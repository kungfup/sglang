# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from llama2.py
# Modify details for the adaptation of Qwen2 model.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
import logging
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from sglang.srt.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.two_batch_overlap import TboForwardBatchPreparer
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.layers.communicator import LayerScatterModes
from sglang.srt.utils import add_prefix, make_layers
import os

_TBO_DEBUG = bool(int(os.environ.get("SGLANG_TBO_DEBUG", "0")))

def _tbo_log(msg: str):
    if _TBO_DEBUG:
        print(f"[TBO][qwen2] {msg}", flush=True)

Qwen2Config = None


logger = logging.getLogger(__name__)


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            dual_chunk_attention_config=dual_chunk_attention_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Initialize layer_scatter_modes for TBO support
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,  # Dense layer
            is_previous_layer_sparse=False,  # Dense model, previous layer is also dense
        )
        
        # 简化版TBO支持 - 专注于功能对齐
        self._tbo_enabled = torch.cuda.is_available()
    
    def _can_use_tbo_optimization(self, state):
        """检查是否可以使用TBO优化"""
        return self._tbo_enabled and hasattr(state, '_tbo_batch_id')
    
    def _get_batch_id(self, state):
        """获取micro-batch ID"""
        return getattr(state, '_tbo_batch_id', 'b1')

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    # Enhanced TBO Operations with TP Communication Overlap
    def op_input_norm_and_qkv(self, **kwargs):
        # 调试信息：查看接收到的参数
        if _TBO_DEBUG:
            _tbo_log(f"[qwen2] op_input_norm_and_qkv received kwargs: {list(kwargs.keys())}")
            for k, v in kwargs.items():
                try:
                    if hasattr(v, 'shape') and hasattr(v, 'dtype'):
                        _tbo_log(f"  {k}: tensor shape {v.shape}")
                    elif v is None:
                        _tbo_log(f"  {k}: None")
                    else:
                        _tbo_log(f"  {k}: {type(v)}")
                except Exception as e:
                    _tbo_log(f"  {k}: {type(v)} (error accessing: {e})")
        
        # 从 kwargs 获取输入
        positions = kwargs.get('positions')
        forward_batch = kwargs.get('forward_batch') 
        hidden_states = kwargs.get('hidden_states')
        residual = kwargs.get('residual')
        
        # 如果缺少关键参数，尝试从其他地方获取
        if positions is None or forward_batch is None:
            # 对于第一个操作，这些可能在state中或者需要从其他源获取
            if _TBO_DEBUG:
                _tbo_log(f"[qwen2] Missing positions or forward_batch, trying to recover...")
            
            # 检查是否可以从state获取
            state = kwargs.get('state')
            if state and hasattr(state, '_data'):
                positions = positions or state._data.get('positions')
                forward_batch = forward_batch or state._data.get('forward_batch')
            
            # 如果仍然缺少，从父调用的kwargs中获取（如果有的话）
            if positions is None or forward_batch is None:
                # 这种情况下，我们可能需要从全局上下文获取或者退回到非TBO模式
                if _TBO_DEBUG:
                    _tbo_log(f"[qwen2] Cannot find positions/forward_batch, falling back")
                # 为了保持兼容性，我们继续执行，但可能需要调整
                if positions is None:
                    # 创建一个假的positions用于测试
                    if hidden_states is not None:
                        seq_len = hidden_states.shape[0]
                        positions = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
                    else:
                        raise ValueError(f"Cannot recover positions, available keys: {list(kwargs.keys())}")
        
        if _TBO_DEBUG:
            _tbo_log(f"[qwen2] positions shape: {positions.shape if positions is not None else None}")
            _tbo_log(f"[qwen2] forward_batch: {type(forward_batch)}")
        
        # 清理之前可能遗留的临时状态
        state = kwargs.get('state')
        if state is not None:
            if state.get('residual_after_input_ln') is not None:
                state.pop('residual_after_input_ln')
            if state.get('residual_after_comm_pre_mlp') is not None:
                state.pop('residual_after_comm_pre_mlp')
        
        if _TBO_DEBUG:
            _tbo_log(f"enter input_norm_and_qkv")
        
        # 添加张量形状调试信息
        if _TBO_DEBUG:
            _tbo_log(f"[qwen2] hidden_states shape: {hidden_states.shape if hidden_states is not None else None}")
            _tbo_log(f"[qwen2] residual shape: {residual.shape if residual is not None else None}")
            _tbo_log(f"[qwen2] hidden_states dtype: {hidden_states.dtype if hidden_states is not None else None}")
            _tbo_log(f"[qwen2] hidden_states device: {hidden_states.device if hidden_states is not None else None}")
        
        # Input layer norm（与forward一致的残差路径）
        if residual is None:
            residual = hidden_states
            if _TBO_DEBUG:
                _tbo_log(f"[qwen2] About to call input_layernorm with shape {hidden_states.shape}")
            hidden_states = self.input_layernorm(hidden_states)
        else:
            if _TBO_DEBUG:
                _tbo_log(f"[qwen2] About to call input_layernorm with residual path, shapes: hidden={hidden_states.shape}, residual={residual.shape}")
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # Self Attention QKV projection
        qkv, _ = self.self_attn.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.self_attn.q_size, self.self_attn.kv_size, self.self_attn.kv_size], dim=-1)
        
        # Apply rotary embedding
        q, k = self.self_attn.rotary_emb(positions, q, k)
        
        # 存储中间状态供后续操作使用
        state = kwargs.get('state')
        if state is not None:
            state['q'] = q
            state['k'] = k 
            state['v'] = v
            state['residual_after_input_ln'] = residual
        # 继续向后传递必要字段
        return dict(
            positions=positions,
            forward_batch=forward_batch,
            hidden_states=hidden_states,
            residual=residual,
            tbo_subbatch_index=kwargs.get('tbo_subbatch_index'),
        )

    def op_attention_compute(self, state, **kwargs):
        """TBO Stage 2: Attention computation (no communication)"""
        if _TBO_DEBUG:
            _tbo_log(f"enter attention_compute")
        
        # 从 state 获取 QKV
        q = state.get('q')
        k = state.get('k') 
        v = state.get('v')
        
        if q is None or k is None or v is None:
            raise ValueError("QKV tensors not found in state")
        
        # 从 kwargs 获取其他参数
        forward_batch = kwargs.get('forward_batch')
        
        # 执行注意力计算
        attn_output = self.self_attn.attn(q, k, v, forward_batch)
        
        # 存储结果到 state
        state['attn_output'] = attn_output
        # 继续传递必要字段
        return dict(
            forward_batch=forward_batch,
            positions=kwargs.get('positions'),
            hidden_states=kwargs.get('hidden_states'),
            residual=kwargs.get('residual'),
            tbo_subbatch_index=kwargs.get('tbo_subbatch_index'),
        )

    def op_attention_output_proj_and_allreduce(self, state, **kwargs):
        """TBO Stage 3: Attention output projection + TP All-Reduce (communication-heavy)"""
        if _TBO_DEBUG:
            _tbo_log(f"enter attention_output_proj_allreduce")
        
        # 从 state 获取注意力输出
        attn_output = state.get('attn_output')
        if attn_output is None:
            raise ValueError("attn_output not found in state")
        
        # 执行输出投影和 all-reduce
        attn_final_output, _ = self.self_attn.o_proj(attn_output)
        
        # 存储结果到 state
        state['attn_final_output'] = attn_final_output
        # 继续传递必要字段
        return dict(
            forward_batch=kwargs.get('forward_batch'),
            positions=kwargs.get('positions'),
            hidden_states=kwargs.get('hidden_states'),
            residual=kwargs.get('residual'),
            tbo_subbatch_index=kwargs.get('tbo_subbatch_index'),
        )

    def op_post_attn_norm_and_gate_up(self, state, **kwargs):
        """TBO Stage 4: Post-attention layernorm + MLP gate_up (no communication)"""
        if _TBO_DEBUG:
            _tbo_log(f"enter post_attn_norm_and_gate_up")
        
        # 从 state 获取注意力最终输出和残差
        attn_final_output = state.get('attn_final_output')
        residual_after_input_ln = state.get('residual_after_input_ln')
        
        if attn_final_output is None or residual_after_input_ln is None:
            raise ValueError("Required tensors not found in state")
        
        # 从 kwargs 获取原始输入
        hidden_states = kwargs.get('hidden_states')
        
        # Add residual connection from input layernorm
        hidden_states = attn_final_output + residual_after_input_ln
        
        # Post-attention layernorm（与forward一致，返回新的 residual）
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual_after_input_ln)
        
        # MLP gate and up projections (ColumnParallel, no all-reduce)  
        gate_up, _ = self.mlp.gate_up_proj(hidden_states)
        gate_up_output = self.mlp.act_fn(gate_up)
        
        # 存储结果到 state
        state['gate_up_output'] = gate_up_output
        state['residual_after_comm_pre_mlp'] = residual
        # 继续传递必要字段
        return dict(
            forward_batch=kwargs.get('forward_batch'),
            positions=kwargs.get('positions'),
            hidden_states=kwargs.get('hidden_states'),
            residual=kwargs.get('residual'),
            tbo_subbatch_index=kwargs.get('tbo_subbatch_index'),
        )

    def op_mlp_down_proj_and_allreduce(self, state, **kwargs):
        """TBO Stage 5: MLP down projection + TP All-Reduce (communication-heavy)"""
        if _TBO_DEBUG:
            _tbo_log(f"enter mlp_down_proj_allreduce")
        
        # 从 state 获取 gate_up 输出
        gate_up_output = state.get('gate_up_output')
        if gate_up_output is None:
            raise ValueError("gate_up_output not found in state")
        
        # 执行下投影和 all-reduce
        mlp_output, _ = self.mlp.down_proj(gate_up_output)
        
        # 存储结果到 state
        state['mlp_output'] = mlp_output
        # 继续传递必要字段（为最终残差加和做准备）
        return dict(
            hidden_states=mlp_output,
            residual=state.get('residual_after_comm_pre_mlp'),
            forward_batch=kwargs.get('forward_batch'),
            positions=kwargs.get('positions'),
            tbo_subbatch_index=kwargs.get('tbo_subbatch_index'),
        )

    def op_residual_add_final(self, state, hidden_states, residual, **kwargs):
        """TBO Stage 6: Final residual addition (no communication)"""
        if _TBO_DEBUG:
            _tbo_log(f"enter residual_add_final")
        
        # 从 state 获取 MLP 输出和残差
        mlp_output = state.get('mlp_output')
        residual_after_comm_pre_mlp = state.get('residual_after_comm_pre_mlp')
        
        if mlp_output is None or residual_after_comm_pre_mlp is None:
            raise ValueError("Required tensors not found in state")
        
        # 最终残差连接
        final_hidden_states = mlp_output + residual_after_comm_pre_mlp
        
        # 清理临时状态
        keys_to_pop = [
            'residual_after_input_ln', 'residual_after_comm_pre_mlp', 'mlp_output',
            'gate_up_output', 'attn_output', 'attn_final_output',
            'q', 'k', 'v'
        ]
        for key in keys_to_pop:
            if key in state:
                state.pop(key)
        # 返回标准字典，供TBO聚合
        return dict(hidden_states=final_hidden_states, residual=residual_after_comm_pre_mlp)


class Qwen2Model(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                enable_tp=not global_server_args_dict["enable_dp_attention"],
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2DecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.config, "scale_emb"):
            return self.get_input_embeddings()(input_ids) * self.config.scale_emb
        else:
            return self.get_input_embeddings()(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        # Check if we can use TBO for the layers
        aux_hidden_states = []
        if hasattr(forward_batch, 'can_run_tbo') and forward_batch.can_run_tbo and len(self.layers) > 0:
            # Use TBO for overlapped execution (children/metadata由调度器/Runner准备)
            for i in range(self.start_layer, self.end_layer):
                if i in self.layers_to_capture:
                    aux_hidden_states.append(
                        hidden_states + residual if residual is not None else hidden_states
                    )
            layers_to_run = [self.layers[i] for i in range(self.start_layer, self.end_layer)]
            if _TBO_DEBUG:
                _tbo_log(f"forward use TBO: can_run_tbo={getattr(forward_batch,'can_run_tbo',None)}, num_layers={len(layers_to_run)}")
            if layers_to_run:
                hidden_states, residual = model_forward_maybe_tbo(
                    layers=layers_to_run,
                    enable_tbo=True,
                    input_data_scatter_mode=ScatterMode.model_input_output(),
                    positions=positions,
                    forward_batch=forward_batch,
                    hidden_states=hidden_states,
                    residual=residual,
                )
        else:
            if _TBO_DEBUG:
                # 追加更详细的诊断信息，便于定位 can_run_tbo 为 False 的原因
                try:
                    bs = (
                        forward_batch.batch_size()
                        if hasattr(forward_batch, "batch_size") and callable(forward_batch.batch_size)
                        else getattr(forward_batch, "batch_size", None)
                    )
                except Exception:
                    bs = None
                _tbo_log(
                    f"forward fallback sequential: can_run_tbo={getattr(forward_batch,'can_run_tbo',None)} "
                    f"split_seq_idx={getattr(forward_batch,'tbo_split_seq_index',None)} "
                    f"global_forward_mode={getattr(forward_batch,'global_forward_mode',None)} "
                    f"local_forward_mode={getattr(forward_batch,'forward_mode',None)} "
                    f"bs={bs} "
                    f"extend_lens_cpu={getattr(forward_batch,'extend_seq_lens_cpu',None)}"
                )
            # Use standard sequential execution
            for i in range(self.start_layer, self.end_layer):
                if i in self.layers_to_capture:
                    aux_hidden_states.append(
                        hidden_states + residual if residual is not None else hidden_states
                    )
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn
            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )


class Qwen2ForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen2Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )

        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        # perform weight tying for PP
        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                )
            else:
                emb_token_weight = self.pp_group.recv(
                    size=(config.vocab_size, config.hidden_size),
                    dtype=next(self.model.parameters()).dtype,
                    src=self.pp_group.first_rank,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embedding(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids, hidden_states, self.lm_head, forward_batch
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    # Handle pp weight tying here
                    # find the embed_tokens.weight in the weights
                    embed_token_weights = next(
                        filter(lambda x: x[0] == "model.embed_tokens.weight", weights)
                    )[1]
                    loaded_weight = embed_token_weights
                else:
                    continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


EntryClass = Qwen2ForCausalLM
