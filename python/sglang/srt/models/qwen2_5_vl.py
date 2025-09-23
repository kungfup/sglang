# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/19e6e80e10118f855137b90740936c0b11ac397f/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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
"""Inference-only Qwen2-VL model compatible with HuggingFace weights."""
import logging
import os
from functools import lru_cache, partial
from typing import Iterable, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
)

from sglang.srt.distributed import get_pp_group, get_tp_group
from sglang.srt.hf_transformers_utils import get_processor
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
    embed_mm_inputs,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.qwen2_vl import Qwen2VLVideoInputs
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Qwen2_5_VLMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        bias: bool = True,
        hidden_act="silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.up_proj = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_parallel_gate, _ = self.gate_proj(x)
        x_parallel_gate = self.act(x_parallel_gate)
        x_parallel_up, _ = self.up_proj(x)
        x_parallel = x_parallel_gate * x_parallel_up
        x, _ = self.down_proj(x_parallel)
        return x


class Qwen2_5_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        hidden_act="silu",
        norm_layer: Type[nn.Module] = None,
        attn_implementation: Optional[str] = "sdpa",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = Qwen2RMSNorm(dim, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(dim, eps=1e-6)
        if attn_implementation == "sdpa":
            softmax_in_single_precision = False
            qkv_backend = "sdpa"
            flatten_batch = True
        elif attn_implementation == "flash_attention_2":
            softmax_in_single_precision = False
            qkv_backend = "triton_attn"
            flatten_batch = True
        elif attn_implementation == "eager":
            softmax_in_single_precision = True
            qkv_backend = "sdpa"
            flatten_batch = True
        elif attn_implementation == "flash_attention_3":
            softmax_in_single_precision = False
            qkv_backend = "fa3"
            flatten_batch = True

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            rotary_embed="normal",
            proj_bias=True,
            qkv_backend=qkv_backend,
            softmax_in_single_precision=softmax_in_single_precision,
            flatten_batch=flatten_batch,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.mlp = Qwen2_5_VLMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.norm1(x)
        hidden_states = rearrange(hidden_states, "s b ... -> b s ...")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        attn = rearrange(attn, "b s ... -> s b ...")
        x = x + attn
        norm2 = self.norm2(x)
        mlp = self.mlp(norm2)
        x = x + mlp
        return x


class Qwen2_5_VisionPatchMerger(nn.Module):

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.0", prefix),
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    dim,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.2", prefix),
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class Qwen2_5_VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        patch_size: int = vision_config.patch_size
        temporal_patch_size: int = vision_config.temporal_patch_size
        spatial_merge_size: int = vision_config.spatial_merge_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit: int = spatial_merge_size * spatial_merge_size
        in_channels: int = vision_config.in_channels
        hidden_size: int = vision_config.hidden_size
        depth: int = vision_config.depth
        num_heads: int = vision_config.num_heads
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        mlp_hidden_size: int = vision_config.intermediate_size
        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VisionBlock(
                    dim=hidden_size,
                    intermediate_dim=mlp_hidden_size,
                    num_heads=num_heads,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                    attn_implementation="sdpa",
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{i}", prefix),
                )
                for i in range(depth)
            ]
        )
        self.merger = Qwen2_5_VisionPatchMerger(
            dim=vision_config.out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
            quant_config=quant_config,
            prefix=add_prefix("merger", prefix),
        )

    def get_window_index(self, grid_thw):
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )
        window_index: list = []
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            # ensure non-negative padding when dimensions are divisible
            pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size) % vit_merger_window_size
            pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size) % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        # downstream expects long indices
        window_index = torch.cat(window_index, dim=0).to(torch.long)
        return window_index, cu_window_seqlens

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].mlp.gate_proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for i in range(grid_thw.size(0)):
            t, h, w = grid_thw[i].tolist()
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)

            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=x.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = x.size()

        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # compute cu_seqlens
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=grid_thw.device),
                (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).cumsum(dim=0),
            ]
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        # transformers
        x = x.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            x = blk(
                x, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings
            )

        # adapter
        x = self.merger(x)

        reverse_indices = torch.argsort(window_index)
        x = x[reverse_indices, :]

        return x


cached_get_processor = lru_cache(get_processor)


class Qwen2_5_VLForConditionalGeneration(nn.Module):
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
        config: Qwen2_5_VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.pp_group = get_pp_group()

        # Debug: PP 信息与设备信息（默认关闭，通过 SGLANG_ENABLE_DEBUG_LOGS 开启）
        try:
            if os.environ.get("SGLANG_ENABLE_DEBUG_LOGS", "0").lower() in ("1", "true", "yes"):
                logger.info(
                    f"[QWEN2_5_VL_INIT] pp_world={self.pp_group.world_size} "
                    f"is_first={self.pp_group.is_first_rank} is_last={self.pp_group.is_last_rank}"
                )
        except Exception:
            pass

        if self.pp_group.is_first_rank:
            self.visual = Qwen2_5_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                # NOTE: Qwen2_5-VL vision encoder currently supports BitsAndBytes 4-bit quantization.
                # Other quantization methods (e.g., GPTQ, AWQ) are untested and may not be supported.
                quant_config=quant_config,
                prefix=add_prefix("visual", prefix),
            )
        else:
            self.visual = PPMissingLayer()

        # Log vision tower ownership for PP diagnostics
        try:
            vis_is_missing = isinstance(self.visual, PPMissingLayer)
            vis_cls = type(self.visual).__name__
            vis_params = 0
            if not vis_is_missing:
                for n, p in self.named_parameters():
                    if n.startswith("visual."):
                        vis_params += p.numel()
            logger.info(
                f"[VLM_PP_OWNERSHIP] pp_world={self.pp_group.world_size} pp_first={self.pp_group.is_first_rank} "
                f"pp_last={self.pp_group.is_last_rank} visual_cls={vis_cls} visual_params={vis_params}"
            )
        except Exception:
            pass

        self.model = Qwen2Model(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
        )

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
            self.lm_head = PPMissingLayer()
        # tie embeddings across pp if needed
        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                )
            if self.pp_group.is_last_rank:
                target_dtype = config.torch_dtype or torch.float16
                emb_token_weight = self.pp_group.recv(
                    size=(config.vocab_size, config.hidden_size),
                    dtype=target_dtype,
                    src=self.pp_group.first_rank,
                )
                self.lm_head.weight.data.copy_(emb_token_weight)
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.image_token_idx = config.image_token_id

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        # Get all special token IDs
        im_token_id: int = mm_inputs.im_token_id
        pattern = MultiModalityDataPaddingPatternMultimodalTokens([im_token_id])
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # Build on CPU first to avoid per-item GPU transfers; move once to GPU.
        pv_list = []
        grid_list = []
        for item in items:
            pv = item.pixel_values
            grid = item.image_grid_thw
            if isinstance(pv, torch.Tensor) and pv.is_cuda:
                pv = pv.cpu()
            if isinstance(grid, torch.Tensor) and grid.is_cuda:
                grid = grid.cpu()
            pv_list.append(pv)
            grid_list.append(grid)
        # In qwen-vl, the last dimension is the same
        pixel_values = torch.cat(pv_list, dim=0)
        image_grid_thw = torch.concat(grid_list, dim=0)
        # Move once to target device with proper dtypes
        pixel_values = pixel_values.to(device=self.device, dtype=self.visual.dtype, non_blocking=True)
        image_grid_thw = image_grid_thw.to(device=self.device, non_blocking=True)
        # Lightweight visibility for vision device usage (always on)
        try:
            logger.info(
                "[VLM_VISION] pp_first=%s target_dev=%s pv_shape=%s grid_shape=%s dtype=%s",
                getattr(getattr(self, "pp_group", None), "is_first_rank", None),
                str(self.device),
                tuple(pixel_values.shape),
                tuple(image_grid_thw.shape),
                str(self.visual.dtype),
            )
        except Exception:
            pass

        # Detailed audit log to confirm who really runs ViT
        try:
            role = getattr(self.model, "instance_role", None)
            pp_rank = getattr(getattr(self, "pp_group", None), "rank_in_group", None)
            try:
                tp_rank = get_tp_group().rank_in_group
            except Exception:
                tp_rank = None
            logger.info(
                "[VLM_VIT_FORWARD] pid=%d role=%s pp_first=%s pp_rank=%s tp_rank=%s dev=%s items=%d",
                os.getpid(), str(role), getattr(getattr(self, "pp_group", None), "is_first_rank", None),
                str(pp_rank), str(tp_rank), str(self.device), len(items)
            )
        except Exception:
            pass

        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        try:
            if os.environ.get("SGLANG_ENABLE_DEBUG_LOGS", "0").lower() in ("1", "true", "yes"):
                _dev = pixel_values.device
                _dtype = pixel_values.dtype
                pv_shape = tuple(pixel_values.shape)
                grid_shape = tuple(image_grid_thw.shape)
                # Estimate patch/token and pixel budgets for visibility
                # total patches across items: should equal pixel_values.shape[0]
                try:
                    total_patches_from_grid = (
                        (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2])
                        .sum()
                        .item()
                    )
                except Exception:
                    total_patches_from_grid = -1
                # Approximate pixel budget using IMAGE_FACTOR=28 (see Qwen2_5VLImageProcessor)
                try:
                    approx_pixels_28 = (
                        (image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum().item() * (28 * 28)
                    )
                except Exception:
                    approx_pixels_28 = -1
                # Memory snapshot before vision forward
                if _dev.type == "cuda":
                    dev_index = _dev.index if _dev.index is not None else torch.cuda.current_device()
                    mem_alloc_pre = torch.cuda.memory_allocated(dev_index) / (1024 ** 3)
                    mem_rsv_pre = torch.cuda.memory_reserved(dev_index) / (1024 ** 3)
                else:
                    mem_alloc_pre = mem_rsv_pre = 0.0
                logger.info(
                    "[QWEN2_5_VL_VISION_MEM][pre] dev=%s dtype=%s pv_shape=%s grid_shape=%s "
                    "alloc=%.2fGiB reserved=%.2fGiB patches(pv/grid)=%s/%s approx_pixels_28=%s (IMAGE_FACTOR=28)",
                    str(_dev), str(_dtype), pv_shape, grid_shape,
                    mem_alloc_pre, mem_rsv_pre, pv_shape[0], total_patches_from_grid, approx_pixels_28,
                )
        except Exception:
            pass
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        try:
            if os.environ.get("SGLANG_ENABLE_DEBUG_LOGS", "0").lower() in ("1", "true", "yes"):
                _dev = pixel_values.device
                if _dev.type == "cuda":
                    dev_index = _dev.index if _dev.index is not None else torch.cuda.current_device()
                    mem_alloc_post = torch.cuda.memory_allocated(dev_index) / (1024 ** 3)
                    mem_rsv_post = torch.cuda.memory_reserved(dev_index) / (1024 ** 3)
                else:
                    mem_alloc_post = mem_rsv_post = 0.0
                logger.info(
                    "[QWEN2_5_VL_VISION_MEM][post] dev=%s alloc=%.2fGiB reserved=%.2fGiB",
                    str(_dev), mem_alloc_post, mem_rsv_post,
                )
        except Exception:
            pass
        return image_embeds.contiguous()

    def _process_video_input(self, video_input: Qwen2VLVideoInputs) -> torch.Tensor:
        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        video_embeds = self.visual(
            pixel_values_videos, grid_thw=video_input["video_grid_thw"]
        )
        return video_embeds

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def _prepare_initial_embeddings(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """首段准备多模态/文本混合的 input_embeds，并清理 mm_inputs 以避免传递到后段。"""
        embed_tokens = self.get_input_embeddings()
        # 在 PP 首段执行多模态嵌入（无论实例角色），仅在 prefill 阶段触发
        try:
            pp_first = getattr(getattr(self, "pp_group", None), "is_first_rank", False)
        except Exception:
            pp_first = False
        # Only allow multimodal embedding on PP first rank in PREFILL (non-decode) instances
        try:
            role_str = str(getattr(self, "instance_role", ""))
        except Exception:
            role_str = ""
        is_prefill_role = role_str.endswith("PREFILL")
        is_decode_role = role_str.endswith("DECODE")
        semi_pd_enabled = os.environ.get("SGLANG_ENABLE_SEMI_PD", "0").lower() in ("1", "true", "yes")
        # In Semi-PD, ViT runs on DECODE@PP0; otherwise, allow on PP-first regardless of role
        allow_mm = bool(pp_first and (not forward_batch.forward_mode.is_decode()) and ((semi_pd_enabled and is_decode_role) or (not semi_pd_enabled)))

        if os.environ.get("SGLANG_ENABLE_DEBUG_LOGS", "0").lower() in ("1", "true", "yes"):
            try:
                rid_dbg = None
                if forward_batch.mm_inputs is not None and len(forward_batch.mm_inputs) > 0:
                    rid_dbg = getattr(forward_batch.mm_inputs[0], "_rid", None)
                logger.info(
                    f"[MM_EMBED_DECISION_QWEN] rid={rid_dbg} role={getattr(self,'instance_role',None)} "
                    f"pp_first={pp_first} allow_mm={allow_mm} "
                    f"mode={'DECODE' if forward_batch.forward_mode.is_decode() else 'EXTEND'}"
                )
            except Exception:
                pass

        if (
            forward_batch.contains_mm_inputs()
            and allow_mm
        ):
            mm_inputs_list = [
                mm_input for mm_input in forward_batch.mm_inputs if mm_input is not None
            ]
            extend_prefix_lens = [
                prefix_len
                for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            extend_seq_lens = [
                seq_len
                for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            inputs_embeds = embed_mm_inputs(
                mm_inputs_list=mm_inputs_list,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens=extend_seq_lens,
                input_ids=input_ids,
                input_embedding=embed_tokens,
                image_data_embedding_func=self.get_image_feature,
                audio_data_embedding_func=None,
                placeholder_tokens=None,
            )
            # 嵌入完成后，避免把原始多模态数据传到后段
            forward_batch.mm_inputs = None
        else:
            inputs_embeds = embed_tokens(input_ids)
            # If this instance is not allowed to process multimodal inputs under Semi-PD, drop them to avoid downstream misuse
            try:
                semi_pd_enabled = os.environ.get("SGLANG_ENABLE_SEMI_PD", "0").lower() in ("1", "true", "yes")
                if semi_pd_enabled and forward_batch.contains_mm_inputs() and not allow_mm:
                    forward_batch.mm_inputs = None
            except Exception:
                pass
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ):
        """Run forward pass for Qwen2_5-VL with PP-aware multimodal embedding.
        首段：准备 input_embeds 并传入 self.model；后段：仅接收 hidden_states。
        末段：计算 logits/pooling。
        """
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        # 仅首段构建多模态嵌入，后段不再触发任何多模态嵌入逻辑
        input_embeds = None
        if self.pp_group.is_first_rank:
            input_embeds = self._prepare_initial_embeddings(
                input_ids=input_ids,
                forward_batch=forward_batch,
            )

        # 模型前向：传入 positions/forward_batch/input_embeds/pp_proxy_tensors
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        try:
            if os.environ.get("SGLANG_ENABLE_DEBUG_LOGS", "0").lower() in ("1", "true", "yes"):
                logger.info(
                    f"[QWEN2_5_VL_FWD] mode={'DECODE' if forward_batch.forward_mode.is_decode() else 'EXTEND'} "
                    f"mm={'N/A' if forward_batch.mm_inputs is None else True} "
                    f"pp_first={self.pp_group.is_first_rank} pp_last={self.pp_group.is_last_rank} "
                    f"hs_dev={getattr(hidden_states, 'device', 'cpu')} hs_shape={getattr(hidden_states, 'shape', None)}"
                )
        except Exception:
            pass

        # 非末段直接返回 hidden_states
        if not self.pp_group.is_last_rank:
            return hidden_states

        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Robust weight loader that supports fused/non-fused naming and PP slicing.

        - Handles q/k/v→qkv_proj and gate/up→gate_up_proj mapping when available;
          falls back to original separate weights otherwise.
        - Skips vision weights on non-first PP stage and lm_head weights on non-last PP stage.
        - Silently skips any tensor not present on this PP stage to avoid KeyError.
        """
        from inspect import signature

        prefix_map = {
            "model.": "model.",
            "lm_head.": "lm_head.",
            "visual.": "visual.",
        }
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Normalize prefix (handle llava-style vision prefix if any)
            if name.startswith("visual.vision_tower.vision_model."):
                sglang_name = name.replace(
                    "visual.vision_tower.vision_model.", "visual."
                )
            else:
                sglang_name = name
                for old_prefix, new_prefix in prefix_map.items():
                    if sglang_name.startswith(old_prefix):
                        sglang_name = new_prefix + sglang_name[len(old_prefix) :]
                        break

            # Non-first PP stage does not own vision module
            if (
                self.pp_group.world_size > 1
                and not self.pp_group.is_first_rank
                and sglang_name.startswith("visual.")
            ):
                continue
            # Non-last PP stage does not own lm_head
            if (
                self.pp_group.world_size > 1
                and not self.pp_group.is_last_rank
                and sglang_name.startswith("lm_head")
            ):
                continue

            # Adapt qkv naming for vision if needed
            if ".attn.qkv." in sglang_name:
                sglang_name = sglang_name.replace(".attn.qkv.", ".attn.qkv_proj.")

            # Try bitsandbytes-stacked mapping first (q/k/v and gate/up fusions)
            is_stacked = False
            for src_proj_name, (dst_fused, shard_idx) in (
                self.bitsandbytes_stacked_params_mapping.items()
            ):
                pattern_base = f".{src_proj_name}"
                if sglang_name.endswith(f"{pattern_base}.weight") or sglang_name.endswith(
                    f"{pattern_base}.bias"
                ):
                    fused_name = sglang_name.replace(pattern_base, f".{dst_fused}")
                    if fused_name in params_dict:
                        # Use fused param loader with appropriate shard id
                        param = params_dict[fused_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        # Prefer keyword arg if supported to avoid mismatched signatures
                        try:
                            sig = signature(weight_loader)
                            if "loaded_shard_id" in sig.parameters:
                                # Map q/k/v to string identifiers expected by attention loaders
                                if src_proj_name in ["q_proj", "k_proj", "v_proj"]:
                                    loaded_shard_id = src_proj_name.split("_")[0]
                                else:
                                    loaded_shard_id = shard_idx
                                weight_loader(
                                    param,
                                    loaded_weight,
                                    loaded_shard_id=loaded_shard_id,
                                )
                            else:
                                # Fall back to positional if only 2-arg loaders exist
                                weight_loader(param, loaded_weight)
                        except Exception:
                            # Last-resort fallback for any unforeseen loader signature
                            weight_loader(param, loaded_weight)
                        is_stacked = True
                    elif sglang_name in params_dict:
                        # Fallback: load original separate weight
                        param = params_dict[sglang_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        is_stacked = True
                    break
            if is_stacked:
                continue

            # Regular load path
            if sglang_name.endswith(".bias") and sglang_name not in params_dict:
                continue
            if sglang_name not in params_dict:
                continue
            param = params_dict[sglang_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = [Qwen2_5_VLForConditionalGeneration]
