from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt import operations
from sglang.srt.layers.moe.token_dispatcher import DeepEPConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.operations import Operation


@dataclass
class OperationsStrategy:
    operations: List[Operation]
    deep_gemm_num_sms: Optional[int] = None
    tbo_delta_stages: Optional[int] = None

    @classmethod
    def concat(cls, items: List["OperationsStrategy"]) -> "OperationsStrategy":
        return OperationsStrategy(
            operations=[x for item in items for x in item.operations],
            deep_gemm_num_sms=_assert_all_same(
                [item.deep_gemm_num_sms for item in items]
            ),
            tbo_delta_stages=_assert_all_same(
                [item.tbo_delta_stages for item in items]
            ),
        )

    @staticmethod
    def init_new_tbo(
        layers: torch.nn.ModuleList,
        forward_mode: ForwardMode,
    ) -> "OperationsStrategy":
        layer_name = layers[0].__class__.__name__
        if layer_name == "DeepseekV2DecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_deepseek_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        elif layer_name == "Qwen3MoeDecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_qwen3_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        # Add support for dense layers
        elif layer_name in ["LlamaDecoderLayer", "Qwen2DecoderLayer", "Qwen3DecoderLayer", 
                            "GemmaDecoderLayer", "Gemma2DecoderLayer", "MistralDecoderLayer"]:
            return OperationsStrategy.concat(
                [
                    _compute_dense_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        else:
            raise NotImplementedError(f"TBO not implemented for layer type: {layer_name}")


def _assert_all_same(items: List):
    assert all(item == items[0] for item in items)
    return items[0]


# -------------------------------- Strategy for DeepSeek ---------------------------------------


# TODO can refactor to make it more fancy if we have more complex strategies
def _compute_moe_deepseek_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    # Remove the dense layer restriction since we're now supporting dense layers too
    # assert layer.is_layer_sparse, "dense layer TBO not yet implemented"
    if hasattr(layer, 'is_layer_sparse') and not layer.is_layer_sparse:
        # This is actually a dense layer, redirect to dense strategy
        return _compute_dense_layer_operations_strategy_tbo(layer, forward_mode)
    
    if forward_mode == ForwardMode.EXTEND:
        return _compute_moe_deepseek_blog_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_moe_deepseek_blog_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_deepseek_blog_prefill(layer):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    deep_gemm_num_sms = total_num_sms - DeepEPConfig.get_instance().num_sms

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_shared_experts,
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_deepseek_blog_decode(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=2,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_a,
            layer.mlp.op_shared_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            operations.YieldOperation(),
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


# -------------------------------- Strategy for Qwen3 ---------------------------------------


# TODO: unstable, current strategy is almost the same as DeepSeek, keep redundant code here for
# convenience to adjust strategy
def _compute_moe_qwen3_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    # Support both sparse and dense layers
    if hasattr(layer, 'is_layer_sparse') and not layer.is_layer_sparse:
        # This is actually a dense layer, redirect to dense strategy
        return _compute_dense_layer_operations_strategy_tbo(layer, forward_mode)
        
    assert layer.is_layer_sparse, "qwen3 moe only support sparse layers"
    if forward_mode == ForwardMode.EXTEND:
        return _compute_moe_qwen3_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_moe_qwen3_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_qwen3_prefill(layer):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    deep_gemm_num_sms = total_num_sms - DeepEPConfig.get_instance().num_sms

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_qwen3_decode(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=2,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
            operations.YieldOperation(),
        ],
    )


# -------------------------------- Strategy for Dense Layers ---------------------------------------


def _compute_dense_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    """TBO strategy for dense layers (Llama, Qwen2, Gemma, etc.)"""
    if forward_mode == ForwardMode.EXTEND:
        return _compute_dense_layer_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_dense_layer_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=} for dense layer TBO")


def _compute_dense_layer_prefill(layer):
    """Dense layer prefill strategy - fine-grained TP communication overlap"""
    from sglang.srt.managers.schedule_batch import global_server_args_dict
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=global_server_args_dict.get("tbo_delta_extend", 0),
        operations=[
            # Stage 1: Compute-only operations
            layer.op_input_norm_and_qkv,
            layer.op_attention_compute,
            operations.YieldOperation(),  # Allow batch overlap before attention communication
            # Stage 2: TP communication for attention
            layer.op_attention_output_proj_and_allreduce,
            operations.YieldOperation(),  # Allow batch overlap after attention communication
            # Stage 3: Compute-only operations  
            layer.op_post_attn_norm_and_gate_up,
            operations.YieldOperation(),  # Allow batch overlap before MLP communication
            # Stage 4: TP communication for MLP
            layer.op_mlp_down_proj_and_allreduce,
            operations.YieldOperation(),  # Allow batch overlap after MLP communication
            layer.op_residual_add_final,
        ],
    )


def _compute_dense_layer_decode(layer):
    """Dense layer decode strategy - aggressive TP communication overlap"""
    from sglang.srt.managers.schedule_batch import global_server_args_dict
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        tbo_delta_stages=global_server_args_dict.get("tbo_delta_decode", 2),
        operations=[
            # Stage 1: Compute-only operations
            layer.op_input_norm_and_qkv,
            operations.YieldOperation(),  # Switch to second batch
            layer.op_attention_compute,
            operations.YieldOperation(),  # Switch back
            # Stage 2: TP communication for attention
            layer.op_attention_output_proj_and_allreduce,
            operations.YieldOperation(),  # Switch to second batch
            # Stage 3: Compute-only operations
            layer.op_post_attn_norm_and_gate_up,
            operations.YieldOperation(),  # Switch back
            # Stage 4: TP communication for MLP
            layer.op_mlp_down_proj_and_allreduce,
            operations.YieldOperation(),  # Switch to second batch
            layer.op_residual_add_final,
        ],
    )
