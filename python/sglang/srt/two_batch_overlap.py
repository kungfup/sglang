from __future__ import annotations

import copy
import dataclasses
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import torch
import os

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSummableTensorPairFn,
    ScatterMode,
)
from sglang.srt.layers.moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.managers.schedule_batch import ScheduleBatch, global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.operations import execute_operations, execute_overlapped_operations
from sglang.srt.operations_strategy import OperationsStrategy
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
from sglang.srt.utils import BumpAllocator, get_bool_env_var, is_hip
import os

# 在文件开头添加资源清理和错误处理
import atexit
import threading
import weakref

# 在文件开头的import后添加vLLM风格的微批处理管理
import threading
import weakref
from typing import Dict, Any
from contextlib import contextmanager

# 参考vLLM的UBatchContext，为SGLang实现微批处理上下文管理
class SGLangMicroBatchContext:
    """
    SGLang微批处理上下文管理器
    参考vLLM的UBatchContext实现，用于管理多请求batch的micro-batch执行
    """
    
    def __init__(self, 
                 micro_batch_id: int,
                 total_micro_batches: int,
                 forward_batch,
                 parent_batch_info: Dict[str, Any]):
        self.micro_batch_id = micro_batch_id
        self.total_micro_batches = total_micro_batches
        self.forward_batch = forward_batch
        self.parent_batch_info = parent_batch_info
        
        # 同步事件
        self.execution_done = threading.Event()
        self.ready_for_merge = threading.Event()
        
        # 结果存储
        self.execution_result = None
        self.execution_error = None
        
    def mark_execution_done(self, result=None, error=None):
        """标记micro-batch执行完成"""
        self.execution_result = result
        self.execution_error = error
        self.execution_done.set()
        
    def wait_for_execution(self, timeout=None):
        """等待micro-batch执行完成"""
        return self.execution_done.wait(timeout)
        
    def is_ready_for_merge(self):
        """检查是否准备好合并"""
        return self.ready_for_merge.is_set()

# 全局micro-batch管理器
_global_micro_batch_contexts: Dict[str, List[SGLangMicroBatchContext]] = {}
_global_context_lock = threading.Lock()

def create_micro_batch_contexts(batch_id: str, 
                               num_micro_batches: int,
                               forward_batch,
                               batch_info: Dict[str, Any]) -> List[SGLangMicroBatchContext]:
    """
    创建micro-batch上下文列表
    参考vLLM的make_ubatch_contexts
    """
    with _global_context_lock:
        contexts = []
        for i in range(num_micro_batches):
            context = SGLangMicroBatchContext(
                micro_batch_id=i,
                total_micro_batches=num_micro_batches,
                forward_batch=forward_batch,
                parent_batch_info=batch_info
            )
            contexts.append(context)
        
        _global_micro_batch_contexts[batch_id] = contexts
        return contexts

def cleanup_micro_batch_contexts(batch_id: str):
    """清理micro-batch上下文"""
    with _global_context_lock:
        if batch_id in _global_micro_batch_contexts:
            del _global_micro_batch_contexts[batch_id]

@contextmanager
def sglang_micro_batch_execution(context: SGLangMicroBatchContext):
    """
    SGLang micro-batch执行上下文管理器
    参考vLLM的UBatchContext.__enter__/__exit__
    """
    try:
        if _TBO_DEBUG:
            _tbo_log(f"SGLang micro-batch {context.micro_batch_id} execution started")
        yield context
    except Exception as e:
        context.mark_execution_done(error=e)
        raise
    finally:
        if _TBO_DEBUG:
            _tbo_log(f"SGLang micro-batch {context.micro_batch_id} execution completed")

# 全局资源追踪
_active_tbo_resources = weakref.WeakSet()
_resource_lock = threading.Lock()

def _cleanup_tbo_resources():
    """清理TBO相关资源"""
    with _resource_lock:
        try:
            for resource in list(_active_tbo_resources):
                if hasattr(resource, 'cleanup'):
                    resource.cleanup()
        except Exception as e:
            print(f"[TBO] Warning: Error during resource cleanup: {e}")

# 注册退出时清理
atexit.register(_cleanup_tbo_resources)

_TBO_DEBUG = bool(int(os.environ.get("SGLANG_TBO_DEBUG", "0")))

def _tbo_log(msg: str):
    if _TBO_DEBUG:
        print(f"[TBO] {msg}", flush=True)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutput

_is_hip = is_hip()

_tbo_debug = get_bool_env_var("SGLANG_TBO_DEBUG")

logger = logging.getLogger(__name__)


# -------------------------------- Compute Basic Info ---------------------------------------


def get_token_num_per_seq(
    forward_mode: ForwardMode,
    spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]] = None,
):
    if forward_mode.is_target_verify():
        return spec_info.draft_token_num
    elif forward_mode.is_decode():
        return 1
    elif forward_mode.is_idle():
        return 0
    else:
        # For extend, we should not use `token_num_per_seq`.
        return None


# TBO强制模式：EXTEND强制执行，DECODE单token跳过
def compute_split_seq_index(
    forward_mode: "ForwardMode",
    num_tokens: int,
    extend_lens: Optional[Sequence[int]],
    token_num_per_seq: Optional[int],
) -> Optional[int]:
    """
    Dense模型TBO切分索引：EXTEND强制执行，DECODE单token跳过
    """
    
    # 1. EXTEND模式的切分逻辑
    if forward_mode == ForwardMode.EXTEND:
        assert extend_lens is not None
        num_requests = len(extend_lens)
        total_extend_tokens = sum(extend_lens)
        
        # 1.1 单请求two-chunk特殊处理：参考vLLM，允许单请求通过token切分启用TBO
        if num_requests == 1 and _is_two_chunk_split_enabled(extend_lens):
            if _TBO_DEBUG:
                _tbo_log(f"compute_split_seq_index EXTEND: single request two-chunk enabled, extend_lens={list(extend_lens)}, return idx=0")
            return 0
            
        # 1.2 强制切分：无论请求数量多少，都执行切分
        if num_requests < 2:
            if _TBO_DEBUG:
                _tbo_log(f"compute_split_seq_index EXTEND: FORCE MODE, num_requests={num_requests}, forcing split at idx=0")
            return 0
        
        # 1.3 执行切分（参考vLLM的智能切分算法）
        idx = _split_extend_seqs(extend_lens)
        
        # 1.4 强制确保有效索引
        if idx <= 0:
            idx = 1
        elif idx >= num_requests:
            idx = num_requests - 1
        
        # 1.5 多请求batch切分质量评估（参考vLLM的负载均衡检查）
        left_tokens = sum(extend_lens[:idx])
        right_tokens = sum(extend_lens[idx:])
        imbalance_ratio = abs(left_tokens - right_tokens) / max(left_tokens, right_tokens) if max(left_tokens, right_tokens) > 0 else 0
        
        if _TBO_DEBUG:
            _tbo_log(f"compute_split_seq_index EXTEND: FORCE MODE split analysis:")
            _tbo_log(f"  extend_lens={list(extend_lens)}, split_idx={idx}")
            _tbo_log(f"  micro_batch_A: {idx} requests, {left_tokens} tokens")
            _tbo_log(f"  micro_batch_B: {num_requests-idx} requests, {right_tokens} tokens")
            _tbo_log(f"  load_imbalance_ratio={imbalance_ratio:.3f}")
        
        return idx
        
    # 2. DECODE/VERIFY模式的切分逻辑
    elif forward_mode.is_target_verify() or forward_mode.is_decode():
        assert token_num_per_seq is not None
        num_requests = num_tokens // token_num_per_seq if token_num_per_seq > 0 else 1
        
        # 2.1 DECODE阶段单token保护：避免产生空micro-batch
        if num_requests < 2:
            if _TBO_DEBUG:
                _tbo_log(f"compute_split_seq_index {forward_mode}: single request DECODE (num_requests={num_requests}, num_tokens={num_tokens})")
                _tbo_log(f"compute_split_seq_index {forward_mode}: DECODE single token cannot be meaningfully split, skipping TBO")
            # DECODE阶段单token无法有效切分，静默跳过TBO
            return None
            
        # 2.2 执行切分（参考vLLM的均衡切分）
        idx = max(1, num_requests // 2)
        
        # 2.3 强制确保有效索引
        if idx >= num_requests:
            idx = num_requests - 1
        
        # 2.4 多请求batch切分质量评估（DECODE模式）
        micro_batch_a_size = idx
        micro_batch_b_size = num_requests - idx
        micro_batch_a_tokens = micro_batch_a_size * token_num_per_seq
        micro_batch_b_tokens = micro_batch_b_size * token_num_per_seq
        
        if _TBO_DEBUG:
            _tbo_log(f"compute_split_seq_index {forward_mode}: FORCE MODE split analysis:")
            _tbo_log(f"  total_requests={num_requests}, split_idx={idx}")
            _tbo_log(f"  micro_batch_A: {micro_batch_a_size} requests, {micro_batch_a_tokens} tokens")
            _tbo_log(f"  micro_batch_B: {micro_batch_b_size} requests, {micro_batch_b_tokens} tokens")
            _tbo_log(f"  token_per_seq={token_num_per_seq}")
            
        return idx
        
    # 3. IDLE模式
    elif forward_mode.is_idle():
        assert num_tokens == 0
        return 0
    else:
        raise NotImplementedError(f"Unsupported forward_mode: {forward_mode}")


def _is_two_chunk_split_enabled(extend_lens: Sequence[int]) -> bool:
    if extend_lens is None:
        return False

    vanilla_split_seq_index = _split_array_by_balanced_sum(extend_lens)
    left_sum = sum(extend_lens[:vanilla_split_seq_index])
    overall_sum = sum(extend_lens)
    threshold = global_server_args_dict["tbo_token_distribution_threshold"]
    assert threshold <= 0.5, f"{threshold=}"
    enabled = left_sum < overall_sum * threshold or left_sum > overall_sum * (
        1 - threshold
    )
    if _TBO_DEBUG:
        _tbo_log(f"two_chunk_check: extend_lens={list(extend_lens)}, vanilla_idx={vanilla_split_seq_index}, left_sum={left_sum}, overall={overall_sum}, threshold={threshold}, enabled={enabled}")
    return enabled


def _split_extend_seqs(arr: Sequence[int]) -> int:
    if _is_two_chunk_split_enabled(arr):
        return _split_array_by_cum_less_than_half(arr)

    return _split_array_by_balanced_sum(arr)


def _split_array_by_cum_less_than_half(arr: Sequence[int]) -> int:
    left_sum = 0
    overall_sum = sum(arr)
    half_sum = overall_sum // 2
    chosen_index = 0

    for i in range(len(arr)):
        left_sum += arr[i]
        if left_sum > half_sum:
            chosen_index = i
            break

    return chosen_index


def _split_array_by_balanced_sum(arr: Sequence[int]) -> int:
    overall_sum = sum(arr)
    left_sum = 0
    min_diff = float("inf")
    best_index = 0

    for i in range(1, len(arr)):
        left_sum += arr[i - 1]
        right_sum = overall_sum - left_sum
        diff = abs(left_sum - right_sum)
        if diff <= min_diff:
            min_diff = diff
            best_index = i
        else:
            break

    return best_index


def _update_device_and_sum_field_from_cpu_field(
    batch: ForwardBatch, cpu_field: str, device_field: str, sum_field: str = None
):
    cpu_value = getattr(batch, cpu_field, None)
    old_device_value = getattr(batch, device_field, None)
    if (
        cpu_value is None
        or old_device_value is None
        or not (isinstance(cpu_value, torch.Tensor) or isinstance(cpu_value, list))
    ):
        return

    # TBO强制模式：安全的tensor创建和转移，多重保护机制
    target_device = global_server_args_dict["device"]
    
    # 重置CUDA环境确保稳定性
    if torch.cuda.is_available() and hasattr(target_device, 'type') and target_device.type == 'cuda':
        try:
            torch.cuda.synchronize()
            current_device = torch.cuda.current_device()
            torch.cuda.set_device(current_device)
        except Exception as reset_e:
            if _TBO_DEBUG:
                _tbo_log(f"_update_device_and_sum_field_from_cpu_field: CUDA reset warning: {reset_e}")
    
    # 多级tensor创建和转移策略
    new_device_value = None
    success = False
    
    try:
        # 方法1：直接转移到目标设备
        if isinstance(cpu_value, torch.Tensor):
            new_device_value = cpu_value.to(device=target_device, non_blocking=True)
        else:
            temp_tensor = torch.tensor(cpu_value, dtype=old_device_value.dtype)
            new_device_value = temp_tensor.to(device=target_device, non_blocking=True)
        success = True
        
        if _TBO_DEBUG:
            _tbo_log(f"_update_device_and_sum_field_from_cpu_field: Direct transfer successful to {target_device}")
            
    except Exception as e:
        if _TBO_DEBUG:
            _tbo_log(f"_update_device_and_sum_field_from_cpu_field: Direct transfer failed: {e}")
        
        try:
            # 方法2：同步转移（更安全）
            if isinstance(cpu_value, torch.Tensor):
                new_device_value = cpu_value.to(device=target_device, non_blocking=False)
            else:
                temp_tensor = torch.tensor(cpu_value, dtype=old_device_value.dtype)
                new_device_value = temp_tensor.to(device=target_device, non_blocking=False)
            success = True
            
            if _TBO_DEBUG:
                _tbo_log(f"_update_device_and_sum_field_from_cpu_field: Sync transfer successful")
                
        except Exception as sync_e:
            if _TBO_DEBUG:
                _tbo_log(f"_update_device_and_sum_field_from_cpu_field: Sync transfer failed: {sync_e}")
            
            # 方法3：保持原值（最安全）
            new_device_value = old_device_value
            if _TBO_DEBUG:
                _tbo_log("_update_device_and_sum_field_from_cpu_field: Keeping original value")
    
    # 设置新值
    if new_device_value is not None:
        setattr(batch, device_field, new_device_value)

    if sum_field is not None:
        sum_value = (
            cpu_value.sum().item()
            if isinstance(cpu_value, torch.Tensor)
            else sum(cpu_value)
        )
        setattr(batch, sum_field, sum_value)


def _compute_mask_offset(seq_index: int, spec_info: Optional[EagleVerifyInput]) -> int:
    if seq_index == 0:
        return 0

    offset = 0
    max_seq_len = min(seq_index, spec_info.seq_lens_cpu.shape[0])
    for i in range(max_seq_len):
        offset += (
            spec_info.seq_lens_cpu[i] + spec_info.draft_token_num
        ) * spec_info.draft_token_num
    return offset


def split_spec_info(
    spec_info: Optional[EagleVerifyInput],
    start_seq_index: int,
    end_seq_index: int,
    start_token_index: int,
    end_token_index: int,
):
    if spec_info is None:
        return None
    if spec_info.draft_token is not None:
        draft_token = spec_info.draft_token[start_token_index:end_token_index]
    else:
        draft_token = None
    if spec_info.custom_mask is not None and spec_info.draft_token is not None:
        custom_mask_start = _compute_mask_offset(start_seq_index, spec_info)
        if end_seq_index == spec_info.seq_lens_cpu.shape[0]:
            custom_mask_end = spec_info.custom_mask.shape[0]
        else:
            custom_mask_end = _compute_mask_offset(end_seq_index, spec_info)

        if custom_mask_end > custom_mask_start:
            custom_mask = spec_info.custom_mask[custom_mask_start:custom_mask_end]
        else:
            custom_mask = spec_info.custom_mask
    else:
        custom_mask = spec_info.custom_mask
    if spec_info.positions is not None:
        positions = spec_info.positions[start_token_index:end_token_index]
    else:
        positions = None
    if spec_info.retrive_index is not None:
        retrive_index = spec_info.retrive_index[start_seq_index:end_seq_index]
    else:
        retrive_index = None
    if spec_info.retrive_next_token is not None:
        retrive_next_token = spec_info.retrive_next_token[start_seq_index:end_seq_index]
    else:
        retrive_next_token = None
    if spec_info.retrive_next_sibling is not None:
        retrive_next_sibling = spec_info.retrive_next_sibling[
            start_seq_index:end_seq_index
        ]
    else:
        retrive_next_sibling = None
    if spec_info.retrive_cum_len is not None:
        retrive_cum_len = spec_info.retrive_cum_len[start_seq_index:end_seq_index]
    else:
        retrive_cum_len = None

    if spec_info.seq_lens_cpu is not None:
        seq_lens_cpu = spec_info.seq_lens_cpu[start_seq_index:end_seq_index]
    else:
        seq_lens_cpu = None
    if seq_lens_cpu is not None:
        seq_lens_sum = seq_lens_cpu.sum()
    else:
        seq_lens_sum = None
    output_spec_info = replace(
        spec_info,
        custom_mask=custom_mask,
        draft_token=draft_token,
        positions=positions,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        retrive_cum_len=retrive_cum_len,
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_sum=seq_lens_sum,
    )
    return output_spec_info


def compute_split_token_index(
    split_seq_index: int,
    forward_mode: "ForwardMode",
    extend_seq_lens: Optional[Sequence[int]],
    token_num_per_seq: Optional[int],
) -> int:
    if forward_mode == ForwardMode.EXTEND:
        assert extend_seq_lens is not None
        if _is_two_chunk_split_enabled(extend_seq_lens):
            return sum(extend_seq_lens) // 2
        return sum(extend_seq_lens[:split_seq_index])
    elif forward_mode.is_target_verify() or forward_mode.is_decode():
        assert token_num_per_seq is not None
        return split_seq_index * token_num_per_seq
    elif forward_mode.is_idle():
        assert split_seq_index == 0
        return 0
    else:
        raise NotImplementedError


def compute_split_indices_for_cuda_graph_replay(
    forward_mode: ForwardMode,
    cuda_graph_num_tokens: int,
    spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
):
    forward_mode_for_tbo_split = (
        forward_mode if forward_mode != ForwardMode.IDLE else ForwardMode.DECODE
    )
    token_num_per_seq = get_token_num_per_seq(
        forward_mode=forward_mode, spec_info=spec_info
    )
    tbo_split_seq_index = compute_split_seq_index(
        forward_mode=forward_mode_for_tbo_split,
        num_tokens=cuda_graph_num_tokens,
        extend_lens=None,
        token_num_per_seq=token_num_per_seq,
    )
    tbo_split_token_index = compute_split_token_index(
        split_seq_index=tbo_split_seq_index,
        forward_mode=forward_mode_for_tbo_split,
        extend_seq_lens=None,
        token_num_per_seq=token_num_per_seq,
    )
    return tbo_split_seq_index, tbo_split_token_index


# -------------------------------- Preparation ---------------------------------------


class TboCudaGraphRunnerPlugin:
    def __init__(self):
        self._tbo_children_num_token_non_padded = torch.zeros((2,), dtype=torch.int32)

    def capture_one_batch_size(self, batch: ForwardBatch, num_tokens: int):
        if not global_server_args_dict["enable_two_batch_overlap"]:
            return
        token_num_per_seq = get_token_num_per_seq(
            forward_mode=batch.forward_mode, spec_info=batch.spec_info
        )

        batch.tbo_split_seq_index = compute_split_seq_index(
            forward_mode=batch.forward_mode,
            num_tokens=num_tokens,
            extend_lens=None,
            token_num_per_seq=token_num_per_seq,
        )
        # For simplicity, when two_batch_overlap is enabled, we only capture CUDA Graph for tbo=true
        assert batch.tbo_split_seq_index is not None, f"{num_tokens=}"

        self._tbo_children_num_token_non_padded[...] = (
            TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded(batch)
        )

        TboForwardBatchPreparer.prepare_raw(
            batch,
            tbo_children_num_token_non_padded=self._tbo_children_num_token_non_padded,
        )

    def replay_prepare(
        self,
        forward_mode: ForwardMode,
        bs: int,
        num_token_non_padded: int,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        token_num_per_seq = get_token_num_per_seq(
            forward_mode=forward_mode, spec_info=spec_info
        )
        tbo_split_seq_index, tbo_split_token_index = (
            compute_split_indices_for_cuda_graph_replay(
                forward_mode=forward_mode,
                cuda_graph_num_tokens=bs * token_num_per_seq,
                spec_info=spec_info,
            )
        )

        self._tbo_children_num_token_non_padded[...] = (
            TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded_raw(
                tbo_split_token_index=tbo_split_token_index,
                num_token_non_padded=num_token_non_padded,
            )
        )


class TboDPAttentionPreparer:
    def prepare_all_gather(
        self,
        local_batch: ScheduleBatch,
        deepep_mode: DeepEPMode,
        enable_deepep_moe: bool,
        enable_two_batch_overlap: bool,
    ):
        self.enable_two_batch_overlap = enable_two_batch_overlap

        if local_batch is not None:
            token_num_per_seq = get_token_num_per_seq(
                forward_mode=local_batch.forward_mode, spec_info=local_batch.spec_info
            )

            if (
                local_batch.forward_mode.is_target_verify()
                or local_batch.forward_mode.is_decode()
            ):
                num_tokens = local_batch.batch_size() * token_num_per_seq
            else:
                num_tokens = local_batch.extend_num_tokens
            
            # TBO强制模式：跳过batch size检查，直接计算split index
            batch_size = local_batch.batch_size() if hasattr(local_batch, 'batch_size') else 0
            if _TBO_DEBUG:
                _tbo_log(f"TboDPAttentionPreparer: FORCE MODE, batch_size={batch_size}, proceeding with TBO")
            
            self.local_tbo_split_seq_index = compute_split_seq_index(
                forward_mode=local_batch.forward_mode,
                num_tokens=num_tokens,
                extend_lens=local_batch.extend_lens,
                token_num_per_seq=token_num_per_seq,
            )
            resolved_deepep_mode = deepep_mode.resolve(local_batch.is_extend_in_batch)
            local_can_run_tbo = (self.local_tbo_split_seq_index is not None) and not (
                (
                    local_batch.forward_mode.is_extend()
                    and not local_batch.forward_mode.is_target_verify()
                )
                and enable_deepep_moe
                and (resolved_deepep_mode == DeepEPMode.LOW_LATENCY)
            )
            if _TBO_DEBUG:
                _tbo_log(
                    f"tbo_gate_local: enable_two_batch_overlap={enable_two_batch_overlap}, forward_mode={local_batch.forward_mode}, bs={local_batch.batch_size() if hasattr(local_batch,'batch_size') else None}, extend_num_tokens={getattr(local_batch,'extend_num_tokens',None)}, split_seq_idx={self.local_tbo_split_seq_index}, local_can_run_tbo={local_can_run_tbo}, resolved_deepep_mode={resolved_deepep_mode}, enable_deepep_moe={enable_deepep_moe}"
                )
        else:
            self.local_tbo_split_seq_index = 0
            local_can_run_tbo = True

        local_forward_mode = self._compute_local_forward_mode(local_batch)

        return local_can_run_tbo, local_forward_mode

    def compute_output(self, partial_global_info):
        local_can_run_tbo_aggregated = min(partial_global_info[:, 0, 0].tolist())
        forward_modes = partial_global_info[:, 0, 1].tolist()

        global_forward_mode, forward_mode_agree = self._compute_global_forward_mode(
            forward_modes
        )

        can_run_tbo = (
            self.enable_two_batch_overlap
            and local_can_run_tbo_aggregated
            and forward_mode_agree
        )
        # 强制放行：用于 TP-only 或混杂模式快速验证 TBO
        force_tbo = os.environ.get("SGLANG_TBO_FORCE_TBO", "0") == "1"
        if force_tbo and self.enable_two_batch_overlap:
            can_run_tbo = True
            # 如果全局模式未解析，默认按 EXTEND 处理以运行 prefill TBO
            if global_forward_mode is None:
                global_forward_mode = ForwardMode.EXTEND
            # 兜底 split 索引：若本地未计算到，则设为 1（至少能分成两份）
            if self.local_tbo_split_seq_index is None:
                self.local_tbo_split_seq_index = 1

        if _TBO_DEBUG:
            _tbo_log(
                f"tbo_gate_global: can_run_tbo={can_run_tbo}, enable_two_batch_overlap={self.enable_two_batch_overlap}, local_can_run_tbo_aggregated={local_can_run_tbo_aggregated}, forward_mode_agree={forward_mode_agree}, global_forward_mode={global_forward_mode}, local_split_seq_idx={self.local_tbo_split_seq_index}, force_tbo={force_tbo}"
            )

        tbo_split_seq_index = self.local_tbo_split_seq_index if can_run_tbo else None
        global_forward_mode = global_forward_mode if can_run_tbo else None
        return tbo_split_seq_index, global_forward_mode

    @staticmethod
    def _compute_local_forward_mode(local_batch):
        return (
            local_batch.forward_mode if local_batch is not None else ForwardMode.IDLE
        ).value

    @staticmethod
    def _compute_global_forward_mode(forward_modes):
        forward_modes_excluding_idle = [
            x for x in forward_modes if x != ForwardMode.IDLE.value
        ]

        if not forward_modes_excluding_idle:
            return ForwardMode.IDLE, False

        # 原逻辑：必须完全一致
        forward_mode_agree = TboDPAttentionPreparer._is_all_same(
            forward_modes_excluding_idle
        )

        # 放宽：若仅包含 DECODE/TARGET_VERIFY 的混合，也按 DECODE 处理
        unique_modes = set(forward_modes_excluding_idle)
        relaxed_decode_ok = unique_modes.issubset(
            {ForwardMode.DECODE.value, ForwardMode.TARGET_VERIFY.value}
        )
        # 放宽：若仅包含 EXTEND/TARGET_VERIFY 的混合，也按 EXTEND 处理
        relaxed_extend_ok = unique_modes.issubset(
            {ForwardMode.EXTEND.value, ForwardMode.TARGET_VERIFY.value}
        )

        # 环境变量强制放行
        force_decode = os.environ.get("SGLANG_TBO_FORCE_DECODE", "0") == "1"
        force_extend = os.environ.get("SGLANG_TBO_FORCE_EXTEND", "0") == "1"

        agree = (
            forward_mode_agree
            or (relaxed_decode_ok and force_decode)
            or (relaxed_extend_ok and force_extend)
        )
        if agree:
            if relaxed_decode_ok and (force_decode or forward_mode_agree):
                resolved = ForwardMode.DECODE
            elif relaxed_extend_ok and (force_extend or forward_mode_agree):
                resolved = ForwardMode.EXTEND
            else:
                resolved = ForwardMode(forward_modes_excluding_idle[0])
        else:
            resolved = None

        if _TBO_DEBUG:
            _tbo_log(
                f"global_mode_check: forward_modes={forward_modes}, excl_idle={forward_modes_excluding_idle}, "
                f"unique={list(unique_modes)}, forward_mode_agree={forward_mode_agree}, "
                f"relaxed_decode_ok={relaxed_decode_ok}, relaxed_extend_ok={relaxed_extend_ok}, "
                f"force_decode={force_decode}, force_extend={force_extend}, agree={agree}, resolved={resolved}"
            )

        return resolved, agree

    @staticmethod
    def _is_all_same(x):
        return all(value == x[0] for value in x)


class TboForwardBatchPreparer:
    @classmethod
    def prepare(cls, batch: ForwardBatch, is_draft_worker: bool = False):
        if batch.tbo_split_seq_index is None or is_draft_worker:
            return

        # TBO强制模式：跳过batch size验证，强制执行TBO
        if _TBO_DEBUG:
            _tbo_log(f"TboForwardBatchPreparer.prepare: FORCE MODE, batch_size={batch.batch_size}, proceeding with TBO")

        tbo_children_num_token_non_padded = (
            cls.compute_tbo_children_num_token_non_padded(batch)
        )
        cls.prepare_raw(
            batch, tbo_children_num_token_non_padded=tbo_children_num_token_non_padded
        )

    @classmethod
    def prepare_raw(
        cls, batch: ForwardBatch, tbo_children_num_token_non_padded: torch.Tensor
    ):
        from sglang.srt.layers.attention.tbo_backend import TboAttnBackend

        tbo_split_token_index = cls._compute_split_token_index(batch)

        is_enable_two_chunk = (
            batch.forward_mode == ForwardMode.EXTEND
            and _is_two_chunk_split_enabled(batch.extend_seq_lens_cpu)
        )

        if _tbo_debug:
            logger.info(
                f"TboForwardBatchPreparer.prepare "
                f"is_enable_two_chunk={is_enable_two_chunk} "
                f"tbo_split_seq_index={batch.tbo_split_seq_index} "
                f"tbo_split_token_index={tbo_split_token_index} "
                f"extend_seq_lens={batch.extend_seq_lens_cpu} "
                f"bs={batch.batch_size} "
                f"forward_mode={batch.forward_mode}"
            )

        assert isinstance(batch.attn_backend, TboAttnBackend)
        attn_backend_child_a, attn_backend_child_b = batch.attn_backend.children

        [out_num_token_non_padded_a, out_num_token_non_padded_b] = (
            tbo_children_num_token_non_padded
        )

        child_a = cls.filter_batch(
            batch,
            start_token_index=0,
            end_token_index=tbo_split_token_index,
            start_seq_index=0,
            end_seq_index=(
                batch.tbo_split_seq_index + 1
                if is_enable_two_chunk
                else batch.tbo_split_seq_index
            ),
            output_attn_backend=attn_backend_child_a,
            out_num_token_non_padded=out_num_token_non_padded_a,
        )
        child_b = cls.filter_batch(
            batch,
            start_token_index=tbo_split_token_index,
            end_token_index=batch.input_ids.shape[0],
            start_seq_index=batch.tbo_split_seq_index,
            end_seq_index=batch.batch_size,
            output_attn_backend=attn_backend_child_b,
            out_num_token_non_padded=out_num_token_non_padded_b,
        )

        if is_enable_two_chunk:
            cls.derive_fields_related_to_seq_len_for_two_chunk(
                batch,
                child_a=child_a,
                child_b=child_b,
                tbo_split_seq_index=batch.tbo_split_seq_index,
            )

        assert batch.tbo_children is None
        batch.tbo_children = [child_a, child_b]

    @classmethod
    def derive_fields_related_to_seq_len_for_two_chunk(
        cls,
        batch: ForwardBatch,
        *,
        child_a: ForwardBatch,
        child_b: ForwardBatch,
        tbo_split_seq_index: int,
    ):
        extend_seq_lens_cpu = batch.extend_seq_lens_cpu
        overall_seq_lens_sum = sum(extend_seq_lens_cpu)
        half_seq_lens_sum = overall_seq_lens_sum // 2
        left_last_seq_token_num = half_seq_lens_sum - sum(
            extend_seq_lens_cpu[:tbo_split_seq_index]
        )
        right_first_seq_token_num = (
            extend_seq_lens_cpu[tbo_split_seq_index] - left_last_seq_token_num
        )

        # making deepcopy to be extra safe
        child_a.extend_seq_lens_cpu = copy.deepcopy(child_a.extend_seq_lens_cpu)
        child_a.extend_seq_lens_cpu[-1] = left_last_seq_token_num
        child_b.extend_seq_lens_cpu = copy.deepcopy(child_b.extend_seq_lens_cpu)
        child_b.extend_seq_lens_cpu[0] = right_first_seq_token_num
        for child in [child_a, child_b]:
            _update_device_and_sum_field_from_cpu_field(
                batch=child,
                cpu_field="extend_seq_lens_cpu",
                device_field="extend_seq_lens",
                sum_field="extend_num_tokens",
            )

        assert (
            child_a.extend_num_tokens == half_seq_lens_sum
        ), f"{child_a.extend_num_tokens=}, {half_seq_lens_sum=}"

        child_a.seq_lens_cpu = copy.deepcopy(child_a.seq_lens_cpu)
        child_a.seq_lens_cpu[-1] = (
            child_a.extend_seq_lens_cpu[-1] + child_a.extend_prefix_lens_cpu[-1]
        )
        _update_device_and_sum_field_from_cpu_field(
            batch=child_a,
            cpu_field="seq_lens_cpu",
            device_field="seq_lens",
            sum_field="seq_lens_sum",
        )

        child_b.extend_prefix_lens_cpu = copy.deepcopy(child_b.extend_prefix_lens_cpu)
        child_b.extend_prefix_lens_cpu[0] += left_last_seq_token_num
        _update_device_and_sum_field_from_cpu_field(
            batch=child_b,
            cpu_field="extend_prefix_lens_cpu",
            device_field="extend_prefix_lens",
            sum_field=None,
        )
        _, child_b.extend_start_loc = compute_position(
            global_server_args_dict["attention_backend"],
            child_b.extend_prefix_lens,
            child_b.extend_seq_lens,
            child_b.extend_num_tokens,
        )

    @classmethod
    def filter_batch(
        cls,
        batch: ForwardBatch,
        *,
        start_token_index: int,
        end_token_index: int,
        start_seq_index: int,
        end_seq_index: int,
        output_attn_backend: AttentionBackend,
        out_num_token_non_padded: torch.Tensor,
    ):
        assert (
            end_token_index >= start_token_index
        ), f"{end_token_index=}, {start_token_index=}, batch={batch}"
        num_tokens = batch.input_ids.shape[0]
        num_seqs = batch.batch_size

        output_dict = dict()

        for key in [
            "input_ids",
            "positions",
            "out_cache_loc",
        ]:
            old_value = getattr(batch, key)
            assert (
                old_value.shape[0] == num_tokens
            ), f"{key=} {old_value=} {num_tokens=} {batch=}"
            output_dict[key] = old_value[start_token_index:end_token_index]

        for key in [
            "req_pool_indices",
            "seq_lens",
            "seq_lens_cpu",
            "extend_seq_lens",
            "extend_prefix_lens",
            "extend_start_loc",
            "extend_prefix_lens_cpu",
            "extend_seq_lens_cpu",
            "extend_logprob_start_lens_cpu",
            "lora_ids",
        ]:
            old_value = getattr(batch, key)
            if old_value is None:
                continue
            elif batch.forward_mode.is_target_verify() and (
                key == "extend_seq_lens"
                or key == "extend_prefix_lens"
                or key == "extend_start_loc"
                or key == "extend_prefix_lens_cpu"
                or key == "extend_seq_lens_cpu"
                or key == "extend_logprob_start_lens_cpu"
            ):
                output_dict[key] = None
                continue
            assert (
                len(old_value) == num_seqs
            ), f"{key=} {old_value=} {num_seqs=} {batch=}"
            output_dict[key] = old_value[start_seq_index:end_seq_index]

        spec_info = getattr(batch, "spec_info")
        output_spec_info = split_spec_info(
            spec_info=spec_info,
            start_token_index=start_token_index,
            end_token_index=end_token_index,
            start_seq_index=start_seq_index,
            end_seq_index=end_seq_index,
        )
        output_dict["spec_info"] = output_spec_info
        for key in [
            "forward_mode",
            "is_extend_in_batch",
            "return_logprob",
            "req_to_token_pool",
            "token_to_kv_pool",
            "can_run_dp_cuda_graph",
            "global_forward_mode",
            "spec_algorithm",
            "capture_hidden_mode",
            "padded_static_len",
            "mrope_positions",  # only used by qwen2-vl, thus not care
            "split_index",  # for split prefill
            "orig_seq_lens",  # only used by qwen-1m, thus not care
        ]:
            output_dict[key] = getattr(batch, key)
        if not batch.forward_mode.is_target_verify():
            assert (
                _compute_extend_num_tokens(batch.input_ids, batch.forward_mode)
                == batch.extend_num_tokens
            ), f"{batch=}"
        extend_num_tokens = _compute_extend_num_tokens(
            output_dict["input_ids"], output_dict["forward_mode"]
        )

        # TODO improve, e.g. unify w/ `init_raw`
        if (
            global_server_args_dict["moe_dense_tp_size"] == 1
            and batch.gathered_buffer is not None
        ):
            sum_len = end_token_index - start_token_index
            gathered_buffer = torch.zeros(
                (sum_len, batch.gathered_buffer.shape[1]),
                dtype=batch.gathered_buffer.dtype,
                device=batch.gathered_buffer.device,
            )
        else:
            gathered_buffer = None

        # TBO强制模式：安全的tensor创建，确保CUDA状态正确
        child_num_tokens = end_token_index - start_token_index
        if batch.global_num_tokens_gpu is not None:
            # 强制重置CUDA环境，确保tensor创建的稳定性
            if torch.cuda.is_available():
                try:
                    # 重置CUDA环境
                    torch.cuda.synchronize()
                    current_device = torch.cuda.current_device()
                    torch.cuda.set_device(current_device)
                    torch.cuda.empty_cache()
                except Exception as reset_e:
                    if _TBO_DEBUG:
                        _tbo_log(f"TBO filter_batch: CUDA reset warning: {reset_e}")
            
            # 安全创建子批次tensor - 多重保护机制
            original_device = batch.global_num_tokens_gpu.device
            original_dtype = batch.global_num_tokens_gpu.dtype
            
            try:
                # 方法1：直接在目标设备上创建
                if torch.cuda.is_available() and original_device.type == 'cuda':
                    target_device_idx = original_device.index if original_device.index is not None else torch.cuda.current_device()
                    
                    # 确保设备状态正确
                    with torch.cuda.device(target_device_idx):
                        # 验证设备可访问性
                        torch.cuda.synchronize()
                        child_global_num_tokens_gpu = torch.tensor(
                            [child_num_tokens], 
                            dtype=original_dtype,
                            device=original_device
                        )
                else:
                    # CPU设备
                    child_global_num_tokens_gpu = torch.tensor(
                        [child_num_tokens], 
                        dtype=original_dtype,
                        device=original_device
                    )
                
                if _TBO_DEBUG:
                    _tbo_log(f"TBO filter_batch: Successfully created child tensor on {original_device}")
                    
            except Exception as e:
                if _TBO_DEBUG:
                    _tbo_log(f"TBO filter_batch: Primary tensor creation failed: {e}")
                
                # 方法2：先在CPU创建，再转移（更安全）
                try:
                    cpu_tensor = torch.tensor([child_num_tokens], dtype=torch.int32)
                    if original_device.type == 'cuda' and torch.cuda.is_available():
                        child_global_num_tokens_gpu = cpu_tensor.to(device=original_device, dtype=original_dtype, non_blocking=False)
                    else:
                        child_global_num_tokens_gpu = cpu_tensor.to(dtype=original_dtype)
                    
                    if _TBO_DEBUG:
                        _tbo_log(f"TBO filter_batch: Used CPU->GPU transfer method successfully")
                        
                except Exception as transfer_e:
                    if _TBO_DEBUG:
                        _tbo_log(f"TBO filter_batch: CPU->GPU transfer also failed: {transfer_e}")
                    
                    # 方法3：最后手段 - 使用None（让上层处理）
                    child_global_num_tokens_gpu = None
                    if _TBO_DEBUG:
                        _tbo_log("TBO filter_batch: Using None for child_global_num_tokens_gpu")
        else:
            child_global_num_tokens_gpu = None

        output_dict.update(
            dict(
                batch_size=end_seq_index - start_seq_index,
                seq_lens_sum=(
                    output_dict["seq_lens_cpu"].sum()
                    if "seq_lens_cpu" in output_dict
                    else None
                ),
                extend_num_tokens=extend_num_tokens,
                attn_backend=output_attn_backend,
                num_token_non_padded=out_num_token_non_padded,
                tbo_split_seq_index=None,
                tbo_parent_token_range=(start_token_index, end_token_index),
                tbo_children=None,
                global_num_tokens_gpu=child_global_num_tokens_gpu,
                global_num_tokens_cpu=child_num_tokens,
                dp_padding_mode=None,
                gathered_buffer=gathered_buffer,
                global_num_tokens_for_logprob_gpu=None,
                global_num_tokens_for_logprob_cpu=None,
                sampling_info=None,
                # For logits and logprobs post processing, thus we do not care
                temp_scaled_logprobs=False,
                temperature=None,
                top_p_normalized_logprobs=False,
                top_p=None,
                mm_inputs=None,
                top_logprobs_nums=None,
                token_ids_logprobs=None,
                next_token_logits_buffer=None,
            )
        )

        errors = []
        for field in dataclasses.fields(ForwardBatch):
            if getattr(batch, field.name) is not None and field.name not in output_dict:
                errors.append(
                    f"Field {field.name} has value, but is not yet supported (value={getattr(batch, field.name)} batch={batch})"
                )
        if len(errors) > 0:
            raise Exception(f"{len(errors)} errors happen:\n" + "\n\n".join(errors))

        return ForwardBatch(**output_dict)

    @classmethod
    def compute_tbo_children_num_token_non_padded(cls, batch: ForwardBatch):
        return cls.compute_tbo_children_num_token_non_padded_raw(
            tbo_split_token_index=cls._compute_split_token_index(batch),
            num_token_non_padded=len(batch.input_ids),
        )

    @classmethod
    def compute_tbo_children_num_token_non_padded_raw(
        cls, tbo_split_token_index: int, num_token_non_padded: int
    ):
        # TODO we may make padding on both sub-batches to make it slightly more balanced
        value_a = min(tbo_split_token_index, num_token_non_padded)
        value_b = max(0, num_token_non_padded - tbo_split_token_index)
        return torch.tensor([value_a, value_b], dtype=torch.int32).to(
            device=global_server_args_dict["device"], non_blocking=True
        )

    @classmethod
    def _compute_split_token_index(cls, batch: ForwardBatch):
        token_num_per_seq = get_token_num_per_seq(
            forward_mode=batch.forward_mode, spec_info=batch.spec_info
        )
        return compute_split_token_index(
            split_seq_index=batch.tbo_split_seq_index,
            forward_mode=batch.forward_mode,
            extend_seq_lens=batch.extend_seq_lens_cpu,
            token_num_per_seq=token_num_per_seq,
        )


def _compute_extend_num_tokens(input_ids, forward_mode: ForwardMode):
    if (
        forward_mode.is_decode()
        or forward_mode.is_idle()
        or forward_mode.is_target_verify()
    ):
        return None
    elif forward_mode.is_extend():
        return input_ids.shape[0]
    raise NotImplementedError


# -------------------------------- Execution ---------------------------------------


def model_forward_maybe_tbo(
    layers: torch.nn.ModuleList,
    enable_tbo: bool,
    input_data_scatter_mode: ScatterMode,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    zero_allocator: Optional[BumpAllocator] = None,
):
    inputs = dict(
        positions=positions,
        hidden_states=hidden_states,
        forward_batch=forward_batch,
        residual=residual,
        zero_allocator=zero_allocator,
    )
    
    # 增强的TBO安全检查
    tbo_safety_checks_passed = True
    disable_reason = ""
    
    # TBO强制模式 - 基础CUDA环境检查，确保执行环境稳定
    if torch.cuda.is_available():
        try:
            # 基础CUDA环境检查和重置
            torch.cuda.synchronize()
            current_device = torch.cuda.current_device()
            torch.cuda.set_device(current_device)
            
            # 重置到默认流状态
            default_stream = torch.cuda.default_stream()
            torch.cuda.set_stream(default_stream)
            
            if _TBO_DEBUG:
                _tbo_log("TBO FORCE MODE: Basic CUDA environment validated and reset")
        except Exception as basic_e:
            if _TBO_DEBUG:
                _tbo_log(f"TBO FORCE MODE: Basic CUDA check failed: {basic_e}")
    
    if _TBO_DEBUG:
        _tbo_log("TBO FORCE MODE: Proceeding with enforced TBO execution")
    
    # 强制初始化operations_strategy - 不允许失败
    layer_input_scatter_mode = layers[0].layer_scatter_modes.layer_input_mode
    operations_strategy = OperationsStrategy.init_new_tbo(
        layers, forward_batch.global_forward_mode
    )
    if _TBO_DEBUG:
        _tbo_log("TBO operations_strategy initialized successfully")
    
    # TBO强制模式：跳过所有safety checks，强制执行TBO
    if _TBO_DEBUG:
        _tbo_log(
            f"TBO FORCE MODE: enable={enable_tbo}, mode={forward_batch.global_forward_mode}, delta={getattr(operations_strategy, 'tbo_delta_stages', 'N/A')}"
        )
        _tbo_log("TBO FORCE MODE: Skipping all safety checks, enforcing TBO execution")
    
    if enable_tbo:
        return _model_forward_tbo(
            inputs=inputs,
            operations_strategy=operations_strategy,
            input_data_scatter_mode=input_data_scatter_mode,
            layer_input_scatter_mode=layer_input_scatter_mode,
        )
    else:
        # 当禁用TBO时，使用标准的逐层forward，避免TBO operations
        if _TBO_DEBUG:
            _tbo_log(f"using standard layer-by-layer forward for {len(layers)} layers")
        current_hidden_states = hidden_states
        current_residual = residual
        for layer in layers:
            current_hidden_states, current_residual = layer(
                positions=positions,
                hidden_states=current_hidden_states,
                forward_batch=forward_batch,
                residual=current_residual,
            )
        return current_hidden_states, current_residual


def _model_forward_tbo(
    inputs: dict,
    operations_strategy: OperationsStrategy,
    input_data_scatter_mode: ScatterMode,
    layer_input_scatter_mode: ScatterMode,
):
    if _TBO_DEBUG:
        _tbo_log(f"_model_forward_tbo ENTRY: operations_strategy.tbo_delta_stages={operations_strategy.tbo_delta_stages}")
    
    # 获取并分割输入
    split_inputs = _model_forward_tbo_split_inputs(
        hidden_states=inputs["hidden_states"],
        residual=inputs["residual"],
        positions=inputs["positions"],
        forward_batch=inputs["forward_batch"],
        zero_allocator=inputs.get("zero_allocator"),
        input_data_scatter_mode=input_data_scatter_mode,
        layer_input_scatter_mode=layer_input_scatter_mode,
    )
    if _TBO_DEBUG:
        _tbo_log(f"split_inputs: micro_batch_a.shape={split_inputs[0]['hidden_states'].shape}, micro_batch_b.shape={split_inputs[1]['hidden_states'].shape}")
    
    # 检查空批次回退
    for i, micro_batch_inputs in enumerate(split_inputs):
        if micro_batch_inputs['hidden_states'].shape[0] == 0:
            if _TBO_DEBUG:
                _tbo_log(f"micro_batch_{i} is empty (shape[0]=0), falling back to execute_operations on original inputs")
            # 有空批次，回退到标准执行
            outputs = execute_operations(inputs, operations_strategy.operations)
            return outputs["hidden_states"], outputs["residual"]
    
    if _TBO_DEBUG:
        _tbo_log(f"🚀 ENTERING TBO OVERLAP EXECUTION with {len(split_inputs)} micro-batches")
        _tbo_log(f"   operations count: {len(operations_strategy.operations)}")
        _tbo_log(f"   delta_stages: {operations_strategy.tbo_delta_stages}")
        _tbo_log(f"   batch info: forward_mode={inputs['forward_batch'].forward_mode}, batch_size={inputs['forward_batch'].batch_size}")
    
    # 创建micro-batch上下文管理（参考vLLM的UBatchContext）
    batch_id = f"tbo_batch_{id(inputs['forward_batch'])}"
    micro_contexts = create_micro_batch_contexts(
        batch_id=batch_id,
        num_micro_batches=len(split_inputs),
        forward_batch=inputs['forward_batch'],
        batch_info={
            'forward_mode': inputs['forward_batch'].forward_mode,
            'batch_size': inputs['forward_batch'].batch_size,
            'operations_strategy': operations_strategy
        }
    )
    
    try:
        # 🔧 TBO前环境重置：确保每次TBO都从干净的CUDA环境开始
        enable_tbo_cuda_sync = os.environ.get("SGLANG_TBO_CUDA_SYNC", "1") == "1"
        if enable_tbo_cuda_sync and torch.cuda.is_available():
            try:
                # 1. 重置CUDA环境到已知状态
                torch.cuda.synchronize()
                
                # 2. 确保所有设备的流状态正确
                for device_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_id):
                        default_stream = torch.cuda.default_stream()
                        torch.cuda.set_stream(default_stream)
                        default_stream.synchronize()
                
                # 3. 清理内存状态
                torch.cuda.empty_cache()
                
                # 4. 重置当前设备
                current_device = torch.cuda.current_device()
                torch.cuda.set_device(current_device)
                
                # 5. 尝试清除任何累积的CUDA错误状态
                try:
                    # 执行一个简单的CUDA操作来"flush"任何潜在错误
                    test_tensor = torch.tensor([1.0], device=current_device)
                    _ = test_tensor + 1.0
                    del test_tensor, _
                except Exception:
                    pass  # 忽略测试操作的错误
                
                if _TBO_DEBUG:
                    _tbo_log("TBO pre-execution: CUDA environment reset successful")
            except Exception as e:
                if _TBO_DEBUG:
                    _tbo_log(f"TBO pre-execution: CUDA reset failed: {e}")
        
        # 执行重叠操作 - 这是TBO的核心！（参考vLLM的微批处理模式）
        try:
            output_a, output_b = execute_overlapped_operations(
                inputs_arr=[split_inputs[0], split_inputs[1]],
                operations_arr=[operations_strategy.operations, operations_strategy.operations],
                delta_stages=[0, operations_strategy.tbo_delta_stages],
            )
        except Exception as e:
            if _TBO_DEBUG:
                _tbo_log(f"TBO execution error: {e}")
            # 发生错误时强制同步并清理
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            raise
        
        # 标记micro-batch执行完成
        if len(micro_contexts) >= 2:
            micro_contexts[0].mark_execution_done(result=output_a)
            micro_contexts[1].mark_execution_done(result=output_b)
            
    finally:
        # 清理micro-batch上下文
        cleanup_micro_batch_contexts(batch_id)
    
    if _TBO_DEBUG:
        _tbo_log(f"✅ TBO OVERLAP EXECUTION COMPLETED")
        _tbo_log(f"   output_a.shape: {output_a['hidden_states'].shape if isinstance(output_a, dict) and 'hidden_states' in output_a else 'N/A'}")
        _tbo_log(f"   output_b.shape: {output_b['hidden_states'].shape if isinstance(output_b, dict) and 'hidden_states' in output_b else 'N/A'}")
    
    # 合并输出
    hidden_states, residual = _model_forward_tbo_merge_outputs(output_a, output_b)
    
    # TBO后的状态恢复：确保主forward_batch的关键状态正确
    main_batch = inputs["forward_batch"]
    if hasattr(main_batch, 'tbo_children') and main_batch.tbo_children:
        try:
            # 恢复主batch的token数量信息（安全版本）
            total_tokens = hidden_states.shape[0] if hidden_states is not None else 0
            if main_batch.global_num_tokens_gpu is None and total_tokens > 0:
                try:
                    # 安全地创建tensor
                    device = hidden_states.device if hidden_states is not None else 'cpu'
                    if torch.cuda.is_available() and device.type == 'cuda':
                        with torch.cuda.device(device.index if device.index is not None else 0):
                            main_batch.global_num_tokens_gpu = torch.tensor(
                                [total_tokens], dtype=torch.int32, device=device
                            )
                    else:
                        main_batch.global_num_tokens_gpu = torch.tensor(
                            [total_tokens], dtype=torch.int32, device=device
                        )
                except Exception as tensor_e:
                    # tensor创建失败，使用CPU fallback
                    if _TBO_DEBUG:
                        _tbo_log(f"TBO post-merge: tensor creation failed, using CPU: {tensor_e}")
                    main_batch.global_num_tokens_gpu = torch.tensor([total_tokens], dtype=torch.int32)
                    
            if _TBO_DEBUG:
                _tbo_log(f"TBO post-merge: restored main_batch global_num_tokens_gpu to {total_tokens}")
        except Exception as e:
            if _TBO_DEBUG:
                _tbo_log(f"TBO post-merge: failed to restore main_batch state: {e}")
    
    # 🔥 关键修复：彻底重置CUDA状态，确保下一次TBO执行的环境干净
    if enable_tbo_cuda_sync and torch.cuda.is_available():
        try:
            # 1. 全面同步所有设备和流
            torch.cuda.synchronize()
            
            # 2. 重置所有CUDA流到默认状态
            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    default_stream = torch.cuda.default_stream()
                    torch.cuda.set_stream(default_stream)
                    default_stream.synchronize()
            
            # 3. 清理GPU内存碎片
            torch.cuda.empty_cache()
            
            # 4. 重置CUDA错误状态
            try:
                torch.cuda.get_device_properties(torch.cuda.current_device())
            except Exception:
                pass
            
            # 5. 确保当前设备状态正确
            current_device = torch.cuda.current_device()
            torch.cuda.set_device(current_device)
            
            if _TBO_DEBUG:
                _tbo_log("TBO post-execution: Complete CUDA environment reset successful")
        except Exception as e:
            if _TBO_DEBUG:
                _tbo_log(f"TBO post-execution: CUDA reset failed: {e}")
            # 最后的安全网：强制同步和设备重置
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                current_device = torch.cuda.current_device()
                torch.cuda.set_device(current_device)
            except Exception:
                pass
    
    return hidden_states, residual


def _model_forward_non_tbo(inputs, operations_strategy: OperationsStrategy):
    outputs = execute_operations(inputs, operations_strategy.operations)
    return outputs["hidden_states"], outputs["residual"]


def _model_forward_tbo_split_inputs(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    zero_allocator: Optional[BumpAllocator],
    input_data_scatter_mode: ScatterMode,
    layer_input_scatter_mode: ScatterMode,
) -> List[Dict]:
    tbo_splitter_scatter_mode = ScatterMode.TP_ATTN_FULL
    context = CommunicateContext.init_new()

    hidden_states, residual = CommunicateSummableTensorPairFn.execute(
        hidden_states_input_mode=input_data_scatter_mode,
        residual_input_mode=input_data_scatter_mode,
        output_mode=tbo_splitter_scatter_mode,
        hidden_states=hidden_states,
        residual=residual,
        forward_batch=forward_batch,
        context=context,
    )

    inputs_arr = _model_forward_tbo_split_inputs_raw(
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
        zero_allocator=zero_allocator,
    )

    def _post_transform(hidden_states, residual, forward_batch, **kwargs):
        hidden_states, residual = CommunicateSummableTensorPairFn.execute(
            hidden_states_input_mode=tbo_splitter_scatter_mode,
            residual_input_mode=tbo_splitter_scatter_mode,
            output_mode=layer_input_scatter_mode,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            context=context,
        )
        return dict(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            **kwargs,
        )

    return [_post_transform(**inputs) for inputs in inputs_arr]


def _model_forward_tbo_split_inputs_raw(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    zero_allocator: Optional[BumpAllocator],
) -> List[Dict]:
    out = [
        dict(
            **_model_forward_filter_inputs(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                output_forward_batch=output_forward_batch,
                tbo_subbatch_index=tbo_subbatch_index,
            ),
            **(
                dict(zero_allocator=zero_allocator)
                if zero_allocator is not None
                else {}
            ),
        )
        for tbo_subbatch_index, output_forward_batch in enumerate(
            forward_batch.tbo_children
        )
    ]
    if _TBO_DEBUG:
        lens = [
            x["hidden_states"].shape[0] if x["hidden_states"] is not None else 0
            for x in out
        ]
        _tbo_log(f"split_inputs: micro_batch_sizes={lens}")
    return out


def _model_forward_filter_inputs(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    output_forward_batch: ForwardBatch,
    tbo_subbatch_index: int,
) -> Dict:
    token_slice = slice(*output_forward_batch.tbo_parent_token_range)
    start, end = output_forward_batch.tbo_parent_token_range
    if end <= start:
        empty_hs = hidden_states.new_empty((0,) + hidden_states.shape[1:])
        empty_pos = positions.new_empty((0,), dtype=positions.dtype)
        return dict(
            hidden_states=empty_hs,
            residual=None if residual is None else residual.new_empty((0,) + residual.shape[1:]),
            positions=empty_pos,
            forward_batch=output_forward_batch,
            tbo_subbatch_index=tbo_subbatch_index,
        )
    return dict(
        hidden_states=hidden_states[token_slice],
        residual=None if residual is None else residual[token_slice],
        positions=positions[token_slice],
        forward_batch=output_forward_batch,
        tbo_subbatch_index=tbo_subbatch_index,
    )


def _model_forward_tbo_merge_outputs(output_a, output_b):
    def _handle_key(name):
        value_a = output_a[name]
        value_b = output_b[name]
        assert (value_a is None) == (value_b is None)
        if value_a is None:
            return None
        return torch.concat([value_a, value_b], dim=0)

    return _handle_key("hidden_states"), _handle_key("residual")


# -------------------------------- Utilities and wrappers ---------------------------------------


class MaybeTboDeepEPDispatcher:
    def __init__(self, **kwargs):
        num_inner_dispatchers = (
            2 if global_server_args_dict["enable_two_batch_overlap"] else 1
        )
        self._inners = [
            DeepEPDispatcher(**kwargs) for _ in range(num_inner_dispatchers)
        ]

    def _execute(self, name, tbo_subbatch_index: Optional[int] = None, **kwargs):
        return getattr(self._inners[tbo_subbatch_index or 0], name)(**kwargs)

    def dispatch(self, **kwargs) -> DispatchOutput:
        return self._execute("dispatch", **kwargs)

    def dispatch_a(self, **kwargs):
        return self._execute("dispatch_a", **kwargs)

    def dispatch_b(self, **kwargs):
        return self._execute("dispatch_b", **kwargs)

    def combine(self, **kwargs) -> torch.Tensor:
        return self._execute("combine", **kwargs)

    def combine_a(self, **kwargs):
        return self._execute("combine_a", **kwargs)

    def combine_b(self, **kwargs):
        return self._execute("combine_b", **kwargs)
