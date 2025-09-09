# vLLM Dense模型TBO实现技术文档
# 目录
# 系统架构概述
# 核心组件实现
# 微批处理执行逻辑
# 通信重叠机制
# 配置与集成
# 使用示例
# 1. 系统架构概述
# TBO系统主要由以下几个核心组件构成：
# ┌─────────────────────────────────────────────────────────┐
# │                    TBO系统架构                          │
# ├─────────────────────────────────────────────────────────┤
# │  ┌─────────────────┐    ┌─────────────────────────────┐ │
# │  │  UBatchContext  │    │     GPUModelRunner          │ │
# │  │  (上下文管理)    │◄───┤     (执行调度)              │ │
# │  └─────────────────┘    └─────────────────────────────┘ │
# │           ▲                          ▲                  │
# │           │                          │                  │
# │  ┌─────────────────┐    ┌─────────────────────────────┐ │
# │  │ Communication   │    │     智能切分算法             │ │
# │  │ Overlap         │    │     (Batch Splitting)       │ │
# │  └─────────────────┘    └─────────────────────────────┘ │
# └─────────────────────────────────────────────────────────┘
# 2. 核心组件实现
# 2.1 UBatchContext - 微批处理上下文管理器
# 这是TBO系统的核心类，负责管理微批次的执行上下文和同步机制。
# vllm/v1/worker/ubatching.py

import threading
from typing import Optional, Dict
import torch
from vllm import forward_context
from vllm.forward_context import ForwardContext
from vllm.utils import current_stream

class UBatchContext:
    """
    微批处理上下文管理器
    功能：管理单个micro-batch的执行环境，包括CUDA流、同步事件等
    """
    
    def __init__(self,
                 id: int,                                    # 微批次ID (0或1)
                 comm_stream: torch.cuda.Stream,             # 通信专用流
                 compute_stream: torch.cuda.Stream,          # 计算专用流
                 forward_context: ForwardContext,            # 前向推理上下文
                 cpu_wait_event: threading.Event,           # CPU线程等待事件
                 cpu_signal_event: threading.Event,         # CPU线程信号事件
                 gpu_comm_done_event: torch.cuda.Event,     # GPU通信完成事件
                 gpu_compute_done_event: torch.cuda.Event,  # GPU计算完成事件
                 schedule: str = "default"):
        
        # 基础属性
        self.id = id
        self.comm_stream = comm_stream
        self.compute_stream = compute_stream
        self.forward_context = forward_context
        
        # CPU同步事件
        self.cpu_wait_event = cpu_wait_event
        self.cpu_signal_event = cpu_signal_event
        
        # 当前活跃流
        self.current_stream = compute_stream
        
        # 默认GPU事件（向后兼容）
        self._default_gpu_comm_done_event = gpu_comm_done_event
        self._default_gpu_compute_done_event = gpu_compute_done_event
        
        # Per-schedule事件映射（支持不同通信模式）
        self._gpu_comm_done_events: Dict[str, torch.cuda.Event] = {}
        self._gpu_compute_done_events: Dict[str, torch.cuda.Event] = {}
        
        self.schedule = schedule

    def __enter__(self):
        """
        进入微批次执行上下文
        功能：设置当前线程的上下文，等待CPU调度信号
        """
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT[threading.get_ident()] = self
        
        # 清除并等待调度信号
        self.cpu_wait_event.clear()
        self.cpu_wait_event.wait()  # 阻塞直到被调度
        self.cpu_wait_event.clear()
        
        # 恢复执行环境
        self._restore_context()
        
        # 确保从计算流开始
        assert current_stream() == self.compute_stream
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出微批次执行上下文
        功能：清理上下文，发送完成信号
        """
        global _CURRENT_CONTEXT
        _CURRENT_CONTEXT[threading.get_ident()] = None
        
        # 发送完成信号给其他线程
        self.cpu_signal_event.set()
        self.cpu_wait_event.clear()
        
        # 重置到计算流
        self.current_stream = self.compute_stream
        torch.cuda.set_stream(self.current_stream)
        return False

    def _restore_context(self):
        """
        恢复执行上下文
        功能：恢复forward context和CUDA流状态
        """
        forward_context._forward_context = self.forward_context
        torch.cuda.set_stream(self.current_stream)

    def update_stream(self, stream):
        """
        更新当前活跃流
        功能：切换到指定的CUDA流
        """
        self.current_stream = stream
        torch.cuda.set_stream(self.current_stream)

    def _get_compute_event(self, schedule: str) -> torch.cuda.Event:
        """
        获取计算完成事件
        功能：支持per-schedule的事件管理，按需创建
        """
        if schedule == "default":
            return self._default_gpu_compute_done_event
        
        # 按需创建schedule专用事件
        evt = self._gpu_compute_done_events.get(schedule)
        if evt is None:
            evt = torch.cuda.Event()
            self._gpu_compute_done_events[schedule] = evt
        return evt

    def _get_comm_event(self, schedule: str) -> torch.cuda.Event:
        """
        获取通信完成事件
        功能：支持per-schedule的事件管理，按需创建
        """
        if schedule == "default":
            return self._default_gpu_comm_done_event
        
        # 按需创建schedule专用事件
        evt = self._gpu_comm_done_events.get(schedule)
        if evt is None:
            evt = torch.cuda.Event()
            self._gpu_comm_done_events[schedule] = evt
        return evt

    def _signal_comm_done(self, schedule: str):
        """
        标记通信完成
        功能：在通信流上记录完成事件
        """
        self._get_comm_event(schedule).record(self.comm_stream)

    def _signal_compute_done(self, schedule: str):
        """
        标记计算完成
        功能：在计算流上记录完成事件
        """
        self._get_compute_event(schedule).record(self.compute_stream)

    def _wait_compute_done(self, schedule: str):
        """
        等待计算完成
        功能：通信流等待计算流完成
        """
        self.comm_stream.wait_event(self._get_compute_event(schedule))

    def _wait_comm_done(self, schedule: str):
        """
        等待通信完成
        功能：计算流等待通信流完成
        """
        self.compute_stream.wait_event(self._get_comm_event(schedule))

    def _cpu_yield(self):
        """
        CPU线程协作切换
        功能：实现两个micro-batch线程之间的协作调度
        """
        # 确保只有一个线程在运行（关键的正确性检查）
        assert forward_context._forward_context == self.forward_context
        assert current_stream() == self.current_stream
        assert not self.cpu_wait_event.is_set()

        # 唤醒另一个线程，自己进入等待
        self.cpu_signal_event.set()     # 发信号："该你了"
        self.cpu_wait_event.wait()      # 等待信号："我等着"
        self.cpu_wait_event.clear()
        
        # 被唤醒后恢复上下文
        self._restore_context()

    def yield_and_switch_from_compute_to_comm(self, schedule: str = "default"):
        """
        从计算流切换到通信流
        功能：实现计算->通信的流切换，是重叠的关键
        """
        assert current_stream() == self.compute_stream
        
        # 1. 标记计算完成
        self._signal_compute_done(schedule)
        
        # 2. CPU线程切换（让另一个micro-batch运行）
        self._cpu_yield()
        
        # 3. 切换到通信流
        assert self.current_stream == self.compute_stream
        self.update_stream(self.comm_stream)
        
        # 4. 等待计算确实完成
        self._wait_compute_done(schedule)

    def yield_and_switch_from_comm_to_compute(self, schedule: str = "default"):
        """
        从通信流切换到计算流
        功能：实现通信->计算的流切换
        """
        assert current_stream() == self.comm_stream
        
        # 1. 标记通信完成
        self._signal_comm_done(schedule)
        
        # 2. CPU线程切换
        self._cpu_yield()
        
        # 3. 切换到计算流
        assert self.current_stream == self.comm_stream
        self.update_stream(self.compute_stream)
        
        # 4. 等待通信确实完成
        self._wait_comm_done(schedule)


# 全局上下文字典和状态
_CURRENT_CONTEXT: Dict = {}
_UBATCHING_ACTIVE: bool = False

def is_ubatching_globally_enabled() -> bool:
    """
    检查全局是否启用了ubatching
    功能：为通信原语提供快速检查，避免在未启用时的额外开销
    """
    return _UBATCHING_ACTIVE

def get_current_ubatch_context() -> Optional[UBatchContext]:
    """
    获取当前线程的UBatch上下文
    功能：供通信原语查询当前执行环境
    """
    try:
        return _CURRENT_CONTEXT.get(threading.get_ident(), None)
    except Exception:
        # TorchDynamo等编译模式下的兼容性处理
        return None

def yield_and_switch_from_compute_to_comm(schedule="default"):
    """
    全局的计算->通信切换函数
    功能：供All-Reduce等通信原语调用
    """
    ctx = get_current_ubatch_context()
    if ctx is not None:
        ctx.yield_and_switch_from_compute_to_comm(schedule)

def yield_and_switch_from_comm_to_compute(schedule="default"):
    """
    全局的通信->计算切换函数
    功能：供All-Reduce等通信原语调用
    """
    ctx = get_current_ubatch_context()
    if ctx is not None:
        ctx.yield_and_switch_from_comm_to_compute(schedule)

def make_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    forward_contexts: list[ForwardContext],
    device: Optional[torch.device] = None,
    schedule: str = "default",
) -> list[UBatchContext]:
    """
    创建微批处理上下文列表
    功能：初始化TBO执行环境
    """
    assert num_micro_batches == 2, "当前只支持2个micro-batch"
    
    global _UBATCHING_ACTIVE
    _UBATCHING_ACTIVE = True

    # 创建CPU同步事件（环形连接）
    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    
    # 创建GPU同步事件
    gpu_comm_done_events = [torch.cuda.Event() for _ in range(num_micro_batches)]
    gpu_compute_done_events = [torch.cuda.Event() for _ in range(num_micro_batches)]
    
    device = device or torch.cuda.current_device()
    assert len(forward_contexts) == 2

    ctxs = []
    for i in range(num_micro_batches):
        ctx = UBatchContext(
            id=i,
            compute_stream=compute_stream,
            comm_stream=comm_stream,
            forward_context=forward_contexts[i],
            cpu_wait_event=cpu_events[i],
            cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],  # 环形连接
            gpu_comm_done_event=gpu_comm_done_events[i],
            gpu_compute_done_event=gpu_compute_done_events[i],
            schedule=schedule
        )
        ctxs.append(ctx)

    return ctxs


# 2.2 智能批次切分算法
# 这是TBO系统中负责将大批次拆分为微批次的核心算法。
# vllm/v1/worker/gpu_model_runner.py (部分实现)

from typing import Optional, Tuple
import torch
import os
from vllm.v1.attention.backends.utils import UbatchSlice

# 类型别名定义
UBatchSlices = list[Tuple[slice, slice]]  # [(request_slice, token_slice), ...]

class GPUModelRunner:
    def _ubatch_split(
        self, 
        max_num_scheduled_tokens: int,
        scheduler_output: "SchedulerOutput"
    ) -> Tuple[Optional[UBatchSlices], int, Optional[torch.Tensor]]:
        """
        智能批次切分算法
        功能：将宏观批次切分为2个微批次，实现负载均衡
        
        参数：
        - max_num_scheduled_tokens: 最大调度token数
        - scheduler_output: 调度器输出
        
        返回：
        - ubatch_slices: 微批次切片信息
        - num_pad_tokens: 填充token数量
        - num_tokens_after_padding: 填充后总token数
        """
        
        # 1. 检查microbatching是否启用
        if not self.parallel_config.enable_microbatching:
            return (None, 0, None)

        # 2. 获取批次基本信息
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_reqs = self.input_batch.num_reqs
        
        # 3. 判断执行模式
        # Decode模式：每个请求只有1个token
        decode_only = (max_num_scheduled_tokens == 1)
        
        # Prefill模式：只有新请求，没有缓存的decode
        prefill_only = (
            hasattr(scheduler_output, "scheduled_cached_reqs") and
            getattr(scheduler_output.scheduled_cached_reqs, "req_ids", []) == [] and
            getattr(scheduler_output, "scheduled_new_reqs", []) != []
        )
        
        # 4. 检查是否允许prefill ubatch（实验性功能）
        allow_prefill_ubatch = (
            os.getenv("VLLM_EXPERIMENTAL_PREFILL_DBO", "0") == "1"
        )
        
        # 5. 核心启用条件判断
        should_attempt_ubatching = (
            self.parallel_config.enable_microbatching and
            total_num_scheduled_tokens >= self.parallel_config.microbatching_token_threshold and
            (decode_only or (allow_prefill_ubatch and prefill_only))
        )

        # 6. 数据并行协调：所有worker必须一致决策
        should_ubatch = self.should_ubatch(should_attempt_ubatching)
        if not should_ubatch:
            return (None, 0, None)

        # 7. 执行双维度切分算法
        
        # 7.1 Token维度切分（取中点）
        b0_tokens_end = max(1, total_num_scheduled_tokens // 2)
        if b0_tokens_end >= total_num_scheduled_tokens:
            # 无法有效切分token
            return (None, 0, None)

        # 7.2 特殊情况：单请求prefill无法切分
        if prefill_only and allow_prefill_ubatch and num_reqs < 2:
            # 避免单请求的元数据不一致问题
            return (None, 0, None)
        
        # 7.3 请求维度切分
        b0_reqs_end = max(1, num_reqs // 2)
        if b0_reqs_end >= num_reqs:
            # 无法有效切分请求
            return (None, 0, None)
        
        # 7.4 生成切片信息
        ubatch_slices = [
            # Micro-batch 1: 前半部分请求和token
            (slice(0, b0_reqs_end), slice(0, b0_tokens_end)),
            # Micro-batch 2: 后半部分请求和token  
            (slice(b0_reqs_end, num_reqs), slice(b0_tokens_end, total_num_scheduled_tokens)),
        ]

        # 8. 处理数据并行填充
        num_pad_tokens = 0
        num_tokens_after_padding = None
        ubatch_abort = False
        
        # 8.1 计算所需填充
        num_pad_tokens, num_tokens_after_padding = self.get_dp_padding_ubatch(ubatch_slices)
        
        if num_pad_tokens > 0:
            # 8.2 检查填充是否会导致空的微批次
            if num_pad_tokens < scheduler_output.total_num_scheduled_tokens:
                self.pad_out_ubatch_first_stage(ubatch_slices, num_pad_tokens)
            else:
                ubatch_abort = True

        # 8.3 最终协调检查
        should_ubatch = self.should_ubatch(not ubatch_abort)
        if not should_ubatch:
            return (None, 0, None)
            
        return (ubatch_slices, num_pad_tokens, num_tokens_after_padding)

    def get_dp_padding_ubatch(
        self, 
        ubatch_slices: UBatchSlices
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """
        计算数据并行所需的填充
        功能：确保所有DP worker处理相同数量的token
        """
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        
        if dp_size == 1:
            return 0, None

        # 获取两个微批次的token数
        first_ubatch_slice = ubatch_slices[0]
        second_ubatch_slice = ubatch_slices[1]

        first_ubatch_num_tokens = (
            first_ubatch_slice[1].stop - first_ubatch_slice[1].start
        )
        second_ubatch_num_tokens = (
            second_ubatch_slice[1].stop - second_ubatch_slice[1].start
        )
        
        # 由于只支持decode，两个微批次最多相差1个token
        assert abs(first_ubatch_num_tokens - second_ubatch_num_tokens) <= 1

        num_tokens_unpadded = first_ubatch_num_tokens + second_ubatch_num_tokens
        num_tokens_padded = round_up(num_tokens_unpadded, 2)  # 向上取整到2的倍数

        num_tokens_per_ubatch = num_tokens_padded // 2

        # 计算每个微批次的DP填充
        num_pad_tokens, num_tokens_after_padding = self.get_dp_padding(
            num_tokens_per_ubatch
        )

        # 转换为总填充数量
        num_pad_tokens = ((num_pad_tokens + num_tokens_per_ubatch) * 2) - num_tokens_unpadded
        return num_pad_tokens, num_tokens_after_padding

    def pad_out_ubatch_first_stage(
        self, 
        ubatch_slices: UBatchSlices,
        num_pad_tokens: int
    ):
        """
        第一阶段填充：调整切分点
        功能：重新分配token，为后续填充做准备
        """
        original_num_tokens = ubatch_slices[1][1].stop
        assert num_pad_tokens < original_num_tokens
        
        # 计算填充后每个微批次的token数
        total_num_tokens_per_ubatch = (original_num_tokens + num_pad_tokens) // 2
        
        # 重新调整切分点
        padded_first_ubatch_slice = slice(0, total_num_tokens_per_ubatch)
        padded_second_ubatch_slice = slice(total_num_tokens_per_ubatch, original_num_tokens)

        # 更新切片信息（请求和token切片都更新）
        ubatch_slices[0] = (padded_first_ubatch_slice, padded_first_ubatch_slice)
        ubatch_slices[1] = (padded_second_ubatch_slice, padded_second_ubatch_slice)

    def pad_out_ubatch_second_stage(
        self, 
        ubatch_slices: UBatchSlices,
        num_total_tokens: int
    ):
        """
        第二阶段填充：实际添加填充token
        功能：将第二个微批次扩展到总token数
        """
        # 扩展第二个微批次到包含所有填充token
        padded_second_ubatch_slice = slice(ubatch_slices[1][1].start, num_total_tokens)
        ubatch_slices[1] = (padded_second_ubatch_slice, padded_second_ubatch_slice)

# 3. 微批处理执行逻辑
# 3.1 微批处理执行器
# vllm/v1/worker/gpu_model_runner.py (执行部分)

import threading
from contextlib import contextmanager
from typing import List
from dataclasses import dataclass

@dataclass
class UbatchMetadata:
    """
    微批次元数据
    功能：封装单个微批次的执行所需信息
    """
    context: UBatchContext              # 执行上下文
    input_ids: Optional[torch.Tensor]   # 输入token IDs
    positions: Optional[torch.Tensor]   # token位置信息
    inputs_embeds: Optional[torch.Tensor]  # 输入嵌入（多模态）
    intermediate_tensors: Optional[dict]    # 中间张量

class GPUModelRunner:
    def _run_model(self,
                   attn_metadata,
                   num_scheduled_tokens: int,
                   scheduler_output: "SchedulerOutput",
                   ubatch_slices: Optional[UBatchSlices] = None,
                   num_tokens_across_dp: Optional[torch.Tensor] = None,
                   skip_cuda_graphs: bool = False):
        """
        模型执行入口
        功能：根据是否有ubatch_slices决定执行路径
        """
        
        # 检查是否启用微批处理
        if ubatch_slices is not None:
            assert len(ubatch_slices) == 2, "当前只支持2个micro-batch"

            # 获取当前计算流
            compute_stream = torch.cuda.current_stream()
            
            # 创建微批处理元数据
            ubatch_metadata = self._make_ubatch_metadata(
                ubatch_slices=ubatch_slices,
                attn_metadata=attn_metadata,
                compute_stream=compute_stream,
                num_tokens_across_dp=num_tokens_across_dp,
                skip_cuda_graphs=skip_cuda_graphs,
                scheduler_output=scheduler_output
            )
            
            # 执行微批处理
            return self._run_ubatches(ubatch_metadata, self.model)
        else:
            # 传统的单批次执行路径
            input_ids, positions, inputs_embeds, intermediate_tensors = \
                self._get_model_inputs(slice(0, num_scheduled_tokens), scheduler_output)
            
            with set_forward_context(attn_metadata,
                                   vllm_config=self.vllm_config,
                                   num_tokens=num_scheduled_tokens or 1,
                                   num_tokens_across_dp=num_tokens_across_dp,
                                   skip_cuda_graphs=skip_cuda_graphs):
                return self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )

    def _make_ubatch_metadata(
        self,
        ubatch_slices: UBatchSlices,
        attn_metadata,
        compute_stream: torch.cuda.Stream,
        num_tokens_across_dp: Optional[torch.Tensor],
        skip_cuda_graphs: bool,
        scheduler_output: "SchedulerOutput"
    ) -> List[UbatchMetadata]:
        """
        创建微批处理元数据
        功能：为每个微批次准备执行所需的所有信息
        """
        
        # 1. 创建forward contexts（每个微批次独立的推理上下文）
        forward_contexts = []
        for i, (_, token_slice) in enumerate(ubatch_slices):
            num_tokens = (token_slice.stop - token_slice.start)
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i] if attn_metadata is not None else None,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    skip_cuda_graphs=skip_cuda_graphs
                )
            )

        # 2. 创建UBatch上下文
        ubatch_ctxs = make_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,       # 通信流
            compute_stream=compute_stream,       # 计算流
            forward_contexts=forward_contexts,
            device=self.device
        )

        # 3. 为每个微批次准备输入数据
        ubatch_metadata: List[UbatchMetadata] = []
        for i, (_, token_slice) in enumerate(ubatch_slices):
            # 根据token_slice获取该微批次的模型输入
            input_ids, positions, inputs_embeds, intermediate_tensors = \
                self._get_model_inputs(token_slice, scheduler_output)
            
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    intermediate_tensors=intermediate_tensors
                )
            )

        return ubatch_metadata

    def _run_ubatches(
        self, 
        ubatch_metadata: List[UbatchMetadata], 
        model
    ) -> torch.Tensor:
        """
        并行执行微批处理
        功能：TBO的核心执行逻辑，实现两个微批次的并行执行
        """
        
        @torch.inference_mode()
        def _ubatch_thread(results: List, model, ubatch_metadata: UbatchMetadata):
            """
            单个微批次的执行线程
            功能：在独立线程中执行一个微批次
            """
            # 进入微批次上下文（这里会触发线程调度）
            with ubatch_metadata.context:
                # 执行模型前向推理
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )
            
            # 保存结果（包含微批次ID用于排序）
            results.append((ubatch_metadata.context.id, model_output))

        results: List[Tuple[int, torch.Tensor]] = []

        # 创建并启动微批次线程
        # 注意：使用override_forward_context(None)确保线程独立管理上下文
        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=_ubatch_thread,
                    args=(results, model, metadata)
                )
                ubatch_threads.append(thread)
                thread.start()

            # 启动第一个微批次（发送CPU调度信号）
            ubatch_metadata[0].context.cpu_wait_event.set()
            
            # 等待所有线程完成
            for thread in ubatch_threads:
                thread.join()

        # 按微批次ID排序并合并结果
        sorted_results = [value for position, value in sorted(results)]
        result = torch.cat(sorted_results, dim=0)
        return result

# 4. 通信重叠机制
# 4.1 All-Reduce包装层
# vllm/distributed/communication_op.py

import torch
from typing import Callable, Tuple
from vllm.distributed import get_tensor_model_parallel_group

def _get_yield_funcs() -> Tuple[Callable, Callable]:
    """
    懒加载微批处理yield函数
    功能：避免循环导入，按需导入ubatching模块
    
    返回：
    - to_comm: 切换到通信流的函数
    - to_compute: 切换到计算流的函数
    """
    try:
        from vllm.v1.worker.ubatching import (
            yield_and_switch_from_compute_to_comm as _to_comm,
            yield_and_switch_from_comm_to_compute as _to_compute,
        )
        return _to_comm, _to_compute
    except Exception:
        # 降级处理：返回无操作函数
        def _noop_to_comm(schedule: str = "default"):
            return None

        def _noop_to_compute(schedule: str = "default"):
            return None

        return _noop_to_comm, _noop_to_compute


def tensor_model_parallel_all_reduce(
    input_: torch.Tensor,
    *,
    schedule: str = "default"
) -> torch.Tensor:
    """
    带重叠优化的All-Reduce操作
    功能：在微批处理启用时，插入yield点实现计算通信重叠
    
    参数：
    - input_: 需要All-Reduce的张量
    - schedule: 通信调度标识符，支持不同类型的通信
    
    返回：
    - All-Reduce后的张量
    """
    
    # 快速路径：如果未启用ubatching，直接执行原始All-Reduce
    try:
        from vllm.v1.worker.ubatching import is_ubatching_globally_enabled
        if not is_ubatching_globally_enabled():
            return get_tensor_model_parallel_group().all_reduce(input_)
    except Exception:
        # 任何导入或查询失败：降级到传统All-Reduce
        return get_tensor_model_parallel_group().all_reduce(input_)

    # TBO模式：带重叠的All-Reduce执行
    to_comm, to_compute = _get_yield_funcs()
    
    # 1. 切换到通信流，等待前置计算完成
    to_comm(schedule=schedule)
    
    # 2. 在通信流上执行All-Reduce
    result = get_tensor_model_parallel_group().all_reduce(input_)
    
    # 3. 切换回计算流，标记通信完成
    to_compute(schedule=schedule)
    
    return result

# 5. 配置与集成
# 5.1 系统配置
# vllm/config.py (配置定义)

@dataclass
class ParallelConfig:
    """并行配置类"""
    
    enable_microbatching: bool = False
    """启用微批处理优化"""
    
    microbatching_token_threshold: int = 4
    """微批处理启用阈值。只有当批次token数大于等于该值时才启用微批处理"""
    
    def __post_init__(self):
        """配置后处理"""
        # 检查microbatching与其他功能的兼容性
        if self.enable_microbatching:
            # 微批处理需要eager模式执行
            assert self.enforce_eager, "微批处理需要enforce_eager=True"
            
            # 与CUDA Graph冲突检查
            if hasattr(self, 'compilation_config'):
                if self.compilation_config.level >= CompilationLevel.PIECEWISE:
                    logger.warning_once(
                        "微批处理与分段编译不兼容，禁用分段编译"
                    )
                    self.compilation_config.level = CompilationLevel.NO_COMPILATION