import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Sequence, Union

import torch

_ENABLE_PROFILE = bool(int(os.environ.get("SGLANG_OPERATIONS_ENABLE_PROFILE", "0")))
_TBO_DEBUG = bool(int(os.environ.get("SGLANG_TBO_DEBUG", "0")))
# 是否启用独立通信流（默认关闭，避免在小批次时引入额外开销）
_ENABLE_SEPARATE_COMM_STREAM = bool(int(os.environ.get("SGLANG_TBO_SEPARATE_COMM_STREAM", "0")))

if _ENABLE_PROFILE:
    import nvtx


def _tbo_log(message: str):
    if _TBO_DEBUG:
        print(f"[TBO] {message}", flush=True)

# CUDA availability
try:
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False


def execute_operations(inputs, operations):
    stages = _convert_operations_to_stages(operations)
    executor = _StageExecutor("primary", stages, inputs=inputs)
    for _ in range(executor.num_stages):
        executor.next()
    assert executor.done
    return executor.output


def execute_overlapped_operations(
    inputs_arr: Sequence,
    operations_arr: Sequence,
    delta_stages: Sequence[int],
) -> Sequence:
    # Make it explicit for clarity; if we need multi-batch overlap, this can be generalized
    inputs_a, inputs_b = inputs_arr
    operations_a, operations_b = operations_arr
    delta_stage_a, delta_stage_b = delta_stages
    assert delta_stage_a == 0
    delta_stage = delta_stage_b

    stages_a = _convert_operations_to_stages(operations_a)
    stages_b = _convert_operations_to_stages(operations_b)

    if _TBO_DEBUG:
        _tbo_log(
            f"overlap plan: delta={delta_stage}, stages_a={len(stages_a)}, stages_b={len(stages_b)}"
        )
        # 打印每个阶段的 op 名称列表，便于识别通信阶段
        def _stage_op_names(stages):
            return ["+".join(op.debug_name for op in stage) for stage in stages]

        _tbo_log(f"stages_a_ops={_stage_op_names(stages_a)}")
        _tbo_log(f"stages_b_ops={_stage_op_names(stages_b)}")

    executor_a = _StageExecutor("a", stages_a, inputs=inputs_a)
    executor_b = _StageExecutor("b", stages_b, inputs=inputs_b)

    for _ in range(delta_stage):
        executor_a.next()

    for _ in range(executor_a.num_stages - delta_stage):
        executor_a.next()
        executor_b.next()

    for _ in range(delta_stage):
        executor_b.next()

    assert executor_a.done and executor_b.done
    return [executor_a.output, executor_b.output]


class YieldOperation:
    pass


@dataclass
class ExecutionOperation:
    debug_name: str
    fn: Callable
    is_comm: bool = False


Operation = Union[YieldOperation, ExecutionOperation, Callable]
Stage = List[ExecutionOperation]


class _StageExecutor:
    def __init__(self, debug_name: str, stages: List[Stage], inputs):
        self._debug_name = debug_name
        self._stages = stages
        self._index = 0
        self._stage_state = _StateDict()
        self._stage_output = inputs
        # 记录上一阶段是否为通信阶段，用于在回到计算阶段时做依赖同步
        self._last_stage_was_comm = False
        # 每个微批自己的通信流与事件，避免跨微批相互等待（仅在开关开启且 CUDA 可用时创建）
        self._comm_stream = (
            torch.cuda.Stream() if (_HAS_CUDA and _ENABLE_SEPARATE_COMM_STREAM) else None
        )
        self._last_compute_event = (
            torch.cuda.Event(enable_timing=False)
            if (_HAS_CUDA and _ENABLE_SEPARATE_COMM_STREAM)
            else None
        )
        self._last_comm_event = (
            torch.cuda.Event(enable_timing=False)
            if (_HAS_CUDA and _ENABLE_SEPARATE_COMM_STREAM)
            else None
        )

    def _record_event_on_current(self, event: torch.cuda.Event):
        if _HAS_CUDA and event is not None:
            event.record(torch.cuda.current_stream())

    def _wait_event_on_stream(self, stream: torch.cuda.Stream, event: torch.cuda.Event):
        if _HAS_CUDA and stream is not None and event is not None:
            stream.wait_event(event)

    def next(self):
        assert not self.done

        stage = self._stages[self._index]
        op_names = "+".join(op.debug_name for op in stage)
        is_comm_stage = any(getattr(op, "is_comm", False) for op in stage)
        t0 = time.time() if _TBO_DEBUG else None

        # 在阶段边界处理同一微批的跨流依赖（仅等待本微批产生的事件）
        use_comm_stream = self._comm_stream is not None and is_comm_stage
        if use_comm_stream:
            # 通信阶段：在通信流上执行，并等待本微批最近一次计算事件
            stream_ctx = torch.cuda.stream(self._comm_stream)
            if self._last_compute_event is not None:
                self._comm_stream.wait_event(self._last_compute_event)
        else:
            # 计算阶段：如上一阶段是通信，等待本微批的通信事件（仅在启用独立通信流时）
            if (
                self._last_stage_was_comm
                and self._comm_stream is not None
                and self._last_comm_event is not None
            ):
                torch.cuda.current_stream().wait_event(self._last_comm_event)
            stream_ctx = contextmanager(lambda: (yield))()

        with _annotate_region(debug_name=f"{self._debug_name}{self._index}"):
            with stream_ctx:
                for op in stage:
                    with _annotate_region(debug_name=op.debug_name):
                        self._stage_output = op.fn(
                            state=self._stage_state,
                            **(
                                self._stage_output if self._stage_output is not None else {}
                            ),
                        )

        # 执行完成后，在对应流上记录事件，供下一阶段依赖
        if _HAS_CUDA and _ENABLE_SEPARATE_COMM_STREAM:
            if use_comm_stream and self._last_comm_event is not None:
                # 事件要在通信流上记录
                with torch.cuda.stream(self._comm_stream):
                    self._last_comm_event.record(self._comm_stream)
            elif (not is_comm_stage) and self._last_compute_event is not None:
                # 事件要在当前（计算）流上记录
                self._last_compute_event.record(torch.cuda.current_stream())

        if _TBO_DEBUG:
            dt_ms = (time.time() - t0) * 1000.0
            _tbo_log(
                f"stage {self._debug_name}{self._index}: ops=[{op_names}] time_ms={dt_ms:.3f}"
            )

        self._last_stage_was_comm = is_comm_stage
        self._index += 1

    @property
    def output(self):
        assert self.done
        return self._stage_output

    @property
    def done(self):
        return self._index >= self.num_stages

    @property
    def num_stages(self):
        return len(self._stages)


@contextmanager
def _annotate_region(debug_name):
    if _ENABLE_PROFILE:
        with torch.autograd.profiler.record_function(debug_name):
            with nvtx.annotate(debug_name):
                yield
    else:
        yield


class _StateDict:
    def __init__(self):
        self._data = {}

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__(key, value)
            return
        assert (
            key not in self._data
        ), f"`{key}` already exist, are you sure you want to override it?"
        self._data[key] = value

    def __getattr__(self, item):
        return self._data[item]

    def __delattr__(self, item):
        del self._data[item]

    def pop(self, item):
        return self._data.pop(item)

    def update(self, values: Dict[str, Any]):
        for k, v in values.items():
            setattr(self, k, v)

    def get(self, item):
        return self._data.get(item)

    def clear(self, expect_keys: Sequence[str]):
        if set(self._data.keys()) != set(expect_keys):
            raise Exception(
                f"Unexpected keys when clearning. This may indicate you do not release memory early enough but leave it to here. {list(self._data.keys())=} {expect_keys=}"
            )

        self._data.clear()


def _convert_operations_to_stages(operations: List[Operation]) -> List[Stage]:
    operations = _decorate_operations(operations)
    operation_chunks = list(
        _chunk_by_separator(operations, lambda op: isinstance(op, YieldOperation))
    )
    assert all(len(chunk) > 0 for chunk in operation_chunks)
    return operation_chunks


def _chunk_by_separator(
    items: List[Any], is_separator: Callable[[Any], bool]
) -> Generator[List[Any], None, None]:
    pending_items = []
    for item in items:
        if is_separator(item):
            yield pending_items
            pending_items = []
        else:
            pending_items.append(item)
    if len(pending_items) > 0:
        yield pending_items


def _decorate_operations(operations: List[Operation], debug_name_prefix: str = ""):
    return [_decorate_operation(op, debug_name_prefix) for op in operations]


def _is_comm_op_name(name: str) -> bool:
    n = name.lower()
    return (
        ("allreduce" in n)
        or ("all_reduce" in n)
        or ("reduce_scatter" in n)
    )


def _decorate_operation(operation: Operation, debug_name_prefix: str):
    if isinstance(operation, YieldOperation):
        return operation
    name = getattr(operation, "__name__", "unknown").replace("op_", "")
    debug_name = debug_name_prefix + name
    return ExecutionOperation(
        debug_name=debug_name,
        fn=operation,
        is_comm=_is_comm_op_name(name),
    )
