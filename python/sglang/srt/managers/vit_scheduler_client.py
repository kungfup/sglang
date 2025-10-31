"""
VIT Scheduler Client - 主 Scheduler 侧的客户端

功能:
1. 通过 ZMQ 向 VIT Scheduler 发送请求
2. 通过共享内存传递 tensor
3. 管理待处理请求的状态
"""

import os
import time
import zmq
import pickle
import logging
import threading
from collections import deque
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict
from multiprocessing import shared_memory

import torch

from sglang.srt.managers.vit_shm_utils import (
    cleanup_embedding_shm,
    read_embedding_from_shm,
)

logger = logging.getLogger(__name__)


@dataclass
class VITRequest:
    """VIT 计算请求"""
    request_id: str
    pixel_values_shm_name: str
    pixel_values_shape: Tuple[int, ...]
    pixel_values_dtype: str
    image_grid_thw_shm_name: str
    image_grid_thw_shape: Tuple[int, ...]
    image_grid_thw_dtype: str
    hash_val: Optional[int] = None  # 🔑 在 submit_async 时计算，避免重复读 SHM


@dataclass
class VITResponse:
    """VIT 计算响应（使用 CUDA IPC）

    Attributes:
        request_id: 请求 ID
        embedding_ipc_handle: CUDA IPC handle 和 offset
            - handle: List[int] - 序列化的 cudaIpcMemHandle_t (64 字节)
            - offset: int - tensor 在 storage 中的偏移量（字节）
        embedding_shape: Embedding tensor 的 shape
        embedding_dtype: Embedding tensor 的 dtype (如 "float32")
        embedding_device: Embedding tensor 的 device (如 "cuda:0")
        image_hash: 图片的 hash 值（用于 cache 管理，保留兼容性）
        cache_id: 缓存 ID（用于 free，新架构使用）
        compute_time: 计算耗时（秒）
        from_cache: 是否从 cache 中获取
        error: 是否发生错误
        error_message: 错误信息
        vit_compute_start_time: VIT 计算开始时间（用于并行性分析）
        vit_compute_end_time: VIT 计算结束时间（用于并行性分析）
    """
    request_id: str
    # 🔧 CUDA IPC: 使用 IPC handle 代替 CPU 共享内存
    embedding_ipc_handle: Tuple[List[int], int]  # (handle, offset)
    embedding_shape: Tuple[int, ...]
    embedding_dtype: str
    embedding_device: str  # 新增: 设备信息
    image_hash: int  # 保留兼容性: 用于旧架构 cache 管理
    cache_id: Optional[int] = None  # 🔧 新增: 缓存 ID (用于新架构 free)
    # 保留原有字段
    compute_time: float = 0.0
    from_cache: bool = False
    # 🔧 新增: 错误处理
    error: bool = False
    error_message: str = ""
    # 🔧 新增: 时间戳（用于并行性分析和缓存生命周期跟踪）
    vit_compute_start_time: float = 0.0
    vit_compute_end_time: float = 0.0
    # 🔑 方案 1: 引用计数 + 完成通知
    input_shm_names: Optional[List[str]] = None  # Worker 返回输入 SHM 名称，用于 PP0 release


@dataclass
class VITResult:
    """VIT 计算结果（包含 embedding 和 metadata）

    Attributes:
        embedding: 计算得到的 embedding tensor
        image_hash: VIT Scheduler 计算的 hash 值（用于旧架构 cache 释放，保留兼容性）
        cache_id: 缓存 ID（用于新架构 cache 释放）
        compute_time: 计算耗时（毫秒）
        from_cache: 是否从 cache 中获取
        error: 是否发生错误
        error_message: 错误信息
    """
    embedding: torch.Tensor
    image_hash: int  # 保留兼容性: 旧架构使用
    cache_id: Optional[int] = None  # 🔧 新增: 新架构使用
    compute_time: float = 0.0
    from_cache: bool = False
    error: bool = False  # 🔧 新增: 错误标志
    error_message: str = ""  # 🔧 新增: 错误信息


class VITSchedulerClient:
    """
    VIT Scheduler 异步客户端

    在主 Scheduler 进程中运行，负责:
    1. 异步提交 ViT 计算请求到 VIT Scheduler
    2. 异步查询计算结果
    3. 管理共享内存
    """

    def __init__(
        self,
        zmq_host: str = "localhost",
        zmq_port: int = 5555,
        timeout_ms: int = 5000,
        enable: bool = True,
    ):
        """
        Args:
            zmq_host: VIT Scheduler 的主机地址
            zmq_port: VIT Scheduler 的端口
            timeout_ms: ZMQ 请求超时时间（毫秒）
            enable: 是否启用（False 则回退到同步计算）
        """
        self.zmq_host = zmq_host
        self.zmq_port = zmq_port
        self.timeout_ms = timeout_ms
        self.max_inflight = int(os.environ.get("SGLANG_VIT_MAX_INFLIGHT", "4"))
        self.enable = enable

        if not self.enable:
            logger.info("[VIT Client] Disabled, will use synchronous ViT computation")
            return

        # 初始化 ZMQ 客户端（使用 PAIR 模式，一对一最可靠）
        self.context = zmq.Context()

        # 使用 127.0.0.1 而不是 localhost，避免 DNS 问题
        host = "127.0.0.1" if zmq_host == "localhost" else zmq_host

        # PAIR socket - 双向通信，一对一
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.RCVTIMEO, 0)  # 非阻塞接收
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{host}:{zmq_port}")

        # ZMQ Poller for efficient polling
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        # 等待连接建立
        import time
        time.sleep(0.3)

        logger.info(f"[VIT Client] ✅ Connected to VIT Scheduler at {host}:{zmq_port} (PAIR mode)")
        logger.info(f"[VIT Client] ✅ Socket (bidirectional) <-> {host}:{zmq_port}")

        # 测试连接：发送一个测试消息并等待响应
        test_msg = {"test": "connection_test"}
        try:
            self.socket.send(pickle.dumps(test_msg), zmq.NOBLOCK)
            logger.info(f"[VIT Client] ✅ Test message sent")

            # 等待测试响应
            time.sleep(0.1)
            socks = dict(self.poller.poll(timeout=1000))  # 等待 1 秒
            if self.socket in socks:
                test_response = self.socket.recv(zmq.NOBLOCK)
                logger.info(f"[VIT Client] ✅ Test response received: {pickle.loads(test_response)}")
            else:
                logger.warning(f"[VIT Client] ⚠️  No test response received")
        except Exception as e:
            logger.error(f"[VIT Client] ❌ Test failed: {e}")

        # ✅ 环境变量控制 SHM/IPC 模式
        self.use_shm = os.environ.get("SGLANG_VIT_USE_SHM", "true").lower() == "true"
        if self.use_shm:
            logger.info("[VIT Client] ✅ Using SHM mode for embedding transfer")
        else:
            logger.info("[VIT Client] ⚠️ Using CUDA IPC mode (legacy, SGLANG_VIT_USE_SHM=false)")

        # 管理 pixel/grid 的 multiprocessing.shared_memory
        self.shm_objects: Dict[str, shared_memory.SharedMemory] = {}

        # 异步请求管理
        self.pending_requests: Dict[str, Dict] = {}  # request_id -> request_info

        # 统计信息
        self.submitted_count = 0
        self.completed_count = 0
        self.timeout_count = 0

        # ✅✅✅ 核心修复: 增加超时时间（从 10 秒增加到 30 秒）
        # 根据日志分析，处理 8 个请求需要 11.6 秒，10 秒超时太短
        # 增加到 30 秒留有余量
        self._send_timeout_s = 1.0   # 未成功发出后重发阈值
        self._drop_timeout_s = float(
            os.environ.get("SGLANG_VIT_CLIENT_DROP_TIMEOUT_S", "300.0")
        )  # 连续未发出/未应答的丢弃阈值
        self._max_retries = 5        # 最多重发次数
        # 🔧 Phase 2.C: 响应超时阈值（避免 TP 计算时被误判为超时，默认 120s，可通过环境变量调整）
        self._response_timeout_s = float(
            os.environ.get("SGLANG_VIT_CLIENT_RESPONSE_TIMEOUT_S", "120.0")
        )
        if self._drop_timeout_s <= self._response_timeout_s:
            self._drop_timeout_s = self._response_timeout_s * 1.5
        self._last_free_ts: Dict[int, float] = {}  # free 信号去重节流

        # 后台通信线程（唯一持有 socket）
        # 🔧 使用 deque 代替 Queue，支持从队列开头插入（用于重新入队）
        self._tx_queue: deque = deque()
        self._tx_queue_lock = threading.Lock()
        self._results: Dict[str, torch.Tensor] = {}
        self._results_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._disable_drop = os.environ.get(
            "SGLANG_VIT_CLIENT_DISABLE_DROP", "0"
        ) == "1"
        self._worker_thread = threading.Thread(
            target=self._worker_main, name="VITClientWorker", daemon=True
        )
        self._worker_thread.start()
        logger.info(f"[VIT Client] ✅ Worker thread started. client_id={id(self)}, socket_id={id(self.socket)}")

    def _create_shm_for_tensor(self, tensor: torch.Tensor, shm_name: str) -> shared_memory.SharedMemory:
        """为 tensor 创建共享内存"""
        tensor_cpu = tensor.cpu()
        nbytes = tensor_cpu.element_size() * tensor_cpu.nelement()

        # 创建共享内存
        shm = shared_memory.SharedMemory(create=True, size=nbytes, name=shm_name)

        # 写入数据
        shm_tensor = torch.frombuffer(shm.buf, dtype=tensor_cpu.dtype).reshape(tensor_cpu.shape)
        shm_tensor.copy_(tensor_cpu)

        # 保存引用（防止被 GC）
        self.shm_objects[shm_name] = shm

        return shm

    def _load_tensor_from_shm(self, shm_name: str, shape: Tuple, dtype: str) -> torch.Tensor:
        """从共享内存加载 tensor"""
        # 计算 tensor 大小
        dtype_obj = getattr(torch, dtype)
        element_size = torch.tensor([], dtype=dtype_obj).element_size()
        size = 1
        for dim in shape:
            size *= dim
        nbytes = size * element_size

        # 打开共享内存
        shm = shared_memory.SharedMemory(name=shm_name)

        # 创建 tensor
        tensor = torch.frombuffer(shm.buf[:nbytes], dtype=dtype_obj).reshape(shape).clone()

        # 关闭并删除共享内存
        shm.close()
        shm.unlink()

        return tensor

    def _cleanup_shm(self, shm_name: str, reason: str = "unspecified"):
        """释放共享内存引用（不直接 unlink，改由引用计数管理）"""
        if not shm_name:
            return

        from sglang.srt.managers.vit_shm_manager import get_global_shm_manager
        shm_manager = get_global_shm_manager()

        # 关闭本地 handle（如果有），避免 fd 泄漏
        if shm_name in self.shm_objects:
            shm = self.shm_objects.pop(shm_name)
            try:
                shm.close()
            except FileNotFoundError:
                logger.debug(
                    "[VIT Client] 🧹 SHM %s already closed when cleaning (reason=%s)",
                    shm_name,
                    reason,
                )
            except Exception as exc:
                logger.warning(
                    "[VIT Client] ⚠️ Failed to close SHM %s (reason=%s): %s",
                    shm_name,
                    reason,
                    exc,
                )

        # 引用计数减 1；只有归零时才真正 unlink
        try:
            if shm_manager.release(shm_name):
                if shm_manager.cleanup_if_zero(shm_name):
                    logger.debug(
                        "[VIT Client] 🧹 cleaned SHM %s via ref-count (reason=%s)",
                        shm_name,
                        reason,
                    )
        except Exception as exc:
            logger.warning(
                "[VIT Client] ⚠️ Failed to release SHM %s (reason=%s): %s",
                shm_name,
                reason,
                exc,
            )

    def _read_embedding_from_shm(
        self,
        request_id: str,
        shape: Tuple[int, ...],
        dtype_str: str,
    ) -> Optional[torch.Tensor]:
        embedding = read_embedding_from_shm(request_id, max_retries=5, retry_delay_ms=10.0)
        if embedding is None:
            logger.error("[VIT Client] ❌ Failed to read SHM embedding: %s", request_id)
        return embedding

    def submit_async(
        self,
        request_id: str,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> bool:
        """
        异步提交 ViT 计算请求（立即返回，不等待）

        Args:
            request_id: 请求 ID
            pixel_values: 图片像素值 tensor
            image_grid_thw: 图片网格信息

        Returns:
            是否成功提交
        """
        if not self.enable:
            return False

        try:
            # 创建共享内存
            pixel_values_shm_name = f"vit_pv_{request_id}"
            image_grid_thw_shm_name = f"vit_grid_{request_id}"

            self._create_shm_for_tensor(pixel_values, pixel_values_shm_name)
            self._create_shm_for_tensor(image_grid_thw, image_grid_thw_shm_name)

            # 🔑 修复 4: 计算完整 hash (pixel_values + image_grid_thw)
            import hashlib
            hash_val = None
            try:
                # 🔑 修复 4: 包含 pixel_values 和 image_grid_thw，对齐 LightLLM
                pixel_values_cpu = pixel_values.cpu() if pixel_values.is_cuda else pixel_values
                image_grid_thw_cpu = image_grid_thw.cpu() if image_grid_thw.is_cuda else image_grid_thw

                # 拼接两个 tensor 的字节流
                buffer = pixel_values_cpu.numpy().tobytes() + image_grid_thw_cpu.numpy().tobytes()

                # 计算 MD5 hash（取前 16 位，对齐 _compute_request_hash）
                hash_val = int(hashlib.md5(buffer).hexdigest()[:16], 16)
                logger.debug(f"[VIT Client] Computed hash_val for {request_id}: {hash_val:x}")
            except Exception as e:
                logger.warning(f"[VIT Client] Failed to compute hash_val for {request_id}: {e}")

            # 🔑 方案 1: Acquire SHM 引用计数
            from sglang.srt.managers.vit_shm_manager import get_global_shm_manager
            shm_manager = get_global_shm_manager()
            shm_manager.acquire(pixel_values_shm_name)
            shm_manager.acquire(image_grid_thw_shm_name)
            logger.debug(
                f"[VIT Client] Acquired SHM: {pixel_values_shm_name}, {image_grid_thw_shm_name}"
            )

            # 构造请求
            request = VITRequest(
                request_id=request_id,
                pixel_values_shm_name=pixel_values_shm_name,
                pixel_values_shape=tuple(pixel_values.shape),
                pixel_values_dtype=str(pixel_values.dtype).replace('torch.', ''),
                image_grid_thw_shm_name=image_grid_thw_shm_name,
                image_grid_thw_shape=tuple(image_grid_thw.shape),
                image_grid_thw_dtype=str(image_grid_thw.dtype).replace('torch.', ''),
                hash_val=hash_val,  # 🔑 携带 hash_val
            )

            # 将请求投递到后台线程发送（避免跨线程触碰 ZMQ socket）
            try:
                with self._tx_queue_lock:
                    self._tx_queue.append((request_id, asdict(request)))
            except Exception as e:
                logger.error(f"[VIT Client] ❌ Failed to enqueue request {request_id}: {e}")
                raise
            logger.info(f"[VIT Client]  submit_async on thread={threading.current_thread().name}, client_id={id(self)}, socket_id={id(self.socket)}, pending_dict_id={id(self.pending_requests)}")

            self.submitted_count += 1

            # 记录待处理请求（加入发送确认与重试元数据）
            self.pending_requests[request_id] = {
                'pixel_values_shm_name': pixel_values_shm_name,
                'image_grid_thw_shm_name': image_grid_thw_shm_name,
                'submit_time': time.time(),
                'sent': False,                 # 发送确认标志：仅在 worker 成功 send 后置 True
                'retry': 0,                    # 重发次数
                'payload': asdict(request),    # 原始请求体，便于重发
            }

            logger.info(f"[VIT Client] ✅ Async submitted request: {request_id}, pending={len(self.pending_requests)}")
            logger.debug(f"[VIT Client] 📋 pending_requests size={len(self.pending_requests)}")
            logger.debug(f"[VIT Client] 📋 pending_requests id: {id(self.pending_requests)}")

            # 立即验证 pending_requests（降为 debug）
            if request_id in self.pending_requests:
                logger.debug(f"[VIT Client] ✅ Verified: {request_id} is in pending_requests")
            else:
                logger.error(f"[VIT Client] ❌ ERROR: {request_id} NOT in pending_requests after submit!")

            return True

        except Exception as e:
            logger.error(f"[VIT Client] Error submitting request {request_id}: {e}", exc_info=True)

            # 🔑 修复 2: 统一使用 shm_manager.release() 清理
            try:
                from sglang.srt.managers.vit_shm_manager import get_global_shm_manager
                shm_manager = get_global_shm_manager()

                pixel_values_shm_name = f"vit_pv_{request_id}"
                image_grid_thw_shm_name = f"vit_grid_{request_id}"

                # Release 引用计数
                if shm_manager.release(pixel_values_shm_name):
                    shm_manager.cleanup_if_zero(pixel_values_shm_name)
                if shm_manager.release(image_grid_thw_shm_name):
                    shm_manager.cleanup_if_zero(image_grid_thw_shm_name)

                logger.debug(f"[VIT Client] Released SHM after submit failure: {request_id}")
            except Exception as cleanup_error:
                logger.warning(f"[VIT Client] Failed to release SHM after submit failure: {cleanup_error}")

            return False

    def try_get_result(self, request_id: str) -> Optional[torch.Tensor]:
        """
        尝试获取 ViT 计算结果（非阻塞）

        Args:
            request_id: 请求 ID

        Returns:
            embedding tensor，如果还未完成则返回 None
        """
        if not self.enable:
            return None

        if request_id not in self.pending_requests:
            logger.warning(f"[VIT Client] Request {request_id} not found in pending requests")
            return None

        try:
            # 非阻塞接收响应
            # DEALER socket 接收 [empty_frame, message]
            frames = self.socket.recv_multipart(zmq.NOBLOCK)
            if len(frames) < 2:
                return None
            message = frames[1]
            response_dict = pickle.loads(message)
            response = VITResponse(**response_dict)

            # 检查是否是我们要的请求
            if response.request_id != request_id:
                # 不是我们要的，可能是其他请求的响应
                # 暂时忽略（简化处理）
                logger.warning(
                    f"[VIT Client] Received response for {response.request_id}, "
                    f"but expected {request_id}"
                )
                return None

            self.completed_count += 1

            logger.debug(
                f"[VIT Client] Received response: {response.request_id}, "
                f"from_cache={response.from_cache}, "
                f"compute_time={response.compute_time*1000:.1f}ms"
            )

            # 从共享内存加载 embedding
            embedding = self._load_tensor_from_shm(
                response.embedding_shm_name,
                response.embedding_shape,
                response.embedding_dtype,
            )

            # 清理共享内存
            req_info = self.pending_requests.pop(request_id)
            self._cleanup_shm(req_info['pixel_values_shm_name'], reason="recv_success")
            self._cleanup_shm(req_info['image_grid_thw_shm_name'], reason="recv_success")

            return embedding

        except zmq.Again:
            # 还没有响应，返回 None
            return None

        except Exception as e:
            logger.error(f"[VIT Client] Error getting result for {request_id}: {e}", exc_info=True)
            return None

    def poll_results(self) -> Dict[str, VITResult]:
        """
        非阻塞地取走后台线程收到的全部结果；不再触碰 ZMQ socket。

        Returns:
            字典：request_id -> VITResult（包含 embedding 和 metadata）
        """
        if not self.enable:
            return {}

        # 统计与诊断日志
        if not hasattr(self, '_poll_count'):
            self._poll_count = 0
            logger.info("[VIT Client] 🔍 poll_results() is being called for the first time")
        self._poll_count += 1
        if self._poll_count <= 10:
            logger.info(
                f"[VIT Client] 🔍 Poll #{self._poll_count}: pending={len(self.pending_requests)}, "
                f"ids={list(self.pending_requests.keys())}, dict_id={id(self.pending_requests)}"
            )

        # 仅访问内存缓冲区，不跨线程访问 socket
        out: Dict[str, VITResult] = {}
        lock = getattr(self, "_results_lock", None)
        if lock is not None:
            with lock:
                if hasattr(self, "_results") and self._results:
                    out = self._results.copy()
                    self._results.clear()
        else:
            if hasattr(self, "_results") and self._results:
                out = self._results.copy()
                self._results.clear()

        if out:
            try:
                _keys = list(out.keys())
                _sample = _keys[:3]
                _more = "" if len(_keys) <= 3 else f" (+{len(_keys)-3} more)"
                logger.info(
                    f"[VIT Client] ✅ Drain {len(out)} results from worker, pending={len(self.pending_requests)}; "
                    f"ids={_sample}{_more}"
                )
            except Exception:
                logger.info(f"[VIT Client] ✅ Drain {len(out)} results from worker, pending={len(self.pending_requests)}")
        else:
            # 低频诊断，避免刷屏
            if self._poll_count <= 20 or (len(self.pending_requests) > 0 and self._poll_count % 10000 == 0):
                res_len = len(self._results) if hasattr(self, "_results") and self._results is not None else 0
                logger.info(
                    f"[VIT Client] 🔍 poll_results: no result, _results_len={res_len}, pending={len(self.pending_requests)}, poll_count={self._poll_count}"
                )

            # Watchdog：处理未发送成功或长期未应答的 pending（幽灵 pending 防护）
            now = time.time()
            to_drop = []
            for rid, info in list(self.pending_requests.items()):
                sent = info.get('sent', False)
                submit_time = info.get('submit_time', now)
                retry = info.get('retry', 0)

                # 情况 1：未发送成功（sent=False）
                if not sent and (now - submit_time) > self._send_timeout_s:
                    payload = info.get('payload')
                    if payload is not None:
                        with self._tx_queue_lock:
                            # 优先重发，确保尽快出队
                            self._tx_queue.appendleft((rid, payload))
                        info['retry'] = retry + 1
                        info['submit_time'] = now
                        logger.warning(f"[VIT Client] 🔁 Requeue unsent request {rid} (retry={info['retry']})")

                # 情况 2：已发送但长期无响应（sent=True but no response）
                # 这种情况可能是消息在 ZMQ 缓冲区中丢失，或 VIT Scheduler 未收到
                # ✅ 优化: 从 3秒 增加到 15秒，避免在 VIT 计算过程中重试（VIT 计算约 9.4秒）
                if sent and (now - submit_time) > self._response_timeout_s:
                    payload = info.get('payload')
                    if payload is not None:
                        with self._tx_queue_lock:
                            # 重发请求
                            self._tx_queue.appendleft((rid, payload))
                        info['retry'] = retry + 1
                        info['submit_time'] = now
                        info['sent'] = False  # 重置 sent 标志，等待重新发送
                        logger.warning(f"[VIT Client] 🔁 Requeue sent-but-no-response request {rid} (retry={info['retry']}, elapsed={now - submit_time:.1f}s)")

                # 丢弃条件：长时间未响应 或 重试过多
                if (now - submit_time) > self._drop_timeout_s or info.get('retry', 0) >= self._max_retries:
                    to_drop.append(rid)

            for rid in to_drop:
                if self._disable_drop:
                    info = self.pending_requests.get(rid)
                    logger.warning(
                        "[VIT Client] ⏱️ Drop suppressed for request %s (retry=%s, elapsed=%.1fs)",
                        rid,
                        info.get('retry', 0) if info else "n/a",
                        now - info.get('submit_time', now) if info else -1.0,
                    )
                    if info:
                        info['submit_time'] = now
                    continue
                info = self.pending_requests.pop(rid, None)
                if info:
                    self._cleanup_shm(info.get('pixel_values_shm_name', ''), reason="drop_timeout")
                    self._cleanup_shm(info.get('image_grid_thw_shm_name', ''), reason="drop_timeout")
                    self.timeout_count += 1
                    logger.error(f"[VIT Client] ⏱️ Drop pending request {rid} after retries={info.get('retry',0)}, elapsed={now - info.get('submit_time', now):.1f}s; cleaned SHM and marked timeout")
        return out

    def _worker_main(self):
        """Background worker that exclusively owns the ZMQ socket (send + recv)."""
        tname = threading.current_thread().name
        logger.info(f"[VIT Client] 🧵 worker start: thread={tname}, client_id={id(self)}, socket_id={id(self.socket)}")
        while not self._stop_event.is_set():
            # Send phase: drain a small burst to reduce syscalls
            try:
                # 尽量一次性发送多条，避免频繁系统调用
                for _ in range(32):
                    msg_type = None
                    msg_data = None

                    # 从 deque 中取出消息
                    with self._tx_queue_lock:
                        if len(self._tx_queue) == 0:
                            break
                        msg_type, msg_data = self._tx_queue.popleft()

                    try:
                        # msg_type 可以是 request_id（计算请求）或 "free"（释放信号）
                        self.socket.send(pickle.dumps(msg_data), zmq.NOBLOCK)
                        if msg_type == "free":
                            logger.debug(f"[VIT Client] 📤 sent free signal: hash={msg_data.get('image_hash')}")
                        else:
                            # 发送成功，标记 pending.sent = True
                            info = self.pending_requests.get(msg_type)
                            if info is not None:
                                info['sent'] = True
                            logger.info(f"[VIT Client] 📤 sent VIT request: {msg_type}")
                    except zmq.Again:
                        # Socket not ready. Requeue to the TAIL and retry later.
                        logger.warning(f"[VIT Client] ⚠️ Socket not ready (zmq.Again), requeuing message to TAIL: {msg_type}")
                        time.sleep(0.0005)
                        with self._tx_queue_lock:
                            self._tx_queue.append((msg_type, msg_data))  # 放到队列尾部，避免饿死
                        break
                    except Exception as e:
                        logger.error(f"[VIT Client] ❌ send error for {msg_type}: {e}")
            except Exception as e:
                logger.error(f"[VIT Client] ❌ send-phase error: {e}")

            # Recv phase: pull all available responses
            while True:
                try:
                    msg = self.socket.recv(zmq.NOBLOCK)
                    resp_dict = pickle.loads(msg)
                    resp = VITResponse(**resp_dict)

                    # ✅ 根据模式选择不同的处理方式
                    try:
                        # 🔧 检查错误标志
                        if resp.error or resp.embedding_shape == (0, 0):
                            logger.error(
                                "[VIT Client] ❌ Received error response for %s: %s (cache_id=%s, hash=%s)",
                                resp.request_id,
                                resp.error_message or "Unknown error",
                                resp.cache_id,
                                resp.image_hash,
                            )
                            emb = torch.empty(0)
                        elif self.use_shm and resp.embedding_device == "cpu":
                            # ✅ SHM 模式: 从 POSIX 共享内存读取
                            logger.info(f"[VIT Client] 📖 Reading embedding from SHM: {resp.request_id}")

                            # 读取 embedding
                            emb = self._read_embedding_from_shm(
                                resp.request_id,
                                resp.embedding_shape,
                                resp.embedding_dtype
                            )

                            if emb is None:
                                logger.error(f"[VIT Client] ❌ Failed to read SHM: {resp.request_id}")
                                continue

                            cleanup_embedding_shm(resp.request_id)

                            logger.info(
                                f"[VIT Client] ✅ Read embedding from SHM: "
                                f"shape={emb.shape}, device={emb.device}"
                            )
                        else:
                            # ❌ CUDA IPC 模式
                            from functools import reduce
                            from sglang.semi_pd.utils import convert_ipc_handle_to_tensor

                            ipc_handle, offset = resp.embedding_ipc_handle
                            shape = resp.embedding_shape
                            dtype_str = resp.embedding_dtype
                            device_str = resp.embedding_device

                            # 转换 dtype
                            dtype = getattr(torch, dtype_str)

                            # 🔧 关键: 使用 PP0 的设备 (GPU 0)
                            device = torch.device("cuda:0")  # PP0 在 GPU 0

                            # 计算 size
                            size = reduce(lambda x, y: x * y, shape)

                            logger.info(
                                f"[VIT Client] 🔗 Opening CUDA IPC handle: "
                                f"request_id={resp.request_id}, shape={shape}, dtype={dtype_str}, "
                                f"device={device}, offset={offset}"
                            )

                            # 从 IPC handle 重建 tensor
                            emb = convert_ipc_handle_to_tensor(
                                (ipc_handle, offset),
                                size,
                                dtype,
                                device
                            )

                            # Reshape
                            emb = emb.view(shape)

                            logger.info(
                                f"[VIT Client] ✅ Opened CUDA IPC handle: "
                                f"shape={emb.shape}, device={emb.device}"
                            )

                    except Exception as e:
                        logger.error(
                            f"[VIT Client] ❌ Failed to load embedding for "
                            f"{resp.request_id}: {e}",
                            exc_info=True
                        )
                        # 跳过这个响应，继续处理下一个
                        continue

                    # 验证 embedding
                    if emb.numel() == 0:
                        logger.warning(
                            f"[VIT Client] ⚠️ Received empty embedding for {resp.request_id}"
                        )
                    elif torch.isnan(emb).any():
                        logger.error(
                            f"[VIT Client] ❌ Loaded embedding contains NaN! "
                            f"shape={emb.shape}"
                        )
                    else:
                        logger.info(
                            f"[VIT Client] ✅ Loaded embedding is valid: "
                            f"shape={emb.shape}, device={emb.device}, "
                            f"min={emb.min().item():.4f}, max={emb.max().item():.4f}"
                        )

                    # ✅ CPU Staging: 确保 embedding 留在 CPU，对齐 LightLLM
                    # LightLLM 在 ViT worker 中将嵌入 to(cpu)，只有在准备拼文本时才拷贝回 GPU
                    if emb.device.type != 'cpu':
                        logger.warning(
                            f"[VIT Client] ⚠️ Embedding on GPU, moving to CPU for staging: "
                            f"{resp.request_id}, device={emb.device}"
                        )
                        emb = emb.cpu()
                        logger.info(f"[VIT Client] ✅ Moved embedding to CPU: {resp.request_id}")

                    # 🔑 方案 1: Release SHM 引用计数，引用计数为 0 时清理
                    # 优先使用 Worker 返回的 input_shm_names（更可靠）
                    from sglang.srt.managers.vit_shm_manager import get_global_shm_manager
                    shm_manager = get_global_shm_manager()

                    input_shm_names = getattr(resp, 'input_shm_names', None)
                    if input_shm_names:
                        # Worker 返回了 input_shm_names，使用它们
                        for shm_name in input_shm_names:
                            if shm_name and shm_manager.release(shm_name):
                                # 引用计数为 0，清理 SHM
                                shm_manager.cleanup_if_zero(shm_name)
                                logger.debug(
                                    f"[VIT Client] Released and cleaned SHM: {shm_name} (from worker)"
                                )
                    else:
                        # 兼容旧版本：从 pending_requests 获取 SHM 名称
                        info = self.pending_requests.get(resp.request_id)
                        if info:
                            pixel_values_shm_name = info.get('pixel_values_shm_name', '')
                            image_grid_thw_shm_name = info.get('image_grid_thw_shm_name', '')

                            if pixel_values_shm_name and shm_manager.release(pixel_values_shm_name):
                                shm_manager.cleanup_if_zero(pixel_values_shm_name)
                                logger.debug(
                                    f"[VIT Client] Released and cleaned SHM: {pixel_values_shm_name} (from pending)"
                                )

                            if image_grid_thw_shm_name and shm_manager.release(image_grid_thw_shm_name):
                                shm_manager.cleanup_if_zero(image_grid_thw_shm_name)
                                logger.debug(
                                    f"[VIT Client] Released and cleaned SHM: {image_grid_thw_shm_name} (from pending)"
                                )

                    # 从 pending_requests 移除
                    self.pending_requests.pop(resp.request_id, None)

                    # 🔧 关键修改: 存储 VITResult（包含 embedding、cache_id 和错误信息）
                    vit_result = VITResult(
                        embedding=emb,
                        image_hash=resp.image_hash,  # 保留兼容性: 旧架构使用
                        cache_id=resp.cache_id,  # 🔧 新增: 新架构使用
                        compute_time=resp.compute_time,
                        from_cache=resp.from_cache,
                        error=resp.error,  # 🔧 新增: 错误标志
                        error_message=resp.error_message,  # 🔧 新增: 错误信息
                    )

                    with self._results_lock:
                        self._results[resp.request_id] = vit_result
                        logger.info(
                            f"[VIT Client] 🔧 Worker put result into _results: {resp.request_id}, "
                            f"_results_len={len(self._results)}, cache_id={resp.cache_id}, image_hash={resp.image_hash}, error={resp.error}"
                        )

                    self.completed_count += 1
                    logger.info(
                        f"[VIT Client] 📥 received {resp.request_id}, "
                        f"compute_time={resp.compute_time*1000:.1f}ms, "
                        f"from_cache={resp.from_cache}, hash={resp.image_hash}"
                    )
                except zmq.Again:
                    break
                except Exception as e:
                    logger.error(f"[VIT Client] ❌ recv error: {e}")
                    break

            # Avoid busy spinning
            with self._tx_queue_lock:
                queue_is_empty = len(self._tx_queue) == 0
            if queue_is_empty:
                time.sleep(0.0005)

        logger.info("[VIT Client] 🧵 worker exit")

    def notify_embedding_consumed(self, image_hash: int = 0, cache_id: Optional[int] = None):
        """通知 VIT Scheduler 释放 embedding（事件驱动释放）

        Args:
            image_hash: 图片 hash (旧架构使用，保留兼容性)
            cache_id: 缓存 ID (新架构使用)
        """
        if not self.enable:
            logger.warning(f"[VIT Client] ⚠️ notify_embedding_consumed called but VIT client is disabled")
            return

        # 🔧 新架构优先使用 cache_id
        if cache_id is not None:
            message = {
                "type": "free_embedding",
                "cache_id": cache_id,
                "image_hash": 0,  # 保留兼容性
            }
            dedup_key = cache_id
        else:
            # 旧架构使用 image_hash
            message = {
                "type": "free_embedding",
                "image_hash": image_hash,
                "cache_id": None,
            }
            dedup_key = image_hash

        try:
            # free 信号去重：0.5s 内重复信号直接跳过，避免刷屏与队列膨胀
            now = time.time()
            last = self._last_free_ts.get(dedup_key, 0.0)
            if now - last < 0.5:
                logger.debug(f"[VIT Client] ⏭️ Skip duplicate free signal within 0.5s for key={dedup_key}")
                return
            self._last_free_ts[dedup_key] = now

            # 通过后台线程发送（非阻塞）
            with self._tx_queue_lock:
                self._tx_queue.append(("free", message))
            logger.debug(f"[VIT Client] 📝 Queued free signal for cache_id={cache_id}, hash={image_hash}")
        except Exception as e:
            logger.warning(f"[VIT Client] Failed to send free signal for cache_id={cache_id}, hash={image_hash}: {e}")

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'submitted': self.submitted_count,
            'completed': self.completed_count,
            'timeout': self.timeout_count,
            'pending': len(self.pending_requests),
            'success_rate': self.completed_count / max(self.submitted_count, 1),
        }

    def cleanup(self):
        """清理资源"""
        if not self.enable:
            return

        logger.info("[VIT Client] Cleaning up...")

        # 停止后台线程
        try:
            if hasattr(self, "_stop_event"):
                self._stop_event.set()
            if hasattr(self, "_worker_thread") and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=1.0)
                logger.info("[VIT Client] Worker thread stopped")
        except Exception as e:
            logger.warning(f"[VIT Client] Error stopping worker: {e}")

        # 清理所有共享内存
        for shm_name in list(self.shm_objects.keys()):
            try:
                self._cleanup_shm(shm_name, reason="client_cleanup")
            except Exception as e:
                logger.warning(f"[VIT Client] Error cleaning up {shm_name}: {e}")

        # 关闭 ZMQ
        try:
            self.socket.close()
        except Exception:
            pass
        try:
            self.context.term()
        except Exception:
            pass

        logger.info("[VIT Client] Cleanup complete")

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass


class VITSchedulerClientAsync:
    """
    VIT Scheduler 异步客户端（未来扩展）

    支持:
    1. 异步提交请求（非阻塞）
    2. 异步查询结果（非阻塞）
    3. 批量提交
    """

    def __init__(self, *args, **kwargs):
        # TODO: 实现异步版本
        raise NotImplementedError("Async client not implemented yet")

    def submit_task(self, request_id: str, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> bool:
        """异步提交任务"""
        raise NotImplementedError()

    def try_get_result(self, request_id: str) -> Optional[torch.Tensor]:
        """非阻塞查询结果"""
        raise NotImplementedError()
