"""
LightLLM-aligned ViT scheduler.

This implementation removes the legacy POSIX IPC helpers and GPU cache logic,
and instead mirrors the high-level structure of LightLLM's visual pipeline:

- A dedicated cache server that persists embeddings in multiprocessing.shared_memory.
- Distinct IO and compute loops connected via thread-safe queues.
- Responses delivered exclusively through SHM buffers created by vit_shm_utils.
- Event-driven cache release triggered by the client through free signals.

The design keeps the public surface of VITScheduler compatible with the rest of
SGLang while modernising the internals to the stable LightLLM pattern.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import pickle
import threading
import time
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Dict, List, Optional, Sequence, Tuple

import multiprocessing as mp
import psutil
import rpyc
import torch
import zmq
from torch import distributed as dist

from sglang.srt.managers.vit_shm_utils import (
    cleanup_embedding_shm,
    read_embedding_from_shm,
    read_embedding_from_shm_raw,
    write_embedding_to_shm,
    write_embedding_to_shm_raw,
)
from rpyc.utils.classic import obtain

logger = logging.getLogger(__name__)

# 🔑 Phase 4: asyncio/uvloop 支持
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info("[VIT Scheduler] Using uvloop for high-performance async I/O")
except ImportError:
    logger.warning("[VIT Scheduler] uvloop not available, using default asyncio event loop")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_tensor_from_shared_memory(
    shm_name: str,
    shape: Sequence[int],
    dtype_str: str,
) -> torch.Tensor:
    """Load tensor from multiprocessing.shared_memory and clone to CPU tensor."""
    import multiprocessing.shared_memory as shm

    dtype = getattr(torch, dtype_str)
    shared_memory = shm.SharedMemory(name=shm_name)
    try:
        tensor = torch.frombuffer(shared_memory.buf, dtype=dtype).reshape(shape).clone()
    finally:
        shared_memory.close()
    return tensor


def _compute_request_hash(pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> int:
    """Compute deterministic hash for a request (matches LightLLM strategy)."""
    buffer = pixel_values.cpu().numpy().tobytes() + image_grid_thw.cpu().numpy().tobytes()
    return int(hashlib.md5(buffer).hexdigest()[:16], 16)


def _worker_process_bootstrap(
    model_config,
    worker_id: int,
    tp_rank: int,
    tp_size: int,
    rpc_port: int,
    cache_rpc_port: int,
    ready_conn,
    env_overrides: Optional[Dict[str, str]] = None,
) -> None:
    """Spawn helper used for in-process worker restarts."""
    try:
        if env_overrides:
            os.environ.update(env_overrides)
        os.environ.setdefault("SGLANG_VIT_TP_RANK", str(tp_rank))
        os.environ.setdefault("SGLANG_VIT_TP_SIZE", str(tp_size))
        os.environ.setdefault("SGLANG_VIT_WORKER_ID", str(worker_id))
        # 让每个 Worker 拥有独立的 NCCL 端口，保持与 engine 中的逻辑一致
        base_tp_port = int(os.environ.get("SGLANG_VIT_TP_PORT", "29500"))
        os.environ["SGLANG_VIT_TP_PORT"] = str(base_tp_port + worker_id)
        if ready_conn is not None:
            ready_conn.send("ready")

        from sglang.srt.managers.vit_worker_rpc import start_vit_worker_process

        start_vit_worker_process(
            worker_id=worker_id,
            model_config=model_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            rpc_port=rpc_port,
            cache_rpc_host="localhost",
            cache_rpc_port=cache_rpc_port,
        )
    except Exception as exc:
        if ready_conn is not None:
            try:
                ready_conn.send(f"error:{exc}")
            except Exception:
                pass
        raise
    finally:
        if ready_conn is not None:
            ready_conn.close()


# ---------------------------------------------------------------------------
# Cache server (SHM backed)
# ---------------------------------------------------------------------------


@dataclass
class CacheRecord:
    shm_key: str
    size_bytes: int
    last_access: float
    ref_count: int = 0


class VITCacheServer:
    """Shared-memory cache for ViT embeddings (LightLLM-style)."""

    def __init__(self, max_cache_bytes: int):
        self._max_cache_bytes = max_cache_bytes
        self._total_bytes = 0
        self._records: "OrderedDict[int, CacheRecord]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _cache_request_id(hash_val: int) -> str:
        return f"cache_{hash_val:x}"

    def contains(self, hash_val: int) -> bool:
        if self._max_cache_bytes <= 0:
            return False
        with self._lock:
            return hash_val in self._records

    def retain(self, hash_val: int) -> bool:
        if self._max_cache_bytes <= 0:
            return False
        with self._lock:
            record = self._records.get(hash_val)
            if record is None:
                return False
            record.ref_count += 1
            record.last_access = time.time()
            self._records.move_to_end(hash_val, last=True)
            return True

    def release(self, hash_val: int) -> None:
        if self._max_cache_bytes <= 0:
            return
        with self._lock:
            record = self._records.get(hash_val)
            if record is None:
                return
            if record.ref_count > 0:
                record.ref_count -= 1
            record.last_access = time.time()
            if record.ref_count == 0:
                self._evict_if_needed(0)

    def get(self, hash_val: int) -> Optional[torch.Tensor]:
        with self._lock:
            record = self._records.get(hash_val)
            if record is None:
                return None
            record.last_access = time.time()
            self._records.move_to_end(hash_val, last=True)
            shm_key = record.shm_key
        return read_embedding_from_shm(shm_key)

    def put(self, hash_val: int, embedding: torch.Tensor) -> None:
        if self._max_cache_bytes <= 0:
            return
        if embedding.is_cuda:
            embedding = embedding.cpu()
        size_bytes = embedding.nelement() * embedding.element_size()
        shm_key = self._cache_request_id(hash_val)

        with self._lock:
            old = self._records.pop(hash_val, None)
            if old is not None:
                cleanup_embedding_shm(old.shm_key)
                self._total_bytes -= old.size_bytes

            self._evict_if_needed(size_bytes)

            success = write_embedding_to_shm(shm_key, embedding)
            if not success:
                logger.error("[VIT Cache] Failed to write embedding to SHM: hash=%s", hash_val)
                return

            record = CacheRecord(
                shm_key=shm_key,
                size_bytes=size_bytes,
                last_access=time.time(),
                ref_count=0,
            )
            self._records[hash_val] = record
            self._total_bytes += size_bytes
            logger.info(
                "[VIT Cache] added hash=%s size=%.2fMB total=%.2fMB",
                hash_val,
                size_bytes / (1024**2),
                self._total_bytes / (1024**2),
            )

    def _evict_if_needed(self, extra_bytes: int) -> None:
        if self._max_cache_bytes <= 0:
            return
        while (
            self._records
            and self._total_bytes + extra_bytes > self._max_cache_bytes
        ):
            hash_val, record = next(
                ((k, r) for k, r in self._records.items() if r.ref_count == 0),
                (None, None),
            )
            if hash_val is None:
                logger.warning(
                    "[VIT Cache] Cannot evict more entries (all in use). "
                    "current=%.2fMB limit=%.2fMB",
                    self._total_bytes / (1024**2),
                    self._max_cache_bytes / (1024**2),
                )
                break

            self._records.pop(hash_val, None)
            cleanup_embedding_shm(record.shm_key)
            self._total_bytes -= record.size_bytes
            logger.info(
            "[VIT Cache] evicted hash=%s size=%.2fMB remaining=%.2fMB",
            hash_val,
            record.size_bytes / (1024**2),
            self._total_bytes / (1024**2),
        )

    def get_stats(self) -> Dict[str, float]:
        with self._lock:
            return {
                "num_entries": len(self._records),
                "total_bytes": self._total_bytes,
                "max_bytes": self._max_cache_bytes,
            }

    def cleanup(self) -> None:
        with self._lock:
            records = list(self._records.values())
            self._records.clear()
            self._total_bytes = 0
        for record in records:
            cleanup_embedding_shm(record.shm_key)
        logger.info("[VIT Cache] cleaned %d entries", len(records))


# ---------------------------------------------------------------------------
# Request / response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VITRequest:
    request_id: str
    pixel_values_shm_name: str
    pixel_values_shape: Tuple[int, ...]
    pixel_values_dtype: str
    image_grid_thw_shm_name: str
    image_grid_thw_shape: Tuple[int, ...]
    image_grid_thw_dtype: str
    hash_val: Optional[int] = None  # 🔑 Phase 3 补充: 在 submit_async 时计算，避免重复读 SHM
    cache_hit_retry_count: int = 0  # 🔧 Phase 3 补充: 缓存命中回退重试次数


@dataclass
class VITResponse:
    request_id: str
    embedding_ipc_handle: Tuple[List[int], int]
    embedding_shape: Tuple[int, ...]
    embedding_dtype: str
    embedding_device: str
    image_hash: int  # 旧架构使用
    compute_time: float
    from_cache: bool
    cache_id: Optional[int] = None  # 🔧 新架构使用
    error: bool = False  # 🔧 错误标志
    error_message: str = ""  # 🔧 错误信息
    vit_compute_start_time: float = 0.0
    vit_compute_end_time: float = 0.0
    # 🔑 方案 1: 引用计数 + 完成通知
    input_shm_names: Optional[List[str]] = None  # Worker 返回输入 SHM 名称，用于 PP0 release


# ---------------------------------------------------------------------------
# VITModelRunner (mostly unchanged from legacy implementation)
# ---------------------------------------------------------------------------


class VITModelRunner:
    """Wrapper that hosts the ViT model (single or tensor parallel)."""

    def __init__(self, model_config, device: str = "cuda:0", tp_size: int = 1):
        self.model_config = model_config
        self.device = device
        self.vit_model = None
        self.tp_size = tp_size

        from sglang.srt.configs.load_config import LoadConfig

        self.load_config = LoadConfig()

    def load_model(self):
        model_type = self.model_config.hf_config.model_type

        logger.info(f"[VIT Runner] Loading ViT model type: {model_type}")
        if self.tp_size > 1:
            logger.info(f"[VIT Runner] TP enabled: size={self.tp_size}")

        # 🔧 关键修复: 复用主流程的量化配置逻辑
        # 参考: sglang/python/sglang/srt/model_loader/loader.py L110-140
        from sglang.srt.model_loader.weight_utils import get_quant_config

        quant_config = None
        if self.model_config.quantization is not None:
            try:
                # 获取量化配置 (支持 fp8, bf16, int8 等)
                quant_config = get_quant_config(
                    self.model_config,
                    self.load_config,
                    packed_modules_mapping={},  # VIT 不需要 packed modules
                )
                logger.info(
                    f"[VIT Runner] Quantization enabled: {self.model_config.quantization}, "
                    f"config={quant_config.get_name() if quant_config else 'None'}"
                )
            except Exception as e:
                logger.warning(
                    f"[VIT Runner] Failed to get quant_config for {self.model_config.quantization}: {e}. "
                    "Falling back to no quantization."
                )
                quant_config = None

        # 🔧 关键修复: 使用 set_default_torch_dtype 确保 dtype 与主模型一致
        # 参考: sglang/python/sglang/srt/model_loader/loader.py L612-614
        from sglang.srt.model_loader.utils import set_default_torch_dtype

        logger.info(
            f"[VIT Runner] Creating VIT model with dtype={self.model_config.dtype}, "
            f"quant_config={quant_config.get_name() if quant_config else 'None'}"
        )

        with set_default_torch_dtype(self.model_config.dtype):
            if model_type == "qwen2_5_vl":
                from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer

                self.vit_model = Qwen2_5_VisionTransformer(
                    self.model_config.hf_config.vision_config,
                    norm_eps=getattr(self.model_config.hf_config, "rms_norm_eps", 1e-6),
                    quant_config=quant_config,  # ✅ 传入量化配置
                )
            elif model_type == "qwen2_vl":
                from sglang.srt.models.qwen2_vl import Qwen2VisionTransformer

                self.vit_model = Qwen2VisionTransformer(
                    self.model_config.hf_config.vision_config,
                    norm_eps=getattr(self.model_config.hf_config, "rms_norm_eps", 1e-6),
                    quant_config=quant_config,  # ✅ 传入量化配置
                )
            else:
                raise ValueError(f"Unsupported model type for VIT Scheduler: {model_type}")

        logger.info(
            "[VIT Runner] Loading ViT weights (visual.* only) "
            f"from {self.model_config.model_path}"
        )

        from inspect import signature

        from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        # 🔧 Phase 2.C: 获取 TP rank 用于权重分片
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        from sglang.srt.layers.parameter import (
            _ColumnvLLMParameter,
            RowvLLMParameter,
        )

        tp_rank = get_tensor_model_parallel_rank() if self.tp_size > 1 else 0
        logger.info(
            f"[VIT Runner] Loading weights with tp_rank={tp_rank}, tp_size={self.tp_size}, "
            f"dtype={self.model_config.dtype}"
        )

        # 🔧 关键修复: 使用 set_default_torch_dtype 包裹权重加载
        # 参考: sglang/python/sglang/srt/model_loader/loader.py L612-614
        # 确保权重加载时使用正确的 dtype (bf16/fp8)
        with set_default_torch_dtype(self.model_config.dtype):
            loader = get_model_loader(self.load_config)
            weight_iter = loader._get_weights_iterator(
                DefaultModelLoader.Source(
                    model_or_path=self.model_config.model_path,
                    revision=getattr(self.model_config, "revision", None),
                    prefix="",
                    fall_back_to_pt=True,
                )
            )

            params_dict = dict(self.vit_model.named_parameters())
            buffers_dict = dict(self.vit_model.named_buffers())

            loaded = 0
            loaded_column_parallel = 0
            loaded_row_parallel = 0

            for name, tensor in weight_iter:
                if not name.startswith("visual."):
                    continue

                vit_name = name[len("visual.") :]

                if ".attn.qkv." in vit_name:
                    vit_name = vit_name.replace(".attn.qkv.", ".attn.qkv_proj.")

                target_param = params_dict.get(vit_name)
                if target_param is None:
                    target_param = buffers_dict.get(vit_name)

                if target_param is None:
                    continue

                # 🔧 Phase 2.C: 根据参数类型选择正确的 weight_loader
                if isinstance(target_param, _ColumnvLLMParameter):
                    # ColumnParallelLinear: 按列分片
                    target_param.load_column_parallel_weight(
                        tensor,
                        tp_rank=tp_rank,
                        use_presharded_weights=False,
                    )
                    loaded_column_parallel += 1
                    logger.debug(f"[VIT Runner] Loaded column-parallel weight: {vit_name}, shape={target_param.data.shape}")
                elif isinstance(target_param, RowvLLMParameter):
                    # RowParallelLinear: 按行分片
                    target_param.load_row_parallel_weight(
                        tensor,
                        tp_rank=tp_rank,
                        use_presharded_weights=False,
                    )
                    loaded_row_parallel += 1
                    logger.debug(f"[VIT Runner] Loaded row-parallel weight: {vit_name}, shape={target_param.data.shape}")
                else:
                    # 普通参数: 不分片
                    weight_loader = getattr(target_param, "weight_loader", default_weight_loader)

                    if "qkv_proj" in vit_name and (
                        ".q_proj." in name or ".k_proj." in name or ".v_proj." in name
                    ):
                        if ".q_proj." in name:
                            loaded_shard_id = "q"
                        elif ".k_proj." in name:
                            loaded_shard_id = "k"
                        else:
                            loaded_shard_id = "v"

                        sig = signature(weight_loader)
                        if "loaded_shard_id" in sig.parameters:
                            weight_loader(target_param, tensor, loaded_shard_id=loaded_shard_id)
                        else:
                            weight_loader(target_param, tensor)
                    else:
                        weight_loader(target_param, tensor)
                    logger.debug(f"[VIT Runner] Loaded normal weight: {vit_name}")

                loaded += 1

            if loaded == 0:
                raise RuntimeError("Failed to load any ViT visual weights; check model path")

            # 🔧 关键修复: 量化后处理 (参考主流程)
            # 参考: sglang/python/sglang/srt/model_loader/loader.py L615-618
            if quant_config is not None:
                logger.info(f"[VIT Runner] Processing quantization weights for {quant_config.get_name()}")
                for _, module in self.vit_model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        quant_method.process_weights_after_loading(module)
                logger.info(f"[VIT Runner] Quantization processing completed")

            self.vit_model.to(self.device)
            self.vit_model.eval()

            logger.info(
                f"[VIT Runner] ViT model loaded successfully: "
                f"total={loaded}, column_parallel={loaded_column_parallel}, row_parallel={loaded_row_parallel}, "
                f"dtype={self.model_config.dtype}, quant={quant_config.get_name() if quant_config else 'None'}"
            )

    @torch.inference_mode()
    def compute_batch(
        self,
        pixel_values_list: Sequence[torch.Tensor],
        image_grid_thw_list: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        """批量计算 VIT embeddings

        🔧 P0 修复: 直接复用 SGLang 原生 Qwen2_5_VisionTransformer.forward()

        关键发现:
        1. SGLang 的 Qwen2_5_VisionTransformer 已经实现了完整的窗口化注意力
        2. get_window_index() 会自动处理多张图的窗口索引
        3. forward() 内部已经正确处理 window_index 和 reverse_indices
        4. 我们之前的 torch.cat() 破坏了窗口语义，导致 OOM

        修复方案 (P0 止血版):
        - 按图循环调用 self.vit_model.forward()
        - 让原生代码自己处理窗口化注意力
        - 避免手动拼接破坏窗口语义

        Args:
            pixel_values_list: List of pixel_values tensors, each shape [num_images, C, H, W]
            image_grid_thw_list: List of image_grid_thw tensors, each shape [num_grids, 3]

        Returns:
            List of embedding tensors, each shape [num_tokens, embedding_dim]
        """
        if self.vit_model is None:
            raise RuntimeError("VIT model not loaded")

        batch_size = len(pixel_values_list)
        if batch_size == 0:
            return []

        # ✅ P0 修复: 按图循环，直接复用 SGLang 原生 forward()
        # 这样可以保证窗口化注意力正确工作，避免 OOM
        logger.debug(f"[VIT Runner] Starting batch compute (per-image): batch_size={batch_size}")

        embeddings = []
        total_time = 0.0

        for i, (pixel_values, image_grid_thw) in enumerate(zip(pixel_values_list, image_grid_thw_list)):
            start_time = time.time()

            # 🔧 关键修复 1: 先迁移到 GPU (non_blocking 提高性能)
            # 避免模型内部隐式搬运，保证性能和正确性
            pixel_values = pixel_values.to(self.device, non_blocking=True)
            image_grid_thw = image_grid_thw.to(self.device, non_blocking=True)

            # 直接调用 SGLang 原生 forward()
            # 内部会自动处理:
            # 1. get_window_index(grid_thw) - 计算窗口索引
            # 2. 窗口化注意力 - 避免 O(N²) 显存爆炸
            # 3. reverse_indices - 恢复原始顺序
            embedding = self.vit_model(pixel_values, grid_thw=image_grid_thw)

            compute_time = time.time() - start_time
            total_time += compute_time

            logger.debug(
                f"[VIT Runner] Image {i+1}/{batch_size} completed: "
                f"time={compute_time*1000:.1f}ms, embedding={embedding.shape}"
            )

            # 🔧 关键修复 2: 迁移到 CPU 再返回
            # 避免 GPU tensor 写 SHM 出错，减少 GPU 显存占用
            embeddings.append(embedding.detach().cpu())

        avg_time = total_time / batch_size
        logger.info(
            f"[VIT Runner] Batch compute completed: "
            f"batch_size={batch_size}, total_time={total_time*1000:.1f}ms, "
            f"avg_time={avg_time*1000:.1f}ms per image"
        )

        return embeddings

    @torch.inference_mode()
    def compute(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        if self.vit_model is None:
            raise RuntimeError("VIT model not loaded")

        pixel_values = pixel_values.to(self.device, non_blocking=True)
        image_grid_thw = image_grid_thw.to(self.device, non_blocking=True)

        embedding = self.vit_model(pixel_values, grid_thw=image_grid_thw)
        return embedding.detach().cpu()


# ---------------------------------------------------------------------------
# VITScheduler
# ---------------------------------------------------------------------------


class VITScheduler:
    """LightLLM-aligned ViT scheduler."""

    def __init__(
        self,
        model_config,
        device: str = "cuda:0",
        zmq_port: int = 5555,
        batch_size: int = 4,
        batch_timeout_ms: float = 10.0,
        cache_size_mb: Optional[int] = None,
        cache_rpc_port: Optional[int] = None,
        worker_rpc_port_start: Optional[int] = None,  # 🔧 新增: Worker Pool RPC 端口起始
        vit_dp: Optional[int] = None,  # 🔧 新增: ViT DP 大小
        use_dynamic_batching: bool = False,  # 🔑 Phase 4: 动态批处理开关
    ):
        self.model_config = model_config
        self.device = device
        self.zmq_port = zmq_port
        self.batch_size = max(1, batch_size)

        # 🔧 P2 修复: 优化 batch 超时时间
        # 参考 LightLLM: 使用更长的超时时间以提高批处理利用率
        # 默认从 10ms 增加到 100ms，可通过环境变量调整
        default_batch_timeout_ms = float(os.environ.get("SGLANG_VIT_BATCH_TIMEOUT_MS", "100.0"))
        if batch_timeout_ms == 10.0:  # 如果使用默认值，则使用环境变量
            batch_timeout_ms = default_batch_timeout_ms
        self.batch_timeout = max(0.001, batch_timeout_ms / 1000.0)
        logger.info(f"[VIT Scheduler] Batch timeout set to {self.batch_timeout*1000:.1f}ms")

        # 🔑 Phase 4: 动态批处理配置
        self.use_dynamic_batching = use_dynamic_batching or (os.environ.get("SGLANG_VIT_DYNAMIC_BATCHING", "0") == "1")
        if self.use_dynamic_batching:
            # 动态批处理: 初始接收数量
            self.visual_recv_max_count = int(os.environ.get("SGLANG_VIT_RECV_MAX_COUNT", "64"))
            logger.info(f"[VIT Scheduler] Dynamic batching enabled, initial recv_max_count={self.visual_recv_max_count}")
        else:
            # 固定批处理: 使用 batch_size
            logger.info(f"[VIT Scheduler] Fixed batching enabled, batch_size={self.batch_size}")
        self.max_pixel_values = int(os.environ.get("SGLANG_VIT_MAX_PIXEL_VALUES", "400000"))
        self.max_batch_pixel_tokens = int(os.environ.get("SGLANG_VIT_MAX_BATCH_PIXEL_TOKENS", "800000"))

        env_cache_mb = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "2048"))
        if cache_size_mb is None:
            cache_size_mb = env_cache_mb
        self.cache_size_bytes = cache_size_mb * 1024 * 1024

        self.benchmark_mode = os.environ.get("SGLANG_VIT_BENCHMARK_MODE", "0") == "1"

        # 🔧 缓存禁用开关 (参考 LightLLM)
        self.cache_enabled = os.environ.get("SGLANG_VIT_DISABLE_CACHE", "0") != "1"
        if not self.cache_enabled:
            logger.info("[VIT Scheduler] ViT embedding cache is DISABLED")

        self.tp_rank = int(os.environ.get("SGLANG_VIT_TP_RANK", "0"))
        self.tp_size = int(os.environ.get("SGLANG_VIT_TP_SIZE", "1"))
        self.vit_tp_port = int(os.environ.get("SGLANG_VIT_TP_PORT", "29500"))

        # 🔧 新架构开关
        self.use_new_arch = os.environ.get("SGLANG_VIT_NEW_ARCH", "1") == "1"
        self.cache_rpc_port = cache_rpc_port or int(os.environ.get("SGLANG_VIT_CACHE_RPC_PORT", "18888"))

        # 🔧 Worker Pool 配置
        default_worker_pool = "1" if self.use_new_arch else "0"
        self.use_worker_pool = os.environ.get("SGLANG_VIT_USE_WORKER_POOL", default_worker_pool) == "1"
        self.vit_dp = vit_dp or int(os.environ.get("SGLANG_VIT_DP", "1"))
        self.worker_rpc_port_start = worker_rpc_port_start or int(os.environ.get("SGLANG_VIT_WORKER_RPC_PORT_START", "19000"))
        self.worker_clients = []  # Worker Pool RPC 客户端列表
        self.next_worker_id = 0  # DP 轮询计数器
        self.worker_executor = None  # ThreadPoolExecutor for concurrent RPC calls
        self.worker_rpc_timeout = float(os.environ.get("SGLANG_VIT_WORKER_RPC_TIMEOUT", "120.0"))
        self.worker_processes: Dict[int, List[mp.Process]] = {}
        self.worker_proc_ctx = mp.get_context("spawn")
        self.worker_gpu_high_water_frac = float(
            os.environ.get("SGLANG_VIT_WORKER_GPU_HIGH_WATER_FRAC", "0.995")
        )
        if not (0.0 < self.worker_gpu_high_water_frac < 1.0):
            # Interpret <=0 or >=1 as disabling high-water throttling (LightLLM-style).
            self.worker_gpu_high_water_frac = 0.0
        self.worker_gpu_cooldown_s = float(
            os.environ.get("SGLANG_VIT_WORKER_GPU_COOLDOWN_S", "1.5")
        )
        self.worker_restart_backoff = float(
            os.environ.get("SGLANG_VIT_WORKER_RESTART_BACKOFF_S", "5.0")
        )
        self.worker_gpu_overflow_limit = int(
            os.environ.get("SGLANG_VIT_WORKER_GPU_OVERFLOW_LIMIT", "3")
        )
        if self.worker_gpu_overflow_limit < 1:
            self.worker_gpu_overflow_limit = 1
        self.worker_gpu_overflow_count: Dict[int, int] = defaultdict(int)

        if self.tp_size > 1:
            self._init_distributed()

        self.context = None
        self.socket = None
        if self.tp_rank == 0:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PAIR)
            self.socket.bind(f"tcp://127.0.0.1:{zmq_port}")
            self.socket.setsockopt(zmq.RCVTIMEO, 0)
            self.socket.setsockopt(zmq.LINGER, 0)
            logger.info("[VIT Scheduler] ZMQ server listening on port %d", zmq_port)

        # 🔧 根据架构选择缓存实现
        if self.use_new_arch and self.cache_enabled:
            # 连接 CacheServer (RPyC)
            import rpyc
            try:
                self.cache_client = rpyc.connect(
                    "localhost",
                    self.cache_rpc_port,
                    config={"allow_pickle": True, "sync_request_timeout": 30}
                )
                logger.info(f"[VIT Scheduler] Connected to CacheServer at port {self.cache_rpc_port} (NEW ARCH)")
                self.cache_server = None  # 不使用旧的 cache_server
            except Exception as e:
                logger.error(f"[VIT Scheduler] Failed to connect to CacheServer: {e}")
                raise
        elif self.use_new_arch and not self.cache_enabled:
            # 新架构但缓存禁用
            self.cache_client = None
            self.cache_server = None
            logger.info("[VIT Scheduler] CacheServer NOT connected (cache disabled)")
        else:
            # 使用旧的进程内 CacheServer
            self.cache_server = VITCacheServer(self.cache_size_bytes)
            self.cache_client = None
            logger.info("[VIT Scheduler] Using legacy in-process CacheServer")

        # 🔧 连接 Worker Pool (如果启用)
        if self.use_worker_pool:
            self._init_worker_pool()
        else:
            # 使用进程内 ModelRunner
            self.model_runner = VITModelRunner(
                model_config=model_config,
                device=self.device,
                tp_size=self.tp_size,
            )

        self._request_queue: "Queue[VITRequest]" = Queue()
        self._free_queue: "deque[int]" = deque()
        self._free_lock = threading.Lock()
        self._stop_event = threading.Event()

        # 🔑 Phase 4: asyncio 队列（仅在 asyncio 模式下使用）
        self._request_queue_async: Optional[asyncio.Queue] = None
        self._free_queue_async: Optional[asyncio.Queue] = None

        self.total_requests = 0
        self.cache_hits = 0
        self.total_compute_time = 0.0

        # 🔧 Phase 2.B: 健壮性统计
        self.worker_timeout_count = 0  # Worker RPC 超时次数
        self.worker_retry_count = 0  # 任务重试次数
        self.worker_failure_count = 0  # Worker 失败次数
        self.worker_last_health_check = {}  # {worker_id: timestamp}
        self.worker_health_check_interval = 30.0  # 健康检查间隔 (秒)

        self._io_thread: Optional[threading.Thread] = None
        self._compute_thread: Optional[threading.Thread] = None
        self._free_thread: Optional[threading.Thread] = None
        self._health_check_thread: Optional[threading.Thread] = None  # 🔧 健康检查线程

        logger.info(
            "[VIT Scheduler] initialised (batch=%d timeout=%.1fms cache=%dMB)",
            self.batch_size,
            self.batch_timeout * 1000,
            cache_size_mb,
        )

    # ------------------------------------------------------------------
    # Worker Pool initialization
    # ------------------------------------------------------------------

    def _init_worker_pool(self) -> None:
        """连接 Worker Pool (RPyC)"""
        import rpyc

        logger.info(f"[VIT Scheduler] Connecting to Worker Pool: vit_dp={self.vit_dp}, port_start={self.worker_rpc_port_start}")

        # 🔧 Phase 0: Worker Watchdog 配置
        if not hasattr(self, "worker_last_health_check"):
            self.worker_last_health_check = {}
        if not hasattr(self, "worker_health_check_interval"):
            self.worker_health_check_interval = float(os.environ.get("SGLANG_VIT_WORKER_HEALTH_CHECK_INTERVAL", "30.0"))
        if not hasattr(self, "worker_timeout_threshold"):
            self.worker_timeout_threshold = float(os.environ.get("SGLANG_VIT_WORKER_TIMEOUT_THRESHOLD", "180.0"))  # 3分钟无响应视为超时
        if not hasattr(self, "worker_restart_enabled"):
            self.worker_restart_enabled = (
                os.environ.get("SGLANG_VIT_WORKER_RESTART_ENABLED", "0") == "1"
            )

        # Worker 进程引用（用于重启）
        self.worker_processes = {}
        self.worker_restart_count = {}

        for worker_id in range(self.vit_dp):
            port = self.worker_rpc_port_start + worker_id
            try:
                client = rpyc.connect(
                    "localhost",
                    port,
                    config={
                        "allow_public_attrs": True,
                        "allow_pickle": True,
                        "sync_request_timeout": 300,
                    },
                )
                self.worker_clients.append(client)
                self.worker_restart_count[worker_id] = 0
                self.worker_last_health_check[worker_id] = time.time()
                self.worker_gpu_overflow_count[worker_id] = 0
                logger.info(f"[VIT Scheduler] Connected to Worker {worker_id} on port {port}")
            except Exception as e:
                logger.error(f"[VIT Scheduler] Failed to connect to Worker {worker_id} on port {port}: {e}")
                raise

        logger.info(f"[VIT Scheduler] Worker Pool initialized with {len(self.worker_clients)} workers")

        # 🔧 初始化 ThreadPoolExecutor 用于并发 RPC 调用
        self.worker_executor = ThreadPoolExecutor(max_workers=self.vit_dp, thread_name_prefix="vit_worker_rpc")
        logger.info(f"[VIT Scheduler] ThreadPoolExecutor initialized with {self.vit_dp} workers")

        # 🔧 Phase 0: 初始化健康检查时间戳
        for worker_id in range(self.vit_dp):
            self.worker_last_health_check[worker_id] = time.time()

        logger.info(
            f"[VIT Scheduler] Worker Watchdog enabled: health_check_interval={self.worker_health_check_interval}s, "
            f"timeout_threshold={self.worker_timeout_threshold}s, restart_enabled={self.worker_restart_enabled}"
        )

    def _check_worker_health(self, worker_id: int) -> bool:
        """检查 Worker 健康状态

        🔧 Phase 0: Worker Watchdog - 心跳检测

        Args:
            worker_id: Worker ID

        Returns:
            bool: True 表示健康，False 表示不健康
        """
        if worker_id >= len(self.worker_clients):
            return False
        client = self.worker_clients[worker_id]
        if client is None:
            return False
        try:
            result = client.root.ping()

            # 转换 RPyC netref 为本地对象
            if hasattr(result, "_getvalue"):
                result = result._getvalue()
            elif hasattr(result, "value"):
                result = result.value
            else:
                result = obtain(result)

            if result.get("status") == "ok":
                self.worker_last_health_check[worker_id] = time.time()
                return True
            else:
                logger.warning(f"[VIT Scheduler] Worker {worker_id} health check failed: {result}")
                return False

        except Exception as e:
            logger.error(f"[VIT Scheduler] Worker {worker_id} health check error: {e}")
            return False

    def _restart_worker(self, worker_id: int) -> bool:
        """重启 Worker 进程

        🔧 Phase 0: Worker Watchdog - 快速恢复机制

        Args:
            worker_id: Worker ID

        Returns:
            bool: True 表示重启成功，False 表示重启失败
        """
        if not self.worker_restart_enabled:
            logger.warning(f"[VIT Scheduler] Worker {worker_id} restart disabled by config")
            return False

        max_restart_attempts = 3
        if self.worker_restart_count.get(worker_id, 0) >= max_restart_attempts:
            logger.error(
                f"[VIT Scheduler] Worker {worker_id} has been restarted {max_restart_attempts} times, "
                f"giving up to avoid restart loop"
            )
            return False

        logger.warning(f"[VIT Scheduler] Attempting to restart Worker {worker_id}...")

        try:
            # 1. 关闭旧连接
            if worker_id < len(self.worker_clients):
                old_client = self.worker_clients[worker_id]
                if old_client is not None:
                    try:
                        old_client.close()
                    except Exception as e:
                        logger.warning(
                            f"[VIT Scheduler] Failed to close old client for Worker {worker_id}: {e}"
                        )
                self.worker_clients[worker_id] = None

            # 2. 终止残留进程
            self._terminate_worker_processes(worker_id)

            # 3. 启动新的 Worker 进程组
            if not self._spawn_worker_group(worker_id):
                logger.error(f"[VIT Scheduler] Worker {worker_id} spawn failed")
                return False

            time.sleep(self.worker_restart_backoff)

            # 4. 重建连接（最多重试 5 次）
            port = self.worker_rpc_port_start + worker_id
            for attempt in range(5):
                try:
                    new_client = rpyc.connect(
                        "localhost",
                        port,
                        config={
                            "allow_public_attrs": True,
                            "allow_pickle": True,
                            "sync_request_timeout": 300,
                        },
                    )
                    result = new_client.root.ping()
                    if hasattr(result, "_getvalue"):
                        result = result._getvalue()
                    elif hasattr(result, "value"):
                        result = result.value
                    else:
                        result = obtain(result)

                    if result.get("status") != "ok":
                        raise RuntimeError(
                            f"Worker {worker_id} ping failed after restart: {result}"
                        )

                    self.worker_clients[worker_id] = new_client
                    self.worker_last_health_check[worker_id] = time.time()
                    self.worker_restart_count[worker_id] = (
                        self.worker_restart_count.get(worker_id, 0) + 1
                    )
                    logger.info(
                        "[VIT Scheduler] ✅ Worker %d restarted successfully (restart_count=%d)",
                        worker_id,
                        self.worker_restart_count[worker_id],
                    )
                    return True
                except Exception as retry_exc:
                    logger.warning(
                        "[VIT Scheduler] Worker %d reconnect attempt %d failed: %s",
                        worker_id,
                        attempt + 1,
                        retry_exc,
                    )
                    time.sleep(self.worker_restart_backoff)

            logger.error(f"[VIT Scheduler] Worker {worker_id} failed to reconnect after restart")
            return False
        except Exception as e:
            logger.error(f"[VIT Scheduler] Failed to restart Worker {worker_id}: {e}", exc_info=True)
            return False

    @staticmethod
    def _kill_process(proc: mp.Process, timeout: float = 5.0) -> None:
        """Terminate a multiprocessing.Process safely."""
        if proc is None:
            return
        if not proc.is_alive():
            proc.close()
            return
        proc.terminate()
        proc.join(timeout=timeout)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=timeout)
        proc.close()

    def _terminate_worker_processes(self, worker_id: int) -> None:
        """Terminate worker subprocesses started by the scheduler."""
        processes = self.worker_processes.pop(worker_id, [])
        for proc in processes:
            self._kill_process(proc)

        if processes:
            logger.info("[VIT Scheduler] Terminated tracked worker processes for worker %d", worker_id)
            return

        # Fallback: inspect system processes by environment variables
        target_port = str(self.vit_tp_port + worker_id)
        terminated_pids = []
        for proc in psutil.process_iter(attrs=["pid", "name"]):
            try:
                env = proc.environ()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
            if env.get("SGLANG_VIT_TP_PORT") == target_port:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
                terminated_pids.append(proc.pid)
        if terminated_pids:
            logger.info(
                "[VIT Scheduler] Terminated worker %d processes via psutil: %s",
                worker_id,
                terminated_pids,
            )

    def _spawn_worker_group(self, worker_id: int) -> bool:
        """Spawn a fresh worker group (all TP ranks) for the given worker_id."""
        if self.tp_size <= 0:
            return False

        processes: List[Tuple[mp.Process, mp.connection.Connection]] = []
        rpc_port = self.worker_rpc_port_start + worker_id
        env_overrides = {"SGLANG_VIT_WORKER_ID": str(worker_id)}
        try:
            success = True
            for tp_rank in range(self.tp_size):
                parent_conn, child_conn = self.worker_proc_ctx.Pipe(duplex=False)
                proc = self.worker_proc_ctx.Process(
                    target=_worker_process_bootstrap,
                    args=(
                        self.model_config,
                        worker_id,
                        tp_rank,
                        self.tp_size,
                        rpc_port,
                        self.cache_rpc_port,
                        child_conn,
                        env_overrides,
                    ),
                    daemon=True,
                )
                proc.start()
                processes.append((proc, parent_conn))

            for proc, conn in processes:
                if not conn.poll(timeout=30):
                    logger.error(
                        "[VIT Scheduler] Worker %d TP rank process startup timeout (pid=%s)",
                        worker_id,
                        proc.pid,
                    )
                    success = False
                    break
                message = conn.recv()
                if isinstance(message, str) and message.startswith("error:"):
                    logger.error(
                        "[VIT Scheduler] Worker %d TP rank process failed: %s",
                        worker_id,
                        message,
                    )
                    success = False
                    break

            if success:
                self.worker_processes[worker_id] = [proc for proc, _ in processes]
            else:
                for proc, _ in processes:
                    self._kill_process(proc)

            return success
        finally:
            for _, conn in processes:
                try:
                    conn.close()
                except Exception:
                    pass

    def _is_worker_gpu_saturated(self, worker_id: int) -> Tuple[bool, float]:
        """Check whether worker GPU usage exceeds configured high-water mark."""
        if (
            self.worker_gpu_high_water_frac <= 0.0
            or self.worker_gpu_high_water_frac >= 1.0
            or worker_id >= len(self.worker_clients)
        ):
            return False, 0.0

        client = self.worker_clients[worker_id]
        if client is None:
            return True, 1.0

        try:
            stats = client.root.get_memory_stats()
            total = float(stats.get("total", 0.0))
            used = float(stats.get("used", 0.0))
            if total <= 0:
                return False, 0.0
            usage_ratio = used / total
            self.worker_last_health_check[worker_id] = time.time()
            if usage_ratio >= self.worker_gpu_high_water_frac:
                return True, usage_ratio
            return False, usage_ratio
        except Exception as exc:
            logger.warning(
                "[VIT Scheduler] Failed to obtain GPU memory stats from worker %d: %s",
                worker_id,
                exc,
            )
        return False, 0.0

    def _check_and_recover_workers(self) -> None:
        """检查所有 Worker 的健康状态，必要时重启

        🔧 Phase 0: Worker Watchdog - 主动监控循环
        """
        if not self.use_worker_pool:
            return
        if not self.worker_restart_enabled:
            return

        current_time = time.time()

        for worker_id in range(len(self.worker_clients)):
            if self.worker_clients[worker_id] is None:
                continue
            last_check = self.worker_last_health_check.get(worker_id, current_time)
            time_since_last_check = current_time - last_check

            # 检查是否超时
            if time_since_last_check > self.worker_timeout_threshold:
                logger.error(
                    f"[VIT Scheduler] ❌ Worker {worker_id} timeout detected: "
                    f"no response for {time_since_last_check:.1f}s (threshold={self.worker_timeout_threshold}s)"
                )

                # 尝试重启
                if self._restart_worker(worker_id):
                    logger.info(f"[VIT Scheduler] Worker {worker_id} recovered after timeout")
                else:
                    logger.error(f"[VIT Scheduler] Worker {worker_id} recovery failed, marking as unhealthy")

            # 定期心跳检测
            elif time_since_last_check > self.worker_health_check_interval:
                is_healthy = self._check_worker_health(worker_id)
                if not is_healthy:
                    logger.warning(
                        f"[VIT Scheduler] Worker {worker_id} health check failed, "
                        f"attempting restart..."
                    )
                    self._restart_worker(worker_id)

    def _health_check_loop(self):
        """健康检查循环

        🔧 Phase 2.B: 定期检查 Worker 健康状态
        """
        logger.info("[VIT Scheduler] Health check thread started")

        while not self._stop_event.is_set():
            try:
                # 等待一段时间
                if self._stop_event.wait(self.worker_health_check_interval):
                    break

                # 检查所有 Worker
                for worker_id in range(self.vit_dp):
                    if not self._check_worker_health(worker_id):
                        logger.error(f"[VIT Scheduler] Worker {worker_id} is unhealthy!")
                        self.worker_failure_count += 1
                        # TODO: 实现 Worker 重启逻辑

            except Exception as e:
                logger.error(f"[VIT Scheduler] Health check loop error: {e}", exc_info=True)

        logger.info("[VIT Scheduler] Health check thread stopped")

    # ------------------------------------------------------------------
    # distributed initialisation
    # ------------------------------------------------------------------

    def _init_distributed(self) -> None:
        if not dist.is_initialized():
            logger.info(
                "[VIT Scheduler] Initialising distributed env rank=%d size=%d port=%d",
                self.tp_rank,
                self.tp_size,
                self.vit_tp_port,
            )
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://localhost:{self.vit_tp_port}",
                world_size=self.tp_size,
                rank=self.tp_rank,
            )
        torch.cuda.set_device(torch.device(f"cuda:{self.tp_rank}"))

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def run(self):
        """运行 VIT Scheduler

        🔑 Phase 4: 根据环境变量选择 threading 或 asyncio
        """
        use_asyncio = os.environ.get("SGLANG_VIT_USE_ASYNCIO", "0") == "1"

        if use_asyncio:
            logger.info("[VIT Scheduler] Using asyncio event loop")
            self.run_async()
        else:
            logger.info("[VIT Scheduler] Using threading model")
            self.run_sync()

    def run_sync(self):
        """同步运行模式（原 run() 方法）

        🔑 Phase 4: 重命名为 run_sync()，保留原有 threading 实现
        """
        if self.tp_rank > 0:
            self._run_tp_worker()
            return

        # 🔧 只在不使用 Worker Pool 时加载模型
        if not self.use_worker_pool:
            self.model_runner.load_model()

        self._io_thread = threading.Thread(
            target=self._io_loop, name="VIT-IO", daemon=True
        )

        # 🔧 Compute 线程改名为 Dispatch 线程 (Worker Pool 模式)
        thread_name = "VIT-Dispatch" if self.use_worker_pool else "VIT-Compute"
        self._compute_thread = threading.Thread(
            target=self._compute_loop, name=thread_name, daemon=True
        )

        self._free_thread = threading.Thread(
            target=self._free_loop, name="VIT-Free", daemon=True
        )

        # 🔧 监控线程 (定期输出统计信息)
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="VIT-Monitor", daemon=True
        )

        # 🔧 Phase 2.B: 健康检查线程 (Worker Pool 模式)
        if self.use_worker_pool:
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop, name="VIT-HealthCheck", daemon=True
            )

        self._io_thread.start()
        self._compute_thread.start()
        self._free_thread.start()
        self._monitor_thread.start()

        # 🔧 Phase 2.B: 启动健康检查线程
        if self._health_check_thread is not None:
            self._health_check_thread.start()
            logger.info("[VIT Scheduler] Health check thread started")

        mode = "Worker Pool" if self.use_worker_pool else "In-Process"
        logger.info(f"[VIT Scheduler] threads started (mode={mode})")

        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("[VIT Scheduler] keyboard interrupt, shutting down")
            self._stop_event.set()
        finally:
            self.cleanup()

    def cleanup(self):
        self._stop_event.set()

        # 🔧 Phase 2.B: 包含健康检查线程
        for thread in (self._io_thread, self._compute_thread, self._free_thread, self._monitor_thread, self._health_check_thread):
            if thread is not None:
                thread.join(timeout=1.0)

        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()

    # ------------------------------------------------------------------
    # Phase 4: asyncio event loop
    # ------------------------------------------------------------------

    def run_async(self):
        """asyncio 事件循环入口

        🔑 Phase 4: 新增方法，使用 asyncio/uvloop 事件循环
        """
        if self.tp_rank > 0:
            self._run_tp_worker()
            return

        # 只在不使用 Worker Pool 时加载模型
        if not self.use_worker_pool:
            self.model_runner.load_model()

        # 初始化 asyncio 队列
        self._request_queue_async = asyncio.Queue()
        self._free_queue_async = asyncio.Queue()

        # 创建 asyncio 事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        logger.info("[VIT Scheduler] Starting asyncio event loop")

        # 启动协程
        try:
            loop.run_until_complete(self._run_event_loop_async())
        except KeyboardInterrupt:
            logger.info("[VIT Scheduler] keyboard interrupt, shutting down")
            self._stop_event.set()
        finally:
            loop.close()
            self.cleanup()

    async def _run_event_loop_async(self):
        """运行所有异步循环

        🔑 Phase 4: 使用 asyncio.gather 并发运行所有循环
        """
        tasks = [
            self._io_loop_async(),
            self._compute_loop_async(),
            self._free_loop_async(),
            self._monitor_loop_async(),
        ]

        # 🔧 Phase 2.B: 健康检查循环（仅 Worker Pool 模式）
        if self.use_worker_pool:
            tasks.append(self._health_check_loop_async())

        await asyncio.gather(*tasks)

    async def _io_loop_async(self):
        """异步 I/O 循环 - 接收请求

        🔑 Phase 4: 动态批处理 - 自适应调整接收数量
        """
        if self.use_dynamic_batching and not hasattr(self, "visual_recv_max_count"):
            self.visual_recv_max_count = 64

        while not self._stop_event.is_set():
            try:
                pulled = 0

                if self.use_dynamic_batching:
                    # 🔑 动态批处理: 尝试拉取 visual_recv_max_count 个请求
                    for _ in range(self.visual_recv_max_count):
                        try:
                            data = self.socket.recv(zmq.NOBLOCK)
                            request = pickle.loads(data)
                            await self._request_queue_async.put(request)
                            pulled += 1
                        except zmq.ZMQError:
                            break

                    # 🔑 动态调整: 拉满了就增加上限
                    if pulled == self.visual_recv_max_count:
                        old_count = self.visual_recv_max_count
                        self.visual_recv_max_count = min(int(self.visual_recv_max_count * 1.3), 256)
                        if self.visual_recv_max_count != old_count:
                            logger.debug(f"[VIT Scheduler] Increased recv_max_count: {old_count} → {self.visual_recv_max_count}")
                    elif pulled == 0:
                        # 🔑 队列清空时下调
                        old_count = self.visual_recv_max_count
                        self.visual_recv_max_count = max(32, int(self.visual_recv_max_count / 1.3))
                        if self.visual_recv_max_count != old_count:
                            logger.debug(f"[VIT Scheduler] Decreased recv_max_count: {old_count} → {self.visual_recv_max_count}")
                else:
                    # 🔑 固定批处理: 单次接收
                    try:
                        data = self.socket.recv(zmq.NOBLOCK)
                        request = pickle.loads(data)
                        await self._request_queue_async.put(request)
                    except zmq.ZMQError:
                        pass

            except Exception as e:
                logger.error(f"[VIT Scheduler] I/O loop error: {e}", exc_info=True)

            await asyncio.sleep(0.01)  # 10ms

    async def _compute_loop_async(self):
        """异步计算循环 - 批处理 + Worker 调用

        🔑 Phase 4: 立即发送 + 流水线 + 动态批处理
        """
        pending = []
        first_enqueue_time = None

        while not self._stop_event.is_set():
            # 计算 flush deadline
            if first_enqueue_time is not None:
                elapsed = time.time() - first_enqueue_time
                if elapsed >= self.batch_timeout:
                    flush_deadline = time.time()
                else:
                    flush_deadline = first_enqueue_time + self.batch_timeout
            else:
                flush_deadline = time.time() + self.batch_timeout

            # 尝试接收请求
            timeout = max(0.001, flush_deadline - time.time())
            try:
                request = await asyncio.wait_for(
                    self._request_queue_async.get(),
                    timeout=timeout
                )

                # 🔑 Phase 3.1: 立即发送机制 - 检查缓存
                if self.cache_enabled and self.cache_client is not None:
                    cache_id = self._check_cache_hit(request)
                    if cache_id is not None:
                        # 🔑 cache hit，立即发送
                        self._send_cache_hit_response(request, cache_id)
                        logger.info(
                            f"[VIT Scheduler] ✅ Cache hit, sent immediately: "
                            f"request_id={request.request_id}, cache_id={cache_id}, hash={request.hash_val}"
                        )
                        continue  # 不加入 pending

                # cache miss，加入 pending
                pending.append(request)
                if first_enqueue_time is None:
                    first_enqueue_time = time.time()

            except asyncio.TimeoutError:
                pass

            # 检查是否需要 flush
            should_flush = False
            if len(pending) >= self.batch_size:
                should_flush = True
            elif len(pending) > 0 and time.time() >= flush_deadline:
                should_flush = True

            if should_flush:
                # 🔑 异步调用 Worker（简化版本：复用同步方法）
                # TODO: 完整实现 _dispatch_to_workers_async
                if self.use_worker_pool:
                    self._dispatch_to_workers(pending)
                else:
                    self._process_batch(pending)

                pending.clear()
                first_enqueue_time = None

            await asyncio.sleep(0.001)  # 1ms

    async def _free_loop_async(self):
        """异步释放循环 - 缓存释放

        🔑 Phase 4: 转换为 asyncio
        """
        while not self._stop_event.is_set():
            try:
                request_id = await asyncio.wait_for(
                    self._free_queue_async.get(),
                    timeout=0.1
                )

                # 释放逻辑（复用同步方法）
                with self._free_lock:
                    cache_id = self._request_to_cache.pop(request_id, None)
                    if cache_id is not None and self.cache_client is not None:
                        try:
                            self.cache_client.root.release(cache_id)
                            logger.debug(f"[VIT Scheduler] Released cache: request_id={request_id}, cache_id={cache_id}")
                        except Exception as e:
                            logger.error(f"[VIT Scheduler] Failed to release cache: {e}")

            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.error(f"[VIT Scheduler] Free loop error: {e}", exc_info=True)

            await asyncio.sleep(0.001)

    async def _monitor_loop_async(self):
        """异步监控循环 - 统计信息

        🔑 Phase 4: 转换为 asyncio
        """
        while not self._stop_event.is_set():
            await asyncio.sleep(30.0)

            # 输出统计信息（复用同步方法的逻辑）
            if self.total_requests > 0:
                hit_rate = self.cache_hits / self.total_requests * 100
                avg_time = self.total_compute_time / self.total_requests * 1000
                logger.info(
                    f"[VIT Scheduler] Stats: total={self.total_requests}, "
                    f"cache_hit_rate={hit_rate:.1f}%, avg_compute_time={avg_time:.1f}ms"
                )

    async def _health_check_loop_async(self):
        """异步健康检查循环 - Worker Pool 健康检查

        🔑 Phase 4: 转换为 asyncio
        """
        while not self._stop_event.is_set():
            await asyncio.sleep(self.worker_health_check_interval)

            # 健康检查逻辑（复用同步方法）
            for worker_id in range(len(self.worker_clients)):
                try:
                    # 简单的 ping 检查
                    self.worker_clients[worker_id].ping(timeout=5.0)
                    self.worker_last_health_check[worker_id] = time.time()
                except Exception as e:
                    logger.warning(f"[VIT Scheduler] Worker {worker_id} health check failed: {e}")

        # 🔧 关闭 Worker Pool ThreadPoolExecutor
        if self.worker_executor is not None:
            logger.info("[VIT Scheduler] Shutting down worker executor")
            self.worker_executor.shutdown(wait=False)
            self.worker_executor = None

        if self.worker_processes:
            for worker_id in list(self.worker_processes.keys()):
                self._terminate_worker_processes(worker_id)

        # 🔧 关闭 Worker Pool RPC 连接
        if self.use_worker_pool and self.worker_clients:
            for worker_id, client in enumerate(self.worker_clients):
                try:
                    client.close()
                    logger.info(f"[VIT Scheduler] Closed Worker {worker_id} connection")
                except Exception as e:
                    logger.warning(f"[VIT Scheduler] Failed to close Worker {worker_id}: {e}")
            self.worker_clients = []

        # 🔧 根据架构选择清理方式
        if self.use_new_arch:
            if self.cache_client is not None:
                self.cache_client.close()
                logger.info("[VIT Scheduler] Closed CacheServer connection")
        else:
            if self.cache_server is not None:
                self.cache_server.cleanup()

        logger.info("[VIT Scheduler] cleanup complete")

    # ------------------------------------------------------------------
    # IO / compute loops
    # ------------------------------------------------------------------

    def _io_loop(self):
        assert self.socket is not None
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        while not self._stop_event.is_set():
            events = dict(poller.poll(timeout=50))
            if self.socket not in events:
                continue
            try:
                message = self.socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                continue
            except Exception as exc:
                logger.error("[VIT Scheduler][IO] receive error: %s", exc, exc_info=True)
                continue

            try:
                payload = pickle.loads(message)
            except Exception as exc:
                logger.error("[VIT Scheduler][IO] decode error: %s", exc, exc_info=True)
                continue

            if payload.get("type") == "free_embedding":
                # 🔧 新架构优先使用 cache_id
                cache_id = payload.get("cache_id")
                hash_val = payload.get("image_hash")

                if self.use_new_arch and cache_id is not None:
                    # 新架构: 使用 cache_id
                    with self._free_lock:
                        self._free_queue.append(cache_id)
                    logger.debug(f"[VIT Scheduler][IO] Queued free signal: cache_id={cache_id}")
                elif hash_val is not None:
                    # 旧架构: 使用 hash_val
                    with self._free_lock:
                        self._free_queue.append(hash_val)
                    logger.debug(f"[VIT Scheduler][IO] Queued free signal: hash={hash_val}")
                continue

            if payload.get("test") == "connection_test":
                self.socket.send(pickle.dumps({"test_response": "ok"}))
                continue

            request = VITRequest(**payload)
            self._request_queue.put(request)
            self.total_requests += 1
            logger.debug(
                "[VIT Scheduler][IO] queued request=%s pending=%d",
                request.request_id,
                self._request_queue.qsize(),
            )

    def _compute_loop(self):
        """计算/调度循环

        - Worker Pool 模式: 调度到 Worker Pool
        - In-Process 模式: 本地计算

        🔧 批处理聚合逻辑:
        - 达到 batch_size 时立即处理
        - 超过 batch_timeout 时处理当前批次
        - 这样可以平衡延迟和吞吐量

        🔧 Phase 0: Worker Watchdog 集成
        - 定期检查 Worker 健康状态
        - 超时自动重启
        """
        pending: List[VITRequest] = []
        flush_deadline = time.time() + self.batch_timeout
        first_enqueue_time = None  # 🔧 记录第一个请求入队时间
        last_watchdog_check = time.time()  # 🔧 Phase 0: 上次 Watchdog 检查时间

        # 🔧 根据模式选择处理函数
        process_fn = self._dispatch_to_workers if self.use_worker_pool else self._process_batch

        while not self._stop_event.is_set():
            # 🔧 Phase 0: Worker Watchdog - 定期检查 Worker 健康状态
            current_time = time.time()
            if self.use_worker_pool and (current_time - last_watchdog_check) > self.worker_health_check_interval:
                self._check_and_recover_workers()
                last_watchdog_check = current_time
            # 🔧 达到 batch_size，立即处理
            if len(pending) >= self.batch_size:
                wait_time = time.time() - first_enqueue_time if first_enqueue_time else 0
                logger.info(
                    f"[VIT Scheduler] Batch full: size={len(pending)}, wait_time={wait_time*1000:.1f}ms"
                )
                process_fn(pending)
                pending = []
                flush_deadline = time.time() + self.batch_timeout
                first_enqueue_time = None
                continue

            timeout = max(0.0, flush_deadline - time.time())
            try:
                request = self._request_queue.get(timeout=timeout)

                # 🔑 Phase 3.1: 立即发送机制 - 检查缓存
                if self.cache_enabled and self.cache_client is not None:
                    cache_id = self._check_cache_hit(request)
                    if cache_id is not None:
                        # 🔑 cache hit，立即发送
                        self._send_cache_hit_response(request, cache_id)
                        logger.info(
                            f"[VIT Scheduler] ✅ Cache hit, sent immediately: "
                            f"request_id={request.request_id}, cache_id={cache_id}, hash={request.hash_val}"
                        )
                        continue  # 不加入 pending

                # cache miss，加入 pending
                pending.append(request)
                if first_enqueue_time is None:
                    first_enqueue_time = time.time()
                logger.debug(
                    f"[VIT Scheduler] Request queued: {request.request_id}, pending={len(pending)}/{self.batch_size}"
                )
            except Empty:
                # 🔧 超时，处理当前批次
                if pending:
                    wait_time = time.time() - first_enqueue_time if first_enqueue_time else 0
                    logger.info(
                        f"[VIT Scheduler] Batch timeout: size={len(pending)}, wait_time={wait_time*1000:.1f}ms, timeout={self.batch_timeout*1000:.1f}ms"
                    )
                    process_fn(pending)
                    pending = []
                    first_enqueue_time = None
                flush_deadline = time.time() + self.batch_timeout

        # 🔧 退出时处理剩余请求
        if pending:
            logger.info(f"[VIT Scheduler] Processing remaining batch: size={len(pending)}")
            process_fn(pending)

    def _free_loop(self):
        while not self._stop_event.is_set():
            hash_val = None
            with self._free_lock:
                if self._free_queue:
                    hash_val = self._free_queue.popleft()
            if hash_val is None:
                time.sleep(0.05)
                continue

            # 🔧 根据架构选择释放方式
            if self.use_new_arch:
                # 新架构: hash_val 实际上是 cache_id
                if self.cache_client is not None:
                    try:
                        self.cache_client.root.release(hash_val)
                        logger.debug("[VIT Scheduler][Free] released cache_id=%s (NEW ARCH)", hash_val)
                    except Exception as e:
                        logger.error(f"[VIT Scheduler][Free] Failed to release cache_id={hash_val}: {e}")
            else:
                # 旧架构: hash_val 是内容哈希
                self.cache_server.release(hash_val)
                logger.debug("[VIT Scheduler][Free] released hash=%s", hash_val)

    def _monitor_loop(self):
        """监控线程：定期输出统计信息"""
        monitor_interval = 30.0  # 每 30 秒输出一次统计

        while not self._stop_event.is_set():
            time.sleep(monitor_interval)

            try:
                # 🔧 输出 CacheServer 统计
                if self.use_new_arch and self.cache_client is not None:
                    stats_ref = self.cache_client.root.get_stats()
                    stats = obtain(stats_ref)
                    if not isinstance(stats, dict):
                        stats = dict(stats)
                    total_requests = stats.get("total_requests", 0)
                    cache_hits = stats.get("cache_hits", 0)
                    cache_misses = stats.get("cache_misses", 0)
                    evictions = stats.get("evictions", 0)
                    hit_rate = cache_hits / total_requests * 100 if total_requests > 0 else 0.0

                    logger.info(
                        f"[VIT Monitor] CacheServer Stats: "
                        f"requests={total_requests}, hits={cache_hits}, misses={cache_misses}, "
                        f"evictions={evictions}, hit_rate={hit_rate:.1f}%"
                    )

                # 🔧 输出 Worker Pool 统计
                if self.use_worker_pool and self.worker_clients:
                    for worker_id, client in enumerate(self.worker_clients):
                        try:
                            worker_stats_ref = client.root.get_stats()
                            worker_stats = obtain(worker_stats_ref)
                            if not isinstance(worker_stats, dict):
                                worker_stats = dict(worker_stats)
                            logger.info(
                                f"[VIT Monitor] Worker {worker_id} Stats: "
                                f"requests={worker_stats.get('total_requests', 0)}, "
                                f"cache_hits={worker_stats.get('total_cache_hits', 0)}, "
                                f"avg_compute_time={worker_stats.get('avg_compute_time', 0):.3f}s"
                            )
                        except Exception as e:
                            logger.warning(f"[VIT Monitor] Failed to get stats from Worker {worker_id}: {e}")

                    # 🔧 Phase 2.B: 输出健壮性统计
                    logger.info(
                        f"[VIT Monitor] Scheduler Stats: "
                        f"timeouts={self.worker_timeout_count}, "
                        f"retries={self.worker_retry_count}, "
                        f"failures={self.worker_failure_count}"
                    )

                # 🔧 输出旧架构统计（兼容性）
                if not self.use_new_arch and hasattr(self, 'cache_server') and self.cache_server is not None:
                    try:
                        stats = self.cache_server.get_stats()
                        logger.info(
                            "[VIT Monitor] Legacy Cache Stats: entries=%d total=%.2fMB max=%.2fMB",
                            stats.get("num_entries", 0),
                            stats.get("total_bytes", 0) / (1024**2),
                            stats.get("max_bytes", 0) / (1024**2),
                        )
                    except Exception as e:
                        logger.warning(f"[VIT Monitor] Failed to get legacy cache stats: {e}")

            except Exception as e:
                logger.error(f"[VIT Monitor] Error in monitor loop: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # batch processing
    # ------------------------------------------------------------------

    def _dispatch_to_workers(self, requests: List[VITRequest]) -> None:
        """将请求分配到 Worker Pool (DP 轮询) - 并发版本"""
        if not requests:
            return

        total_batch = len(requests)
        logger.info(
            "[VIT Scheduler] 🚀 Dispatching %d request(s) to worker pool (batch_timeout=%.1fms)",
            total_batch,
            self.batch_timeout * 1000,
        )

        # 按 Worker 分组 (DP 轮询)
        if not self.worker_clients:
            for request in requests:
                self._request_queue.put(request)
            logger.warning("[VIT Scheduler] No worker clients available; re-queued %d request(s)", len(requests))
            return

        tasks_per_worker = [[] for _ in range(len(self.worker_clients))]
        for request in requests:
            assigned = False
            attempts = 0
            while attempts < len(self.worker_clients):
                worker_id = self.next_worker_id % len(self.worker_clients)
                self.next_worker_id += 1
                if self.worker_clients[worker_id] is None:
                    attempts += 1
                    continue
                tasks_per_worker[worker_id].append(request)
                assigned = True
                break
            if not assigned:
                self._request_queue.put(request)


        # 🔧 并发调用 Worker (使用 ThreadPoolExecutor)
        # 🔑 Phase 3.2: 流水线发送 - Worker 完成后立即发送结果
        def call_worker(worker_id: int, tasks: List[VITRequest]):
            """调用单个 Worker 的辅助函数

            🔧 Phase 2.B: 支持重试逻辑
            🔑 Phase 3.2: 流水线发送 - 立即处理结果
            """
            if not tasks:
                return worker_id, []

            # 准备参数
            request_ids = [t.request_id for t in tasks]
            pixel_values_shm_keys = [t.pixel_values_shm_name for t in tasks]
            pixel_values_shapes = [t.pixel_values_shape for t in tasks]
            pixel_values_dtypes = [t.pixel_values_dtype for t in tasks]
            image_grid_thw_shm_keys = [t.image_grid_thw_shm_name for t in tasks]
            image_grid_thw_shapes = [t.image_grid_thw_shape for t in tasks]
            image_grid_thw_dtypes = [t.image_grid_thw_dtype for t in tasks]
            content_hashes = [
                t.hash_val if t.hash_val is not None else 0 for t in tasks
            ]

            # 🔧 Phase 2.B: 重试逻辑 (最多重试 1 次)
            max_retries = 1
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    results = self.worker_clients[worker_id].root.forward(
                        request_ids,
                        pixel_values_shm_keys,
                        pixel_values_shapes,
                        pixel_values_dtypes,
                        image_grid_thw_shm_keys,
                        image_grid_thw_shapes,
                        image_grid_thw_dtypes,
                        content_hashes,
                    )

                    # 成功，记录重试次数
                    if attempt > 0:
                        self.worker_retry_count += 1
                        logger.info(f"[VIT Scheduler] Worker {worker_id} succeeded after {attempt} retries")

                    self.worker_last_health_check[worker_id] = time.time()

                    # 🔑 Phase 3.2: 流水线发送 - 立即处理结果，不等待其他 Worker
                    for task, result in zip(tasks, results):
                        self._handle_worker_result(task, result)

                    logger.debug(f"[VIT Scheduler] Worker {worker_id} completed and sent {len(tasks)} results")
                    return worker_id, []  # 已经处理完毕，返回空列表

                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(f"[VIT Scheduler] Worker {worker_id} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        time.sleep(0.1)  # 短暂延迟后重试
                    else:
                        logger.error(f"[VIT Scheduler] Worker {worker_id} failed after {max_retries + 1} attempts: {e}", exc_info=True)
                        self.worker_failure_count += 1

            # 🔑 Phase 3.2: 所有重试都失败，立即发送错误结果
            for task in tasks:
                self._handle_worker_result(
                    task,
                    {
                        "request_id": task.request_id,
                        "error": True,
                        "error_message": str(last_error),
                    }
                )
            return worker_id, []  # 已经处理完毕，返回空列表

        # 提交所有 Worker 任务
        futures = []
        future_to_tasks: Dict = {}
        high_water_enabled = 0.0 < self.worker_gpu_high_water_frac < 1.0
        for worker_id, tasks in enumerate(tasks_per_worker):
            if not tasks:
                continue
            if worker_id >= len(self.worker_clients) or self.worker_clients[worker_id] is None:
                for task in tasks:
                    self._request_queue.put(task)
                continue
            if high_water_enabled:
                saturated, usage_ratio = self._is_worker_gpu_saturated(worker_id)
                if saturated:
                    self.worker_gpu_overflow_count[worker_id] += 1
                    if self.worker_gpu_overflow_count[worker_id] >= self.worker_gpu_overflow_limit:
                        logger.warning(
                            "[VIT Scheduler] Worker %d GPU usage %.2f%% exceeds %.2f%% for %d consecutive checks; cooling down %.1fs",
                            worker_id,
                            usage_ratio * 100.0,
                            self.worker_gpu_high_water_frac * 100.0,
                            self.worker_gpu_overflow_limit,
                            self.worker_gpu_cooldown_s,
                        )
                        for task in tasks:
                            self._request_queue.put(task)
                        self.worker_gpu_overflow_count[worker_id] = 0
                        time.sleep(self.worker_gpu_cooldown_s)
                        continue
                    else:
                        logger.debug(
                            "[VIT Scheduler] Worker %d temporarily above GPU high-water mark (%.2f%%); allowing dispatch (count=%d/%d)",
                            worker_id,
                            usage_ratio * 100.0,
                            self.worker_gpu_overflow_count[worker_id],
                            self.worker_gpu_overflow_limit,
                        )
                else:
                    self.worker_gpu_overflow_count[worker_id] = 0
            else:
                self.worker_gpu_overflow_count[worker_id] = 0

            future = self.worker_executor.submit(call_worker, worker_id, tasks)
            futures.append(future)
            future_to_tasks[future] = (worker_id, tasks)

        # 🔑 Phase 3.2: 流水线发送 - 等待所有任务完成（结果已在 call_worker 中处理）
        if not futures:
            return

        try:
            for future in as_completed(futures, timeout=self.worker_rpc_timeout):
                try:
                    worker_id, task_results = future.result()
                    # 🔑 Phase 3.2: task_results 已经是空列表（结果已在 call_worker 中处理）
                    if task_results:
                        logger.warning(f"[VIT Scheduler] Unexpected task_results from worker {worker_id}: {len(task_results)}")
                except Exception as e:
                    logger.error(f"[VIT Scheduler] Failed to get worker result: {e}", exc_info=True)
        except FuturesTimeoutError:
            # 🔧 Phase 2.B: 记录超时统计
            self.worker_timeout_count += 1
            logger.error(
                "[VIT Scheduler] Worker RPC timeout after %.1fs. Cancelling pending tasks.",
                self.worker_rpc_timeout,
            )
            for future in futures:
                if not future.done():
                    worker_id, tasks = future_to_tasks.get(future, (None, []))
                    future.cancel()
                    # 🔑 Phase 3.2: 立即发送超时错误
                    for task in tasks:
                        self._handle_worker_result(
                            task,
                            {
                                "cache_id": None,
                                "from_cache": False,
                                "compute_time": 0.0,
                                "error": True,
                                "error_message": f"Worker {worker_id} timeout",
                            },
                        )

    def _handle_worker_result(self, request: VITRequest, result: Dict) -> None:
        """处理 Worker 返回的结果

        🔧 缓存禁用时，cache_id=None 是正常的，Worker 会直接通过 SHM 返回 embedding
        """
        if result.get("error", False):
            # 错误结果
            self._send_error_response(
                request,
                result.get("cache_id"),
                result.get("error_message", "Unknown error")
            )
        else:
            # 成功结果
            cache_id = result.get("cache_id")
            from_cache = result.get("from_cache", False)
            compute_time = result.get("compute_time", 0.0)

            # 🔧 缓存启用时，从缓存读取 embedding
            if cache_id is not None and self.cache_enabled:
                shm_key = self.cache_client.root.get_shm_key(cache_id)
                from sglang.srt.managers.vit_shm_utils import read_embedding_from_shm_raw

                embedding = read_embedding_from_shm_raw(shm_key)
                if embedding is not None:
                    # 写入请求级 SHM
                    from sglang.srt.managers.vit_shm_utils import (
                        cleanup_embedding_shm,
                        write_embedding_to_shm,
                    )
                    # 若已有残留，先清理再写入，避免 FileExistsError
                    cleanup_embedding_shm(request.request_id)
                    success = write_embedding_to_shm(request.request_id, embedding)
                    if success:
                        self._send_embedding(
                            request, embedding, cache_id, from_cache, compute_time
                        )
                    else:
                        self._send_error_response(request, cache_id, "Failed to write embedding to SHM")
                else:
                    self._send_error_response(request, cache_id, "Failed to read embedding from cache")
            elif cache_id is None and not self.cache_enabled:
                # 🔧 缓存禁用时，Worker 直接通过请求级 SHM 返回 embedding
                # 直接从请求级 SHM 读取（Worker 已经写入）
                from sglang.srt.managers.vit_shm_utils import read_embedding_from_shm

                embedding = read_embedding_from_shm(request.request_id)
                if embedding is not None:
                    self._send_embedding(
                        request, embedding, None, from_cache, compute_time
                    )
                else:
                    self._send_error_response(request, None, "Failed to read embedding from request SHM")
            else:
                # 🔧 其他情况：cache_id=None 但缓存启用，或 cache_id!=None 但缓存禁用
                self._send_error_response(request, cache_id, f"Invalid state: cache_id={cache_id}, cache_enabled={self.cache_enabled}")

    def _process_batch(self, requests: List[VITRequest]) -> None:
        if not requests:
            return

        t_start = time.time()

        cache_hits: List[Tuple[VITRequest, int]] = []
        cache_misses: List[Tuple[VITRequest, torch.Tensor, torch.Tensor, int]] = []
        batch_pixel_tokens = 0

        for request in requests:
            # 🔑 修复 1: Scheduler 不再读取 SHM，hash_val 由 PP0 预计算
            # 如果 hash_val 未设置，说明 PP0 版本过旧，跳过该请求
            hash_val = request.hash_val
            if hash_val is None:
                message = "hash_val not set (PP0 version too old or hash computation failed)"
                logger.error(
                    "[VIT Scheduler] ❌ hash_val not set for request=%s, skipping",
                    request.request_id,
                )
                self._send_error_response(request, 0, message)
                continue

            # 🔑 修复 1: 大小检查也在 PP0 完成，Scheduler 只检查 shape
            pixel_values_size = request.pixel_values_shape[0] if request.pixel_values_shape else 0
            if pixel_values_size > self.max_pixel_values:
                message = (
                    f"pixel_values size {pixel_values_size} exceeds limit {self.max_pixel_values}"
                )
                logger.warning(
                    "[VIT Scheduler] request %s too large: pixel_values.shape=%s limit=%d",
                    request.request_id,
                    tuple(request.pixel_values_shape),
                    self.max_pixel_values,
                )
                self._send_error_response(request, hash_val, message)
                continue

            # 🔧 根据架构选择缓存检查方式
            cache_contains = False
            if not self.benchmark_mode and self.cache_enabled:
                if self.use_new_arch:
                    if self.cache_client is not None:
                        try:
                            cache_contains = self.cache_client.root.contains(hash_val)
                        except Exception as e:
                            logger.error(f"[VIT Scheduler] Failed to check cache: {e}")
                else:
                    cache_contains = self.cache_server.contains(hash_val)

            if cache_contains:
                cache_hits.append((request, hash_val))
            else:
                # 🔑 修复 1: 使用 shape 而不是读取 tensor
                pixel_values_size = request.pixel_values_shape[0] if request.pixel_values_shape else 0
                if self.max_batch_pixel_tokens > 0:
                    projected = batch_pixel_tokens + pixel_values_size
                    if projected > self.max_batch_pixel_tokens:
                        message = (
                            f"Batch pixel token budget exceeded: current={batch_pixel_tokens}, "
                            f"request={pixel_values_size}, limit={self.max_batch_pixel_tokens}"
                        )
                        logger.warning(
                            "[VIT Scheduler] request %s skipped due to batch pixel limit: %s",
                            request.request_id,
                            message,
                        )
                        self._send_error_response(request, hash_val, message)
                        continue
                    batch_pixel_tokens = projected
                # 🔑 修复 1: cache_misses 不再包含 tensor，只包含 request 和 hash_val
                cache_misses.append((request, hash_val))

        if cache_misses:
            self._handle_cache_misses(cache_misses)

        for request, hash_val in cache_hits:
            self._handle_cache_hit(request, hash_val)

        batch_time = time.time() - t_start
        if cache_misses:
            self.total_compute_time += batch_time
        logger.info(
            "[VIT Scheduler] processed batch size=%d hits=%d misses=%d time=%.1fms",
            len(requests),
            len(cache_hits),
            len(cache_misses),
            batch_time * 1000,
        )

    def _handle_cache_hit(self, request: VITRequest, hash_val: int) -> None:
        # 🔧 根据架构选择缓存读取方式
        tensor = None
        cache_id = None

        if self.use_new_arch:
            # 新架构: 通过 CacheServer RPC 获取
            if self.cache_client is not None:
                try:
                    # 🔧 分配缓存槽位 (增加引用计数)，返回 (cache_id, is_new)
                    result = self.cache_client.root.alloc(hash_val, 0)
                    if result is not None:
                        cache_id, is_new = result
                        # 获取 SHM key
                        shm_key = self.cache_client.root.get_shm_key(cache_id)
                        if shm_key:
                            # 🔧 使用 raw 函数读取 (不添加前缀)
                            tensor = read_embedding_from_shm_raw(shm_key)
                except Exception as e:
                    logger.error(f"[VIT Scheduler] Failed to read from cache: {e}")
        else:
            # 旧架构: 直接从进程内 CacheServer 读取
            retained = self.cache_server.retain(hash_val)
            tensor = self.cache_server.get(hash_val) if retained else None
            if tensor is None and retained:
                self.cache_server.release(hash_val)

        if tensor is None:
            logger.warning(
                "[VIT Scheduler] cache hit fallback: hash=%s missing, forcing compute",
                hash_val,
            )
            pixel_values = _load_tensor_from_shared_memory(
                request.pixel_values_shm_name,
                request.pixel_values_shape,
                request.pixel_values_dtype,
            )
            image_grid_thw = _load_tensor_from_shared_memory(
                request.image_grid_thw_shm_name,
                request.image_grid_thw_shape,
                request.image_grid_thw_dtype,
            )
            self._handle_cache_misses([(request, pixel_values, image_grid_thw, hash_val)])
            return

        self.cache_hits += 1
        # 🔧 新架构使用 cache_id, 旧架构使用 hash_val
        hash_or_id = cache_id if self.use_new_arch else hash_val
        self._send_embedding(request, tensor, hash_or_id, from_cache=True, compute_time=0.0)

    def _handle_cache_misses(
        self,
        misses: List[Tuple[VITRequest, int]],
    ) -> None:
        """处理缓存未命中的请求 (派发给 Worker)

        🔑 修复 1: Scheduler 不再读取 SHM 和计算，直接派发给 Worker
        Worker 负责读取 SHM、计算 embedding、写入缓存
        """
        if not misses:
            return

        # 🔑 修复 1: 直接派发给 Worker，Worker 会读取 SHM
        requests = [item[0] for item in misses]

        # 🔧 批处理日志
        total_pixels = sum(req.pixel_values_shape[0] if req.pixel_values_shape else 0 for req in requests)
        logger.info(
            f"[VIT Scheduler] 🚀 Dispatching cache misses to worker: "
            f"batch_size={len(misses)}, total_pixels={total_pixels}"
        )

        # 🔑 修复 1: 调用 Worker RPC
        # Worker 会处理：读取 SHM → 计算 embedding → 写入缓存 → 返回结果
        self._dispatch_to_workers(requests)

    def _check_cache_hit(self, request: VITRequest) -> Optional[int]:
        """检查缓存是否命中

        🔑 Phase 3.1: 新增方法，用于 Scheduler 端检查缓存
        🔑 修复 1: 直接使用 request.hash_val，不再读取 SHM

        Returns:
            cache_id: 如果命中返回 cache_id，否则返回 None
        """
        try:
            # 🔑 修复 1: 直接使用 request.hash_val，不再读取 SHM
            content_hash = request.hash_val
            if content_hash is None:
                # hash_val 未设置，说明 PP0 版本过旧或计算失败
                logger.warning(f"[VIT Scheduler] hash_val not set for request={request.request_id}, cannot check cache")
                return None

            # 🔑 RPC 调用 CacheServer 检查缓存
            cache_id = self.cache_client.root.get_cache_id(content_hash)

            return cache_id
        except Exception as e:
            logger.warning(f"[VIT Scheduler] Failed to check cache: {e}")
            return None

    def _send_cache_hit_response(self, request: VITRequest, cache_id: int):
        """立即发送 cache hit 响应

        🔑 Phase 3.1: 新增方法，用于立即发送 cache hit 响应
        🔧 Phase 3 补充: 添加重试次数限制，防止回旋
        """
        # 🔧 Phase 3 补充: 检查重试次数
        MAX_CACHE_HIT_RETRY = 2
        if request.cache_hit_retry_count >= MAX_CACHE_HIT_RETRY:
            logger.warning(
                f"[VIT Scheduler] Cache hit fallback: request {request.request_id} exceeded retry limit ({MAX_CACHE_HIT_RETRY}), "
                f"forcing compute as cache miss"
            )
            # 强制走 miss 流程
            request.hash_val = None  # 清空 hash_val，强制重新计算
            request.cache_hit_retry_count = 0  # 重置计数
            self._request_queue.put(request)
            return

        try:
            # 🔑 从 CacheServer 读取 embedding
            shm_key = self.cache_client.root.get_shm_key(cache_id)
            if shm_key is None:
                logger.error(
                    f"[VIT Scheduler] ❌ Cache hit fallback: SHM key not found - "
                    f"request_id={request.request_id}, cache_id={cache_id}, retry={request.cache_hit_retry_count + 1}/{MAX_CACHE_HIT_RETRY}"
                )
                # 回退到正常流程，增加重试计数
                request.cache_hit_retry_count += 1
                self._request_queue.put(request)
                return

            embedding = read_embedding_from_shm_raw(shm_key)
            if embedding is None:
                logger.error(
                    f"[VIT Scheduler] ❌ Cache hit fallback: failed to read embedding - "
                    f"request_id={request.request_id}, cache_id={cache_id}, shm_key={shm_key}, "
                    f"retry={request.cache_hit_retry_count + 1}/{MAX_CACHE_HIT_RETRY}"
                )
                # 回退到正常流程，增加重试计数
                request.cache_hit_retry_count += 1
                self._request_queue.put(request)
                # 释放引用计数
                self.cache_client.root.release(cache_id)
                return

            # 🔑 写入请求级 SHM
            from sglang.srt.managers.vit_shm_utils import cleanup_embedding_shm
            cleanup_embedding_shm(request.request_id)

            success = write_embedding_to_shm(request.request_id, embedding)
            if not success:
                logger.error(f"[VIT Scheduler] Failed to write cache hit embedding to SHM: {request.request_id}")
                # 回退到正常流程
                self._request_queue.put(request)
                # 释放引用计数
                self.cache_client.root.release(cache_id)
                return

            # 🔑 发送响应
            response = VITResponse(
                request_id=request.request_id,
                embedding_ipc_handle=([], 0),
                embedding_shape=tuple(embedding.shape),
                embedding_dtype=str(embedding.dtype).replace("torch.", ""),
                embedding_device="cpu",
                image_hash=0,  # 旧架构兼容性
                cache_id=cache_id,
                compute_time=0.0,
                from_cache=True,
                error=False,
                error_message="",
                vit_compute_start_time=time.time(),
                vit_compute_end_time=time.time(),
                # 🔑 方案 1: 返回输入 SHM 名称，用于 PP0 release
                input_shm_names=[
                    request.pixel_values_shm_name,
                    request.image_grid_thw_shm_name,
                ],
            )
            self._send_response(response)

            # 统计
            self.cache_hits += 1

            logger.debug(f"[VIT Scheduler] Cache hit response sent: {request.request_id}, cache_id={cache_id}")
        except Exception as e:
            logger.error(f"[VIT Scheduler] Failed to send cache hit response: {e}", exc_info=True)
            # 失败时回退到正常流程
            self._request_queue.put(request)
            # 释放引用计数
            try:
                self.cache_client.root.release(cache_id)
            except:
                pass

    def _send_embedding(
        self,
        request: VITRequest,
        embedding: torch.Tensor,
        hash_or_id: Optional[int],  # 🔧 新架构传 cache_id，旧架构传 hash_val
        from_cache: bool,
        compute_time: float,
    ) -> None:
        """发送 embedding 到 Client (通过 SHM)

        🔧 CPU Staging: 确保 embedding 在 CPU 上，对齐 LightLLM
        - ViT 计算后立即 to(cpu)
        - 写入 SHM (CPU)
        - Client 读取后保持在 CPU
        - 只在 LLM Prefill 时搬到 GPU
        """
        # 🔧 CPU Staging: 确保 embedding 在 CPU
        if embedding.is_cuda:
            logger.debug(
                f"[VIT Scheduler] Moving embedding to CPU for staging: {request.request_id}, device={embedding.device}"
            )
            embedding = embedding.cpu()

        logger.debug(
            f"[VIT Scheduler] Embedding ready for SHM write: {request.request_id}, "
            f"shape={tuple(embedding.shape)}, dtype={embedding.dtype}, device={embedding.device}"
        )

        from sglang.srt.managers.vit_shm_utils import cleanup_embedding_shm

        # 清理旧的请求级 SHM，避免重复写入失败
        cleanup_embedding_shm(request.request_id)

        success = write_embedding_to_shm(request.request_id, embedding)
        if not success:
            logger.error(
                "[VIT Scheduler] failed to write embedding shm for request=%s",
                request.request_id,
            )
            # 🔧 发送错误响应
            self._send_error_response(request, hash_or_id, "Failed to write embedding to SHM")
            return

        # 🔧 区分新旧架构
        if self.use_new_arch:
            # 新架构: hash_or_id 是 cache_id
            image_hash = 0  # 旧架构兼容性
            cache_id = hash_or_id
        else:
            # 旧架构: hash_or_id 是 image_hash
            image_hash = hash_or_id if hash_or_id is not None else 0
            cache_id = None

        response = VITResponse(
            request_id=request.request_id,
            embedding_ipc_handle=([], 0),
            embedding_shape=tuple(embedding.shape),
            embedding_dtype=str(embedding.dtype).replace("torch.", ""),
            embedding_device="cpu",
            image_hash=image_hash,  # 🔧 旧架构使用
            cache_id=cache_id,  # 🔧 新架构使用
            compute_time=compute_time,
            from_cache=from_cache,
            error=False,
            error_message="",
            vit_compute_start_time=time.time(),
            vit_compute_end_time=time.time(),
            # 🔑 方案 1: 返回输入 SHM 名称，用于 PP0 release
            input_shm_names=[
                request.pixel_values_shm_name,
                request.image_grid_thw_shm_name,
            ],
        )
        self._send_response(response)

    def _send_response(self, response: VITResponse) -> None:
        if self.socket is None:
            logger.error("[VIT Scheduler] socket is None, cannot send response")
            return
        try:
            self.socket.send(pickle.dumps(response.__dict__), flags=zmq.NOBLOCK)
        except Exception as exc:
            logger.error("[VIT Scheduler] failed to send response: %s", exc, exc_info=True)

    def _send_error_response(self, request: VITRequest, hash_or_id: Optional[int], error_message: str = "Unknown error") -> None:
        """发送错误响应

        🔑 方案 1: 即使失败也要返回 input_shm_names，让 PP0 可以 release SHM
        """
        # 🔧 区分新旧架构
        if self.use_new_arch:
            image_hash = 0
            cache_id = hash_or_id
        else:
            image_hash = hash_or_id if hash_or_id is not None else 0
            cache_id = None

        response = VITResponse(
            request_id=request.request_id,
            embedding_ipc_handle=([], 0),
            embedding_shape=(0, 0),
            embedding_dtype="float32",
            embedding_device="cpu",
            image_hash=image_hash,  # 🔧 旧架构使用
            cache_id=cache_id,  # 🔧 新架构使用
            compute_time=0.0,
            from_cache=False,
            error=True,  # 🔧 标记错误
            error_message=error_message,  # 🔧 错误信息
            vit_compute_start_time=time.time(),
            vit_compute_end_time=time.time(),
            # 🔑 方案 1: 返回输入 SHM 名称，用于 PP0 release
            input_shm_names=[
                request.pixel_values_shm_name,
                request.image_grid_thw_shm_name,
            ],
        )
        self._send_response(response)

    # ------------------------------------------------------------------
    # TP worker (kept as simple placeholder)
    # ------------------------------------------------------------------

    def _run_tp_worker(self):
        logger.info("[VIT Scheduler] TP worker rank=%d entering idle loop", self.tp_rank)
        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("[VIT Scheduler] TP worker interrupt")


# ---------------------------------------------------------------------------
# process entry point
# ---------------------------------------------------------------------------


def start_vit_scheduler(
    model_config,
    device: str = "cuda:0",
    zmq_port: int = 5555,
    batch_size: int = 4,
    batch_timeout_ms: float = 10.0,
    cache_size_mb: int = 1024,
    cache_rpc_port: Optional[int] = None,
    pipe_writer=None,
):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [VIT Scheduler] %(message)s")

    scheduler = VITScheduler(
        model_config=model_config,
        device=device,
        zmq_port=zmq_port,
        batch_size=batch_size,
        batch_timeout_ms=batch_timeout_ms,
        cache_size_mb=cache_size_mb,
        cache_rpc_port=cache_rpc_port,
    )

    if pipe_writer is not None:
        pipe_writer.send("ready")

    scheduler.run()
