"""
ViT Worker RPC Service for Phase 2 Worker Pool Architecture

This module implements the RPC service for ViT workers that can run in separate processes.
Supports both Data Parallel (DP) and Tensor Parallel (TP) modes.

Architecture:
- Each worker is an independent process with RPyC service
- DP: Multiple workers process different batches in parallel
- TP: Multiple ranks of the same worker process the same batch across GPUs
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import rpyc
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class VITWorkerService(rpyc.Service):
    """ViT Worker RPC 服务
    
    每个 Worker 是一个独立进程，通过 RPyC 提供服务。
    支持 TP (Tensor Parallel) 模式，多个 rank 协同处理同一批次。

    """
    def __init__(
        self,
        model_config,
        device: str,
        tp_rank: int,
        tp_size: int,
        worker_id: int,
        cache_rpc_host: str = "localhost",
        cache_rpc_port: int = 18888,
    ):
        """初始化 Worker 服务
        
        Args:
            model_config: 模型配置
            device: GPU 设备 (e.g., "cuda:0")
            tp_rank: TP rank (0 到 tp_size-1)
            tp_size: TP 大小 (1 表示单卡，2 表示双卡)
            worker_id: Worker ID (用于 DP)
            cache_rpc_host: CacheServer RPC 主机
            cache_rpc_port: CacheServer RPC 端口
        """
        super().__init__()
        self.model_config = model_config
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.worker_id = worker_id
        self.cache_rpc_host = cache_rpc_host
        self.cache_rpc_port = cache_rpc_port

        # 统计信息
        self.total_requests = 0
        self.total_compute_time = 0.0
        self.total_cache_hits = 0
        self.total_cache_misses = 0

        if torch.cuda.is_available() and self.device.startswith("cuda"):
            try:
                torch.cuda.set_device(torch.device(self.device))
            except Exception as exc:
                logger.warning(
                    f"[Worker {self.worker_id}] Failed to set CUDA device {self.device}: {exc}"
                )

        # 🔧 Phase 2.B: 健康检查相关
        self.last_ping_time = time.time()
        self.is_healthy = True

        # 🔧 缓存禁用开关 (参考 LightLLM)
        self.cache_enabled = os.environ.get("SGLANG_VIT_DISABLE_CACHE", "0") != "1"
        if not self.cache_enabled:
            logger.info(f"[Worker {self.worker_id}] ViT embedding cache is DISABLED")

        # 🔧 显存池管理 (对齐 LightLLM)
        memory_pool_gb = float(os.environ.get("SGLANG_VIT_MEMORY_POOL_GB", "0"))
        if memory_pool_gb > 0:
            from sglang.srt.managers.vit_memory_pool import VITMemoryPool
            self.memory_pool = VITMemoryPool(
                max_memory_gb=memory_pool_gb,
                enable_monitoring=True,
                monitoring_interval=30.0,
            )
            logger.info(
                f"[Worker {self.worker_id}] Memory pool enabled: {memory_pool_gb:.2f} GB"
            )
        else:
            self.memory_pool = None
            logger.info(f"[Worker {self.worker_id}] Memory pool disabled")

        # 🔧 限制单进程显存占用 (torch.cuda.set_per_process_memory_fraction)
        self.memory_fraction = float(os.environ.get("SGLANG_VIT_MEMORY_FRACTION", "0"))
        if self.memory_fraction > 0 and torch.cuda.is_available():
            try:
                fraction = min(max(self.memory_fraction, 0.0), 0.99)
                torch.cuda.set_per_process_memory_fraction(
                    fraction, device=torch.device(self.device)
                )
                logger.info(
                    f"[Worker {self.worker_id}] Set per-process memory fraction to {fraction:.3f} on {self.device}"
                )
            except Exception as e:
                logger.warning(
                    f"[Worker {self.worker_id}] Failed to set per-process memory fraction ({self.memory_fraction}): {e}"
                )

        # 🔧 Phase 2.C: TP 真正并行 - 初始化顺序调整
        # 1. 先初始化 NCCL (如果 tp_size > 1)，确保 parallel_state 正确设置
        if self.tp_size > 1:
            self._init_nccl()

        # 2. 再初始化模型 (所有 rank 都加载权重切片)
        self._init_model()

        # 3. 只有 rank0 连接 CacheServer (仅当缓存启用时)
        if self.tp_rank == 0 and self.cache_enabled:
            self._init_cache_client()
        else:
            self.cache_client = None
            if self.tp_rank == 0:
                logger.info(f"[Worker {self.worker_id}] Rank {self.tp_rank}: CacheServer NOT connected (cache disabled)")
            else:
                logger.info(f"[Worker {self.worker_id}] Rank {self.tp_rank}: skipping CacheServer connection")

        logger.info(
            f"[Worker {self.worker_id}] Initialized: tp_rank={self.tp_rank}, "
            f"tp_size={self.tp_size}, device={self.device}"
        )

    def _init_model(self):
        """初始化 ViT 模型

        🔧 Phase 2.C: TP 真正并行 - 所有 rank 都加载模型权重切片
        """
        from sglang.srt.managers.vit_scheduler import VITModelRunner

        self.model_runner = VITModelRunner(
            self.model_config, device=self.device, tp_size=self.tp_size
        )

        # 🔧 Phase 2.C: 加载模型权重 (TP 模式下会自动分片)
        logger.info(
            f"[Worker {self.worker_id}] Rank {self.tp_rank}: loading model weights..."
        )
        self.model_runner.load_model()
        logger.info(
            f"[Worker {self.worker_id}] Rank {self.tp_rank}: model loaded on {self.device}"
        )

    def _init_nccl(self):
        """初始化 NCCL (用于 TP)

        🔧 Phase 2.C: TP 真正并行 - 初始化 parallel_state
        """
        try:
            # 设置环境变量
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(29500 + self.worker_id)
            os.environ["RANK"] = str(self.tp_rank)
            os.environ["WORLD_SIZE"] = str(self.tp_size)

            # 初始化进程组
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    init_method=f"env://",
                    rank=self.tp_rank,
                    world_size=self.tp_size,
                )
                logger.info(
                    f"[Worker {self.worker_id}] NCCL initialized: rank={self.tp_rank}, "
                    f"world_size={self.tp_size}"
                )

            # 🔧 Phase 2.C: 初始化 parallel_state (关键！)
            # 这样 ColumnParallelLinear 和 RowParallelLinear 才能正确分片权重
            from sglang.srt.distributed.parallel_state import (
                init_world_group,
                model_parallel_is_initialized,
            )
            from sglang.srt.distributed import initialize_model_parallel
            import sglang.srt.distributed.parallel_state as parallel_state

            # 初始化 _WORLD group
            if parallel_state._WORLD is None:
                ranks = list(range(self.tp_size))
                parallel_state._WORLD = init_world_group(
                    ranks=ranks,
                    local_rank=self.tp_rank,
                    backend="nccl",
                )
                logger.info(f"[Worker {self.worker_id}] _WORLD group initialized")

            # 初始化 TP group
            if not model_parallel_is_initialized():
                initialize_model_parallel(
                    tensor_model_parallel_size=self.tp_size,
                    pipeline_model_parallel_size=1,
                )
                logger.info(
                    f"[Worker {self.worker_id}] Model parallel initialized: "
                    f"tp_size={self.tp_size}"
                )

            # 🔧 Phase 2.C: 验证初始化结果
            from sglang.srt.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            actual_tp_rank = get_tensor_model_parallel_rank()
            actual_tp_size = get_tensor_model_parallel_world_size()

            if actual_tp_rank != self.tp_rank:
                raise RuntimeError(
                    f"[Worker {self.worker_id}] TP rank mismatch: "
                    f"expected {self.tp_rank}, got {actual_tp_rank}"
                )

            if actual_tp_size != self.tp_size:
                raise RuntimeError(
                    f"[Worker {self.worker_id}] TP size mismatch: "
                    f"expected {self.tp_size}, got {actual_tp_size}"
                )

            logger.info(
                f"[Worker {self.worker_id}] TP verified: "
                f"rank={actual_tp_rank}, size={actual_tp_size}"
            )

            # ✅ NCCL Warmup broadcast (避免首次 broadcast 出错)
            logger.info(
                f"[Worker {self.worker_id}] Rank {self.tp_rank}: "
                f"Performing NCCL warmup broadcast..."
            )
            warmup_tensor = torch.zeros([1], dtype=torch.int32, device=self.device)
            dist.broadcast(warmup_tensor, src=0)
            logger.info(
                f"[Worker {self.worker_id}] Rank {self.tp_rank}: "
                f"NCCL warmup broadcast successful"
            )

        except Exception as e:
            logger.error(
                f"[Worker {self.worker_id}] Failed to initialize NCCL: {e}",
                exc_info=True,
            )
            raise

    def _init_cache_client(self):
        """连接 CacheServer"""
        try:
            self.cache_client = rpyc.connect(
                self.cache_rpc_host,
                self.cache_rpc_port,
                config={
                    "allow_public_attrs": True,
                    "allow_pickle": True,
                    "sync_request_timeout": 300,
                },
            )
            logger.info(
                f"[Worker {self.worker_id}] Connected to CacheServer at "
                f"{self.cache_rpc_host}:{self.cache_rpc_port}"
            )
        except Exception as e:
            logger.error(
                f"[Worker {self.worker_id}] Failed to connect to CacheServer: {e}",
                exc_info=True,
            )
            raise

    def exposed_forward(
        self,
        request_ids: List[str],
        pixel_values_shm_keys: List[str],
        pixel_values_shapes: List[Tuple[int, ...]],
        pixel_values_dtypes: List[str],
        image_grid_thw_shm_keys: List[str],
        image_grid_thw_shapes: List[Tuple[int, ...]],
        image_grid_thw_dtypes: List[str],
        content_hashes: List[int],
    ) -> List[Dict]:
        """执行 forward 推理（仅 Rank 0 暴露给 RPC）

        Rank 0:
            1. 读取输入 / 检查缓存
            2. 如果需要计算，通过 NCCL broadcast 通知其他 rank
            3. 负责写回缓存 / 返回结果

        Rank > 0:
            - 通过 _run_tp_worker() 等待 broadcast，无需调用该 RPC。
        """
        if self.tp_rank != 0:
            logger.warning(
                f"[Worker {self.worker_id}] Rank {self.tp_rank}: "
                "exposed_forward called on non-zero rank; ignoring"
            )
            return []

        start_time = time.time()
        results: List[Dict] = []

        try:
            (
                pixel_values_list,
                image_grid_thw_list,
                cache_results,
            ) = self._read_inputs_and_check_cache(
                pixel_values_shm_keys,
                pixel_values_shapes,
                pixel_values_dtypes,
                image_grid_thw_shm_keys,
                image_grid_thw_shapes,
                image_grid_thw_dtypes,
                content_hashes,
            )

            cache_hits: List[Tuple[int, Dict]] = []
            cache_misses: List[
                Tuple[int, torch.Tensor, torch.Tensor, int, Optional[int]]
            ] = []

            for i, cache_result in enumerate(cache_results):
                if cache_result and cache_result["from_cache"]:
                    cache_hits.append((i, cache_result))
                else:
                    cache_misses.append(
                        (
                            i,
                            pixel_values_list[i],
                            image_grid_thw_list[i],
                            content_hashes[i],
                            cache_result["cache_id"] if cache_result else None,
                        )
                    )

            for idx, cache_result in cache_hits:
                results.append(
                    {
                        "request_id": request_ids[idx],
                        "cache_id": cache_result["cache_id"],
                        "from_cache": True,
                        "compute_time": 0.0,
                        "error": False,
                        "error_message": "",
                        "input_shm_names": [
                            pixel_values_shm_keys[idx],
                            image_grid_thw_shm_keys[idx],
                        ],
                    }
                )
                self.total_cache_hits += 1

            if not cache_misses:
                if self.tp_size > 1:
                    try:
                        self._broadcast_inputs([], [], shutdown=False, cache_hit=True)
                    except Exception as e:
                        logger.error(
                            f"[Worker {self.worker_id}] Rank 0: failed to broadcast cache-hit signal: {e}",
                            exc_info=True,
                        )
                self.total_requests += len(request_ids)
                self.total_compute_time += time.time() - start_time
                results.sort(key=lambda x: request_ids.index(x["request_id"]))
                return results

            miss_results = self._compute_embeddings(
                cache_misses,
                request_ids,
                pixel_values_shm_keys=pixel_values_shm_keys,
                image_grid_thw_shm_keys=image_grid_thw_shm_keys,
            )
            results.extend(miss_results)
            self.total_cache_misses += len(cache_misses)

            self.total_requests += len(request_ids)
            self.total_compute_time += time.time() - start_time
            results.sort(key=lambda x: request_ids.index(x["request_id"]))
            return results

        except Exception as e:
            logger.error(
                f"[Worker {self.worker_id}] Rank 0: Forward failed: {e}",
                exc_info=True,
            )
            if self.tp_size > 1:
                try:
                    self._broadcast_inputs([], [], shutdown=True, cache_hit=False)
                except Exception as broadcast_err:
                    logger.error(
                        f"[Worker {self.worker_id}] Rank 0: Failed to send shutdown signal: {broadcast_err}"
                    )

            error_results = []
            for i, req_id in enumerate(request_ids):
                error_results.append(
                    {
                        "request_id": req_id,
                        "cache_id": None,
                        "from_cache": False,
                        "compute_time": 0.0,
                        "error": True,
                        "error_message": str(e),
                        "input_shm_names": [
                            pixel_values_shm_keys[i],
                            image_grid_thw_shm_keys[i],
                        ],
                    }
                )
            return error_results

    def exposed_ping(self) -> Dict:
        """健康检查 (心跳)

        🔧 Phase 2.B: Worker 健康检查

        Returns:
            Dict: {"status": "ok", "timestamp": float, "worker_id": int, "tp_rank": int}
        """
        self.last_ping_time = time.time()
        return {
            "status": "ok",
            "timestamp": self.last_ping_time,
            "worker_id": self.worker_id,
            "tp_rank": self.tp_rank,
        }

    def exposed_get_health(self) -> Dict:
        """获取健康状态

        🔧 Phase 2.B: Worker 健康状态

        Returns:
            Dict: 健康状态信息
        """
        return {
            "is_healthy": self.is_healthy,
            "last_ping_time": self.last_ping_time,
            "worker_id": self.worker_id,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "total_requests": self.total_requests,
            "total_cache_hits": self.total_cache_hits,
            "total_cache_misses": self.total_cache_misses,
            "total_compute_time": self.total_compute_time,
        }

    def exposed_get_memory_stats(self) -> Dict[str, float]:
        """Return current GPU memory statistics for watchdog/high-water checks."""
        stats = {"total": 0.0, "free": 0.0, "used": 0.0}
        try:
            if torch.cuda.is_available():
                device = torch.device(self.device)
                torch.cuda.synchronize(device)
                free, total = torch.cuda.mem_get_info(device=device)
                used = total - free
                stats.update(
                    {
                        "total": float(total),
                        "free": float(free),
                        "used": float(used),
                    }
                )
        except Exception as exc:
            logger.warning(
                f"[Worker {self.worker_id}] Failed to query GPU memory stats: {exc}"
            )
            stats["error"] = str(exc)
        return stats

    def _read_inputs_and_check_cache(
        self,
        pixel_values_shm_keys: List[str],
        pixel_values_shapes: List[Tuple[int, ...]],
        pixel_values_dtypes: List[str],
        image_grid_thw_shm_keys: List[str],
        image_grid_thw_shapes: List[Tuple[int, ...]],
        image_grid_thw_dtypes: List[str],
        content_hashes: List[int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Optional[Dict]]]:
        """读取输入并检查缓存 (仅 TP Rank 0)

        🔧 修复: 使用 read_tensor_from_shared_memory 读取 pixel/grid，
        避免 read_embedding_from_shm 添加 vit_embed_ 前缀导致找不到 SHM
        """
        from sglang.srt.managers.vit_shm_utils import read_tensor_from_shared_memory

        pixel_values_list = []
        image_grid_thw_list = []
        cache_results = []

        for pv_key, pv_shape, pv_dtype, grid_key, grid_shape, grid_dtype, content_hash in zip(
            pixel_values_shm_keys,
            pixel_values_shapes,
            pixel_values_dtypes,
            image_grid_thw_shm_keys,
            image_grid_thw_shapes,
            image_grid_thw_dtypes,
            content_hashes,
        ):
            # 🔧 读取 pixel_values (使用 read_tensor_from_shared_memory)
            pixel_values = read_tensor_from_shared_memory(
                pv_key, pv_shape, pv_dtype
            )

            # 🔧 读取 image_grid_thw (使用 read_tensor_from_shared_memory)
            image_grid_thw = read_tensor_from_shared_memory(
                grid_key, grid_shape, grid_dtype
            )

            pixel_values_list.append(pixel_values.to(self.device))
            image_grid_thw_list.append(image_grid_thw.to(self.device))

            # 🔧 检查缓存并分配 cache_id (只分配一次)
            size_bytes = pixel_values.nelement() * pixel_values.element_size()
            cache_result = self._check_cache(content_hash, size_bytes)
            cache_results.append(cache_result)

        return pixel_values_list, image_grid_thw_list, cache_results

    def _check_cache(
        self, content_hash: int, size_bytes: int
    ) -> Optional[Dict]:
        """检查缓存是否命中

        🔧 缓存禁用时，总是返回 None (缓存未命中)
        """
        # 🔧 缓存禁用时，跳过缓存检查
        if not self.cache_enabled or self.cache_client is None:
            return None

        try:
            result = self.cache_client.root.alloc(content_hash, size_bytes)
            if result is None:
                return None

            cache_id, is_new = result
            if not is_new:
                # 缓存命中
                return {"cache_id": cache_id, "from_cache": True}
            else:
                # 缓存未命中，需要计算
                return {"cache_id": cache_id, "from_cache": False}
        except Exception as e:
            logger.error(
                f"[Worker {self.worker_id}] Cache check failed: {e}", exc_info=True
            )
            return None

    def _broadcast_inputs(
        self,
        pixel_values_list: List[torch.Tensor],
        image_grid_thw_list: List[torch.Tensor],
        shutdown: bool = False,
        cache_hit: bool = False,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        bool,
        bool,
    ]:
        """通过 NCCL broadcast 输入张量给其他 TP ranks。

        Args:
            pixel_values_list: Rank 0 的输入张量列表
            image_grid_thw_list: Rank 0 的 grid 列表
            shutdown: 是否发送退出信号
            cache_hit: 是否发送缓存命中信号

        Returns:
            Tuple[pixel_values_list, image_grid_thw_list, shutdown_flag, cache_hit_flag]
            对于非 0 Rank，会返回通过 broadcast 接收到的新张量。
        """
        if self.tp_size <= 1:
            return pixel_values_list, image_grid_thw_list, shutdown, cache_hit

        try:
            batch_size = len(pixel_values_list)
            if self.tp_rank == 0:
                metadata = {
                    "batch_size": batch_size,
                    "pixel_values_shapes": [list(pv.shape) for pv in pixel_values_list],
                    "pixel_values_dtypes": [
                        str(pv.dtype).replace("torch.", "") for pv in pixel_values_list
                    ],
                    "image_grid_thw_shapes": [
                        list(grid.shape) for grid in image_grid_thw_list
                    ],
                    "image_grid_thw_dtypes": [
                        str(grid.dtype).replace("torch.", "") for grid in image_grid_thw_list
                    ],
                    "shutdown": bool(shutdown),
                    "cache_hit": bool(cache_hit),
                }
                metadata_list = [metadata]
            else:
                metadata_list = [None]

            dist.broadcast_object_list(metadata_list, src=0)
            metadata = metadata_list[0]

            shutdown_flag = bool(metadata.get("shutdown", False))
            cache_hit_flag = bool(metadata.get("cache_hit", False))
            batch_size = int(metadata.get("batch_size", 0))

            if self.tp_rank != 0:
                pixel_values_list = []
                image_grid_thw_list = []
                for i in range(batch_size):
                    pv_shape = tuple(metadata["pixel_values_shapes"][i])
                    pv_dtype = getattr(torch, metadata["pixel_values_dtypes"][i])
                    pixel_values = torch.empty(pv_shape, dtype=pv_dtype, device=self.device)
                    pixel_values_list.append(pixel_values)

                    grid_shape = tuple(metadata["image_grid_thw_shapes"][i])
                    grid_dtype = getattr(torch, metadata["image_grid_thw_dtypes"][i])
                    image_grid_thw = torch.empty(
                        grid_shape, dtype=grid_dtype, device=self.device
                    )
                    image_grid_thw_list.append(image_grid_thw)

            for i in range(batch_size):
                dist.broadcast(pixel_values_list[i], src=0)
                dist.broadcast(image_grid_thw_list[i], src=0)

            return pixel_values_list, image_grid_thw_list, shutdown_flag, cache_hit_flag
        except Exception as e:
            logger.error(
                f"[Worker {self.worker_id}] Rank {self.tp_rank}: TP broadcast failed: {e}",
                exc_info=True,
            )
            raise

    def _run_tp_worker(self):
        """Rank > 0 worker loop: 等待 Rank 0 的 NCCL broadcast 并参与计算。"""
        logger.info(
            f"[Worker {self.worker_id}] Rank {self.tp_rank}: TP worker loop started (device={self.device})"
        )
        try:
            while True:
                try:
                    (
                        pixel_values_list,
                        image_grid_thw_list,
                        shutdown_flag,
                        cache_hit_flag,
                    ) = self._broadcast_inputs([], [], shutdown=False, cache_hit=False)
                except Exception as e:
                    logger.error(
                        f"[Worker {self.worker_id}] Rank {self.tp_rank}: "
                        f"NCCL broadcast failed: {e}",
                        exc_info=True,
                    )
                    time.sleep(0.001)
                    continue

                if shutdown_flag:
                    logger.info(
                        f"[Worker {self.worker_id}] Rank {self.tp_rank}: received shutdown signal, exiting"
                    )
                    break

                if cache_hit_flag:
                    logger.debug(
                        f"[Worker {self.worker_id}] Rank {self.tp_rank}: received cache-hit signal, skipping compute"
                    )
                    continue

                if not pixel_values_list:
                    logger.debug(
                        f"[Worker {self.worker_id}] Rank {self.tp_rank}: received empty payload, continue"
                    )
                    continue

                batch_size = len(pixel_values_list)
                logger.info(
                    f"[Worker {self.worker_id}] Rank {self.tp_rank}: "
                    f"received batch_size={batch_size} from NCCL broadcast"
                )
                try:
                    start = time.time()
                    _ = self.model_runner.compute_batch(
                        pixel_values_list, image_grid_thw_list
                    )
                    logger.info(
                        f"[Worker {self.worker_id}] Rank {self.tp_rank}: "
                        f"computation finished in {(time.time() - start) * 1000:.1f} ms"
                    )
                except Exception as compute_err:
                    logger.error(
                        f"[Worker {self.worker_id}] Rank {self.tp_rank}: "
                        f"error during compute: {compute_err}",
                        exc_info=True,
                    )
        except KeyboardInterrupt:
            logger.info(
                f"[Worker {self.worker_id}] Rank {self.tp_rank}: received interrupt, exiting loop"
            )
        except Exception as e:
            logger.error(
                f"[Worker {self.worker_id}] Rank {self.tp_rank}: error in worker loop: {e}",
                exc_info=True,
            )

    def _compute_embeddings(
        self,
        cache_misses: List[Tuple],
        request_ids: List[str],
        pixel_values_shm_keys: List[str] = None,
        image_grid_thw_shm_keys: List[str] = None,
    ) -> List[Dict]:
        """计算 embeddings (所有 TP rank 都参与)

        🔧 Phase 2.C: rank0 通过 NCCL broadcast (_broadcast_inputs) 通知其他 rank 参与计算
        🔧 修复: 所有 rank 都执行 forward，只有 rank0 写缓存和返回结果
        🔧 修复: 复用 _check_cache 返回的 cache_id，避免重复分配
        🔧 显存池: 估算显存需求，不满足则拒绝请求 (backpressure)
        🔑 方案 1: 接收 SHM 名称列表，用于返回 input_shm_names
        """
        from sglang.srt.managers.vit_shm_utils import write_embedding_to_shm_raw

        results = []

        # 🔧 Phase 2.C: 准备批量数据
        pixel_values_list = []
        image_grid_thw_list = []
        for idx, pixel_values, image_grid_thw, content_hash, cache_id in cache_misses:
            pixel_values_list.append(pixel_values)
            image_grid_thw_list.append(image_grid_thw)

        if not pixel_values_list:
            return results

        sample_ids = [request_ids[idx] for idx, *_ in cache_misses][:3]
        more = "" if len(cache_misses) <= 3 else f" (+{len(cache_misses) - 3} more)"
        logger.info(
            f"[Worker {self.worker_id}] Rank {self.tp_rank}: computing batch size={len(cache_misses)} "
            f"request_ids={sample_ids}{more}"
        )

        # 🔧 Phase 0: 显存池估算与监控
        estimated_memory = 0
        gpu_memory_before = 0
        predicted_oom = False
        available_mb = 0.0
        safety_ratio = float(os.environ.get("SGLANG_VIT_MEMORY_SAFETY_RATIO", "0.9"))

        if self.tp_rank == 0:
            from sglang.srt.managers.vit_memory_pool import estimate_batch_memory
            import torch

            # 记录 forward 前的显存使用
            if torch.cuda.is_available():
                gpu_memory_before = torch.cuda.memory_allocated(self.device) / 1024**2  # MB

            estimated_memory = estimate_batch_memory(
                pixel_values_list=pixel_values_list,
                image_grid_list=image_grid_thw_list,
                embedding_dim=3584,  # Qwen2.5-VL
                dtype_size=2,  # fp16
                overhead_factor=1.5,
            )

            # 🔧 预估显存不足 -> 直接拒绝，避免触发 CUDA OOM
            if torch.cuda.is_available():
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(
                        device=torch.device(self.device)
                    )
                    available_bytes = free_bytes * max(min(safety_ratio, 1.0), 0.1)
                    available_mb = available_bytes / 1024**2
                    if estimated_memory > available_bytes:
                        predicted_oom = True
                except Exception as e:
                    logger.warning(
                        f"[Worker {self.worker_id}] Failed to query GPU free memory before forward: {e}"
                    )

            if predicted_oom:
                logger.warning(
                    f"[Worker {self.worker_id}] ❌ OOM predicted: batch size={len(cache_misses)}, "
                    f"estimated={estimated_memory / 1024**2:.2f} MB, "
                    f"available~={available_mb:.2f} MB (safety_ratio={safety_ratio})"
                )
                for idx, *_ in cache_misses:
                    results.append(
                        {
                            "request_id": request_ids[idx],
                            "error": True,
                            "error_message": "OOM predicted on ViT worker; please reduce batch size",
                            "cache_id": None,
                        }
                    )
                return results

        if self.memory_pool is not None and self.tp_rank == 0:
            if not self.memory_pool.can_allocate(estimated_memory):
                # 显存池满，拒绝请求
                logger.warning(
                    f"[Worker {self.worker_id}] ❌ Memory pool full, rejecting batch size={len(cache_misses)}, "
                    f"estimated_memory={estimated_memory / 1024**2:.2f} MB, "
                    f"pool_usage={self.memory_pool.current_usage / 1024**2:.2f}/{self.memory_pool.max_memory_bytes / 1024**2:.2f} MB, "
                    f"gpu_allocated={gpu_memory_before:.2f} MB"
                )
                # 返回错误给所有请求
                for idx, *_ in cache_misses:
                    results.append({
                        "request_id": request_ids[idx],
                        "error": True,
                        "error_message": "Memory pool full, please retry later",
                        "cache_id": None,
                    })
                return results

            # 分配显存
            if not self.memory_pool.allocate(estimated_memory):
                # 分配失败（理论上不应该发生，因为 can_allocate 已经检查过）
                logger.error(
                    f"[Worker {self.worker_id}] ❌ Memory allocation failed, batch size={len(cache_misses)}"
                )
                for idx, *_ in cache_misses:
                    results.append({
                        "request_id": request_ids[idx],
                        "error": True,
                        "error_message": "Memory allocation failed",
                        "cache_id": None,
                    })
                return results

            logger.info(
                f"[Worker {self.worker_id}] 📊 Memory allocated: estimated={estimated_memory / 1024**2:.2f} MB, "
                f"pool_usage={self.memory_pool.current_usage / 1024**2:.2f}/{self.memory_pool.max_memory_bytes / 1024**2:.2f} MB, "
                f"gpu_before={gpu_memory_before:.2f} MB"
            )

        if self.tp_size > 1:
            pixel_values_list, image_grid_thw_list, shutdown_flag, cache_hit_flag = (
                self._broadcast_inputs(
                    pixel_values_list,
                    image_grid_thw_list,
                    shutdown=False,
                    cache_hit=False,
                )
            )
            if shutdown_flag:
                logger.info(
                    f"[Worker {self.worker_id}] Rank {self.tp_rank}: "
                    "received shutdown signal during compute"
                )
                return results
            if cache_hit_flag:
                logger.debug(
                    f"[Worker {self.worker_id}] Rank {self.tp_rank}: "
                    "received cache-hit signal, skipping compute"
                )
                return results

        # ✅ 在 forward 前清理 CUDA 缓存，减少内存碎片化
        if self.tp_rank == 0:
            torch.cuda.empty_cache()
            logger.debug(
                f"[Worker {self.worker_id}] Rank {self.tp_rank}: CUDA cache cleared before forward"
            )

        # 🔧 所有 rank 都执行 forward (TP 模式下需要同步)
        start_time = time.time()
        embeddings = self.model_runner.compute_batch(
            pixel_values_list, image_grid_thw_list
        )
        compute_time = time.time() - start_time
        logger.info(
            f"[Worker {self.worker_id}] Rank {self.tp_rank}: forward finished in {compute_time*1000:.1f} ms"
        )

        # ✅ 在 forward 后清理 CUDA 缓存，释放未使用的内存
        if self.tp_rank == 0:
            torch.cuda.empty_cache()
            logger.debug(
                f"[Worker {self.worker_id}] Rank {self.tp_rank}: CUDA cache cleared after forward"
            )

        # 🔧 只有 rank0 写缓存和返回结果
        if self.tp_rank != 0:
            return []

        # rank0: 处理结果
        for i, (idx, pixel_values, image_grid_thw, content_hash, cache_id) in enumerate(cache_misses):
            try:
                embedding = embeddings[i]

                # 🔧 缓存禁用时，跳过缓存写入
                if self.cache_enabled and self.cache_client is not None:
                    # 使用已分配的 cache_id，不再重复分配
                    if cache_id is None:
                        # 如果没有 cache_id（缓存检查失败），重新分配
                        size_bytes = embedding.nelement() * embedding.element_size()
                        result = self.cache_client.root.alloc(content_hash, size_bytes)
                        if result is None:
                            raise RuntimeError("Failed to allocate cache")
                        cache_id, is_new = result
                    else:
                        # 复用已分配的 cache_id
                        is_new = True  # 已经在 _check_cache 中分配，需要写入

                    # 写入缓存 (如果是新分配)
                    if is_new:
                        shm_key = self.cache_client.root.get_shm_key(cache_id)
                        if shm_key:
                            from sglang.srt.managers.vit_shm_utils import cleanup_shared_memory

                            # 确保旧的缓存块被清理，避免 FileExistsError
                            cleanup_shared_memory(shm_key)

                            # 🔧 关键修复: 保持原始 dtype (bf16/fp8)，不强制转换
                            # vit_shm_utils._tensor_to_bytes() 会自动搬到 CPU 并保留 dtype
                            # 参考: sglang/python/sglang/srt/managers/vit_shm_utils.py L37-45
                            success = write_embedding_to_shm_raw(shm_key, embedding)
                            if not success:
                                logger.error(
                                    f"[Worker {self.worker_id}] Failed to write embedding to cache"
                                )
                                # 🔧 写入失败，释放 cache_id
                                try:
                                    self.cache_client.root.release(cache_id)
                                except Exception as release_err:
                                    logger.error(f"[Worker {self.worker_id}] Failed to release cache_id: {release_err}")
                else:
                    # 🔧 缓存禁用，不分配 cache_id，直接通过请求级 SHM 返回 embedding
                    cache_id = None

                    # 写入请求级 SHM
                    from sglang.srt.managers.vit_shm_utils import (
                        cleanup_embedding_shm,
                        write_embedding_to_shm,
                    )

                    request_id = request_ids[idx]
                    cleanup_embedding_shm(request_id)

                    # 🔧 关键修复: 保持原始 dtype (bf16/fp8)，不强制转换
                    # vit_shm_utils._tensor_to_bytes() 会自动搬到 CPU 并保留 dtype
                    success = write_embedding_to_shm(request_id, embedding)
                    if not success:
                        logger.error(
                            f"[Worker {self.worker_id}] Failed to write embedding to request SHM: {request_id}"
                        )
                        # 写入失败，返回错误
                        results.append(
                            {
                                "request_id": request_id,
                                "cache_id": None,
                                "from_cache": False,
                                "compute_time": 0.0,
                                "error": True,
                                "error_message": "Failed to write embedding to request SHM",
                            }
                        )
                        continue

                results.append(
                    {
                        "request_id": request_ids[idx],  # 🔧 使用真实 request_id
                        "cache_id": cache_id,
                        "from_cache": False,
                        "compute_time": compute_time / len(cache_misses),  # 平均时间
                        "error": False,
                        "error_message": "",
                        # 🔑 方案 1: 返回输入 SHM 名称，用于 PP0 release
                        "input_shm_names": [
                            pixel_values_shm_keys[idx] if pixel_values_shm_keys else None,
                            image_grid_thw_shm_keys[idx] if image_grid_thw_shm_keys else None,
                        ] if pixel_values_shm_keys and image_grid_thw_shm_keys else None,
                    }
                )

            except Exception as e:
                logger.error(
                    f"[Worker {self.worker_id}] Compute failed: {e}", exc_info=True
                )
                # 释放 cache_id
                if cache_id is not None:
                    try:
                        self.cache_client.root.release(cache_id)
                    except Exception as release_err:
                        logger.error(f"[Worker {self.worker_id}] Failed to release cache_id: {release_err}")

                results.append(
                    {
                        "request_id": request_ids[idx],  # 🔧 使用真实 request_id
                        "cache_id": None,
                        "from_cache": False,
                        "compute_time": 0.0,
                        "error": True,
                        "error_message": str(e),
                        # 🔑 方案 1: 即使失败也要返回 input_shm_names
                        "input_shm_names": [
                            pixel_values_shm_keys[idx] if pixel_values_shm_keys else None,
                            image_grid_thw_shm_keys[idx] if image_grid_thw_shm_keys else None,
                        ] if pixel_values_shm_keys and image_grid_thw_shm_keys else None,
                    }
                )

        # 🔧 Phase 0: 显存池释放与监控
        if self.memory_pool is not None and self.tp_rank == 0 and estimated_memory > 0:
            import torch

            # 记录 forward 后的显存使用
            gpu_memory_after = 0
            if torch.cuda.is_available():
                gpu_memory_after = torch.cuda.memory_allocated(self.device) / 1024**2  # MB

            self.memory_pool.release(estimated_memory)

            # 计算实际显存增量
            actual_memory_delta = gpu_memory_after - gpu_memory_before

            logger.info(
                f"[Worker {self.worker_id}] 📊 Memory released: estimated={estimated_memory / 1024**2:.2f} MB, "
                f"actual_delta={actual_memory_delta:.2f} MB, "
                f"gpu_after={gpu_memory_after:.2f} MB, "
                f"pool_usage={self.memory_pool.current_usage / 1024**2:.2f}/{self.memory_pool.max_memory_bytes / 1024**2:.2f} MB, "
                f"pool_peak={self.memory_pool.peak_usage / 1024**2:.2f} MB"
            )

        return results

    def exposed_health_check(self) -> bool:
        """健康检查"""
        return True

    def exposed_get_stats(self) -> Dict:
        """获取统计信息"""
        avg_compute_time = (
            self.total_compute_time / self.total_requests
            if self.total_requests > 0
            else 0.0
        )
        return {
            "worker_id": self.worker_id,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "total_requests": self.total_requests,
            "total_compute_time": self.total_compute_time,
            "avg_compute_time": avg_compute_time,
            "total_cache_hits": self.total_cache_hits,
            "total_cache_misses": self.total_cache_misses,
            "cache_hit_rate": (
                self.total_cache_hits / (self.total_cache_hits + self.total_cache_misses)
                if (self.total_cache_hits + self.total_cache_misses) > 0
                else 0.0
            ),
        }

    # ✅ 删除 _run_tp_worker() 方法（-72 行）
    # Rank>0 不再需要独立的工作循环
    # 它们会在 exposed_forward() 中自动参与计算

    def cleanup(self):
        """清理资源

        🔧 Phase 2.B: 确保正确释放 NCCL / GPU 资源
        """
        try:
            logger.info(f"[Worker {self.worker_id}] Rank {self.tp_rank} cleaning up...")

            if self.tp_rank == 0 and self.tp_size > 1:
                try:
                    self._broadcast_inputs([], [], shutdown=True, cache_hit=False)
                    logger.info(
                        f"[Worker {self.worker_id}] Rank 0: sent shutdown signal to other ranks"
                    )
                except Exception as e:
                    logger.error(
                        f"[Worker {self.worker_id}] Rank 0: failed to send shutdown signal: {e}",
                        exc_info=True,
                    )

            # 关闭 CacheServer 连接
            if hasattr(self, "cache_client") and self.cache_client is not None:
                try:
                    self.cache_client.close()
                    logger.info(f"[Worker {self.worker_id}] Cache client closed")
                except Exception as e:
                    logger.error(f"[Worker {self.worker_id}] Failed to close cache client: {e}")

            # 销毁 NCCL 进程组
            if self.tp_size > 1 and dist.is_initialized():
                try:
                    dist.destroy_process_group()
                    logger.info(f"[Worker {self.worker_id}] Rank {self.tp_rank} NCCL destroyed")
                except Exception as e:
                    logger.error(f"[Worker {self.worker_id}] Failed to destroy NCCL: {e}")

            # 清理 GPU 缓存
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.info(f"[Worker {self.worker_id}] GPU cache cleared")
                except Exception as e:
                    logger.error(f"[Worker {self.worker_id}] Failed to clear GPU cache: {e}")

            logger.info(f"[Worker {self.worker_id}] Rank {self.tp_rank} cleanup complete")

        except Exception as e:
            logger.error(f"[Worker {self.worker_id}] Cleanup failed: {e}", exc_info=True)


def start_vit_worker_process(
    worker_id: int,
    model_config,
    tp_rank: int,
    tp_size: int,
    rpc_port: int,
    cache_rpc_host: str = "localhost",
    cache_rpc_port: int = 18888,
):
    """启动 Worker 进程

    🔧 修复: 只有 rank0 启动 RPC server，避免端口冲突
    其他 rank 只创建 service 并参与计算

    Args:
        worker_id: Worker ID
        model_config: 模型配置
        tp_rank: TP rank
        tp_size: TP 大小
        rpc_port: Worker RPC 端口 (只有 rank0 使用)
        cache_rpc_host: CacheServer RPC 主机
        cache_rpc_port: CacheServer RPC 端口
    """
    # 设置 GPU 设备
    base_device_index = int(os.environ.get("SGLANG_VIT_TP_BASE_DEVICE", "0"))
    device_index = base_device_index + tp_rank
    device = f"cuda:{device_index}"

    # 创建 Worker 服务
    service = VITWorkerService(
        model_config=model_config,
        device=device,
        tp_rank=tp_rank,
        tp_size=tp_size,
        worker_id=worker_id,
        cache_rpc_host=cache_rpc_host,
        cache_rpc_port=cache_rpc_port,
    )

    from sglang.srt.utils.graceful_utils import graceful_registry
    graceful_registry(f"VITWorker_{worker_id}_Rank_{tp_rank}")

    if tp_rank == 0:
        from rpyc.utils.server import ThreadedServer

        server = ThreadedServer(
            service,
            port=rpc_port,
            protocol_config={
                "allow_public_attrs": True,
                "allow_pickle": True,
                "sync_request_timeout": 300,
            },
        )

        logger.info(
            f"[Worker {worker_id}] Rank {tp_rank}: starting RPC server on port {rpc_port}, "
            f"tp_size={tp_size}, device={device}"
        )

        try:
            server.start()
        except KeyboardInterrupt:
            logger.info(f"[Worker {worker_id}] Rank {tp_rank}: shutting down...")
        except Exception as e:
            logger.error(
                f"[Worker {worker_id}] Rank {tp_rank}: failed to start: {e}",
                exc_info=True,
            )
            raise
        finally:
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                    logger.info(
                        f"[Worker {worker_id}] Rank {tp_rank}: destroyed process group"
                    )
                except Exception as cleanup_err:
                    logger.error(
                        f"[Worker {worker_id}] Rank {tp_rank}: "
                        f"failed to destroy process group: {cleanup_err}"
                    )
            service.cleanup()
    else:
        logger.info(
            f"[Worker {worker_id}] Rank {tp_rank}: entering TP worker loop (device={device})"
        )
        try:
            service._run_tp_worker()
        except KeyboardInterrupt:
            logger.info(
                f"[Worker {worker_id}] Rank {tp_rank}: received interrupt, shutting down..."
            )
        except Exception as e:
            logger.error(
                f"[Worker {worker_id}] Rank {tp_rank}: worker loop failed: {e}",
                exc_info=True,
            )
            raise
        finally:
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                    logger.info(
                        f"[Worker {worker_id}] Rank {tp_rank}: destroyed process group"
                    )
                except Exception as cleanup_err:
                    logger.error(
                        f"[Worker {worker_id}] Rank {tp_rank}: "
                        f"failed to destroy process group: {cleanup_err}"
                    )
            service.cleanup()
