"""
VIT Scheduler - 独立的 ViT 计算调度器

架构设计:
- 独立进程运行，专门负责 ViT 计算
- 通过 ZMQ 接收来自主 Scheduler 的请求
- 通过共享内存传递 pixel_values 和 embedding
- 支持批量计算和缓存优化
"""

import os
import time
import zmq
import pickle
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import threading

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


@dataclass
class VITRequest:
    """VIT 计算请求"""
    request_id: str
    pixel_values_shm_name: str  # 共享内存名称
    pixel_values_shape: Tuple[int, ...]
    pixel_values_dtype: str
    image_grid_thw_shm_name: str
    image_grid_thw_shape: Tuple[int, ...]
    image_grid_thw_dtype: str
    hash_val: Optional[int] = None


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
        image_hash: 图片的 hash 值（用于 cache 管理）
        compute_time: 计算耗时（秒）
        from_cache: 是否从 cache 中获取
    """
    request_id: str
    # 🔧 CUDA IPC: 使用 IPC handle 代替 CPU 共享内存
    embedding_ipc_handle: Tuple[List[int], int]  # (handle, offset)
    embedding_shape: Tuple[int, ...]
    embedding_dtype: str
    embedding_device: str  # 新增: 设备信息
    image_hash: int  # 新增: 用于 cache 管理
    # 保留原有字段
    compute_time: float
    from_cache: bool


class VITModelRunner:
    """VIT 模型运行器（支持 TP）"""

    def __init__(self, model_config, device: str = "cuda:0", tp_size: int = 1):
        self.model_config = model_config
        self.device = device
        self.vit_model = None
        self.tp_size = tp_size

        # 初始化 load_config（复用官方加载逻辑需要）
        from sglang.srt.configs.load_config import LoadConfig
        self.load_config = LoadConfig()
        
    def load_model(self):
        """加载 ViT 模型（支持 TP）

        注意: 当 TP > 1 时，VIT 模型的权重会自动按 TP rank 分片。
        这是通过 SGLang 的 ColumnParallelLinear 和 RowParallelLinear 层实现的。
        """
        model_type = self.model_config.hf_config.model_type

        logger.info(f"[VIT Runner] Loading ViT model type: {model_type}")
        if self.tp_size > 1:
            logger.info(f"[VIT Runner] TP enabled: size={self.tp_size}")

        # 🔧 VIT 模型不使用量化（FP8 量化会导致 NaN）
        # 即使主模型使用了量化，VIT 模型也应该使用 float32
        logger.info(f"[VIT Runner] VIT model will use float32 (no quantization)")

        # 创建 ViT 模型
        if model_type == "qwen2_5_vl":
            from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer
            self.vit_model = Qwen2_5_VisionTransformer(
                self.model_config.hf_config.vision_config,
                norm_eps=getattr(self.model_config.hf_config, "rms_norm_eps", 1e-6),
                quant_config=None,  # 🔧 不使用量化
            )
        elif model_type == "qwen2_vl":
            from sglang.srt.models.qwen2_vl import Qwen2VisionTransformer
            self.vit_model = Qwen2VisionTransformer(
                self.model_config.hf_config.vision_config,
                norm_eps=getattr(self.model_config.hf_config, "rms_norm_eps", 1e-6),
                quant_config=None,  # 🔧 不使用量化
            )
        else:
            raise ValueError(f"Unsupported model type for VIT Scheduler: {model_type}")

        # 加载权重（复用官方加载逻辑，仅过滤 visual.* 权重）
        logger.info(f"[VIT Runner] Loading ViT weights (official loader, visual.* only) from {self.model_config.model_path}")
        from sglang.srt.model_loader.loader import get_model_loader, DefaultModelLoader
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        # 🔧 关键修复：不创建完整模型（会触发 PP group 初始化），直接用 loader 获取权重
        loader = get_model_loader(self.load_config)

        # 获取所有权重的迭代器（不需要真实模型对象，只需要 model_config）
        all_weights_iter = loader._get_weights_iterator(
            DefaultModelLoader.Source(
                model_or_path=self.model_config.model_path,
                revision=getattr(self.model_config, 'revision', None),
                prefix="",
                fall_back_to_pt=True
            )
        )

        # 收集 visual.* 权重，应用官方命名映射逻辑
        params_dict = dict(self.vit_model.named_parameters())
        loaded_count = 0

        for name, tensor in all_weights_iter:
            if not name.startswith("visual."):
                continue

            # 去掉 "visual." 前缀
            vit_name = name[len("visual."):]

            # 🔧 应用官方命名映射：attn.qkv. → attn.qkv_proj.
            if ".attn.qkv." in vit_name:
                vit_name = vit_name.replace(".attn.qkv.", ".attn.qkv_proj.")

            # 🔧 处理堆叠参数（qkv_proj）
            # Qwen2-VL 的视觉模块使用 qkv_proj 融合 q/k/v
            if vit_name in params_dict:
                param = params_dict[vit_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                # 检查是否是 qkv_proj 的分片加载
                if "qkv_proj" in vit_name:
                    # 需要判断当前是 q/k/v 中的哪一个
                    if ".q_proj." in name:
                        loaded_shard_id = "q"
                    elif ".k_proj." in name:
                        loaded_shard_id = "k"
                    elif ".v_proj." in name:
                        loaded_shard_id = "v"
                    else:
                        # 已经是 qkv_proj，直接加载
                        loaded_shard_id = None

                    if loaded_shard_id is not None:
                        # 使用 loaded_shard_id 参数
                        from inspect import signature
                        sig = signature(weight_loader)
                        if "loaded_shard_id" in sig.parameters:
                            weight_loader(param, tensor, loaded_shard_id=loaded_shard_id)
                        else:
                            weight_loader(param, tensor)
                    else:
                        weight_loader(param, tensor)
                else:
                    weight_loader(param, tensor)

                loaded_count += 1
            else:
                # 参数不存在，可能是 q_proj/k_proj/v_proj 需要融合到 qkv_proj
                # 尝试找到对应的 qkv_proj 参数
                if ".q_proj." in name or ".k_proj." in name or ".v_proj." in name:
                    # 替换为 qkv_proj
                    qkv_name = vit_name.replace(".q_proj.", ".qkv_proj.").replace(".k_proj.", ".qkv_proj.").replace(".v_proj.", ".qkv_proj.")
                    if qkv_name in params_dict:
                        param = params_dict[qkv_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)

                        # 确定是 q/k/v 中的哪一个
                        if ".q_proj." in name:
                            loaded_shard_id = "q"
                        elif ".k_proj." in name:
                            loaded_shard_id = "k"
                        elif ".v_proj." in name:
                            loaded_shard_id = "v"

                        from inspect import signature
                        sig = signature(weight_loader)
                        if "loaded_shard_id" in sig.parameters:
                            weight_loader(param, tensor, loaded_shard_id=loaded_shard_id)
                            loaded_count += 1
                        else:
                            weight_loader(param, tensor)
                            loaded_count += 1

        logger.info(f"[VIT Runner] Loaded {loaded_count} visual weights from checkpoint")

        self.vit_model.to(self.device)
        self.vit_model.eval()

        logger.info(f"[VIT Runner] ✅ ViT weights loaded via official path; model ready on {self.device}")

        # 🔍 检查 VIT 模型的 dtype
        first_param = next(self.vit_model.parameters())
        logger.info(f"[VIT Runner] ViT model loaded on {self.device}, dtype={first_param.dtype}")

        # 🔍 检查是否有 NaN 参数
        nan_count = 0
        for name, param in self.vit_model.named_parameters():
            if torch.isnan(param).any():
                logger.error(f"[VIT Runner] ❌ Parameter {name} contains NaN!")
                nan_count += 1
        if nan_count > 0:
            logger.error(f"[VIT Runner] ❌ Found {nan_count} parameters with NaN!")
        else:
            logger.info(f"[VIT Runner] ✅ All parameters are valid (no NaN)")
    
    def compute(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """计算 ViT embedding（支持 TP + all-reduce）

        当 TP > 1 时:
        1. 每个 TP rank 计算部分 heads 的 embedding
        2. 使用 NCCL all-reduce 合并结果
        3. 只有 TP rank 0 返回完整的 embedding
        """
        try:
            logger.info(f"[VIT Runner] 🚀 Starting compute: pixel_values.shape={pixel_values.shape}, device={pixel_values.device}")

            # 确保 CUDA 操作在正确的设备上执行
            with torch.cuda.device(self.device):
                # 移动到 GPU（添加超时保护）
                logger.info(f"[VIT Runner] 📥 Moving tensors to {self.device}...")
                pixel_values = pixel_values.to(self.device, non_blocking=False)
                image_grid_thw = image_grid_thw.to(self.device, non_blocking=False)

                # 同步 CUDA 操作，确保数据已经传输完成
                torch.cuda.synchronize(self.device)
                logger.info(f"[VIT Runner] ✅ Tensors moved to {self.device}")

                # 🔍 检查输入数据
                if torch.isnan(pixel_values).any():
                    logger.error(f"[VIT Runner] ❌ Input pixel_values contains NaN!")
                if torch.isinf(pixel_values).any():
                    logger.error(f"[VIT Runner] ❌ Input pixel_values contains Inf!")
                logger.info(f"[VIT Runner] 📊 Input pixel_values: shape={pixel_values.shape}, dtype={pixel_values.dtype}, min={pixel_values.min().item():.4f}, max={pixel_values.max().item():.4f}")
                logger.info(f"[VIT Runner] 📊 Input image_grid_thw: {image_grid_thw}")

                # 执行 ViT 前向传播
                logger.info(f"[VIT Runner] 🔄 Running ViT forward pass...")
                with torch.no_grad():
                    embedding = self.vit_model(pixel_values, grid_thw=image_grid_thw)

                # 同步 CUDA 操作，确保计算完成
                torch.cuda.synchronize(self.device)
                logger.info(f"[VIT Runner] ✅ ViT forward pass completed")

                # 🔧 VIT TP: All-reduce（如果 TP > 1）
                # 注意: SGLang 的 VIT 模型（Qwen2_5_VisionTransformer）使用了 RowParallelLinear
                # RowParallelLinear 会自动执行 all-reduce，所以这里不需要额外的操作
                # 所有 TP ranks 都会得到完整的 embedding
                if self.tp_size > 1:
                    logger.info(f"[VIT Runner] 🔄 TP mode: embedding already all-reduced by RowParallelLinear")

                # 🔍 检查输出数据
                if torch.isnan(embedding).any():
                    logger.error(f"[VIT Runner] ❌ Output embedding contains NaN!")
                if torch.isinf(embedding).any():
                    logger.error(f"[VIT Runner] ❌ Output embedding contains Inf!")
                logger.info(f"[VIT Runner] 📊 Output embedding: shape={embedding.shape}, dtype={embedding.dtype}, min={embedding.min().item() if not torch.isnan(embedding).any() else 'nan'}, max={embedding.max().item() if not torch.isnan(embedding).any() else 'nan'}")

                return embedding

        except Exception as e:
            logger.error(f"[VIT Runner] ❌ Error in compute: {e}", exc_info=True)
            raise


class VITScheduler:
    """
    VIT Scheduler - 独立的 ViT 计算调度器

    功能:
    1. 接收来自主 Scheduler 的 ViT 计算请求
    2. 批量计算 ViT
    3. 缓存 embedding（LRU 策略）
    4. 返回结果给主 Scheduler
    5. 事件驱动释放缓存
    """

    def __init__(
        self,
        model_config,
        device: str = "cuda:0",
        zmq_port: int = 5555,
        batch_size: int = 4,
        batch_timeout_ms: float = 10.0,
        cache_size_mb: int = None,
    ):
        """
        Args:
            model_config: 模型配置
            device: ViT 运行的设备
            zmq_port: ZMQ 通信端口
            batch_size: 批量计算的最大 batch size
            batch_timeout_ms: 批量计算的超时时间（毫秒）
            cache_size_mb: Embedding 缓存大小（MB），默认从环境变量读取
        """
        self.model_config = model_config

        # 🔧 VIT TP: 初始化 TP group
        self.tp_rank = int(os.environ.get("SGLANG_VIT_TP_RANK", "0"))
        self.tp_size = int(os.environ.get("SGLANG_VIT_TP_SIZE", "1"))
        self.vit_tp_port = int(os.environ.get("SGLANG_VIT_TP_PORT", "29500"))

        if self.tp_size > 1:
            # 初始化 distributed
            if not dist.is_initialized():
                logger.info(
                    f"[VIT Scheduler] Initializing distributed: "
                    f"rank={self.tp_rank}, world_size={self.tp_size}, "
                    f"port={self.vit_tp_port}"
                )
                dist.init_process_group(
                    backend="nccl",
                    init_method=f"tcp://localhost:{self.vit_tp_port}",
                    world_size=self.tp_size,
                    rank=self.tp_rank,
                )
                logger.info(f"[VIT Scheduler] Distributed initialized")

            # 设置 device（基于 TP rank）
            self.device = torch.device(f"cuda:{self.tp_rank}")
            torch.cuda.set_device(self.device)

            # 🔧 VIT TP: 初始化 _WORLD group（必须在 initialize_model_parallel 之前）
            from sglang.srt.distributed.parallel_state import (
                init_world_group,
                _WORLD,
            )

            # 检查 _WORLD 是否已初始化
            global_parallel_state = __import__(
                "sglang.srt.distributed.parallel_state", fromlist=["_WORLD"]
            )
            if global_parallel_state._WORLD is None:
                logger.info(
                    f"[VIT Scheduler] Initializing _WORLD group: "
                    f"rank={self.tp_rank}, world_size={self.tp_size}"
                )
                ranks = list(range(self.tp_size))
                global_parallel_state._WORLD = init_world_group(
                    ranks=ranks,
                    local_rank=self.tp_rank,
                    backend="nccl",
                )
                logger.info(f"[VIT Scheduler] _WORLD group initialized")

            # 初始化 TP group
            from sglang.srt.distributed import initialize_model_parallel
            initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                pipeline_model_parallel_size=1,
            )

            logger.info(
                f"[VIT Scheduler] ✅ Initialized TP group: rank={self.tp_rank}, "
                f"size={self.tp_size}, device={self.device}"
            )
        else:
            # 单 GPU 模式
            self.device = device

        self.zmq_port = zmq_port
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0

        # 缓存大小配置（从环境变量读取，默认 2048 MB）
        if cache_size_mb is None:
            cache_size_mb = int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "2048"))
        self.cache_size_mb = cache_size_mb
        self.max_cache_size_bytes = cache_size_mb * 1024 * 1024

        # 测压模式开关（禁用缓存以准确测试性能）
        self.benchmark_mode = os.environ.get("SGLANG_VIT_BENCHMARK_MODE", "0") == "1"

        # 🔧 VIT TP: 只有 TP rank 0 初始化 ZMQ（其他 ranks 只参与计算）
        if self.tp_rank == 0:
            # 初始化 ZMQ（使用 PAIR 模式，一对一最可靠）
            self.context = zmq.Context()

            # 使用 PAIR socket - 一对一通信，最简单最可靠
            self.socket = self.context.socket(zmq.PAIR)
            self.socket.bind(f"tcp://127.0.0.1:{zmq_port}")
            self.socket.setsockopt(zmq.RCVTIMEO, 0)  # 非阻塞接收
            self.socket.setsockopt(zmq.LINGER, 0)

            logger.info(f"[VIT Scheduler] ZMQ server listening (PAIR mode - most reliable)")
            logger.info(f"[VIT Scheduler] Port: {zmq_port} (bidirectional)")
            logger.info(f"[VIT Scheduler] Waiting for client to connect...")
        else:
            # TP rank > 0: 不初始化 ZMQ，只参与计算
            self.context = None
            self.socket = None
            logger.info(f"[VIT Scheduler] TP rank {self.tp_rank}: no ZMQ (only rank 0 handles communication)")

        # 初始化模型（传递 tp_size）
        self.model_runner = VITModelRunner(model_config, self.device, tp_size=self.tp_size)
        self.model_runner.load_model()

        # 初始化缓存（使用 OrderedDict 实现 LRU）
        from collections import OrderedDict
        self.embedding_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.cache_size_bytes = 0

        # 日志输出
        if self.benchmark_mode:
            logger.warning("[VIT Scheduler] ⚠️ Benchmark mode enabled: cache is DISABLED")
        else:
            logger.info(f"[VIT Scheduler] Cache enabled: {cache_size_mb} MB (LRU strategy, ~{cache_size_mb // 132} embeddings)")
            logger.info(f"[VIT Scheduler] Event-driven cache release enabled")

        # 批处理队列
        self.pending_requests: List[VITRequest] = []  # 不再需要 client_id
        self.last_batch_time = time.time()

        # 统计信息
        self.total_requests = 0
        self.cache_hits = 0
        self.total_compute_time = 0.0

        # 🔧 方案 2B: 更频繁地检查 free 信号（简化版，不使用单独线程）
        import threading
        self._cache_lock = threading.Lock()  # 保护 embedding_cache 的锁

        # 🔧 方案 4A: 共享内存超时清理
        self._shm_registry: Dict[str, float] = {}  # shm_name -> 创建时间
        self._shm_registry_lock = threading.Lock()
        self._shm_timeout_seconds = 10.0  # 共享内存超时时间（秒）

        # 🔧 新优化: 单独线程处理 free 信号
        self._free_signal_queue = []  # free 信号队列
        self._free_signal_queue_lock = threading.Lock()
        self._free_signal_thread = None
        self._stop_free_signal_thread = threading.Event()

        # 🔧 新优化: 动态批处理大小
        self._dynamic_batch_size = batch_size  # 当前动态批处理大小
        self._min_batch_size = max(1, batch_size // 2)  # 最小批处理大小
        self._max_batch_size = batch_size * 2  # 最大批处理大小
        self._last_gpu_memory_check = time.time()
        self._gpu_memory_check_interval = 1.0  # GPU 显存检查间隔（秒）

        # 🔧 新增: 正在处理的请求集合（防止重复处理）
        self._processing_requests: Set[str] = set()  # 正在处理的 request_id
        self._processing_lock = threading.Lock()

        logger.info("[VIT Scheduler] Initialized successfully")
        logger.info(f"[VIT Scheduler] 🔧 Dynamic batch size enabled: min={self._min_batch_size}, max={self._max_batch_size}, initial={self._dynamic_batch_size}")
    
    def _compute_hash(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> int:
        """计算输入的 hash 值"""
        hash_input = (
            pixel_values.cpu().numpy().tobytes() +
            image_grid_thw.cpu().numpy().tobytes()
        )
        return int(hashlib.md5(hash_input).hexdigest()[:16], 16)
    
    def _load_tensor_from_shm(self, shm_name: str, shape: Tuple, dtype: str) -> torch.Tensor:
        """从共享内存加载 tensor"""
        from multiprocessing import shared_memory
        
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
        
        # 关闭共享内存（不删除）
        shm.close()
        
        return tensor
    
    def _save_tensor_to_shm(self, tensor: torch.Tensor, shm_name: str):
        """保存 tensor 到共享内存"""
        from multiprocessing import shared_memory

        # 创建共享内存
        tensor_cpu = tensor.cpu()
        nbytes = tensor_cpu.element_size() * tensor_cpu.nelement()
        shm = shared_memory.SharedMemory(create=True, size=nbytes, name=shm_name)

        # 写入数据
        shm_tensor = torch.frombuffer(shm.buf, dtype=tensor_cpu.dtype).reshape(tensor_cpu.shape)
        shm_tensor.copy_(tensor_cpu)

        # 关闭共享内存（不删除，等待接收方读取）
        shm.close()

        # 🔧 方案 4A: 注册共享内存，记录创建时间
        with self._shm_registry_lock:
            self._shm_registry[shm_name] = time.time()
            logger.debug(f"[VIT Scheduler] 📝 Registered SHM: {shm_name}")

    def _cleanup_timeout_shm(self):
        """🔧 方案 4A: 清理超时的共享内存"""
        from multiprocessing import shared_memory

        now = time.time()
        to_cleanup = []

        with self._shm_registry_lock:
            for shm_name, create_time in list(self._shm_registry.items()):
                if now - create_time > self._shm_timeout_seconds:
                    to_cleanup.append(shm_name)

        for shm_name in to_cleanup:
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
                with self._shm_registry_lock:
                    self._shm_registry.pop(shm_name, None)
                logger.warning(
                    f"[VIT Scheduler] 🗑️ Cleaned up timeout SHM: {shm_name} "
                    f"(age: {now - create_time:.1f}s > {self._shm_timeout_seconds}s)"
                )
            except FileNotFoundError:
                # 共享内存已被删除（正常情况）
                with self._shm_registry_lock:
                    self._shm_registry.pop(shm_name, None)
                logger.debug(f"[VIT Scheduler] ✅ SHM already cleaned: {shm_name}")
            except Exception as e:
                logger.error(f"[VIT Scheduler] ❌ Failed to cleanup SHM {shm_name}: {e}")

    def _unregister_shm(self, shm_name: str):
        """🔧 方案 4A: 从注册表中移除共享内存（当 Client 成功读取后调用）"""
        with self._shm_registry_lock:
            if shm_name in self._shm_registry:
                self._shm_registry.pop(shm_name)
                logger.debug(f"[VIT Scheduler] ✅ Unregistered SHM: {shm_name}")

    def _get_gpu_memory_usage(self) -> float:
        """🔧 新优化: 获取 GPU 显存使用率"""
        try:
            # 获取当前设备
            device_id = int(self.device.split(':')[1]) if ':' in self.device else 0

            # 获取显存信息
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)

            usage_ratio = allocated_memory / total_memory
            return usage_ratio
        except Exception as e:
            logger.error(f"[VIT Scheduler] ❌ Failed to get GPU memory usage: {e}")
            return 0.5  # 默认返回 50%

    def _adjust_dynamic_batch_size(self):
        """🔧 新优化: 根据 GPU 显存使用情况动态调整批处理大小"""
        now = time.time()

        # 检查是否需要更新（避免频繁检查）
        if now - self._last_gpu_memory_check < self._gpu_memory_check_interval:
            return

        self._last_gpu_memory_check = now

        # 获取 GPU 显存使用率
        memory_usage = self._get_gpu_memory_usage()

        old_batch_size = self._dynamic_batch_size

        # 根据显存使用率调整批处理大小
        if memory_usage > 0.85:
            # 显存使用率 > 85%，减小批处理大小
            self._dynamic_batch_size = max(self._min_batch_size, self._dynamic_batch_size - 1)
            logger.warning(
                f"[VIT Scheduler] 🔧 GPU memory high ({memory_usage:.1%}), "
                f"reducing batch size: {old_batch_size} -> {self._dynamic_batch_size}"
            )
        elif memory_usage < 0.60:
            # 显存使用率 < 60%，增加批处理大小
            self._dynamic_batch_size = min(self._max_batch_size, self._dynamic_batch_size + 1)
            logger.info(
                f"[VIT Scheduler] 🔧 GPU memory low ({memory_usage:.1%}), "
                f"increasing batch size: {old_batch_size} -> {self._dynamic_batch_size}"
            )

        # 如果批处理大小发生变化，记录日志
        if self._dynamic_batch_size != old_batch_size:
            logger.info(
                f"[VIT Scheduler] 📊 Dynamic batch size adjusted: {old_batch_size} -> {self._dynamic_batch_size} "
                f"(memory usage: {memory_usage:.1%})"
            )

    def _process_batch(self):
        """🔧 新优化: 批量处理请求（动态批处理大小 + 分批处理）"""
        if not self.pending_requests:
            return

        # 🔧 新优化: 动态调整批处理大小
        self._adjust_dynamic_batch_size()

        batch_start_time = time.time()
        total_pending = len(self.pending_requests)

        # 🔧 新优化: 使用动态批处理大小
        actual_batch_size = min(total_pending, self._dynamic_batch_size)
        batch_to_process = self.pending_requests[:actual_batch_size]
        remaining_requests = self.pending_requests[actual_batch_size:]

        if total_pending > self._dynamic_batch_size:
            logger.warning(
                f"[VIT Scheduler] ⚠️ Pending requests ({total_pending}) exceeds dynamic_batch_size ({self._dynamic_batch_size}). "
                f"Processing {actual_batch_size} requests, {len(remaining_requests)} remaining in queue."
            )
        else:
            logger.info(f"[VIT Scheduler] Processing batch of {actual_batch_size} requests (dynamic_batch_size={self._dynamic_batch_size})")

        # 逐个处理（后续可以优化为真正的批量计算）
        for request in batch_to_process:
            try:
                response = self._process_single_request(request)

                # 🔧 新增: 跳过 None 响应（重复请求或共享内存不存在）
                if response is None:
                    logger.warning(f"[VIT Scheduler] ⚠️ Skipping None response for {request.request_id}")
                    continue

                # 发送响应：直接发送消息
                message_data = pickle.dumps(asdict(response))

                logger.info(
                    f"[VIT Scheduler] 📤 Sending response: {response.request_id}, "
                    f"message_size={len(message_data)} bytes"
                )

                # PAIR socket 发送（更可靠）
                try:
                    # 检查 socket 状态
                    events_before = self.socket.getsockopt(zmq.EVENTS)
                    logger.info(f"[VIT Scheduler] Socket events before send: {events_before} (POLLOUT={zmq.POLLOUT}, POLLIN={zmq.POLLIN})")

                    self.socket.send(message_data, zmq.NOBLOCK)

                    # 等待一小段时间确保消息被发送到网络层
                    # 这对于 PAIR socket 很重要，避免消息丢失
                    time.sleep(0.001)  # 1ms

                    # 检查发送后的状态
                    events_after = self.socket.getsockopt(zmq.EVENTS)
                    logger.info(
                        f"[VIT Scheduler] ✅ Sent response: {response.request_id}, "
                        f"from_cache={response.from_cache}, "
                        f"compute_time={response.compute_time*1000:.1f}ms, "
                        f"events_after={events_after}"
                    )

                    # 立即尝试接收（看看是否有 ACK 或其他消息）
                    try:
                        ack = self.socket.recv(zmq.NOBLOCK)
                        logger.info(f"[VIT Scheduler] 📥 Received immediate response: {len(ack)} bytes")
                    except zmq.Again:
                        pass  # 没有立即响应，正常

                except zmq.Again:
                    logger.error(f"[VIT Scheduler] ❌ Send would block!")
                    # 尝试阻塞发送
                    self.socket.send(message_data)
                    logger.info(f"[VIT Scheduler] ✅ Sent with blocking mode")
            except Exception as e:
                logger.error(f"[VIT Scheduler] Error processing request {request.request_id}: {e}", exc_info=True)

        # 🔧 方案 3B: 将剩余请求放回队列
        self.pending_requests = remaining_requests

        batch_time = time.time() - batch_start_time
        logger.info(
            f"[VIT Scheduler] Batch processed in {batch_time*1000:.1f}ms, "
            f"remaining in queue: {len(self.pending_requests)}"
        )

        # 定期打印统计信息
        if self.total_requests % 100 == 0:
            self._log_stats()

        # 定期清理显存碎片（每 10 个批次）
        if self.total_requests % 10 == 0:
            torch.cuda.empty_cache()

        # 🔧 方案 4A: 定期清理超时的共享内存（每 10 个批次）
        if self.total_requests % 10 == 0:
            self._cleanup_timeout_shm()

    def _process_single_request(self, request: VITRequest) -> VITResponse:
        """🔧 CUDA IPC: 处理单个请求（使用 CUDA IPC 代替 CPU 共享内存）

        Args:
            request: VIT 请求

        Returns:
            VITResponse: 包含 CUDA IPC handle 的响应

        Raises:
            FileNotFoundError: 共享内存不存在（重复请求）
            RuntimeError: CUDA IPC 创建失败
        """
        start_time = time.time()

        try:
            logger.info(f"[VIT Scheduler] 🔄 Processing request: {request.request_id}")

            # 🔧 检查是否正在处理中（防止重复处理）
            with self._processing_lock:
                if request.request_id in self._processing_requests:
                    logger.warning(
                        f"[VIT Scheduler] ⚠️ Request {request.request_id} is already being processed, "
                        "skipping duplicate"
                    )
                    return None
                self._processing_requests.add(request.request_id)

            try:
                # 从共享内存加载输入
                logger.info(f"[VIT Scheduler] 📥 Loading tensors from shared memory...")
                try:
                    pixel_values = self._load_tensor_from_shm(
                        request.pixel_values_shm_name,
                        request.pixel_values_shape,
                        request.pixel_values_dtype,
                    )
                    logger.info(f"[VIT Scheduler] ✅ Loaded pixel_values: shape={pixel_values.shape}")

                    image_grid_thw = self._load_tensor_from_shm(
                        request.image_grid_thw_shm_name,
                        request.image_grid_thw_shape,
                        request.image_grid_thw_dtype,
                    )
                    logger.info(f"[VIT Scheduler] ✅ Loaded image_grid_thw: shape={image_grid_thw.shape}")

                except FileNotFoundError as e:
                    # 共享内存不存在，可能是重复请求
                    logger.warning(
                        f"[VIT Scheduler] ⚠️ Shared memory not found for {request.request_id}: {e}"
                    )
                    logger.warning(
                        f"[VIT Scheduler] 🔍 This is likely a duplicate/retry request. Skipping."
                    )
                    return None

                # 计算 hash
                logger.info(f"[VIT Scheduler] 🔢 Computing hash...")
                hash_val = self._compute_hash(pixel_values, image_grid_thw)
                logger.info(f"[VIT Scheduler] ✅ Hash computed: {hash_val}")

                # 查询缓存（使用 LRU 更新）
                from_cache = False
                embedding = self._get_cached_embedding(hash_val)
                if embedding is not None:
                    from_cache = True
                    self.cache_hits += 1
                    logger.info(f"[VIT Scheduler] ✅ Cache hit: hash={hash_val}")

                    # 🔧 VIT TP: 即使 cache hit，也需要通知其他 ranks（发送空的 broadcast）
                    # 这样 TP rank 1 就不会一直等待
                    if self.tp_size > 1:
                        from sglang.srt.distributed.communication_op import broadcast_tensor_dict

                        logger.info(f"[VIT Scheduler] 📡 Broadcasting cache hit signal to other TP ranks...")
                        # 发送一个空的 tensor_dict，表示不需要计算
                        # 注意: broadcast_tensor_dict 需要所有 ranks 都调用，所以我们发送一个特殊标记
                        cache_hit_signal = {
                            "cache_hit": True,
                            "exit": False,
                        }
                        broadcast_tensor_dict(tensor_dict=cache_hit_signal, src=0)
                        logger.info(f"[VIT Scheduler] ✅ Cache hit signal broadcasted")
                else:
                    # 计算 ViT
                    logger.info(f"[VIT Scheduler] 🚀 Cache miss, computing ViT embedding...")

                    # 🔧 VIT TP: 如果 TP > 1，通过 broadcast 同步请求数据给其他 ranks
                    if self.tp_size > 1:
                        from sglang.srt.distributed.communication_op import broadcast_tensor_dict

                        logger.info(f"[VIT Scheduler] 📡 Broadcasting request data to other TP ranks...")
                        request_data = {
                            "pixel_values": pixel_values,
                            "image_grid_thw": image_grid_thw,
                            "exit": False,
                        }
                        broadcast_tensor_dict(tensor_dict=request_data, src=0)
                        logger.info(f"[VIT Scheduler] ✅ Request data broadcasted")

                    # 所有 ranks 执行计算（RowParallelLinear 会自动 all-reduce）
                    embedding = self.model_runner.compute(pixel_values, image_grid_thw)
                    logger.info(f"[VIT Scheduler] ✅ ViT computation completed")

                    # 验证 embedding
                    if torch.isnan(embedding).any():
                        logger.error(
                            f"[VIT Scheduler] ❌ Computed embedding contains NaN! "
                            f"shape={embedding.shape}"
                        )
                    else:
                        logger.info(
                            f"[VIT Scheduler] ✅ Computed embedding is valid: "
                            f"shape={embedding.shape}, min={embedding.min().item():.4f}, "
                            f"max={embedding.max().item():.4f}"
                        )

                    # 更新缓存（LRU）
                    logger.info(f"[VIT Scheduler] 💾 Updating cache...")
                    self._update_cache(hash_val, embedding)
                    logger.info(f"[VIT Scheduler] ✅ Cache updated")

                # 🔧 VIT TP: 只有 TP rank 0 创建 CUDA IPC handle 并返回响应
                # TP rank 1 只参与计算，不返回响应
                if self.tp_rank == 0:
                    # 🔧 CUDA IPC: 创建 IPC handle（代替保存到 CPU 共享内存）
                    logger.info(f"[VIT Scheduler] 🔗 Creating CUDA IPC handle...")
                    try:
                        from sglang.semi_pd.utils import get_ipc_handle
                        ipc_handle, offset = get_ipc_handle(embedding)
                        logger.info(
                            f"[VIT Scheduler] ✅ Created CUDA IPC handle: "
                            f"offset={offset}, device={embedding.device}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[VIT Scheduler] ❌ Failed to create CUDA IPC handle: {e}",
                            exc_info=True
                        )
                        raise RuntimeError(f"Failed to create CUDA IPC handle: {e}")

                    compute_time = time.time() - start_time
                    self.total_compute_time += compute_time

                    # 🔧 CUDA IPC: 构造响应（使用 IPC handle）
                    response = VITResponse(
                        request_id=request.request_id,
                        embedding_ipc_handle=(ipc_handle, offset),
                        embedding_shape=tuple(embedding.shape),
                        embedding_dtype=str(embedding.dtype).replace('torch.', ''),
                        embedding_device=str(embedding.device),
                        image_hash=hash_val,
                        compute_time=compute_time,
                        from_cache=from_cache,
                    )

                    logger.info(
                        f"[VIT Scheduler] ✅ Request processed: {request.request_id}, "
                        f"compute_time={compute_time*1000:.1f}ms, from_cache={from_cache}, "
                        f"hash={hash_val}"
                    )
                    return response
                else:
                    # TP rank 1: 只参与计算，不返回响应
                    logger.info(
                        f"[VIT Scheduler] ✅ TP rank {self.tp_rank}: computation completed, "
                        f"no response returned (only rank 0 returns response)"
                    )
                    return None

            finally:
                # 清理 processing 标记
                with self._processing_lock:
                    self._processing_requests.discard(request.request_id)

        except Exception as e:
            logger.error(
                f"[VIT Scheduler] ❌ Error processing request {request.request_id}: {e}",
                exc_info=True
            )
            # 清理 processing 标记
            with self._processing_lock:
                self._processing_requests.discard(request.request_id)
            raise
    
    def _update_cache(self, hash_val: int, embedding: torch.Tensor):
        """🔧 CUDA IPC: 更新缓存（LRU 策略，线程安全，保持 embedding 在 GPU 上）

        Args:
            hash_val: 图片的 hash 值
            embedding: 计算出的 embedding tensor（必须在 GPU 上）

        Notes:
            - Embedding 必须保持在 GPU 上，以便通过 CUDA IPC 共享
            - 使用 LRU 策略驱逐最久未使用的 embedding
            - 线程安全（使用 _cache_lock）
        """
        # 测压模式：不缓存
        if self.benchmark_mode:
            return

        with self._cache_lock:
            # 如果已在缓存中，更新访问时间
            if hash_val in self.embedding_cache:
                self.embedding_cache.move_to_end(hash_val)
                logger.debug(f"[VIT Scheduler] 🔄 Cache hit (LRU update): hash={hash_val}")
                return

            embedding_size = embedding.element_size() * embedding.nelement()

            # 🔧 CUDA IPC: 确保 embedding 在 GPU 上
            if not embedding.is_cuda:
                logger.warning(
                    f"[VIT Scheduler] ⚠️ Embedding is not on GPU (device={embedding.device}), "
                    "moving to GPU for CUDA IPC sharing"
                )
                embedding = embedding.to(self.device)

            # LRU 驱逐：移除最久未使用的条目
            while self.cache_size_bytes + embedding_size > self.max_cache_size_bytes:
                if len(self.embedding_cache) == 0:
                    break
                oldest_hash, oldest_emb = self.embedding_cache.popitem(last=False)
                evicted_size = oldest_emb.element_size() * oldest_emb.nelement()
                self.cache_size_bytes -= evicted_size
                logger.info(
                    f"[VIT Scheduler] 🗑️ Evicted embedding (LRU): hash={oldest_hash}, "
                    f"size={evicted_size/1024/1024:.1f}MB"
                )
                # 🔧 CUDA IPC: 不需要手动释放，Python GC 会自动处理

            # 添加到缓存（保持在 GPU 上）
            self.embedding_cache[hash_val] = embedding
            self.cache_size_bytes += embedding_size

            logger.info(
                f"[VIT Scheduler] 💾 Added to cache: hash={hash_val}, "
                f"size={embedding_size/1024/1024:.1f}MB, "
                f"total_cache_size={self.cache_size_bytes/1024/1024:.1f}MB"
            )

    def _get_cached_embedding(self, hash_val: int) -> Optional[torch.Tensor]:
        """🔧 方案 2B: 查询缓存（更新 LRU，线程安全）"""
        # 测压模式：不使用缓存
        if self.benchmark_mode:
            return None

        with self._cache_lock:
            if hash_val in self.embedding_cache:
                # 更新访问时间
                self.embedding_cache.move_to_end(hash_val)
                return self.embedding_cache[hash_val]
            return None

    def _free_cache(self, hash_val: int):
        """🔧 方案 2B: 释放缓存（事件驱动，线程安全）"""
        with self._cache_lock:
            if hash_val in self.embedding_cache:
                embedding = self.embedding_cache.pop(hash_val)
                freed_size = embedding.element_size() * embedding.nelement()
                self.cache_size_bytes -= freed_size
                logger.info(f"[VIT Scheduler] 🗑️ Freed embedding (event-driven): hash={hash_val}, size={freed_size/1024/1024:.2f} MB")

    def _start_free_signal_thread(self):
        """🔧 新优化: 启动单独线程处理 free 信号"""
        import threading
        self._free_signal_thread = threading.Thread(
            target=self._free_signal_thread_worker,
            daemon=True,
            name="VIT-FreeSignal"
        )
        self._free_signal_thread.start()
        logger.info("[VIT Scheduler] 🧵 Started free signal processing thread")

    def _stop_free_signal_thread_func(self):
        """🔧 新优化: 停止 free signal 线程"""
        if self._free_signal_thread is not None:
            logger.info("[VIT Scheduler] 🧵 Stopping free signal thread...")
            self._stop_free_signal_thread.set()
            self._free_signal_thread.join(timeout=2.0)
            logger.info("[VIT Scheduler] 🧵 Free signal thread stopped")

    def _free_signal_thread_worker(self):
        """🔧 新优化: 单独线程处理 free 信号（持续运行）"""
        logger.info("[VIT Scheduler] 🧵 Free signal thread worker started")

        processed_count = 0
        last_log_time = time.time()

        while not self._stop_free_signal_thread.is_set():
            try:
                # 从队列中获取 free 信号
                free_signals_to_process = []
                with self._free_signal_queue_lock:
                    if self._free_signal_queue:
                        free_signals_to_process = self._free_signal_queue[:]
                        self._free_signal_queue.clear()

                # 处理 free 信号
                for image_hash in free_signals_to_process:
                    self._free_cache(image_hash)
                    processed_count += 1

                # 定期打印统计信息
                if time.time() - last_log_time > 10.0 and processed_count > 0:
                    logger.info(f"[VIT Scheduler] 🧵 Free signal thread: processed {processed_count} signals")
                    last_log_time = time.time()

                # 短暂休眠，避免 CPU 空转
                time.sleep(0.001)

            except Exception as e:
                if not self._stop_free_signal_thread.is_set():
                    logger.error(f"[VIT Scheduler] 🧵 Error in free signal thread: {e}", exc_info=True)
                time.sleep(0.1)  # 出错后休眠更长时间

        logger.info(f"[VIT Scheduler] 🧵 Free signal thread stopped (processed {processed_count} signals)")

    def _drain_free_signals(self):
        """🔧 新优化: 检查并将 free 信号添加到队列（由单独线程处理）"""
        queued_count = 0
        while True:
            try:
                message = self.socket.recv(zmq.NOBLOCK)
                request_dict = pickle.loads(message)

                # 只处理释放信号，其他消息不处理（避免打乱请求顺序）
                if request_dict.get("type") == "free_embedding":
                    image_hash = request_dict["image_hash"]
                    # 🔧 新优化: 将 free 信号添加到队列，由单独线程处理
                    with self._free_signal_queue_lock:
                        self._free_signal_queue.append(image_hash)
                    queued_count += 1
                else:
                    # 非释放信号，直接处理
                    if "test" in request_dict and request_dict["test"] == "connection_test":
                        self.socket.send(pickle.dumps({"test_response": "ok"}))
                    else:
                        # 这是一个 VIT 计算请求，添加到待处理队列
                        request = VITRequest(**request_dict)
                        self.pending_requests.append(request)
                        self.total_requests += 1
                        logger.info(f"[VIT Scheduler] Received request (via drain): {request.request_id}")
                    # 🔧 修复：继续处理后续消息，而不是 break
                    # 这样可以确保所有待处理的消息都被接收
            except zmq.Again:
                # 没有更多消息了
                break
            except Exception as e:
                logger.error(f"[VIT Scheduler] Error draining free signals: {e}", exc_info=True)
                break

        if queued_count > 0:
            logger.debug(f"[VIT Scheduler] 🔄 Queued {queued_count} free signals for processing")

    def run(self):
        """🔧 新优化: 主循环（单独线程处理 free 信号 + 动态批处理大小）

        🔧 VIT TP 支持:
        - TP rank 0: 监听 ZMQ 请求，通过 broadcast 同步给其他 ranks
        - TP rank > 0: 等待 broadcast，参与计算
        - 所有 ranks 同时执行 forward pass (RowParallelLinear 需要 all-reduce)
        """
        logger.info("[VIT Scheduler] Starting main loop (dynamic batch size + dedicated free signal thread)")

        # 🔧 VIT TP: TP rank > 0 使用 worker 模式（等待 broadcast）
        if self.tp_rank > 0:
            logger.info(f"[VIT Scheduler] TP rank {self.tp_rank}: entering worker mode (waiting for broadcast)")
            self._run_tp_worker()
            return

        # 🔧 新优化: 启动单独线程处理 free 信号（只在 TP rank 0）
        self._start_free_signal_thread()

        try:
            while True:
                # 🔧 新优化: 在收集请求前先检查 free 信号（将信号添加到队列）
                self._drain_free_signals()

                # 收集请求（非阻塞）
                # 🔧 新优化: 使用动态批处理大小
                while len(self.pending_requests) < self._dynamic_batch_size:
                    try:
                        # PAIR socket 接收消息
                        message = self.socket.recv(zmq.NOBLOCK)

                        request_dict = pickle.loads(message)

                        # 🔧 新优化: 处理释放信号（添加到队列，由单独线程处理）
                        if request_dict.get("type") == "free_embedding":
                            image_hash = request_dict["image_hash"]
                            with self._free_signal_queue_lock:
                                self._free_signal_queue.append(image_hash)
                            continue

                        # 检查是否是测试消息
                        if "test" in request_dict and request_dict["test"] == "connection_test":
                            logger.info(f"[VIT Scheduler] ✅ Received test message from client")
                            # 发送测试响应
                            self.socket.send(pickle.dumps({"test_response": "ok"}))
                            logger.info(f"[VIT Scheduler] ✅ Sent test response")
                            continue

                        request = VITRequest(**request_dict)

                        self.pending_requests.append(request)
                        self.total_requests += 1

                        logger.info(f"[VIT Scheduler] Received request: {request.request_id}")
                    except zmq.Again:
                        # 没有更多请求了
                        break
                    except Exception as e:
                        logger.error(f"[VIT Scheduler] Error receiving request: {e}", exc_info=True)
                        break

                # 🔧 新优化: 判断是否执行批量计算（使用动态批处理大小）
                should_compute = (
                    len(self.pending_requests) >= self._dynamic_batch_size or
                    (len(self.pending_requests) > 0 and
                     time.time() - self.last_batch_time > self.batch_timeout)
                )

                if should_compute:
                    # 🔧 方案 2B: 处理批次前再次检查 free 信号
                    self._drain_free_signals()

                    self._process_batch()
                    self.last_batch_time = time.time()

                    # 🔧 方案 2B: 处理批次后立即检查释放信号（避免延迟）
                    self._drain_free_signals()
                else:
                    # 🔧 休眠前也检查释放信号
                    self._drain_free_signals()
                    # 短暂休眠，避免 CPU 空转
                    time.sleep(0.001)
                    
        except KeyboardInterrupt:
            logger.info("[VIT Scheduler] Received interrupt signal")
        except Exception as e:
            logger.error(f"[VIT Scheduler] Error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def _log_stats(self):
        """打印统计信息"""
        cache_hit_rate = self.cache_hits / max(self.total_requests, 1)
        avg_compute_time = self.total_compute_time / max(self.total_requests, 1)
        
        logger.info(
            f"[VIT Scheduler Stats] "
            f"total_requests={self.total_requests}, "
            f"cache_hits={self.cache_hits}, "
            f"cache_hit_rate={cache_hit_rate:.2%}, "
            f"avg_compute_time={avg_compute_time*1000:.1f}ms, "
            f"cache_size={self.cache_size_bytes/1024/1024:.1f}MB"
        )
    
    def _run_tp_worker(self):
        """🔧 VIT TP: TP rank > 0 的 worker 运行循环

        TP rank > 0 不监听 ZMQ，而是等待 TP rank 0 的 broadcast。
        收到 broadcast 后，参与计算（RowParallelLinear 需要 all-reduce）。
        """
        logger.info(f"[VIT Scheduler] TP rank {self.tp_rank}: worker mode started")

        from sglang.srt.distributed.communication_op import broadcast_tensor_dict

        try:
            while True:
                # 1. 等待 TP rank 0 的 broadcast（接收请求数据）
                # broadcast_tensor_dict 会阻塞直到收到数据
                request_data = broadcast_tensor_dict(tensor_dict=None, src=0)

                if request_data is None:
                    # 没有请求，继续等待
                    time.sleep(0.001)
                    continue

                # 检查是否是退出信号
                if request_data.get("exit", False):
                    logger.info(f"[VIT Scheduler] TP rank {self.tp_rank}: received exit signal")
                    break

                # 检查是否是 cache hit 信号
                if request_data.get("cache_hit", False):
                    logger.info(f"[VIT Scheduler] TP rank {self.tp_rank}: received cache hit signal, skipping computation")
                    continue

                # 2. 提取请求数据
                pixel_values = request_data.get("pixel_values")
                image_grid_thw = request_data.get("image_grid_thw")

                if pixel_values is None or image_grid_thw is None:
                    logger.warning(f"[VIT Scheduler] TP rank {self.tp_rank}: received invalid request data")
                    continue

                logger.info(
                    f"[VIT Scheduler] TP rank {self.tp_rank}: received broadcast, "
                    f"pixel_values.shape={pixel_values.shape}"
                )

                # 3. 执行计算（参与 all-reduce）
                try:
                    embedding = self.model_runner.compute(pixel_values, image_grid_thw)
                    logger.info(
                        f"[VIT Scheduler] TP rank {self.tp_rank}: computation completed, "
                        f"embedding.shape={embedding.shape}"
                    )
                except Exception as e:
                    logger.error(
                        f"[VIT Scheduler] TP rank {self.tp_rank}: error in compute: {e}",
                        exc_info=True
                    )

        except KeyboardInterrupt:
            logger.info(f"[VIT Scheduler] TP rank {self.tp_rank}: received interrupt signal")
        except Exception as e:
            logger.error(f"[VIT Scheduler] TP rank {self.tp_rank}: error in worker loop: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """🔧 新优化: 清理资源（包括停止 free signal 线程）"""
        logger.info("[VIT Scheduler] Cleaning up...")

        # 🔧 VIT TP: TP rank 0 发送退出信号给其他 ranks
        if self.tp_rank == 0 and self.tp_size > 1:
            try:
                from sglang.srt.distributed.communication_op import broadcast_tensor_dict

                logger.info("[VIT Scheduler] Sending exit signal to other TP ranks...")
                exit_signal = {
                    "exit": True,
                }
                broadcast_tensor_dict(tensor_dict=exit_signal, src=0)
                logger.info("[VIT Scheduler] Exit signal sent")
            except Exception as e:
                logger.error(f"[VIT Scheduler] Failed to send exit signal: {e}")

        # 🔧 新优化: 停止 free signal 线程（只在 TP rank 0）
        if self.tp_rank == 0:
            self._stop_free_signal_thread_func()

        # 关闭 ZMQ（只在 TP rank 0）
        if self.tp_rank == 0 and self.socket is not None:
            self.socket.close()
            self.context.term()

        logger.info("[VIT Scheduler] Cleanup complete")


def start_vit_scheduler(
    model_config,
    device: str = "cuda:0",
    zmq_port: int = 5555,
    batch_size: int = 4,
    batch_timeout_ms: float = 10.0,
    cache_size_mb: int = 1024,
    pipe_writer=None,
):
    """启动 VIT Scheduler（在独立进程中调用）"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [VIT Scheduler] %(message)s',
    )

    logger.info(f"Starting VIT Scheduler process, PID={os.getpid()}")

    # 创建 scheduler
    scheduler = VITScheduler(
        model_config=model_config,
        device=device,
        zmq_port=zmq_port,
        batch_size=batch_size,
        batch_timeout_ms=batch_timeout_ms,
        cache_size_mb=cache_size_mb,
    )

    # 通知父进程已经准备好
    if pipe_writer is not None:
        pipe_writer.send("ready")
        logger.info("[VIT Scheduler] Sent ready signal to parent process")

    # 运行主循环
    scheduler.run()

