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
        vit_compute_start_time: VIT 计算开始时间（用于并行性分析）
        vit_compute_end_time: VIT 计算结束时间（用于并行性分析）
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
    # 🔧 新增: 时间戳（用于并行性分析和缓存生命周期跟踪）
    vit_compute_start_time: float = 0.0
    vit_compute_end_time: float = 0.0


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

    def compute_batch(self, pixel_values_list: List[torch.Tensor], image_grid_thw_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """🔧 批量计算 ViT embedding（支持 TP + all-reduce）

        Args:
            pixel_values_list: 批量的 pixel_values tensors
            image_grid_thw_list: 批量的 image_grid_thw tensors

        Returns:
            List[torch.Tensor]: 批量的 embedding tensors

        Notes:
            - 将多个请求的 pixel_values 拼接成一个大的 batch
            - 一次性执行 VIT forward pass
            - 然后将结果拆分回各个请求
            - 这样可以充分利用 GPU 并行计算能力
        """
        try:
            batch_size = len(pixel_values_list)
            logger.info(f"[VIT Runner] 🚀 Starting batch compute: batch_size={batch_size}")

            if batch_size == 0:
                return []

            # 如果只有一个请求，直接调用单个 compute
            if batch_size == 1:
                embedding = self.compute(pixel_values_list[0], image_grid_thw_list[0])
                return [embedding]

            # 确保 CUDA 操作在正确的设备上执行
            with torch.cuda.device(self.device):
                # 🔧 批量拼接 pixel_values
                # pixel_values 的 shape: (num_images, num_channels, height, width)
                # 我们需要将多个请求的 pixel_values 拼接成一个大的 batch
                logger.info(f"[VIT Runner] 📦 Concatenating {batch_size} pixel_values...")

                # 移动到 GPU 并拼接
                pixel_values_gpu = [pv.to(self.device, non_blocking=False) for pv in pixel_values_list]
                image_grid_thw_gpu = [igt.to(self.device, non_blocking=False) for igt in image_grid_thw_list]

                # 拼接 pixel_values (在第 0 维拼接)
                batched_pixel_values = torch.cat(pixel_values_gpu, dim=0)

                # 拼接 image_grid_thw (在第 0 维拼接)
                batched_image_grid_thw = torch.cat(image_grid_thw_gpu, dim=0)

                # 同步 CUDA 操作
                torch.cuda.synchronize(self.device)
                logger.info(f"[VIT Runner] ✅ Batched tensors: pixel_values.shape={batched_pixel_values.shape}, image_grid_thw.shape={batched_image_grid_thw.shape}")

                # 执行批量 ViT 前向传播
                logger.info(f"[VIT Runner] 🔄 Running batched ViT forward pass...")
                with torch.no_grad():
                    batched_embedding = self.vit_model(batched_pixel_values, grid_thw=batched_image_grid_thw)

                # 同步 CUDA 操作
                torch.cuda.synchronize(self.device)
                logger.info(f"[VIT Runner] ✅ Batched ViT forward pass completed: embedding.shape={batched_embedding.shape}")

                # 🔧 VIT TP: All-reduce（如果 TP > 1）
                if self.tp_size > 1:
                    logger.info(f"[VIT Runner] 🔄 TP mode: embedding already all-reduced by RowParallelLinear")

                # 🔧 拆分 embedding 回各个请求
                # batched_embedding 的 shape: (total_num_images, seq_len, hidden_size)
                # 我们需要根据每个请求的 num_images 拆分
                logger.info(f"[VIT Runner] 📦 Splitting batched embedding back to {batch_size} requests...")

                embeddings = []
                start_idx = 0
                for i, pv in enumerate(pixel_values_list):
                    num_images = pv.shape[0]  # 第 0 维是 num_images
                    end_idx = start_idx + num_images
                    embedding = batched_embedding[start_idx:end_idx]
                    embeddings.append(embedding)
                    start_idx = end_idx
                    logger.info(f"[VIT Runner] 📦 Request {i}: embedding.shape={embedding.shape}")

                logger.info(f"[VIT Runner] ✅ Batch compute completed: {batch_size} embeddings")
                return embeddings

        except Exception as e:
            logger.error(f"[VIT Runner] ❌ Error in batch compute: {e}", exc_info=True)
            raise


class EmbeddingPagePool:
    """VIT Embedding 页面池 - 参考 SGLang 的 TokenToKVPoolAllocator 设计

    核心功能:
    1. 管理 embedding 显存页面的分配和释放
    2. 支持成组释放（free_group_begin/end）避免频繁操作
    3. 提供可用页面查询接口
    4. 在初始化时预分配固定大小的显存池

    设计参考:
    - sglang/python/sglang/srt/mem_cache/memory_pool.py::TokenToKVPoolAllocator
    - sglang/docs/vit_scheduler_decoupling_plan.md 第 6.2 节
    """

    def __init__(self, pool_size_gb: float, page_size_mb: float, device: str):
        """初始化 embedding 页面池

        Args:
            pool_size_gb: 显存池总大小（GB）
            page_size_mb: 每个页面大小（MB），通常为单张图片 embedding 的大小
            device: CUDA 设备
        """
        self.pool_size_bytes = int(pool_size_gb * 1024**3)
        self.page_size_bytes = int(page_size_mb * 1024**2)
        self.device = device

        # 计算总页面数
        self.total_pages = self.pool_size_bytes // self.page_size_bytes

        # 维护空闲页面列表（参考 TokenToKVPoolAllocator）
        self.free_pages = list(range(self.total_pages))

        # 成组释放支持（参考 TokenToKVPoolAllocator.free_group_begin/end）
        self.is_not_in_free_group = True
        self.free_group = []

        # ✅ 核心: 预分配显存池（使用 torch.empty 占用显存）
        logger.info(f"[Embedding Page Pool] 🔧 Pre-allocating {pool_size_gb:.2f} GB memory pool on {device}...")
        logger.info(f"[Embedding Page Pool]   - Page size: {page_size_mb:.2f} MB")
        logger.info(f"[Embedding Page Pool]   - Total pages: {self.total_pages}")
        try:
            # 使用 float32 预分配显存（4 bytes per element）
            num_elements = self.pool_size_bytes // 4
            self._pool_tensor = torch.empty(
                num_elements, dtype=torch.float32, device=device
            )
            logger.info(f"[Embedding Page Pool] ✅ Memory pool pre-allocated: {pool_size_gb:.2f} GB")

            # 验证显存确实被占用
            allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
            logger.info(f"[Embedding Page Pool] 📊 GPU memory status:")
            logger.info(f"[Embedding Page Pool]   - Allocated: {allocated_gb:.2f} GB")
            logger.info(f"[Embedding Page Pool]   - Reserved: {reserved_gb:.2f} GB")
            logger.info(f"[Embedding Page Pool]   - Pool size: {pool_size_gb:.2f} GB")
        except Exception as e:
            logger.error(f"[Embedding Page Pool] ❌ Failed to pre-allocate memory pool: {e}")
            raise

    def available_size(self) -> int:
        """返回可用页面数（参考 TokenToKVPoolAllocator.available_size）"""
        return len(self.free_pages)

    def available_bytes(self) -> int:
        """返回可用显存大小（bytes）"""
        return len(self.free_pages) * self.page_size_bytes

    def alloc(self, need_pages: int) -> Optional[List[int]]:
        """分配页面（参考 TokenToKVPoolAllocator.alloc）

        Args:
            need_pages: 需要分配的页面数

        Returns:
            分配的页面索引列表，失败返回 None
        """
        if need_pages > len(self.free_pages):
            return None

        # 从空闲列表中取出前 need_pages 个
        select_pages = self.free_pages[:need_pages]
        self.free_pages = self.free_pages[need_pages:]

        return select_pages

    def free(self, page_indices: List[int]):
        """释放页面（参考 TokenToKVPoolAllocator.free）

        Args:
            page_indices: 要释放的页面索引列表
        """
        if self.is_not_in_free_group:
            # 直接释放
            self.free_pages.extend(page_indices)
        else:
            # 成组释放：先收集，等待 free_group_end
            self.free_group.extend(page_indices)

    def free_group_begin(self):
        """开始成组释放（参考 TokenToKVPoolAllocator.free_group_begin）"""
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        """结束成组释放（参考 TokenToKVPoolAllocator.free_group_end）"""
        self.is_not_in_free_group = True
        if self.free_group:
            self.free_pages.extend(self.free_group)
            self.free_group = []

    def usage_ratio(self) -> float:
        """返回显存使用率（0.0 - 1.0）"""
        used_pages = self.total_pages - len(self.free_pages)
        return used_pages / self.total_pages if self.total_pages > 0 else 0.0

    def get_stats(self) -> dict:
        """返回显存池统计信息"""
        used_pages = self.total_pages - len(self.free_pages)
        return {
            "pool_size_gb": self.pool_size_bytes / 1024**3,
            "page_size_mb": self.page_size_bytes / 1024**2,
            "total_pages": self.total_pages,
            "used_pages": used_pages,
            "free_pages": len(self.free_pages),
            "used_gb": (used_pages * self.page_size_bytes) / 1024**3,
            "available_gb": self.available_bytes() / 1024**3,
            "usage_ratio": self.usage_ratio(),
        }


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
        import os
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

        # ✅✅✅ 核心重新设计: 使用 EmbeddingPagePool（参考 SGLang 的 TokenToKVPoolAllocator）
        # 参考文档: sglang/docs/vit_scheduler_decoupling_plan.md 第 6.2 节

        # 从环境变量读取显存池大小，默认 10.0 GB
        vit_memory_pool_gb = float(os.environ.get("SGLANG_VIT_MEMORY_POOL_GB", "10.0"))

        # 从环境变量读取页面大小，默认 100 MB（约为单张图片 embedding 的大小）
        # 页面大小 = seq_len * hidden_dim * dtype_size
        # 例如: Qwen2.5-VL 的 embedding 约为 (25920, 3584) * 2 bytes ≈ 186 MB
        # 保守估计使用 100 MB
        vit_page_size_mb = float(os.environ.get("SGLANG_VIT_PAGE_SIZE_MB", "100.0"))

        logger.info(f"[VIT Scheduler] 🔧 Initializing Embedding Page Pool:")
        logger.info(f"[VIT Scheduler]   - Pool size: {vit_memory_pool_gb:.2f} GB")
        logger.info(f"[VIT Scheduler]   - Page size: {vit_page_size_mb:.2f} MB")

        self.embedding_page_pool = EmbeddingPagePool(
            pool_size_gb=vit_memory_pool_gb,
            page_size_mb=vit_page_size_mb,
            device=self.device
        )

        # ✅✅✅ 核心重新设计: 每个请求需要的页面数（参考 PrefillAdder 的 token 预算）
        # 从环境变量读取，默认 2 页/请求（2 * 100 MB = 200 MB）
        self._pages_per_request = int(os.environ.get("SGLANG_VIT_PAGES_PER_REQUEST", "2"))
        logger.info(f"[VIT Scheduler] 🔧 Pages per request: {self._pages_per_request}")

        # 🔧 OOM 统计与回退策略（参考 vit_scheduler_decoupling_plan.md 第 6.3 节）
        self._oom_count = 0  # OOM 次数统计
        self._last_oom_time = 0.0  # 上次 OOM 时间
        self._oom_cooldown_seconds = 5.0  # OOM 冷却时间（秒）
        self._consecutive_oom_count = 0  # 连续 OOM 次数
        self._max_consecutive_oom = 3  # 最大连续 OOM 次数，超过则降级到单请求模式

        # ✅ 初始化显存预算（不查询 GPU 显存）
        self._init_memory_budget()

        # 🔧 新增: 正在处理的请求集合（防止重复处理）
        self._processing_requests: Set[str] = set()  # 正在处理的 request_id
        self._processing_lock = threading.Lock()

        logger.info("[VIT Scheduler] Initialized successfully")
        logger.info(f"[VIT Scheduler] 🔧 Dynamic batch size enabled: min={self._min_batch_size}, max={self._max_batch_size}, initial={self._dynamic_batch_size}")

    def _init_memory_budget(self):
        """初始化显存预算（不查询 GPU 显存）

        ✅✅✅ 核心重新设计: 基于 EmbeddingPagePool 的预算管理
        参考: sglang/docs/batch_scheduler_memory.md 第 3 节
        """
        # 使用 embedding 页面池统计信息
        pool_stats = self.embedding_page_pool.get_stats()
        logger.info(f"[VIT Scheduler] 📊 Embedding Page Pool 初始化完成:")
        logger.info(f"[VIT Scheduler]   - 池大小: {pool_stats['pool_size_gb']:.2f} GB")
        logger.info(f"[VIT Scheduler]   - 页面大小: {pool_stats['page_size_mb']:.2f} MB")
        logger.info(f"[VIT Scheduler]   - 总页面数: {pool_stats['total_pages']}")
        logger.info(f"[VIT Scheduler]   - 可用页面数: {pool_stats['free_pages']}")
        logger.info(f"[VIT Scheduler]   - 每请求页面数: {self._pages_per_request}")
        logger.info(f"[VIT Scheduler]   - 最大缓存大小: {self.max_cache_size_bytes / (1024**2):.2f} MB")

        # 估算最大 batch size（参考 PrefillAdder 的 token 预算）
        max_batch_size = pool_stats['free_pages'] // self._pages_per_request
        logger.info(f"[VIT Scheduler]   - 估算最大 batch size: {max_batch_size}")

    def _estimate_batch_pages(self, batch_size: int) -> int:
        """估算批量计算需要的页面数

        ✅✅✅ 核心重新设计: 基于 EmbeddingPagePool 的页面估算
        参考: sglang/docs/batch_scheduler_memory.md 第 3 节 (PrefillAdder 的 token 预算)

        Args:
            batch_size: 批量大小

        Returns:
            估算的页面数
        """
        # ✅ 核心重新设计: 使用页面数估算（每个请求需要 _pages_per_request 页）
        estimated_pages = batch_size * self._pages_per_request

        logger.debug(
            f"[VIT Scheduler] 📊 页面估算: batch_size={batch_size}, "
            f"estimated_pages={estimated_pages} "
            f"({self._pages_per_request} pages/request)"
        )

        return estimated_pages

    def _can_process_batch(self, batch_size: int) -> tuple[bool, str]:
        """检查是否可以处理指定大小的 batch

        ✅✅✅ 核心重新设计: 使用 EmbeddingPagePool 检查，不再查询 GPU 显存
        参考: sglang/docs/batch_scheduler_memory.md 第 3 节 (PrefillAdder.budget_state)

        Args:
            batch_size: 批量大小

        Returns:
            (can_process, reason): 是否可以处理, 原因
        """
        logger.info(f"[VIT Scheduler] 🔍 检查是否可以处理 batch_size={batch_size}")

        # 1. 估算批量计算需要的页面数
        estimated_pages = self._estimate_batch_pages(batch_size)

        # 2. ✅✅✅ 核心重新设计: 检查 embedding 页面池是否有足够页面
        pool_stats = self.embedding_page_pool.get_stats()
        available_pages = self.embedding_page_pool.available_size()

        logger.info(
            f"[VIT Scheduler]   Embedding Page Pool 状态: "
            f"需求={estimated_pages} pages, "
            f"可用={available_pages} pages, "
            f"已用={pool_stats['used_pages']} pages, "
            f"总计={pool_stats['total_pages']} pages, "
            f"使用率={pool_stats['usage_ratio']*100:.1f}%"
        )

        # 3. 检查页面池是否有足够页面（参考 PrefillAdder.budget_state）
        if estimated_pages > available_pages:
            reason = (
                f"Embedding 页面池空间不足: "
                f"需求 {estimated_pages} pages, "
                f"可用 {available_pages} pages"
            )
            logger.warning(f"[VIT Scheduler] ❌ {reason}")
            return False, reason

        # 4. 检查缓存占用（如果缓存太满，可能需要驱逐）
        cache_usage_ratio = self.cache_size_bytes / self.max_cache_size_bytes if self.max_cache_size_bytes > 0 else 0.0
        logger.info(
            f"[VIT Scheduler]   缓存占用: {cache_usage_ratio*100:.1f}% "
            f"({self.cache_size_bytes / (1024**2):.1f} MB / {self.max_cache_size_bytes / (1024**2):.1f} MB)"
        )

        # 注意: 不阻止处理，只是警告
        # 如果缓存满了，会在添加新缓存时自动驱逐
        if cache_usage_ratio > 0.9:
            logger.warning(
                f"[VIT Scheduler] ⚠️ 缓存占用较高: {cache_usage_ratio*100:.1f}%, "
                f"可能会触发 LRU 驱逐"
            )

        # 4. 检查 OOM 冷却时间
        # ✅ 修复: 在显存充足时（可用显存 > 10 GiB）跳过 OOM 冷却检查
        if self._oom_count > 0:
            time_since_oom = time.time() - self._last_oom_time
            if time_since_oom < self._oom_cooldown_seconds:
                # 检查实际可用显存，如果充足则跳过冷却
                try:
                    free_memory, _ = torch.cuda.mem_get_info(self.device)
                    free_memory_gb = free_memory / (1024**3)

                    if free_memory_gb > 10.0:
                        # 显存充足，跳过 OOM 冷却
                        logger.info(
                            f"[VIT Scheduler] ✅ 显存充足 ({free_memory_gb:.2f} GiB > 10 GiB), "
                            f"跳过 OOM 冷却检查 (剩余 {self._oom_cooldown_seconds - time_since_oom:.1f}s)"
                        )
                    else:
                        # 显存不足，继续冷却
                        reason = f"OOM 冷却中: 剩余 {self._oom_cooldown_seconds - time_since_oom:.1f}s, 可用显存 {free_memory_gb:.2f} GiB"
                        logger.warning(f"[VIT Scheduler] ❌ {reason}")
                        return False, reason
                except Exception as e:
                    logger.warning(f"[VIT Scheduler] ⚠️ 无法获取 GPU 显存信息: {e}")
                    # 无法获取显存信息，继续冷却
                    reason = f"OOM 冷却中: 剩余 {self._oom_cooldown_seconds - time_since_oom:.1f}s"
                    logger.warning(f"[VIT Scheduler] ❌ {reason}")
                    return False, reason

        logger.info(f"[VIT Scheduler] ✅ 可以处理 batch_size={batch_size}")
        return True, "OK"

    def _adjust_batch_size_for_memory(self, requested_batch_size: int) -> int:
        """根据显存情况动态调整批量大小

        Args:
            requested_batch_size: 请求的批量大小

        Returns:
            调整后的批量大小
        """
        logger.info(
            f"[VIT Scheduler] 🔧 Adjusting batch size for memory: requested={requested_batch_size}"
        )

        # 从请求的批量大小开始,逐步减小直到可以处理
        for batch_size in range(requested_batch_size, 0, -1):
            can_process, reason = self._can_process_batch(batch_size)
            if can_process:
                if batch_size < requested_batch_size:
                    logger.warning(
                        f"[VIT Scheduler] 📉 Reduced batch size: {requested_batch_size} -> {batch_size} "
                        f"(reason: {reason})"
                    )
                else:
                    logger.info(
                        f"[VIT Scheduler] ✅ Batch size OK: {batch_size}"
                    )
                return batch_size

        # 如果连 batch_size=1 都不能处理,返回 0
        logger.warning(f"[VIT Scheduler] ⚠️ Cannot process any batch (even size=1)")
        return 0

    def _update_dynamic_batch_size(self):
        """更新动态批量大小

        ✅ 参考 SGLang 设计: 基于预算和 OOM 历史，不查询 GPU 显存
        """
        # 检查是否需要更新
        now = time.time()
        if now - self._last_gpu_memory_check < self._gpu_memory_check_interval:
            return

        self._last_gpu_memory_check = now

        # ✅ 基于预算使用率调整批量大小
        max_batch_memory_bytes = int(self._max_batch_memory_gb * 1024**3)
        memory_usage_ratio = self._current_batch_memory_bytes / max_batch_memory_bytes if max_batch_memory_bytes > 0 else 0.0

        # 根据预算使用率调整批量大小
        if memory_usage_ratio > 0.9:  # 预算使用率 > 90%
            # 减小批量大小
            new_batch_size = max(self._min_batch_size, self._dynamic_batch_size - 1)
            if new_batch_size != self._dynamic_batch_size:
                logger.info(
                    f"[VIT Scheduler] 📉 减小批量大小: {self._dynamic_batch_size} -> {new_batch_size} "
                    f"(预算使用率: {memory_usage_ratio*100:.1f}%)"
                )
                self._dynamic_batch_size = new_batch_size
        elif memory_usage_ratio < 0.7 and self._oom_count == 0:  # 预算使用率 < 70% 且无 OOM
            # 增加批量大小（只在没有 OOM 时）
            new_batch_size = min(self._max_batch_size, self._dynamic_batch_size + 1)
            if new_batch_size != self._dynamic_batch_size:
                logger.info(
                    f"[VIT Scheduler] 📈 增加批量大小: {self._dynamic_batch_size} -> {new_batch_size} "
                    f"(预算使用率: {memory_usage_ratio*100:.1f}%)"
                )
                self._dynamic_batch_size = new_batch_size

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

    def _adjust_dynamic_batch_size(self):
        """动态调整批处理大小

        ✅✅✅ 核心重新设计: 基于 EmbeddingPagePool 使用率调整批量大小
        参考: sglang/docs/batch_scheduler_memory.md 第 5 节 (update_running_batch 的 new_token_ratio 调整)
        """
        now = time.time()

        # 检查是否需要更新（避免频繁检查）
        if now - self._last_gpu_memory_check < self._gpu_memory_check_interval:
            return

        self._last_gpu_memory_check = now

        # ✅✅✅ 核心重新设计: 使用 embedding 页面池使用率
        pool_stats = self.embedding_page_pool.get_stats()
        memory_usage = pool_stats['usage_ratio']  # 0.0 - 1.0

        old_batch_size = self._dynamic_batch_size

        # 根据页面池使用率调整批处理大小（参考 new_token_ratio 的调整逻辑）
        if memory_usage > 0.85:
            # 页面池使用率 > 85%，减小批处理大小
            self._dynamic_batch_size = max(self._min_batch_size, self._dynamic_batch_size - 1)
            logger.warning(
                f"[VIT Scheduler] 🔧 Embedding 页面池使用率高 ({memory_usage:.1%}), "
                f"减小批量大小: {old_batch_size} -> {self._dynamic_batch_size}"
            )
        elif memory_usage < 0.60 and self._consecutive_oom_count == 0:
            # 页面池使用率 < 60% 且无连续 OOM，增加批处理大小
            self._dynamic_batch_size = min(self._max_batch_size, self._dynamic_batch_size + 1)
            logger.info(
                f"[VIT Scheduler] 🔧 Embedding 页面池使用率低 ({memory_usage:.1%}), "
                f"增加批量大小: {old_batch_size} -> {self._dynamic_batch_size}"
            )

        # 如果批处理大小发生变化，记录日志
        if self._dynamic_batch_size != old_batch_size:
            logger.info(
                f"[VIT Scheduler] 📊 动态批量大小已调整: {old_batch_size} -> {self._dynamic_batch_size} "
                f"(页面池使用率: {memory_usage:.1%}, 已用: {pool_stats['used_pages']} pages, "
                f"可用: {pool_stats['free_pages']} pages)"
            )

    def _process_batch(self):
        """🔧 批量处理请求（真正的批量计算 + 动态批处理大小）"""
        if not self.pending_requests:
            return

        # 🔧 动态调整批处理大小
        self._adjust_dynamic_batch_size()

        batch_start_time = time.time()
        total_pending = len(self.pending_requests)

        # 🔧 使用动态批处理大小
        actual_batch_size = min(total_pending, self._dynamic_batch_size)
        batch_to_process = self.pending_requests[:actual_batch_size]
        remaining_requests = self.pending_requests[actual_batch_size:]

        if total_pending > self._dynamic_batch_size:
            logger.warning(
                f"[VIT Scheduler] ⚠️ Pending requests ({total_pending}) exceeds dynamic_batch_size ({self._dynamic_batch_size}). "
                f"Processing {actual_batch_size} requests, {len(remaining_requests)} remaining in queue."
            )
        else:
            logger.info(f"[VIT Scheduler] 📦 Processing batch of {actual_batch_size} requests (dynamic_batch_size={self._dynamic_batch_size})")

        # 🔧 批量处理: 分离 cache hit 和 cache miss
        cache_hits = []
        cache_misses = []

        for request in batch_to_process:
            try:
                # 检查是否正在处理中
                with self._processing_lock:
                    if request.request_id in self._processing_requests:
                        logger.warning(f"[VIT Scheduler] ⚠️ Request {request.request_id} is already being processed, skipping")
                        continue
                    self._processing_requests.add(request.request_id)

                # 从共享内存加载输入
                try:
                    pixel_values = self._load_tensor_from_shm(
                        request.pixel_values_shm_name,
                        request.pixel_values_shape,
                        request.pixel_values_dtype
                    )
                    image_grid_thw = self._load_tensor_from_shm(
                        request.image_grid_thw_shm_name,
                        request.image_grid_thw_shape,
                        request.image_grid_thw_dtype
                    )
                except FileNotFoundError as e:
                    logger.warning(f"[VIT Scheduler] ⚠️ Shared memory not found for {request.request_id}: {e}")
                    with self._processing_lock:
                        self._processing_requests.discard(request.request_id)
                    continue

                # 计算 hash
                hash_val = self._compute_hash(pixel_values, image_grid_thw)

                # 查询缓存
                cached_embedding = self._get_cached_embedding(hash_val)

                if cached_embedding is not None:
                    # Cache hit
                    cache_hits.append((request, cached_embedding, hash_val))
                    logger.info(f"[VIT Scheduler] 🎯 Cache hit for {request.request_id}")
                else:
                    # Cache miss
                    cache_misses.append((request, pixel_values, image_grid_thw, hash_val))
                    logger.info(f"[VIT Scheduler] 🚀 Cache miss for {request.request_id}, will compute")

            except Exception as e:
                logger.error(f"[VIT Scheduler] ❌ Error preparing request {request.request_id}: {e}", exc_info=True)
                with self._processing_lock:
                    self._processing_requests.discard(request.request_id)

        # 🔧 批量计算 cache misses (带显存检查)
        if cache_misses:
            logger.info(f"[VIT Scheduler] 🔍 Processing {len(cache_misses)} cache misses with memory check")

            # ✅ 修复: 批量处理前强制清理 PyTorch 显存碎片
            torch.cuda.empty_cache()
            logger.info(f"[VIT Scheduler] 🧹 Cleared CUDA cache before batch processing")

            # 检查是否可以处理这个批量大小
            cache_miss_batch_size = len(cache_misses)
            adjusted_batch_size = self._adjust_batch_size_for_memory(cache_miss_batch_size)

            logger.info(
                f"[VIT Scheduler] 📊 Batch size adjustment result: "
                f"requested={cache_miss_batch_size}, adjusted={adjusted_batch_size}"
            )

            if adjusted_batch_size == 0:
                # 显存不足,无法处理任何请求
                logger.warning(
                    f"[VIT Scheduler] ⚠️ 显存不足，无法处理任何请求。"
                    f"等待显存释放..."
                )

                # ✅ 不调用 torch.cuda.empty_cache()，让 Python GC 自动处理

                # 清理所有请求的 processing 标记
                for request, _, _, _ in cache_misses:
                    with self._processing_lock:
                        self._processing_requests.discard(request.request_id)
                # 将请求放回队列
                self.pending_requests = [req for req, _, _, _ in cache_misses] + remaining_requests
                return

            elif adjusted_batch_size < cache_miss_batch_size:
                # 需要减小批量大小
                logger.warning(
                    f"[VIT Scheduler] 📉 Adjusting cache miss batch size: {cache_miss_batch_size} -> {adjusted_batch_size} "
                    f"(memory constraint)"
                )
                # 只处理前 adjusted_batch_size 个请求
                cache_misses_to_process = cache_misses[:adjusted_batch_size]
                cache_misses_deferred = cache_misses[adjusted_batch_size:]

                # 将延迟的请求放回队列
                deferred_requests = [req for req, _, _, _ in cache_misses_deferred]
                self.pending_requests = deferred_requests + remaining_requests

                # ✅ 修复: 添加详细日志，显示剩余请求重新入队
                logger.warning(
                    f"[VIT Scheduler] ⚠️ 剩余 {len(deferred_requests)} 个请求重新入队 "
                    f"(request_ids: {[req.request_id for req in deferred_requests[:3]]}{'...' if len(deferred_requests) > 3 else ''})"
                )

                # 清理延迟请求的 processing 标记
                for request, _, _, _ in cache_misses_deferred:
                    with self._processing_lock:
                        self._processing_requests.discard(request.request_id)

                # 处理调整后的批量
                self._process_cache_misses_batch(cache_misses_to_process)
            else:
                # 批量大小合适,直接处理
                self._process_cache_misses_batch(cache_misses)

        # 🔧 处理 cache hits
        for request, cached_embedding, hash_val in cache_hits:
            try:
                self._send_cached_response(request, cached_embedding, hash_val)
            except Exception as e:
                logger.error(f"[VIT Scheduler] ❌ Error sending cached response for {request.request_id}: {e}", exc_info=True)
            finally:
                with self._processing_lock:
                    self._processing_requests.discard(request.request_id)

        # 🔧 将剩余请求放回队列
        self.pending_requests = remaining_requests

        batch_time = time.time() - batch_start_time
        logger.info(
            f"[VIT Scheduler] ✅ Batch processed in {batch_time*1000:.1f}ms, "
            f"cache_hits={len(cache_hits)}, cache_misses={len(cache_misses)}, "
            f"remaining in queue: {len(self.pending_requests)}"
        )

        # 定期打印统计信息
        if self.total_requests % 100 == 0:
            self._log_stats()

        # 🔧 新增: 如果队列不为空，立即重新触发批量处理
        if self.pending_requests:
            logger.info(
                f"[VIT Scheduler] 🔄 Queue not empty ({len(self.pending_requests)} requests), "
                f"scheduling next batch processing in 10ms"
            )
            # 使用 threading.Timer 异步触发，避免递归
            # 延迟 10ms，给系统一点时间处理其他事情（例如显存释放）
            threading.Timer(0.01, self._process_batch).start()
        else:
            logger.debug(f"[VIT Scheduler] ✅ 队列为空，批量处理完成")

        # ✅ 不调用 torch.cuda.empty_cache()，让 Python GC 自动处理

        # 🔧 定期清理超时的共享内存（每 10 个批次）
        if self.total_requests % 10 == 0:
            self._cleanup_timeout_shm()

    def _process_cache_misses_batch(self, cache_misses: List[Tuple]):
        """批量处理 cache miss 的请求

        ✅ 参考 SGLang 设计: 批量计算前分配显存预算，计算后释放预算

        Args:
            cache_misses: List of (request, pixel_values, image_grid_thw, hash_val)
        """
        if not cache_misses:
            return

        batch_size = len(cache_misses)
        logger.info(f"[VIT Scheduler] 📦 批量处理 {batch_size} 个 cache miss...")

        # ✅✅✅ 核心重新设计: 从 embedding 页面池分配页面
        # 参考: sglang/docs/batch_scheduler_memory.md 第 4 节 (prepare_for_extend 的页面分配)
        estimated_pages = self._estimate_batch_pages(batch_size)
        allocated_pages = self.embedding_page_pool.alloc(estimated_pages)

        if allocated_pages is None:
            pool_stats = self.embedding_page_pool.get_stats()
            logger.error(
                f"[VIT Scheduler] ❌ Embedding 页面池分配失败: "
                f"需求 {estimated_pages} pages, "
                f"可用 {pool_stats['free_pages']} pages"
            )
            return

        pool_stats = self.embedding_page_pool.get_stats()
        logger.info(
            f"[VIT Scheduler] 💾 Embedding 页面池分配成功: "
            f"{len(allocated_pages)} pages (需求 {estimated_pages}), "
            f"已用: {pool_stats['used_pages']} pages, "
            f"使用率: {pool_stats['usage_ratio']*100:.1f}%"
        )

        # 只有 TP rank 0 需要处理
        if self.tp_rank == 0:
            # 提取数据
            requests = [item[0] for item in cache_misses]
            pixel_values_list = [item[1] for item in cache_misses]
            image_grid_thw_list = [item[2] for item in cache_misses]
            hash_vals = [item[3] for item in cache_misses]

            try:
                # 🔧 VIT TP: Broadcast 批量数据给其他 TP ranks
                if self.tp_size > 1:
                    from sglang.srt.distributed.communication_op import broadcast_tensor_dict

                    logger.info(f"[VIT Scheduler] 📡 广播批量数据到其他 TP ranks...")
                    batch_data = {
                        "pixel_values_list": pixel_values_list,
                        "image_grid_thw_list": image_grid_thw_list,
                    }
                    broadcast_tensor_dict(tensor_dict=batch_data, src=0)
                    logger.info(f"[VIT Scheduler] ✅ 批量数据已广播")

                # 批量计算
                compute_start = time.time()
                embeddings = self.model_runner.compute_batch(pixel_values_list, image_grid_thw_list)
                compute_end = time.time()
                compute_time = compute_end - compute_start

                logger.info(
                    f"[VIT Scheduler] ✅ 批量计算完成: "
                    f"耗时 {compute_time*1000:.1f}ms, "
                    f"平均 {compute_time/batch_size*1000:.1f}ms/请求"
                )

                # 🔧 并行性分析: 记录批量计算的时间范围
                logger.info(
                    f"[VIT Scheduler] 📊 并行性分析: "
                    f"VIT 批量计算时间范围: [{compute_start:.6f}, {compute_end:.6f}], "
                    f"持续时间: {compute_time:.6f}s"
                )

                # 处理每个结果
                for i, (request, embedding, hash_val) in enumerate(zip(requests, embeddings, hash_vals)):
                    try:
                        # 更新缓存
                        cache_add_time = time.time()
                        self._update_cache(hash_val, embedding)

                        # 🔧 缓存生命周期跟踪: 记录缓存添加时间
                        logger.info(
                            f"[VIT Scheduler] 🕐 缓存生命周期 - 添加: "
                            f"request_id={request.request_id}, "
                            f"hash={hash_val}, "
                            f"添加时间={cache_add_time:.6f}"
                        )

                        # 创建 CUDA IPC handle
                        from sglang.semi_pd.utils import get_ipc_handle
                        ipc_handle, offset = get_ipc_handle(embedding)

                        # 构造响应
                        response = VITResponse(
                            request_id=request.request_id,
                            embedding_ipc_handle=(ipc_handle, offset),
                            embedding_shape=tuple(embedding.shape),
                            embedding_dtype=str(embedding.dtype).replace('torch.', ''),
                            embedding_device=str(embedding.device),
                            image_hash=hash_val,
                            compute_time=compute_time / batch_size,  # 平均时间
                            from_cache=False,
                            vit_compute_start_time=compute_start,
                            vit_compute_end_time=compute_end,
                        )

                        # 发送响应
                        response_send_time = time.time()
                        self._send_response(response)

                        # 🔧 并行性分析: 记录响应发送时间
                        logger.info(
                            f"[VIT Scheduler] 📊 并行性分析 - 响应发送: "
                            f"request_id={request.request_id}, "
                            f"发送时间={response_send_time:.6f}, "
                            f"VIT 计算结束到发送延迟={(response_send_time - compute_end)*1000:.2f}ms"
                        )

                        logger.info(f"[VIT Scheduler] ✅ 批量请求 {i+1}/{batch_size} 已处理: {request.request_id}")

                    except Exception as e:
                        logger.error(f"[VIT Scheduler] ❌ Error processing batch request {request.request_id}: {e}", exc_info=True)
                    finally:
                        with self._processing_lock:
                            self._processing_requests.discard(request.request_id)

            except RuntimeError as e:
                # 🔧 OOM 或其他 CUDA 错误: 降级到单个请求处理
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    # 记录 OOM 统计
                    self._oom_count += 1
                    self._last_oom_time = time.time()
                    self._last_oom_batch_size = batch_size
                    self._consecutive_oom_count += 1

                    logger.warning(
                        f"[VIT Scheduler] ⚠️ 批量计算失败 (OOM/CUDA 错误), 降级到单请求处理: {e}"
                    )
                    logger.warning(
                        f"[VIT Scheduler] 📊 OOM 统计: count={self._oom_count}, "
                        f"batch_size={batch_size}, "
                        f"当前预算占用={self._current_batch_memory_bytes / (1024**3):.2f} GB, "
                        f"连续 OOM 次数={self._consecutive_oom_count}"
                    )

                    # ✅ 新增: OOM 自适应调整 - 增加单请求显存预算估算
                    if self._consecutive_oom_count >= 2:
                        # 连续 2 次 OOM，显存需求估算严重偏低
                        old_multiplier = self._per_request_memory_multiplier
                        self._per_request_memory_multiplier *= 1.5  # 增加 50%
                        logger.warning(
                            f"[VIT Scheduler] 🔧 OOM 自适应调整: 连续 {self._consecutive_oom_count} 次 OOM, "
                            f"增加显存需求倍数: {old_multiplier:.2f} -> {self._per_request_memory_multiplier:.2f}"
                        )
                        logger.warning(
                            f"[VIT Scheduler] 📊 新的单请求显存预算: "
                            f"{self._per_request_memory_bytes * self._per_request_memory_multiplier / (1024**3):.2f} GB"
                        )

                    # ❌ 不调用 torch.cuda.empty_cache()
                    # 让 Python GC 自动处理

                    # 逐个处理请求
                    for request, pixel_values, image_grid_thw, hash_val in cache_misses:
                        try:
                            self._process_single_request_fallback(request, pixel_values, image_grid_thw, hash_val)
                        except Exception as single_e:
                            logger.error(f"[VIT Scheduler] ❌ 单请求降级也失败: {request.request_id}: {single_e}", exc_info=True)
                        finally:
                            with self._processing_lock:
                                self._processing_requests.discard(request.request_id)
                else:
                    # 其他错误: 清理所有请求
                    logger.error(f"[VIT Scheduler] ❌ 批量计算失败 (非 OOM 错误): {e}", exc_info=True)
                    for request, _, _, _ in cache_misses:
                        with self._processing_lock:
                            self._processing_requests.discard(request.request_id)
                    raise
            finally:
                # ✅✅✅ 核心重新设计: 释放 embedding 页面池
                # 参考: sglang/docs/batch_scheduler_memory.md 第 7 节 (free_group_end 的成组释放)
                self.embedding_page_pool.free(allocated_pages)
                pool_stats = self.embedding_page_pool.get_stats()
                logger.info(
                    f"[VIT Scheduler] 💾 Embedding 页面池释放成功: "
                    f"{len(allocated_pages)} pages, "
                    f"已用: {pool_stats['used_pages']} pages, "
                    f"使用率: {pool_stats['usage_ratio']*100:.1f}%"
                )

                # ✅ 修复: 批量处理后强制清理 PyTorch 显存碎片
                torch.cuda.empty_cache()
                logger.info(f"[VIT Scheduler] 🧹 Cleared CUDA cache after batch processing")

        else:
            # TP rank > 0: 接收 broadcast 并参与计算
            try:
                from sglang.srt.distributed.communication_op import broadcast_tensor_dict

                # 接收 broadcast
                batch_data = broadcast_tensor_dict(tensor_dict=None, src=0)

                if batch_data is None:
                    logger.error(f"[VIT Scheduler] ❌ TP rank {self.tp_rank}: 接收批量数据失败")
                    return

                pixel_values_list = batch_data.get("pixel_values_list")
                image_grid_thw_list = batch_data.get("image_grid_thw_list")

                # 批量计算（参与 all-reduce）
                embeddings = self.model_runner.compute_batch(pixel_values_list, image_grid_thw_list)

                logger.info(f"[VIT Scheduler] ✅ TP rank {self.tp_rank}: 批量计算完成")

            except RuntimeError as e:
                # OOM 或 CUDA 错误
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    logger.warning(f"[VIT Scheduler] ⚠️ TP rank {self.tp_rank}: 批量计算失败 (OOM/CUDA 错误): {e}")
                    # ❌ 不调用 torch.cuda.empty_cache()
                else:
                    logger.error(f"[VIT Scheduler] ❌ TP rank {self.tp_rank}: 批量计算失败: {e}", exc_info=True)
                    raise
            finally:
                # 清理 processing 标记
                for request, _, _, _ in cache_misses:
                    with self._processing_lock:
                        self._processing_requests.discard(request.request_id)

                # ✅✅✅ 核心重新设计: 释放 embedding 页面池 (TP rank > 0)
                self.embedding_page_pool.free(allocated_pages)
                logger.info(
                    f"[VIT Scheduler] 💾 Embedding 页面池释放成功 (TP rank {self.tp_rank}): "
                    f"{len(allocated_pages)} pages"
                )

    def _process_single_request_fallback(self, request: VITRequest, pixel_values: torch.Tensor,
                                          image_grid_thw: torch.Tensor, hash_val: str):
        """🔧 单个请求处理 (OOM 降级时使用)

        Args:
            request: VIT 请求
            pixel_values: 图像像素值
            image_grid_thw: 图像网格尺寸
            hash_val: 图像哈希值
        """
        logger.info(f"[VIT Scheduler] 🔄 Processing single request (fallback): {request.request_id}")

        # 🔧 VIT TP: Broadcast 数据给其他 TP ranks
        if self.tp_size > 1:
            from sglang.srt.distributed.communication_op import broadcast_tensor_dict

            logger.info(f"[VIT Scheduler] 📡 Broadcasting single request data to other TP ranks...")
            single_data = {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
            broadcast_tensor_dict(tensor_dict=single_data, src=0)

        # 单个计算
        compute_start = time.time()
        # ✅ 修复: VITModelRunner 没有 forward 方法，应该使用 compute_batch
        embeddings = self.model_runner.compute_batch([pixel_values], [image_grid_thw])
        embedding = embeddings[0]
        compute_time = time.time() - compute_start

        # 更新缓存
        self._update_cache(hash_val, embedding)

        # 创建 CUDA IPC handle
        from sglang.semi_pd.utils import get_ipc_handle
        ipc_handle, offset = get_ipc_handle(embedding)

        # 构造响应
        response = VITResponse(
            request_id=request.request_id,
            embedding_ipc_handle=(ipc_handle, offset),
            embedding_shape=tuple(embedding.shape),
            embedding_dtype=str(embedding.dtype).replace('torch.', ''),
            embedding_device=str(embedding.device),
            image_hash=hash_val,
            compute_time=compute_time,
            from_cache=False,
        )

        # 发送响应
        self._send_response(response)

        logger.info(f"[VIT Scheduler] ✅ Single request (fallback) processed: {request.request_id}, compute_time={compute_time*1000:.1f}ms")

    def _send_cached_response(self, request: VITRequest, cached_embedding: torch.Tensor, hash_val: int):
        """发送缓存的响应"""
        # 创建 CUDA IPC handle
        from sglang.semi_pd.utils import get_ipc_handle
        ipc_handle, offset = get_ipc_handle(cached_embedding)

        # 🔧 缓存命中: 记录时间
        cache_hit_time = time.time()

        # 构造响应
        response = VITResponse(
            request_id=request.request_id,
            embedding_ipc_handle=(ipc_handle, offset),
            embedding_shape=tuple(cached_embedding.shape),
            embedding_dtype=str(cached_embedding.dtype).replace('torch.', ''),
            embedding_device=str(cached_embedding.device),
            image_hash=hash_val,
            compute_time=0.0,
            from_cache=True,
            vit_compute_start_time=cache_hit_time,  # 缓存命中时间
            vit_compute_end_time=cache_hit_time,
        )

        # 发送响应
        self._send_response(response)

        # 🔧 缓存生命周期跟踪: 记录缓存命中
        logger.info(
            f"[VIT Scheduler] 🕐 缓存生命周期 - 命中: "
            f"request_id={request.request_id}, "
            f"hash={hash_val}, "
            f"命中时间={cache_hit_time:.6f}"
        )

        logger.info(f"[VIT Scheduler] ✅ 缓存响应已发送: {request.request_id}")

    def _send_response(self, response: VITResponse):
        """发送响应到 ZMQ socket"""
        message_data = pickle.dumps(asdict(response))

        logger.info(
            f"[VIT Scheduler] 📤 Sending response: {response.request_id}, "
            f"message_size={len(message_data)} bytes, from_cache={response.from_cache}"
        )

        try:
            self.socket.send(message_data, zmq.NOBLOCK)
            time.sleep(0.001)  # 1ms
            logger.info(f"[VIT Scheduler] ✅ Response sent: {response.request_id}")
        except zmq.Again:
            logger.error(f"[VIT Scheduler] ❌ Send would block!")
            self.socket.send(message_data)
            logger.info(f"[VIT Scheduler] ✅ Sent with blocking mode")

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
                        # 只发送 cache_hit 标记，不需要其他数据
                        cache_hit_signal = {
                            "cache_hit": True,
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
                        # 只发送 VIT 计算必需的数据: pixel_values 和 image_grid_thw
                        request_data = {
                            "pixel_values": pixel_values,
                            "image_grid_thw": image_grid_thw,
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

    def _evict_cache(self, required_bytes: int) -> bool:
        """LRU 缓存驱逐

        ✅ 参考 SGLang 设计: 只驱逐缓存，不调用 torch.cuda.empty_cache()

        Args:
            required_bytes: 需要的显存大小 (bytes)

        Returns:
            是否成功驱逐足够的空间
        """
        logger.info(
            f"[VIT Scheduler] 🗑️ 开始 LRU 驱逐: "
            f"需求={required_bytes / (1024**2):.1f} MB, "
            f"当前缓存={self.cache_size_bytes / (1024**2):.1f} MB, "
            f"最大缓存={self.max_cache_size_bytes / (1024**2):.1f} MB"
        )

        evicted_count = 0
        evicted_bytes = 0

        # LRU 驱逐：移除最久未使用的条目
        while self.cache_size_bytes + required_bytes > self.max_cache_size_bytes:
            if len(self.embedding_cache) == 0:
                logger.warning(
                    f"[VIT Scheduler] ⚠️ 缓存已空，但仍需 "
                    f"{(required_bytes - (self.max_cache_size_bytes - self.cache_size_bytes)) / (1024**2):.1f} MB"
                )
                return False

            # 驱逐最久未使用的条目
            oldest_hash, oldest_emb = self.embedding_cache.popitem(last=False)
            evicted_size = oldest_emb.element_size() * oldest_emb.nelement()
            self.cache_size_bytes -= evicted_size
            evicted_count += 1
            evicted_bytes += evicted_size

            logger.info(
                f"[VIT Scheduler] 🗑️ 驱逐 embedding (LRU): hash={oldest_hash}, "
                f"size={evicted_size / (1024**2):.1f} MB"
            )
            # ✅ 不调用 torch.cuda.empty_cache()，让 Python GC 自动处理

        logger.info(
            f"[VIT Scheduler] ✅ LRU 驱逐完成: "
            f"驱逐了 {evicted_count} 个条目, "
            f"释放了 {evicted_bytes / (1024**2):.1f} MB, "
            f"剩余缓存 {self.cache_size_bytes / (1024**2):.1f} MB"
        )
        return True

    def _update_cache(self, hash_val: int, embedding: torch.Tensor):
        """更新缓存（LRU 策略，线程安全，保持 embedding 在 GPU 上）

        ✅ 参考 SGLang 设计: 使用 LRU 驱逐，不调用 torch.cuda.empty_cache()

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
                logger.debug(f"[VIT Scheduler] 🔄 缓存命中 (LRU 更新): hash={hash_val}")
                return

            embedding_size = embedding.element_size() * embedding.nelement()

            # 🔧 CUDA IPC: 确保 embedding 在 GPU 上
            if not embedding.is_cuda:
                logger.warning(
                    f"[VIT Scheduler] ⚠️ Embedding 不在 GPU 上 (device={embedding.device}), "
                    "移动到 GPU 以支持 CUDA IPC 共享"
                )
                embedding = embedding.to(self.device)

            # 检查是否需要驱逐
            if self.cache_size_bytes + embedding_size > self.max_cache_size_bytes:
                if not self._evict_cache(embedding_size):
                    logger.warning(
                        f"[VIT Scheduler] ⚠️ 无法驱逐足够的缓存空间，跳过缓存此 embedding"
                    )
                    return

            # 添加到缓存（保持在 GPU 上）
            self.embedding_cache[hash_val] = embedding
            self.cache_size_bytes += embedding_size

            logger.info(
                f"[VIT Scheduler] 💾 添加到缓存: hash={hash_val}, "
                f"size={embedding_size / (1024**2):.1f} MB, "
                f"总缓存大小={self.cache_size_bytes / (1024**2):.1f} MB"
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
        """释放缓存（事件驱动，线程安全）"""
        free_time = time.time()
        with self._cache_lock:
            if hash_val in self.embedding_cache:
                embedding = self.embedding_cache.pop(hash_val)
                freed_size = embedding.element_size() * embedding.nelement()
                self.cache_size_bytes -= freed_size

                # 🔧 缓存生命周期跟踪: 记录缓存释放
                logger.info(
                    f"[VIT Scheduler] 🕐 缓存生命周期 - 释放: "
                    f"hash={hash_val}, "
                    f"size={freed_size / (1024**2):.2f} MB, "
                    f"释放时间={free_time:.6f}"
                )

                logger.info(
                    f"[VIT Scheduler] 🗑️ 释放 embedding (事件驱动): "
                    f"hash={hash_val}, "
                    f"size={freed_size / (1024**2):.2f} MB"
                )

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
                        request_id = request.request_id

                        # ✅ 优化: 请求去重，避免处理重复请求
                        # 检查是否已在队列中
                        if any(r.request_id == request_id for r in self.pending_requests):
                            logger.warning(
                                f"[VIT Scheduler] ⚠️ 忽略重复请求: {request_id} "
                                f"(已在队列中，pending={len(self.pending_requests)})"
                            )
                            continue

                        # 检查是否正在处理
                        if hasattr(self, '_processing_requests') and request_id in self._processing_requests:
                            logger.warning(
                                f"[VIT Scheduler] ⚠️ 忽略重复请求: {request_id} "
                                f"(正在处理中)"
                            )
                            continue

                        self.pending_requests.append(request)
                        self.total_requests += 1
                        logger.info(f"[VIT Scheduler] Received request (via drain): {request_id}")
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
                        request_id = request.request_id

                        # ✅ 优化: 请求去重，避免处理重复请求
                        # 检查是否已在队列中
                        if any(r.request_id == request_id for r in self.pending_requests):
                            logger.warning(
                                f"[VIT Scheduler] ⚠️ 忽略重复请求: {request_id} "
                                f"(已在队列中，pending={len(self.pending_requests)})"
                            )
                            continue

                        # 检查是否正在处理
                        if hasattr(self, '_processing_requests') and request_id in self._processing_requests:
                            logger.warning(
                                f"[VIT Scheduler] ⚠️ 忽略重复请求: {request_id} "
                                f"(正在处理中)"
                            )
                            continue

                        self.pending_requests.append(request)
                        self.total_requests += 1

                        logger.info(f"[VIT Scheduler] Received request: {request_id}")
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
        """🔧 VIT TP: TP rank > 0 的 worker 运行循环（支持批量 broadcast）

        TP rank > 0 不监听 ZMQ，而是等待 TP rank 0 的 broadcast。
        收到 broadcast 后，参与计算（RowParallelLinear 需要 all-reduce）。
        """
        logger.info(f"[VIT Scheduler] TP rank {self.tp_rank}: worker mode started (batch support)")

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

                # 2. 检查是否是批量数据
                pixel_values_list = request_data.get("pixel_values_list")
                image_grid_thw_list = request_data.get("image_grid_thw_list")

                if pixel_values_list is not None and image_grid_thw_list is not None:
                    # 批量计算
                    batch_size = len(pixel_values_list)
                    logger.info(
                        f"[VIT Scheduler] TP rank {self.tp_rank}: received batch broadcast, "
                        f"batch_size={batch_size}"
                    )

                    try:
                        embeddings = self.model_runner.compute_batch(pixel_values_list, image_grid_thw_list)
                        logger.info(
                            f"[VIT Scheduler] TP rank {self.tp_rank}: batch computation completed, "
                            f"batch_size={batch_size}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[VIT Scheduler] TP rank {self.tp_rank}: error in batch compute: {e}",
                            exc_info=True
                        )
                else:
                    # 单个请求计算（向后兼容）
                    pixel_values = request_data.get("pixel_values")
                    image_grid_thw = request_data.get("image_grid_thw")

                    if pixel_values is None or image_grid_thw is None:
                        logger.warning(f"[VIT Scheduler] TP rank {self.tp_rank}: received invalid request data")
                        continue

                    logger.info(
                        f"[VIT Scheduler] TP rank {self.tp_rank}: received single broadcast, "
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

