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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import threading

import torch
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
    """VIT 计算响应"""
    request_id: str
    embedding_shm_name: str  # 共享内存名称
    embedding_shape: Tuple[int, ...]
    embedding_dtype: str
    compute_time: float
    from_cache: bool


class VITModelRunner:
    """VIT 模型运行器"""

    def __init__(self, model_config, device: str = "cuda:0"):
        self.model_config = model_config
        self.device = device
        self.vit_model = None

        # 初始化 load_config（复用官方加载逻辑需要）
        from sglang.srt.configs.load_config import LoadConfig
        self.load_config = LoadConfig()
        
    def load_model(self):
        """加载 ViT 模型"""
        model_type = self.model_config.hf_config.model_type

        logger.info(f"[VIT Runner] Loading ViT model type: {model_type}")

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
        """计算 ViT embedding"""
        pixel_values = pixel_values.to(self.device)
        image_grid_thw = image_grid_thw.to(self.device)

        # 🔍 检查输入数据
        if torch.isnan(pixel_values).any():
            logger.error(f"[VIT Runner] ❌ Input pixel_values contains NaN!")
        if torch.isinf(pixel_values).any():
            logger.error(f"[VIT Runner] ❌ Input pixel_values contains Inf!")
        logger.info(f"[VIT Runner] 📊 Input pixel_values: shape={pixel_values.shape}, dtype={pixel_values.dtype}, min={pixel_values.min().item():.4f}, max={pixel_values.max().item():.4f}")
        logger.info(f"[VIT Runner] 📊 Input image_grid_thw: {image_grid_thw}")

        with torch.no_grad():
            embedding = self.vit_model(pixel_values, grid_thw=image_grid_thw)

        # 🔍 检查输出数据
        if torch.isnan(embedding).any():
            logger.error(f"[VIT Runner] ❌ Output embedding contains NaN!")
        if torch.isinf(embedding).any():
            logger.error(f"[VIT Runner] ❌ Output embedding contains Inf!")
        logger.info(f"[VIT Runner] 📊 Output embedding: shape={embedding.shape}, dtype={embedding.dtype}, min={embedding.min().item() if not torch.isnan(embedding).any() else 'nan'}, max={embedding.max().item() if not torch.isnan(embedding).any() else 'nan'}")

        return embedding


class VITScheduler:
    """
    VIT Scheduler - 独立的 ViT 计算调度器
    
    功能:
    1. 接收来自主 Scheduler 的 ViT 计算请求
    2. 批量计算 ViT
    3. 缓存 embedding
    4. 返回结果给主 Scheduler
    """
    
    def __init__(
        self,
        model_config,
        device: str = "cuda:0",
        zmq_port: int = 5555,
        batch_size: int = 4,
        batch_timeout_ms: float = 10.0,
        cache_size_mb: int = 1024,
    ):
        """
        Args:
            model_config: 模型配置
            device: ViT 运行的设备
            zmq_port: ZMQ 通信端口
            batch_size: 批量计算的最大 batch size
            batch_timeout_ms: 批量计算的超时时间（毫秒）
            cache_size_mb: Embedding 缓存大小（MB）
        """
        self.model_config = model_config
        self.device = device
        self.zmq_port = zmq_port
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0
        self.cache_size_mb = cache_size_mb
        
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

        # 初始化模型
        self.model_runner = VITModelRunner(model_config, device)
        self.model_runner.load_model()

        # 初始化缓存
        self.embedding_cache: Dict[int, torch.Tensor] = {}
        self.cache_size_bytes = 0
        self.max_cache_size_bytes = cache_size_mb * 1024 * 1024

        # 批处理队列
        self.pending_requests: List[VITRequest] = []  # 不再需要 client_id
        self.last_batch_time = time.time()
        
        # 统计信息
        self.total_requests = 0
        self.cache_hits = 0
        self.total_compute_time = 0.0
        
        logger.info("[VIT Scheduler] Initialized successfully")
    
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
    
    def _process_batch(self):
        """批量处理请求"""
        if not self.pending_requests:
            return

        batch_start_time = time.time()
        batch_size = len(self.pending_requests)

        logger.info(f"[VIT Scheduler] Processing batch of {batch_size} requests")

        # 逐个处理（后续可以优化为真正的批量计算）
        for request in self.pending_requests:
            try:
                response = self._process_single_request(request)

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

        # 清空队列
        self.pending_requests.clear()

        batch_time = time.time() - batch_start_time
        logger.info(f"[VIT Scheduler] Batch processed in {batch_time*1000:.1f}ms")

        # 定期打印统计信息
        if self.total_requests % 100 == 0:
            self._log_stats()

    def _process_single_request(self, request: VITRequest) -> VITResponse:
        """处理单个请求"""
        start_time = time.time()
        
        # 从共享内存加载输入
        pixel_values = self._load_tensor_from_shm(
            request.pixel_values_shm_name,
            request.pixel_values_shape,
            request.pixel_values_dtype,
        )
        image_grid_thw = self._load_tensor_from_shm(
            request.image_grid_thw_shm_name,
            request.image_grid_thw_shape,
            request.image_grid_thw_dtype,
        )
        
        # 计算 hash
        hash_val = self._compute_hash(pixel_values, image_grid_thw)
        
        # 查询缓存
        from_cache = False
        if hash_val in self.embedding_cache:
            embedding = self.embedding_cache[hash_val]
            from_cache = True
            self.cache_hits += 1
        else:
            # 计算 ViT
            embedding = self.model_runner.compute(pixel_values, image_grid_thw)

            # 🔍 检查 embedding 是否包含 NaN
            if torch.isnan(embedding).any():
                logger.error(f"[VIT Scheduler] ❌ Computed embedding contains NaN! shape={embedding.shape}")
            else:
                logger.info(f"[VIT Scheduler] ✅ Computed embedding is valid: shape={embedding.shape}, min={embedding.min().item():.4f}, max={embedding.max().item():.4f}")

            # 更新缓存
            self._update_cache(hash_val, embedding)

        # 保存到共享内存
        embedding_shm_name = f"vit_emb_{request.request_id}"
        self._save_tensor_to_shm(embedding, embedding_shm_name)

        # 🔍 检查保存到共享内存后是否仍然有效
        logger.info(f"[VIT Scheduler] 📤 Saved embedding to SHM: {embedding_shm_name}")
        
        compute_time = time.time() - start_time
        self.total_compute_time += compute_time
        
        # 构造响应
        response = VITResponse(
            request_id=request.request_id,
            embedding_shm_name=embedding_shm_name,
            embedding_shape=tuple(embedding.shape),
            embedding_dtype=str(embedding.dtype).replace('torch.', ''),
            compute_time=compute_time,
            from_cache=from_cache,
        )
        
        return response
    
    def _update_cache(self, hash_val: int, embedding: torch.Tensor):
        """更新缓存"""
        embedding_size = embedding.element_size() * embedding.nelement()
        
        # 如果缓存已满，移除最旧的条目
        while self.cache_size_bytes + embedding_size > self.max_cache_size_bytes:
            if len(self.embedding_cache) == 0:
                break
            old_hash = next(iter(self.embedding_cache))
            old_embedding = self.embedding_cache.pop(old_hash)
            self.cache_size_bytes -= old_embedding.element_size() * old_embedding.nelement()
        
        # 添加新条目
        self.embedding_cache[hash_val] = embedding.cpu()
        self.cache_size_bytes += embedding_size
    
    def run(self):
        """主循环（支持批量处理）"""
        logger.info("[VIT Scheduler] Starting main loop (batch mode)")

        try:
            while True:
                # 收集请求（非阻塞）
                while len(self.pending_requests) < self.batch_size:
                    try:
                        # PAIR socket 接收消息
                        message = self.socket.recv(zmq.NOBLOCK)

                        request_dict = pickle.loads(message)

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

                # 判断是否执行批量计算
                should_compute = (
                    len(self.pending_requests) >= self.batch_size or
                    (len(self.pending_requests) > 0 and
                     time.time() - self.last_batch_time > self.batch_timeout)
                )

                if should_compute:
                    self._process_batch()
                    self.last_batch_time = time.time()
                else:
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
    
    def cleanup(self):
        """清理资源"""
        logger.info("[VIT Scheduler] Cleaning up...")
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

