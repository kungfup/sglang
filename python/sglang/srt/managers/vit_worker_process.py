"""
ViT Worker Process - 独立进程中运行 ViT 计算

核心功能:
1. 在独立进程中加载和运行 ViT 模型
2. 接收来自 Scheduler 的 ViT 计算任务
3. 支持批量计算多个请求的 ViT
4. 通过共享内存返回 embedding 结果
5. 完全异步，不阻塞 Scheduler 主线程
"""

import os
import time
import logging
import hashlib
from typing import Dict, List, Optional, Tuple
from queue import Empty
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


@dataclass
class ViTTask:
    """ViT 计算任务"""
    request_id: str
    pixel_values: torch.Tensor  # [1, num_patches] or [batch, num_patches]
    image_grid_thw: torch.Tensor  # [1, 3] or [batch, 3]
    hash_val: Optional[int] = None  # 用于缓存查找
    submit_time: float = 0.0


@dataclass
class ViTResult:
    """ViT 计算结果"""
    request_id: str
    embedding: torch.Tensor  # [num_image_tokens, hidden_dim]
    compute_time: float
    from_cache: bool


class ViTWorkerProcess:
    """
    ViT Worker 进程
    
    在独立进程中运行，负责:
    1. 加载 ViT 模型
    2. 接收计算任务
    3. 批量计算 ViT
    4. 返回 embedding 结果
    """
    
    def __init__(
        self,
        model_config,
        device: str = "cuda:0",
        batch_size: int = 4,
        batch_timeout_ms: float = 10.0,  # 毫秒
        cache_size_mb: int = 1024,
        enable: bool = True,
    ):
        """
        Args:
            model_config: 模型配置对象
            device: ViT 运行的设备
            batch_size: 批量计算的最大 batch size
            batch_timeout_ms: 批量计算的超时时间（毫秒）
            cache_size_mb: Embedding 缓存大小（MB）
            enable: 是否启用 ViT Worker（False 则回退到同步计算）
        """
        self.model_config = model_config
        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0  # 转换为秒
        self.cache_size_mb = cache_size_mb
        self.enable = enable
        
        if not self.enable:
            logger.info("[ViT Worker] Disabled, will use synchronous ViT computation")
            return
        
        # 创建进程间通信队列
        mp_context = mp.get_context('spawn')  # 使用 spawn 模式，避免 CUDA 初始化问题
        self.task_queue = mp_context.Queue(maxsize=100)
        self.result_queue = mp_context.Queue(maxsize=100)
        self.control_queue = mp_context.Queue(maxsize=10)
        
        # 启动 Worker 进程
        self.process = mp_context.Process(
            target=self._worker_main,
            daemon=True,
        )
        self.process.start()
        
        logger.info(
            f"[ViT Worker] Started process PID={self.process.pid}, "
            f"device={device}, batch_size={batch_size}, "
            f"batch_timeout={batch_timeout_ms}ms"
        )
        
        # 统计信息
        self.submitted_count = 0
        self.completed_count = 0
        self.cache_hit_count = 0
    
    def _worker_main(self):
        """Worker 进程主函数"""
        try:
            # 设置日志
            logging.basicConfig(
                level=logging.INFO,
                format='[%(asctime)s] [ViT Worker] %(message)s',
            )
            
            logger.info(f"Worker process started, PID={os.getpid()}")
            
            # 加载 ViT 模型
            logger.info(f"Loading ViT model on {self.device}...")
            self.vit_model = self._load_vit_model()
            logger.info("ViT model loaded successfully")
            
            # 初始化缓存
            self.embedding_cache: Dict[int, torch.Tensor] = {}
            self.cache_size_bytes = 0
            self.max_cache_size_bytes = self.cache_size_mb * 1024 * 1024
            
            # 批处理缓冲区
            self.pending_tasks: List[ViTTask] = []
            self.last_batch_time = time.time()
            
            logger.info("Ready to process ViT tasks")
            
            # 主循环
            while True:
                try:
                    # 检查控制命令
                    try:
                        cmd = self.control_queue.get_nowait()
                        if cmd == 'shutdown':
                            logger.info("Received shutdown command")
                            break
                    except Empty:
                        pass
                    
                    # 获取任务（带超时）
                    try:
                        task_dict = self.task_queue.get(timeout=self.batch_timeout)
                        task = ViTTask(**task_dict)
                        self.pending_tasks.append(task)
                    except Empty:
                        pass
                    
                    # 判断是否执行 batch 计算
                    should_compute = (
                        len(self.pending_tasks) >= self.batch_size or
                        (len(self.pending_tasks) > 0 and 
                         time.time() - self.last_batch_time > self.batch_timeout)
                    )
                    
                    if should_compute:
                        self._batch_compute_and_send()
                        self.last_batch_time = time.time()
                        
                except Exception as e:
                    logger.error(f"Error in worker loop: {e}", exc_info=True)
                    
        except Exception as e:
            logger.error(f"Fatal error in worker process: {e}", exc_info=True)
        finally:
            logger.info("Worker process exiting")
    
    def _load_vit_model(self):
        """加载 ViT 模型"""
        # 根据模型类型加载对应的 ViT
        model_type = self.model_config.hf_config.model_type
        
        if model_type == "qwen2_5_vl":
            from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer
            vit_model = Qwen2_5_VisionTransformer(
                self.model_config.hf_config.vision_config,
                norm_eps=getattr(self.model_config.hf_config, "rms_norm_eps", 1e-6),
            )
        elif model_type == "qwen2_vl":
            from sglang.srt.models.qwen2_vl import Qwen2VisionTransformer
            vit_model = Qwen2VisionTransformer(
                self.model_config.hf_config.vision_config,
                norm_eps=getattr(self.model_config.hf_config, "rms_norm_eps", 1e-6),
            )
        else:
            raise ValueError(f"Unsupported model type for ViT Worker: {model_type}")
        
        # 加载权重（从主进程传递的路径）
        # TODO: 实现权重加载逻辑
        
        vit_model.to(self.device)
        vit_model.eval()
        
        return vit_model
    
    def _compute_hash(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> int:
        """计算输入的 hash 值，用于缓存查找"""
        # 使用 tensor 的内容计算 hash
        hash_input = (
            pixel_values.cpu().numpy().tobytes() +
            image_grid_thw.cpu().numpy().tobytes()
        )
        return int(hashlib.md5(hash_input).hexdigest()[:16], 16)
    
    def _batch_compute_and_send(self):
        """批量计算并发送结果"""
        if len(self.pending_tasks) == 0:
            return
        
        batch_start_time = time.time()
        
        # 分离缓存命中和未命中的任务
        cache_hit_tasks = []
        cache_miss_tasks = []
        
        for task in self.pending_tasks:
            if task.hash_val is None:
                task.hash_val = self._compute_hash(task.pixel_values, task.image_grid_thw)
            
            if task.hash_val in self.embedding_cache:
                cache_hit_tasks.append(task)
            else:
                cache_miss_tasks.append(task)
        
        # 处理缓存命中
        for task in cache_hit_tasks:
            embedding = self.embedding_cache[task.hash_val]
            result = ViTResult(
                request_id=task.request_id,
                embedding=embedding,
                compute_time=0.0,
                from_cache=True,
            )
            self.result_queue.put(result.__dict__)
        
        # 批量计算缓存未命中的任务
        if len(cache_miss_tasks) > 0:
            # 拼接输入
            pixel_values_list = [t.pixel_values for t in cache_miss_tasks]
            grid_thw_list = [t.image_grid_thw for t in cache_miss_tasks]
            
            pixel_values_batch = torch.cat(pixel_values_list, dim=0).to(self.device)
            grid_thw_batch = torch.cat(grid_thw_list, dim=0).to(self.device)
            
            # 批量计算 ViT
            compute_start_time = time.time()
            with torch.no_grad():
                embeddings_batch = self.vit_model(pixel_values_batch, grid_thw=grid_thw_batch)
            compute_time = time.time() - compute_start_time
            
            # 拆分结果
            split_sizes = [pv.shape[0] for pv in pixel_values_list]
            embeddings_list = torch.split(embeddings_batch, split_sizes, dim=0)
            
            # 发送结果并更新缓存
            for task, embedding in zip(cache_miss_tasks, embeddings_list):
                # 更新缓存
                self._update_cache(task.hash_val, embedding)
                
                # 发送结果
                result = ViTResult(
                    request_id=task.request_id,
                    embedding=embedding.cpu(),  # 移到 CPU 以便跨进程传输
                    compute_time=compute_time / len(cache_miss_tasks),
                    from_cache=False,
                )
                self.result_queue.put(result.__dict__)
        
        batch_time = time.time() - batch_start_time
        logger.info(
            f"Processed batch: {len(self.pending_tasks)} tasks "
            f"({len(cache_hit_tasks)} cache hits, {len(cache_miss_tasks)} cache misses), "
            f"batch_time={batch_time*1000:.1f}ms"
        )
        
        # 清空缓冲区
        self.pending_tasks = []
    
    def _update_cache(self, hash_val: int, embedding: torch.Tensor):
        """更新 embedding 缓存"""
        # 计算 embedding 大小
        embedding_size = embedding.element_size() * embedding.nelement()
        
        # 如果缓存已满，移除最旧的条目（简单 FIFO 策略）
        while self.cache_size_bytes + embedding_size > self.max_cache_size_bytes:
            if len(self.embedding_cache) == 0:
                break
            # 移除第一个条目
            old_hash = next(iter(self.embedding_cache))
            old_embedding = self.embedding_cache.pop(old_hash)
            self.cache_size_bytes -= old_embedding.element_size() * old_embedding.nelement()
        
        # 添加新条目
        self.embedding_cache[hash_val] = embedding.cpu()  # 存储在 CPU 上
        self.cache_size_bytes += embedding_size
    
    # ========== 主进程侧接口 ==========
    
    def submit_task(
        self,
        request_id: str,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> bool:
        """
        提交 ViT 计算任务（非阻塞）
        
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
            task = ViTTask(
                request_id=request_id,
                pixel_values=pixel_values.cpu(),  # 移到 CPU 以便跨进程传输
                image_grid_thw=image_grid_thw.cpu(),
                submit_time=time.time(),
            )
            self.task_queue.put_nowait(task.__dict__)
            self.submitted_count += 1
            return True
        except:
            logger.warning(f"[ViT Worker] Task queue full, dropping task {request_id}")
            return False
    
    def try_get_result(self, request_id: str) -> Optional[torch.Tensor]:
        """
        非阻塞查询结果
        
        Args:
            request_id: 请求 ID
        
        Returns:
            embedding tensor，如果未就绪则返回 None
        """
        if not self.enable:
            return None
        
        try:
            result_dict = self.result_queue.get_nowait()
            result = ViTResult(**result_dict)
            
            if result.request_id == request_id:
                self.completed_count += 1
                if result.from_cache:
                    self.cache_hit_count += 1
                return result.embedding
            else:
                # 不是我们要的结果，放回队列
                self.result_queue.put(result_dict)
                return None
        except Empty:
            return None
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'submitted': self.submitted_count,
            'completed': self.completed_count,
            'cache_hits': self.cache_hit_count,
            'cache_hit_rate': self.cache_hit_count / max(self.completed_count, 1),
            'queue_size': self.task_queue.qsize() if self.enable else 0,
        }
    
    def shutdown(self):
        """关闭 Worker 进程"""
        if not self.enable:
            return
        
        logger.info("[ViT Worker] Shutting down...")
        self.control_queue.put('shutdown')
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            logger.warning("[ViT Worker] Process did not exit gracefully, terminating...")
            self.process.terminate()
            self.process.join()
        logger.info("[ViT Worker] Shutdown complete")

