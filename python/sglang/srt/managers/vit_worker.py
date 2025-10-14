"""
ViT Worker: 使用线程池 + CUDA Stream 实现 ViT 异步计算

核心思想:
1. ViT 在主进程中加载（使用 SGLang 优化的模块）
2. 使用独立的 CUDA Stream 进行异步计算
3. 使用线程池管理任务队列
4. 主线程异步提交 ViT 任务，不等待结果
5. Prefill 时从缓存读取已完成的 embedding

性能优势:
- 使用 SGLang 优化的 ViT（FA3、量化等）
- ViT 计算与 LLM 并行
- 避免进程间通信开销
- 预期吞吐量提升 2-3 倍
"""

import threading
import time
import torch
from typing import Dict, Optional
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)


class ViTWorkerThread:
    """ViT 工作线程（在后台处理 ViT 计算）"""

    def __init__(
        self,
        vit_model,
        device: str,
        result_cache: Dict,
        task_queue: "Queue",
    ):
        self.vit_model = vit_model
        self.device = device
        self.result_cache = result_cache
        self.task_queue = task_queue
        self.running = True

        # 创建独立的 CUDA Stream
        self.vit_stream = torch.cuda.Stream()

        # 启动工作线程
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

        logger.info(f"[ViT Worker Thread] Started on device {device}")

    def _worker_loop(self):
        """工作线程主循环"""
        from queue import Empty

        while self.running:
            try:
                # 从队列获取任务（阻塞等待，超时 0.1s）
                task = self.task_queue.get(timeout=0.1)

                if task is None:  # 终止信号
                    logger.info("[ViT Worker Thread] Received termination signal")
                    break

                # 执行 ViT 计算
                request_id = task['request_id']
                pixel_values = task['pixel_values']
                grid_thw = task['grid_thw']

                start_time = time.time()

                # 在独立的 CUDA Stream 中执行
                with torch.cuda.stream(self.vit_stream):
                    with torch.no_grad():
                        embedding = self.vit_model(pixel_values, grid_thw=grid_thw)

                # 同步 stream，确保计算完成
                self.vit_stream.synchronize()

                compute_time = time.time() - start_time

                # 存储结果
                self.result_cache[request_id] = {
                    'embedding': embedding,
                    'timestamp': time.time(),
                    'compute_time': compute_time,
                }

                logger.info(
                    f"[ViT Worker Thread] Completed ViT for request {request_id}, "
                    f"compute_time={compute_time:.3f}s, embedding_shape={embedding.shape}"
                )

            except Empty:
                continue
            except Exception as e:
                logger.error(f"[ViT Worker Thread] Error processing task: {e}")
                import traceback
                traceback.print_exc()
                continue

    def stop(self):
        """停止工作线程"""
        self.running = False
        self.task_queue.put(None)  # 发送终止信号
        self.thread.join(timeout=5)


class ViTWorkerManager:
    """ViT Worker 管理器（主线程侧）"""

    def __init__(
        self,
        vit_model,  # 直接传入已加载的 ViT 模型
        device: str = "cuda:0",
        enable: bool = True,
    ):
        self.vit_model = vit_model
        self.device = device
        self.enable = enable

        if not self.enable:
            logger.info("[ViT Worker] ViT worker disabled")
            return

        # 创建任务队列和结果缓存
        from queue import Queue
        self.task_queue = Queue(maxsize=100)
        self.result_cache = {}  # 存储计算结果

        # 启动工作线程
        self.worker_thread = ViTWorkerThread(
            vit_model=self.vit_model,
            device=self.device,
            result_cache=self.result_cache,
            task_queue=self.task_queue,
        )

        logger.info("[ViT Worker] ViT worker initialized (thread-based)")

        # 统计信息
        self.submitted_count = 0
        self.completed_count = 0
    
    def submit_task(
        self,
        request_id: str,
        pixel_values: torch.Tensor,
        grid_thw: list,
    ) -> bool:
        """
        提交 ViT 计算任务（非阻塞）

        Args:
            request_id: 请求 ID
            pixel_values: 图片像素值 tensor
            grid_thw: 图片网格信息

        Returns:
            是否成功提交
        """
        if not self.enable:
            return False

        try:
            # 构造任务
            task = {
                'request_id': request_id,
                'pixel_values': pixel_values,  # 直接传递 tensor，不需要转换
                'grid_thw': grid_thw,
                'timestamp': time.time(),
            }

            # 非阻塞提交（如果队列满则立即返回 False）
            self.task_queue.put_nowait(task)
            self.submitted_count += 1

            logger.info(
                f"[ViT Worker] Submitted ViT task for request {request_id}, "
                f"queue_size={self.task_queue.qsize()}"
            )
            return True

        except Exception as e:
            # 捕获所有异常，包括队列满的情况
            if "Full" in str(type(e).__name__):
                logger.warning(f"[ViT Worker] Task queue is full, dropping task for {request_id}")
            else:
                logger.error(f"[ViT Worker] Failed to submit task: {e}")
            return False
    
    def get_result(
        self,
        request_id: str,
        timeout: float = 10.0,
        device: str = "cuda:0",
    ) -> Optional[torch.Tensor]:
        """
        获取 ViT 计算结果（阻塞等待）

        Args:
            request_id: 请求 ID
            timeout: 超时时间（秒）
            device: 目标设备（忽略，因为结果已经在正确的设备上）

        Returns:
            embedding tensor，如果超时则返回 None
        """
        if not self.enable:
            return None

        start_time = time.time()

        while time.time() - start_time < timeout:
            if request_id in self.result_cache:
                result = self.result_cache.pop(request_id)
                self.completed_count += 1

                # 结果已经是 torch tensor，直接返回
                embedding = result['embedding']

                wait_time = time.time() - start_time
                logger.info(
                    f"[ViT Worker] Retrieved ViT result for request {request_id}, "
                    f"wait_time={wait_time:.3f}s, compute_time={result['compute_time']:.3f}s"
                )
                return embedding

            time.sleep(0.001)  # 1ms 轮询间隔

        logger.warning(
            f"[ViT Worker] Timeout waiting for ViT result for request {request_id} "
            f"after {timeout}s"
        )
        return None
    
    def try_get_result(
        self,
        request_id: str,
        device: str = "cuda:0",
    ) -> Optional[torch.Tensor]:
        """
        尝试获取 ViT 结果（非阻塞）

        Returns:
            embedding tensor，如果未完成则返回 None
        """
        if not self.enable:
            return None

        if request_id in self.result_cache:
            result = self.result_cache.pop(request_id)
            self.completed_count += 1
            embedding = result['embedding']
            return embedding

        return None

    def shutdown(self):
        """关闭 ViT 工作线程"""
        if not self.enable:
            return

        logger.info("[ViT Worker] Shutting down ViT worker...")

        # 停止工作线程
        self.worker_thread.stop()

        logger.info(
            f"[ViT Worker] ViT worker shutdown complete. "
            f"Stats: submitted={self.submitted_count}, completed={self.completed_count}"
        )

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "submitted": self.submitted_count,
            "completed": self.completed_count,
            "pending": self.submitted_count - self.completed_count,
            "queue_size": self.task_queue.qsize() if self.enable else 0,
        }

