"""
Asynchronous ViT worker implemented with a background thread and dedicated CUDA stream.

Key ideas:
1. The ViT model stays in the main process (leveraging SGLang optimisations).
2. A dedicated CUDA stream executes ViT computation asynchronously.
3. A thread-safe queue coordinates work submission and completion.
4. The main thread enqueues ViT jobs without blocking.
5. Prefill reads cached embeddings once the worker finishes.

Performance benefits:
- Reuses SGLang-optimised ViT features (FA3, quantisation, etc.).
- Overlaps ViT computation with LLM execution.
- Avoids interprocess communication overhead.
- Expected 2-3x throughput improvement.
"""

import threading
import time
import torch
from typing import Dict, Optional
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)


class ViTWorkerThread:
    """Background thread that processes ViT inference tasks."""

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

        # Create a dedicated CUDA stream for ViT work
        self.vit_stream = torch.cuda.Stream()

        # Start background worker thread
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

        logger.info(f"[ViT Worker Thread] Started on device {device}")

    def _worker_loop(self):
        """Main loop executed by the worker thread."""
        from queue import Empty

        while self.running:
            try:
                # Pull task from the queue (blocks up to 0.1s)
                task = self.task_queue.get(timeout=0.1)

                if task is None:  # Termination signal
                    logger.info("[ViT Worker Thread] Received termination signal")
                    break

                # Run ViT inference
                request_id = task['request_id']
                pixel_values = task['pixel_values']
                grid_thw = task['grid_thw']

                start_time = time.time()

                # Execute on the dedicated CUDA stream
                with torch.cuda.stream(self.vit_stream):
                    with torch.no_grad():
                        embedding = self.vit_model(pixel_values, grid_thw=grid_thw)

                # Synchronise stream to make sure work is done
                self.vit_stream.synchronize()

                compute_time = time.time() - start_time

                # Cache result for retrieval
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
        """Stop worker thread."""
        self.running = False
        self.task_queue.put(None)  # Signal termination
        self.thread.join(timeout=5)


class ViTWorkerManager:
    """Coordinator for the thread-based ViT worker."""

    def __init__(
        self,
        vit_model,  # Already-loaded ViT model
        device: str = "cuda:0",
        enable: bool = True,
    ):
        self.vit_model = vit_model
        self.device = device
        self.enable = enable

        if not self.enable:
            logger.info("[ViT Worker] ViT worker disabled")
            return

        # Create work queue and result cache
        from queue import Queue
        self.task_queue = Queue(maxsize=100)
        self.result_cache = {}  # Store outputs keyed by request id

        # Start worker thread
        self.worker_thread = ViTWorkerThread(
            vit_model=self.vit_model,
            device=self.device,
            result_cache=self.result_cache,
            task_queue=self.task_queue,
        )

        logger.info("[ViT Worker] ViT worker initialized (thread-based)")

        # Simple counters for observability
        self.submitted_count = 0
        self.completed_count = 0
    
    def submit_task(
        self,
        request_id: str,
        pixel_values: torch.Tensor,
        grid_thw: list,
    ) -> bool:
        """
        Submit a ViT inference task without blocking.

        Args:
            request_id: Identifier for the request.
            pixel_values: Tensor containing image pixels.
            grid_thw: Grid metadata for the image.

        Returns:
            Whether the task was enqueued successfully.
        """
        if not self.enable:
            return False

        try:
            # Build task payload
            task = {
                'request_id': request_id,
                'pixel_values': pixel_values,  # Pass tensor directly
                'grid_thw': grid_thw,
                'timestamp': time.time(),
            }

            # Non-blocking submit; drop immediately if the queue is full
            self.task_queue.put_nowait(task)
            self.submitted_count += 1

            logger.info(
                f"[ViT Worker] Submitted ViT task for request {request_id}, "
                f"queue_size={self.task_queue.qsize()}"
            )
            return True

        except Exception as e:
            # Catch all exceptions, including a full queue
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
        Retrieve a ViT result, blocking until it is ready or the timeout elapses.

        Args:
            request_id: Identifier for the request.
            timeout: Maximum time to wait in seconds.
            device: Target device (ignored; tensor already resides on the right device).

        Returns:
            Embedding tensor, or None on timeout.
        """
        if not self.enable:
            return None

        start_time = time.time()

        while time.time() - start_time < timeout:
            if request_id in self.result_cache:
                result = self.result_cache.pop(request_id)
                self.completed_count += 1

                # Already a tensor on the correct device
                embedding = result['embedding']

                wait_time = time.time() - start_time
                logger.info(
                    f"[ViT Worker] Retrieved ViT result for request {request_id}, "
                    f"wait_time={wait_time:.3f}s, compute_time={result['compute_time']:.3f}s"
                )
                return embedding

            time.sleep(0.001)  # 1ms polling interval

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
        Try to retrieve a ViT result without blocking.

        Returns:
            Embedding tensor, or None if the task is still running.
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
        """Shut down the ViT worker thread."""
        if not self.enable:
            return

        logger.info("[ViT Worker] Shutting down ViT worker...")

        # Stop worker thread
        self.worker_thread.stop()

        logger.info(
            f"[ViT Worker] ViT worker shutdown complete. "
            f"Stats: submitted={self.submitted_count}, completed={self.completed_count}"
        )

    def get_stats(self) -> Dict:
        """Return current queue statistics."""
        return {
            "submitted": self.submitted_count,
            "completed": self.completed_count,
            "pending": self.submitted_count - self.completed_count,
            "queue_size": self.task_queue.qsize() if self.enable else 0,
        }
