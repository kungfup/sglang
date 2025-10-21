"""
ViT Memory Pool - 显存池管理

对齐 LightLLM 的显存管理机制，防止 ViT Worker 在高并发时 OOM。

功能:
1. 限制 Worker 显存使用
2. 估算 batch forward 所需显存
3. 显存不足时拒绝请求（backpressure）
4. 自动释放显存

使用示例:
```python
# 初始化显存池
memory_pool = VITMemoryPool(max_memory_gb=10.0)

# 估算显存需求
estimated_memory = sum(pv.nelement() * pv.element_size() for pv in pixel_values_list)

# 检查显存池
if not memory_pool.can_allocate(estimated_memory):
    return {"error": True, "error_message": "Memory pool full"}

# 分配显存
memory_pool.allocate(estimated_memory)

try:
    embeddings = model_runner.compute_batch(...)
finally:
    memory_pool.release(estimated_memory)
```
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class VITMemoryPool:
    """
    ViT 显存池管理器
    
    对齐 LightLLM 的显存管理机制:
    - 限制 Worker 显存使用
    - 估算 batch forward 所需显存
    - 显存不足时拒绝请求（backpressure）
    - 自动释放显存
    
    Args:
        max_memory_gb: 最大显存限制（GB）
        enable_monitoring: 是否启用监控日志
        monitoring_interval: 监控日志间隔（秒）
    """
    
    def __init__(
        self,
        max_memory_gb: float,
        enable_monitoring: bool = False,
        monitoring_interval: float = 10.0,
    ):
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.current_usage = 0
        self.peak_usage = 0
        self.total_allocations = 0
        self.total_releases = 0
        self.rejected_allocations = 0
        self.lock = threading.Lock()
        
        # 监控
        self.enable_monitoring = enable_monitoring
        self.monitoring_interval = monitoring_interval
        self.last_monitor_time = time.time()
        
        logger.info(
            f"[VIT Memory Pool] Initialized: max_memory={max_memory_gb:.2f} GB "
            f"({self.max_memory_bytes:,} bytes)"
        )
    
    def can_allocate(self, size_bytes: int) -> bool:
        """
        检查是否可以分配指定大小的显存
        
        Args:
            size_bytes: 需要分配的显存大小（字节）
        
        Returns:
            True if can allocate, False otherwise
        """
        with self.lock:
            return self.current_usage + size_bytes <= self.max_memory_bytes
    
    def allocate(self, size_bytes: int) -> bool:
        """
        分配显存
        
        Args:
            size_bytes: 需要分配的显存大小（字节）
        
        Returns:
            True if allocated successfully, False if pool is full
        """
        with self.lock:
            if self.current_usage + size_bytes > self.max_memory_bytes:
                self.rejected_allocations += 1
                logger.warning(
                    f"[VIT Memory Pool] ❌ Allocation rejected: "
                    f"requested={size_bytes / 1024**2:.2f} MB, "
                    f"current={self.current_usage / 1024**2:.2f} MB, "
                    f"max={self.max_memory_bytes / 1024**2:.2f} MB, "
                    f"available={(self.max_memory_bytes - self.current_usage) / 1024**2:.2f} MB"
                )
                return False
            
            self.current_usage += size_bytes
            self.total_allocations += 1
            
            # 更新峰值
            if self.current_usage > self.peak_usage:
                self.peak_usage = self.current_usage
            
            logger.debug(
                f"[VIT Memory Pool] ✅ Allocated: "
                f"size={size_bytes / 1024**2:.2f} MB, "
                f"current={self.current_usage / 1024**2:.2f} MB, "
                f"usage={self.current_usage / self.max_memory_bytes * 100:.1f}%"
            )
            
            # 监控
            self._maybe_log_stats()
            
            return True
    
    def release(self, size_bytes: int):
        """
        释放显存
        
        Args:
            size_bytes: 需要释放的显存大小（字节）
        """
        with self.lock:
            self.current_usage = max(0, self.current_usage - size_bytes)
            self.total_releases += 1
            
            logger.debug(
                f"[VIT Memory Pool] 🔄 Released: "
                f"size={size_bytes / 1024**2:.2f} MB, "
                f"current={self.current_usage / 1024**2:.2f} MB, "
                f"usage={self.current_usage / self.max_memory_bytes * 100:.1f}%"
            )
    
    def get_stats(self) -> dict:
        """
        获取显存池统计信息
        
        Returns:
            Dictionary containing memory pool statistics
        """
        with self.lock:
            return {
                "max_memory_bytes": self.max_memory_bytes,
                "max_memory_gb": self.max_memory_bytes / 1024**3,
                "current_usage_bytes": self.current_usage,
                "current_usage_gb": self.current_usage / 1024**3,
                "current_usage_percent": self.current_usage / self.max_memory_bytes * 100,
                "peak_usage_bytes": self.peak_usage,
                "peak_usage_gb": self.peak_usage / 1024**3,
                "peak_usage_percent": self.peak_usage / self.max_memory_bytes * 100,
                "available_bytes": self.max_memory_bytes - self.current_usage,
                "available_gb": (self.max_memory_bytes - self.current_usage) / 1024**3,
                "total_allocations": self.total_allocations,
                "total_releases": self.total_releases,
                "rejected_allocations": self.rejected_allocations,
            }
    
    def _maybe_log_stats(self):
        """
        定期输出统计信息（如果启用监控）
        """
        if not self.enable_monitoring:
            return
        
        now = time.time()
        if now - self.last_monitor_time >= self.monitoring_interval:
            stats = self.get_stats()
            logger.info(
                f"[VIT Memory Pool] 📊 Stats: "
                f"current={stats['current_usage_gb']:.2f} GB ({stats['current_usage_percent']:.1f}%), "
                f"peak={stats['peak_usage_gb']:.2f} GB ({stats['peak_usage_percent']:.1f}%), "
                f"available={stats['available_gb']:.2f} GB, "
                f"allocations={stats['total_allocations']}, "
                f"releases={stats['total_releases']}, "
                f"rejected={stats['rejected_allocations']}"
            )
            self.last_monitor_time = now
    
    def reset_stats(self):
        """
        重置统计信息（保留当前使用量和峰值）
        """
        with self.lock:
            self.total_allocations = 0
            self.total_releases = 0
            self.rejected_allocations = 0
            logger.info("[VIT Memory Pool] Stats reset")
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"VITMemoryPool("
            f"max={stats['max_memory_gb']:.2f} GB, "
            f"current={stats['current_usage_gb']:.2f} GB, "
            f"usage={stats['current_usage_percent']:.1f}%)"
        )


def estimate_batch_memory(
    pixel_values_list: list,
    image_grid_list: list,
    embedding_dim: int = 3584,
    dtype_size: int = 2,  # fp16 = 2 bytes, fp32 = 4 bytes
    overhead_factor: float = 1.5,  # 考虑中间变量和梯度
) -> int:
    """
    估算 batch forward 所需的显存
    
    Args:
        pixel_values_list: List of pixel values tensors
        image_grid_list: List of image grid tensors
        embedding_dim: Embedding dimension (default: 3584 for Qwen2.5-VL)
        dtype_size: Size of data type in bytes (default: 2 for fp16)
        overhead_factor: Overhead factor for intermediate variables (default: 1.5)
    
    Returns:
        Estimated memory in bytes
    """
    # 1. 输入显存: pixel_values
    input_memory = sum(pv.nelement() * pv.element_size() for pv in pixel_values_list)
    
    # 2. 输出显存: embeddings
    # 假设每个 pixel_value 对应一个 embedding
    total_pixels = sum(pv.shape[0] for pv in pixel_values_list)
    output_memory = total_pixels * embedding_dim * dtype_size
    
    # 3. 中间变量和梯度（估算）
    intermediate_memory = (input_memory + output_memory) * (overhead_factor - 1.0)
    
    # 4. 总显存
    total_memory = input_memory + output_memory + intermediate_memory
    
    logger.debug(
        f"[VIT Memory Pool] Estimated memory: "
        f"input={input_memory / 1024**2:.2f} MB, "
        f"output={output_memory / 1024**2:.2f} MB, "
        f"intermediate={intermediate_memory / 1024**2:.2f} MB, "
        f"total={total_memory / 1024**2:.2f} MB"
    )
    
    return int(total_memory)

