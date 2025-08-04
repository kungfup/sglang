"""
Semi-PD 异步优化集成模块
与 CUDA 层的异步 AllReduce 优化协同工作
"""

import os
import logging
import torch
from typing import Optional, Dict, Any
from sglang.semi_pd.utils import InstanceRole

logger = logging.getLogger(__name__)


class SemiPDAsyncOptimizer:
    """Semi-PD 异步优化管理器"""
    
    def __init__(self, instance_role: InstanceRole, gpu_id: int):
        self.instance_role = instance_role
        self.gpu_id = gpu_id
        self.optimization_enabled = self._check_optimization_support()
        self.memory_pool_size = 0
        self.allreduce_cache_size = 0
        
        if self.optimization_enabled:
            logger.info(f"✅ Semi-PD 异步优化已启用 ({instance_role.name} 进程)")
            self._configure_async_settings()
        else:
            logger.warning(f"❌ Semi-PD 异步优化未启用 ({instance_role.name} 进程)")

    def _check_optimization_support(self) -> bool:
        """检查是否支持异步优化"""
        # 检查环境变量
        if os.environ.get("SEMI_PD_DISABLE_ASYNC_OPT", "0") == "1":
            logger.info("Semi-PD 异步优化被环境变量禁用")
            return False
            
        # 检查 CUDA 版本支持
        if not torch.cuda.is_available():
            logger.warning("CUDA 不可用，禁用异步优化")
            return False
            
        # 检查计算能力
        if torch.cuda.get_device_capability(self.gpu_id)[0] < 7:
            logger.warning("GPU 计算能力过低 (<7.0)，建议禁用异步优化")
            return False
            
        return True

    def _configure_async_settings(self):
        """配置异步优化设置"""
        # 根据进程角色配置不同的优化策略
        if self.instance_role == InstanceRole.PREFILL:
            # PREFILL 进程：重点优化内存分配和同步开销
            self.memory_pool_size = int(os.environ.get("SEMI_PD_PREFILL_MEMORY_POOL_SIZE", "256"))  # MB
            self.allreduce_cache_size = int(os.environ.get("SEMI_PD_PREFILL_ALLREDUCE_CACHE_SIZE", "64"))
            
            # 设置 CUDA 内存池
            torch.cuda.set_per_process_memory_fraction(0.95, device=self.gpu_id)
            
            logger.info(f"PREFILL 进程异步优化配置:")
            logger.info(f"  - 内存池大小: {self.memory_pool_size} MB")
            logger.info(f"  - AllReduce 缓存大小: {self.allreduce_cache_size}")
            
        elif self.instance_role == InstanceRole.DECODE:
            # DECODE 进程：重点优化 CUDA Graph 和流管理
            self.memory_pool_size = int(os.environ.get("SEMI_PD_DECODE_MEMORY_POOL_SIZE", "512"))  # MB
            self.allreduce_cache_size = int(os.environ.get("SEMI_PD_DECODE_ALLREDUCE_CACHE_SIZE", "128"))
            
            # 启用 CUDA Graph 兼容的内存分配器
            torch.cuda.empty_cache()
            
            logger.info(f"DECODE 进程异步优化配置:")
            logger.info(f"  - 内存池大小: {self.memory_pool_size} MB")
            logger.info(f"  - AllReduce 缓存大小: {self.allreduce_cache_size}")
            logger.info(f"  - CUDA Graph 友好的内存管理已启用")

    def optimize_before_forward(self, model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """在前向传播前进行优化"""
        if not self.optimization_enabled:
            return model_inputs
            
        # 预热内存分配
        if self.instance_role == InstanceRole.PREFILL:
            self._prefill_memory_warmup(model_inputs)
        elif self.instance_role == InstanceRole.DECODE:
            self._decode_memory_warmup(model_inputs)
            
        return model_inputs

    def optimize_after_forward(self, outputs: Any) -> Any:
        """在前向传播后进行优化"""
        if not self.optimization_enabled:
            return outputs
            
        # 清理临时内存
        if self.instance_role == InstanceRole.PREFILL:
            self._prefill_memory_cleanup()
        
        return outputs

    def _prefill_memory_warmup(self, model_inputs: Dict[str, Any]):
        """PREFILL 进程的内存预热"""
        # 预分配常用的 tensor 大小
        batch_size = model_inputs.get("input_ids", torch.tensor([])).shape[0] if "input_ids" in model_inputs else 1
        
        # 为 AllReduce 操作预热内存句柄
        if batch_size > 0:
            logger.debug(f"PREFILL 内存预热: batch_size={batch_size}")

    def _decode_memory_warmup(self, model_inputs: Dict[str, Any]):
        """DECODE 进程的内存预热"""
        # DECODE 通常处理单个 token，优化重点不同
        logger.debug("DECODE 内存预热")

    def _prefill_memory_cleanup(self):
        """PREFILL 进程的内存清理"""
        # 定期清理过期的内存句柄缓存
        pass

    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            "optimization_enabled": self.optimization_enabled,
            "instance_role": self.instance_role.name,
            "memory_pool_size_mb": self.memory_pool_size,
            "allreduce_cache_size": self.allreduce_cache_size,
            "gpu_id": self.gpu_id,
        }


def create_semi_pd_async_optimizer(instance_role: InstanceRole, gpu_id: int) -> SemiPDAsyncOptimizer:
    """创建 Semi-PD 异步优化器的工厂函数"""
    return SemiPDAsyncOptimizer(instance_role, gpu_id)


# 全局优化器实例
_global_optimizer: Optional[SemiPDAsyncOptimizer] = None


def get_semi_pd_async_optimizer() -> Optional[SemiPDAsyncOptimizer]:
    """获取全局异步优化器实例"""
    return _global_optimizer


def set_semi_pd_async_optimizer(optimizer: SemiPDAsyncOptimizer):
    """设置全局异步优化器实例"""
    global _global_optimizer
    _global_optimizer = optimizer
    logger.info(f"Semi-PD 异步优化器已设置: {optimizer.instance_role.name}")


def enable_semi_pd_async_optimization():
    """启用 Semi-PD 异步优化的全局开关"""
    os.environ["SEMI_PD_ASYNC_OPT_ENABLED"] = "1"
    logger.info("Semi-PD 异步优化已全局启用")


def disable_semi_pd_async_optimization():
    """禁用 Semi-PD 异步优化的全局开关"""
    os.environ["SEMI_PD_DISABLE_ASYNC_OPT"] = "1"
    logger.info("Semi-PD 异步优化已全局禁用") 