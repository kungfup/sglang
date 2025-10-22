"""SHM Reference Manager for VIT Scheduler

对齐 LightLLM 的 SHM 生命周期管理：
- 引用计数管理（acquire/release）
- 完成通知机制
- 线程安全
- 防止过早清理

参考: LightLLM/lightllm/server/embed_cache/manager.py
"""

from __future__ import annotations

import logging
import threading
import time
from multiprocessing import shared_memory as shm
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class SHMReferenceManager:
    """SHM 引用计数管理器
    
    核心功能:
    1. acquire(shm_name): 增加引用计数
    2. release(shm_name): 减少引用计数
    3. cleanup_if_zero(shm_name): 引用计数为 0 时清理
    
    对齐 LightLLM:
    - LightLLM 在 VisualServer 中为每张图分配 UUID
    - 请求写入 SHM 时 ref++
    - Worker 读取完成后告知 router
    - Router/PP0 在消费完 embedding 后 ref--
    - 引用归零才 unlink
    """
    
    def __init__(self):
        self._refs: Dict[str, int] = {}  # {shm_name: ref_count}
        self._lock = threading.Lock()
        self._created_shm: Set[str] = set()  # 记录已创建的 SHM
        
        # 统计信息
        self.acquire_count = 0
        self.release_count = 0
        self.cleanup_count = 0
        self.leak_count = 0  # 引用计数泄漏次数
        
        logger.info("[SHM Manager] Initialized")
    
    def acquire(self, shm_name: str) -> None:
        """增加引用计数
        
        Args:
            shm_name: 共享内存名称
        """
        with self._lock:
            if shm_name not in self._refs:
                self._refs[shm_name] = 0
            
            self._refs[shm_name] += 1
            self.acquire_count += 1
            
            logger.debug(
                "[SHM Manager] Acquired %s, ref_count=%d",
                shm_name,
                self._refs[shm_name]
            )
    
    def release(self, shm_name: str) -> bool:
        """减少引用计数
        
        Args:
            shm_name: 共享内存名称
        
        Returns:
            是否可以清理（引用计数为 0）
        """
        with self._lock:
            if shm_name not in self._refs:
                logger.warning(
                    "[SHM Manager] Release unknown SHM: %s (possible double-release or never acquired)",
                    shm_name
                )
                self.leak_count += 1
                return True  # 未知 SHM，可以尝试清理
            
            self._refs[shm_name] -= 1
            self.release_count += 1
            
            logger.debug(
                "[SHM Manager] Released %s, ref_count=%d",
                shm_name,
                self._refs[shm_name]
            )
            
            if self._refs[shm_name] <= 0:
                # 引用计数为 0，可以清理
                if self._refs[shm_name] < 0:
                    logger.warning(
                        "[SHM Manager] Negative ref_count for %s: %d (double-release detected)",
                        shm_name,
                        self._refs[shm_name]
                    )
                    self.leak_count += 1
                
                del self._refs[shm_name]
                return True  # 可以清理
            
            return False  # 还有引用，不能清理
    
    def cleanup_if_zero(self, shm_name: str) -> bool:
        """如果引用计数为 0，清理 SHM
        
        Args:
            shm_name: 共享内存名称
        
        Returns:
            是否成功清理
        """
        # 检查引用计数
        with self._lock:
            if shm_name in self._refs and self._refs[shm_name] > 0:
                logger.debug(
                    "[SHM Manager] Cannot cleanup %s, ref_count=%d > 0",
                    shm_name,
                    self._refs[shm_name]
                )
                return False
        
        # 引用计数为 0，执行清理
        try:
            shared_memory = shm.SharedMemory(name=shm_name)
            shared_memory.close()
            shared_memory.unlink()
            
            with self._lock:
                self.cleanup_count += 1
                self._created_shm.discard(shm_name)
            
            logger.debug("[SHM Manager] Cleaned up SHM: %s", shm_name)
            return True
        
        except FileNotFoundError:
            logger.debug("[SHM Manager] SHM already gone: %s", shm_name)
            return True
        
        except Exception as exc:
            logger.warning(
                "[SHM Manager] Failed to cleanup SHM %s: %s",
                shm_name,
                exc
            )
            return False
    
    def force_cleanup(self, shm_name: str) -> bool:
        """强制清理 SHM（忽略引用计数）
        
        ⚠️ 仅用于异常情况（如进程崩溃后的清理）
        
        Args:
            shm_name: 共享内存名称
        
        Returns:
            是否成功清理
        """
        with self._lock:
            if shm_name in self._refs:
                logger.warning(
                    "[SHM Manager] Force cleanup %s with ref_count=%d > 0",
                    shm_name,
                    self._refs[shm_name]
                )
                del self._refs[shm_name]
        
        return self.cleanup_if_zero(shm_name)
    
    def get_ref_count(self, shm_name: str) -> int:
        """获取引用计数
        
        Args:
            shm_name: 共享内存名称
        
        Returns:
            引用计数（如果不存在返回 0）
        """
        with self._lock:
            return self._refs.get(shm_name, 0)
    
    def get_stats(self) -> Dict:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                "active_shm_count": len(self._refs),
                "total_acquire": self.acquire_count,
                "total_release": self.release_count,
                "total_cleanup": self.cleanup_count,
                "leak_count": self.leak_count,
                "active_shm_names": list(self._refs.keys()),
            }
    
    def cleanup_all(self) -> int:
        """清理所有 SHM（用于进程退出）
        
        Returns:
            清理的 SHM 数量
        """
        with self._lock:
            shm_names = list(self._refs.keys())
        
        count = 0
        for shm_name in shm_names:
            if self.force_cleanup(shm_name):
                count += 1
        
        logger.info("[SHM Manager] Cleaned up %d SHM objects", count)
        return count


# 全局单例
_global_shm_manager: Optional[SHMReferenceManager] = None
_global_shm_manager_lock = threading.Lock()


def get_global_shm_manager() -> SHMReferenceManager:
    """获取全局 SHM 管理器单例
    
    Returns:
        全局 SHM 管理器
    """
    global _global_shm_manager
    
    if _global_shm_manager is None:
        with _global_shm_manager_lock:
            if _global_shm_manager is None:
                _global_shm_manager = SHMReferenceManager()
    
    return _global_shm_manager

