"""Graceful shutdown utilities for VIT workers.

参考 LightLLM 的实现，确保子进程能够优雅地处理 SIGTERM 信号。
"""

import signal
import logging

logger = logging.getLogger(__name__)


def graceful_registry(sub_module_name: str):
    """注册 graceful shutdown 处理
    
    子进程在收到 SIGTERM 时，不能自己就提前退出，
    而是由主进程来决定退出时机。
    
    Args:
        sub_module_name: 子模块名称（用于日志）
    """
    def graceful_shutdown(signum, frame):
        logger.info(f"{sub_module_name} Received signal to shutdown. Performing graceful shutdown...")
        if signum == signal.SIGTERM:
            # 不退出，由主进程来决定退出时机
            logger.info(f"{sub_module_name} receive sigterm")
    
    signal.signal(signal.SIGTERM, graceful_shutdown)

