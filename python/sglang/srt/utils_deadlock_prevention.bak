"""
Semi-PD 死锁预防模块
解决 PREFILL-DECODE 调度器间通信死锁问题
"""

import os
import time
import threading
import signal
import logging
import multiprocessing
from typing import Optional, Callable, Any
import queue
import weakref

logger = logging.getLogger(__name__)

# 全局配置
IPC_TIMEOUT = int(os.environ.get('SEMI_PD_IPC_TIMEOUT', '30'))
WATCHDOG_TIMEOUT = int(os.environ.get('SEMI_PD_WATCHDOG_TIMEOUT', '60'))
MAX_RETRY_COUNT = int(os.environ.get('SEMI_PD_MAX_RETRY_COUNT', '3'))

class DeadlockPreventionManager:
    """死锁预防管理器"""
    
    def __init__(self):
        self.active_requests = {}
        self.request_timeouts = {}
        self.lock = threading.Lock()
        self.monitor_thread = None
        self.is_monitoring = False
        
    def register_request(self, request_id: str, timeout: float = None):
        """注册请求，开始监控"""
        if timeout is None:
            timeout = IPC_TIMEOUT
            
        with self.lock:
            self.active_requests[request_id] = time.time()
            self.request_timeouts[request_id] = timeout
            
        if not self.is_monitoring:
            self.start_monitoring()
            
    def unregister_request(self, request_id: str):
        """取消注册请求"""
        with self.lock:
            self.active_requests.pop(request_id, None)
            self.request_timeouts.pop(request_id, None)
            
    def start_monitoring(self):
        """启动监控线程"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_requests,
            daemon=True,
            name="DeadlockMonitor"
        )
        self.monitor_thread.start()
        logger.info("🔍 死锁监控线程已启动")
        
    def _monitor_requests(self):
        """监控请求超时"""
        while self.is_monitoring:
            try:
                current_time = time.time()
                timeout_requests = []
                
                with self.lock:
                    for req_id, start_time in self.active_requests.items():
                        timeout = self.request_timeouts.get(req_id, IPC_TIMEOUT)
                        if current_time - start_time > timeout:
                            timeout_requests.append(req_id)
                
                # 处理超时请求
                for req_id in timeout_requests:
                    logger.warning(f"⚠️ 请求 {req_id} 超时，可能发生死锁")
                    self.unregister_request(req_id)
                    # 这里可以添加死锁恢复逻辑
                    
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"❌ 死锁监控线程出错: {e}")
                
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

# 全局实例
_deadlock_manager = DeadlockPreventionManager()

class TimeoutException(Exception):
    """超时异常"""
    pass

def with_timeout(timeout: float):
    """超时装饰器"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            request_id = f"{func.__name__}_{int(time.time() * 1000)}"
            
            # 注册请求
            _deadlock_manager.register_request(request_id, timeout)
            
            try:
                # 创建超时机制
                result_queue = queue.Queue()
                exception_queue = queue.Queue()
                
                def target():
                    try:
                        result = func(*args, **kwargs)
                        result_queue.put(result)
                    except Exception as e:
                        exception_queue.put(e)
                
                thread = threading.Thread(target=target, daemon=True)
                thread.start()
                
                # 等待结果或超时
                start_time = time.time()
                while thread.is_alive():
                    if time.time() - start_time > timeout:
                        logger.warning(f"⚠️ 函数 {func.__name__} 执行超时 ({timeout}s)")
                        raise TimeoutException(f"函数 {func.__name__} 执行超时")
                    
                    try:
                        result = result_queue.get_nowait()
                        return result
                    except queue.Empty:
                        pass
                        
                    try:
                        exception = exception_queue.get_nowait()
                        raise exception
                    except queue.Empty:
                        pass
                        
                    time.sleep(0.1)
                
                # 线程结束，获取结果
                try:
                    return result_queue.get_nowait()
                except queue.Empty:
                    try:
                        exception = exception_queue.get_nowait()
                        raise exception
                    except queue.Empty:
                        raise TimeoutException(f"函数 {func.__name__} 未返回结果")
                        
            finally:
                # 取消注册
                _deadlock_manager.unregister_request(request_id)
                
        return wrapper
    return decorator

class SafeQueue:
    """安全队列，带超时和重试机制"""
    
    def __init__(self, maxsize: int = 0):
        self.queue = queue.Queue(maxsize)
        self.put_timeout = IPC_TIMEOUT
        self.get_timeout = IPC_TIMEOUT
        
    def put(self, item: Any, timeout: float = None) -> bool:
        """安全放入，带超时"""
        if timeout is None:
            timeout = self.put_timeout
            
        try:
            self.queue.put(item, timeout=timeout)
            return True
        except queue.Full:
            logger.warning(f"⚠️ 队列放入超时 ({timeout}s)")
            return False
            
    def get(self, timeout: float = None) -> Optional[Any]:
        """安全获取，带超时"""
        if timeout is None:
            timeout = self.get_timeout
            
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            logger.warning(f"⚠️ 队列获取超时 ({timeout}s)")
            return None
            
    def qsize(self) -> int:
        """队列大小"""
        return self.queue.qsize()

class ProcessSafeManager:
    """进程安全管理器"""
    
    def __init__(self):
        self.process_registry = {}
        self.lock = multiprocessing.Lock()
        
    def register_process(self, process_name: str, pid: int):
        """注册进程"""
        with self.lock:
            self.process_registry[process_name] = {
                'pid': pid,
                'start_time': time.time(),
                'last_heartbeat': time.time()
            }
            
    def heartbeat(self, process_name: str):
        """进程心跳"""
        with self.lock:
            if process_name in self.process_registry:
                self.process_registry[process_name]['last_heartbeat'] = time.time()
                
    def check_health(self) -> dict:
        """检查所有进程健康状态"""
        current_time = time.time()
        health_status = {}
        
        with self.lock:
            for name, info in self.process_registry.items():
                last_heartbeat = info['last_heartbeat']
                is_healthy = (current_time - last_heartbeat) < WATCHDOG_TIMEOUT
                
                health_status[name] = {
                    'healthy': is_healthy,
                    'pid': info['pid'],
                    'last_heartbeat_ago': current_time - last_heartbeat
                }
                
        return health_status

# 全局进程管理器
_process_manager = ProcessSafeManager()

def register_semi_pd_process(process_name: str):
    """注册Semi-PD进程"""
    pid = os.getpid()
    _process_manager.register_process(process_name, pid)
    logger.info(f"📝 注册 Semi-PD 进程: {process_name} (PID: {pid})")
    
def semi_pd_heartbeat(process_name: str):
    """发送Semi-PD进程心跳"""
    _process_manager.heartbeat(process_name)
    
def check_semi_pd_health() -> dict:
    """检查Semi-PD进程健康状态"""
    return _process_manager.check_health()

def setup_deadlock_prevention():
    """设置死锁预防机制"""
    logger.info(f"🛡️ 设置死锁预防机制")
    logger.info(f"  - IPC超时: {IPC_TIMEOUT}s")
    logger.info(f"  - Watchdog超时: {WATCHDOG_TIMEOUT}s")
    logger.info(f"  - 最大重试: {MAX_RETRY_COUNT}次")
    
    # 启动监控
    _deadlock_manager.start_monitoring()
    
def cleanup_deadlock_prevention():
    """清理死锁预防资源"""
    logger.info("🧹 清理死锁预防资源")
    _deadlock_manager.stop_monitoring()

# 上下文管理器
class DeadlockPreventionContext:
    """死锁预防上下文管理器"""
    
    def __enter__(self):
        setup_deadlock_prevention()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_deadlock_prevention()

# 环境变量控制
ENABLE_DEADLOCK_PREVENTION = os.environ.get('SEMI_PD_DEADLOCK_PREVENTION', '1') == '1'

if ENABLE_DEADLOCK_PREVENTION:
    logger.info("✅ Semi-PD 死锁预防已启用")
else:
    logger.info("❌ Semi-PD 死锁预防已禁用") 