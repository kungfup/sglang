#!/usr/bin/env python3
"""
测试 Semi-PD 修复的脚本
"""

import os
import sys
import logging

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """测试导入"""
    try:
        from sglang.srt.managers.semi_pd_scheduler import run_scheduler_process
        logger.info("✅ run_scheduler_process 导入成功")
        
        from sglang.srt.managers.semi_pd_decode_scheduler import SemiPDDecodeScheduler
        logger.info("✅ SemiPDDecodeScheduler 导入成功")
        
        from sglang.srt.managers.semi_pd_prefill_scheduler import SemiPDPrefillScheduler
        logger.info("✅ SemiPDPrefillScheduler 导入成功")
        
        from sglang.srt.entrypoints.engine import InstanceRole
        logger.info("✅ InstanceRole 导入成功")
        
        return True
    except Exception as e:
        logger.error(f"❌ 导入失败: {e}")
        return False

def test_scheduler_constructors():
    """测试调度器构造函数"""
    try:
        from sglang.srt.managers.semi_pd_decode_scheduler import SemiPDDecodeScheduler
        from sglang.srt.managers.semi_pd_prefill_scheduler import SemiPDPrefillScheduler
        import inspect
        
        # 检查 SemiPDDecodeScheduler 构造函数
        decode_sig = inspect.signature(SemiPDDecodeScheduler.__init__)
        decode_params = list(decode_sig.parameters.keys())
        logger.info(f"✅ SemiPDDecodeScheduler 参数: {decode_params}")
        
        # 检查 SemiPDPrefillScheduler 构造函数
        prefill_sig = inspect.signature(SemiPDPrefillScheduler.__init__)
        prefill_params = list(prefill_sig.parameters.keys())
        logger.info(f"✅ SemiPDPrefillScheduler 参数: {prefill_params}")
        
        # 检查是否包含必需的参数
        required_params = ['server_args', 'port_args', 'gpu_id', 'tp_rank', 'pp_rank', 'dp_rank', 'bypass_load_weight']
        
        for param in required_params:
            if param in decode_params:
                logger.info(f"✅ SemiPDDecodeScheduler 包含参数 {param}")
            else:
                logger.error(f"❌ SemiPDDecodeScheduler 缺少参数 {param}")
                return False
                
            if param in prefill_params:
                logger.info(f"✅ SemiPDPrefillScheduler 包含参数 {param}")
            else:
                logger.error(f"❌ SemiPDPrefillScheduler 缺少参数 {param}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"❌ 构造函数测试失败: {e}")
        return False

def test_function_signatures():
    """测试函数签名"""
    try:
        from sglang.srt.managers.semi_pd_scheduler import run_scheduler_process
        import inspect
        
        # 获取函数签名
        sig = inspect.signature(run_scheduler_process)
        params = list(sig.parameters.keys())
        
        logger.info(f"✅ run_scheduler_process 参数: {params}")
        
        # 检查关键参数是否存在
        expected_params = ['server_args', 'port_args', 'gpu_id', 'tp_rank', 'pp_rank', 'dp_rank', 'pipe_writer', 'ipc_info_queue', 'bypass_load_weight', 'instance_role']
        
        for param in expected_params:
            if param in params:
                logger.info(f"✅ 参数 {param} 存在")
            else:
                logger.error(f"❌ 参数 {param} 缺失")
                return False
        
        return True
    except Exception as e:
        logger.error(f"❌ 函数签名测试失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("🚀 开始测试 Semi-PD 修复...")
    
    # 测试导入
    if not test_imports():
        logger.error("❌ 导入测试失败")
        return False
    
    # 测试调度器构造函数
    if not test_scheduler_constructors():
        logger.error("❌ 调度器构造函数测试失败")
        return False
    
    # 测试函数签名
    if not test_function_signatures():
        logger.error("❌ 函数签名测试失败")
        return False
    
    logger.info("🎉 所有测试通过！Semi-PD 修复成功")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 