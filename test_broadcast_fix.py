#!/usr/bin/env python3
"""
测试 Semi-PD 模式下 broadcast_pyobj 的修复
"""

import os
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_broadcast_fix():
    """测试 broadcast_pyobj 修复"""
    
    # 模拟 Semi-PD 环境
    os.environ["SGLANG_ENABLE_SEMI_PD"] = "1"
    os.environ["SGLANG_PP_RANK"] = "0"
    os.environ["SGLANG_GPU_ID"] = "0"
    
    logger.info("🔧 测试 Semi-PD 模式下的 broadcast_pyobj 修复")
    logger.info(f"   - SGLANG_ENABLE_SEMI_PD: {os.environ.get('SGLANG_ENABLE_SEMI_PD')}")
    logger.info(f"   - SGLANG_PP_RANK: {os.environ.get('SGLANG_PP_RANK')}")
    logger.info(f"   - SGLANG_GPU_ID: {os.environ.get('SGLANG_GPU_ID')}")
    
    try:
        # 导入修复后的模块
        from sglang.srt.managers.tp_worker import TpModelWorker
        logger.info("✅ 成功导入 TpModelWorker")
        
        # 检查修复的代码
        with open("python/sglang/srt/managers/tp_worker.py", "r") as f:
            content = f.read()
            
        if "broadcast_rank = tp_rank" in content and "get_tp_group().cpu_group" in content:
            logger.info("✅ 找到修复代码")
            logger.info("   - 使用 tp_rank 作为 broadcast_rank")
            logger.info("   - 使用 TP 组而不是 world 组")
        else:
            logger.error("❌ 未找到修复代码")
            return False
            
        logger.info("✅ 所有测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_broadcast_fix()
    sys.exit(0 if success else 1) 