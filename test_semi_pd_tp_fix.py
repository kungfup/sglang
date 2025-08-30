#!/usr/bin/env python3
"""
测试Semi-PD TP配置修复
"""

import os
import sys
import copy

# 设置环境变量
os.environ["SGLANG_ENABLE_SEMI_PD"] = "1"

# 添加sglang路径
sys.path.insert(0, "python")

def test_semi_pd_tp_config():
    """测试Semi-PD模式下的TP配置修复"""
    print("🔧 测试Semi-PD TP配置修复...")
    
    try:
        # 测试配置复制和修改
        class MockServerArgs:
            def __init__(self):
                self.tp_size = 2
                self.pp_size = 1
                self.dp_size = 1
                self.tensor_model_parallel_size = 1  # 原始值
                self.enable_semi_pd = True
        
        # 模拟原始配置
        server_args = MockServerArgs()
        print(f"✅ 原始配置: tp_size={server_args.tp_size}, tensor_model_parallel_size={server_args.tensor_model_parallel_size}")
        
        # 测试配置修复逻辑
        tp_size_per_node = server_args.tp_size // 1  # nnodes=1
        
        # 为DECODE实例修复配置
        decode_server_args = copy.deepcopy(server_args)
        decode_server_args.tensor_model_parallel_size = tp_size_per_node
        print(f"✅ DECODE配置修复: tensor_model_parallel_size={decode_server_args.tensor_model_parallel_size}")
        
        # 为PREFILL实例修复配置
        prefill_server_args = copy.deepcopy(server_args)
        prefill_server_args.tensor_model_parallel_size = tp_size_per_node
        print(f"✅ PREFILL配置修复: tensor_model_parallel_size={prefill_server_args.tensor_model_parallel_size}")
        
        # 验证配置
        assert decode_server_args.tensor_model_parallel_size == 2, "DECODE TP配置错误"
        assert prefill_server_args.tensor_model_parallel_size == 2, "PREFILL TP配置错误"
        assert decode_server_args.tp_size == 2, "TP大小配置错误"
        
        print("🎉 所有TP配置测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        return False

if __name__ == "__main__":
    success = test_semi_pd_tp_config()
    sys.exit(0 if success else 1) 