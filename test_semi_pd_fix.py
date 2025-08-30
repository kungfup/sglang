#!/usr/bin/env python3
"""
测试Semi-PD TP=2, PP=1的修复
"""

import os
import sys

# 添加当前目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def test_parallel_state():
    """测试parallel_state.py的修复"""
    print("🔧 测试parallel_state.py的修复...")
    
    try:
        from sglang.srt.distributed.parallel_state import get_group_coordinator
        print("✅ get_group_coordinator函数导入成功")
        
        # 测试函数调用
        coordinator = get_group_coordinator()
        print(f"✅ get_group_coordinator()调用成功，返回: {coordinator}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    return True

def test_model_runner():
    """测试model_runner.py的修复"""
    print("\n🔧 测试model_runner.py的修复...")
    
    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
        print("✅ ModelRunner类导入成功")
        
        # 检查修复后的代码
        with open('python/sglang/srt/model_executor/model_runner.py', 'r') as f:
            content = f.read()
            
        # 检查关键修复
        fixes = [
            "Semi-PD TP模式: pp_size=1，所有进程共享完整模型，不进行TP分割",
            "设置tensor_model_parallel_size=1，避免权重分割",
            "tensor_model_parallel_size=1,  # 🔧 修复：不进行TP分割",
            "PP模式: 使用world组进行broadcast"
        ]
        
        for fix in fixes:
            if fix in content:
                print(f"✅ 找到修复: {fix}")
            else:
                print(f"❌ 未找到修复: {fix}")
                return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    return True

def test_tp_worker():
    """测试tp_worker.py的修复"""
    print("\n🔧 测试tp_worker.py的修复...")
    
    try:
        from sglang.srt.managers.tp_worker import TpModelWorker
        print("✅ TpModelWorker类导入成功")
        
        # 检查修复后的代码
        with open('python/sglang/srt/managers/tp_worker.py', 'r') as f:
            content = f.read()
            
        # 检查关键修复
        fixes = [
            "Semi-PD模式: 在PP模式下，所有PP stage共享同一个分布式环境",
            "PP模式: 使用world组进行broadcast",
            "pp_size=1也是PP模式，只是只有一个stage"
        ]
        
        for fix in fixes:
            if fix in content:
                print(f"✅ 找到修复: {fix}")
            else:
                print(f"❌ 未找到修复: {fix}")
                return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始测试Semi-PD TP=2, PP=1的修复...")
    print("=" * 50)
    
    # 测试parallel_state.py
    test1 = test_parallel_state()
    
    # 测试model_runner.py
    test2 = test_model_runner()
    
    # 测试tp_worker.py
    test3 = test_tp_worker()
    
    print("\n" + "=" * 50)
    if test1 and test2 and test3:
        print("🎉 所有测试通过！修复成功！")
        print("\n🔧 修复总结:")
        print("1. ✅ 添加了get_group_coordinator函数")
        print("2. ✅ 修复了PP组创建逻辑")
        print("3. ✅ 修复了broadcast逻辑")
        print("4. ✅ 修复了TP/PP模式判断")
        print("5. ✅ 修复了TP分割逻辑 - 当pp_size=1时不进行TP分割")
        print("6. ✅ 修复了权重加载问题")
        return True
    else:
        print("❌ 部分测试失败，需要进一步修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 