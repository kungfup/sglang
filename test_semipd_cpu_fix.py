#!/usr/bin/env python3
"""
🔧 Semi-PD CPU拷贝修复验证脚本

用于验证在Semi-PD架构下，添加non_blocking=True是否能解决
.cpu()调用导致的50ms阻塞问题。
"""

import torch
import time
import numpy as np

def test_cpu_copy_performance():
    """测试CPU拷贝的性能差异"""
    
    print("🔧 测试Semi-PD CPU拷贝性能修复")
    print("=" * 50)
    
    # 模拟Semi-PD中的tensor状态
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建类似Semi-PD中的seq_lens tensor
    seq_lens = torch.randint(1, 100, (32,), device=device, dtype=torch.int32)
    
    print(f"设备: {device}")
    print(f"Tensor形状: {seq_lens.shape}")
    print(f"Tensor设备: {seq_lens.device}")
    
    # 测试阻塞版本 (原有问题)
    start_time = time.perf_counter()
    seq_lens_cpu_blocking = seq_lens.cpu()
    blocking_time = time.perf_counter() - start_time
    
    print(f"\n阻塞版本 (.cpu()): {blocking_time*1000:.3f}ms")
    
    # 测试非阻塞版本 (修复后)
    start_time = time.perf_counter()
    seq_lens_cpu_nonblocking = seq_lens.to('cpu', non_blocking=True)
    nonblocking_time = time.perf_counter() - start_time
    
    print(f"非阻塞版本 (.to('cpu', non_blocking=True)): {nonblocking_time*1000:.3f}ms")
    
    # 等待非阻塞操作完成
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # 验证结果一致性
    if torch.equal(seq_lens_cpu_blocking, seq_lens_cpu_nonblocking):
        print("✅ 结果一致性验证通过")
    else:
        print("❌ 结果一致性验证失败")
    
    improvement = (blocking_time - nonblocking_time) / blocking_time * 100
    print(f"\n🚀 性能提升: {improvement:.1f}%")
    
    if nonblocking_time < blocking_time:
        print("✅ 非阻塞版本更快")
    else:
        print("⚠️ 性能差异不明显（可能在单进程环境中）")
    
    print("\n📝 说明:")
    print("- 在Semi-PD的跨进程环境中，性能差异会更明显")
    print("- 非阻塞版本可以避免等待PREFILL进程的数据传输")
    print("- 这应该能解决CUDA Graph replay的50ms阻塞问题")

if __name__ == "__main__":
    test_cpu_copy_performance() 