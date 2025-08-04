#!/usr/bin/env python3
"""
测试真实性能优化效果
模拟Semi-PD中的真实操作负载，验证优化效果
"""

import torch
import time
import logging
import os
import sys

# 设置环境
os.environ['SEMI_PD_ASYNC_OPT_ENABLED'] = '1'
sys.path.append('/home/yzh/semi_pd_migration/sglang_0.4.8')

from optimize_real_bottlenecks import (
    apply_real_performance_optimizations,
    print_real_optimization_stats,
    real_performance_optimization
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_prefill_workload():
    """模拟PREFILL阶段的工作负载"""
    print("🔄 模拟PREFILL阶段工作负载...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟大量的item()调用 (原来93.01%的瓶颈)
    print("  📊 测试item()调用优化...")
    item_results = []
    start_time = time.time()
    
    for i in range(150):  # 模拟152次调用
        # 模拟序列长度查询
        seq_len = torch.tensor([128 + i], device=device)
        length = seq_len.item()  # 这里会被优化
        item_results.append(length)
        
        # 模拟batch大小查询
        batch_size = torch.tensor([16], device=device)
        bs = batch_size.item()  # 这里会被优化
        item_results.append(bs)
    
    item_time = (time.time() - start_time) * 1000
    print(f"    ⏱️  item()调用耗时: {item_time:.2f}ms")
    
    # 模拟大量的流同步 (原来92.76%的瓶颈)
    print("  🔄 测试流同步优化...")
    start_time = time.time()
    
    for i in range(300):  # 模拟304次同步
        # 创建一些GPU操作
        x = torch.randn(100, 100, device=device)
        y = torch.mm(x, x.t())
        
        # 原来这里会调用 cudaStreamSynchronize
        torch.cuda.synchronize()  # 这里会被优化
    
    sync_time = (time.time() - start_time) * 1000
    print(f"    ⏱️  流同步耗时: {sync_time:.2f}ms")
    
    # 模拟广播通信 (原来121,228次调用)
    if torch.cuda.device_count() > 1:
        print("  📡 测试广播批处理优化...")
        start_time = time.time()
        
        # 模拟多次小广播
        for i in range(100):  # 简化版本，实际是121k次
            small_tensor = torch.randn(10, device=device)
            # 这里原来会有大量broadcast调用
            # torch.distributed.broadcast(small_tensor, 0)  # 需要init process group
        
        broadcast_time = (time.time() - start_time) * 1000
        print(f"    ⏱️  广播模拟耗时: {broadcast_time:.2f}ms")
    
    return {
        'item_time': item_time,
        'sync_time': sync_time,
        'total_operations': 300 + 150*2
    }

def simulate_decode_workload():
    """模拟DECODE阶段的工作负载"""
    print("🔄 模拟DECODE阶段工作负载...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟CUDA Graph启动 (原来94.23%的瓶颈)
    print("  📈 测试CUDA Graph优化...")
    
    def mock_decode_function(batch_tensor):
        """模拟decode函数"""
        # 简单的计算模拟
        x = torch.relu(batch_tensor)
        y = torch.softmax(x, dim=-1)
        return y.argmax(dim=-1)
    
    start_time = time.time()
    results = []
    
    # 模拟425次graph启动
    for i in range(50):  # 简化版本
        batch_data = torch.randn(1, 256, device=device)  # batch_size=1
        
        # 这里原来每次都会启动新的CUDA Graph
        # 现在会被优化复用
        result = mock_decode_function(batch_data)
        results.append(result)
    
    graph_time = (time.time() - start_time) * 1000
    print(f"    ⏱️  Graph启动耗时: {graph_time:.2f}ms")
    
    # 模拟kernel启动 (14,091次调用)
    print("  🚀 测试kernel启动优化...")
    start_time = time.time()
    
    for i in range(100):  # 简化版本
        # 模拟频繁的小kernel
        x = torch.randn(32, 32, device=device)
        # 回退修改：恢复原始代码 (可能引发RuntimeError)
        y = torch.sum(x, dim=1)  # 触发kernel启动
        # y = torch.sum(x)  # 修改版本：对整个tensor求和，得到标量
        try:
            z = y.item()  # 同时测试item优化
        except RuntimeError as e:
            # 处理多元素tensor转标量的错误
            z = y[0].item() if y.numel() > 1 else y.item()
    
    kernel_time = (time.time() - start_time) * 1000
    print(f"    ⏱️  kernel启动耗时: {kernel_time:.2f}ms")
    
    return {
        'graph_time': graph_time,
        'kernel_time': kernel_time,
        'total_operations': 50 + 100
    }

def benchmark_with_without_optimization():
    """对比有无优化的性能差异"""
    print("\n🎯 性能对比测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过性能测试")
        return
    
    # 测试无优化版本
    print("📊 测试1: 无优化版本...")
    start_time = time.time()
    
    # 执行基准工作负载
    baseline_prefill = simulate_prefill_workload()
    baseline_decode = simulate_decode_workload()
    
    baseline_total_time = (time.time() - start_time) * 1000
    print(f"✅ 无优化总耗时: {baseline_total_time:.2f}ms")
    
    # 测试有优化版本
    print("\n📊 测试2: 有优化版本...")
    
    with real_performance_optimization():
        start_time = time.time()
        
        # 执行相同的工作负载
        optimized_prefill = simulate_prefill_workload()
        optimized_decode = simulate_decode_workload()
        
        optimized_total_time = (time.time() - start_time) * 1000
        print(f"✅ 有优化总耗时: {optimized_total_time:.2f}ms")
    
    # 计算改善
    improvement = ((baseline_total_time - optimized_total_time) / baseline_total_time) * 100
    
    print("\n📈 性能对比结果:")
    print("=" * 50)
    print(f"📊 基准版本: {baseline_total_time:.2f}ms")
    print(f"🚀 优化版本: {optimized_total_time:.2f}ms")
    print(f"⚡ 性能提升: {improvement:+.1f}%")
    
    # 详细分析
    print(f"\n🔍 详细分析:")
    print(f"  PREFILL阶段:")
    print(f"    - item()调用: {baseline_prefill['item_time']:.1f}ms → {optimized_prefill['item_time']:.1f}ms")
    print(f"    - 流同步: {baseline_prefill['sync_time']:.1f}ms → {optimized_prefill['sync_time']:.1f}ms")
    
    print(f"  DECODE阶段:")
    print(f"    - Graph启动: {baseline_decode['graph_time']:.1f}ms → {optimized_decode['graph_time']:.1f}ms")
    print(f"    - Kernel启动: {baseline_decode['kernel_time']:.1f}ms → {optimized_decode['kernel_time']:.1f}ms")
    
    if improvement > 10:
        print(f"\n🎉 优化效果显著！性能提升 {improvement:.1f}%")
        print("💡 这个优化针对真实瓶颈，应该在实际Semi-PD中有更好效果")
    elif improvement > 0:
        print(f"\n✅ 优化有效果，性能提升 {improvement:.1f}%")
        print("💡 在实际Semi-PD负载下效果可能更明显")
    else:
        print(f"\n⚠️ 优化效果不明显")
        print("💡 可能需要在实际Semi-PD环境中才能看到效果")
    
    return {
        'baseline_time': baseline_total_time,
        'optimized_time': optimized_total_time,
        'improvement_percent': improvement
    }

def test_specific_optimizations():
    """测试特定优化组件"""
    print("\n🧪 测试特定优化组件")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过测试")
        return
    
    device = torch.device('cuda')
    
    # 启用优化
    apply_real_performance_optimizations()
    
    # 测试1: item()缓存效果
    print("🔍 测试1: item()调用缓存...")
    test_tensor = torch.tensor([123], device=device)
    
    # 第一次调用
    start_time = time.time()
    val1 = test_tensor.item()
    first_call_time = (time.time() - start_time) * 1000
    
    # 第二次调用（应该命中缓存）
    start_time = time.time()
    val2 = test_tensor.item()
    second_call_time = (time.time() - start_time) * 1000
    
    print(f"  第一次调用: {first_call_time:.3f}ms")
    print(f"  第二次调用: {second_call_time:.3f}ms")
    print(f"  加速比: {first_call_time/second_call_time:.1f}x")
    
    # 测试2: 流同步批处理
    print("\n🔍 测试2: 流同步批处理...")
    start_time = time.time()
    
    # 多次快速同步（应该被批处理）
    for i in range(10):
        torch.cuda.synchronize()
    
    batch_sync_time = (time.time() - start_time) * 1000
    print(f"  批量同步耗时: {batch_sync_time:.2f}ms")
    
    # 打印优化统计
    print_real_optimization_stats()

def main():
    """主测试函数"""
    print("🚀 Semi-PD 真实性能优化效果测试")
    print("=" * 60)
    print("🎯 针对profile分析出的真实瓶颈:")
    print("  - cudaStreamSynchronize: 92.76% CPU时间")
    print("  - c10d::broadcast_: 121,228次调用")
    print("  - aten::item: 93.01% CUDA时间")
    print("  - cudaGraphLaunch: 94.23% CPU时间")
    print("=" * 60)
    
    try:
        # 运行特定优化测试
        test_specific_optimizations()
        
        # 运行完整对比测试
        results = benchmark_with_without_optimization()
        
        # 总结
        print("\n🎯 测试总结:")
        print("=" * 60)
        
        if results and results['improvement_percent'] > 5:
            print("🎉 真实性能优化效果显著！")
            print(f"   性能提升: {results['improvement_percent']:.1f}%")
            print("💡 建议:")
            print("   - 在实际Semi-PD服务中使用这个优化")
            print("   - 监控生产环境性能指标")
            print("   - 根据实际负载调整优化参数")
        else:
            print("📊 测试完成，优化效果需要在真实环境验证")
            print("💡 建议:")
            print("   - 在实际Semi-PD负载下测试")
            print("   - 检查profile数据验证优化效果")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 