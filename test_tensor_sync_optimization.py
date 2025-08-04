#!/usr/bin/env python3
"""
测试张量同步优化效果
验证 .item() 调用的缓存和批量处理是否有效
"""

import os
import time
import torch
import logging

# 设置环境变量
os.environ['SEMI_PD_ASYNC_OPT_ENABLED'] = '1'
os.environ['SEMI_PD_CACHE_SIZE'] = '1000' 
os.environ['SEMI_PD_CACHE_TTL'] = '0.1'

# 导入优化模块
from sglang.srt.utils_async_optimization import (
    apply_tensor_optimization_patches,
    get_sync_optimization_stats,
    print_sync_optimization_stats,
    batch_extract_items
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tensor_sync_optimization():
    """测试张量同步优化效果"""
    print("🧪 测试 Semi-PD 张量同步优化")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，跳过测试")
        return
        
    device = torch.device('cuda')
    
    # 应用优化补丁
    print("🚀 应用张量同步优化补丁...")
    apply_tensor_optimization_patches()
    
    # 测试数据
    test_tensors = []
    print("📊 创建测试张量...")
    for i in range(20):
        # 创建不同类型的张量来测试
        seq_len = torch.tensor([128 + i], device=device, dtype=torch.int32)
        mask_sum = torch.tensor([64 + i], device=device, dtype=torch.int32) 
        max_len = torch.tensor([256 + i], device=device, dtype=torch.int32)
        
        test_tensors.extend([seq_len, mask_sum, max_len])
    
    print(f"✅ 创建了 {len(test_tensors)} 个测试张量")
    
    # 测试1: 传统方式（多次单独调用）
    print("\n🔄 测试1: 传统 .item() 调用...")
    start_time = time.time()
    
    traditional_results = []
    for tensor in test_tensors:
        value = tensor.item()  # 这会触发优化
        traditional_results.append(value)
        
    traditional_time = time.time() - start_time
    
    # 测试2: 批量方式
    print("🔄 测试2: 批量 .item() 调用...")
    start_time = time.time()
    
    batch_results = batch_extract_items(test_tensors)
    
    batch_time = time.time() - start_time
    
    # 测试3: 重复调用（测试缓存效果）
    print("🔄 测试3: 重复调用（测试缓存）...")
    start_time = time.time()
    
    cached_results = []
    for tensor in test_tensors[:10]:  # 重复调用前10个
        value = tensor.item()
        cached_results.append(value)
        
    cached_time = time.time() - start_time
    
    # 结果对比
    print("\n📈 性能测试结果:")
    print(f"  - 传统方式: {traditional_time*1000:.2f}ms")
    print(f"  - 批量方式: {batch_time*1000:.2f}ms") 
    print(f"  - 缓存测试: {cached_time*1000:.2f}ms")
    
    if batch_time < traditional_time:
        improvement = (traditional_time - batch_time) / traditional_time * 100
        print(f"  🚀 批量方式提升: {improvement:.1f}%")
    
    # 验证结果正确性
    if traditional_results == batch_results:
        print("✅ 结果验证通过：传统方式 == 批量方式")
    else:
        print("❌ 结果验证失败：结果不一致")
    
    # 打印优化统计
    print("\n" + "="*50)
    print_sync_optimization_stats()
    
    return get_sync_optimization_stats()

def test_common_patterns():
    """测试常见的张量操作模式"""
    print("\n🎯 测试常见张量操作模式")
    print("=" * 50)
    
    device = torch.device('cuda')
    
    # 模拟注意力机制中的常见操作
    batch_size = 16
    seq_lens = torch.randint(100, 500, (batch_size,), device=device)
    attention_masks = torch.randint(0, 2, (batch_size, 512), device=device, dtype=torch.bool)
    
    print("🔍 测试序列长度提取...")
    start_time = time.time()
    
    # 常见模式1: 获取最大序列长度
    max_seq_len = seq_lens.max().item()
    
    # 常见模式2: 获取掩码统计
    mask_counts = []
    for i in range(batch_size):
        count = attention_masks[i].sum().item()
        mask_counts.append(count)
    
    # 常见模式3: 序列长度总和
    total_tokens = seq_lens.sum().item()
    
    pattern_time = time.time() - start_time
    
    print(f"  - 最大序列长度: {max_seq_len}")
    print(f"  - 总token数: {total_tokens}")
    print(f"  - 处理时间: {pattern_time*1000:.2f}ms")
    
    return pattern_time

def main():
    """主测试函数"""
    print("🚀 Semi-PD 张量同步优化测试")
    print("=" * 60)
    
    try:
        # 基础优化测试
        stats = test_tensor_sync_optimization()
        
        # 常见模式测试  
        pattern_time = test_common_patterns()
        
        # 总结
        print("\n📋 测试总结:")
        print("=" * 60)
        print(f"✅ 总 .item() 调用: {stats['total_item_calls']}")
        print(f"💾 缓存命中次数: {stats['cached_item_calls']}")
        print(f"🎯 缓存命中率: {stats['cache_hit_rate']:.1%}")
        print(f"⏱️  节省同步时间: {stats['estimated_sync_time_saved_ms']:.1f}ms")
        
        if stats['cache_hit_rate'] > 0.3:
            print("🎉 优化效果显著！缓存命中率良好")
        elif stats['total_item_calls'] > 0:
            print("⚠️  优化效果一般，考虑调整缓存策略")
        else:
            print("❌ 未检测到优化效果")
            
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 