#!/usr/bin/env python3
"""
Semi-PD 真实性能瓶颈分析工具
基于实际profile数据找到根本问题
"""

import re
import sys
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

def parse_profile_file(file_path: str) -> Dict:
    """解析性能分析文件"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return {}
    
    print(f"📊 分析文件: {file_path}")
    
    operations = []
    total_cpu_time = 0
    total_cuda_time = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 查找表格数据
        data_started = False
        for line in lines:
            line = line.strip()
            if 'Self CPU %' in line and 'Self CUDA %' in line:
                data_started = True
                continue
                
            if data_started and line and not line.startswith('-'):
                # 解析数据行
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        name = parts[0]
                        self_cpu_percent = float(parts[1].replace('%', ''))
                        self_cpu_time = parts[2]
                        self_cuda_percent = float(parts[3].replace('%', ''))
                        self_cuda_time = parts[4]
                        calls = int(parts[-1]) if parts[-1].isdigit() else 0
                        
                        operations.append({
                            'name': name,
                            'self_cpu_percent': self_cpu_percent,
                            'self_cpu_time': self_cpu_time,
                            'self_cuda_percent': self_cuda_percent,
                            'self_cuda_time': self_cuda_time,
                            'calls': calls
                        })
                        
                        # 计算总时间
                        if 'total' in name.lower():
                            if 's' in self_cpu_time:
                                total_cpu_time = float(self_cpu_time.replace('s', ''))
                                
                    except (ValueError, IndexError):
                        continue
                        
    except Exception as e:
        print(f"❌ 解析文件出错: {e}")
        return {}
        
    return {
        'operations': operations,
        'total_cpu_time': total_cpu_time,
        'total_cuda_time': total_cuda_time
    }

def analyze_bottlenecks(profile_data: Dict) -> Dict:
    """分析性能瓶颈"""
    if not profile_data:
        return {}
        
    operations = profile_data['operations']
    
    # 按CPU时间排序
    cpu_bottlenecks = sorted(operations, key=lambda x: x['self_cpu_percent'], reverse=True)[:10]
    
    # 按CUDA时间排序
    cuda_bottlenecks = sorted(operations, key=lambda x: x['self_cuda_percent'], reverse=True)[:10]
    
    # 分类分析
    categories = {
        'memory_ops': [],
        'compute_ops': [],
        'communication_ops': [],
        'sync_ops': [],
        'kernel_ops': [],
        'other_ops': []
    }
    
    for op in operations:
        name = op['name'].lower()
        if any(keyword in name for keyword in ['memory', 'alloc', 'free', 'copy']):
            categories['memory_ops'].append(op)
        elif any(keyword in name for keyword in ['sync', 'wait', 'barrier']):
            categories['sync_ops'].append(op)
        elif any(keyword in name for keyword in ['nccl', 'allreduce', 'broadcast']):
            categories['communication_ops'].append(op)
        elif any(keyword in name for keyword in ['kernel', 'cuda', 'flash']):
            categories['kernel_ops'].append(op)
        elif any(keyword in name for keyword in ['matmul', 'linear', 'attention', 'silu']):
            categories['compute_ops'].append(op)
        else:
            categories['other_ops'].append(op)
    
    return {
        'cpu_bottlenecks': cpu_bottlenecks,
        'cuda_bottlenecks': cuda_bottlenecks,
        'categories': categories
    }

def get_optimization_suggestions(analysis: Dict) -> List[str]:
    """根据分析结果给出优化建议"""
    suggestions = []
    
    if not analysis:
        return ["❌ 无法分析，请检查数据"]
    
    cpu_bottlenecks = analysis.get('cpu_bottlenecks', [])
    categories = analysis.get('categories', {})
    
    # 检查主要瓶颈
    if cpu_bottlenecks:
        top_bottleneck = cpu_bottlenecks[0]
        
        if top_bottleneck['self_cpu_percent'] > 80:
            if 'item' in top_bottleneck['name'].lower():
                suggestions.append("🎯 **关键瓶颈**: GPU→CPU数据传输 (.item()调用)")
                suggestions.append("💡 优化方案: 减少.item()调用频率，使用批量处理")
                
            elif 'sync' in top_bottleneck['name'].lower():
                suggestions.append("🎯 **关键瓶颈**: CUDA流同步")
                suggestions.append("💡 优化方案: 优化CUDA流管理，减少不必要的同步")
                
            elif 'allreduce' in top_bottleneck['name'].lower():
                suggestions.append("🎯 **关键瓶颈**: 多GPU通信")
                suggestions.append("💡 优化方案: 优化AllReduce策略，考虑使用MSCCL++")
    
    # 检查特定类别的问题
    if categories.get('sync_ops'):
        sync_total = sum(op['self_cpu_percent'] for op in categories['sync_ops'])
        if sync_total > 50:
            suggestions.append(f"⚠️ **同步操作占用{sync_total:.1f}%CPU时间**")
            suggestions.append("💡 建议: 审查同步策略，考虑异步处理")
    
    if categories.get('memory_ops'):
        memory_total = sum(op['self_cpu_percent'] for op in categories['memory_ops'])
        if memory_total > 30:
            suggestions.append(f"⚠️ **内存操作占用{memory_total:.1f}%CPU时间**")
            suggestions.append("💡 建议: 优化内存分配策略，考虑内存池")
    
    if categories.get('communication_ops'):
        comm_total = sum(op['self_cpu_percent'] for op in categories['communication_ops'])
        if comm_total > 40:
            suggestions.append(f"⚠️ **通信操作占用{comm_total:.1f}%CPU时间**")
            suggestions.append("💡 建议: 优化通信拓扑，减少通信频率")
    
    # 检查调用频率
    high_freq_ops = [op for op in cpu_bottlenecks if op['calls'] > 1000]
    if high_freq_ops:
        suggestions.append("🔄 **高频调用操作检测到**:")
        for op in high_freq_ops[:3]:
            suggestions.append(f"   - {op['name']}: {op['calls']}次调用")
        suggestions.append("💡 建议: 考虑批量处理或缓存优化")
    
    if not suggestions:
        suggestions.append("✅ 未检测到明显性能瓶颈")
        suggestions.append("💡 建议: 进行更详细的profiling分析")
    
    return suggestions

def compare_profiles(baseline_path: str, optimized_path: str) -> Dict:
    """比较两个profile文件"""
    baseline_data = parse_profile_file(baseline_path)
    optimized_data = parse_profile_file(optimized_path)
    
    if not baseline_data or not optimized_data:
        return {}
    
    comparison = {}
    
    # 比较总时间
    if baseline_data.get('total_cpu_time') and optimized_data.get('total_cpu_time'):
        baseline_time = baseline_data['total_cpu_time']
        optimized_time = optimized_data['total_cpu_time']
        improvement = (baseline_time - optimized_time) / baseline_time * 100
        comparison['total_time_improvement'] = improvement
    
    # 比较主要操作
    baseline_ops = {op['name']: op for op in baseline_data.get('operations', [])}
    optimized_ops = {op['name']: op for op in optimized_data.get('operations', [])}
    
    operation_improvements = {}
    for op_name in baseline_ops:
        if op_name in optimized_ops:
            baseline_percent = baseline_ops[op_name]['self_cpu_percent']
            optimized_percent = optimized_ops[op_name]['self_cpu_percent']
            if baseline_percent > 0:
                improvement = (baseline_percent - optimized_percent) / baseline_percent * 100
                operation_improvements[op_name] = improvement
    
    comparison['operation_improvements'] = operation_improvements
    return comparison

def main():
    print("🔍 Semi-PD 性能瓶颈深度分析")
    print("=" * 50)
    
    # 分析PREFILL性能
    prefill_path = "/home/yzh/semi_pd_migration/profile/0.4.8_4k_d64/stats_semipd_PREFILL_1754231730.txt"
    
    print("\n📊 分析PREFILL阶段性能...")
    prefill_data = parse_profile_file(prefill_path)
    
    if prefill_data:
        prefill_analysis = analyze_bottlenecks(prefill_data)
        
        print("\n🎯 **PREFILL阶段 - Top 5 CPU瓶颈**:")
        for i, op in enumerate(prefill_analysis['cpu_bottlenecks'][:5], 1):
            print(f"  {i}. {op['name']}: {op['self_cpu_percent']:.2f}% ({op['calls']}次调用)")
        
        print("\n🎯 **PREFILL阶段 - Top 5 CUDA瓶颈**:")
        for i, op in enumerate(prefill_analysis['cuda_bottlenecks'][:5], 1):
            print(f"  {i}. {op['name']}: {op['self_cuda_percent']:.2f}% ({op['calls']}次调用)")
        
        print("\n💡 **PREFILL优化建议**:")
        suggestions = get_optimization_suggestions(prefill_analysis)
        for suggestion in suggestions:
            print(f"  {suggestion}")
    
    # 分析DECODE性能
    decode_path = "/home/yzh/semi_pd_migration/profile/0.4.8_4k_d64/stats_semipd_DECODE_1754231702.txt"
    
    print("\n" + "=" * 50)
    print("📊 分析DECODE阶段性能...")
    decode_data = parse_profile_file(decode_path)
    
    if decode_data:
        decode_analysis = analyze_bottlenecks(decode_data)
        
        print("\n🎯 **DECODE阶段 - Top 5 CPU瓶颈**:")
        for i, op in enumerate(decode_analysis['cpu_bottlenecks'][:5], 1):
            print(f"  {i}. {op['name']}: {op['self_cpu_percent']:.2f}% ({op['calls']}次调用)")
        
        print("\n🎯 **DECODE阶段 - Top 5 CUDA瓶颈**:")
        for i, op in enumerate(decode_analysis['cuda_bottlenecks'][:5], 1):
            print(f"  {i}. {op['name']}: {op['self_cuda_percent']:.2f}% ({op['calls']}次调用)")
        
        print("\n💡 **DECODE优化建议**:")
        suggestions = get_optimization_suggestions(decode_analysis)
        for suggestion in suggestions:
            print(f"  {suggestion}")
    
    # 对比分析
    baseline_prefill = "/home/yzh/semi_pd_migration/profile/0.4.4_4k_d64/prefill.txt"
    baseline_decode = "/home/yzh/semi_pd_migration/profile/0.4.4_4k_d64/decoder.txt"
    
    print("\n" + "=" * 50)
    print("📈 性能对比分析 (0.4.4 vs 0.4.8)...")
    
    if os.path.exists(baseline_prefill):
        print("\n🔄 PREFILL对比:")
        prefill_comparison = compare_profiles(baseline_prefill, prefill_path)
        if prefill_comparison:
            total_improvement = prefill_comparison.get('total_time_improvement', 0)
            print(f"  总体性能变化: {total_improvement:+.1f}%")
            
            print("  主要操作变化:")
            for op_name, improvement in list(prefill_comparison.get('operation_improvements', {}).items())[:5]:
                print(f"    - {op_name}: {improvement:+.1f}%")
    
    if os.path.exists(baseline_decode):
        print("\n🔄 DECODE对比:")
        decode_comparison = compare_profiles(baseline_decode, decode_path)
        if decode_comparison:
            total_improvement = decode_comparison.get('total_time_improvement', 0)
            print(f"  总体性能变化: {total_improvement:+.1f}%")
            
            print("  主要操作变化:")
            for op_name, improvement in list(decode_comparison.get('operation_improvements', {}).items())[:5]:
                print(f"    - {op_name}: {improvement:+.1f}%")
    
    print("\n" + "=" * 50)
    print("🎯 **总体优化策略建议**:")
    print("1. 🚀 **立即可行的优化**:")
    print("   - 减少.item()调用频率")
    print("   - 优化CUDA流管理")
    print("   - 使用更高效的kernel")
    
    print("\n2. 🔧 **中期优化方向**:")
    print("   - 优化内存分配策略")
    print("   - 改进多GPU通信")
    print("   - 调整批处理大小")
    
    print("\n3. ⚡ **深度优化考虑**:")
    print("   - 自定义kernel开发")
    print("   - 混合精度优化")
    print("   - 流水线并行优化")

if __name__ == "__main__":
    main() 