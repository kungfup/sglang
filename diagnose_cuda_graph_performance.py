#!/usr/bin/env python3
"""
CUDA Graph 性能诊断脚本
目标: 定位58ms CUDA Graph replay延迟的根本原因
"""

import os
import time
import re
from pathlib import Path

def analyze_cuda_graph_performance_issue():
    """分析CUDA Graph性能问题"""
    
    print("🔍 [DIAGNOSIS] CUDA Graph 性能问题诊断")
    print("="*70)
    
    print("📋 性能对比分析:")
    print("  原生 0.4.4: cudaGraphLaunch = 1.4ms/call (正常)")
    print("  迁移 0.4.8: cudaGraphLaunch = 58.7ms/call (异常)")
    print("  性能回退: +4213% (42倍慢)")
    print()
    
    # 分析可能的原因
    possible_causes = [
        {
            "name": "CUDA Graph 重复capture",
            "description": "recapture_if_needed被频繁调用",
            "check_method": "check_recapture_frequency",
            "severity": "HIGH"
        },
        {
            "name": "性能监控代码残留",
            "description": "热路径中仍有time.perf_counter()调用",
            "check_method": "check_performance_monitoring",
            "severity": "HIGH"
        },
        {
            "name": "内存同步问题",
            "description": "过多的torch.cuda.synchronize()调用",
            "check_method": "check_cuda_synchronization",
            "severity": "MEDIUM"
        },
        {
            "name": "CUDA Graph capture逻辑变化",
            "description": "0.4.8版本capture逻辑与0.4.4不同",
            "check_method": "check_capture_logic",
            "severity": "HIGH"
        },
        {
            "name": "批处理大小不匹配",
            "description": "CUDA Graph的batch size配置问题",
            "check_method": "check_batch_size_config",
            "severity": "MEDIUM"
        }
    ]
    
    results = {}
    
    for cause in possible_causes:
        print(f"🔧 [{cause['severity']}] 检查: {cause['name']}")
        check_func = globals().get(cause['check_method'])
        if check_func:
            results[cause['name']] = check_func()
        else:
            results[cause['name']] = "未实现检查"
        print()
    
    # 生成诊断报告
    generate_diagnosis_report(results)

def check_recapture_frequency():
    """检查CUDA Graph重新capture的频率"""
    
    print("  🔍 分析recapture_if_needed逻辑...")
    
    issues = []
    file_path = "python/sglang/srt/model_executor/cuda_graph_runner.py"
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 检查recapture条件
        if "if self.capture_hidden_mode != required_capture_hidden_mode:" in content:
            if "self.capture()" in content:
                issues.append("✅ recapture逻辑存在，但条件可能过于频繁")
        
        # 检查是否有简化的recapture逻辑
        if "SEMI_PD_FIX: 简化判断逻辑" in content:
            issues.append("⚠️ 发现简化逻辑注释，可能有修改")
        
        # 检查capture方法本身
        capture_count = content.count("def capture")
        if capture_count > 0:
            issues.append(f"✅ 找到{capture_count}个capture方法定义")
    
    if not issues:
        issues.append("❌ 无法分析，文件可能不存在")
    
    for issue in issues:
        print(f"    {issue}")
    
    return issues

def check_performance_monitoring():
    """检查性能监控代码残留"""
    
    print("  🔍 搜索性能监控代码残留...")
    
    issues = []
    monitoring_patterns = [
        r"time\.perf_counter\(\)",
        r"print.*ms.*replay",
        r"print.*CUDA.*Graph",
        r"torch\.cuda\.synchronize\(\).*# 性能",
        r"DEEP_CUDA_GRAPH_DIAGNOSIS",
        r"CUDA Graph replay.*seconds"
    ]
    
    files_to_check = [
        "python/sglang/srt/model_executor/cuda_graph_runner.py",
        "python/sglang/srt/model_executor/model_runner.py"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            
            for pattern in monitoring_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    issues.append(f"⚠️ {file_path}: 发现{len(matches)}个'{pattern}'匹配")
        else:
            issues.append(f"❌ 文件不存在: {file_path}")
    
    if not issues:
        issues.append("✅ 未发现明显的性能监控代码残留")
    
    for issue in issues:
        print(f"    {issue}")
    
    return issues

def check_cuda_synchronization():
    """检查CUDA同步调用"""
    
    print("  🔍 分析CUDA同步调用频率...")
    
    issues = []
    file_path = "python/sglang/srt/model_executor/cuda_graph_runner.py"
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 统计同步调用
        sync_count = content.count("torch.cuda.synchronize()")
        empty_cache_count = content.count("torch.cuda.empty_cache()")
        
        issues.append(f"📊 torch.cuda.synchronize(): {sync_count}次")
        issues.append(f"📊 torch.cuda.empty_cache(): {empty_cache_count}次")
        
        if sync_count > 5:
            issues.append("⚠️ CUDA同步调用可能过多")
        
        # 检查replay方法中的同步
        replay_section = re.search(r'def replay\(.*?\n(.*?)\n    def ', content, re.DOTALL)
        if replay_section:
            replay_code = replay_section.group(1)
            replay_sync_count = replay_code.count("torch.cuda.synchronize()")
            issues.append(f"📊 replay方法中的同步: {replay_sync_count}次")
            
            if replay_sync_count > 1:
                issues.append("🚨 replay方法中同步过多，可能是性能瓶颈")
    
    for issue in issues:
        print(f"    {issue}")
    
    return issues

def check_capture_logic():
    """检查CUDA Graph capture逻辑变化"""
    
    print("  🔍 对比capture逻辑...")
    
    issues = []
    
    # 检查是否存在原版文件用于对比
    current_file = "python/sglang/srt/model_executor/cuda_graph_runner.py"
    semipd_file = "python/sglang/srt/model_executor/cuda_graph_runner.semipd"
    new_file = "python/sglang/srt/model_executor/cuda_graph_runner.new"
    
    files_exist = {
        "current": os.path.exists(current_file),
        "semipd": os.path.exists(semipd_file),
        "new": os.path.exists(new_file)
    }
    
    issues.append(f"📁 文件存在情况: {files_exist}")
    
    if files_exist["current"] and files_exist["new"]:
        # 简单的行数对比
        with open(current_file, 'r') as f:
            current_lines = len(f.readlines())
        with open(new_file, 'r') as f:
            new_lines = len(f.readlines())
        
        issues.append(f"📊 行数对比: current={current_lines}, new={new_lines}")
        
        if abs(current_lines - new_lines) > 50:
            issues.append("⚠️ 文件差异较大，可能有逻辑变化")
    
    for issue in issues:
        print(f"    {issue}")
    
    return issues

def check_batch_size_config():
    """检查批处理大小配置"""
    
    print("  🔍 分析批处理大小配置...")
    
    issues = []
    
    # 从性能报告中提取的信息
    original_calls = 618
    migrated_calls = 545
    
    issues.append(f"📊 调用次数变化: {original_calls} → {migrated_calls} ({((migrated_calls-original_calls)/original_calls*100):+.1f}%)")
    
    # 检查capture batch size配置
    file_path = "python/sglang/srt/model_executor/cuda_graph_runner.py"
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 寻找batch size相关配置
        if "get_batch_sizes_to_capture" in content:
            issues.append("✅ 找到batch size capture逻辑")
        
        if "self.capture_bs" in content:
            issues.append("✅ 找到capture_bs配置")
        
        # 检查是否有固定的batch size限制
        bs_patterns = [
            r"max_bs.*=.*(\d+)",
            r"capture_bs.*=.*\[([^\]]+)\]"
        ]
        
        for pattern in bs_patterns:
            matches = re.findall(pattern, content)
            if matches:
                issues.append(f"📊 发现batch size配置: {matches}")
    
    for issue in issues:
        print(f"    {issue}")
    
    return issues

def generate_diagnosis_report(results):
    """生成诊断报告"""
    
    print("📋 [DIAGNOSIS REPORT] CUDA Graph 性能问题诊断报告")
    print("="*70)
    
    # 问题优先级分析
    high_priority_issues = []
    medium_priority_issues = []
    
    for check_name, result in results.items():
        if isinstance(result, list):
            for issue in result:
                if "🚨" in issue or "异常" in issue:
                    high_priority_issues.append(f"{check_name}: {issue}")
                elif "⚠️" in issue:
                    medium_priority_issues.append(f"{check_name}: {issue}")
    
    print("🚨 高优先级问题:")
    if high_priority_issues:
        for issue in high_priority_issues:
            print(f"  • {issue}")
    else:
        print("  • 未发现明显的高优先级问题")
    
    print("\n⚠️ 中优先级问题:")
    if medium_priority_issues:
        for issue in medium_priority_issues:
            print(f"  • {issue}")
    else:
        print("  • 未发现明显的中优先级问题")
    
    print("\n💡 推荐修复方案:")
    recommendations = [
        "1. 检查replay()方法中的torch.cuda.synchronize()调用，移除不必要的同步",
        "2. 确认recapture_if_needed()的触发频率，可能需要优化判断条件",
        "3. 对比原生0.4.4的CUDA Graph实现，找出性能回退的具体原因",
        "4. 添加详细的replay耗时监控，定位具体的性能瓶颈点",
        "5. 考虑暂时禁用CUDA Graph来验证其他功能是否正常"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print(f"\n📊 总结:")
    print(f"  • CUDA Graph Launch性能回退: +4213% (关键问题)")
    print(f"  • cudaMemcpyAsync已修复: -99.6% (成功)")
    print(f"  • 核心GEMM计算正常: ~90% (正常)")
    print(f"  • 建议优先修复CUDA Graph replay逻辑")

def create_quick_fix_script():
    """创建快速修复脚本"""
    
    print("\n🔧 [QUICK FIX] 创建快速修复脚本...")
    
    fix_script = """#!/bin/bash
# CUDA Graph 性能快速修复脚本

echo "🔧 开始CUDA Graph性能修复..."

# 1. 备份当前文件
cp python/sglang/srt/model_executor/cuda_graph_runner.py python/sglang/srt/model_executor/cuda_graph_runner.py.backup

# 2. 移除可能的性能监控代码
echo "📝 移除性能监控代码..."
sed -i '/time\.perf_counter/d' python/sglang/srt/model_executor/cuda_graph_runner.py
sed -i '/print.*CUDA.*Graph/d' python/sglang/srt/model_executor/cuda_graph_runner.py

# 3. 优化CUDA同步
echo "⚡ 优化CUDA同步调用..."
# 这需要手动检查和修改

# 4. 测试
echo "🧪 建议运行小规模测试验证修复效果"

echo "✅ 快速修复完成，请手动验证结果"
"""
    
    with open("quick_cuda_graph_fix.sh", "w") as f:
        f.write(fix_script)
    
    os.chmod("quick_cuda_graph_fix.sh", 0o755)
    print("  ✅ 快速修复脚本已创建: quick_cuda_graph_fix.sh")

def main():
    """主函数"""
    
    analyze_cuda_graph_performance_issue()
    create_quick_fix_script()
    
    print("\n🎯 [CONCLUSION] 结论:")
    print("  核心问题: CUDA Graph replay从1.4ms暴涨到58.7ms")
    print("  可能原因: replay()方法中的性能回退")
    print("  修复方向: 优化CUDA Graph的replay逻辑")
    print("  紧急措施: 如果修复困难，可考虑临时禁用CUDA Graph")

if __name__ == "__main__":
    main() 