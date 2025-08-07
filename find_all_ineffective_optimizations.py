#!/usr/bin/env python3
"""
全面分析sglang_0.4.8中的无效优化代码
找出所有看起来像优化但实际无效甚至有害的代码
"""

import os
import subprocess
import re

def find_ineffective_optimizations():
    """全面查找无效优化"""
    
    print("🔍 sglang_0.4.8 全面无效优化分析报告")
    print("=" * 70)
    
    # 要检查的无效优化模式
    patterns = {
        "无用变量定义": [
            r"role_suffix = getattr.*但未使用",
            r"定义.*从未引用的变量"
        ],
        "危险的跳过逻辑": [
            r"Skip.*memory.*leak.*detection",
            r"Semi-PD.*Skip.*检测"
        ],
        "阻塞性代码": [
            r"time\.sleep\([^0-1]",  # sleep超过1秒
            r"强制.*等待.*秒",
            r"阻塞.*整个.*进程"
        ],
        "过度复杂化": [
            r"改进.*角色识别.*逻辑",
            r"复杂.*但仍.*UNKNOWN"
        ],
        "调试/日志开销": [
            r"getattr.*instance_role.*每次调用",
            r"频繁.*角色.*获取"
        ],
        "中文注释优化": [
            r"# 记录.*角色.*信息",
            r"# 优化.*异步.*处理",
            r"# Semi-PD.*优化"
        ]
    }
    
    # 检查的文件
    key_files = [
        "python/sglang/srt/model_executor/model_runner.py",
        "python/sglang/srt/managers/scheduler.py", 
        "python/sglang/srt/managers/schedule_batch.py",
        "python/sglang/srt/server_args.py",
        "python/sglang/srt/utils.py"
    ]
    
    issues_found = []
    
    for file_path in key_files:
        if not os.path.exists(file_path):
            continue
            
        print(f"\n📋 检查文件: {file_path}")
        
        # 比较行数
        try:
            original_path = file_path.replace("sglang_0.4.8", "sglang_origin_0.4.8")
            original_path = "../" + original_path.replace("python/", "")
            
            current_lines = sum(1 for _ in open(file_path))
            if os.path.exists(original_path):
                original_lines = sum(1 for _ in open(original_path))
                diff = current_lines - original_lines
                if diff > 0:
                    print(f"   📊 行数增加: +{diff} 行 ({current_lines} vs {original_lines})")
            else:
                print(f"   📊 当前行数: {current_lines}")
        except:
            pass
        
        # 搜索具体的无效模式
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            # 检查特定的无效优化模式
            file_issues = []
            
            # 1. 检查未使用的role_suffix变量
            for i, line in enumerate(lines, 1):
                if 'role_suffix = getattr(self, \'instance_role\', \'UNKNOWN\')' in line:
                    # 检查接下来几行是否使用了这个变量
                    used = False
                    for j in range(i, min(i+10, len(lines))):
                        if j < len(lines) and 'role_suffix' in lines[j] and lines[j] != line:
                            used = True
                            break
                    if not used:
                        file_issues.append({
                            "line": i,
                            "type": "🗑️ 未使用变量",
                            "code": line.strip(),
                            "problem": "定义了role_suffix但从未使用"
                        })
            
            # 2. 检查危险的跳过逻辑
            for i, line in enumerate(lines, 1):
                if 'Semi-PD: Skip' in line and 'memory leak detection' in line:
                    file_issues.append({
                        "line": i,
                        "type": "🚨 危险跳过",
                        "code": line.strip(),
                        "problem": "跳过内存泄漏检测"
                    })
            
            # 3. 检查强制sleep
            for i, line in enumerate(lines, 1):
                if re.search(r'time\.sleep\([3-9]|[1-9][0-9]', line):
                    file_issues.append({
                        "line": i,
                        "type": "🐌 阻塞代码",
                        "code": line.strip(),
                        "problem": "强制sleep超过2秒"
                    })
            
            # 4. 检查频繁的角色获取
            role_getattr_count = content.count('getattr(self, \'instance_role\'')
            if role_getattr_count > 5:
                file_issues.append({
                    "line": "多处",
                    "type": "⚡ 性能开销",
                    "code": f"getattr(self, 'instance_role') 出现 {role_getattr_count} 次",
                    "problem": "频繁获取角色信息，应该缓存"
                })
            
            # 5. 检查中文注释的优化
            chinese_optimization_comments = 0
            for i, line in enumerate(lines, 1):
                if re.search(r'#.*记录.*角色|#.*优化.*异步|#.*Semi-PD.*优化', line):
                    chinese_optimization_comments += 1
            
            if chinese_optimization_comments > 3:
                file_issues.append({
                    "line": "多处",
                    "type": "📝 过度注释",
                    "code": f"包含 {chinese_optimization_comments} 个优化相关中文注释",
                    "problem": "过多的调试/优化注释，增加维护负担"
                })
            
            if file_issues:
                issues_found.extend(file_issues)
                for issue in file_issues:
                    print(f"   {issue['type']} (行 {issue['line']}): {issue['problem']}")
            else:
                print(f"   ✅ 未发现明显的无效优化")
                
        except Exception as e:
            print(f"   ❌ 读取文件失败: {e}")
    
    # 总结报告
    print("\n" + "=" * 70)
    print(f"📊 总结: 发现 {len(issues_found)} 个潜在的无效优化")
    
    if issues_found:
        print("\n🔧 建议的修复操作:")
        print("1. 移除未使用的role_suffix变量定义")
        print("2. 恢复被跳过的安全检测逻辑")
        print("3. 替换阻塞性的sleep调用")
        print("4. 缓存频繁获取的角色信息")
        print("5. 清理过多的调试注释")
    else:
        print("\n✅ 恭喜！未发现明显的无效优化")
    
    print("\n⚡ 预期收益:")
    print("- 减少CPU开销（移除无用计算）")
    print("- 提高安全性（恢复检测机制）")
    print("- 减少阻塞时间（消除不必要的sleep）")
    print("- 简化代码维护（清理冗余注释）")

if __name__ == "__main__":
    find_ineffective_optimizations() 