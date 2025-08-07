#!/usr/bin/env python3
"""
精确分析真正的无效优化代码
避免误报，只找出确实有问题的优化
"""

import os
import re

def analyze_precise_ineffective_optimizations():
    """精确分析真正的无效优化"""
    
    print("🎯 精确无效优化分析报告")
    print("=" * 60)
    
    key_files = [
        "python/sglang/srt/model_executor/model_runner.py",
        "python/sglang/srt/managers/scheduler.py", 
        "python/sglang/srt/managers/schedule_batch.py",
        "python/sglang/srt/server_args.py"
    ]
    
    real_issues = []
    
    for file_path in key_files:
        if not os.path.exists(file_path):
            continue
            
        print(f"\n📋 检查文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        file_issues = []
        
        # 1. 检查未使用的role_suffix变量 (更精确)
        for i, line in enumerate(lines, 1):
            if 'role_suffix = getattr(self, \'instance_role\', \'UNKNOWN\')' in line:
                # 检查接下来的10行是否真的使用了这个变量
                used = False
                for j in range(i, min(i+15, len(lines))):
                    if j < len(lines):
                        next_line = lines[j]
                        # 排除定义行本身和只是hasattr检查的情况
                        if ('role_suffix' in next_line and 
                            next_line != line and 
                            'hasattr(role_suffix' not in next_line and
                            'role_suffix = role_suffix.name' not in next_line):
                            used = True
                            break
                
                if not used:
                    file_issues.append({
                        "line": i,
                        "type": "🗑️ 未使用变量",
                        "code": line.strip(),
                        "problem": "定义了role_suffix但从未真正使用",
                        "fix": "删除这个变量定义"
                    })
        
        # 2. 检查真正的阻塞性sleep (time.sleep(数字))
        for i, line in enumerate(lines, 1):
            # 只匹配真正的time.sleep调用，且睡眠时间>2秒
            if re.search(r'time\.sleep\s*\(\s*([3-9]|[1-9][0-9])\s*\)', line):
                file_issues.append({
                    "line": i,
                    "type": "🐌 阻塞性sleep",
                    "code": line.strip(),
                    "problem": "强制sleep超过2秒，会阻塞系统",
                    "fix": "使用异步等待或移除"
                })
        
        # 3. 检查频繁但无意义的角色信息获取
        role_getattr_pattern = r'role_suffix = getattr\(self, [\'"]instance_role[\'"], [\'"]UNKNOWN[\'"]?\)'
        role_matches = re.findall(role_getattr_pattern, content)
        if len(role_matches) > 8:  # 超过8次就认为是过度使用
            file_issues.append({
                "line": "多处",
                "type": "⚡ 频繁无用调用",
                "code": f"role_suffix = getattr(...) 出现 {len(role_matches)} 次",
                "problem": "频繁获取角色信息但大多未使用",
                "fix": "缓存角色信息或移除不必要的获取"
            })
        
        # 4. 检查Semi-PD相关的"伪优化"注释
        semi_pd_comments = []
        for i, line in enumerate(lines, 1):
            if re.search(r'#.*Semi-PD.*[优化|Skip|优化|异步]', line):
                semi_pd_comments.append((i, line.strip()))
        
        if len(semi_pd_comments) > 5:
            file_issues.append({
                "line": "多处",
                "type": "📝 过度Semi-PD注释",
                "code": f"发现 {len(semi_pd_comments)} 个Semi-PD相关注释",
                "problem": "过多的Semi-PD优化相关注释，增加维护负担",
                "fix": "保留必要的，移除调试性质的注释"
            })
        
        # 5. 检查已知的具体问题
        specific_issues = [
            (r'Semi-PD: Skip memory leak detection', "🚨 危险跳过", "跳过内存泄漏检测"),
            (r'PREFILL进程等待.*秒确保文件写入', "🐌 强制等待", "强制等待文件写入"),
            (r'改进的角色识别逻辑.*确保不会出现UNKNOWN', "🔄 过度复杂", "复杂的角色识别逻辑")
        ]
        
        for pattern, issue_type, description in specific_issues:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                file_issues.append({
                    "line": line_num,
                    "type": issue_type,
                    "code": lines[line_num-1].strip() if line_num <= len(lines) else "N/A",
                    "problem": description,
                    "fix": f"移除或简化此{description}"
                })
        
        if file_issues:
            real_issues.extend(file_issues)
            for issue in file_issues:
                print(f"   {issue['type']} (行 {issue['line']}): {issue['problem']}")
        else:
            print(f"   ✅ 未发现真正的无效优化")
    
    # 总结
    print("\n" + "=" * 60)
    print(f"📊 发现 {len(real_issues)} 个真正的无效优化问题")
    
    if real_issues:
        print("\n🔧 具体修复建议:")
        for i, issue in enumerate(real_issues, 1):
            print(f"{i}. {issue['type']}: {issue['fix']}")
    
    return real_issues

if __name__ == "__main__":
    analyze_precise_ineffective_optimizations() 