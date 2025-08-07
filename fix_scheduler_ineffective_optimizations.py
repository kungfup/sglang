#!/usr/bin/env python3
"""
修复scheduler.py中的无效优化代码
恢复到原生版本的高效实现
"""

import re

def fix_scheduler_optimizations():
    """修复scheduler.py中的无效优化"""
    
    scheduler_path = "python/sglang/srt/managers/scheduler.py"
    
    print("🔧 开始修复scheduler.py中的无效优化...")
    
    # 读取文件
    with open(scheduler_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_lines = len(content.split('\n'))
    
    # 修复1: 移除内存泄漏检测的跳过逻辑
    print("1. 恢复内存泄漏检测机制...")
    
    # 找到并移除Semi-PD跳过逻辑
    memory_check_pattern = r'(def check_memory\(self\):)\s*\n\s*# Semi-PD: Skip memory leak detection in Semi-PD mode\s*\n\s*if self\.server_args\.enable_semi_pd:\s*\n\s*return\s*\n'
    
    if re.search(memory_check_pattern, content):
        content = re.sub(memory_check_pattern, r'\1\n', content)
        print("   ✅ 已移除内存泄漏检测跳过逻辑")
    else:
        print("   ⚠️ 未找到内存泄漏检测跳过逻辑")
    
    # 修复2: 移除run_batch中未使用的role_suffix变量
    print("2. 移除未使用的角色信息变量...")
    
    # 在run_batch方法中查找并移除未使用的role_suffix定义
    run_batch_pattern = r'(\s+)# 记录当前角色信息和每100次前向的批次状态\s*\n\s*role_suffix = getattr\(self, \'instance_role\', \'UNKNOWN\'\)\s*\n\s*if hasattr\(role_suffix, \'name\'\):\s*\n\s*role_suffix = role_suffix\.name\s*\n\s*\n'
    
    if re.search(run_batch_pattern, content):
        content = re.sub(run_batch_pattern, r'\1', content)
        print("   ✅ 已移除run_batch中未使用的role_suffix变量")
    else:
        print("   ⚠️ 未找到run_batch中的未使用变量")
    
    # 修复3: 移除强制sleep等待
    print("3. 移除强制sleep等待...")
    
    # 移除PREFILL进程的强制等待
    sleep_pattern = r'\s*# 如果是PREFILL进程，等待一段时间确保文件写入完成\s*\n\s*role_suffix = getattr\(self, \'instance_role\', None\)\s*\n\s*if hasattr\(role_suffix, \'name\'\) and role_suffix\.name == \'PREFILL\':\s*\n\s*logger\.info\("PREFILL进程等待3秒确保文件写入完成\.\.\."\)\s*\n\s*time\.sleep\(3\)\s*\n'
    
    if re.search(sleep_pattern, content):
        content = re.sub(sleep_pattern, '', content)
        print("   ✅ 已移除强制sleep等待")
    else:
        print("   ⚠️ 未找到强制sleep等待代码")
    
    # 修复4: 简化复杂的角色识别逻辑
    print("4. 简化角色识别逻辑...")
    
    # 简化过度复杂的角色识别
    complex_role_pattern = r'(\s+)# 改进的角色识别逻辑，确保不会出现UNKNOWN\s*\n\s*role_suffix = getattr\(self, \'instance_role\', None\)\s*\n\s*if hasattr\(role_suffix, \'name\'\):\s*\n\s*role_suffix = role_suffix\.name\s*\n\s*\n(\s*# 如果仍为None，尝试从服务器参数获取.*?)\n(.*?)(\s*# 强制记录角色信息，即使为UNKNOWN也输出)'
    
    # 使用更简单的替换
    simple_replacement = r'\1role_suffix = getattr(self, "instance_role", "UNKNOWN")\n\1if hasattr(role_suffix, "name"):\n\1    role_suffix = role_suffix.name\n\4'
    
    if re.search(r'# 改进的角色识别逻辑，确保不会出现UNKNOWN', content):
        # 简化整个复杂的角色识别块
        lines = content.split('\n')
        new_lines = []
        skip_until_logger = False
        
        for line in lines:
            if '# 改进的角色识别逻辑，确保不会出现UNKNOWN' in line:
                skip_until_logger = True
                # 添加简化的版本
                indent = ' ' * (len(line) - len(line.lstrip()))
                new_lines.append(f'{indent}role_suffix = getattr(self, "instance_role", "UNKNOWN")')
                new_lines.append(f'{indent}if hasattr(role_suffix, "name"):')
                new_lines.append(f'{indent}    role_suffix = role_suffix.name')
                new_lines.append('')
                continue
            
            if skip_until_logger and 'logger.info(f"Semi-PD Profiler Role: {role_suffix}")' in line:
                skip_until_logger = False
                new_lines.append(line)
                continue
            
            if not skip_until_logger:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        print("   ✅ 已简化复杂的角色识别逻辑")
    else:
        print("   ⚠️ 未找到复杂的角色识别逻辑")
    
    # 修复5: 移除调试代码中的无用角色获取（保留有用的）
    print("5. 优化调试代码...")
    
    # 在_profile_batch_predicate中，角色信息是有用的，保留
    # 但可以优化为缓存版本
    
    # 写入修复后的文件
    with open(scheduler_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    new_lines = len(content.split('\n'))
    
    print()
    print("✅ 修复完成！")
    print(f"📊 文件行数变化: {original_lines} → {new_lines} (减少 {original_lines - new_lines} 行)")
    print()
    print("🎯 修复效果:")
    print("- ✅ 恢复了内存泄漏检测的安全性")
    print("- ✅ 移除了每次batch运行时的无用角色获取")
    print("- ✅ 消除了阻塞性的强制sleep")
    print("- ✅ 简化了过度复杂的角色识别逻辑")
    print("- ✅ 保留了有用的调试功能")
    
    return True

if __name__ == "__main__":
    fix_scheduler_optimizations() 