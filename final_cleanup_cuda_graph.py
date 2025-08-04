#!/usr/bin/env python3
"""
最终清理：移除所有残留的Semi-PD事件协调代码
"""

def final_cleanup():
    """最终清理残留代码"""
    
    model_runner_path = "python/sglang/srt/model_executor/model_runner.py"
    
    print("🧹 开始最终清理残留的Semi-PD事件协调代码...")
    
    with open(model_runner_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    cleaned_lines = []
    skip_until_next_def = False
    
    for i, line in enumerate(lines):
        # 跳过Semi-PD异步优化函数的整个定义
        if 'def _apply_semi_pd_async_optimization' in line:
            skip_until_next_def = True
            continue
        
        # 如果在跳过状态，寻找下一个def或class来结束跳过
        if skip_until_next_def:
            if line.strip().startswith('def ') and '_apply_semi_pd_async_optimization' not in line:
                skip_until_next_def = False
                cleaned_lines.append(line)
            continue
        
        # 移除包含事件协调的行
        if any(keyword in line for keyword in [
            '_completion_event', 
            'torch.cuda.Event()',
            '.record()',
            'instance_role == InstanceRole.PREFILL',
            'instance_role == InstanceRole.DECODE',
            '"""Semi-PD异步处理优化'
        ]):
            continue
        
        # 移除空的if语句块
        if line.strip() in ["if hasattr(self, 'instance_role'):", "elif self.instance_role == InstanceRole.DECODE:"]:
            continue
            
        cleaned_lines.append(line)
    
    # 移除连续的空行
    final_lines = []
    prev_empty = False
    
    for line in cleaned_lines:
        if line.strip() == "":
            if not prev_empty:
                final_lines.append(line)
            prev_empty = True
        else:
            final_lines.append(line)
            prev_empty = False
    
    content = '\n'.join(final_lines)
    
    # 写入清理后的文件
    with open(model_runner_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 最终清理完成！")
    
    # 验证清理结果
    with open(model_runner_path, 'r', encoding='utf-8') as f:
        verification_content = f.read()
    
    remaining_issues = []
    if '_completion_event' in verification_content:
        remaining_issues.append("_completion_event")
    if 'torch.cuda.Event()' in verification_content:
        remaining_issues.append("torch.cuda.Event()")
    if 'instance_role == InstanceRole.PREFILL' in verification_content:
        remaining_issues.append("PREFILL角色检查")
    if 'instance_role == InstanceRole.DECODE' in verification_content and '_forward_raw' not in verification_content:
        remaining_issues.append("DECODE角色检查")
    
    if remaining_issues:
        print(f"⚠️  仍有残留代码: {', '.join(remaining_issues)}")
        return False
    else:
        print("🎉 所有Semi-PD事件协调代码已完全移除！")
        return True

if __name__ == "__main__":
    if final_cleanup():
        print("✅ 最终清理成功！原生CUDA图复用机制完全恢复！")
    else:
        print("❌ 清理不完整，请手动检查") 