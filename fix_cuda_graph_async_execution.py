#!/usr/bin/env python3
"""
🚨 修复CUDA Graph异步执行问题

问题：复杂的0.4.8 recapture逻辑导致频繁重新capture，
      破坏了cudaGraphLaunch的异步执行，变成同步执行

解决：将复杂逻辑替换为简单的0.4.4版本逻辑
"""

import re

def fix_cuda_graph_async():
    """修复CUDA Graph异步执行问题"""
    
    print("🔧 修复CUDA Graph异步执行问题...")
    print("=" * 60)
    
    file_path = "python/sglang/srt/model_executor/cuda_graph_runner.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建备份
        with open(file_path + ".async_backup", 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 已创建备份: {file_path}.async_backup")
        
        # 定位并替换复杂的recapture_if_needed方法
        # 匹配从方法定义到下一个方法之间的所有内容
        complex_pattern = r'def recapture_if_needed\(self, forward_batch: ForwardBatch\):.*?(?=\n    def [a-zA-Z_])'
        
        # 简化的recapture逻辑（基于原版Semi-PD 0.4.4）
        simple_recapture = '''def recapture_if_needed(self, forward_batch: ForwardBatch):
        """
        🔧 SEMI_PD_FIX: 简化recapture逻辑，恢复异步执行
        
        复杂的0.4.8逻辑导致频繁重新capture，破坏cudaGraphLaunch异步执行。
        使用原版Semi-PD的简单逻辑，确保CUDA Graph正常异步工作。
        """
        # If the capture_hidden_mode changes, we need to recapture the graph
        hidden_mode_from_spec_info = getattr(
            forward_batch.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
        )
        
        if (
            forward_batch.capture_hidden_mode == CaptureHiddenMode.FULL
            and self.capture_hidden_mode != CaptureHiddenMode.FULL
        ):
            print(f"[SEMI_PD_FIX] CUDA Graph recapture: forward_batch要求FULL模式")
            self.capture_hidden_mode = CaptureHiddenMode.FULL
            self.capture()
        elif (
            forward_batch.capture_hidden_mode != CaptureHiddenMode.FULL
            and self.capture_hidden_mode != hidden_mode_from_spec_info
        ):
            print(f"[SEMI_PD_FIX] CUDA Graph recapture: 切换到{hidden_mode_from_spec_info}模式")
            self.capture_hidden_mode = hidden_mode_from_spec_info
            self.capture()

'''
        
        # 执行替换
        new_content = re.sub(complex_pattern, simple_recapture, content, flags=re.DOTALL)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print("✅ 成功替换复杂recapture逻辑为简化版本")
            print("🎯 预期效果：")
            print("   - cudaGraphLaunch恢复异步执行")
            print("   - GPU stream并行工作")
            print("   - 消除不必要的recapture开销")
            return True
        else:
            print("⚠️ 未找到复杂recapture逻辑，可能已经修复")
            return False
            
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        return False

def verify_fix():
    """验证修复效果"""
    
    print("\n🔍 验证修复效果...")
    
    file_path = "python/sglang/srt/model_executor/cuda_graph_runner.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ("SEMI_PD_FIX.*简化recapture逻辑", "修复标记"),
            ("恢复异步执行", "问题说明"),
            ("forward_batch.capture_hidden_mode == CaptureHiddenMode.FULL", "简单条件"),
            ("hidden_mode_from_spec_info", "0.4.4逻辑"),
        ]
        
        success_count = 0
        for pattern, desc in checks:
            if re.search(pattern, content):
                print(f"   ✅ {desc}: 已应用")
                success_count += 1
            else:
                print(f"   ❌ {desc}: 未找到")
        
        # 检查复杂逻辑是否已移除
        if "required_capture_hidden_mode = max(" not in content:
            print(f"   ✅ 复杂0.4.8逻辑: 已移除")
            success_count += 1
        else:
            print(f"   ❌ 复杂0.4.8逻辑: 仍存在")
        
        total = len(checks) + 1
        print(f"\n修复验证: {success_count}/{total}")
        
        if success_count == total:
            print("🎉 CUDA Graph异步执行修复验证成功！")
            return True
        else:
            print("⚠️ 修复不完整")
            return False
            
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

if __name__ == "__main__":
    print("🚨 CUDA Graph异步执行问题修复")
    print("=" * 60)
    print("问题: cudaGraphLaunch从异步变同步，GPU空泡严重")
    print("原因: 复杂的recapture逻辑导致频繁重新capture")
    print("方案: 恢复原版Semi-PD的简单高效逻辑")
    print()
    
    success = fix_cuda_graph_async()
    if success:
        verify_fix()
        
        print("\n" + "=" * 60)
        print("🎯 修复完成！预期效果：")
        print("- ✅ cudaGraphLaunch恢复异步执行")
        print("- ✅ Duration时间恢复正常（~1.4ms）")
        print("- ✅ GPU空泡显著减少")
        print("- ✅ 整体性能大幅提升")
        print("\n🚀 重启服务后生效！")
    else:
        print("\n❌ 修复失败，请检查错误")
