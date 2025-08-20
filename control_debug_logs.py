#!/usr/bin/env python3
"""
控制SGLang调试日志的环境变量演示脚本

使用方法：
1. 启用调试日志（默认）：
   python control_debug_logs.py

2. 禁用调试日志：
   SGLANG_DISABLE_DEBUG_LOGS=1 python control_debug_logs.py
   或
   export SGLANG_DISABLE_DEBUG_LOGS=1
   python control_debug_logs.py

3. 在服务器启动时禁用调试日志：
   SGLANG_DISABLE_DEBUG_LOGS=1 python -m sglang.launch_server --model-path ...
"""

import os
import subprocess
import sys

def show_current_setting():
    """显示当前的调试日志设置"""
    debug_disabled = os.environ.get("SGLANG_DISABLE_DEBUG_LOGS", "0").lower() in ("1", "true", "yes")
    
    print("=" * 50)
    print("🔧 SGLang 调试日志控制设置")
    print("=" * 50)
    print(f"环境变量 SGLANG_DISABLE_DEBUG_LOGS: {os.environ.get('SGLANG_DISABLE_DEBUG_LOGS', '未设置')}")
    
    if debug_disabled:
        print("📴 状态: 调试日志已禁用")
        print("🤫 以下日志将不会显示：")
        print("   - [CG-LAUNCH] cudaGraphLaunch host_cost=...")
        print("   - [CG-DETAILED-TIMING] prepare=...ms, core_replay=...ms")
        print("   - [CG-DEVICE] replay:entry device=...")
        print("   - [CG-STREAM] replay_prepare bs=...")
        print("   - [CG-STREAM-FIX] about_to_replay current=...")
        print("   - [DBG_SCHEDULER] rid=...")
        print("   - [DBG_DETOKENIZER] batch=...")
        print("   - [DBG_DETOKENIZER_DECODE] head_text=...")
    else:
        print("📝 状态: 调试日志已启用")
        print("🔍 将显示详细的CUDA Graph和处理过程日志")
    
    print("=" * 50)

def create_launch_scripts():
    """创建启动脚本示例"""
    
    # 带调试日志的启动脚本
    debug_script = """#!/bin/bash
# 启动Semi-PD服务器 - 带详细调试日志
export SGLANG_DISABLE_DEBUG_LOGS=0

echo "🔍 启动Semi-PD服务器 - 调试日志启用"
conda activate Sepd

python -m sglang.launch_server \\
    --model-path /path/to/your/model \\
    --enable-semi-pd \\
    --tp-size 2 \\
    --semi-pd-decode-sm-percentage 50 \\
    --mem-fraction-static 0.78 \\
    --port 30000
"""

    # 不带调试日志的启动脚本
    quiet_script = """#!/bin/bash  
# 启动Semi-PD服务器 - 静默模式（无调试日志）
export SGLANG_DISABLE_DEBUG_LOGS=1

echo "🤫 启动Semi-PD服务器 - 调试日志禁用"
conda activate Sepd

python -m sglang.launch_server \\
    --model-path /path/to/your/model \\
    --enable-semi-pd \\
    --tp-size 2 \\
    --semi-pd-decode-sm-percentage 50 \\
    --mem-fraction-static 0.78 \\
    --port 30000
"""

    with open("launch_debug.sh", "w") as f:
        f.write(debug_script)
    
    with open("launch_quiet.sh", "w") as f:
        f.write(quiet_script)
    
    # 设置执行权限
    os.chmod("launch_debug.sh", 0o755)
    os.chmod("launch_quiet.sh", 0o755)
    
    print("\n📁 已创建启动脚本：")
    print("   - launch_debug.sh  (启用调试日志)")
    print("   - launch_quiet.sh  (禁用调试日志)")

def main():
    show_current_setting()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "enable":
            print("\n✅ 启用调试日志...")
            os.environ["SGLANG_DISABLE_DEBUG_LOGS"] = "0"
            show_current_setting()
            
        elif command == "disable":
            print("\n❌ 禁用调试日志...")
            os.environ["SGLANG_DISABLE_DEBUG_LOGS"] = "1"
            show_current_setting()
            
        elif command == "create-scripts":
            create_launch_scripts()
            
        else:
            print(f"\n❓ 未知命令: {command}")
            print("可用命令: enable, disable, create-scripts")
    
    else:
        print("\n💡 使用提示：")
        print("   python control_debug_logs.py enable        # 启用调试日志")
        print("   python control_debug_logs.py disable       # 禁用调试日志")
        print("   python control_debug_logs.py create-scripts # 创建启动脚本")

if __name__ == "__main__":
    main() 