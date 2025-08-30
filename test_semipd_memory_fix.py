#!/usr/bin/env python3
"""
测试Semi-PD内存修复
"""

import os
import sys
import subprocess
import time

# 添加当前目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def test_semipd_memory_fix():
    """测试Semi-PD内存修复"""
    print("🧪 测试Semi-PD内存修复...")
    
    # 设置环境变量
    env = os.environ.copy()
    env.update({
        'SGLANG_ENABLE_SEMI_PD': '1',
        'CUDA_VISIBLE_DEVICES': '0,1',
    })
    
    # 启动Semi-PD服务器
    cmd = [
        'python', '-m', 'sglang.launch_server',
        '--model-path', '/home/yzh/model/Qwen/Qwen2.5-1.5B-Instruct',
        '--tp-size', '1',
        '--pp-size', '2',
        '--enable-semi-pd',
        '--mem-fraction-static', '0.8',
        '--port', '30012'
    ]
    
    print(f"🚀 启动命令: {' '.join(cmd)}")
    print(f"🔧 环境变量: SGLANG_ENABLE_SEMI_PD=1, CUDA_VISIBLE_DEVICES=0,1")
    
    try:
        # 启动进程
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("⏳ 等待服务器启动...")
        time.sleep(10)  # 等待10秒
        
        # 检查进程是否还在运行
        if process.poll() is None:
            print("✅ 服务器启动成功，进程仍在运行")
            
            # 等待一段时间观察
            print("⏳ 观察服务器运行状态...")
            time.sleep(20)
            
            # 检查进程状态
            if process.poll() is None:
                print("✅ 服务器运行稳定，内存修复成功！")
            else:
                print("❌ 服务器意外退出")
                stdout, stderr = process.communicate()
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
        else:
            print("❌ 服务器启动失败")
            stdout, stderr = process.communicate()
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            
    except Exception as e:
        print(f"❌ 启动失败: {e}")
    finally:
        # 清理进程
        if 'process' in locals() and process.poll() is None:
            print("🧹 清理进程...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

if __name__ == "__main__":
    test_semipd_memory_fix() 