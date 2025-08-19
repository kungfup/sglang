#!/usr/bin/env python3
"""
测试CUDA Graph修复效果
监控cudaGraphLaunch时间
"""

import subprocess
import re
import time
import statistics

def monitor_cuda_graph_time(duration=30):
    """监控CUDA Graph启动时间"""
    
    print(f"🔍 监控CUDA Graph性能 ({duration}秒)...")
    
    # 启动日志监控
    log_file = "/tmp/semi_pd_cuda_graph.log"
    
    # 使用journalctl或tail监控日志
    cmd = f"timeout {duration} tail -f /path/to/semi_pd.log | grep -E 'cudaGraphLaunch|CG-LAUNCH' > {log_file}"
    subprocess.run(cmd, shell=True)
    
    # 解析日志提取时间
    times = []
    with open(log_file, 'r') as f:
        for line in f:
            # 提取host_cost时间
            match = re.search(r'host_cost=([0-9.]+)\s*ms', line)
            if match:
                times.append(float(match.group(1)))
    
    if times:
        print(f"\n📊 CUDA Graph性能统计：")
        print(f"  样本数: {len(times)}")
        print(f"  平均时间: {statistics.mean(times):.2f} ms")
        print(f"  最小时间: {min(times):.2f} ms")
        print(f"  最大时间: {max(times):.2f} ms")
        print(f"  中位数: {statistics.median(times):.2f} ms")
        
        # 判断是否修复成功
        avg_time = statistics.mean(times)
        if avg_time < 5:
            print("\n✅ 性能优秀！CUDA Graph启动时间正常")
        elif avg_time < 20:
            print("\n⚠️  性能一般，可能需要进一步优化")
        else:
            print("\n❌ 性能问题仍然存在，需要深入调试")
    else:
        print("\n⚠️  未能收集到性能数据")

def test_32b_inference():
    """测试32B模型推理"""
    
    import requests
    import json
    
    print("\n🧪 测试32B模型推理性能...")
    
    url = "http://127.0.0.1:40066/v1/chat/completions"
    
    test_prompts = [
        "你好",
        "请解释什么是人工智能",
        "写一个Python快速排序算法"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n测试 {i}: {prompt[:20]}...")
        
        payload = {
            "model": "model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        start = time.time()
        try:
            response = requests.post(url, json=payload, timeout=60)
            elapsed = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                print(f"  ✅ 响应时间: {elapsed:.2f}s")
                print(f"  输出预览: {content[:50]}...")
            else:
                print(f"  ❌ 请求失败: {response.status_code}")
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")

if __name__ == "__main__":
    # 先测试推理
    test_32b_inference()
    
    # 然后监控性能
    print("\n" + "=" * 60)
    monitor_cuda_graph_time(30)
