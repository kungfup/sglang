#!/usr/bin/env python3
"""
Semi-PD 性能测试脚本
"""

import time
import requests
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

def test_single_request(prompt="Hello, how are you?", max_tokens=100):
    """测试单个请求的性能"""
    url = "http://localhost:30001/generate"
    data = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7
        }
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, json=data, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            latency = end_time - start_time
            output_text = result.get("text", "")
            tokens_generated = len(output_text.split())
            
            return {
                "success": True,
                "latency": latency,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_generated / latency if latency > 0 else 0,
                "output_length": len(output_text)
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def run_performance_test(num_requests=10, concurrency=1):
    """运行性能测试"""
    print(f"🧪 运行性能测试: {num_requests}个请求, 并发度={concurrency}")
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(test_single_request) for _ in range(num_requests)]
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["success"]:
                print(f"✅ 请求完成: {result['latency']:.2f}s, {result['tokens_per_second']:.1f} tokens/s")
            else:
                print(f"❌ 请求失败: {result['error']}")
    
    total_time = time.time() - start_time
    
    # 统计结果
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        latencies = [r["latency"] for r in successful_results]
        throughputs = [r["tokens_per_second"] for r in successful_results]
        
        print(f"\n📊 性能测试结果:")
        print(f"总请求数: {num_requests}")
        print(f"成功请求数: {len(successful_results)}")
        print(f"成功率: {len(successful_results)/num_requests*100:.1f}%")
        print(f"总耗时: {total_time:.2f}s")
        print(f"平均延迟: {statistics.mean(latencies):.2f}s")
        print(f"延迟中位数: {statistics.median(latencies):.2f}s")
        print(f"平均吞吐量: {statistics.mean(throughputs):.1f} tokens/s")
        print(f"QPS: {len(successful_results)/total_time:.2f} requests/s")
    else:
        print("❌ 所有请求都失败了")

if __name__ == "__main__":
    # 运行基础性能测试
    run_performance_test(num_requests=5, concurrency=1)
    
    print("\n" + "="*50)
    
    # 运行并发性能测试
    run_performance_test(num_requests=10, concurrency=2)
