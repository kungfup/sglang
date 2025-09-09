import requests
import threading
import json

# --- 配置 ---
URL = "http://127.0.0.1:30009/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

# 所有请求的Payload
PAYLOADS = [
    {
        "model": "Qwen2.5-32B-Instruct",
        "messages": [{"role": "user", "content": "请详细介绍人工智能的发展历史，包括重要里程碑事件"}],
        "max_tokens": 1024,
        "temperature": 0.7
    },
    {
        "model": "Qwen2.5-32B-Instruct",
        "messages": [{"role": "user", "content": "请分析深度学习在计算机视觉领域的应用和发展趋势"}],
        "max_tokens": 1024,
        "temperature": 0.7
    }
]

# --- 执行逻辑 ---
def send_request(payload, req_id):
    """发送单个请求的函数"""
    print(f"线程 {req_id}: 发送请求...")
    try:
        response = requests.post(URL, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status() # 如果状态码不是2xx，则抛出异常
        print(f"线程 {req_id}: 收到响应 (状态码: {response.status_code})")
        # print(f"线程 {req_id} 响应内容: {response.json()}") # 可以取消注释来查看完整响应
    except requests.exceptions.RequestException as e:
        print(f"线程 {req_id}: 请求失败 - {e}")

# 创建并启动线程
threads = []
for i, p in enumerate(PAYLOADS):
    # target 是线程要执行的函数，args 是传递给函数的参数
    thread = threading.Thread(target=send_request, args=(p, i + 1))
    threads.append(thread)
    thread.start() # 启动线程

# 等待所有线程执行完毕
for thread in threads:
    thread.join()

print("所有请求已发送完毕。")