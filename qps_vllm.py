import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import os
import math
import random
import sys
import time
from concurrent import futures
from transformers import AutoTokenizer
from queue import Queue
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import aiohttp
import numpy as np
import requests
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from tqdm.asyncio import tqdm
from qwen_vl_utils import fetch_image
import io
import json
import socket
from argparse import ArgumentParser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

AIOHTTP_TIMEOUT = 10000

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0

async def send_request(session, url, data, headers):
    async with session.post(url=url, json=data, headers=headers) as response:
        return await response.json()


async def async_request_openai_chat_completions(
    payloads,
    api_url,
    api_key,
):
    # api_url = random.choice(api_url_list)
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    timeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        st = time.perf_counter()
        for payload in payloads:
            tasks.append(send_request(session, api_url, payload, headers))
        responses = await asyncio.gather(*tasks)
        timestamp = time.perf_counter()
        prompt_tokens = 0
        generate_tokens = 0
        for o in responses:
            try:
                prompt_tokens += o["usage"]["prompt_tokens"]
                generate_tokens += o["usage"]["completion_tokens"]
            except Exception as e:
                logging.info("bad stat")
                continue
        prompt_tokens = prompt_tokens / len(payloads)
        generate_tokens = generate_tokens / len(payloads)
        latency = (timestamp - st) * 1000
        # logging.info(f"prompt_tokens: {responses}")
        return latency, prompt_tokens, generate_tokens


async def get_next_payloads(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    默认使用均匀qps请求，保证对比的一致性
    """
    input_requests = iter(input_requests)
    low_bound = request_rate - request_rate * 0.2
    high_bound = request_rate + request_rate * 0.2
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        # interval = np.random.exponential(1.0 / request_rate)
        interval = 1.0 / request_rate
        # interval = np.random.uniform(1.0 / high_bound, 1.0 / low_bound, size=[1])[0]
        # interval = np.random.uniform(1.0 / request_rate-0.2, 1.0 / request_rate+0.2, size=[1])[0]
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def resize_image_to_base64(imgpath):
    base64_image = encode_image(imgpath)
    img = fetch_image({"type": "image",
                        "image": f"data:image/jpeg;base64,{base64_image}",
                        "min_pixels": 28 * 28 * 4,  # 最小 4 个 image patch
                        "max_pixels": 512 * 512})
    img_size = img.size
    num_tokens = math.ceil(img_size[0] / 28) * math.ceil(img_size[1] / 28) + 2
   
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    image_bytes = img_bytes.getvalue()
    img_base64 = base64.b64encode(image_bytes)
    # 将Base64字节数据转换为字符串
    img_base64_str = img_base64.decode('utf-8')
    img_base64_str = "data:image/jpeg;base64," + img_base64_str
    return img_base64_str, num_tokens


def build_payloads(model_path, input_len=4096, num_samples=300, max_images=20, max_output_tokens=128):
    # TODO 需要check decoding 长度是不是满足要求

    image_path = '/home/yzh/apple.png'
    input_requests = []
    base64_image, image_tokens = resize_image_to_base64(image_path)
    num_images = min(input_len // image_tokens, max_images)
    logging.info(f"input contains {num_images} images")
    system_prompt = '你是一个全能的AI助手, 回答问题的时候请忽略用户输入的重复"A "部分'
    prompt = 'please describe thoses image one by one in details and answer as longer as possible'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    used_tokens = len(tokenizer.encode(system_prompt)) + len(tokenizer.encode(prompt)) + num_images * image_tokens + 20
    space_tokens = input_len - used_tokens
    question = "A "* (space_tokens + 7+7500+8192-9172) + prompt

    content_user = [
                    {
                        "type": "text",
                        "text": question
                    }
                ]
    for i in range(num_images):
        content_user.append({
            "type": "image_url",
            "image_url": {
                "url": base64_image
            },
        })
    logging.info(f"NUM Images:{len(content_user)} ii:{max_images}")
    payloads = { 
        "model": model_path,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content_user
            }
        ],
        "temperature": 0.0,
        "ignore_eos":True,
        "max_tokens": max_output_tokens,
        "top_k": 1,
        "top_p":0.001,
        "stream": False,
        "logits": False,  # 如不需要请删除，会造成性能损失
        "logits_care_tokens":[1,2],
        "return_hidden_states": False, # 如不需要返回请删除，会造成性能损失
        # "top_logprobs": 4 # 如不需要请删除，会造成性能损失
    }
    input_requests = [payloads] * num_samples
    return input_requests

async def async_request_profile(api_url: str) -> RequestFuncOutput:
    timeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        output = RequestFuncOutput()
        try:
            async with session.post(url=api_url) as response:
                if response.status == 200:
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output

async def benchmark(
    payloads, qps, port, api_key, profile=False
):
    base_url = f"http://127.0.0.1:{port}"
    generate_url = base_url + "/v1/chat/completions"
    
    # 注意：当前vLLM版本不支持 /start_profile 和 /stop_profile 路由
    # 如果需要profiling，请在启动服务时设置 VLLM_TORCH_PROFILER_DIR 环境变量
    if profile:
        print("注意：当前vLLM版本不支持HTTP profiling路由")
        print("如需profiling，请在启动服务时设置 VLLM_TORCH_PROFILER_DIR 环境变量")

    tasks: List[asyncio.Task] = []
    async for payload in get_next_payloads(payloads, qps):
        tasks.append(
            asyncio.create_task(
                async_request_openai_chat_completions(
                        payloads=[payload],
                        api_url=generate_url,
                        api_key=api_key,
                )
            )
        )
    outputs: List[float] = await asyncio.gather(*tasks)

    return outputs

def warm_up(args, payloads):
    time_start = time.perf_counter()
    warmup_payloads = payloads[:10]
    benchmark_result = asyncio.run(
        benchmark(
            payloads=warmup_payloads,
            qps=1, 
            port=args.port, 
            api_key=args.api_key, 
            profile=False
        )
    )
    time_end = time.perf_counter()
    true_qps = len(warmup_payloads) / (time_end - time_start)
    logging.info(f"Total time: {(time_end - time_start) * 1000}")
    logging.info(f"True QPS: {true_qps}")
    benchmark_prompt_tokens = [i[1] for i in benchmark_result]
    benchmark_generate_tokens = [i[2] for i in benchmark_result]
    logging.info(f"Total prompt tokens: {np.sum(benchmark_prompt_tokens)}")
    logging.info(f"Total generate tokens: {np.sum(benchmark_generate_tokens)}")
    logging.info(f"Avg prompt tokens: {np.mean(benchmark_prompt_tokens)}")
    logging.info(f"Avg generate tokens: {np.mean(benchmark_generate_tokens)}")
    pass

def run_bench(args, payloads, qps):
    time_start = time.perf_counter()
    benchmark_result = asyncio.run(
        benchmark(
            payloads=payloads, 
            qps=qps, 
            port=args.port, 
            api_key=args.api_key, 
            profile=args.profile
        )
    )
    time_end = time.perf_counter()
    result_dict = {}
    true_qps = len(payloads) / (time_end - time_start)
    logging.info(f"Total time: {(time_end - time_start) * 1000}")
    logging.info(f"True QPS: {true_qps}")
    benchmark_prompt_tokens = [i[1] for i in benchmark_result]
    benchmark_generate_tokens = [i[2] for i in benchmark_result]
    logging.info(f"Total prompt tokens: {np.sum(benchmark_prompt_tokens)}")
    logging.info(f"Total generate tokens: {np.sum(benchmark_generate_tokens)}")
    logging.info(f"Avg prompt tokens: {np.mean(benchmark_prompt_tokens)}")
    logging.info(f"Avg generate tokens: {np.mean(benchmark_generate_tokens)}")
    benchmark_latency = [i[0] for i in benchmark_result]
    latencies = np.array(benchmark_latency)
    percentages = [10, 25, 50, 75, 90, 95, 99]
    percentiles = np.percentile(latencies, percentages)
    logging.info(f"Avg latency: {np.mean(latencies)} ms")
    avg_latency = np.mean(latencies)
    result_dict['target_qps'] = qps
    result_dict['true_qps'] = true_qps
    result_dict['avg_latency'] = avg_latency
    result_dict['avg_prompt_tokens'] = np.mean(benchmark_prompt_tokens)
    result_dict['avg_generate_tokens'] = np.mean(benchmark_generate_tokens)
    for percentage, percentile in zip(percentages, percentiles):
        logging.info(f"{percentage}% percentile latency: {percentile} ms")
        result_dict[f'{percentage}%_latency'] = percentile
    diff = abs(true_qps - qps) / qps
    result_dict['qps_diff'] = diff
    return result_dict

def main(args):
    if args.profile:
        if args.num_samples > 10:
            logging.warning("开启profile模式时，num_samples > 10, 可能导致生成文件过大，导致无法打开")

    payloads = build_payloads(model_path=args.model_name,
                              input_len=args.prefill_len, 
                              num_samples=args.num_samples, 
                              max_images=args.max_img_num, 
                              max_output_tokens=args.decoding_len)
    logging.info("Warm up...")
    warm_up(args, payloads)

    # 处理qps测试逻辑
    keep_test = True
    hit_top = False
    qps = args.start_qps
    prev_qps = qps
    with open(args.output_path,'w') as w:
        while keep_test:
            logging.info("="*20)
            logging.info(f"QPS:{qps}")
            result_dict = run_bench(args, payloads, qps)
            w.write(json.dumps(result_dict)+"\n")
            if result_dict['qps_diff'] >= 0.05 and qps > args.start_qps:
                if prev_qps == qps:
                    keep_test = False
                elif qps > prev_qps:
                    qps -= 0.01
                hit_top=True
            else:
                prev_qps = qps
                if hit_top:
                    keep_test = False
                else:
                    qps += args.qps_gap

if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online serving throughput.")
    # 模型tokenizer信息
    parser.add_argument(
        "--model_name",
        type=str,
        default="/home/yzh/model/Qwen/Qwen2.5-VL-32B-Instruct/",
        help="模型路径",
    )
    # 服务信息
    parser.add_argument(
        "--port",
        type=str,
        default="25000",
        help="服务访问端口",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="服务访问端口",
    )
    # 压测信息
    # 压测数据配置
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="压测数据量",
    )
    parser.add_argument(
        "--prefill_len",
        type=int,
        default=8192,
        help="输入数据长度",
    )
    parser.add_argument(
        "--decoding_len",
        type=int,
        default=128,
        help="模型生成长度",
    )
    parser.add_argument(
        "--max_img_num",
        type=int,
        default=20,
        help="图片数量，一般情况下，图片数量越多，吞吐越低",
    )
    # 压测信息
    parser.add_argument(
        "--start_qps",
        type=float,
        default=0.7,
        help="测试起始qps",
    )
    parser.add_argument(
        "--qps_gap",
        type=float,
        default=0.03,
        help="用于控制qps的增长速度",
    )

    # 控制是否进行peofile
    parser.add_argument(
        "--profile",
        action="store_true",
        help="是否进行profile",
    )
    # 控制输出结果保存位置
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output.json",
        help="用于控制qps的增长速度",
    )
    args = parser.parse_args()
    main(args)

    pass

"""
参考命令：Python auto_qps.py --port 25000 --num_samples 200 --prefill_len 8192 --decoding_len 128 --max_img_num 20 --start_qps 0.1 --qps_gap 0.03 --output_path./output.json
"""


