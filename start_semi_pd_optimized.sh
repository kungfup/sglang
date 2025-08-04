#!/bin/bash

# Semi-PD SGLang v0.4.8 性能优化启动脚本

set -e

echo "🚀 启动Semi-PD SGLang v0.4.8 (性能优化版本)..."

# 应用内核优化
source ./set_kernel_optimization.sh

# 应用内存优化
python semi_pd_memory_config.py



# 启动Semi-PD服务器
echo "🚀 启动Semi-PD服务器..."
cd python

python -m sglang.launch_server --model-path /home/yzh/model/Qwen/Qwen2.5-32B-Instruct --tp-size 2 --disable-radix-cache --disable-overlap-schedule --mem-fraction-static 0.8 --max-prefill-tokens 32768 --port 40069 --enable-semi-pd --chunked-prefill-size 8192  

echo "✅ Semi-PD服务器启动完成"
