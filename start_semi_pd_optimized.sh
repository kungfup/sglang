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

python -m sglang.launch_server \
    --model-path /home/yzh/model/Qwen/Qwen2.5-1.5B-Instruct \
    --tp-size 2 \
    --enable-semi-pd \
    --max-total-tokens 512 \
    --log-level debug \
    --mem-fraction-static 0.80 \
    --disable-radix-cache \
    --port 30001 \
    --chunked-prefill-size 8192 \
    --max-prefill-tokens 32768 \
    --disable-overlap-schedule

echo "✅ Semi-PD服务器启动完成"
