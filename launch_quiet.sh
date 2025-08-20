#!/bin/bash  
# 启动Semi-PD服务器 - 静默模式（无调试日志）
export SGLANG_DISABLE_DEBUG_LOGS=1

echo "🤫 启动Semi-PD服务器 - 调试日志禁用"
conda activate Sepd

python -m sglang.launch_server \
    --model-path /path/to/your/model \
    --enable-semi-pd \
    --tp-size 2 \
    --semi-pd-decode-sm-percentage 50 \
    --mem-fraction-static 0.78 \
    --port 30000
