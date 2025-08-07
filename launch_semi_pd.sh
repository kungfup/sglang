#!/bin/bash

# 设置正确的 Python 路径
export PYTHONPATH="/home/yzh/semi_pd_migration/sglang_0.4.8/python:$PYTHONPATH"

# 设置 PyTorch 库路径（如果需要）
export LD_LIBRARY_PATH="/home/yzh/ENTER/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"

# 进入正确的目录
cd /home/yzh/semi_pd_migration/sglang_0.4.8

# 启动 Semi-PD 服务器
python -m sglang.launch_server \
    --model-path /path/to/your/model \
    --enable-semi-pd \
    --host 0.0.0.0 \
    --port 30000 \
    "$@" 