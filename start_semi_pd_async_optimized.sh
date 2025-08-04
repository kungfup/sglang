#!/bin/bash

# Semi-PD 异步优化启动脚本
# 移除 MSCCL++ 强制流同步，提升性能

set -e

echo "🚀 启动 Semi-PD 异步优化版本"
echo "===================================="

# 真正有效的异步优化配置：减少 .item() 调用的同步开销
export SEMI_PD_ASYNC_OPT_ENABLED=1
export SEMI_PD_CACHE_SIZE=2000              # 张量缓存大小
export SEMI_PD_CACHE_TTL=0.1                # 缓存存活时间（秒）

# 防死锁配置
export SEMI_PD_IPC_TIMEOUT=30               # IPC通信超时时间（秒）
export SEMI_PD_WATCHDOG_TIMEOUT=60          # Watchdog超时时间（秒）
export SEMI_PD_MAX_RETRY_COUNT=3            # 最大重试次数

# CUDA 优化设置
export CUDA_LAUNCH_BLOCKING=0  # 启用异步 CUDA 启动
export CUDA_DEVICE_MAX_CONNECTIONS=32  # 增加并发连接数

# 内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6

# MPS 优化配置
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

echo "✅ 张量同步优化环境变量已设置"
echo "  - SEMI_PD_ASYNC_OPT_ENABLED: $SEMI_PD_ASYNC_OPT_ENABLED"
echo "  - 张量缓存大小: ${SEMI_PD_CACHE_SIZE}"
echo "  - 缓存TTL: ${SEMI_PD_CACHE_TTL}s"

# 检查 CUDA 可用性
if ! nvidia-smi &>/dev/null; then
    echo "❌ 错误: 无法检测到 NVIDIA GPU"
    exit 1
fi

# 检查 MPS 服务
if ! pgrep nvidia-cuda-mps-control &>/dev/null; then
    echo "⚠️  警告: NVIDIA MPS 未运行，Semi-PD 可能无法正常工作"
    echo "请先启动 MPS 服务:"
    echo "  sudo nvidia-cuda-mps-control -d"
fi

# 默认参数
MODEL_PATH=${1:-"/data/models/Meta-Llama-3.1-8B-Instruct"}
HOST=${2:-"0.0.0.0"}
PORT=${3:-8000}
TENSOR_PARALLEL_SIZE=${4:-2}

echo ""
echo "🔧 启动参数:"
echo "  - 模型路径: $MODEL_PATH"
echo "  - 监听地址: $HOST:$PORT"  
echo "  - 张量并行大小: $TENSOR_PARALLEL_SIZE"
echo ""

# 构建启动命令
START_CMD="python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host $HOST \
    --port $PORT \
    --tp-size $TENSOR_PARALLEL_SIZE \
    --semi-pd \
    --disable-cuda-graph-for-prefill \
    --enable-cuda-graph \
    --chunked-prefill-size 4096 \
    --max-running-requests 256 \
    --max-total-tokens 32768"

echo "🚀 执行启动命令:"
echo "$START_CMD"
echo ""

# 启动性能监控
if command -v nvidia-ml-py3 &>/dev/null; then
    echo "📊 启动性能监控..."
    # 后台监控 GPU 使用率
    (
        while true; do
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
            awk '{printf "[%s] GPU使用率: %s%%, 内存: %s/%sMB\n", strftime("%H:%M:%S"), $1, $2, $3}'
            sleep 5
        done
    ) &
    MONITOR_PID=$!
    echo "性能监控进程 PID: $MONITOR_PID"
fi

# 捕获退出信号，清理资源
cleanup() {
    echo ""
    echo "🧹 清理资源..."
    if [[ -n $MONITOR_PID ]]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
    echo "✅ 清理完成"
    exit 0
}

trap cleanup SIGINT SIGTERM

# 启动 Semi-PD 服务器
echo "🎯 启动 Semi-PD 异步优化服务器..."
eval $START_CMD

# 如果到达这里，说明服务器已退出
cleanup 