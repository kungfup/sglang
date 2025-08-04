#!/bin/bash

echo "🛡️ Semi-PD 安全启动脚本 (含死锁预防)"
echo "============================================"

# 首先运行死锁修复
echo "📋 步骤1: 清理环境..."
./fix_deadlock.sh

# 环境配置
echo ""
echo "📋 步骤2: 配置环境变量..."

# 张量同步优化
export SEMI_PD_ASYNC_OPT_ENABLED=1
export SEMI_PD_CACHE_SIZE=2000
export SEMI_PD_CACHE_TTL=0.1

# 死锁预防配置
export SEMI_PD_DEADLOCK_PREVENTION=1
export SEMI_PD_IPC_TIMEOUT=30
export SEMI_PD_WATCHDOG_TIMEOUT=60
export SEMI_PD_MAX_RETRY_COUNT=3

# 保守内存配置
export SEMI_PD_CONSERVATIVE_MEMORY=1
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 减少内存压力
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8

echo "✅ 环境变量配置完成"
echo "  - 死锁预防: 启用"
echo "  - IPC超时: ${SEMI_PD_IPC_TIMEOUT}s"
echo "  - Watchdog超时: ${SEMI_PD_WATCHDOG_TIMEOUT}s"
echo "  - 张量优化: 启用"

# 启动前检查
echo ""
echo "📋 步骤3: 启动前检查..."

# 检查GPU状态
echo "🔍 检查GPU状态:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits | head -2

# 检查共享内存
echo "🔍 检查共享内存:"
df -h /dev/shm | grep shm

# 检查端口占用
echo "🔍 检查端口占用:"
netstat -tuln | grep -E ":(30000|30001|30002|30003)" || echo "端口空闲"

# 启动服务
echo ""
echo "📋 步骤4: 启动Semi-PD服务..."

# 设置PYTHONPATH
export PYTHONPATH="/home/yzh/semi_pd_migration/sglang_0.4.8/python:$PYTHONPATH"

# 启动命令 - 使用保守配置
echo "🚀 使用保守配置启动..."

python -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 2 \
    --semi-pd \
    --mem-fraction-static 0.8 \
    --max-total-tokens 4096 \
    --context-length 4096 \
    --chunked-prefill-size 1024 \
    --max-num-reqs 32 \
    --schedule-conservativeness 0.8 \
    --disable-flashinfer-sampling \
    --enable-mixed-chunk \
    --disable-cuda-graph \
    --watchdog-timeout 60 \
    --log-level info \
    2>&1 | tee semi_pd_safe_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "📝 启动日志已保存到 semi_pd_safe_*.log"
echo "💡 如果仍有问题，请检查日志或运行: ./diagnose_deadlock.sh" 