#!/bin/bash
# VIT SHM 模式集成测试脚本
#
# 功能:
# 1. 启动 SGLang server (SHM 模式)
# 2. 发送测试请求
# 3. 监控显存使用
# 4. 验证性能指标
# 5. 清理资源

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置
MODEL_PATH=${MODEL_PATH:-"/path/to/model"}
IMAGE_PATH=${IMAGE_PATH:-"/path/to/test/image.jpg"}
HOST="127.0.0.1"
PORT=30017
SERVER_PID=""

# 清理函数
cleanup() {
    echo_info "Cleaning up..."
    
    # 停止 server
    if [ -n "$SERVER_PID" ]; then
        echo_info "Stopping server (PID: $SERVER_PID)..."
        kill -TERM $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    
    # 清理 SHM
    echo_info "Cleaning up shared memory..."
    python sglang/scripts/cleanup_vit_shm.py --force || true
    
    echo_info "Cleanup complete"
}

# 注册清理函数
trap cleanup EXIT INT TERM

# 检查依赖
check_dependencies() {
    echo_info "Checking dependencies..."
    
    # 检查 Python
    if ! command -v python &> /dev/null; then
        echo_error "Python not found"
        exit 1
    fi
    
    # 检查 posix_ipc
    if ! python -c "import posix_ipc" 2>/dev/null; then
        echo_error "posix_ipc not installed. Install with: pip install posix-ipc"
        exit 1
    fi
    
    # 检查 CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        echo_error "nvidia-smi not found. CUDA required for testing"
        exit 1
    fi
    
    echo_info "All dependencies OK"
}

# 启动 server
start_server() {
    echo_info "Starting SGLang server in SHM mode..."
    
    # 设置环境变量
    export SGLANG_VIT_USE_SHM=true
    export SGLANG_VIT_SHM_SIZE_GB=20.0
    export SGLANG_VIT_SAFETY_MARGIN_GB=0.5
    export SGLANG_VIT_OVERHEAD_RATIO=4.0
    
    # 启动 server
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        --device cuda \
        > /tmp/sglang_server.log 2>&1 &
    
    SERVER_PID=$!
    echo_info "Server started (PID: $SERVER_PID)"
    
    # 等待 server 启动
    echo_info "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
            echo_info "Server is ready"
            return 0
        fi
        sleep 1
    done
    
    echo_error "Server failed to start within 30 seconds"
    echo_error "Server log:"
    tail -n 50 /tmp/sglang_server.log
    exit 1
}

# 监控显存
monitor_memory() {
    echo_info "GPU Memory Usage:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s (%s): %s / %s MB (%.1f%%)\n", $1, $2, $3, $4, ($3/$4)*100}'
}

# 发送测试请求
send_test_requests() {
    echo_info "Sending test requests..."
    
    # 设置测试参数
    export MODEL_PATH="$MODEL_PATH"
    export IMAGE_PATH="$IMAGE_PATH"
    
    # 发送请求
    python test_image_request.py
    
    if [ $? -eq 0 ]; then
        echo_info "Test requests completed successfully"
    else
        echo_error "Test requests failed"
        exit 1
    fi
}

# 验证性能
verify_performance() {
    echo_info "Verifying performance metrics..."
    
    # 检查 server 日志
    if grep -q "OOM" /tmp/sglang_server.log; then
        echo_error "OOM detected in server log"
        return 1
    fi
    
    if grep -q "✅ Embedding moved to CPU" /tmp/sglang_server.log; then
        echo_info "✅ CPU transfer working"
    else
        echo_warn "⚠️  CPU transfer not detected in logs"
    fi
    
    if grep -q "✅ Read embedding from SHM" /tmp/sglang_server.log; then
        echo_info "✅ SHM reading working"
    else
        echo_warn "⚠️  SHM reading not detected in logs"
    fi
    
    echo_info "Performance verification complete"
}

# 主流程
main() {
    echo_info "=== VIT SHM Integration Test ==="
    
    # 1. 检查依赖
    check_dependencies
    
    # 2. 清理旧的 SHM
    echo_info "Cleaning up old shared memory..."
    python sglang/scripts/cleanup_vit_shm.py --force || true
    
    # 3. 显示初始显存
    echo_info "Initial GPU memory:"
    monitor_memory
    
    # 4. 启动 server
    start_server
    
    # 5. 显示启动后显存
    echo_info "GPU memory after server start:"
    monitor_memory
    
    # 6. 发送测试请求
    send_test_requests
    
    # 7. 显示测试后显存
    echo_info "GPU memory after test requests:"
    monitor_memory
    
    # 8. 验证性能
    verify_performance
    
    # 9. 显示统计
    echo_info "=== Test Summary ==="
    echo_info "Server log: /tmp/sglang_server.log"
    echo_info "Check logs for detailed metrics"
    
    echo_info "=== Test Complete ==="
}

# 运行主流程
main

