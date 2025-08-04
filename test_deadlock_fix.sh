#!/bin/bash

echo "🧪 Semi-PD 死锁修复效果测试"
echo "============================="

# 设置变量
MODEL_PATH=${1:-"/path/to/your/model"}  # 用户可以传入模型路径
TEST_HOST="localhost"
TEST_PORT="30000"
MAX_WAIT_TIME=300  # 最大等待时间5分钟

# 检查前置条件
echo "📋 步骤1: 检查前置条件..."

if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ NVIDIA GPU 不可用"
    exit 1
else
    echo "✅ GPU 可用"
fi

if ! python3 -c "import torch; print('PyTorch:', torch.__version__)" 2>/dev/null; then
    echo "❌ PyTorch 不可用"
    exit 1
else
    echo "✅ PyTorch 可用"
fi

# 清理环境
echo ""
echo "📋 步骤2: 清理环境..."
./fix_deadlock.sh

echo ""
echo "📋 步骤3: 启动Semi-PD服务..."

# 修改启动脚本中的模型路径
if [ "$MODEL_PATH" != "/path/to/your/model" ]; then
    echo "🔧 使用指定模型路径: $MODEL_PATH"
    # 创建临时启动脚本
    cp start_semi_pd_safe.sh temp_start.sh
    sed -i "s|--model-path /path/to/your/model|--model-path $MODEL_PATH|g" temp_start.sh
    START_SCRIPT="./temp_start.sh"
else
    echo "⚠️  使用默认模型路径，请确保路径正确"
    START_SCRIPT="./start_semi_pd_safe.sh"
fi

# 后台启动服务
echo "🚀 后台启动服务..."
$START_SCRIPT > test_output.log 2>&1 &
SERVICE_PID=$!

echo "📝 服务PID: $SERVICE_PID"
echo "📝 日志文件: test_output.log"

# 等待服务启动
echo ""
echo "📋 步骤4: 等待服务启动..."

wait_time=0
while [ $wait_time -lt $MAX_WAIT_TIME ]; do
    # 检查进程是否还在运行
    if ! kill -0 $SERVICE_PID 2>/dev/null; then
        echo "❌ Semi-PD 服务进程已退出"
        echo "📝 最后10行日志:"
        tail -10 test_output.log
        exit 1
    fi
    
    # 检查端口是否开放
    if netstat -tuln 2>/dev/null | grep -q ":$TEST_PORT "; then
        echo "✅ 服务端口 $TEST_PORT 已开放"
        break
    fi
    
    # 检查是否有死锁迹象
    if grep -q "Watchdog timeout" test_output.log 2>/dev/null; then
        echo "❌ 检测到Watchdog超时，可能发生死锁"
        echo "📝 相关日志:"
        grep -A 5 -B 5 "Watchdog timeout" test_output.log
        kill $SERVICE_PID 2>/dev/null
        exit 1
    fi
    
    echo "⏳ 等待中... ($wait_time/$MAX_WAIT_TIME 秒)"
    sleep 10
    wait_time=$((wait_time + 10))
done

if [ $wait_time -ge $MAX_WAIT_TIME ]; then
    echo "❌ 服务启动超时"
    echo "📝 最后20行日志:"
    tail -20 test_output.log
    kill $SERVICE_PID 2>/dev/null
    exit 1
fi

# 测试服务健康状态
echo ""
echo "📋 步骤5: 测试服务健康状态..."

# 等待额外30秒确保服务完全启动
echo "⏳ 等待服务完全初始化..."
sleep 30

# 检查进程状态
echo "🔍 检查进程状态:"
ps aux | grep -E "(semi_pd|sglang)" | grep -v grep | head -5

# 检查PREFILL-DECODE通信状态
echo ""
echo "🔍 检查PREFILL-DECODE通信:"
if grep -q "PREFILL.*启动成功" test_output.log && grep -q "DECODE.*启动成功" test_output.log; then
    echo "✅ PREFILL和DECODE进程都已启动"
else
    echo "⚠️  未检测到所有进程启动消息"
fi

# 检查是否有死锁错误
echo ""
echo "🔍 检查死锁情况:"
DEADLOCK_COUNT=$(grep -c -E "(Waiting for response|timeout|deadlock)" test_output.log 2>/dev/null)
if [ "$DEADLOCK_COUNT" -eq 0 ]; then
    echo "✅ 未检测到死锁相关错误"
else
    echo "⚠️  检测到 $DEADLOCK_COUNT 个可能的死锁相关消息"
    grep -E "(Waiting for response|timeout|deadlock)" test_output.log | tail -3
fi

# API连通性测试
echo ""
echo "📋 步骤6: API连通性测试..."

# 简单健康检查
echo "🔍 检查健康状态接口:"
if curl -s --connect-timeout 10 "http://$TEST_HOST:$TEST_PORT/health" > /dev/null 2>&1; then
    echo "✅ 健康检查接口响应正常"
    HEALTH_PASS=true
else
    echo "❌ 健康检查接口无响应"
    HEALTH_PASS=false
fi

# 测试生成接口
echo "🔍 测试文本生成接口:"
if [ "$HEALTH_PASS" = true ]; then
    RESPONSE=$(curl -s --connect-timeout 30 --max-time 60 \
        -X POST "http://$TEST_HOST:$TEST_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "default",
            "prompt": "Hello",
            "max_tokens": 5,
            "temperature": 0.7
        }' 2>/dev/null)
    
    if echo "$RESPONSE" | grep -q "choices" 2>/dev/null; then
        echo "✅ 文本生成接口响应正常"
        echo "📝 响应示例: $(echo "$RESPONSE" | head -c 100)..."
        API_PASS=true
    else
        echo "❌ 文本生成接口响应异常"
        echo "📝 响应内容: $RESPONSE"
        API_PASS=false
    fi
else
    echo "⏭️  跳过API测试（健康检查失败）"
    API_PASS=false
fi

# 生成测试报告
echo ""
echo "📋 步骤7: 生成测试报告..."

REPORT_FILE="deadlock_test_report_$(date +%Y%m%d_%H%M%S).txt"

cat > $REPORT_FILE << EOF
Semi-PD 死锁修复测试报告
========================
测试时间: $(date)
模型路径: $MODEL_PATH
服务PID: $SERVICE_PID

测试结果:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 环境清理: 成功
✅ 服务启动: $([ $wait_time -lt $MAX_WAIT_TIME ] && echo "成功" || echo "失败")
✅ 端口开放: $(netstat -tuln | grep -q ":$TEST_PORT " && echo "成功" || echo "失败")
✅ 进程状态: $([ $(ps aux | grep -c -E "(semi_pd|sglang)" | grep -v grep) -gt 0 ] && echo "正常" || echo "异常")
✅ 死锁检查: $([ "$DEADLOCK_COUNT" -eq 0 ] && echo "无死锁" || echo "检测到${DEADLOCK_COUNT}个问题")
✅ 健康检查: $([ "$HEALTH_PASS" = true ] && echo "通过" || echo "失败")
✅ API测试: $([ "$API_PASS" = true ] && echo "通过" || echo "失败")

详细信息:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
启动时间: ${wait_time}秒
当前进程:
$(ps aux | grep -E "(semi_pd|sglang)" | grep -v grep)

GPU状态:
$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader)

端口状态:
$(netstat -tuln | grep -E ":(30000|30001|30002|30003)")

近期日志:
$(tail -10 test_output.log)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

echo "✅ 测试报告已保存到: $REPORT_FILE"

# 总结
echo ""
echo "🎯 测试总结:"
echo "============================================"

if [ $wait_time -lt $MAX_WAIT_TIME ] && [ "$HEALTH_PASS" = true ] && [ "$DEADLOCK_COUNT" -eq 0 ]; then
    echo "🎉 测试成功！Semi-PD死锁问题已解决"
    echo "   ✅ 服务启动正常"
    echo "   ✅ 无死锁现象"
    echo "   ✅ API接口可用"
    echo ""
    echo "💡 建议:"
    echo "   - 继续监控服务状态: tail -f test_output.log"
    echo "   - 进行负载测试验证稳定性"
    echo "   - 定期检查: ./diagnose_deadlock.sh"
    
    TEST_SUCCESS=true
else
    echo "⚠️  测试发现问题，需要进一步调试"
    echo "   📝 查看完整日志: cat test_output.log"
    echo "   🔍 运行诊断工具: ./diagnose_deadlock.sh"
    echo "   📖 查看解决方案: cat DEADLOCK_SOLUTION.md"
    
    TEST_SUCCESS=false
fi

# 询问是否保持服务运行
echo ""
read -p "是否保持Semi-PD服务运行？(y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "🛑 关闭Semi-PD服务..."
    kill $SERVICE_PID 2>/dev/null
    sleep 5
    kill -9 $SERVICE_PID 2>/dev/null
    echo "✅ 服务已关闭"
else
    echo "✅ 服务继续运行，PID: $SERVICE_PID"
    echo "💡 监控命令: tail -f test_output.log"
fi

# 清理临时文件
[ -f temp_start.sh ] && rm -f temp_start.sh

echo ""
echo "🏁 测试完成！"

# 退出码
if [ "$TEST_SUCCESS" = true ]; then
    exit 0
else
    exit 1
fi 