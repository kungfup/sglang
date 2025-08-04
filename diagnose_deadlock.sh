#!/bin/bash

echo "🔍 Semi-PD 死锁诊断工具"
echo "========================"

# 检查是否有Semi-PD进程
echo "📊 1. 检查Semi-PD进程状态..."
SEMI_PD_PROCS=$(ps aux | grep -E "(semi_pd|sglang)" | grep -v grep)

if [ -z "$SEMI_PD_PROCS" ]; then
    echo "❌ 没有发现Semi-PD进程运行"
    echo "💡 建议: 重新启动服务"
    exit 1
else
    echo "✅ 发现Semi-PD进程:"
    echo "$SEMI_PD_PROCS"
    
    # 提取PID
    PIDS=$(echo "$SEMI_PD_PROCS" | awk '{print $2}')
    echo ""
    echo "🔍 进程PID列表: $PIDS"
fi

echo ""
echo "📊 2. 检查进程资源使用..."

for PID in $PIDS; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔍 进程 $PID 详细信息:"
    
    # 进程基本信息
    if ps -p $PID > /dev/null 2>&1; then
        echo "  📝 进程状态: $(ps -o pid,ppid,state,pcpu,pmem,cmd -p $PID --no-headers)"
        
        # 检查文件描述符
        FD_COUNT=$(ls /proc/$PID/fd 2>/dev/null | wc -l)
        echo "  📁 文件描述符数量: $FD_COUNT"
        
        # 检查线程
        THREAD_COUNT=$(ls /proc/$PID/task 2>/dev/null | wc -l)
        echo "  🧵 线程数量: $THREAD_COUNT"
        
        # 检查内存映射
        if [ -f /proc/$PID/maps ]; then
            MMAP_COUNT=$(cat /proc/$PID/maps | wc -l)
            echo "  🗺️  内存映射数量: $MMAP_COUNT"
        fi
        
        # 检查是否卡在系统调用
        if [ -f /proc/$PID/stack ]; then
            echo "  📚 内核栈信息:"
            head -5 /proc/$PID/stack 2>/dev/null | sed 's/^/    /'
        fi
        
        # 检查打开的文件
        echo "  📂 主要打开文件:"
        lsof -p $PID 2>/dev/null | head -10 | sed 's/^/    /'
        
    else
        echo "  ❌ 进程 $PID 已不存在"
    fi
    echo ""
done

echo ""
echo "📊 3. 检查系统资源..."

# 检查GPU状态
echo "🔍 GPU状态:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo ""
echo "🔍 GPU进程:"
nvidia-smi pmon -c 1 2>/dev/null || nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

# 检查共享内存
echo ""
echo "🔍 共享内存使用:"
df -h /dev/shm
echo ""
echo "🔍 IPC共享内存段:"
ipcs -m | head -10

# 检查信号量
echo ""
echo "🔍 IPC信号量:"
SEM_COUNT=$(ipcs -s | wc -l)
echo "信号量总数: $SEM_COUNT"
if [ $SEM_COUNT -gt 100 ]; then
    echo "⚠️  信号量数量过多，可能存在泄漏"
fi

# 检查网络连接
echo ""
echo "🔍 网络连接状态:"
netstat -tuln | grep -E ":(30000|30001|30002|30003)" || echo "相关端口未监听"

echo ""
echo "📊 4. 检查CUDA上下文..."

# 检查CUDA错误
python3 -c "
import torch
try:
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f'✅ CUDA设备 {device} 可用')
        print(f'📊 GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f}GB')
        
        # 简单的CUDA操作测试
        x = torch.randn(100, 100, device='cuda')
        y = torch.mm(x, x.t())
        print('✅ CUDA操作测试通过')
    else:
        print('❌ CUDA不可用')
except Exception as e:
    print(f'❌ CUDA错误: {e}')
" 2>&1

echo ""
echo "📊 5. 生成诊断报告..."

REPORT_FILE="deadlock_diagnosis_$(date +%Y%m%d_%H%M%S).txt"

cat > $REPORT_FILE << EOF
Semi-PD 死锁诊断报告
生成时间: $(date)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 进程状态:
$SEMI_PD_PROCS

2. 系统资源:
$(free -h)

3. GPU状态:
$(nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv)

4. 共享内存:
$(df -h /dev/shm)

5. IPC资源:
$(ipcs -a | head -20)

6. 网络状态:
$(netstat -tuln | grep -E ":(30000|30001|30002|30003)")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

echo "✅ 诊断报告已保存到: $REPORT_FILE"

echo ""
echo "💡 解决建议:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if echo "$SEMI_PD_PROCS" | grep -q "D.*semi_pd"; then
    echo "🚨 发现进程处于不可中断睡眠状态 (D state)"
    echo "   建议: 强制重启Semi-PD服务"
    echo "   命令: ./fix_deadlock.sh && ./start_semi_pd_safe.sh"
elif [ $SEM_COUNT -gt 100 ]; then
    echo "🚨 发现IPC资源泄漏"
    echo "   建议: 清理IPC资源"
    echo "   命令: ./fix_deadlock.sh"
elif ! nvidia-smi > /dev/null 2>&1; then
    echo "🚨 GPU不可用"
    echo "   建议: 检查NVIDIA驱动和CUDA安装"
else
    echo "📋 常规检查:"
    echo "   1. 运行清理: ./fix_deadlock.sh"
    echo "   2. 安全启动: ./start_semi_pd_safe.sh"
    echo "   3. 监控日志: tail -f *.log"
    echo "   4. 检查配置: 确保模型路径正确"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" 