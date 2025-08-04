#!/bin/bash

echo "🔧 Semi-PD 死锁问题修复工具"
echo "=================================="

# 检查当前系统状态
echo "📊 系统状态检查..."

# 1. 检查GPU内存
echo "🔍 检查GPU内存使用情况："
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader,nounits

# 2. 检查Semi-PD相关进程
echo ""
echo "🔍 检查Semi-PD进程："
ps aux | grep -E "(semi_pd|sglang)" | grep -v grep

# 3. 检查共享内存
echo ""
echo "🔍 检查共享内存使用："
df -h /dev/shm
ipcs -m | head -10

echo ""
echo "🛠️ 应用修复措施..."

# 清理残留进程
echo "1. 清理残留Semi-PD进程..."
pkill -f "semi_pd" 2>/dev/null || true
pkill -f "sglang.*semi" 2>/dev/null || true

# 清理共享内存
echo "2. 清理共享内存资源..."
ipcs -m | awk 'NR>3 {print $2}' | xargs -r ipcrm -m 2>/dev/null || true

# 清理信号量
echo "3. 清理IPC信号量..."
ipcs -s | awk 'NR>3 {print $2}' | xargs -r ipcrm -s 2>/dev/null || true

echo ""
echo "✅ 修复完成！现在可以重新启动Semi-PD服务。" 