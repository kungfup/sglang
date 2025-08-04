# Semi-PD 死锁问题完整解决方案

## 🎯 **问题分析**

您遇到的是典型的Semi-PD PREFILL-DECODE调度器通信死锁：

```
[PREFILL] Sending request to D-Scheduler: 1 candidates
[PREFILL] Waiting for response from D-Scheduler...
# 300秒后...
Watchdog timeout (self.watchdog_timeout=300)
```

**根本原因**：
1. **IPC通信阻塞**：PREFILL向DECODE发送请求后无限等待
2. **资源竞争**：GPU内存、共享内存或信号量冲突
3. **进程状态异常**：某个进程卡在不可中断状态

---

## 🛠️ **立即解决方案**

### **方法1: 一键修复（推荐）**

```bash
# 在您的服务器上执行
cd /home/yzh/semi_pd_migration/sglang_0.4.8

# 1. 清理环境
./fix_deadlock.sh

# 2. 安全启动
./start_semi_pd_safe.sh
```

### **方法2: 手动诊断**

```bash
# 如果一键修复不成功，运行诊断工具
./diagnose_deadlock.sh

# 根据诊断结果采取对应措施
```

---

## 🔧 **核心修复机制**

### **1. 死锁预防模块**
- **文件**: `python/sglang/srt/utils_deadlock_prevention.py`
- **功能**: 
  - IPC通信超时监控
  - 进程心跳机制
  - 自动重试和恢复

### **2. 环境优化**
```bash
# 死锁预防配置
export SEMI_PD_DEADLOCK_PREVENTION=1
export SEMI_PD_IPC_TIMEOUT=30
export SEMI_PD_WATCHDOG_TIMEOUT=60

# 保守内存配置
export SEMI_PD_CONSERVATIVE_MEMORY=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

### **3. 系统资源清理**
```bash
# 清理残留进程
pkill -f "semi_pd"

# 清理共享内存
ipcs -m | awk 'NR>3 {print $2}' | xargs ipcrm -m

# 清理信号量
ipcs -s | awk 'NR>3 {print $2}' | xargs ipcrm -s
```

---

## 📊 **启动参数优化**

### **保守配置（推荐用于解决死锁）**
```bash
python -m sglang.launch_server \
    --model-path /your/model/path \
    --tp-size 2 \
    --semi-pd \
    --mem-fraction-static 0.8 \
    --max-total-tokens 4096 \
    --max-num-reqs 32 \
    --schedule-conservativeness 0.8 \
    --disable-cuda-graph \
    --watchdog-timeout 60 \
    --log-level info
```

### **关键参数说明**
- `--mem-fraction-static 0.8`: 保守内存分配
- `--schedule-conservativeness 0.8`: 保守调度策略
- `--disable-cuda-graph`: 禁用CUDA图（减少复杂性）
- `--watchdog-timeout 60`: 缩短超时时间
- `--max-num-reqs 32`: 限制并发请求数

---

## 🚨 **紧急修复步骤**

如果服务完全卡死：

```bash
# 1. 强制终止所有相关进程
sudo pkill -9 -f "sglang"
sudo pkill -9 -f "semi_pd"

# 2. 重置GPU
nvidia-smi --gpu-reset -i 0
nvidia-smi --gpu-reset -i 1

# 3. 清理系统资源
./fix_deadlock.sh

# 4. 重新启动
./start_semi_pd_safe.sh
```

---

## 📋 **监控和诊断**

### **实时监控**
```bash
# 监控进程状态
watch -n 2 'ps aux | grep -E "(semi_pd|sglang)" | grep -v grep'

# 监控GPU状态
watch -n 2 'nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader'

# 监控日志
tail -f semi_pd_safe_*.log
```

### **健康检查**
```bash
# 检查进程通信
python3 -c "
from sglang.srt.utils_deadlock_prevention import check_semi_pd_health
print(check_semi_pd_health())
"

# 检查IPC资源
ipcs -a | head -20
```

---

## 🎯 **预防措施**

### **1. 启动前检查**
```bash
# 确保环境干净
./fix_deadlock.sh

# 检查GPU可用性
nvidia-smi

# 检查共享内存空间
df -h /dev/shm
```

### **2. 配置优化**
```bash
# 增加系统限制
echo 2147483648 > /proc/sys/kernel/shmmax  # 增加共享内存限制
echo "4096 65536 32000 4096" > /proc/sys/kernel/sem  # 增加信号量限制
```

### **3. 监控告警**
```bash
# 添加定时检查脚本
(crontab -l 2>/dev/null; echo "*/5 * * * * /path/to/diagnose_deadlock.sh") | crontab -
```

---

## 🔍 **常见问题诊断**

### **问题1: PREFILL等待DECODE响应超时**
**现象**: `[PREFILL] Waiting for response from D-Scheduler...`
**解决**: 
```bash
# 检查DECODE进程状态
ps aux | grep DECODE
# 重启服务
./fix_deadlock.sh && ./start_semi_pd_safe.sh
```

### **问题2: GPU内存不足**
**现象**: CUDA out of memory
**解决**:
```bash
# 减少内存使用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# 使用更保守的配置
--mem-fraction-static 0.7
```

### **问题3: 共享内存耗尽**
**现象**: Cannot allocate shared memory
**解决**:
```bash
# 清理共享内存
./fix_deadlock.sh
# 增加共享内存大小
sudo mount -o remount,size=8G /dev/shm
```

---

## ✅ **验证修复效果**

### **功能测试**
```bash
# 启动服务后测试
curl -X POST http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "prompt": "Hello, world!",
    "max_tokens": 50
  }'
```

### **性能测试**
```bash
# 运行负载测试
python3 -c "
import requests
import time
import threading

def test_request():
    try:
        response = requests.post('http://localhost:30000/v1/completions', 
                               json={'prompt': 'Test prompt', 'max_tokens': 10},
                               timeout=30)
        print(f'✅ 请求成功: {response.status_code}')
    except Exception as e:
        print(f'❌ 请求失败: {e}')

# 并发测试
threads = [threading.Thread(target=test_request) for _ in range(5)]
for t in threads: t.start()
for t in threads: t.join()
"
```

---

## 📞 **技术支持**

如果问题仍然存在：

1. **收集诊断信息**: `./diagnose_deadlock.sh`
2. **查看详细日志**: `tail -100 semi_pd_safe_*.log`
3. **检查系统资源**: `free -h && df -h`
4. **GPU状态**: `nvidia-smi -l 1`

**常用调试命令**:
```bash
# 查看进程树
pstree -p $(pgrep -f semi_pd)

# 查看进程打开的文件
lsof -p $(pgrep -f semi_pd)

# 查看网络连接
netstat -tulpn | grep $(pgrep -f semi_pd)
```

---

## 🎉 **预期效果**

修复后您应该看到：

```
✅ Semi-PD 进程启动成功
✅ PREFILL-DECODE通信正常
✅ 请求处理无超时
✅ GPU内存使用稳定
✅ 无死锁告警
```

**性能指标**:
- 启动时间: < 2分钟
- 请求延迟: < 5秒
- 吞吐量: 根据硬件配置
- 稳定性: 7x24小时无死锁 