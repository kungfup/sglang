# SGLang Semi-PD 调试日志控制

## 🎯 概述

为了方便调试和生产环境的使用，我们添加了环境变量 `SGLANG_DISABLE_DEBUG_LOGS` 来控制详细调试日志的显示。

## 📝 控制的日志类型

当设置 `SGLANG_DISABLE_DEBUG_LOGS=1` 时，以下调试日志将被禁用：

### CUDA Graph 相关日志
- `[CG-LAUNCH] cudaGraphLaunch host_cost=...`
- `[CG-DETAILED-TIMING] prepare=...ms, core_replay=...ms, post=...ms, total=...ms`
- `[CG-DEVICE] replay:entry device=..., current_stream=..., graph_stream=...`
- `[CG-STREAM] replay_prepare bs=..., raw_bs=..., mode=..., stream_id=...`
- `[CG-STREAM-FIX] about_to_replay current=..., graph=...`
- `[CG-STREAM-EARLY-FIX] switching from ... to ...`
- `[CG-CRITICAL-STREAM-SWITCH] forced stream switch at replay time`

### 调度器和解码器日志
- `[DBG_SCHEDULER] rid=... send_off=... read_off=... send_len=... head=...`
- `[DBG_DETOKENIZER] batch=... head=... read_offsets=...`
- `[DBG_DETOKENIZER_DECODE] head_text=...`
- `[DBG_DETOKENIZER_INIT] tokenizer_path=... mode=... trust_rc=...`

## 🚀 使用方法

### 方法1：环境变量设置
```bash
# 禁用调试日志
export SGLANG_DISABLE_DEBUG_LOGS=1

# 启用调试日志（默认）
export SGLANG_DISABLE_DEBUG_LOGS=0
# 或
unset SGLANG_DISABLE_DEBUG_LOGS
```

### 方法2：启动时临时设置
```bash
# 禁用调试日志启动服务器
SGLANG_DISABLE_DEBUG_LOGS=1 python -m sglang.launch_server \
    --model-path /path/to/your/model \
    --enable-semi-pd \
    --tp-size 2 \
    --port 30000

# 启用调试日志启动服务器（默认）
python -m sglang.launch_server \
    --model-path /path/to/your/model \
    --enable-semi-pd \
    --tp-size 2 \
    --port 30000
```

### 方法3：使用控制脚本
```bash
# 查看当前设置
python control_debug_logs.py

# 启用调试日志
python control_debug_logs.py enable

# 禁用调试日志
python control_debug_logs.py disable

# 创建启动脚本
python control_debug_logs.py create-scripts
```

## 📊 效果对比

### 启用调试日志时（默认）
```
[2025-08-20 07:57:38 DECODE TP0] [CG-DETAILED-TIMING] prepare=0.036ms, core_replay=0.063ms, post=0.001ms, total=0.102ms
[2025-08-20 07:57:38 DECODE TP1] [CG-DEVICE] replay:exit device=1, current_stream=<torch.cuda.Stream device=cuda:1 cuda_stream=0x3251d100>, graph_stream=<torch.cuda.Stream device=cuda:1 cuda_stream=0x3251d100>
[2025-08-20 07:57:38 DECODE TP1] [CG-LAUNCH] cudaGraphLaunch host_cost=0.083 ms, bs=1, mode=DECODE
[2025-08-20 07:57:38 DECODE TP0] [DBG_SCHEDULER] rid=59037e5d send_off=0 read_off=5 send_len=13 head=[6722, 3283, 315]
[2025-08-20 07:57:38] [DBG_DETOKENIZER] batch=1 head=[[6722, 3283, 315]] read_offsets=[5]
[2025-08-20 07:57:38] [DBG_DETOKENIZER_DECODE] head_text=' capital city of France is Paris'
```

### 禁用调试日志时
```
[2025-08-20 07:57:38] INFO:     127.0.0.1:41146 - "POST /generate HTTP/1.1" 200 OK
```

## 🎯 推荐使用场景

### 🔍 开发/调试阶段
- **启用调试日志**（默认设置）
- 帮助分析性能问题、流同步问题、CUDA Graph问题
- 便于理解系统内部工作流程

### 🚀 生产环境
- **禁用调试日志**（`SGLANG_DISABLE_DEBUG_LOGS=1`）
- 减少日志噪声，提高日志可读性
- 略微减少日志I/O开销
- 保持系统稳定性

### 🧪 性能测试
- **禁用调试日志**
- 避免日志打印影响性能测试结果
- 获得更准确的性能数据

## ⚠️ 注意事项

1. **环境变量优先级**：
   - `1`, `true`, `yes`（不区分大小写）会禁用日志
   - 其他值或未设置会启用日志

2. **仅影响调试日志**：
   - 不影响正常的INFO、WARNING、ERROR日志
   - 不影响API响应和功能

3. **实时生效**：
   - 环境变量在进程启动时读取
   - 修改后需要重启服务器才能生效

## 📁 相关文件

- `control_debug_logs.py` - 日志控制演示脚本
- `launch_debug.sh` - 启用调试日志的启动脚本
- `launch_quiet.sh` - 禁用调试日志的启动脚本

## 💡 小贴士

在调试性能问题时，建议先启用调试日志确定问题位置，然后在解决问题后禁用调试日志以获得清洁的日志输出。 