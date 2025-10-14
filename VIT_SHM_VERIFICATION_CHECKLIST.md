# VIT SHM 模式验证清单

## 📋 验证步骤

### 1. 环境准备

- [ ] 安装 posix-ipc
  ```bash
  pip install posix-ipc
  ```

- [ ] 验证安装
  ```bash
  python -c "import posix_ipc; print('✅ posix_ipc available')"
  ```

- [ ] 清理旧的 SHM
  ```bash
  python scripts/cleanup_vit_shm.py --force
  ```

### 2. 代码验证

- [ ] 检查语法错误
  ```bash
  python -m py_compile sglang/python/sglang/srt/managers/vit_scheduler.py
  python -m py_compile sglang/python/sglang/srt/managers/vit_scheduler_client.py
  ```

- [ ] 运行单元测试
  ```bash
  pytest -q sglang/python/sglang/test/test_vit_shm.py -v
  ```

### 3. 启动 Server (SHM 模式)

- [ ] 设置环境变量
  ```bash
  export SGLANG_VIT_USE_SHM=true
  export SGLANG_VIT_SHM_SIZE_GB=20.0
  export SGLANG_VIT_SAFETY_MARGIN_GB=0.5
  export SGLANG_VIT_OVERHEAD_RATIO=4.0
  ```

- [ ] 启动 server
  ```bash
  python -m sglang.launch_server \
      --model-path /path/to/model \
      --host 127.0.0.1 \
      --port 30017 \
      --device cuda \
      > /tmp/sglang_server.log 2>&1 &
  ```

- [ ] 检查启动日志
  ```bash
  grep "Using POSIX SHM mode" /tmp/sglang_server.log
  ```
  
  **预期输出**:
  ```
  [VIT Scheduler] ✅ Using POSIX SHM mode (LightLLM-inspired)
  [VIT Client] ✅ Using POSIX SHM mode for embedding transfer
  ```

### 4. 显存验证

- [ ] 记录启动前显存
  ```bash
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
  ```

- [ ] 记录启动后显存
  ```bash
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
  ```

- [ ] 验证 VIT Scheduler 显存 < 6 GB
  
  **预期**: VIT Scheduler 显存占用约 5 GB (vs 原来的 16.43 GB)

### 5. 功能验证

- [ ] 发送测试请求
  ```bash
  export MODEL_PATH=/path/to/model
  export IMAGE_PATH=/path/to/test/image.jpg
  python test_image_request.py
  ```

- [ ] 检查 CPU 转移日志
  ```bash
  grep "Embedding moved to CPU" /tmp/sglang_server.log
  ```
  
  **预期输出**:
  ```
  [VIT Runner] ✅ Embedding moved to CPU, GPU memory released (SHM mode)
  ```

- [ ] 检查 SHM 写入日志
  ```bash
  grep "Writing.*embeddings to SHM" /tmp/sglang_server.log
  ```
  
  **预期输出**:
  ```
  [VIT Scheduler] 📝 Writing N embeddings to SHM...
  ```

- [ ] 检查 SHM 读取日志
  ```bash
  grep "Read embedding from SHM" /tmp/sglang_server.log
  ```
  
  **预期输出**:
  ```
  [VIT Client] ✅ Read embedding from SHM: shape=..., device=cpu
  ```

- [ ] 验证响应正确性
  
  **预期**: 请求成功返回,内容正确

### 6. 性能验证

- [ ] 记录请求前显存
  ```bash
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
  ```

- [ ] 发送多个请求 (测试批处理)
  ```bash
  for i in {1..10}; do
      python test_image_request.py &
  done
  wait
  ```

- [ ] 记录请求后显存
  ```bash
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
  ```

- [ ] 检查 batch_size 日志
  ```bash
  grep "batch_size" /tmp/sglang_server.log | tail -20
  ```
  
  **预期**: batch_size > 1 (动态调整)

- [ ] 检查吞吐量
  ```bash
  grep "compute_time" /tmp/sglang_server.log | tail -20
  ```
  
  **预期**: 平均处理时间降低

### 7. 稳定性验证

- [ ] 检查 OOM 错误
  ```bash
  grep -i "out of memory" /tmp/sglang_server.log
  ```
  
  **预期**: 无 OOM 错误

- [ ] 检查 SHM 泄漏
  ```bash
  ls -lh /dev/shm/vit_embed_* | wc -l
  ```
  
  **预期**: SHM 对象数量合理 (< 100)

- [ ] 检查引用计数
  ```bash
  grep "ref_count" /tmp/sglang_server.log | tail -20
  ```
  
  **预期**: 引用计数正常增减

### 8. 清理验证

- [ ] 停止 server
  ```bash
  pkill -f "sglang.launch_server"
  ```

- [ ] 运行清理脚本
  ```bash
  python scripts/cleanup_vit_shm.py --dry-run
  ```
  
  **预期**: 列出残留的 SHM 对象

- [ ] 强制清理
  ```bash
  python scripts/cleanup_vit_shm.py --force
  ```
  
  **预期**: 成功清理所有 SHM 对象

- [ ] 验证清理结果
  ```bash
  ls /dev/shm/vit_embed_* 2>/dev/null
  ```
  
  **预期**: 无残留文件

### 9. 回退验证 (CUDA IPC 模式)

- [ ] 设置环境变量
  ```bash
  export SGLANG_VIT_USE_SHM=false
  ```

- [ ] 启动 server
  ```bash
  python -m sglang.launch_server \
      --model-path /path/to/model \
      --host 127.0.0.1 \
      --port 30017 \
      --device cuda \
      > /tmp/sglang_server_ipc.log 2>&1 &
  ```

- [ ] 检查启动日志
  ```bash
  grep "Using CUDA IPC mode" /tmp/sglang_server_ipc.log
  ```
  
  **预期输出**:
  ```
  [VIT Client] ⚠️ Using CUDA IPC mode (legacy, SGLANG_VIT_USE_SHM=false)
  ```

- [ ] 发送测试请求
  ```bash
  python test_image_request.py
  ```
  
  **预期**: 请求成功 (向后兼容)

- [ ] 停止 server
  ```bash
  pkill -f "sglang.launch_server"
  ```

### 10. 集成测试

- [ ] 运行集成测试脚本
  ```bash
  bash scripts/test_vit_shm_integration.sh
  ```
  
  **预期**: 所有测试通过

## 📊 性能指标对比

### 显存占用

| 指标 | CUDA IPC | SHM | 改进 | 验证 |
|------|---------|-----|------|------|
| VIT Scheduler | 16.43 GB | ~5 GB | ↓ 70% | [ ] |
| GPU 0 可用 | 65 MB | ~11 GB | ↑ 169x | [ ] |

### 性能

| 指标 | CUDA IPC | SHM | 改进 | 验证 |
|------|---------|-----|------|------|
| Batch Size | 1 | 4-8 | ↑ 4-8x | [ ] |
| 吞吐量 | ~5 req/s | ~15-20 req/s | ↑ 2-3x | [ ] |

### 稳定性

| 指标 | 验证 |
|------|------|
| 无 OOM 错误 | [ ] |
| 无内存泄漏 | [ ] |
| 引用计数正常 | [ ] |
| 向后兼容 | [ ] |

## 🐛 常见问题

### 问题 1: posix_ipc 导入失败

**症状**:
```
ImportError: No module named 'posix_ipc'
```

**解决**:
```bash
pip install posix-ipc
```

### 问题 2: SHM 容量不足

**症状**:
```
[VIT Scheduler] ⚠️ SHM capacity exceeded
```

**解决**:
```bash
export SGLANG_VIT_SHM_SIZE_GB=30.0
```

### 问题 3: SHM 未清理

**症状**:
```bash
ls /dev/shm/vit_embed_* | wc -l
# 输出: 100+
```

**解决**:
```bash
python scripts/cleanup_vit_shm.py --force
```

### 问题 4: OOM 错误

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决**:
```bash
# 增加安全余量
export SGLANG_VIT_SAFETY_MARGIN_GB=1.0

# 降低 overhead ratio
export SGLANG_VIT_OVERHEAD_RATIO=3.0
```

## ✅ 验证通过标准

- [ ] 所有单元测试通过
- [ ] 启动日志显示 "Using POSIX SHM mode"
- [ ] VIT Scheduler 显存 < 6 GB
- [ ] GPU 0 可用显存 > 10 GB
- [ ] 请求成功返回,内容正确
- [ ] 日志显示 "Embedding moved to CPU"
- [ ] 日志显示 "Read embedding from SHM"
- [ ] 无 OOM 错误
- [ ] batch_size > 1
- [ ] 吞吐量提升 2-3x
- [ ] SHM 清理脚本工作正常
- [ ] CUDA IPC 模式向后兼容

## 📝 验证报告模板

```markdown
# VIT SHM 模式验证报告

## 环境信息
- GPU: [型号]
- CUDA: [版本]
- PyTorch: [版本]
- SGLang: [版本]

## 显存占用
- VIT Scheduler (SHM): [X] GB
- VIT Scheduler (IPC): [Y] GB
- 改进: [Z]%

## 性能指标
- Batch Size: [N]
- 吞吐量: [X] req/s
- 改进: [Y]x

## 功能验证
- [ ] CPU 转移: [通过/失败]
- [ ] SHM 写入: [通过/失败]
- [ ] SHM 读取: [通过/失败]
- [ ] 引用计数: [通过/失败]
- [ ] LRU 驱逐: [通过/失败]

## 稳定性
- [ ] OOM 错误: [有/无]
- [ ] 内存泄漏: [有/无]
- [ ] 向后兼容: [通过/失败]

## 结论
[通过/失败]

## 备注
[其他说明]
```

## 🎯 下一步

验证通过后:
1. 提交代码到版本控制
2. 更新 CHANGELOG
3. 通知团队成员
4. 部署到生产环境 (可选)

