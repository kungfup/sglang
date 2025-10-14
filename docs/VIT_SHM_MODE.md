# VIT Shared Memory (SHM) Mode

## 概述

VIT SHM 模式是 SGLang 的一个重要优化,通过使用 POSIX 共享内存替代 CUDA IPC 来传递 VIT embedding,显著降低 GPU 显存占用并提升性能。

## 核心改进

### 1. 显存优化

**问题**: 原有 CUDA IPC 模式下,VIT Scheduler 显存占用高达 16.43 GB:
- 10 GB 预分配内存池
- 6.43 GB VIT forward 实际使用
- CUDA IPC 共享的 tensor 继续占用 GPU 显存

**解决方案**: SHM 模式
- VIT forward 后立即将 embedding 移到 CPU
- 使用 POSIX 共享内存传递 (CPU 内存,不占用 GPU)
- 移除 10 GB 预分配内存池
- **预期显存占用**: ~5 GB (降低 70%)

### 2. 动态 Batch Size

**问题**: 原有固定 batch_size=1,无法充分利用 GPU

**解决方案**: DynamicMemoryEstimator
- 实时监控 GPU 可用显存
- 动态调整 batch_size (1-8)
- 预留安全余量避免 OOM
- **预期吞吐提升**: 2-3x

### 3. 内存管理

**问题**: 无引用计数,可能过早释放或内存泄漏

**解决方案**: SharedMemoryManager
- 引用计数管理
- LRU 驱逐策略
- 自动清理过期 SHM
- 线程安全

## 使用方法

### 环境变量配置

```bash
# 启用 SHM 模式 (默认)
export SGLANG_VIT_USE_SHM=true

# SHM 容量 (GB)
export SGLANG_VIT_SHM_SIZE_GB=20.0

# 安全余量 (GB)
export SGLANG_VIT_SAFETY_MARGIN_GB=0.5

# 显存估算倍数
export SGLANG_VIT_OVERHEAD_RATIO=4.0

# 回退到 CUDA IPC 模式
export SGLANG_VIT_USE_SHM=false
```

### 启动 Server

```bash
# 安装依赖
pip install posix-ipc

# 启动 server (SHM 模式)
export SGLANG_VIT_USE_SHM=true
python -m sglang.launch_server \
    --model-path /path/to/model \
    --host 127.0.0.1 \
    --port 30017 \
    --device cuda
```

### 发送请求

```python
# 使用 OpenAI 兼容 API
import openai

client = openai.Client(
    base_url="http://127.0.0.1:30017/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="your-model",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "file:///path/to/image.jpg"}}
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

## 性能对比

### 显存占用

| 模式 | VIT Scheduler | 主 Scheduler | 总计 | 可用 |
|------|--------------|-------------|------|------|
| CUDA IPC | 16.43 GB | 27.10 GB | 43.53 GB | 65 MB |
| **SHM** | **~5 GB** | **27.10 GB** | **~32 GB** | **~11 GB** |

### 吞吐量

| 模式 | Batch Size | 吞吐量 (req/s) |
|------|-----------|---------------|
| CUDA IPC | 1 | ~5 |
| **SHM** | **4-8** | **~15-20** |

## 架构设计

### 组件

1. **SharedMemoryHelper**: POSIX SHM 读写工具
   - `tensor2bytes()`: Tensor 序列化
   - `bytes2tensor()`: Tensor 反序列化
   - `write_to_shm()`: 写入共享内存
   - `read_from_shm()`: 读取共享内存

2. **SharedMemoryManager**: SHM 生命周期管理
   - 引用计数
   - LRU 驱逐
   - 自动清理
   - 容量管理

3. **DynamicMemoryEstimator**: 动态显存估算
   - 实时监控可用显存
   - 估算请求显存需求
   - 计算最大 batch_size

### 工作流程

```
VIT Scheduler:
1. VIT forward (GPU)
2. 立即移到 CPU
3. 写入 POSIX SHM
4. 发送响应 (embedding_device="cpu")
5. GPU 显存释放

主 Scheduler:
1. 接收响应
2. 检测 embedding_device=="cpu"
3. 从 POSIX SHM 读取
4. 使用 embedding
5. 通知释放 SHM
```

## 故障排查

### 1. posix_ipc 未安装

**症状**: 启动时警告 "posix_ipc not available"

**解决**:
```bash
pip install posix-ipc
```

### 2. SHM 容量不足

**症状**: 日志显示 "SHM capacity exceeded"

**解决**:
```bash
# 增加 SHM 容量
export SGLANG_VIT_SHM_SIZE_GB=30.0
```

### 3. SHM 未清理

**症状**: `/dev/shm` 目录下残留 `vit_embed_*` 文件

**解决**:
```bash
# 手动清理
python sglang/scripts/cleanup_vit_shm.py --force

# 或者
rm -f /dev/shm/vit_embed_*
```

### 4. OOM 错误

**症状**: CUDA out of memory

**解决**:
```bash
# 增加安全余量
export SGLANG_VIT_SAFETY_MARGIN_GB=1.0

# 降低 overhead ratio
export SGLANG_VIT_OVERHEAD_RATIO=3.0
```

## 测试

### 单元测试

```bash
# 运行 VIT SHM 单元测试
pytest -q sglang/python/sglang/test/test_vit_shm.py
```

### 集成测试

```bash
# 运行集成测试脚本
bash sglang/scripts/test_vit_shm_integration.sh
```

### 性能测试

```bash
# 监控显存
watch -n 1 nvidia-smi

# 发送测试请求
python test_image_request.py
```

## 迁移指南

### 从 CUDA IPC 迁移到 SHM

1. **安装依赖**:
   ```bash
   pip install posix-ipc
   ```

2. **设置环境变量**:
   ```bash
   export SGLANG_VIT_USE_SHM=true
   ```

3. **重启 server**:
   ```bash
   # 停止旧 server
   pkill -f "sglang.launch_server"
   
   # 清理 SHM
   python sglang/scripts/cleanup_vit_shm.py --force
   
   # 启动新 server
   python -m sglang.launch_server ...
   ```

4. **验证**:
   - 检查日志: `grep "Using POSIX SHM mode" server.log`
   - 监控显存: `nvidia-smi`
   - 发送测试请求

### 回退到 CUDA IPC

如果遇到问题,可以回退到 CUDA IPC 模式:

```bash
export SGLANG_VIT_USE_SHM=false
python -m sglang.launch_server ...
```

## 最佳实践

1. **SHM 容量**: 设置为预期最大并发请求数 × 单个 embedding 大小
2. **安全余量**: 至少保留 0.5 GB,避免 OOM
3. **定期清理**: 使用 cleanup 脚本定期清理残留 SHM
4. **监控显存**: 使用 `nvidia-smi` 监控显存使用
5. **日志分析**: 检查日志中的 "✅ Embedding moved to CPU" 确认工作正常

## 参考

- LightLLM VIT 实现: `/home/yzh/semipd_fp8/LightLLM/`
- 设计文档: `/home/yzh/semipd_fp8/VIT_显存优化_代码修改方案.md`
- 对比分析: `/home/yzh/semipd_fp8/LightLLM_vs_SGLang_VIT架构对比分析.md`

