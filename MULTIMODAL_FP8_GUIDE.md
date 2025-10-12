# Semi-PD 多模态和FP8功能使用指南

本指南介绍如何使用新迁移的多模态（ViT）和FP8量化功能。

---

## 📋 目录

1. [功能概述](#功能概述)
2. [多模态功能](#多模态功能)
3. [FP8量化功能](#fp8量化功能)
4. [测试方法](#测试方法)
5. [故障排查](#故障排查)

---

## 功能概述

### ✅ 已支持的功能

- **多模态推理**：支持Qwen2.5-VL等视觉-语言模型，可以处理图像+文本的混合输入
- **FP8量化**：支持8位浮点量化，减少显存占用并提升推理速度
- **预计算特征缓存**：避免在chunked prefill中重复计算ViT特征，提升性能
- **调试日志**：可选的详细日志，帮助诊断问题

### ❌ 不支持的功能

- **Pipeline Parallel (PP>1)**：本版本专注于Tensor Parallel，不支持流水线并行

---

## 多模态功能

### 1. 启动支持多模态的服务

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-32B-Instruct \
    --tp-size 2 \
    --port 30019 \
    --host 0.0.0.0 \
    --mem-fraction-static 0.85 \
    --context-length 32768 \
    --enable-semi-pd
```

**关键参数说明：**
- `--model-path`: 多模态模型路径（如Qwen2.5-VL）
- `--tp-size 2`: 使用2个GPU进行Tensor Parallel
- `--enable-semi-pd`: 启用Semi-PD模式（PREFILL/DECODE分离）

### 2. 发送多模态请求

#### 方法1：使用测试脚本

```bash
# 测试纯文本
python test_multimodal.py --text-only

# 测试图像+文本
python test_multimodal.py --image /path/to/your/image.jpg
```

#### 方法2：使用curl

```bash
# 准备图像的base64编码
IMAGE_BASE64=$(base64 -w 0 /path/to/your/image.jpg)

# 发送请求
curl http://127.0.0.1:30019/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-VL-32B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,'$IMAGE_BASE64'"
          }
        },
        {
          "type": "text",
          "text": "请描述这张图片的内容。"
        }
      ]
    }],
    "max_tokens": 200
  }'
```

#### 方法3：使用Python SDK

```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:30019/v1",
    api_key="EMPTY"
)

# 纯文本请求
response = client.chat.completions.create(
    model="Qwen2.5-VL-32B-Instruct",
    messages=[
        {"role": "user", "content": "你好！"}
    ]
)
print(response.choices[0].message.content)

# 多模态请求
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_base64 = encode_image("/path/to/image.jpg")

response = client.chat.completions.create(
    model="Qwen2.5-VL-32B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": "这张图片里有什么？"
                }
            ]
        }
    ]
)
print(response.choices[0].message.content)
```

### 3. 多模态性能优化

系统已自动启用以下优化：

- **预计算特征缓存**：ViT特征在第一次计算后会被缓存，后续chunk直接使用
- **进程级缓存**：相同图像的特征会在进程级别缓存，避免重复计算
- **自动清理**：当所有多模态token被消费完毕后，自动释放缓存

**查看多模态日志：**

```bash
# 日志中会显示多模态处理信息
grep "MM_EMBED" /path/to/semipd_tp.log

# 示例输出：
# [MM_EMBED_CALL] rid=abc123 pid=12345 req_idx=0 num_items=1 hash=987654321
# [MM_EMBED_DO_VIT] rid=abc123 precomputed_items=0/1  # 第一次计算
# [MM_EMBED_CALL] rid=abc123 pid=12345 req_idx=0 num_items=1 hash=987654321
# [MM_EMBED_DO_VIT] rid=abc123 precomputed_items=1/1  # 使用缓存
```

---

## FP8量化功能

### 1. 启动FP8量化的服务

```bash
python -m sglang.launch_server \
    --model-path /path/to/fp8-quantized-model \
    --tp-size 2 \
    --port 30019 \
    --quantization fp8 \
    --enable-semi-pd
```

**关键参数：**
- `--quantization fp8`: 启用FP8量化

### 2. 启用FP8调试日志

如果遇到FP8相关问题，可以启用详细的调试日志：

```bash
# 设置环境变量
export SGLANG_FP8_DEBUG=1
# 或
export SEMI_PD_FP8_DEBUG=1

# 然后启动服务
python -m sglang.launch_server ...
```

**调试日志会显示：**
- 权重处理前后的形状、数据类型、设备信息
- Weight scale的形状和元素数量
- 是否检测到权重转置或scale维度错误
- 每个层的量化配置和TP rank信息

**示例日志：**

```
[FP8_DEBUG][process before] prefix=model.layers.0.self_attn.q_proj W.shape=(4096, 4096) dtype=torch.float8_e4m3fn dev=cuda:0 has_w_scale=False
[FP8_DEBUG][process after] prefix=model.layers.0.self_attn.q_proj W.shape=(4096, 4096) dtype=torch.float8_e4m3fn dev=cuda:0 W_scale.shape=(4096,) numel=4096
[FP8_DEBUG][layer] prefix=model.layers.0.self_attn.q_proj tp_rank=0 cutlass=True x.shape=(1, 128, 4096) x.dtype=torch.float16 W.shape=(4096, 4096) W.dtype=torch.float8_e4m3fn
```

### 3. FP8常见问题诊断

**问题1：权重转置错误**

如果日志显示 `suspect_transposed=True`，说明权重可能被错误地转置了。

**问题2：Scale维度错误**

如果日志显示 `suspect_scale_axis='K-dim (expected N)'`，说明weight_scale的维度可能不正确。

**问题3：数据类型不匹配**

检查日志中的 `dtype` 字段，确保权重是 `torch.float8_e4m3fn` 或 `torch.float8_e4m3fnuz`。

---

## 测试方法

### 快速测试

```bash
# 1. 测试纯文本（验证基本功能）
python test_multimodal.py --text-only

# 2. 测试多模态（验证图像处理）
python test_multimodal.py --image /path/to/test_image.jpg

# 3. 同时测试两者
python test_multimodal.py --text-only --image /path/to/test_image.jpg
```

### 性能测试

```bash
# 使用benchmark工具测试吞吐量
python -m sglang.bench_serving \
    --backend sglang \
    --base-url http://127.0.0.1:30019 \
    --dataset-name random \
    --num-prompts 100 \
    --request-rate 1
```

### 压力测试

```bash
# 高并发测试
python -m sglang.bench_serving \
    --backend sglang \
    --base-url http://127.0.0.1:30019 \
    --dataset-name random \
    --num-prompts 1000 \
    --request-rate 10
```

---

## 故障排查

### 问题1：系统卡在warmup阶段

**症状：** 启动后长时间没有响应

**解决方法：**
1. 检查日志中是否有错误信息
2. 确认GPU显存是否充足
3. 尝试降低 `--mem-fraction-static` 参数

### 问题2：多模态请求失败

**症状：** 图像请求返回错误或空响应

**检查清单：**
- [ ] 模型是否支持多模态（如Qwen2.5-VL）
- [ ] 图像是否正确编码为base64
- [ ] 图像大小是否合理（建议<10MB）
- [ ] 请求格式是否正确（参考上面的示例）

**查看日志：**
```bash
# 查看多模态处理日志
grep "MM_EMBED\|Multimodal" /path/to/semipd_tp.log
```

### 问题3：FP8量化效果不佳

**症状：** 输出质量下降或出现异常

**诊断步骤：**
1. 启用FP8调试日志：`export SGLANG_FP8_DEBUG=1`
2. 检查日志中的权重和scale信息
3. 确认模型是否正确量化
4. 尝试使用非量化版本对比

### 问题4：PREFILL-DECODE通信问题

**症状：** 请求卡住或超时

**查看通信日志：**
```bash
# 查看PREFILL-DECODE通信
grep "PREFILL\|DECODE" /path/to/semipd_tp.log | grep "📤\|📥\|⏳\|✅"
```

**正常的通信流程应该是：**
```
[PREFILL] 🚀 get_next_batch_to_run called
[PREFILL] 📤 Send request to D worker
[PREFILL] ⏳ Waiting for response from D worker
[DECODE] 📥 D-Scheduler received N candidate requests
[PREFILL] ✅ Recv response from D worker
```

### 问题5：显存不足

**症状：** OOM (Out of Memory) 错误

**解决方法：**
1. 降低 `--mem-fraction-static` (如从0.85降到0.75)
2. 减少 `--context-length`
3. 使用FP8量化减少显存占用
4. 增加GPU数量（增大 `--tp-size`）

---

## 更多信息

- **迁移总结：** 查看 `MIGRATION_SUMMARY.md` 了解详细的迁移内容
- **日志文件：** 默认位于 `/home/yzh/SemiTP_update/semipd_tp.log`
- **问题反馈：** 如遇到问题，请提供完整的日志和复现步骤

---

**祝使用愉快！** 🎉

