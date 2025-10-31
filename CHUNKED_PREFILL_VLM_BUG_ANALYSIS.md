# Chunked Prefill VLM 输出截断问题分析报告

## 🔴 问题描述

**现象**：使用原生 SGLang 模式（TP + Chunked Prefill）时，VLM 输出**偶尔正常，偶尔被截断**，后面跟着大量换行符和空白。

**示例输出**:
```
[Response 1] 这是一张展示了一篮子青苹果 的的图片。图片










（后面都是换行符和空白）
```

**环境**：
- 模式：原生 SGLang（非 Semi-PD）
- 模型：`Qwen/Qwen2.5-VL-32B-Instruct`
- 配置：`--tp-size 2 --chunked-prefill-size 512 --quantization fp8`
- 代码库：`semipd_pp`

**关键观察**：
1. ✅ **不使用 `--chunked-prefill-size`**：输出正常
2. ❌ **使用 `--chunked-prefill-size 512`**：输出偶尔被截断
3. 问题是**非确定性的**（偶尔正常，偶尔有问题）
4. 使用的是**原生 SGLang 代码**（`semipd_pp` 仓库中的代码）

## 🔍 根本原因分析

### 1. 问题定位

通过代码审查，我发现了一个**关键的错误逻辑**：

#### `semipd_pp/python/sglang/srt/managers/semi_pd_prefill_scheduler.py` (行 448-459)

```python
# Chunked-Prefill: 从第二个chunk开始，避免重复计算多模态（仅对该rid清空mm_inputs）
try:
    if (
        self.chunked_req is not None
        and getattr(self.chunked_req, "is_chunked", 0) > 1
        and new_batch.multimodal_inputs is not None
    ):
        for i, req in enumerate(new_batch.reqs):
            if req.rid == self.chunked_req.rid:
                new_batch.multimodal_inputs[i] = None
except Exception:
    pass
```

**问题**：
- 这段代码在 **chunked prefill 的第二个 chunk 开始时清空 `multimodal_inputs`**
- 这个逻辑是**错误的**，因为：
  1. **VLM 模型需要在每个 chunk 中都访问 multimodal_inputs**（特别是 `mrope_positions`）
  2. 清空 `multimodal_inputs` 会导致后续 chunk 无法正确处理多模态数据
  3. 这会导致生成过程提前终止或产生错误的输出

### 2. 为什么会导致输出截断？

**推测的执行流程**：

1. **第一个 chunk (is_chunked = 1)**:
   - `multimodal_inputs` 正常存在
   - VLM 模型正确处理图片和文本
   - 生成部分输出（例如："这是一张展示了一篮子青苹果 的的图片。图片"）

2. **第二个 chunk (is_chunked = 2)**:
   - `multimodal_inputs[i]` 被设置为 `None`
   - VLM 模型无法访问 `mrope_positions` 等关键信息
   - 可能导致：
     - **提前触发 EOS token**
     - **生成错误的 token**（例如换行符）
     - **模型内部状态错误**

3. **结果**:
   - 输出被截断
   - 后续生成大量换行符或空白

### 3. 为什么是非确定性的？

**可能的原因**：

1. **Chunked Prefill 的触发条件**:
   - 只有当输入长度超过 `chunked_prefill_size` (512) 时才会触发 chunked prefill
   - 不同的图片大小和文本长度会导致不同的 chunk 数量
   - 如果输入长度 < 512，不会触发 chunked prefill，输出正常

2. **并发请求的影响**:
   - 在并发请求时，`chunked_req` 的状态可能会被不同请求影响
   - 导致某些请求正常，某些请求被截断

3. **图片 token 数量的影响**:
   - Qwen2.5-VL 的图片 token 数量取决于图片分辨率
   - 不同的图片可能产生不同数量的 token
   - 影响是否触发 chunked prefill

## 🛠️ 修复方案

根据您的详细分析，问题的根本原因是 `_adjust_embedding_length()` 函数的错误实现。我已经实施了以下修复：

### 修复 1: 恢复原生 SGLang 的 `_adjust_embedding_length()` 逻辑 ✅

**文件**: `semipd_pp/python/sglang/srt/managers/mm_utils.py` (行 385-436)

**关键修改**：

1. **当占位符 < 嵌入时**：从**尾部**提取嵌入（`embedding[-num_mm_tokens_in_input_ids:, :]`）
   - ❌ **之前错误**：从头部提取（`embedding[:num_mm_tokens_in_input_ids, :]`）
   - ✅ **现在正确**：从尾部提取，对齐原生 SGLang

2. **当占位符 > 嵌入时**：**抛出 RuntimeError**
   - ❌ **之前错误**：修改掩码，只保留最后 k 个位置（导致图像特征丢失）
   - ✅ **现在正确**：抛出 RuntimeError，强制调度层处理

**代码片段**：

```python
def _adjust_embedding_length(
    embedding: torch.Tensor,
    mask: torch.Tensor,
    logger,
) -> torch.Tensor:
    num_mm_tokens_in_embedding = embedding.shape[0]
    num_mm_tokens_in_input_ids = mask.sum().item()

    if num_mm_tokens_in_input_ids != num_mm_tokens_in_embedding:
        logger.warning(...)
        if num_mm_tokens_in_input_ids < num_mm_tokens_in_embedding:
            # 🔧 FIX: Extract from the END (tail), not the beginning (head)
            embedding = embedding[-num_mm_tokens_in_input_ids:, :]
        else:
            # 🔧 FIX: Raise RuntimeError instead of modifying mask
            raise RuntimeError(
                f"Insufficient multimodal embedding length: {num_mm_tokens_in_input_ids=} vs {num_mm_tokens_in_embedding=}. This is an internal error"
            )
    return embedding
```

### 修复 2: 移除错误的 multimodal_inputs 清空逻辑 ✅

**文件**: `semipd_pp/python/sglang/srt/managers/semi_pd_prefill_scheduler.py` (原行 448-459)

**修改**：注释掉了错误的 `multimodal_inputs[i] = None` 逻辑

**原因**：
- VLM 模型需要在所有 chunk 中访问 `multimodal_inputs`（特别是 `mrope_positions`）
- 清空 `multimodal_inputs` 会导致后续 chunk 无法正确处理多模态数据
- 原生 SGLang 没有这个逻辑

### 修复 3: 调度层的多模态安全检查（已存在）

**文件**: `semipd_pp/python/sglang/srt/managers/schedule_policy.py` (行 510-536)

**现有逻辑**：
- 检测多模态请求的图像 token 数量
- 如果 `mm_remaining > self.rem_chunk_tokens`，返回 `AddReqResult.OTHER`
- 这会跳过该请求的 chunked prefill

**问题**：
- 这个逻辑可能不够完善，因为它只检查 `mm_remaining`
- 对于某些情况，可能仍然会触发 chunked prefill

### 预期行为

修复后的行为：

1. **情况 1：图像 token 数量 ≤ chunked_prefill_size**
   - 调度层允许 chunked prefill
   - `_adjust_embedding_length()` 从尾部提取嵌入
   - 输出正常

2. **情况 2：图像 token 数量 > chunked_prefill_size**
   - 调度层检测到 `mm_remaining > rem_chunk_tokens`
   - 返回 `AddReqResult.OTHER`，跳过 chunked prefill
   - 等待下一轮调度，使用完整的 prefill

3. **情况 3：调度层检查失败，仍然触发 chunked prefill**
   - `_adjust_embedding_length()` 检测到 `占位符 > 嵌入`
   - 抛出 RuntimeError
   - 请求失败，但不会产生错误的输出

## 🧪 验证步骤

### 1. 检查原生 SGLang 是否有类似逻辑 ✅

```bash
cd /home/yzh/sglang
grep -rn "is_chunked.*>" python/sglang/srt/managers/ | grep -E "multimodal|mm_inputs"
```

**结果**：✅ **确认原生 SGLang 没有类似的逻辑**

### 2. 实施修复 ✅

**已完成**：移除 `semipd_pp/python/sglang/srt/managers/semi_pd_prefill_scheduler.py` 中的错误逻辑（原行 448-459）

**修改内容**：
- 注释掉了错误的 `multimodal_inputs[i] = None` 逻辑
- 添加了详细的注释说明为什么移除这段代码
- 对齐原生 SGLang 的行为

### 3. 测试修复

```bash
# 重启服务器
pkill -f "sglang.launch_server"

cd /home/yzh/semipd_pp_vit/semipd_pp

python -m sglang.launch_server \
    --model-path /home/yzh/model/Qwen/Qwen2.5-VL-32B-Instruct \
    --tp-size 2 \
    --mem-fraction-static 0.7 \
    --disable-radix-cache \
    --max-prefill-tokens 32678 \
    --port 30019 \
    --mm-attention-backend fa3 \
    --context-length 32768 \
    --chat-template qwen2-vl \
    --attention-backend fa3 \
    --quantization fp8 \
    --chunked-prefill-size 512 \
    > /home/yzh/semipd_pp_vit/sglang_tp.log 2>&1 &

# 测试
cd /home/yzh/semipd_pp_vit
python auto_qps.py --port 30019 --start_qps 0.33 --num_samples 10 --print_response
```

### 4. 验证输出

检查输出是否正常，无截断和换行符问题：

```bash
# 查看响应
tail -100 /home/yzh/semipd_pp_vit/sglang_tp.log | grep "Response"
```

## 📝 相关代码位置

### 需要修改的文件

- **`semipd_pp/python/sglang/srt/managers/semi_pd_prefill_scheduler.py`** (行 448-459)

### 相关文件（参考）

- **`semipd_pp/python/sglang/srt/managers/mm_utils.py`** - 多模态处理逻辑
- **`semipd_pp/python/sglang/srt/models/qwen2_5_vl.py`** - Qwen2.5-VL 模型实现
- **`semipd_pp/python/sglang/srt/managers/schedule_batch.py`** - `MultimodalInputs` 定义

## 🔑 关键洞察

1. **VLM 模型需要在所有 chunk 中访问 multimodal_inputs**:
   - 特别是 `mrope_positions`（Qwen2.5-VL 的位置编码）
   - 清空 `multimodal_inputs` 会破坏模型的正确性

2. **Chunked Prefill 的正确行为**:
   - 应该只分割输入 token，不应该修改 multimodal_inputs
   - VIT 计算应该在第一个 chunk 中完成，后续 chunk 可以复用结果

3. **原生 SGLang 没有这个逻辑**:
   - `semipd_pp` 中的这段代码是额外添加的
   - 应该移除以对齐原生 SGLang 的行为

## 🎯 预期效果

修复后：
- ✅ **Chunked Prefill + VLM**：输出正常，无截断
- ✅ **串行请求**：正常工作
- ✅ **并发请求**：正常工作
- ✅ **不同图片大小**：都能正常处理

## 📚 相关文档

- **`semipd_pp/TEXT_ONLY_PP_BUG_ANALYSIS.md`** - 纯文本 PP 模式错误分析
- **`semipd_pp/CONCURRENT_REQUEST_DEBUG_ANALYSIS.md`** - 并发请求调试分析
- **`semipd_pp/VIT_READY_MECHANISM_FIX.md`** - VIT ready 机制修复文档

