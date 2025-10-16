# 纯文本 PP 模式 "A A A A..." 错误分析报告

## 🔴 问题描述

**现象**：使用纯文本请求（无图片）时，输出末尾出现大量重复的 "A"：

```
图中展示的是一个人。这个人穿着一件简单的T恤和一条牛仔裤。他的头发是黑色的，中等长度，自然地垂在肩上。他的面部表情平静，眼神直视前方。他的肤色看起来是中等色调，皮肤健康。这个人站在一个简单的背景前，背景是模糊的，让人将注意力集中在人物本身。整体上，这是一张简洁而富有表现力的照片，捕捉到了一个人的自然状态和内在平静。 A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
```

**环境**：
- 模型：`Qwen/Qwen2.5-32B-Instruct`（纯文本模型，非 VLM）
- 配置：`--pp-size 2 --mem-fraction-static 0.8 --disable-radix-cache --max-prefill-tokens 32678`
- 代码库：`semipd_pp`

**关键发现**：
- ✅ **串行请求**：正常工作
- ❌ **并发请求**：出现 "A A A A..." 重复
- ❌ **纯文本请求**：也出现问题（不是 VLM 特有的）

## 🔍 根本原因分析

### 1. 问题定位

通过对比 `sglang_vit` 和 `semipd_pp` 的代码，我发现了一个关键差异：

#### `sglang_vit` (正确)

```python
# sglang_vit/python/sglang/srt/managers/scheduler.py:887
mbs[next_mb_id].output_ids = next_pp_outputs["next_token_ids"]
output_result = GenerationBatchResult(
    logits_output=None,
    pp_hidden_states_proxy_tensors=None,
    next_token_ids=next_pp_outputs["next_token_ids"],
    ...
)
self.process_batch_result(mbs[next_mb_id], output_result)
```

**特点**：
- **无条件赋值** `output_ids`
- **总是调用** `process_batch_result`

#### `semipd_pp` (错误)

```python
# semipd_pp/python/sglang/srt/managers/scheduler.py:1104-1128
if "next_token_ids" in next_pp_outputs.tensors:
    mbs[next_mb_id].output_ids = next_pp_outputs["next_token_ids"]
    # ... 构建 output_result
    self.process_batch_result(mbs[next_mb_id], output_result)
    last_mbs[next_mb_id] = mbs[next_mb_id]
```

**特点**：
- **条件检查** `if "next_token_ids" in next_pp_outputs.tensors`
- **只有满足条件才调用** `process_batch_result`

### 2. 问题根源

**条件检查导致的问题**：

1. **PP1 发送的 `next_token_ids` 可能不在 `tensors` 字典中**
   - 在某些情况下（例如 Semi-PD 模式），PP1 可能发送空的 `tensors` 字典
   - 或者 `next_token_ids` 以不同的键名存储

2. **PP0 跳过 `process_batch_result` 调用**
   - 如果条件不满足，PP0 不会调用 `process_batch_result`
   - 导致 `req.output_ids` 不会被更新（`req.output_ids.append(next_token_id)` 不执行）

3. **`req.output_ids` 保持旧值**
   - 在下一次 decode 循环中，`req.output_ids` 仍然是旧值
   - 导致重复生成相同的 token

### 3. "A" 的来源

Token ID `32` 对应的字符是 "A"（在 Qwen 的 tokenizer 中）。

**为什么是 "A"？**

可能的原因：
1. **默认 token ID**：当 `next_token_ids` 缺失时，可能使用默认值 `32`
2. **未初始化的内存**：`output_ids` 未更新，导致使用未初始化的值
3. **错误的 token ID 传递**：PP1 发送的 token ID 有误

## 🛠️ 修复方案

### 方案 1: 移除条件检查（推荐）

**对齐 `sglang_vit` 的行为**，移除条件检查，总是调用 `process_batch_result`：

```python
# semipd_pp/python/sglang/srt/managers/scheduler.py:1096-1129
next_pp_outputs = PPProxyTensors(
    self.pp_group.recv_tensor_dict(
        all_gather_group=self.attn_tp_group
    )
)

# 🔧 FIX: Always assign output_ids and process batch result (align with sglang_vit)
# Remove conditional check to ensure process_batch_result is always called
mbs[next_mb_id].output_ids = next_pp_outputs["next_token_ids"]

# Build output_result
logits_output_args = {
    k[len("logits_output.") :]: v
    for k, v in next_pp_outputs.tensors.items()
    if k.startswith("logits_output.")
}
if len(logits_output_args) > 0:
    logits_output = LogitsProcessorOutput(**logits_output_args)
else:
    logits_output = None

output_result = GenerationBatchResult(
    logits_output=logits_output,
    pp_hidden_states_proxy_tensors=None,
    next_token_ids=next_pp_outputs["next_token_ids"],
    extend_input_len_per_req=next_pp_outputs.tensors.get(
        "extend_input_len_per_req", None
    ),
    extend_logprob_start_len_per_req=next_pp_outputs.tensors.get(
        "extend_logprob_start_len_per_req", None
    ),
    bid=bids[next_mb_id],
    can_run_cuda_graph=result.can_run_cuda_graph if result else False,
)

self.process_batch_result(mbs[next_mb_id], output_result)
last_mbs[next_mb_id] = mbs[next_mb_id]
```

**优点**：
- ✅ 对齐 `sglang_vit` 的实现
- ✅ 确保 `process_batch_result` 总是被调用
- ✅ 简化代码逻辑

**缺点**：
- ⚠️ 如果 `next_token_ids` 确实不存在，会抛出 `KeyError`

### 方案 2: 添加日志和错误处理

**保留条件检查，但添加详细的日志和错误处理**：

```python
if "next_token_ids" in next_pp_outputs.tensors:
    # ... 现有逻辑
else:
    logger.error(
        f"[PP{self.pp_rank}] ❌ next_token_ids not found in PP output! "
        f"tensors keys: {list(next_pp_outputs.tensors.keys())}, "
        f"next_mb_id={next_mb_id}"
    )
    # 使用默认值或抛出异常
    raise RuntimeError("next_token_ids not found in PP output")
```

**优点**：
- ✅ 提供详细的错误信息
- ✅ 帮助诊断问题

**缺点**：
- ❌ 不解决根本问题
- ❌ 仍然会跳过 `process_batch_result`

### 方案 3: 检查 PP1 的发送逻辑

**检查 PP1 是否正确发送 `next_token_ids`**：

查看 `semipd_pp/python/sglang/srt/managers/scheduler.py` 中 PP1 发送 token 的逻辑（行 1056-1091）：

```python
if (
    self.pp_group is not None
    and self.pp_group.is_last_rank
    and (
        not getattr(self.server_args, "enable_semi_pd", False)
        or getattr(self, "instance_role", None) == InstanceRole.DECODE
    )
):
    if self.cur_batch:
        next_token_ids, bids[mb_id] = (
            result.next_token_ids,
            result.bid,
        )
        # Send token packet with explicit tag
        if self.cur_batch.return_logprob:
            send_tok = {
                "next_token_ids": next_token_ids,
                "extend_input_len_per_req": result.extend_input_len_per_req,
                "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
            } | {
                f"logits_output.{k}": v
                for k, v in result.logits_output.__dict__.items()
                if result.logits_output is not None
            }
        else:
            send_tok = {"next_token_ids": next_token_ids}
        self.pp_group.send_tensor_dict(
            send_tok,
            all_gather_group=self.attn_tp_group,
        )
```

**检查点**：
1. ✅ `next_token_ids` 是否正确设置
2. ✅ `send_tok` 字典是否包含 `next_token_ids` 键
3. ✅ `send_tensor_dict` 是否正确发送

## 🧪 调试步骤

### 1. 添加调试日志

在 PP0 接收 PP1 输出的位置添加日志：

```python
# semipd_pp/python/sglang/srt/managers/scheduler.py:1096-1105
next_pp_outputs = PPProxyTensors(
    self.pp_group.recv_tensor_dict(
        all_gather_group=self.attn_tp_group
    )
)

# 🔧 DEBUG: Log received tensors
logger.info(
    f"[PP{self.pp_rank}] 📥 Received PP output: "
    f"next_mb_id={next_mb_id}, "
    f"tensors keys={list(next_pp_outputs.tensors.keys())}, "
    f"has_next_token_ids={'next_token_ids' in next_pp_outputs.tensors}"
)

if "next_token_ids" in next_pp_outputs.tensors:
    logger.info(
        f"[PP{self.pp_rank}] ✅ next_token_ids found: "
        f"shape={next_pp_outputs['next_token_ids'].shape if hasattr(next_pp_outputs['next_token_ids'], 'shape') else 'N/A'}"
    )
else:
    logger.error(
        f"[PP{self.pp_rank}] ❌ next_token_ids NOT found! "
        f"Available keys: {list(next_pp_outputs.tensors.keys())}"
    )
```

### 2. 测试并收集日志

```bash
# 重启服务器
pkill -f "sglang.launch_server"

python -m sglang.launch_server \
    --model-path /home/yzh/model/Qwen/Qwen2.5-32B-Instruct \
    --pp-size 2 \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --max-prefill-tokens 32678 \
    --port 30019 \
    > sglang_pp1.log 2>&1 &

# 测试
python auto_qps.py --port 30019 --start_qps 0.33 --num_samples 5 --print_response

# 查看日志
grep "Received PP output\|next_token_ids" sglang_pp1.log | tail -50
```

## ✅ 已实施的修复

### 修复：移除条件检查，对齐 `sglang_vit`

**文件**：`semipd_pp/python/sglang/srt/managers/scheduler.py`

**修改位置**：行 1093-1134

**Before**:
```python
if "next_token_ids" in next_pp_outputs.tensors:
    mbs[next_mb_id].output_ids = next_pp_outputs["next_token_ids"]
    # ... 构建 output_result
    self.process_batch_result(mbs[next_mb_id], output_result)
    last_mbs[next_mb_id] = mbs[next_mb_id]
```

**After**:
```python
# 🔧 FIX: Always assign output_ids and process batch result (align with sglang_vit)
# Remove conditional check to ensure process_batch_result is always called
# This fixes the "A A A A..." bug where req.output_ids is not updated
mbs[next_mb_id].output_ids = next_pp_outputs["next_token_ids"]

# Build logits_output from received tensors
logits_output_args = {
    k[len("logits_output.") :]: v
    for k, v in next_pp_outputs.tensors.items()
    if k.startswith("logits_output.")
}
if len(logits_output_args) > 0:
    logits_output = LogitsProcessorOutput(**logits_output_args)
else:
    logits_output = None

output_result = GenerationBatchResult(
    logits_output=logits_output,
    pp_hidden_states_proxy_tensors=None,
    next_token_ids=next_pp_outputs["next_token_ids"],
    extend_input_len_per_req=next_pp_outputs.tensors.get(
        "extend_input_len_per_req", None
    ),
    extend_logprob_start_len_per_req=next_pp_outputs.tensors.get(
        "extend_logprob_start_len_per_req", None
    ),
    bid=bids[next_mb_id],
    can_run_cuda_graph=result.can_run_cuda_graph if result else False,
)

self.process_batch_result(mbs[next_mb_id], output_result)
last_mbs[next_mb_id] = mbs[next_mb_id]
```

**关键变化**：
1. ✅ **移除条件检查** `if "next_token_ids" in next_pp_outputs.tensors`
2. ✅ **总是调用** `process_batch_result`
3. ✅ **对齐 `sglang_vit`** 的实现
4. ✅ **修复 `can_run_cuda_graph`** 参数（添加 `if result else False` 保护）

## 🧪 测试步骤

1. **重启服务器**:
   ```bash
   pkill -f "sglang.launch_server"

   cd /home/yzh/semipd_pp_vit/semipd_pp

   nohup python -m sglang.launch_server \
       --model-path /home/yzh/model/Qwen/Qwen2.5-32B-Instruct \
       --pp-size 2 \
       --mem-fraction-static 0.8 \
       --disable-radix-cache \
       --max-prefill-tokens 32678 \
       --port 30019 \
       --mm-attention-backend fa3 \
       --context-length 32768 \
       --chat-template qwen2-vl \
       --attention-backend fa3 \
       > /home/yzh/semipd_pp_vit/sglang_pp1.log 2>&1 &
   ```

2. **测试纯文本请求**:
   ```bash
   cd /home/yzh/semipd_pp_vit
   python auto_qps.py --port 30019 --start_qps 0.33 --num_samples 5 --print_response
   ```

3. **检查输出**:
   ```bash
   # 查看响应（应该没有 "A A A A..." 重复）
   tail -100 /home/yzh/semipd_pp_vit/sglang_pp1.log | grep "Response"

   # 查看 PP 通信日志
   tail -500 /home/yzh/semipd_pp_vit/sglang_pp1.log | grep "Forwarding\|Received PP output"
   ```

## 🎯 预期效果

修复后：
- ✅ **纯文本请求**：正常输出，无 "A A A A..." 重复
- ✅ **串行请求**：正常工作
- ✅ **并发请求**：正常工作
- ✅ **VLM 请求**：正常工作（如果模型支持）
- ✅ **PP 通信**：PP0 总是处理 PP1 的输出

## 📝 下一步行动

1. **重启服务器并测试**：运行上述测试步骤
2. **验证修复**：确认输出正常，无 "A A A A..." 重复
3. **扩展测试**：测试不同的并发级别和请求类型
4. **如果仍有问题**：提供详细的错误日志和测试场景

## 🔗 相关文件

- `semipd_pp/python/sglang/srt/managers/scheduler.py` - 主调度器（已修复）
- `sglang_vit/python/sglang/srt/managers/scheduler.py` - 参考实现
- `semipd_pp/python/sglang/srt/managers/scheduler_output_processor_mixin.py` - 输出处理逻辑
- `semipd_pp/TEXT_ONLY_PP_BUG_ANALYSIS.md` - 本文档

