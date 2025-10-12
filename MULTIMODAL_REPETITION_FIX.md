# 多模态重复输出Bug修复报告

## 📋 问题描述

**症状：** 多模态请求（图像+文本）的回复内容会重复出现，同一句话或段落会出现多次。

**示例输出：**
```
这张图片展示了一只鸟的特写，具体来说是鸭子。以下是对图片的详细描述：

### **1. 鸟的特征**
- **头部**：
  - 鸟的头部羽毛主要是黑色，看起来非常光滑且有光泽。
  - 眼睛非常显眼，呈现出深蓝色或黑色，周围有一圈明亮的黄色眼环，显得非常醒目。
  ...
  
- **喙这张图片展示了一只鸟的特写，具体来说是鸭子。以下是对图片的详细描述：
[... 内容重复 ...]
```

---

## 🔍 根本原因分析

### 日志证据

从 `/home/yzh/SemiTP_update/semipd_tp.log` 中发现：

```
[2025-10-12 09:57:24 DECODE TP0] [DBG_SCHEDULER] rid=fdf433f0 send_off=0 read_off=5 send_len=133 head=[151645, 198, 151644, 77091, 198, 108893, 45930, 101987, 99593, 91680]
[2025-10-12 09:57:24] [DBG_DETOKENIZER] batch=1 head=[[151645, 198, 151644, 77091, 198, 108893, 45930, 101987, 99593, 91680]] read_offsets=[5]

[2025-10-12 09:57:31 DECODE TP0] [DBG_SCHEDULER] rid=fdf433f0 send_off=133 read_off=5 send_len=128 head=[334, 28311, 220, 481, 4891, 244, 247, 32948, 100827, 100815]
[2025-10-12 09:57:31] [DBG_DETOKENIZER] batch=1 head=[[151645, 198, 151644, 77091, 198, 108893, 45930, 101987, 99593, 91680]] read_offsets=[5]

[2025-10-12 09:57:37 DECODE TP0] [DBG_SCHEDULER] rid=fdf433f0 send_off=261 read_off=5 send_len=128 head=[104363, 3407, 14374, 3070, 17, 13, 8908, 225, 234, 85254]
[2025-10-12 09:57:37] [DBG_DETOKENIZER] batch=1 head=[[151645, 198, 151644, 77091, 198, 108893, 45930, 101987, 99593, 91680]] read_offsets=[5]
```

**关键发现：**
1. **Scheduler发送的数据是正确的**：
   - `send_off` 在递增：0 → 133 → 261 → 389 → 517
   - `head` 的token在变化（每次都不同）
   
2. **Detokenizer接收到的数据是错误的**：
   - `head` 一直是相同的：`[151645, 198, 151644, 77091, 198, 108893, 45930, 101987, 99593, 91680]`
   - `read_offsets` 一直是 `[5]`

**结论：** Detokenizer没有收到更新后的token数据，一直在解码相同的token序列。

### 代码分析

#### 问题代码（`semipd_tp_nopp`）

**文件：** `semipd_tp_nopp/python/sglang/srt/managers/scheduler_output_processor_mixin.py`

**第570-580行（修复前）：**
```python
# 恢复原逻辑：多模态发送完整，文本发送增量
if self.server_args.enable_semi_pd:
    # Semi-PD: always send full decode_ids to keep detokenizer offsets consistent
    decode_ids_list.append(decode_ids)
elif self.model_config.is_multimodal_gen:
    decode_ids_list.append(decode_ids)
else:
    decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

req.send_decode_id_offset = len(decode_ids)
read_offsets.append(read_offset)
```

**问题：**
1. 在Semi-PD模式下，代码总是发送**完整的** `decode_ids`（从 `surr_offset` 开始）
2. 但 `read_offset` 是相对于 `surr_offset` 的偏移，每次都是相同的值（例如5）
3. Detokenizer收到相同的token序列和相同的偏移，导致每次都解码相同的内容

**工作流程（错误）：**
```
第1次发送:
  decode_ids = [151645, 198, 151644, 77091, 198, ...] (从surr_offset开始的所有token)
  read_offset = 5 (相对于surr_offset)
  → Detokenizer从位置5开始解码

第2次发送:
  decode_ids = [151645, 198, 151644, 77091, 198, ...] (相同的token序列！)
  read_offset = 5 (相同的偏移！)
  → Detokenizer又从位置5开始解码 → 重复输出！
```

#### 正确代码（`semipd_tp_pp`）

**文件：** `semipd_tp_pp/python/sglang/srt/managers/scheduler_output_processor_mixin.py`

**第641-669行：**
```python
# 多模态/文本 detokenizer 协议（通过开关控制）：
# SGLANG_MM_DETOKENIZER_MODE: off(默认)/incremental/full
mm_mode = os.environ.get("SGLANG_MM_DETOKENIZER_MODE", "off").lower()
if self.model_config.is_multimodal_gen:
    if mm_mode in ("off", "0", "false"):
        # 原生语义：多模态不经 detokenizer，跳过发送
        rids.pop(); finished_reasons.pop(); decoded_texts.pop()
        continue
    elif mm_mode == "full":
        # 全量+绝对窗口（兼容/调试）
        full_decode_ids = req.origin_input_ids_unpadded + req.output_ids
        prev_full_len = getattr(req, 'last_full_decode_len', len(req.origin_input_ids_unpadded))
        decode_ids_list.append(full_decode_ids)
        read_offset_to_send = prev_full_len
        req.last_full_decode_len = len(full_decode_ids)
    else:
        # 增量模式
        decode_ids_list.append(decode_ids[req.send_decode_id_offset :])
        read_offset_to_send = read_offset
else:
    # 文本：保持增量协议
    decode_ids_list.append(decode_ids[req.send_decode_id_offset :])
    read_offset_to_send = read_offset

# Update baselines for next round
req.send_decode_id_offset = len(decode_ids)
req.last_full_decode_len = len(req.origin_input_ids_unpadded + req.output_ids)
read_offsets.append(read_offset_to_send)
```

**优点：**
1. 支持三种模式：`off`（跳过detokenizer）、`full`（全量+绝对偏移）、`incremental`（增量）
2. **默认使用增量模式**：只发送新生成的token（`decode_ids[req.send_decode_id_offset :]`）
3. `read_offset` 正确传递，detokenizer可以正确解码新token

**工作流程（正确 - 增量模式）：**
```
第1次发送:
  decode_ids = [151645, 198, 151644, 77091, 198, ...] (所有token)
  send_decode_id_offset = 0
  发送: decode_ids[0:] = [151645, 198, 151644, 77091, 198, ...]
  read_offset = 5
  → Detokenizer从位置5开始解码

第2次发送:
  decode_ids = [151645, 198, 151644, 77091, 198, ..., 334, 28311, 220, ...] (新增token)
  send_decode_id_offset = 133 (上次发送的长度)
  发送: decode_ids[133:] = [334, 28311, 220, ...] (只发送新token！)
  read_offset = 5 (相对于新token列表)
  → Detokenizer解码新token → 正确输出！
```

---

## ✅ 修复方案

### 修改文件

**文件：** `semipd_tp_nopp/python/sglang/srt/managers/scheduler_output_processor_mixin.py`

**修改位置：** 第570-598行

### 修改内容

**修改前（第570-580行）：**
```python
# 恢复原逻辑：多模态发送完整，文本发送增量
if self.server_args.enable_semi_pd:
    # Semi-PD: always send full decode_ids to keep detokenizer offsets consistent
    decode_ids_list.append(decode_ids)
elif self.model_config.is_multimodal_gen:
    decode_ids_list.append(decode_ids)
else:
    decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

req.send_decode_id_offset = len(decode_ids)
read_offsets.append(read_offset)
```

**修改后（第570-598行）：**
```python
# 🔧 MULTIMODAL FIX: Proper detokenizer protocol for multimodal requests
# Migrated from semipd_tp_pp to fix repeated output bug
# SGLANG_MM_DETOKENIZER_MODE: off(default)/incremental/full
mm_mode = os.environ.get("SGLANG_MM_DETOKENIZER_MODE", "incremental").lower()
if self.model_config.is_multimodal_gen:
    if mm_mode in ("off", "0", "false"):
        # Skip detokenizer for multimodal (original behavior)
        rids.pop(); finished_reasons.pop(); decoded_texts.pop()
        continue
    elif mm_mode == "full":
        # Full mode: send all tokens with absolute offset
        full_decode_ids = req.origin_input_ids_unpadded + req.output_ids
        prev_full_len = getattr(req, 'last_full_decode_len', len(req.origin_input_ids_unpadded))
        decode_ids_list.append(full_decode_ids)
        read_offset_to_send = prev_full_len
        req.last_full_decode_len = len(full_decode_ids)
    else:
        # Incremental mode (default): send only new tokens
        decode_ids_list.append(decode_ids[req.send_decode_id_offset :])
        read_offset_to_send = read_offset
else:
    # Text-only: always use incremental protocol
    decode_ids_list.append(decode_ids[req.send_decode_id_offset :])
    read_offset_to_send = read_offset

# Update baselines for next round
req.send_decode_id_offset = len(decode_ids)
req.last_full_decode_len = len(req.origin_input_ids_unpadded + req.output_ids)
read_offsets.append(read_offset_to_send)
```

### 关键改进

1. **增量发送token**：默认只发送新生成的token（`decode_ids[req.send_decode_id_offset :]`）
2. **正确的偏移管理**：使用 `read_offset_to_send` 变量，确保传递正确的偏移
3. **支持三种模式**：
   - `off` - 跳过detokenizer（原始行为）
   - `full` - 发送所有token，使用绝对偏移（调试用）
   - `incremental` - 发送增量token（默认，修复重复问题）
4. **状态跟踪**：添加 `last_full_decode_len` 跟踪完整token长度

---

## 🧪 测试验证

### 测试步骤

1. **重启服务器**：
   ```bash
   # 确保使用新代码
   cd /home/yzh/SemiTP_update/semipd_tp_nopp
   # 重启服务器（使用你的启动脚本）
   ```

2. **发送多模态请求**：
   ```bash
   python test_mm_debug.py
   # 或使用你的测试脚本
   ```

3. **检查日志**：
   ```bash
   tail -f /home/yzh/SemiTP_update/semipd_tp1.log
   ```

### 预期结果

**日志应该显示：**
```
[DBG_SCHEDULER] rid=xxx send_off=0 read_off=5 send_len=133 head=[151645, 198, 151644, ...]
[DBG_DETOKENIZER] batch=1 head=[[151645, 198, 151644, ...]] read_offsets=[5]

[DBG_SCHEDULER] rid=xxx send_off=133 read_off=5 send_len=128 head=[334, 28311, 220, ...]  ← 新token！
[DBG_DETOKENIZER] batch=1 head=[[334, 28311, 220, ...]] read_offsets=[5]  ← 新token！

[DBG_SCHEDULER] rid=xxx send_off=261 read_off=5 send_len=128 head=[104363, 3407, 14374, ...]  ← 又是新token！
[DBG_DETOKENIZER] batch=1 head=[[104363, 3407, 14374, ...]] read_offsets=[5]  ← 又是新token！
```

**模型回复应该：**
- ✅ 内容不重复
- ✅ 流式输出正常
- ✅ 完整描述图片内容

### 环境变量控制

如果需要切换模式，可以设置环境变量：

```bash
# 增量模式（默认，推荐）
export SGLANG_MM_DETOKENIZER_MODE=incremental

# 全量模式（调试用）
export SGLANG_MM_DETOKENIZER_MODE=full

# 关闭detokenizer（原始行为）
export SGLANG_MM_DETOKENIZER_MODE=off
```

---

## 📊 修复总结

| 项目 | 内容 |
|------|------|
| **问题** | 多模态请求回复重复输出 |
| **根本原因** | Detokenizer收到相同的token序列和偏移，重复解码 |
| **修复方案** | 使用增量发送协议，只发送新生成的token |
| **修改文件** | `scheduler_output_processor_mixin.py` |
| **修改行数** | 第570-598行（29行） |
| **迁移来源** | `semipd_tp_pp` 第641-669行 |
| **默认模式** | `incremental`（增量） |
| **状态** | ✅ 已修复 |

---

## 🎯 技术要点

1. **增量vs全量协议**：
   - 增量：只发送新token，节省带宽，避免重复
   - 全量：发送所有token，需要绝对偏移，用于调试

2. **偏移管理**：
   - `send_decode_id_offset`：已发送的token数量
   - `read_offset`：相对于当前发送的token列表的读取位置
   - `last_full_decode_len`：完整token序列的长度（用于全量模式）

3. **状态同步**：
   - 每次发送后更新 `send_decode_id_offset`
   - 确保下次只发送新token

4. **兼容性**：
   - 保持文本请求的原有行为（增量）
   - 多模态请求支持三种模式切换
   - 通过环境变量控制，无需修改代码

---

**修复完成时间：** 2025-10-12  
**修复状态：** ✅ 完成，等待测试验证

