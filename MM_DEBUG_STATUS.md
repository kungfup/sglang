# 多模态功能调试状态

## 问题描述

系统接收到包含图像的多模态请求后，**没有执行ViT计算**，而是返回了通用回复：
```
当然，请提供您想要描述的图片，我会详细为您描述。
```

这表明模型**没有看到图像**，只看到了文本提示。

## 症状分析

从日志 `/home/yzh/SemiTP_update/semipd_tp.log` 可以看到：

1. **请求token数量异常**：
   ```
   [2025-10-11 19:14:34] New request 8f317bb8f3cb4e89ba9ea10b527cbedd, #tokens: 25
   ```
   - 只有25个tokens
   - 正常情况下，一张图像应该被扩展成数百甚至数千个tokens

2. **没有多模态嵌入日志**：
   - 日志中没有 `[MM_EMBED_CALL]` 或 `[MM_EMBED_DO_VIT]` 标记
   - 这说明 `_get_chunked_prefill_embedding` 函数根本没有被调用
   - ViT计算没有被触发

3. **多模态处理被跳过**：
   - 图像数据可能在请求处理过程中丢失
   - 或者 `mm_inputs` 字段为 `None`

## 可能的根本原因

### 假设1：API层面的问题
- OpenAI格式的API请求中的图像数据没有被正确解析
- `image_data` 字段没有被设置到 `GenerateReqInput` 对象中
- `contains_mm_input()` 返回 `False`

### 假设2：Tokenizer Manager的问题
- `mm_processor` 为 `None`（多模态处理器未初始化）
- `process_mm_data_async` 没有被调用
- `image_inputs` 被设置为 `None`

### 假设3：Scheduler的问题
- `recv_req.mm_inputs` 为 `None`
- 多模态处理代码被跳过

## 已添加的调试日志

### 1. Tokenizer Manager (`tokenizer_manager.py`)

在 `_tokenize_one_request` 方法中添加了调试日志（第512-531行）：

```python
logger.info(f"[MM_DEBUG_TOK] rid={obj.rid} has_mm_processor={self.mm_processor is not None} contains_mm={obj.contains_mm_input()}")
logger.info(f"[MM_DEBUG_TOK] rid={obj.rid} image_data type={type(obj.image_data)} is_none={obj.image_data is None}")
logger.info(f"[MM_DEBUG_TOK] rid={obj.rid} Processing multimodal data...")
logger.info(f"[MM_DEBUG_TOK] rid={obj.rid} image_inputs type={type(image_inputs)} is_none={image_inputs is None}")
```

**这些日志会显示：**
- 是否有多模态处理器
- 请求是否包含多模态输入
- `image_data` 的类型和值
- `image_inputs` 的处理结果

### 2. Semi-PD Scheduler (`semi_pd_scheduler.py`)

在 `handle_generate_request` 方法中添加了调试日志（第148-156行和217-233行）：

```python
# 请求接收时
logger.info(f"[MM_DEBUG] Request {recv_req.rid} has mm_inputs: {type(recv_req.mm_inputs)}")
logger.info(f"[MM_DEBUG] Request {recv_req.rid} has NO mm_inputs (mm_inputs is None)")

# 多模态处理时
logger.info(f"[MM_DEBUG] Processing mm_inputs for request {recv_req.rid}")
logger.info(f"[MM_DEBUG] origin_input_ids length before padding: {len(req.origin_input_ids)}")
logger.info(f"[MM_DEBUG] origin_input_ids length after padding: {len(req.origin_input_ids)}")
```

**这些日志会显示：**
- `mm_inputs` 是否为 `None`
- 图像token扩展前后的input_ids长度
- 多模态处理是否被执行

## 测试方法

### 方法1：使用调试脚本

```bash
cd /home/yzh/SemiTP_update/semipd_tp_nopp
python test_mm_debug.py
```

这个脚本会：
1. 创建一个简单的测试图像（100x100红色图片）
2. 发送多模态请求到服务器
3. 分析响应中的token数量
4. 提示查看服务器日志

### 方法2：查看服务器日志

```bash
# 实时查看日志
tail -f /home/yzh/SemiTP_update/semipd_tp.log | grep -E "MM_DEBUG|MM_EMBED"

# 或者在发送请求后查看
tail -200 /home/yzh/SemiTP_update/semipd_tp.log | grep -E "MM_DEBUG|MM_EMBED"
```

## 预期的日志输出

### 如果一切正常，应该看到：

```
[MM_DEBUG_TOK] rid=xxx has_mm_processor=True contains_mm=True
[MM_DEBUG_TOK] rid=xxx image_data type=<class 'list'> is_none=False
[MM_DEBUG_TOK] rid=xxx Processing multimodal data...
[MM_DEBUG_TOK] rid=xxx image_inputs type=<class 'dict'> is_none=False
[MM_DEBUG_TOK] rid=xxx Updated input_ids length=1234  # 应该是几百到几千

[MM_DEBUG] Request xxx has mm_inputs: <class 'dict'>
[MM_DEBUG] Processing mm_inputs for request xxx
[MM_DEBUG] origin_input_ids length before padding: 25
[MM_DEBUG] origin_input_ids length after padding: 1234  # 应该显著增加

[MM_EMBED_CALL] rid=xxx pid=12345 req_idx=0 num_items=1 hash=987654321
[MM_EMBED_DO_VIT] rid=xxx precomputed_items=0/1  # 第一次计算ViT
```

### 如果有问题，可能看到：

```
[MM_DEBUG_TOK] rid=xxx has_mm_processor=False contains_mm=False
# 或
[MM_DEBUG_TOK] rid=xxx has_mm_processor=True contains_mm=False
# 或
[MM_DEBUG] Request xxx has NO mm_inputs (mm_inputs is None)
```

## 下一步诊断步骤

### 步骤1：运行测试脚本

```bash
# 确保服务器正在运行
# 然后运行测试
python semipd_tp_nopp/test_mm_debug.py
```

### 步骤2：查看日志

```bash
# 查看最近的调试日志
tail -200 /home/yzh/SemiTP_update/semipd_tp.log | grep "MM_DEBUG"
```

### 步骤3：根据日志输出诊断

**场景A：没有 `[MM_DEBUG_TOK]` 日志**
- 说明请求没有到达tokenizer_manager
- 检查API层面的问题

**场景B：`has_mm_processor=False`**
- 多模态处理器未初始化
- 检查模型配置和初始化代码

**场景C：`contains_mm=False`**
- `image_data` 字段为空
- 检查API请求解析代码

**场景D：`mm_inputs is None`**
- tokenizer_manager没有生成 `image_inputs`
- 检查 `process_mm_data_async` 的执行

**场景E：有 `[MM_DEBUG]` 日志但token数量没增加**
- `pad_input_ids_func` 或 `extend_image_inputs` 有问题
- 检查这两个函数的实现

## 可能的修复方案

根据诊断结果，可能需要：

1. **修复API解析**：确保OpenAI格式的图像数据被正确解析到 `image_data` 字段
2. **初始化多模态处理器**：确保 `mm_processor` 被正确初始化
3. **修复token扩展**：确保 `pad_input_ids_func` 正确扩展图像tokens
4. **修复ViT调用**：确保ViT模型被正确调用

## 当前状态

- ✅ 调试日志已添加
- ✅ 测试脚本已创建
- ⏳ 等待运行测试并查看日志输出
- ⏳ 根据日志输出确定根本原因
- ⏳ 实施修复

---

**下一步：请运行测试脚本并提供日志输出，我将根据日志确定问题并实施修复。**

