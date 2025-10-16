# 并发请求调试分析报告

## 📊 当前状态

### 1. 代码修改状态

根据代码检查，`semipd_pp/python/sglang/srt/managers/scheduler.py` 已经包含以下修复：

#### ✅ 已实现的修复

1. **`_forwarded_req_ids` 跟踪机制** (行 622, 1152-1156, 1178-1186)
   - 使用 `set` 跟踪已转发的请求 ID
   - 防止重复转发同一个请求

2. **`recv_requests()` 方法** (行 1341-1379)
   - 为所有请求添加 `vit_pending` 属性（默认 `False`）
   - 保存 `_last_recv_reqs` 用于 PP 转发
   - 添加了调试日志（如果 `DEBUG_LOGS_ENABLED`）

3. **`event_loop_pp()` 中的 PP 转发逻辑** (行 1145-1230)
   - 过滤已转发的请求（使用 `_forwarded_req_ids`）
   - 过滤 `vit_pending` 请求
   - 转发 VIT-ready 请求
   - 添加了详细的调试日志

4. **`handle_generate_request()` 方法** (行 1281-1289)
   - 继承 `vit_pending` 标记
   - 保存原始请求 `_original_recv_req`

5. **`poll_vit_results()` 方法** (行 1427-1491)
   - 轮询 VIT 结果
   - 更新请求的 embedding
   - 标记 `vit_pending = False`
   - 添加到 `_vit_ready_reqs` 列表

6. **`get_new_batch_prefill()` 方法** (行 1900-1933)
   - 分类请求为 `vit_ready_reqs` 和 `vit_pending_reqs`
   - 只处理 VIT-ready 请求

### 2. VIT 模式分析

根据日志：
```
[Scheduler] VIT Scheduler disabled, using synchronous ViT computation
```

**当前使用的是同步 VIT 模式**，这意味着：

- ❌ **VIT client 不存在** (`self.vit_client = None`)
- ❌ **`poll_vit_results()` 不会被调用**（因为 VIT client 不存在）
- ❌ **`vit_pending` 机制不会生效**（因为没有异步 VIT 任务）
- ✅ **VIT 计算在 `embed_multimodal()` 中同步完成**

### 3. 问题分析

#### 问题 1: VIT Ready 机制在同步模式下无效

在**同步 VIT 模式**下：
- VIT 计算在 `embed_multimodal()` 中同步完成
- 不需要 `vit_pending` 标记
- 不需要 `poll_vit_results()` 轮询
- **请求在 VIT 计算完成后立即可用**

因此，我们添加的 VIT ready 机制（`poll_vit_results`, `vit_pending` 标记）在当前模式下**不会生效**。

#### 问题 2: 并发请求的真正问题

用户报告：
- ✅ **串行请求（单个请求）**：工作正常
- ❌ **并行请求（多个并发请求）**：仍然出现错误

**可能的原因**：

1. **请求转发时机问题**：
   - 在并发请求时，多个请求可能在同一个 `event_loop_pp()` 循环中被接收
   - 这些请求被添加到 `_last_recv_reqs` 列表
   - 在下一次循环中，所有请求被一次性转发到 PP1
   - **如果某些请求的 VIT 计算尚未完成，它们会被错误地转发**

2. **`_forwarded_req_ids` 清理问题**：
   - `_forwarded_req_ids` 是一个 `set`，会持续累积
   - **没有清理机制**，可能导致内存泄漏
   - 已完成的请求 ID 应该被移除

3. **`_last_recv_reqs` 累积问题**：
   - `_last_recv_reqs` 在每次 `recv_requests()` 时被更新
   - 在并发请求时，可能包含多个请求
   - **如果某些请求的 VIT 计算较慢，它们会在下一次循环中被转发**

## 🔍 需要的信息

为了进一步诊断问题，我需要以下信息：

### 1. 具体的错误现象

请提供：
- 并发请求时的**具体错误输出**（例如：token 重复、"A" 插入、乱码等）
- 错误是**每次都出现**还是**间歇性出现**
- 错误出现在**哪个阶段**（prefill、decode、detokenize）

### 2. 日志分析

请运行以下命令并提供输出：

```bash
# 1. 查看最近的错误输出
tail -5000 /home/yzh/semipd_pp_vit/sglang_pp.log | grep -E "这张图片|Response|的的的|AAAA" | tail -30

# 2. 查看 PP 转发日志
tail -5000 /home/yzh/semipd_pp_vit/sglang_pp.log | grep "Forwarding.*TokenizedGenerateReqInput" | tail -30

# 3. 查看 VIT 计算日志
tail -5000 /home/yzh/semipd_pp_vit/sglang_pp.log | grep "MM_EMBED_DO_VIT\|VLM_VIT_FORWARD" | tail -30

# 4. 查看请求接收日志（如果启用了 DEBUG_LOGS_ENABLED）
tail -5000 /home/yzh/semipd_pp_vit/sglang_pp.log | grep "recv_requests" | tail -30
```

### 3. 测试场景

请提供：
- 并发请求的数量（例如：5 个并发请求）
- 请求是否包含图片
- 请求的 `max_tokens` 设置
- 是否使用了 `--chunked-prefill-size 1024`

## 🛠️ 可能的修复方案

### 方案 1: 在同步 VIT 模式下禁用 VIT ready 机制

如果 VIT client 不存在，直接转发所有请求（因为 VIT 计算已经同步完成）：

```python
# In event_loop_pp()
if self.attn_tp_rank == 0:
    reqs_to_forward = []
    
    # If VIT client doesn't exist, forward all requests immediately
    if not hasattr(self, 'vit_client') or self.vit_client is None:
        # Synchronous VIT mode: VIT computation is already done
        if hasattr(self, '_last_recv_reqs'):
            for req in self._last_recv_reqs:
                if hasattr(req, 'rid') and req.rid not in self._forwarded_req_ids:
                    self._forwarded_req_ids.add(req.rid)
                    reqs_to_forward.append(req)
    else:
        # Asynchronous VIT mode: filter vit_pending requests
        # ... (existing logic)
```

### 方案 2: 清理 `_forwarded_req_ids`

定期清理已完成的请求 ID：

```python
# In process_batch_result() or stream_output()
def _cleanup_forwarded_req_ids(self, finished_req_ids: List[str]):
    """Remove finished request IDs from the forwarded set."""
    for rid in finished_req_ids:
        self._forwarded_req_ids.discard(rid)
```

### 方案 3: 启用异步 VIT 调度器

设置环境变量启用异步 VIT 调度器：

```bash
export SGLANG_VIT_SCHEDULER_ENABLED=1
export SGLANG_VIT_SCHEDULER_HOST=localhost
export SGLANG_VIT_SCHEDULER_PORT=5555
```

然后启动 VIT 调度器服务（如果存在）。

### 方案 4: 添加更详细的调试日志

启用调试日志以查看详细的请求流程：

```bash
export SGLANG_ENABLE_DEBUG_LOGS=1
```

然后重启服务器并重新测试。

## ✅ 已实施的修复

### 修复 1: 对齐 `_last_recv_reqs` 的更新逻辑

**问题**：`semipd_pp` 只保存 `TokenizedGenerateReqInput`，而 `sglang_vit` 保存所有请求

**修复**：
```python
# Before (semipd_pp)
self._last_recv_reqs = [
    req for req in (recv_reqs or [])
    if isinstance(req, TokenizedGenerateReqInput)
]

# After (align with sglang_vit)
self._last_recv_reqs = recv_reqs if recv_reqs is not None else []
```

**原因**：
- `sglang_vit` 直接保存所有请求，确保每批请求只被转发一次
- 过滤 `TokenizedGenerateReqInput` 可能导致某些请求被遗漏

### 修复 2: 移除 `_forwarded_req_ids` 机制

**问题**：`semipd_pp` 使用 `_forwarded_req_ids` 跟踪已转发的请求，但 `sglang_vit` 没有这个机制

**修复**：
- 移除 `self._forwarded_req_ids: set[str] = set()` 初始化
- 移除 PP 转发逻辑中的 `_forwarded_req_ids` 检查和更新

**原因**：
- `sglang_vit` 依赖 `_last_recv_reqs` 的更新来确保每批请求只被转发一次
- `_forwarded_req_ids` 可能导致请求被错误地跳过
- `_forwarded_req_ids` 没有清理机制，会持续累积导致内存泄漏

### 修复 3: 简化 PP 转发逻辑

**Before**:
```python
# 1. Forward newly received non-VIT-pending requests
if hasattr(self, '_last_recv_reqs'):
    for req in self._last_recv_reqs:
        if hasattr(req, 'rid') and req.rid in self._forwarded_req_ids:
            continue
        if not (hasattr(req, 'vit_pending') and req.vit_pending):
            if hasattr(req, 'rid'):
                self._forwarded_req_ids.add(req.rid)
            reqs_to_forward.append(req)
```

**After**:
```python
# 1. Forward newly received non-VIT-pending requests
if hasattr(self, '_last_recv_reqs'):
    for req in self._last_recv_reqs:
        if not (hasattr(req, 'vit_pending') and req.vit_pending):
            reqs_to_forward.append(req)
```

**原因**：
- 对齐 `sglang_vit` 的简洁逻辑
- 移除不必要的 `_forwarded_req_ids` 检查

## 🧪 测试步骤

1. **重启服务器**:
   ```bash
   pkill -f "sglang.launch_server"

   python -m sglang.launch_server \
       --model-path Qwen/Qwen2.5-VL-32B-Instruct \
       --tp-size 2 \
       --pp-size 2 \
       --chunked-prefill-size 1024 \
       --port 30019 \
       > sglang_pp.log 2>&1 &
   ```

2. **测试串行请求**:
   ```bash
   curl -X POST http://localhost:30019/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen2.5-VL-32B-Instruct",
       "messages": [
         {
           "role": "user",
           "content": [
             {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
             {"type": "text", "text": "描述这张图片"}
           ]
         }
       ],
       "max_tokens": 200
     }'
   ```

3. **测试并发请求**:
   ```bash
   for i in {1..5}; do
     curl -X POST http://localhost:30019/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{
         "model": "Qwen/Qwen2.5-VL-32B-Instruct",
         "messages": [
           {
             "role": "user",
             "content": [
               {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
               {"type": "text", "text": "描述这张图片"}
             ]
           }
         ],
         "max_tokens": 200
       }' &
   done
   wait
   ```

4. **检查日志**:
   ```bash
   # 查看 PP 转发日志
   tail -500 sglang_pp.log | grep "Forwarding.*TokenizedGenerateReqInput"

   # 查看 VIT 计算日志
   tail -500 sglang_pp.log | grep "MM_EMBED_DO_VIT\|VLM_VIT_FORWARD"

   # 查看输出结果
   tail -500 sglang_pp.log | grep "这张图片\|Response"
   ```

## 🎯 预期效果

修复后：
- ✅ **串行请求**：正常工作
- ✅ **并发请求**：正常工作，无 token 重复或乱码
- ✅ **PP 转发**：每批请求只被转发一次
- ✅ **内存管理**：无内存泄漏（移除了 `_forwarded_req_ids`）

## 📝 下一步行动

1. **重启服务器并测试**：运行上述测试步骤
2. **提供测试结果**：如果仍有问题，提供详细的错误日志
3. **进一步调试**：如果需要，添加更多调试日志

## 🔗 相关文件

- `semipd_pp/python/sglang/srt/managers/scheduler.py` - 主调度器（已修复）
- `sglang_vit/python/sglang/srt/managers/scheduler.py` - 参考实现
- `semipd_pp/VIT_READY_MECHANISM_FIX.md` - VIT ready 机制修复文档
- `semipd_pp/DIFF_ANALYSIS_SGLANG_VIT_VS_SEMIPD_PP.md` - 差异分析报告
- `semipd_pp/CONCURRENT_REQUEST_DEBUG_ANALYSIS.md` - 本文档

