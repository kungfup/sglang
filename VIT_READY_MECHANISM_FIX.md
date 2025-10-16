# VIT Ready 机制修复方案

## 🎯 问题总结

根据用户的分析和日志，当前 `semipd_pp` 缺少 VIT ready 机制，导致：

1. **PP + Chunked Prefill 下的 token 重复**：PP0 在 VIT embedding 未准备好前就将 VLM 请求加入 chunk 并转发到 PP1
2. **间歇性错误**：取决于 VIT 计算是否恰好在下一轮 microbatch 组装前完成
3. **"A" 插入问题**：Pipeline stage 在多模态状态尚未准备好时就消费了请求

## 🔍 根本原因

### 差异 1: PP 转发时不过滤 vit_pending 请求

**sglang_vit** (`scheduler.py:921-976`):
```python
# 只转发非 VIT pending 的请求
reqs_to_forward = []

# 1. 转发新收到的非 VIT pending 请求
if hasattr(self, '_last_recv_reqs'):
    for req in self._last_recv_reqs:
        if not (hasattr(req, 'vit_pending') and req.vit_pending):
            reqs_to_forward.append(req)

# 2. 转发 VIT 完成的请求
if hasattr(self, '_vit_ready_reqs') and self._vit_ready_reqs:
    for req in self._vit_ready_reqs:
        # 将 VIT embedding 复制到原始请求
        # 设置 vit_pending=False
        reqs_to_forward.append(original_req)
    self._vit_ready_reqs = []

# 转发请求给 PP1
point_to_point_pyobj(reqs_to_forward, ...)
```

**semipd_pp** (`scheduler.py:1001-1105`):
```python
# ❌ 全量转发 recv_reqs，不检查 vit_pending
point_to_point_pyobj(recv_reqs, ...)
```

### 差异 2: 缺少 vit_pending 标记

**sglang_vit** (`scheduler.py:1074-1080`):
```python
# 确保所有接收到的请求都有 vit_pending 属性
if recv_reqs is not None:
    for recv_req in recv_reqs:
        if hasattr(recv_req, 'rid') and not hasattr(recv_req, 'vit_pending'):
            recv_req.vit_pending = False
```

**sglang_vit** (`scheduler.py:1142-1230`):
```python
# 在 handle_generate_request 中标记 vit_pending
if hasattr(recv_req, 'vit_pending'):
    req.vit_pending = recv_req.vit_pending

# 保存原始请求，用于 VIT 完成后转发
if self.pp_group.is_first_rank:
    req._original_recv_req = recv_req
```

**semipd_pp**:
- ❌ 完全缺失 `vit_pending` 标记

### 差异 3: 缺少 VIT 结果轮询和缓存

**sglang_vit** (`scheduler.py:1325-1456`):
```python
def poll_vit_results(self):
    # 获取所有已完成的 VIT 计算结果
    results = self.vit_client.poll_results()
    
    # 更新对应请求的 precomputed_features
    for request_id, vit_result in results.items():
        for req in self.waiting_queue:
            if req.rid == request_id and req.vit_pending:
                # 更新 embedding
                mm_item.precomputed_features = vit_result.embedding.to(self.device)
                
                # 标记 VIT 计算完成
                req.vit_pending = False
                
                # 添加到待转发列表
                self._vit_ready_reqs.append(req)
```

**semipd_pp**:
- ❌ 完全缺失 `poll_vit_results()` 方法
- ❌ 缺失 `_vit_ready_reqs` 列表

### 差异 4: get_new_batch_prefill 不跳过 vit_pending

**sglang_vit** (`scheduler.py:1779-1806`):
```python
# 分类请求，优先处理 VIT 已完成的请求
vit_ready_reqs = []
vit_pending_reqs = []

for req in self.waiting_queue:
    if self.pp_group.is_first_rank:
        if hasattr(req, 'vit_pending') and req.vit_pending:
            vit_pending_reqs.append(req)
            continue
    vit_ready_reqs.append(req)

# 只处理 VIT 已完成的请求
for req in vit_ready_reqs:
    ...
```

**semipd_pp** (`scheduler.py:1797-1839`):
```python
# ❌ 遍历整个等待队列，不检查 vit_pending
for req in self.waiting_queue:
    ...
```

## ✅ 修复方案

### 修复 1: 初始化 VIT ready 相关变量

**文件**: `semipd_pp/python/sglang/srt/managers/scheduler.py`

**位置**: `__init__` 方法中

**添加**:
```python
# VIT ready mechanism for PP mode
self._last_recv_reqs = []
self._vit_ready_reqs = []
self._poll_count = 0
self._skip_vit_log_count = 0
```

### 修复 2: 在 recv_requests 中添加 vit_pending 标记

**文件**: `semipd_pp/python/sglang/srt/managers/scheduler.py`

**位置**: `recv_requests()` 方法末尾

**添加**:
```python
# 🔧 修复：确保所有接收到的请求都有 vit_pending 属性
if recv_reqs is not None:
    for recv_req in recv_reqs:
        if hasattr(recv_req, 'rid') and not hasattr(recv_req, 'vit_pending'):
            recv_req.vit_pending = False

# 保存最后接收的请求，用于 PP 转发
self._last_recv_reqs = recv_reqs if recv_reqs is not None else []

return recv_reqs
```

### 修复 3: 在 handle_generate_request 中标记 vit_pending

**文件**: `semipd_pp/python/sglang/srt/managers/scheduler.py`

**位置**: `handle_generate_request()` 方法中创建 `Req` 对象后

**添加**:
```python
# 🔧 修复：在 PP 模式下，从 TokenizedGenerateReqInput 中读取 vit_pending
if hasattr(recv_req, 'vit_pending'):
    req.vit_pending = recv_req.vit_pending

# 🔧 VIT 流水线并行：保存原始请求，用于 VIT 完成后转发给 PP1
if self.pp_group.is_first_rank:
    req._original_recv_req = recv_req
```

### 修复 4: 实现 poll_vit_results 方法

**文件**: `semipd_pp/python/sglang/srt/managers/scheduler.py`

**位置**: 在类中添加新方法

**添加**:
```python
def poll_vit_results(self):
    """Poll VIT computation results and update requests."""
    if not hasattr(self, 'vit_client') or self.vit_client is None:
        return
    
    self._poll_count = getattr(self, '_poll_count', 0) + 1
    
    # 获取所有已完成的 VIT 计算结果
    results = self.vit_client.poll_results()
    
    if not results:
        return
    
    # 更新对应请求的 precomputed_features
    for request_id, vit_result in results.items():
        for req in self.waiting_queue:
            if req.rid == request_id and hasattr(req, 'vit_pending') and req.vit_pending:
                # 更新 embedding
                if req.multimodal_inputs is not None:
                    for mm_item in req.multimodal_inputs.mm_items:
                        if mm_item.modality == Modality.IMAGE:
                            mm_item.precomputed_features = vit_result.embedding.to(self.device)
                
                # 标记 VIT 计算完成
                req.vit_pending = False
                
                # 添加到待转发列表
                if hasattr(self, '_vit_ready_reqs'):
                    self._vit_ready_reqs.append(req)
                break
```

### 修复 5: 修改 event_loop_pp 中的 PP 转发逻辑

**文件**: `semipd_pp/python/sglang/srt/managers/scheduler.py`

**位置**: `event_loop_pp()` 中的 PP 转发部分

**修改**:
```python
# 🔧 VIT 流水线并行：PP0 转发请求给 PP1
# 只转发非 VIT pending 的请求（VIT 完成后才转发）
if self.attn_tp_rank == 0:
    from sglang.srt.utils import point_to_point_pyobj
    reqs_to_forward = []
    
    # 1. 转发新收到的非 VIT pending 请求
    if hasattr(self, '_last_recv_reqs'):
        for req in self._last_recv_reqs:
            if not (hasattr(req, 'vit_pending') and req.vit_pending):
                reqs_to_forward.append(req)
    
    # 2. 转发 VIT 完成的请求
    if hasattr(self, '_vit_ready_reqs') and self._vit_ready_reqs:
        for req in self._vit_ready_reqs:
            if hasattr(req, '_original_recv_req'):
                original_req = req._original_recv_req
                
                # 将 VIT embedding 复制到原始请求
                if req.multimodal_inputs is not None and original_req.mm_inputs is not None:
                    if isinstance(original_req.mm_inputs, dict):
                        original_req.mm_inputs["mm_items"] = req.multimodal_inputs.mm_items
                
                # 设置 vit_pending=False
                original_req.vit_pending = False
                
                reqs_to_forward.append(original_req)
        # 清空已转发的 VIT 请求
        self._vit_ready_reqs = []
    
    # 转发请求给 PP1
    point_to_point_pyobj(reqs_to_forward, ...)
```

### 修复 6: 修改 get_new_batch_prefill 跳过 vit_pending

**文件**: `semipd_pp/python/sglang/srt/managers/scheduler.py`

**位置**: `get_new_batch_prefill()` 方法中

**修改**:
```python
# 🔧 流水线并行：分类请求，优先处理 VIT 已完成的请求
vit_ready_reqs = []
vit_pending_reqs = []

for req in self.waiting_queue:
    # 只在 PP0 上检查 vit_pending
    if self.pp_group.is_first_rank:
        if hasattr(req, 'vit_pending') and req.vit_pending:
            vit_pending_reqs.append(req)
            continue
    vit_ready_reqs.append(req)

# Get requests from the waiting queue to a new prefill batch
# 优先处理 VIT 已完成的请求
for req in vit_ready_reqs:
    ...
```

## 🎯 预期效果

修复后：
- ✅ PP0 只转发 VIT embedding 已准备好的请求给 PP1
- ✅ Chunked prefill 不会启动半成品 VLM 请求
- ✅ 消除 token 重复和 "A" 插入问题
- ✅ 消除间歇性错误

## 📝 修改文件列表

1. **`semipd_pp/python/sglang/srt/managers/scheduler.py`**
   - ✅ `recv_requests` (行 1200-1221): 添加 vit_pending 标记和保存 _last_recv_reqs
   - ✅ `handle_generate_request` (行 1281-1289): 标记 vit_pending 并保存原始请求
   - ✅ `poll_vit_results` (行 1427-1491): 新增方法
   - ✅ `event_loop_pp` (行 1092-1143): 修改 PP 转发逻辑，过滤 vit_pending 请求
   - ✅ `get_new_batch_prefill` (行 1900-1933): 跳过 vit_pending 请求

2. **`semipd_pp/python/sglang/srt/managers/schedule_batch.py`**
   - ✅ `Req.init_next_round_input` (行 664-705): 添加 `enable_hierarchical_cache` 参数

3. **`semipd_pp/python/sglang/srt/managers/schedule_policy.py`**
   - ✅ `PrefillAdder.add_one_req` (行 492-508): 添加 `enable_hierarchical_cache` 参数

## ✅ 修复完成状态

所有修复已完成！以下是修复的详细内容：

### 1. API 兼容性修复（已完成）

- ✅ `Req.init_next_round_input()`: 添加 `enable_hierarchical_cache` 参数
- ✅ `PrefillAdder.add_one_req()`: 添加 `enable_hierarchical_cache` 参数
- ✅ `calc_priority()` 返回值: 使用 `prefix_computed` 变量
- ✅ VIT 轮询: 在 `event_loop_pp()` 中调用 `poll_vit_results()`

### 2. VIT Ready 机制修复（已完成）

- ✅ `recv_requests()`: 为所有请求添加 `vit_pending` 标记
- ✅ `handle_generate_request()`: 继承 `vit_pending` 标记并保存原始请求
- ✅ `poll_vit_results()`: 新增方法，轮询 VIT 结果并更新请求
- ✅ `event_loop_pp()`: PP 转发时过滤 `vit_pending` 请求
- ✅ `get_new_batch_prefill()`: 跳过 `vit_pending` 请求

## 🧪 测试步骤

1. **重启服务器**:
   ```bash
   pkill -f "sglang.launch_server"

   python -m sglang.launch_server \
       --model-path Qwen/Qwen2.5-VL-32B-Instruct \
       --tp-size 2 \
       --pp-size 2 \
       --chunked-prefill-size 1024 \
       --port 30019
   ```

2. **测试并发请求**:
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

3. **检查输出**:
   ```bash
   grep "DBG_DETOKENIZER_FINAL" sglang_pp.log | tail -10
   ```

   **预期结果**：所有响应都应该正常，无 token 重复或 "A" 插入。

## 🎯 预期效果

修复后：
- ✅ PP0 只转发 VIT embedding 已准备好的请求给 PP1
- ✅ Chunked prefill 不会启动半成品 VLM 请求
- ✅ 消除 token 重复和 "A" 插入问题
- ✅ 消除间歇性错误
- ✅ 原生 SGLang + PP + Chunked Prefill：输出正常
- ✅ Semi-PD + PP + Chunked Prefill：输出正常
- ✅ 单请求和并发请求：都正常
- ✅ VLM 请求：正常

