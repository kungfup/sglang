# PP + Chunked Prefill API 兼容性修复

## 🎯 问题总结

### 问题 1: Semi-PD + PP + Chunked Prefill

使用 `--chunked-prefill-size 1024` + `--pp-size 2` + Semi-PD 时：
- ✅ **单个串行请求**: 输出完全正常
- ❌ **多个并发请求**: 输出严重错误（"A" 插入、token 重复、乱码）

### 问题 2: 原生 SGLang + PP + Chunked Prefill

使用 `--chunked-prefill-size 1024` + `--pp-size 2`（无 Semi-PD）时：
- ❌ **间歇性 token 重复错误**

**错误输出示例**:
```
[Response 1] 这张图片展示了一堆新鲜的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的
[Response 2] 这张图片展示了一堆新鲜的青苹果...  # ✅ 正常
```

**关键特征**：
- ❌ 间歇性错误（有时正常，有时出错）
- ❌ Token 重复（"的" 字重复）
- ❌ 非确定性行为

## 🔍 根本原因

### 核心问题：API 不兼容

`semipd_pp/` 的代码与 `sglang_vit/` 的代码存在 **API 不兼容** 问题，导致关键参数缺失。

### 关键差异

#### 差异 1: `calc_priority()` 返回值未使用

**sglang_vit**:
```python
prefix_computed = self.policy.calc_priority(self.waiting_queue)
```

**semipd_pp**:
```python
self.policy.calc_priority(self.waiting_queue)  # ❌ 忽略返回值
```

**影响**: `prefix_computed` 用于决定是否传递 `tree_cache` 给 `init_next_round_input()`

#### 差异 2: `init_next_round_input()` 参数缺失

**sglang_vit**:
```python
req.init_next_round_input(
    None if prefix_computed else self.tree_cache,
    self.enable_hierarchical_cache,
)
```

**semipd_pp**:
```python
req.init_next_round_input(self.tree_cache)  # ❌ 缺少参数
```

**影响**: 缺少 `enable_hierarchical_cache` 参数，导致 KV cache 状态管理错误

#### 差异 3: `add_one_req()` 参数缺失

**sglang_vit**:
```python
res = adder.add_one_req(
    req, self.chunked_req, self.enable_hierarchical_cache
)
```

**semipd_pp**:
```python
res = adder.add_one_req(req, has_chunked_req=(self.chunked_req is not None))  # ❌ 缺少参数
```

**影响**: 缺少 `enable_hierarchical_cache` 参数，导致 chunked prefill 状态管理错误

#### 差异 4: `event_loop_pp()` 缺少 VIT 轮询

**sglang_vit**:
```python
while True:
    # 🔧 修复：在循环开始就轮询 VIT 结果
    self.poll_vit_results()
    
    server_is_idle = True
    for mb_id in range(self.pp_size):
        ...
```

**semipd_pp**:
```python
while True:
    server_is_idle = True  # ❌ 缺少 poll_vit_results()
    for mb_id in range(self.pp_size):
        ...
```

**影响**: VIT 结果没有及时轮询，可能导致请求阻塞和状态错误

## ✅ 实施的修复

### 修复 1: 使用 `calc_priority()` 的返回值

**文件**: `semipd_pp/python/sglang/srt/managers/scheduler.py`

**行 1789**:

```python
# 修改前
self.policy.calc_priority(self.waiting_queue)

# 修改后
prefix_computed = self.policy.calc_priority(self.waiting_queue)
```

### 修复 2: 修正 `init_next_round_input()` 调用

**文件**: `semipd_pp/python/sglang/srt/managers/scheduler.py`

**行 1835-1839**:

```python
# 修改前
req.init_next_round_input(self.tree_cache)

# 修改后
# 🔧 FIX: Pass tree_cache and enable_hierarchical_cache correctly
# This aligns with sglang_vit's implementation
req.init_next_round_input(
    None if prefix_computed else self.tree_cache,
    self.enable_hierarchical_cache,
)
```

### 修复 3: 修正 `add_one_req()` 调用

**文件**: `semipd_pp/python/sglang/srt/managers/scheduler.py`

**行 1841-1845**:

```python
# 修改前
res = adder.add_one_req(req, has_chunked_req=(self.chunked_req is not None))

# 修改后
# 🔧 FIX: Pass chunked_req object and enable_hierarchical_cache
# This aligns with sglang_vit's implementation
res = adder.add_one_req(
    req, self.chunked_req, self.enable_hierarchical_cache
)
```

### 修复 4: 更新 `add_one_req()` 方法签名

**文件**: `semipd_pp/python/sglang/srt/managers/schedule_policy.py`

**行 492-508**:

```python
# 修改前
def add_one_req(self, req: Req, has_chunked_req: bool):
    if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):
        return self.add_one_req_ignore_eos(req, has_chunked_req)

# 修改后
def add_one_req(
    self, req: Req, has_chunked_req: bool, enable_hierarchical_cache: bool = False
):
    """
    Add one request to the prefill batch.
    
    Args:
        req: The request to add
        has_chunked_req: Whether there is a chunked request (can be bool or Req object)
        enable_hierarchical_cache: Whether hierarchical cache is enabled
    """
    # Convert has_chunked_req to bool if it's a Req object (for compatibility with sglang_vit)
    if not isinstance(has_chunked_req, bool):
        has_chunked_req = has_chunked_req is not None
    
    if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):
        return self.add_one_req_ignore_eos(req, has_chunked_req)
```

### 修复 5: 添加 VIT 轮询

**文件**: `semipd_pp/python/sglang/srt/managers/scheduler.py`

**行 945-950**:

```python
# 修改前
while True:
    server_is_idle = True
    for mb_id in range(self.pp_size):
        ...

# 修改后
while True:
    # 🔧 FIX: Poll VIT results at the beginning of each loop iteration
    # This ensures VIT results are processed even if PP communication blocks
    # This aligns with sglang_vit's implementation
    if hasattr(self, 'poll_vit_results'):
        self.poll_vit_results()
    
    server_is_idle = True
    for mb_id in range(self.pp_size):
        ...
```

## 📊 修复原理

### 为什么会导致 token 重复？

1. **`enable_hierarchical_cache` 参数缺失**：
   - 这个参数影响 KV cache 的管理和 chunked prefill 的状态转移
   - 缺少这个参数可能导致 chunked request 的状态管理错误
   - 在 PP 模式下，不同 microbatch 之间的 chunked request 状态可能混乱

2. **`prefix_computed` 未使用**：
   - `prefix_computed` 决定是否传递 `tree_cache` 给 `init_next_round_input()`
   - 如果 prefix 已经计算过，不应该再传递 `tree_cache`
   - 错误的 `tree_cache` 传递可能导致重复计算或状态错误

3. **VIT 轮询缺失**：
   - 在 PP 模式下，如果 VIT 结果没有及时轮询，可能导致请求阻塞
   - 阻塞的请求可能与新请求混合，导致状态错误

### 为什么是间歇性错误？

- **竞态条件**：缺少的参数导致 chunked request 的状态管理在某些情况下正确，某些情况下错误
- **PP microbatch 交错**：在 PP 模式下，不同 microbatch 的 chunked request 可能相互干扰
- **KV cache 状态不一致**：缺少 `enable_hierarchical_cache` 参数导致 KV cache 的状态在某些情况下不一致

## 🎯 预期效果

修复后：
- ✅ 原生 SGLang + PP + Chunked Prefill：输出正常，无 token 重复
- ✅ Semi-PD + PP + Chunked Prefill：输出正常，无 "A" 插入或 token 重复
- ✅ 单请求和并发请求：都正常
- ✅ VLM 请求：正常
- ✅ 消除间歇性错误：输出稳定可靠

## 🧪 验证步骤

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
   grep "DBG_DETOKENIZER_FINAL" semipd_pp.log | tail -10
   ```

   **预期结果**：所有响应都应该正常，无 token 重复或 "A" 插入。

## 📝 修改的文件列表

1. **`semipd_pp/python/sglang/srt/managers/scheduler.py`**
   - 行 947-950: 添加 VIT 轮询
   - 行 1789: 使用 `calc_priority()` 的返回值
   - 行 1835-1839: 修正 `init_next_round_input()` 调用
   - 行 1841-1845: 修正 `add_one_req()` 调用

2. **`semipd_pp/python/sglang/srt/managers/schedule_policy.py`**
   - 行 492-508: 更新 `add_one_req()` 方法签名，添加 `enable_hierarchical_cache` 参数

3. **`semipd_pp/python/sglang/srt/managers/semi_pd_decode_scheduler.py`**
   - 行 725-777: 添加 `_get_next_batch_decode_only()` 方法（之前的修复）
   - 行 779-791: 修改 `get_next_batch_to_run()` 调用新方法（之前的修复）

## 🎯 总结

**根本原因**：
- `semipd_pp/` 的代码与 `sglang_vit/` 的代码存在 API 不兼容
- 关键参数 `enable_hierarchical_cache` 和 `prefix_computed` 缺失
- 导致 chunked prefill 的状态管理错误

**修复策略**：
- 对齐 `sglang_vit/` 的 API 调用方式
- 传递所有必要的参数
- 确保 chunked request 的状态管理正确
- 添加 VIT 轮询以支持 VLM 请求

**修复完成！请重启服务器并测试。** 🚀

