# 差异分析报告：sglang_vit vs semipd_pp

## 🎯 问题总结

**现象**：
- 原生 SGLang + PP + Chunked Prefill 出现间歇性 token 重复错误
- Semi-PD + PP + Chunked Prefill 出现 "A" 插入和 token 重复错误

**测试环境**：
- `--pp-size 2`
- `--chunked-prefill-size 1024`
- `--tp-size 2`
- Model: `Qwen/Qwen2.5-VL-32B-Instruct`

**错误输出示例**：
```
[Response 1] 这张图片展示了一堆新鲜的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的
[Response 2] 这张图片展示了一堆新鲜的青苹果...  # ✅ 正常
```

**关键特征**：
- ❌ 间歇性错误（有时正常，有时出错）
- ❌ Token 重复（"的" 字重复）
- ❌ 非确定性行为

## 🔍 根本原因分析

### 核心问题：API 不兼容

`semipd_pp/` 的代码与 `sglang_vit/` 的代码存在 **API 不兼容** 问题，导致关键参数缺失。

### 差异 1: `init_next_round_input()` 调用参数不同

**sglang_vit** (`scheduler.py:1831-1834`):
```python
req.init_next_round_input(
    None if prefix_computed else self.tree_cache,
    self.enable_hierarchical_cache,
)
```

**semipd_pp** (`scheduler.py:1835`):
```python
req.init_next_round_input(self.tree_cache)
```

**问题**：
- `sglang_vit` 传递了 **2 个参数**：`tree_cache` 和 `enable_hierarchical_cache`
- `semipd_pp` 只传递了 **1 个参数**：`tree_cache`
- **缺少 `enable_hierarchical_cache` 参数！**

### 差异 2: `add_one_req()` 调用参数不同

**sglang_vit** (`scheduler.py:1836-1838`):
```python
res = adder.add_one_req(
    req, self.chunked_req, self.enable_hierarchical_cache
)
```

**semipd_pp** (`scheduler.py:1836`):
```python
res = adder.add_one_req(req, has_chunked_req=(self.chunked_req is not None))
```

**问题**：
- `sglang_vit` 传递了 **3 个参数**：`req`, `self.chunked_req`, `self.enable_hierarchical_cache`
- `semipd_pp` 只传递了 **2 个参数**：`req`, `has_chunked_req`
- **缺少 `enable_hierarchical_cache` 参数！**

### 差异 3: `PrefillAdder` 构造函数参数不同

**sglang_vit** (`schedule_policy.py:269-278`):
```python
def __init__(
    self,
    tree_cache: BasePrefixCache,
    token_to_kv_pool_allocator: TokenToKVPoolAllocator,
    running_batch: ScheduleBatch,
    new_token_ratio: float,
    rem_input_tokens: int,
    rem_chunk_tokens: Optional[int],
    mixed_with_decode_tokens: int = 0,
):
```

**semipd_pp** (`schedule_policy.py:271-281`):
```python
def __init__(
    self,
    page_size: int,  # ← 额外的参数
    tree_cache: BasePrefixCache,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    running_batch: ScheduleBatch,
    new_token_ratio: float,
    rem_input_tokens: int,
    rem_chunk_tokens: Optional[int],
    mixed_with_decode_tokens: int = 0,
):
```

**问题**：
- `semipd_pp` 多了一个 `page_size` 参数
- 这导致调用时参数位置错位

### 差异 4: `PrefillAdder` 实例化参数不同

**sglang_vit** (`scheduler.py:1762-1770`):
```python
adder = PrefillAdder(
    self.tree_cache,
    self.token_to_kv_pool_allocator,
    self.running_batch,
    self.new_token_ratio,
    self.max_prefill_tokens,
    self.chunked_prefill_size,
    running_bs if self.is_mixed_chunk else 0,
)
```

**semipd_pp** (`scheduler.py:1792-1801`):
```python
adder = PrefillAdder(
    self.page_size,  # ← 额外的参数
    self.tree_cache,
    self.token_to_kv_pool_allocator,
    self.running_batch,
    self.new_token_ratio,
    self.max_prefill_tokens,
    self.chunked_prefill_size,
    running_bs if self.is_mixed_chunk else 0,
)
```

### 差异 5: `add_one_req()` 方法签名不同

**sglang_vit** (`schedule_policy.py:445-447`):
```python
def add_one_req(
    self, req: Req, has_chunked_req: bool, enable_hierarchical_cache: bool = False
):
```

**semipd_pp** (`schedule_policy.py:492`):
```python
def add_one_req(self, req: Req, has_chunked_req: bool):
```

**问题**：
- `sglang_vit` 有一个额外的 `enable_hierarchical_cache` 参数
- `semipd_pp` 缺少这个参数

### 差异 6: `calc_priority()` 返回值不同

**sglang_vit** (`scheduler.py:1759`):
```python
prefix_computed = self.policy.calc_priority(self.waiting_queue)
```

**semipd_pp** (`scheduler.py:1789`):
```python
self.policy.calc_priority(self.waiting_queue)
```

**问题**：
- `sglang_vit` 使用了 `calc_priority()` 的返回值 `prefix_computed`
- `semipd_pp` 忽略了返回值
- `prefix_computed` 用于决定是否传递 `tree_cache` 给 `init_next_round_input()`

### 差异 7: VIT 相关逻辑

**sglang_vit** (`scheduler.py:1779-1806`):
```python
# 🔧 流水线并行：分类请求，优先处理 VIT 已完成的请求
# 只在 PP0 上检查 vit_pending，PP1 不需要检查
vit_ready_reqs = []
vit_pending_reqs = []

for req in self.waiting_queue:
    # 只在 PP0 上检查 vit_pending
    if self.pp_group.is_first_rank:
        has_vit_pending = hasattr(req, 'vit_pending')
        vit_pending_value = getattr(req, 'vit_pending', None)

        if has_vit_pending and vit_pending_value:
            vit_pending_reqs.append(req)
            continue

    vit_ready_reqs.append(req)

# Get requests from the waiting queue to a new prefill batch
# 优先处理 VIT 已完成的请求
for req in vit_ready_reqs:
    ...
```

**semipd_pp** (`scheduler.py:1810-1811`):
```python
# Get requests from the waiting queue to a new prefill batch
for req in self.waiting_queue:
    ...
```

**问题**：
- `sglang_vit` 有 VIT 请求优先级处理逻辑
- `semipd_pp` 缺少这个逻辑

### 差异 8: `event_loop_pp()` 中的 VIT 轮询

**sglang_vit** (`scheduler.py:773-775`):
```python
# 🔧 修复：在循环开始就轮询 VIT 结果
# 这样即使 PP 通信阻塞，也能在下一次循环时获取 VIT 结果
self.poll_vit_results()
```

**semipd_pp** (`scheduler.py:945-954`):
```python
while True:
    server_is_idle = True
    for mb_id in range(self.pp_size):
        self.running_batch = self.running_mbs[mb_id]
        self.last_batch = last_mbs[mb_id]

        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        mbs[mb_id] = self.get_next_batch_to_run()
        # ❌ 缺少 poll_vit_results() 调用
```

**问题**：
- `sglang_vit` 在每次循环开始时调用 `self.poll_vit_results()`
- `semipd_pp` 缺少这个调用

## 📊 影响分析

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

## ✅ 修复方案

### 修复 1: 修正 `init_next_round_input()` 调用

**文件**: `semipd_pp/python/sglang/srt/managers/scheduler.py`

**位置**: 行 1835

**修改前**:
```python
req.init_next_round_input(self.tree_cache)
```

**修改后**:
```python
req.init_next_round_input(
    None if prefix_computed else self.tree_cache,
    self.enable_hierarchical_cache,
)
```

### 修复 2: 修正 `add_one_req()` 调用

**文件**: `semipd_pp/python/sglang/srt/managers/scheduler.py`

**位置**: 行 1836

**修改前**:
```python
res = adder.add_one_req(req, has_chunked_req=(self.chunked_req is not None))
```

**修改后**:
```python
res = adder.add_one_req(
    req, self.chunked_req, self.enable_hierarchical_cache
)
```

### 修复 3: 使用 `calc_priority()` 的返回值

**文件**: `semipd_pp/python/sglang/srt/managers/scheduler.py`

**位置**: 行 1789

**修改前**:
```python
self.policy.calc_priority(self.waiting_queue)
```

**修改后**:
```python
prefix_computed = self.policy.calc_priority(self.waiting_queue)
```

### 修复 4: 添加 VIT 轮询（可选，如果需要 VLM 支持）

**文件**: `semipd_pp/python/sglang/srt/managers/scheduler.py`

**位置**: 行 945（在 `while True:` 循环开始后）

**添加**:
```python
while True:
    # 🔧 修复：在循环开始就轮询 VIT 结果
    # 这样即使 PP 通信阻塞，也能在下一次循环时获取 VIT 结果
    if hasattr(self, 'poll_vit_results'):
        self.poll_vit_results()
    
    server_is_idle = True
    for mb_id in range(self.pp_size):
        ...
```

## 🎯 预期效果

修复后：
- ✅ 原生 SGLang + PP + Chunked Prefill：输出正常，无 token 重复
- ✅ Semi-PD + PP + Chunked Prefill：输出正常，无 "A" 插入或 token 重复
- ✅ 单请求和并发请求：都正常
- ✅ VLM 请求：正常
- ✅ 消除间歇性错误：输出稳定可靠

## 📝 总结

**根本原因**：
- `semipd_pp/` 的代码与 `sglang_vit/` 的代码存在 API 不兼容
- 关键参数 `enable_hierarchical_cache` 和 `prefix_computed` 缺失
- 导致 chunked prefill 的状态管理错误

**修复策略**：
- 对齐 `sglang_vit/` 的 API 调用方式
- 传递所有必要的参数
- 确保 chunked request 的状态管理正确

