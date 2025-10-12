# SGLang 批调度与显存池管理（含核心代码）

本文以中文系统梳理 SGLang 的请求排队、批次（batch）构建、显存预算与释放流程；并配合核心代码片段，解释多请求并发/多 batch 计算时如何在显存不足情况下安全地控制 batch 的组建与回退。

提示：本文覆盖的关键文件路径如下（点击可打开）：
- `sglang/python/sglang/srt/managers/scheduler.py:1676`
- `sglang/python/sglang/srt/managers/schedule_policy.py:268`
- `sglang/python/sglang/srt/managers/schedule_batch.py:1155`
- `sglang/python/sglang/srt/mem_cache/memory_pool.py:54`
- `sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py:200`
- `sglang/python/sglang/srt/disaggregation/decode.py:300`

## 1. 核心组件与职责
- `Scheduler` 负责：请求入队/出队、prefill/decode 批次构建与切换、触发前向、以及与 worker 同步。
- `ScheduleBatch` 代表一次前向（prefill 或 decode），持有本轮参与的请求及其 KV 映射。
- `PrefillAdder` 负责在 prefill 阶段“试装”请求，依据 KV 可用 token 与保守估计 `new_token_ratio` 做预算决策。
- 显存池：
  - `ReqToTokenPool` 管理“请求 → token 索引位置”的映射（限制并发请求数）。
  - `TokenToKVPoolAllocator` 管理“token 索引 → KV 物理页”（限制总 KV 容量）。

## 2. 请求生命周期（入队 → 批次 → 释放）
1) 入队：`_add_request_to_queue` 会放入 `waiting_queue` 并打上队列时间戳。

```python
# sglang/python/sglang/srt/managers/scheduler.py:1422
def _add_request_to_queue(self, req: Req):
    req.queue_time_start = time.perf_counter()
    ...
    self.waiting_queue.append(req)
```

2) 取下一个批：`get_next_batch_to_run()` 先合并上一轮 extend 结果，再优先尝试 prefill，否则进入 decode。

```python
# sglang/python/sglang/srt/managers/scheduler.py:1676
def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
    if self.chunked_req:
        self.tree_cache.cache_unfinished_req(self.chunked_req)
        self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
    ...
    new_batch = self.get_new_batch_prefill()
    if new_batch is not None:
        ret = new_batch
    else:
        if not self.running_batch.is_empty():
            self.running_batch = self.update_running_batch(self.running_batch)
            ret = self.running_batch if not self.running_batch.is_empty() else None
        else:
            ret = None
    return ret
```

3) 构建 prefill 批：`get_new_batch_prefill()` 控制请求数与 token 预算，产出 `ScheduleBatch` 并 `prepare_for_extend()`。

```python
# sglang/python/sglang/srt/managers/scheduler.py:1732
def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
    if (self.running_batch.batch_is_full or len(self.waiting_queue) == 0) and self.chunked_req is None:
        return None
    running_bs = len(self.running_batch.reqs)
    if self.get_num_allocatable_reqs(running_bs) <= 0 and not self.chunked_req:
        self.running_batch.batch_is_full = True
        return None

    adder = PrefillAdder(
        self.tree_cache,
        self.token_to_kv_pool_allocator,
        self.running_batch,
        self.new_token_ratio,
        self.max_prefill_tokens,
        self.chunked_prefill_size,
        running_bs if self.is_mixed_chunk else 0,
    )
    # 遍历 waiting_queue，逐个尝试加入 can_run_list（受预算约束）
    for req in vit_ready_reqs:  # 省略 VIT 分类细节
        if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
            self.running_batch.batch_is_full = True
            break
        res = adder.add_one_req(req, self.chunked_req, self.enable_hierarchical_cache)
        if res != AddReqResult.CONTINUE:
            if res == AddReqResult.NO_TOKEN:
                self.running_batch.batch_is_full = True
            break

    can_run_list = adder.can_run_list
    if len(can_run_list) == 0:
        return None

    new_batch = ScheduleBatch.init_new(...)
    new_batch.prepare_for_extend()
    return new_batch
```

4) 前向：`run_batch()` 将 `ScheduleBatch` 下发到 TP worker。

5) 输出与释放：`SchedulerOutputProcessorMixin` 更新请求状态、判断是否完成，并在 `free_group_end` 时批量释放 KV 页面。

```python
# sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py:200
self.token_to_kv_pool_allocator.free_group_begin()
...  # 逐请求处理输出、判断完成
self.token_to_kv_pool_allocator.free_group_end()
```

6) Decode 循环：`running_batch` 内的请求不断 decode；每轮 decode 前会做显存检查与必要的回退。

## 3. Prefill 阶段：请求与 token 预算双重约束
- 请求数上限：

```python
# sglang/python/sglang/srt/managers/scheduler.py:1726
def get_num_allocatable_reqs(self, running_bs):
    res = global_server_args_dict["max_micro_batch_size"] - running_bs
    if self.pp_size > 1:
        res = min(res, self.req_to_token_pool.available_size())
    return res
```

- token 预算（关键）：`PrefillAdder` 对可用 KV token 做“当前/总量”两种视角的预算，并考虑运行中 batch 的“未来新 token”占用（`new_token_ratio`）。

```python
# sglang/python/sglang/srt/managers/schedule_policy.py:268
class PrefillAdder:
    @property
    def rem_total_tokens(self):
        return (
            self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size()
            - self.rem_total_token_offset
        )
    @property
    def cur_rem_tokens(self):
        return (
            self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size()
            - self.cur_rem_token_offset
        )
    def budget_state(self):
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return AddReqResult.NO_TOKEN
        ...
```

- chunked prefill：当 `chunked_prefill_size` 生效时，长 prompt 会被切分；当前 chunk 入批，剩余部分作为 `new_chunked_req` 等待下一轮。

```python
# sglang/python/sglang/srt/managers/schedule_policy.py:348
def add_chunked_req(self, req: Req):
    truncated = req.extend_input_len > self.rem_chunk_tokens
    req.extend_input_len = min(req.extend_input_len, self.rem_chunk_tokens)
    self.can_run_list.append(req)
    # 如果被截断，本轮不再为其预留未来新 token
    return req if truncated else None
```

## 4. Prefill 批的落地：`prepare_for_extend()`
这一步做了“分配请求槽位、分配 KV 页面、写入索引”三件事：

```python
# sglang/python/sglang/srt/managers/schedule_batch.py:1155
def prepare_for_extend(self):
    bs = len(self.reqs)
    req_pool_indices = self.alloc_req_slots(bs)  # ReqToTokenPool.alloc
    ...
    # 写入已缓存的前缀索引
    if pre_len > 0:
        self.req_to_token_pool.write((req.req_pool_idx, slice(0, pre_len)), req.prefix_indices)
    ...
    # 分配新的 KV 页面（未分页或分页）
    if self.token_to_kv_pool_allocator.page_size == 1:
        out_cache_loc = self.alloc_token_slots(extend_num_tokens)
    else:
        out_cache_loc = self.alloc_paged_token_slots_extend(...)
    ...
    # 最终把 mapping 写入 req_to_token_pool.req_to_token（后续 decode 复用）
```

## 5. Decode 阶段：显存检查与回退（retract）
- 执行 decode 前，对“下一步生成 token 将消耗的 KV 页面”做校验，不足则回退：

```python
# sglang/python/sglang/srt/managers/schedule_batch.py:1378
def check_decode_mem(self, buf_multiplier=1):
    tokens_required = self.new_page_count_next_decode() * buf_multiplier * self.token_to_kv_pool_allocator.page_size
    if self.token_to_kv_pool_allocator.available_size() >= tokens_required:
        return True
    self.tree_cache.evict(tokens_required)
    return self.token_to_kv_pool_allocator.available_size() >= tokens_required
```

```python
# sglang/python/sglang/srt/managers/scheduler.py:1913
def update_running_batch(self, batch: ScheduleBatch):
    if not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier):
        retracted_reqs, new_token_ratio = batch.retract_decode(self.server_args)
        self.new_token_ratio = new_token_ratio  # 调紧预算
        self._extend_requests_to_queue(retracted_reqs)
    else:
        # 显存宽裕则逐步放宽预算（提升吞吐）
        self.new_token_ratio = max(
            self.new_token_ratio - self.new_token_ratio_decay,
            self.min_new_token_ratio,
        )
    batch.prepare_for_decode()
    return batch
```

- 回退过程会释放 KV 页面以及请求槽位，并对 `tree_cache` 解除引用：

```python
# sglang/python/sglang/srt/managers/schedule_batch.py:1420
token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, last_uncached_pos : seq_lens_cpu[idx]]
self.token_to_kv_pool_allocator.free(token_indices)
self.req_to_token_pool.free(req.req_pool_idx)
self.tree_cache.dec_lock_ref(req.last_node)
```

## 6. 显存池细节：请求槽位 + KV 页面
- 请求槽位池（限制并发请求数）：

```python
# sglang/python/sglang/srt/mem_cache/memory_pool.py:54
class ReqToTokenPool:
    def __init__(self, size, max_context_len, device, enable_memory_saver):
        self.req_to_token = torch.zeros((size, max_context_len), dtype=torch.int32, device=device)
        self.free_slots = list(range(size))
    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index
    def free(self, free_index: Union[int, List[int]]):
        ...  # 归还请求槽位
```

- KV 页面池（限制总 KV 页数），支持“批量释放”降低拼接开销：

```python
# sglang/python/sglang/srt/mem_cache/memory_pool.py:180
class TokenToKVPoolAllocator:
    def alloc(self, need_size: int):
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_index
    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []
    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))
```

## 7. 输出阶段的资源回收（成组释放）
Decode 阶段在处理输出后，统一在一次 `free_group_end()` 中释放 KV 页，避免频繁 concat：

```python
# sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py:200
self.token_to_kv_pool_allocator.free_group_begin()
...  # 逐请求 append/finish 判断
self.token_to_kv_pool_allocator.free_group_end()
```

## 8. 拆分（Disaggregation）模式要点
- Prefill 服务器：`PrefillBootstrapQueue` 负责预分配 KV，prefill 侧会额外受 `req_to_token_pool.available_size()` 约束，避免溢出。
  - `sglang/python/sglang/srt/managers/scheduler.py:1732`（prefill）
- Decode 服务器：`DecodePreallocQueue` 会提前为请求分配 req/KV，并保留 decode 头寸：

```python
# sglang/python/sglang/srt/disaggregation/decode.py:300
def _allocatable_tokens(self) -> int:
    allocatable_tokens = (
        self.token_to_kv_pool_allocator.available_size()
        - self.num_reserved_decode_tokens
        * (len(self.scheduler.running_batch.reqs) + len(self.transfer_queue.queue) + len(self.scheduler.waiting_queue))
    )
    ...
```

## 9. 调参建议与监控
- `--max-micro-batch-size`：限制单个 prefill microbatch 的请求数；`max_running_requests` 则限制全局并发请求槽位。
- `--mem-fraction-static` / `--max-total-num-tokens`：决定 KV 物理容量，直接影响 `TokenToKVPoolAllocator` 可用空间。
- `--schedule-conservativeness`：缩放初始 `new_token_ratio`（保守值越大，prefill 越谨慎，更不易 OOM，但吞吐可能下降）。
- `--chunked-prefill-size` / `--enable-mixed-chunk`：长 prompt 切块 + 混合 prefill/decode，可提升利用率；需确保 `tree_cache` 的逐出策略适配。
- 监控：`SchedulerStats` 的 `num_used_tokens/token_usage/num_queue_reqs` 可帮助识别临近内存极限。

## 10. 小结
SGLang 以“请求槽位 + KV 页面”的双层池化来管控 batch：prefill 阶段通过 `PrefillAdder` 的 token 预算与 `new_token_ratio` 保守估计来挑选请求；decode 阶段通过 `check_decode_mem` + `retract_decode` 在显存不足时回退请求；输出阶段用成组释放降低释放开销。配合 `tree_cache` 的共享/逐出策略，便能在多请求、多 batch 的条件下兼顾吞吐与显存安全。
