# VIT 进程解耦与多 Batch 并行技术方案

> 目标：让 ViT 计算像主 Scheduler 一样具备稳定的批量调度能力，不再成为异步生成流程的瓶颈，并彻底解决当前出现的 OOM、积压、请求丢失等问题。本文仅给出方案设计与实施步骤，**暂不改动代码**。

## 1. 背景 & 设计目标

当前 VIT Scheduler 作为单独进程运行，但在大批量请求下会出现：

- **OOM/显存碎片化**：`VITMemoryPool` 只是记录用量，没有真正的“槽位”约束，无法像 `ReqToTokenPool` 那样在调度层面阻止超量请求。
- **队列堵塞/请求丢失**：ZMQ PAIR + 自定义队列在压力下容易堆积，缺乏“调度器—客户端”之间的背压与状态同步。
- **批量不稳**：虽然存在 `compute_batch`，但批大小随缘，遇到 OOM 会整批丢弃，并缺乏像 `PrefillAdder` 那样的预算策略。
- **缓存释放不及时**：事件驱动释放存在 race 条件，导致缓存持有过久进一步挤占显存。

目标是比照主 Scheduler（参见 `sglang/docs/batch_scheduler_memory.md`）的做法，为 ViT 子系统引入完整的调度循环、内存池预算、批量合并与回退机制，并在客户端协议上加上背压/确认，确保「多请求、多 batch」模式下仍然稳定。

## 2. 关键设计原则

1. **请求槽位 + 显存页 双重约束**：仿照 `ReqToTokenPool` + `TokenToKVPoolAllocator`，在 ViT 侧维护 `RequestSlotPool` 和 `EmbeddingPagePool`，实现“入队前先占位”。
2. **调度循环与批处理器分离**：一个线程/协程专门收集请求，另一个线程负责按照预算组 batch 并调用 `compute_batch`。
3. **显存预算可回退**：借鉴 `PrefillAdder` 的 `rem_total_tokens` 逻辑，优先按估算值保守预留，如果发生 OOM 立即缩小 `new_decode_ratio` 并对部分请求回退重排。
4. **严格的请求状态机**：WAITING → PREFETCHED → RUNNING → FINISHED/FAILED，任何失败都返回“可重试”信号，避免 silent drop。
5. **客户端背压**：主 Scheduler 在发送之前必须先申请 RequestSlot，失败则立即回退到同步模式或排队等待。
6. **缓存引用计数**：同一 `image_hash` 的缓存需要和请求引用计数绑定，只有所有请求释放后才真正释放显存。

## 3. 架构总览

```
┌─────────────────────────────────────────────────────┐
│ 主 Scheduler (PP0)                                   │
│   ├─ VITSchedulerClient                              │
│   │   ├─ RequestSlot 申请/释放                       │
│   │   ├─ 共享内存/IPC 栈                             │
│   │   └─ 结果收集与 `notify_embedding_consumed()`   │
│   └─ 主调度器 (参考 batch_scheduler_memory)          │
└─────────────────────────────────────────────────────┘
                ⇅ ZMQ ROUTER/DEALER + 心跳
┌─────────────────────────────────────────────────────┐
│ VIT Scheduler 进程                                   │
│   ├─ RequestIngress 线程 (ZMQ, 心跳, slot 分配)      │
│   ├─ SchedulerCore 线程 (批处理调度、预算)            │
│   ├─ ComputeWorker 池 (1~N, GPU stream)              │
│   ├─ EmbeddingCache (LRU + 引用计数)                 │
│   └─ Metrics / Watchdog                              │
└─────────────────────────────────────────────────────┘
```

## 4. 请求生命周期

| 状态            | 触发方式                              | 说明 |
|-----------------|---------------------------------------|------|
| `WAITING`       | 客户端提交成功 + 获取 slot            | 放入 `waiting_queue` |
| `PREFETCHED`    | 已从共享内存/CPU tensor 拉取到 GPU 前缓冲 | 在 batch 选中后立即搬运 |
| `RUNNING`       | 批次进入 `compute_batch`             | 记录 batch id 与 GPU stream |
| `FINISHED`      | 结果写回 (`embedding_ipc_handle`)     | 推送 response，并登记 cache 引用计数 |
| `FAILED_RETRY`  | OOM/显存不足/异常                     | 释放资源、回退到 `WAITING`，最多 N 次 |
| `FAILED_ABORT`  | 超时或用户主动停止                     | 通知客户端清理共享内存并返回错误 |

状态机中的每次跃迁必须更新：
- `RequestSlotPool`（请求槽位分配/释放）
- `EmbeddingPagePool`（估算显存占用，批次释放时归还）
- `pending_requests` 结构（调度队列）

## 5. 调度循环（伪代码）

```python
while not stopped:
    drain_free_signals()
    pull_new_requests()  # 从 ROUTER 拉消息，完成 slot 申请并入队

    if running_batch and running_batch.done():
        finalize_batch(running_batch)

    if scheduler_core.should_launch_batch():
        batch = scheduler_core.build_next_batch()
        if batch:
            launch_batch(batch)
            continue

    sleep(SMALL_INTERVAL)
```

其中 `build_next_batch()` 需要参考 `scheduler.py:get_new_batch_prefill()`：  
1. 拷贝 `waiting_queue`，按 hash、优先级分组（cache hit 先返回）。  
2. 使用 `VITPrefillAdder` 判断是否还能加入更多请求：  
   - 预算输入尺寸（例如图片面积 × 通道数 × dtype 大小）。  
   - 查询 `EmbeddingPagePool.available_size()`。  
   - 若不足则停止；如果某请求单独也无法分配，则退回并返回错误。  
3. 对于当前 batch 内的所有请求，调用 `_prefetch_to_gpu()`，成功后置为 `PREFETCHED`，失败则回退。

## 6. 内存池设计

### 6.1 RequestSlotPool

仿照 `ReqToTokenPool`：

- 初始化时以 `max_running_vit_requests` 为大小（可从配置或显存自动推算）。
- `alloc(num_requests)` 返回连续/离散请求 id 槽位；失败时客户端必须等待。
- `free(slot_ids)` 在请求完成或失败时释放。

### 6.2 EmbeddingPagePool

对应 `TokenToKVPoolAllocator`：

- 设定 `page_size = 默认每张图片的隐向量大小（例如 seq_len * hidden_dim * dtype_size)`。
- 维护 `free_pages`，每次批处理中，根据实际 `pixel_values` 数量向池中申请 `ceil(num_tokens/page_size)` 页。
- 支持 `free_group_begin/end`，在处理批次结果时统一释放，避免碎片化。
- 当可用页不足时，配合 `cache.evict()` 释放 LRU embedding，或直接缩小批次。

### 6.3 OOM 回退

一旦 `compute_batch` 抛出 OOM：
- 记录 `oom_timestamp`，缩小 `dynamic_batch_size`（类似 `new_token_ratio`）。
- 将批次内请求标记为 `FAILED_RETRY` 并重新入队（带重试计数）。
- 若连续超过阈值（如 3 次），触发“降级到单请求模式”并向客户端告警。

## 7. 批内并行策略

1. **多 Stream**：对同一批次的 `pixel_values` 搬运与前向采用独立 CUDA stream，允许下一个批次在主 stream 上排队，但需通过事件 (`torch.cuda.Event`) 确保顺序。
2. **多 Worker**：若显存允许，可配置 `num_compute_workers > 1`，每个 worker 绑定独立 stream，由 `SchedulerCore` 分发 batch。
3. **TP 支持**：延续现有 `_run_tp_worker` 逻辑，但 broadcast 时传递批量索引及 slot id，保证回写时能定位请求。

## 8. 客户端协议改造

- **Slot 申请**：`VITSchedulerClient.submit()` 在创建共享内存前先发送 `{type: "request_slot", request_id, est_size}`，由 VIT 进程返回 `{"status": "ok", "slot_id": X}` 或 `{"status": "wait", "retry_after_ms": ...}`。只有拿到 slot 才进入正式提交流程。
- **显式 ACK**：VIT 进程在成功接收到 payload 后发送 `ack`，客户端才将 pending 设置为 `sent=True`。
- **心跳与超时**：
  - ROUTER/DEALER 模式下，客户端需要定期发送心跳；若超时则自动重建连接。
  - 客户端检测到连续 `ack` 超时或 `slot` 请求失败时，可降级为同步计算或延迟投递。
- **结果格式**：保持 `VITResponse` 结构，但新增 `slot_id`、`batch_id` 字段，方便调试与释放。

## 9. 缓存与释放

1. **引用计数**：缓存结构改为 `{image_hash: CacheEntry(tensor, ref_count)}`。  
   - 当请求命中缓存时，`ref_count += 1`。  
   - 收到 `free_embedding` 信号时 `ref_count -= 1`，若为 0 则释放并归还 `EmbeddingPagePool`。  
2. **缓存预算**：缓存占用超过设定比例（例如 60%）时，按照 LRU 主动驱逐，并通知相关请求需要重算。
3. **超时清理**：后台线程每 N 秒扫描缓存，对长时间未释放的 hash 做保护性释放或打印告警。

## 10. 可靠性 & 监控

- **Watchdog**：SchedulerCore 定期检查 `running_batches` 是否超时，超时则主动终止并回收资源。
- **指标**：
  - `vit_pending_requests`
  - `vit_running_batches`
  - `vit_cache_bytes` / `capacity`
  - `vit_batch_size` / `dynamic_batch_size`
  - `vit_oom_events_total`
  - `vit_slot_exhausted_total`
- **日志**：统一格式：`[VITScheduler] [batch_id=XX] ...`，便于 grep；错误日志附带 `slot_id`、`hash`。

## 11. 实施步骤

1. **重构骨架**  
   - 新建 `vit_scheduler_core.py`（调度循环、内存池、batch 构建）。  
   - 现有 `vit_scheduler.py` 精简为：启动器 + 线程管理 + cache 实现。  
   - 将 `VITMemoryPool` 替换为 `EmbeddingPagePool`。
2. **客户端更新**  
   - 引入 `request_slot`/`release_slot` RPC。  
   - 背景线程按 ACK 更新 `pending`.sent，删除旧的重试逻辑。  
   - 结果回调增加 `slot_id` 校验。
3. **逐步迁移**  
   - 环境变量控制是否启用新调度。  
   - 先在小 batch（<=2）下验证，再放大。  
   - 加入集成测试：模拟多请求并行（并注入 OOM），确认没有请求泄露、缓存释放正常。
4. **回退策略**  
   - 新实现异常时自动 fallback 到老的同步流程（或直接返回错误）。  
   - 所有变更 behind flag，确保线上可开关。

## 12. 文档与培训

- 更新 `VIT_核心修复_完成总结.md`：加入“调度解耦”章节，说明新增模块与配置项。
- 在 `sglang/docs/` 下撰写 `vit_scheduler_usage.md`，覆盖：
  - 新配置 (`--vit-max-requests`, `--vit-memory-pool-gb` 等)  
  - 监控指标说明  
  - 常见故障排查（OOM、slot exhausted、cache leak）
- 编写新的架构图与请求时序图，便于团队成员理解。

## 13. 预期收益

- **吞吐提升**：稳定支持多 batch 并行，避免单请求串行造成的 GPU 空闲。
- **鲁棒性增强**：显存使用可控，OOM 时自动回退，不会拖垮主进程。
- **问题可观测**：通过 slot、batch、cache 指标快速定位瓶颈。
- **与主 Scheduler 对齐**：整体风格与主调度器一致，后续可重用更多公共组件（如树形 cache、分布式事件）。

---

> 下一步：确定详细的接口定义与数据结构后，即可着手实现与单元测试。建议先在开发环境完成调试，再逐步推到线上。README 本文可作为后续开发的主参考。***
