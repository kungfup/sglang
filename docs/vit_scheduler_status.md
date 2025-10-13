# VIT 进程解耦现状与问题分析

更新时间：2025-10-12  
日志样本：`semipd_PP_vit_optimized.log`

---

## 1. 现有设计概览

### 1.1 架构拆分
- **主 Scheduler (PP0)**：接收多模请求，负责把图片样本异步交给 VIT 进程，并在 `req.vit_pending` 为 `True` 时阻塞后续 Prefill。
- **VIT Scheduler 进程**：独立 ZMQ PAIR 通道；批量加载图片、执行 ViT forward，并通过 CUDA IPC 返回 embedding。
- **VIT Client (PP0 内部)**：后台线程维护一个发送/接收循环，负责写共享内存、发送请求、收结果并向主 Scheduler 回传。

### 1.2 显存与并发控制
- `EmbeddingMemoryManager` 维护 **inflight**（正在执行）和 **cache**（已缓存 embedding）两类显存占用：
  - `try_reserve()` 成功才能进入批处理；
  - `commit()` / `abandon()` / `move_to_cache()` / `release_cache()` 保证额度即时归还；
  - 由 `SGLANG_VIT_CACHE_GPU_GB` 与 `SGLANG_VIT_MAX_INFLIGHT` 配置。
- 每个请求按 `_estimate_request_bytes()` 动态估算显存 (`像素tensor * overhead + margin`)，可由 `SGLANG_VIT_ESTIMATE_OVERHEAD` 等环境变量调节。

### 1.3 延迟重试与降级
- 当预算不足时，`_deferred_until` + `_defer_counter` 记录“最早可调度时间”和失败次数。
- `_schedule_next_batch()` 根据最早可重试时间设置定时器，避免固定 10ms 自旋。
- 若同一请求的失败次数超过 `SGLANG_VIT_MAX_DEFER_COUNT`（默认 5），强制走 `_process_single_request_fallback()`：
  - 不写入缓存，直接返回结果，确保业务继续向前。

### 1.4 共享内存复用
- 首次从共享内存读取 `pixel_values` / `image_grid_thw` 后，缓存到 `VITRequest` 对象属性上，后续重试不用再访问共享内存。
- 请求完成或失败时调用 `_clear_request_tensors()` 释放这些引用。

### 1.5 调度循环
- 主循环 `event_loop()` 在有新请求或超时到达时触发 `_process_batch()`。
- `_process_batch()` 将 pending 队列按“可立即执行 / 需要重试 / cache hit”三类拆分：
  - cache hit 直接返回；
  - cache miss 进入 `_process_cache_misses_batch()`；
  - 资源不足的请求重新放回队列，等待下一次触发。

---

## 2. 当前症状

在 `nohup python -m sglang.launch_server ...` 的运行中，我们串行发送 4 个多模请求，日志显示：

1. **仅第一个请求成功返回**：  
   ```
   [VIT Scheduler] ✅ Response sent: b1fd564a6170486da085f180786bf8c1
   ```
2. **后续请求一直停留在 pending 队列**，反复出现：
   ```
   ⚠️ Request XXXXX cannot be scheduled now (estimate=529.1 MB, total_usage=2.07 GB)
   Queue not empty (1 requests), scheduling next batch processing in 10ms
   ```
3. 某些请求还出现共享内存读取失败：
   ```
   ⚠️ Shared memory not found for 8457c49b391f4ce781b72491e5436705
   ```
4. 主 Scheduler 持续打印 `vit_pending=True` 的等待日志，导致后续 Prefill 阶段无法继续。

结论：**显存预算触发后，重试/降级机制未能让请求完成，造成队列长期阻塞**。

---

## 3. 根因推测

1. **显存额度未被及时归还**  
   - 日志中 `total_usage=2.07 GB` 恒定不变，说明第一批成功执行后占用未释放。
   - 容易导致的场景：fallback 成功执行但 `move_to_cache()` 没被调用、或 `release_cache()` 未触发（例如主 Scheduler 没发 free 信号）。

2. **延迟重试后仍执行 fallback，但结果未送达**  
   - fallback 路径中调用 `_process_single_request_fallback()`，理论上会生成 embedding 并发送响应，但没有 `Response sent` 日志。
   - 可能是在 fallback 后早于 `_send_response()` 就抛异常，或 `_update_cache()` 中断。

3. **客户端提前销毁共享内存**  
   - 当 VIT 进程多次尝试读取同一共享内存，而客户端已经在超时清理，这会触发 `Shared memory not found`，随后请求被清出队列，导致主 Scheduler 永远等不到结果。
   - 需要确保请求在 fallback 或重试时已经将数据深拷贝到进程内存里。

4. **定时器回调过于频繁，未结合实际重试时间**  
   - 即使 `_deferred_until` 记录了重试时间，仍会频繁触发 `_process_batch()`；当资源始终空不出来时，会出现大量空转。

---

## 4. 下一步改进建议

1. **显存占用追踪**
   - 在 `_update_cache()` 和 `_free_cache()` 打印当前 `memory_manager.stats()`，便于确认成功返回后额度是否归还。
   - 若缓存引用计数未归零，需要核对 `notify_embedding_consumed()` 是否被调用。

2. **安全复制 + 共享内存清理协议**
   - 在 `submit_async()` 中增加 ACK / 发送成功后再释放共享内存，避免在重试期间 SHM 被提前 unlink。
   - 或者在 VIT 进程第一次读取后立刻 `clone()` 并告知 client 可以删除。

3. **fallback 返回路径检查**
   - 校验 `_process_single_request_fallback()` 是否总是发送响应（包括 `benchmark_mode` 和失败分支）。
   - 当 fallback 发生时，可以临时打印 `Response sent`，确认主 Scheduler 是否收到了结果。

4. **重试调度策略优化**
   - 将 `delay = self._retry_backoff * count` 改成指数退避（50ms、150ms、350ms...），减少泄洪。
   - 当 `total_usage` 长期占满且没有 free 信号，可自动丢弃显存最大的缓存项，确保新请求能挤入预算。

5. **监控与告警**
   - 增加可视化指标：`vit_pending_queue_length`, `vit_memory_usage_bytes`, `vit_defer_limit_hits` 等。
   - 在主 Scheduler 端若某请求 `vit_pending` 持续超过阈值，主动触发重试或回退逻辑。

---

## 5. 结论

目前的解耦实现已经具备显存预算、延迟重试和 fallback 机制，但出现“只处理首个请求”的瓶颈，说明某些资源释放或重试路径仍有漏洞。下一步应重点排查：

1. fallback 是否真正发送了响应，并在日志中找到对应 `Response sent`。
2. 显存预算在成功路径下是否及时归还，防止 `total_usage` 长时间保持高值。
3. 客户端共享内存是否能坚持到请求完成。

在完成上述修复前，请避免把 `SGLANG_VIT_MAX_INFLIGHT` 设置为大于 1，并适当降低 `SGLANG_VIT_CACHE_GPU_GB`，减轻显存压力，确保服务可用。 
