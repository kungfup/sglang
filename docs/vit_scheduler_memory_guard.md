# VIT 调度限流与显存保护方案

## 背景

近期压测中发现，VIT Scheduler 在接收多并发图像请求时存在两类问题：

1. 首个请求完成后，后续请求长期停留在待处理队列，日志中不断重复 `cannot be scheduled now`，无法继续推进。
2. 当显存预算不足时，请求被反复重试，并因为共享内存被客户端清理导致 `Shared memory not found`，最终主 Scheduler 卡住。

这些现象表明旧的“页面池 + 固定重试”策略没有真正将显存预算、并发限制和失败回退串成闭环。

## 目标

- 严格限制同时执行的 VIT 任务数和总显存占用，避免 GPU OOM。
- 对于临时资源不足的请求，改为带指数退避的“延迟重试”，而不是毫秒级自旋。
- 当请求长时间得不到资源时，自动降级为单请求执行（不写入缓存），确保业务不会停滞。
- 避免重复重复加载共享内存，减少客户端清理后的错误。

## 实现要点

### 1. 统一的显存预算器 `EmbeddingMemoryManager`

- 记录两部分占用：正在执行的 inflight 请求和已经缓存的 embedding。
- `try_reserve()` 失败时不会修改任何预算；成功才计入 inflight。
- `commit()` / `move_to_cache()` / `release_cache()` 保证执行完成或缓存释放后额度及时归还。
- 通过环境变量调整：
  - `SGLANG_VIT_CACHE_GPU_GB`：GPU 显存上限（默认 10GB）
  - `SGLANG_VIT_MAX_INFLIGHT`：同时执行的请求数（默认 4）

### 2. 延迟重试 + 限次强制降级

- 每个请求维护 `_deferred_until` 和 `_defer_counter`。
- 当显存/并发不足导致 `try_reserve()` 失败时，记录延迟时间（默认 50ms，可随重试次数线性增加），请求不会立即重新进入计算。
- 若同一请求连续失败次数超过 `SGLANG_VIT_MAX_DEFER_COUNT`（默认 5），自动触发 `_process_single_request_fallback()`：
  - 即使预算未预留也强制执行单次 ViT 计算。
  - 不写入缓存，直接返回结果，避免一直占用 GPU。

### 3. 复用已加载的共享内存数据

- 首次从共享内存中加载 `pixel_values` / `image_grid_thw` 后，会缓存到请求对象上。
- 后续延迟重试可直接复用，防止客户端提前清理共享内存导致的 `FileNotFoundError`。
- 请求完成或失败时，调用 `_clear_request_tensors()` 释放 CPU 内存。

### 4. 按需调度下一批

- `_process_batch()` 完成后，根据最早可重试时间选择下一次调度，而非固定 10ms 自旋。
- 如果队列为空，则取消定时器。

### 5. 全路径释放

无论成功、失败还是降级，都确保：

- `self._processing_requests` 中移除对应 request_id。
- `_deferred_until`、`_defer_counter` 清零。
- `memory_manager` 归还 inflight 额度。
- 共享内存缓存（若有）被释放。

## 关键环境变量

| 变量 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `SGLANG_VIT_CACHE_GPU_GB` | `10.0` | GPU 显存上限（GB） |
| `SGLANG_VIT_MAX_INFLIGHT` | `4` | 同时执行的 VIT 请求数 |
| `SGLANG_VIT_ESTIMATE_OVERHEAD` | `4.0` | 估算倍数（像素 Tensor 的倍数），用于预留显存 |
| `SGLANG_VIT_ESTIMATE_MARGIN_MB` | `64.0` | 每个请求额外的安全余量（MB） |
| `SGLANG_VIT_RETRY_BACKOFF_MS` | `50.0` | 延迟重试的起始 backoff（毫秒） |
| `SGLANG_VIT_MAX_DEFER_COUNT` | `5` | 达到次数后强制降级为单请求执行 |

## 代码位置

- `sglang/python/sglang/srt/managers/vit_scheduler.py`
  - `_process_batch()`：延迟重试、缓存 Reuse、降级逻辑。
  - `_process_cache_misses_batch()` / `_process_single_request_fallback()`：成功或失败后的资源回收。
  - `EmbeddingMemoryManager`：显存预算的核心实现。

- `sglang/python/sglang/srt/managers/vit_scheduler_client.py`
  - 移除了 client 端的“pending 超限直接拒绝”逻辑，所有限流交由调度器处理。

## 验证建议

1. 设置较低的显存上限和并发（例如 `SGLANG_VIT_CACHE_GPU_GB=3`, `SGLANG_VIT_MAX_INFLIGHT=1`），发送多个图像请求，确认不会出现 `mat2 must be divisible by 16` 崩溃，并且请求会按顺序逐个完成。
2. 查看日志：
   - `Queue not empty ... next check in XXXms` 应与 backoff 相关，重试不会密集刷屏。
   - `Request X hit defer limit ... forcing fallback compute` 仅在极端压力下出现，且请求能完成。
3. 当客户端成功消费 embedding 并发送 `free_embedding` 信号后，看 `total_usage` 是否下降，确保显存额度被归还。

## 后续方向

- 根据真实 workload 校准估算系数，或引入自适应统计（基于 rolling metrics 自动调整）。
- 将 fallback 结果的缓存策略可配置（例如允许“只缓存最后 N 个请求”的 LRU）。
- 引入 Prometheus 指标：`vit_memory_usage_bytes`, `vit_pending_queue_length`, `vit_defer_count`，便于线上监控。
