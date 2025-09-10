### Semi-PD + PP 架构总览（简版）

- 目标
  - **Semi-PD 解耦**：将 PREFILL 与 DECODE 拆分为独立进程，减少空泡、提高利用率。
  - **Pipeline 并行**：当前配置典型为 `pp_size=2`、`tp_size=1`、`dp_size=1`。

- 进程与拓扑（每个 PP Stage 在一块 GPU 上）
  - `DECODE`（主进程）：接收请求、全局协调、返回响应、KV 管理、CUDA Graph 捕获/复用。
  - `PREFILL`（辅助进程）：执行预填充，复用同 Stage `DECODE` 的权重（零拷共享）。
  - 示例（2 路 PP）：
    - GPU0：`PP0-DECODE` ↔ `PP0-PREFILL`
    - GPU1：`PP1-DECODE` ↔ `PP1-PREFILL`
  - 另有独立 `Tokenizer`/`Detokenizer` 进程。

- 通信与权重
  - 同 Stage（GPU 内）：`DECODE` ↔ `PREFILL` 通过 ZMQ IPC；`PREFILL` 使用 `bypass_load_weight=True`，从 `DECODE` 零拷共享权重。
  - 跨 Stage（GPU 间）：通过原生 PP 组的 NCCL 点对点传输中间隐藏态（包含 `hidden_states`、`residual`）。
  - 通信组：初始化 `world`、`pp`、`tp`、`attention_tp` 等分组。

- 启动时序（概要）
  1) 分配 PP 端口、构建通信组；
  2) 启动所有 `DECODE` → 加载各自 PP 段权重 → 捕获 CUDA Graph → 生成 IPC 信息；
  3) 启动所有 `PREFILL` → 通过 IPC 共享权重 → 与 `DECODE` 建立协作通道；
  4) 全部就绪后开始对外服务。

- 数据路径（推理简化流水）
  - 请求 → `PP0-DECODE` 触发 `PP0-PREFILL` 预填充 → 隐藏态经 NCCL 传至 `PP1` →
    `PP1` 首次前向须走 EXTEND（把本段 prompt KV 全写入）→ 其后进入 DECODE 单步生成 → `PP1-DECODE` 产出 `next_token_ids` → 返回客户端。

- 关键运行参数（常见）
  - `disable_overlap_schedule=True`（与 PP 兼容性约束）。
  - `chunked_prefill_size`：预填充分块大小（可调，用于平衡通信粒度与计算效率）。
  - CUDA MPS：默认 `DECODE` 100% SM、`PREFILL` 90% SM（可按需收敛/倾斜）。
  - 环境变量：`SGLANG_PP_SIZE`、`SGLANG_PP_RANK`、`CUDA_VISIBLE_DEVICES`。

- 目录与核心位置（参考）
  - 调度与进程：`sglang_0.4.8/python/sglang/srt/managers/semi_pd_scheduler.py`
  - DECODE/PREFILL 调度：`semi_pd_decode_scheduler.py` / `semi_pd_prefill_scheduler.py`
  - PP 前向与收发：`sglang_0.4.8/python/sglang/srt/managers/tp_worker.py`
  - CUDA Graph：`sglang_0.4.8/python/sglang/srt/model_executor/cuda_graph_runner.py`

- 一句话特性总结
  - 同卡 IPC 权重共享、跨卡 NCCL 传递隐藏态；`DECODE` 统一协调，`PREFILL` 辅助加速，遵循原生 PP 分组与约束。

---

