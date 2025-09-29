# Semi‑PD + PP：统一时钟与单点决策中文技术方案（SGLang v0.4.8）

本文档系统阐述在 SGLang v0.4.8 中，将 Semi‑PD（Prefill/Decode 解耦）与 Pipeline Parallel（PP）结合时，如何通过“统一时钟 + 单点决策（DECODE）”避免 NCCL 死锁与请求卡住问题。文档包括原生 PP 事件循环解析、Semi‑PD 偏差根因、详细的落地方案、协议与日志规范、容错与验证流程，以及与代码实现的映射指南。

---

## 0. 术语与缩写
- P：PREFILL 进程（同段，辅助）
- D：DECODE 进程（同段，主）
- PP：Pipeline Parallel，通常 pp_size=2
- TP：Tensor Parallel
- StepTag：控制面（IPC）微步意图消息（观测为主，非强制）
- AUTH：授权消息（GetNextPrefillBatchOutput），D→P 明确允许哪些请求进入 EXTEND

---

## 1. 原生 SGLang PP 事件循环如何“天然对齐”
原生每个 PP 段只有一个 scheduler；它在一个确定的循环内推进微批（micro‑batch），从而保证跨段 send/recv 调用顺序一致，避免单边等待。

核心特征：
- 一个调度者 per stage（没有 P/D 分裂）。
- 有 batch 才会触发相应的 recv/send；无 batch 不触发，从而不破坏配对。
- 模型前向封装了固定的收发位置：
  - 非首段：先 recv（上一段产出）再 forward；
  - 非末段：forward 后 send（给下一段）；
  - 末段：forward + sample；将 token 送回 PP0。

这意味着在同一“步”上，各段调用 NCCL 的位置与方向一致，配对天然成立。

---

## 2. Semi‑PD 解耦后为何容易卡住
将一个段内的单一 scheduler 拆成 P/D 两个进程，如果两者都在“作决策/调度”，会出现：
- P/D 各自进入不同阶段，导致 PP 调用不配对（比如 P 先 recv 等待，但对端 P 尚未 send）。
- D 还未下发本段的微步意图，P 就自行 get_next_batch_to_run 或发起 PP 调用，时序漂移。
- EXTEND 与 DECODE 的 PP 调用被 P 与 D 混用，或者双边重复，导致串台。
- Idle/noop 未传播，某段无 batch 但另一段仍尝试 PP 调用，单边等待。

根本原因：缺乏“统一的时钟/订单”和“只在一个地方做决策”的机制。

---

## 3. 目标架构与职责（pp_size=2 典型）
- 每个 PP 段两个进程：
  - DECODE（主）：唯一调度者；接收请求、队列与 KV 资源管理、授权 PREFILL、对外返回、参与 D↔D PP。
  - PREFILL（辅）：严格命令驱动；只在授权后执行 EXTEND；在最后一段执行首步前向+采样，并通过 IPC 将 token 回 D；参与 P↔P PP（仅 EXTEND 数据）。
- 段间通信遵循原生规则：
  - EXTEND（prompt/隐藏态）：仅 P↔P（PP0‑P→PP1‑P）。
  - 自回归 token：仅 D↔D（PP1‑D→PP0‑D）。

---

## 4. 统一时钟与单点决策（本方案核心）
- 单点决策：只有 DECODE 调用 `get_next_batch_to_run` 并决定本轮微步（Tick）的意图；PREFILL 不接触请求队列、不自调度。
- 统一时钟：各段 DECODE 都运行与原生一致的 event_loop_pp“骨架”，按 mb_id 顺序推进；P 严格跟随同段 D 的指令。
- 控制面 StepTag：D→P 的轻量“微步意图”消息（phase、mb_id 等），用于观测对齐；实际权限由授权消息（AUTH）决定。

### Tick 内子阶段（pp_size=2）
1) EXTEND 子阶段（P↔P）：
   - PP0‑D 下发对 PP0‑P 的 EXTEND；PP0‑P forward，并通过 PP send 到 PP1‑P；
   - PP1‑D 同步下发对 PP1‑P 的 EXTEND+PRIME；PP1‑P PP recv→forward→sample 首 token→IPC 回给 PP1‑D。
2) DECODE 子阶段（D↔D）：
   - PP1‑D 将 next_token_ids 通过原生 PP 送回 PP0‑D；PP0‑D 继续 AR 解码；
   - 未有 batch 时：D 下发 PREFILL_IDLE，P 快速 ACK，双方都不调用 NCCL。

---

### 4.1 授权矩阵（必须遵守）
- PP0（首段）：允许“同步授权”路径
  - PREFILL-PP0 → DECODE-PP0：GetNextPrefillBatchInput（候选）
  - DECODE-PP0 → PREFILL-PP0：GetNextPrefillBatchOutput（授权）
- PP>0（非首段）：仅允许“异步授权”路径
  - DECODE-PPk（k>0）→ PREFILL-PPk：通过 p_scheduler_input 异步发送 GetNextPrefillBatchOutput
  - PREFILL-PPk 禁止同步“拉取授权”（若误触则收到#rids=0 的空授权用于清等待）

实现要点（与代码一致）：
- DECODE 是唯一决策者与 KV Cache 资源分配者；P 只按授权执行。
- 空授权用于“推进时钟”：当 D 暂无容量时，尽快发送 #rids=0，解除 P 的等待，避免单边阻塞。
- 数据面对称：
  - EXTEND 仅 P↔P；PP>0 的 P 在被授权后进入 run_batch，并在进入模型前向阶段由原生 PP 事件循环确保 send/recv 配对。
  - 自回归 token 仅 D↔D；末段 D 统一把 token 回送到 PP0。


## 5. 段内 IPC 协议（ZMQ）
- D→P：CommandQ（控制面 + 授权）
- P→D：ReplyQ（ACK/RESULT）

消息结构：
- StepTag（控制面，可选）：`{ mb_id, phase∈{EXTEND,PRIME_DECODE,DECODE_NEXT,PREFILL_IDLE}, pp_rank, req_ids, token_pos }`
- GetNextPrefillBatchInput/Output（工作面）：P 提议候选，D 返回授权（授权列表为空表示 idle/无容量）。
- BatchProcessPrefillResultReq：最后一段 P 将 next_token_ids（+可选 logits）回 D。

超时策略：
- PREFILL 的 `bridge_socket` 设置短 `RCVTIMEO`，并以 NOBLOCK 轮询方式“先清控制面、再等授权”，避免 HELLO/控制消息堵塞工作面。

---

## 6. 通信平面划分（避免串台）
- EXTEND 阶段：仅 P↔P（PP0‑P→PP1‑P）传递隐藏态/残差；D 不参与。
- 自回归阶段：仅 D↔D（PP1‑D→PP0‑D）传递 next_token_ids；P 不参与。
- 保持与原生 PP 的“谁在何时发/收”完全等价，确保配对。

---

## 7. 状态机
### DECODE（每段）
Idle → PlanTick → IssuePrefill → WaitPrefill → ArDecode → WaitDPP → Commit → Idle
- IssuePrefill：向 P 发送 StepTag 与授权（AUTH）。
- WaitPrefill：等待 P ACK/RESULT（最后段收到首 token）。
- ArDecode/WaitDPP：沿用原生 PP 完成 D↔D token 交换。

### PREFILL（每段）
WaitCmd → DoExtend → (P↔P PP) → (Prime+Sample at last) → Reply → WaitCmd
- PP0‑P：EXTEND→PP send→ACK
- PP1‑P：PP recv→EXTEND→首步 sample→RESULT（token）

非法转移（例如 PRIME 无匹配 EXTEND）→ 立即上报 ERROR，并由 D 下发 ABORT_MB 清理该微批。

---

## 8. 协议与数据结构（实现）
- StepTag：`sglang_0.4.8/python/sglang/srt/managers/io_struct.py` 新增（控制面观测）。
- 授权：`GetNextPrefillBatchInput/Output`（已存在，P 提议候选，D 同步返回授权）。
- PREFILL 结果回传：`BatchProcessPrefillResultReq`（最后段把 token 回 D）。

---

## 9. 日志规范（强烈建议开启）
- 控制面（IPC）：
  - D→P 授权前置：
    - `[IPC][role=D→P][pp_rank=k][mb_id=m][phase=EXTEND|PRIME_DECODE] SEND AUTH_BEGIN`
  - P 收到 StepTag：
    - `[IPC][role=D→P][pp_rank=k][mb_id=m][phase=...] RECV STEP`
  - P→D 回 token（最后段）：
    - `[PREFILL-PPk] →D tokens: N`
  - D 收到 token（最后段）：
    - `[DECODE-PPk] ←P tokens: N`
- 数据面（PP）：
  - 可选后续增强，在 tp_worker/model_runner 周边对 P↔P EXTEND 与 D↔D token 的 send/recv 打 BEGIN/END，便于对称性核对。

四元组（phase/actor/pp_rank/mb_id）一眼定位谁先卡住。

---

## 10. 死锁保险与背压
- PREFILL `RCVTIMEO` + 短轮询：优先清控制面，避免 HELLO/心跳类消息导致“误判超时/阻塞”。
- 空授权：当容量不足时，D 发送空授权（#rids=0），P 立即清空等待状态并 idle；绝不触发 PP 单边调用。
- 自然背压：D 在收到 P 的 ACK/RESULT 前不下发下一步指令。
- 时钟漂移保险：若检测到不一致的 phase/顺序，不进入 NCCL，直接上报并 ABORT_MB。

---

## 11. 与代码实现的映射
- 新增控制面类型：
  - `io_struct.py`：新增 `StepTag`。
- DECODE（唯一调度者）：
  - `semi_pd_decode_scheduler.py`：
    - 向 P 下发 StepTag（phase=EXTEND for PP0；phase=PRIME_DECODE for PP_last）与同步授权（GetNextPrefillBatchOutput）。
    - 末段接收 P 回 token，走 pending‑token fastpath：避免在末段重复 GPU decode；由原生 PP 统一把 token 回送 PP0。
    - 将新到的 generate 请求转发给同段 PREFILL，确保其 waiting_queue 不空（PP>0）。
- PREFILL（命令驱动）：
  - `semi_pd_prefill_scheduler.py`：
    - 预先 drain bridge（HELLO/StepTag/授权），只用授权构建 EXTEND 批次。
    - 末段将 `BatchProcessPrefillResultReq`（token/logits）回给 D。

环境变量：
- `SGLANG_SEMIPD_TRACE=1` 开启控制面细粒度日志。
- `SEMI_PD_P2D_REQ_TIMEOUT_MS`（默认 200）：P→D 桥接接收超时。

---

## 12. 验证与上线流程
1) 干跑控制面：
   - 按平时方式启动 Semi‑PD + PP；
   - 设置 `SGLANG_SEMIPD_TRACE=1`；
   - 发最小请求，看 D→P StepTag/AUTH 与 P 的 RECV 是否对称；无授权时 P 是否 idle；无 PP 调用。
2) 小请求功能验证：
   - 单请求短 prompt；确认完整返回，无卡顿；
   - 日志包含 `[PREFILL-PP1] →D tokens: N`、`[DECODE-PP1] ←P tokens: N` 等。
3) 并发与稳定性：
   - 多请求并发；检查 StepTag/授权序列是否有序，是否出现 PP stall。
4) 故障演练：
   - 故意延迟 PP1‑P；确认 D 的超时/空授权能避免死锁，其他微批可继续。

---

## 13. 排障清单（Checklist）
- P 一直等待授权：
  - 检查 D 是否向 P 发送了非空/空授权；当容量不足，P 应收到空授权从而 idle。
- NCCL 卡住：
  - 确认 EXTEND 的 PP 调用只发生在 P；D 不参与；
  - 确认自回归 token 的 PP 调用只发生在 D；P 不参与；
  - 核对两个阶段的 StepTag/授权在两个 stage 上是否对称（pp_rank=0/1）。
- 无法早停/文本异常：
  - 核对末段 P 采样的 token 是否经 IPC 回 D，并由 D（末段）交给原生 PP 送回 PP0；
  - 检查 EOS 处理与 logits/ids 有效性（越界 id 会导致 detokenizer 异常）。

---

## 14. 性能与扩展建议
- 在稳定后引入“tick id”广播（控制面）进一步强化时钟一致性。
- 在 `tp_worker`/`model_runner` 周边为 P↔P 和 D↔D 的 PP 调用添加 BEGIN/END 打点（默认关闭）。
- 结合 `chunked_prefill_size` 与滑动窗口微批，提升并行度与吞吐。

---

## 15. 为什么这套方案有效
- 单点决策：只有 DECODE 拥有 `get_next_batch_to_run`，并授权 PREFILL；P 不自调度。
- 统一时钟：DECODE 的 event_loop_pp 与原生同构；P 完全跟随 StepTag/AUTH → 不会相位漂移。
- 固定通信边界：EXTEND 仅 P↔P，自回归仅 D↔D → 与原生等价的位置，保证 send/recv 配对。
- 背压与安全阀：有界 IPC、短超时、空授权、禁止单边 NCCL 调用 → 即使异常也不扩散为全局死锁。

---

## 16. 附：关键代码位置（便于审阅）
- `sglang_0.4.8/python/sglang/srt/managers/io_struct.py`：新增 `StepTag`。
- `sglang_0.4.8/python/sglang/srt/managers/semi_pd_decode_scheduler.py`：
  - D→P 下发 StepTag 与授权、末段 pending‑token fastpath、PP>0 请求前推给 P。
- `sglang_0.4.8/python/sglang/srt/managers/semi_pd_prefill_scheduler.py`：
  - P 预先 drain 控制面，仅以 AUTH 构建 EXTEND，末段回传 token。

如需，可将本文档合并入 `SEMI_PD_PP_OVERVIEW.md`，并追加 Mermaid 时序图与更细致的日志样例。


---

## 17. 当前实现差距与待办（基于最新 semipd_PP.log 与代码对照）

本节对照日志 @semipd_PP.log 与当前代码实现（半精选摘：`semi_pd_prefill_scheduler.py`、`semi_pd_decode_scheduler.py`、`scheduler.py`），列出尚未完全落地/需收敛的点，并给出明确的实现要求与验证要点。

### 17.1 关键现象（证据）
- 日志片段（PP=2，TP=2）：
  - 已出现（P 收到授权并开始执行）：
    - `[DECODE-PP1] (PP>0) →P GetNextPrefillBatchOutput: #rids=1 via p_scheduler_input`
    - `[PREFILL-PP1] inbox+=auth(#rids=1) [bg]`
    - `[PREFILL-PP1] EXTEND from auth rids=[...] size=1`
    - `[PREFILL-PP1] TRACE run_batch(reqlen=1)`
  - 未出现（P→D 回传与 D 接收）：
    - `[PREFILL-PP1] →D will send prefill_result ...`
    - `[PREFILL-PP1] →D sent prefill_result`
    - `[DECODE-PP1] ZMQ recv BatchProcessPrefillResultReq ...`
    - `[DECODE-PP1] ←P prefill_result ... / ←P tokens: ...`
- 结论：最后一段 PREFILL 已执行 EXTEND，但“同段回传给 DECODE”的手合未达成。可能原因包括：
  1) Scheduler 在 PREFILL 侧未按 Tick 内立即调用 `process_batch_result_prefill`；
  2) `is_last_pp_stage` 判定/分支导致早退；
  3) P→D IPC 端点未对齐（bind/connect/权限、TP0 限定）；
  4) D 侧 `recv_requests()/process_input_requests()/process_prefill_result()` 路径未被触发或早退。

### 17.2 必须补齐/核对的实现点
1) PREFILL 最后一段“手合回传”必须执行（不依赖后续 Tick）
   - 位置：`scheduler.event_loop_pp`
   - 规则：当 `instance_role==PREFILL` 且本 Tick 运行了一个非 idle batch，必须在 run 后“当场调用” `self.process_batch_result(self.cur_batch, result)`（当前代码已这样做，但需确保确实触发，参见 17.3/验证）。
   - 位置：`SemiPDPrefillScheduler.process_batch_result_prefill`
     - 若为最后 PP 段：无条件通过 `send_to_d_instance` 发送 `BatchProcessPrefillResultReq`（即使 `next_token_ids` 为空也要发送，一眼可见“→D will/sent ...”日志）；
     - 若非最后段：不处理 token，只交由 PP 事件循环发送隐藏态代理；当前分支已存在，需验证日志覆盖。

2) “最后段判定”的单一可信来源
   - 需求：以 `pp_rank/pp_size` 为唯一真值，`pp_group.is_last_rank` 仅作辅助观测；若二者冲突，记录 ERROR 并选择前者。
   - 代码处：`scheduler.forward_batch_generation` 与 `process_batch_result_prefill` 中均要统一；并在初始化时打印一次对账日志，避免环境差异导致误判。

3) P→D IPC 端点对齐 + TP0 限定
   - D 侧：`SemiPDDecodeScheduler.__init__` 必须在 `attn_tp_rank==0` 绑定 `d_scheduler_input_ipc_name`（PULL, bind）。
   - P 侧：`SemiPDPrefillScheduler.__init__` 在 `attn_tp_rank==0` 连接 `d_scheduler_input_ipc_name`（PUSH, connect）。
   - 验证：启动时打印端点（已有），并在首个 `BatchProcessPrefillResultReq` 发送/接收处打印节流日志（will/sent / ZMQ recv / ←P prefill_result）。

4) D 侧消费 PREFILL 回传路径必须可见
   - 入口：`SemiPDDecodeScheduler.recv_requests()` 应在 `attn_tp_rank==0` 非阻塞收取 `BatchProcessPrefillResultReq`，并打印节流日志 `ZMQ recv BatchProcessPrefillResultReq`。
   - 处理：`process_prefill_result()` 完成 batch 匹配与 `_ready_decode_batches.append(batch)`；打印 `queued decode batch`，便于确认 D 已接棒。
   - 采样快路径：最后段 D 在 `run_batch()` 复用 `_pending_token_ids`；文档与代码应一致（已实现，需日志确认）。

5) Idle 策略一致性（避免单边 NCCL）
   - 文档策略：P idle 时不制造占位 batch（不贴 PP_RECV）；D 可用 `prepare_for_idle()` 产出空 batch 以维持事件循环结构，但不得触发跨段 NCCL。
   - 代码检查：
     - P：`get_next_batch_to_run()` 在 PP>1 时，未授权返回 `None`（已实现）。
     - D：仅在确需对齐时创建 idle batch，且不会发送张量（当前实现 OK）。

6) 授权路径与节流
   - PP0：仅同步授权（GetNextPrefillBatchInput/Output）。
   - PP>0：仅异步授权（D→P 经 `p_scheduler_input` 发送非空授权）；若 P 误触同步拉取，D 需回空授权以清等待。
   - 节流：空授权需限频，避免日志泛滥与无谓轮询；建议 50–200ms 级别，已通过 `SEMI_PD_P2D_REQ_TIMEOUT_MS` 控制。

7) 跨 PP 的 CPU 平面“白名单转发”
   - 保留：`TokenizedGenerateReqInput` / `GenerateReqInput` / `EmbeddingReqInput` / `TokenizedEmbeddingReqInput` / `BatchMultimodalDecodeReq`。
   - 丢弃：StepTag/HELLO/ACK 等控制面，不跨段传播。
   - 现状：已在 `scheduler.event_loop_pp` 实现；文档化为“强约束”，作为回归测试点。

8) 诊断日志基线（必须）
   - PREFILL：
     - `ENTER process_batch_result_prefill is_extend=...`（节流）
     - `→D will send prefill_result: tokens=... rids=...`（节流）
     - `→D sent prefill_result`（节流）
   - DECODE：
     - `ZMQ recv BatchProcessPrefillResultReq rids=[...]`（节流）
     - `←P prefill_result: tokens=... rids=[...]`（节流）
     - `queued decode batch`（一次性）
   - SCHED（P 侧）：
     - `[SCHED-PPk] call process_batch_result for PREFILL (mode=...)`（节流）
   - 判定法：出现 PREFILL 的 will/sent 而 DECODE 无 ZMQ recv → IPC；SCHED 有 call 而 PREFILL 无 ENTER → 分派；P 无 will/sent → 手合未触发。

### 17.3 建议的收敛/重构
- 去线程化的前置排水（推荐）
  - 以“前台短轮询 + NOBLOCK + 超时”的方式处理 HELLO/StepTag/授权，减少后台线程与共享队列的竞态；`semipd_nopp` 参考实现更简单稳健。
- 统一“最后段”的判断与日志
  - 启动时打印一次：`pp_rank/pp_size` 与 `pp_group.is_last_rank` 的对账；冲突则 ERROR 并说明采用前者。
- 将“空授权”与“idle”策略文档化为强约束
  - 明确空授权频率上限、P/D 双方 idle 行为，纳入回归 checklist。

### 17.4 面向实现者的最小变更清单（PR Checklist）
1) 在 `scheduler.event_loop_pp` 的 PREFILL 分支，保持“run 后当场调用 process_batch_result”，并在调用前打印节流日志（已有，需验证出现）。
2) 在 `SemiPDPrefillScheduler.process_batch_result_prefill`：
   - 统一最后段判定，出现则无条件发送 `BatchProcessPrefillResultReq`（即使 token=0）。
   - 增强 will/sent 节流日志（已有）。
3) 在 `SemiPDDecodeScheduler.__init__`/`recv_requests`：确保本段 D 绑定/接收 `d_scheduler_input_ipc_name` 并打印 `ZMQ recv`（已实现，需验证日志出现）。
4) 在 `process_prefill_result`：匹配/入队 `_ready_decode_batches` 后打印 `queued decode batch`（已实现，需验证出现）。
5) 在 PREFILL `get_next_batch_to_run`：PP>1 未授权返回 `None`；删除/关闭任何会制造占位 batch 的路径（已实现）。
6) 将“白名单跨段”与“空授权节流”写入单测/集成用例的断言条件。

---

## 18. 验证步骤（针对本轮问题的精准复现）
1) 启动 Semi‑PD + PP（pp=2, tp=2），设置 `SGLANG_SEMIPD_TRACE=1`；发最小请求。
2) 期望日志顺序（PP1 为例）：
   - D：`(PP>0) →P GetNextPrefillBatchOutput ...`
   - P：`inbox+=auth ...` → `EXTEND from auth ...` → `TRACE run_batch ...`
   - SCHED（P）：`call process_batch_result for PREFILL (mode=EXTEND)`
   - P：`→D will send prefill_result ...` → `→D sent prefill_result`
   - D：`ZMQ recv BatchProcessPrefillResultReq ...` → `←P prefill_result ...` → `queued decode batch`
3) 若断在“P 已 run 但无 will/sent”：进入 17.2-1/2 排查；若断在“D 无 ZMQ recv”：进入 17.2-3/4 排查；若出现 idle 单边 NCCL：进入 17.2-5 排查。

以上条目已与当前代码逐一对齐，后续 PR 请引用本节编号逐项勾选。
