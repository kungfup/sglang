# Semi-PD + PP: Unified Clock and Single-Point Decision Design

This document describes a practical, deadlock-free design for running Semi‑PD (Prefill/Decode disaggregation) together with Pipeline Parallelism (PP) in SGLang v0.4.8. It explains the original SGLang PP event-loop alignment mechanisms, the pitfalls when decoupling Prefill and Decode into separate processes, and the detailed solution we implemented: a unified clock with a single decision point (Decode), control-plane StepTag messages, strict communication-plane partitioning, observability, and deadlock insurance.

The goal is to achieve a truly decoupled P/D architecture that runs correctly under PP (pp_size=2 typical), without request hangs or NCCL deadlocks.

---

## 1. Background: Why requests hung under Semi‑PD + PP

Native SGLang uses a single scheduler per PP stage. That scheduler drives a deterministic micro-batch loop where each stage either sends or receives (never both) at fixed points, so NCCL calls always pair up.

When Semi‑PD splits a stage into two processes (PREFILL=P, DECODE=D) without a single unified “clock/decision maker,” P and D can drift out of step, calling PP send/recv at mismatched moments, leading to stalls.

Symptoms:
- Requests hang; no completion.
- One side waiting on PP recv while the other never reaches the symmetric send.
- Control-plane messages (ZMQ) show one-way traffic without matching acks.

---

## 2. Native SGLang PP event-loop alignment (short overview)

- One scheduler per stage iterates micro-batches mb_id = 0..pp_size-1.
- Per micro-batch:
  - Non-first stage: recv hidden-state proxy, then forward.
  - Non-last stage: forward then send hidden-state proxy.
  - Last stage: forward + sample; send tokens back to stage 0.
- Rule-of-thumb: “Only one place issues recv before forward; only one place issues send after forward.” With one scheduler, the order is inherently aligned across stages.

---

## 3. Semi‑PD deviations that cause stalls

- Two process schedulers (P & D) both making decisions → loss of a single, ordered clock.
- PREFILL calling PP independently (before DECODE authorizes) → PP send/recv mismatch.
- DECODE re-doing work intended for PREFILL (or vice versa) → duplicated responsibilities and timing drift.
- Idle/noop not respected (e.g., one side tries PP while the other is idle) → one-sided waiting.

---

## 4. Target architecture & responsibilities (pp_size=2 typical)

- Per PP stage (on one GPU):
  - DECODE (D, primary): the only decision maker; owns request queue, micro-batch order, KV ownership.
  - PREFILL (P, auxiliary): strictly command-driven; executes EXTEND and, on last stage, first-step forward+sample.
- Cross-stage rules:
  - EXTEND traffic (hidden-state/residual) runs only P↔P (PP0-P → PP1-P). D must not participate.
  - Autoregressive tokens run only D↔D (PP1-D → PP0-D). P must not participate.
- Result: Communication is partitioned to mirror native PP’s fixed locations; only one actor per sub-phase calls PP.

---

## 5. Unified clock & single-point decision

- Each stage’s DECODE runs the native event_loop_pp structure (mb_id ordering). PREFILL never calls `get_next_batch_to_run`.
- DECODE generates micro-step intents (StepTag/authorization) and sends them to same-stage PREFILL over IPC.
- PREFILL executes exactly what DECODE authorized; no autonomous scheduling.

Sub-phases within a tick (conceptual order for pp_size=2):
1) EXTEND sub-phase (P↔P):
   - PP0-D issues EXTEND to PP0-P → PP0-P forwards and PP-send to PP1-P.
   - PP1-D concurrently issues EXTEND+PRIME to PP1-P → PP1-P PP-recv then forward+sample; returns first token to PP1-D via IPC.
2) DECODE sub-phase (D↔D):
   - PP1-D sends next_token_ids to PP0-D via native PP; PP0-D proceeds with decode.

If there is no batch on a stage/tick: DECODE sends PREFILL_IDLE and PREFILL immediately ACKs (no PP calls). Both sides remain aligned.

---

## 6. IPC protocol (intra-stage)

We use ZMQ IPC for intra-stage control. Two queues:
- D→P CommandQ: DECODE sends StepTag & authorization (GetNextPrefillBatchOutput).
- P→D ReplyQ: PREFILL replies ACK/RESULT (BatchProcessPrefillResultReq on last stage).

Control-plane message (optional):
- StepTag: `{ mb_id, phase ∈ {EXTEND, PRIME_DECODE, DECODE_NEXT, PREFILL_IDLE}, pp_rank, req_ids, token_pos }`
  - Purpose: observability (uniform logging, easier triage). Not required for correctness.

Work messages:
- GetNextPrefillBatchInput/Output: controls which rids PREFILL may extend.
- BatchProcessPrefillResultReq: on last stage, PREFILL returns next_token_ids (+optional logits) to its DECODE.

Timeouts:
- PREFILL’s bridge socket uses a short `RCVTIMEO` and drains control messages (HELLO/StepTag) to avoid spurious timeouts.

---

## 7. Communication-plane partitioning (avoid cross-talk)

- EXTEND phase (prompt KV build): only PREFILL participates in P↔P PP send/recv.
- Autoregressive phase: only DECODE participates in D↔D PP token send/recv.
- This mirrors native PP: exactly one side sends/receives at each step, preserving pairing.

---

## 8. State machines

DECODE (per stage):
- Idle → PlanTick → IssuePrefill → WaitPrefill → ArDecode → WaitDPP → Commit → Idle
  - IssuePrefill: send AUTH (GetNextPrefillBatchOutput) and StepTag to same-stage P.
  - WaitPrefill: wait for P’s ACK/RESULT; last stage receives token via IPC.
  - ArDecode/WaitDPP: use native PP for D↔D send/recv of tokens.

PREFILL (per stage):
- WaitCmd → DoExtend → (PP send/recv) → (Prime+Sample at last stage) → Reply → WaitCmd
  - PP0-P: EXTEND then PP-send to PP1-P → ACK_DONE.
  - PP1-P: PP-recv from PP0-P → EXTEND → first-step forward+sample → RESULT to PP1-D.

Invalid transitions (e.g., PRIME without prior EXTEND) → ERROR up to D, which ABORTs the micro-batch.

---

## 9. Scheduling algorithm (pseudocode)

DECODE:
```python
for mb_id in range(pp_size):
    batch = get_next_batch_to_run()  # D only
    phase = 'EXTEND' if pp_rank == 0 else 'PRIME_DECODE'
    send_to_P(StepTag(mb_id, phase, req_ids=batch.rids))
    auth = authorize_prefill(batch.candidate_rids)  # GetNextPrefillBatchOutput
    send_to_P(auth)
    prefill_reply = wait_prefill_reply_or_ack()

    # AR sub-phase (native PP):
    result = decode_step(prime_token=prefill_reply.token if any)
    d_to_d_pp_send_recv(result)
    commit(result)
```

PREFILL:
```python
while True:
    tag_or_auth = recv_from_D()
    if isinstance(tag_or_auth, StepTag):
        continue  # observability only
    if isinstance(tag_or_auth, GetNextPrefillBatchOutput):
        batch = build_extend_batch(tag_or_auth)
        if pp_rank == 0: pp_send_hidden_states(batch)
        if pp_rank == last: pp_recv_then_prime_and_sample(batch)
        reply_to_D(ack_or_tokens)
```

---

## 10. Observability (uniform logging)

Control-plane (IPC):
- `[IPC][role=D→P][pp_rank=k][mb_id=m][phase=EXTEND|PRIME_DECODE] SEND/RECV`
- `[PREFILL-PPk] →D tokens: N` (only last stage)
- `[DECODE-PPk] ←P tokens: N` (only last stage)

Data-plane (PP):
- For future enhancement, add BEGIN/END logs around P↔P EXTEND and D↔D token exchangers in `tp_worker/model_runner` (optional). Current design leverages native PP’s ordering to stay aligned.

These four fields (phase/actor/pp_rank/mb_id) make mismatches obvious.

---

## 11. Deadlock insurance & backpressure

- PREFILL bridge `RCVTIMEO` + short NOBLOCK loops to drain control messages.
- Empty authorization (`#rids=0`) lets PREFILL clear awaiting state and idle without PP calls.
- DECODE only sends the next StepTag/authorization after it receives PREFILL’s reply → natural backpressure.
- “Clock drift” preemption: Do not call NCCL if a matching phase is not observed; report and ABORT micro-batch.

---

## 12. Implementation mapping (what changed)

- `sglang_0.4.8/python/sglang/srt/managers/io_struct.py`
  - Added `StepTag` dataclass (control-plane, optional).

- `sglang_0.4.8/python/sglang/srt/managers/semi_pd_decode_scheduler.py`
  - Decode is the only scheduler making decisions. It:
    - Issues StepTag (phase=EXTEND on PP0; PRIME_DECODE on last stage) and synchronous authorizations (GetNextPrefillBatchOutput) to P.
    - Receives PREFILL tokens on last stage and uses a pending-token fastpath to avoid re-running GPU on last stage; native PP then returns tokens to PP0.
    - Forwards newly arrived generate requests to same-stage PREFILL to ensure its waiting queue is populated (PP>0).

- `sglang_0.4.8/python/sglang/srt/managers/semi_pd_prefill_scheduler.py`
  - PREFILL is command-driven:
    - Pre-drains bridge messages, records StepTag for observability, and only builds EXTEND batches from explicit authorizations.
    - On last stage, returns `BatchProcessPrefillResultReq` with `next_token_ids` (and optional logits) via IPC to its DECODE.

Environment toggles:
- `SGLANG_SEMIPD_TRACE=1` (verbose control-plane logs)
- `SEMI_PD_P2D_REQ_TIMEOUT_MS` (default 200): PREFILL bridge receive timeout

---

## 13. Validation & runbook

1) Dry-run control-plane:
   - Launch Semi‑PD + PP as usual.
   - With `SGLANG_SEMIPD_TRACE=1`, submit a tiny request.
   - Observe symmetric logs: D→P AUTH_BEGIN + P’s RECV, P→D tokens only on last stage, PP0-D continues AR.
2) Small functional test:
   - Single short prompt; verify response completes (no hang).
3) Concurrency:
   - Multiple concurrent requests; check that StepTag/authorization streams remain ordered and no PP stalls appear.
4) Fault-injection (optional):
   - Delay PP1-P processing; confirm DECODE’s timeouts/empty auths avoid deadlock.

---

## 14. Troubleshooting checklist

- PREFILL never leaves awaiting state:
  - Check that DECODE sends non-empty authorization for candidate rids; if capacity is full, P must receive empty auth to idle.
- NCCL stalls:
  - Ensure EXTEND PP calls are only in PREFILL; AR token PP calls only in DECODE.
  - Verify matching phases (EXTEND vs PRIME_DECODE) across stages.
- Endless generation:
  - Verify EOS handling on last stage and that tokens are actually sent D↔D to PP0.

---

## 15. Future enhancements

- Broadcast a lightweight tick id (control-plane) for even stronger clock alignment.
- Add BEGIN/END PP logs around P↔P and D↔D calls deep in `tp_worker`/`model_runner` (off by default).
- Sliding-window micro-batches and chunked_prefill tuning after stability is confirmed.

---

## 16. Why this works

- Single-point decision: only DECODE owns `get_next_batch_to_run` and authorizes EXTEND; PREFILL does not self-schedule.
- Unified clock: DECODE’s event_loop_pp mirrors native PP; PREFILL obeys StepTag/authorization → no phase drift.
- Fixed comm partitioning: EXTEND=P↔P, AR=D↔D → symmetric pairing like native PP.
- Backpressure and safety valves: bounded IPC, short timeouts, empty authorizations, and no unilateral NCCL calls.

