# SemiPD 多模态输出修复与改动说明

本文件整理了近期在 `semipd_tp_nopp` 分支中为解决多模态输出异常所做的全部代码修改、原因、可能的性能影响以及验证方法，方便后续复盘与维护。

---

## 1. 背景与目标

- **问题现象**  
  - 多模态请求在首批响应正常后，后续出现重复段落、模板提示词和大量 `"A "` 等噪声。  
  - 同一请求的 detokenizer 收到重复 token 切片，导致输出被多次拼接。
- **根本原因**  
  1. SemiPD 分支强行开启“增量 detokenizer”，但请求被撤回或状态重置后，`last_full_decode_len` 等基准未同步，offset 错位。  
  2. 为规避错位我们曾跳过 detokenizer，结果 prompt（含大量测试用 `"A "`）被原样 decode，输出仍受污染。  
  3. 环境变量残留，导致部分实例继续走增量协议，与新的修复逻辑冲突。
- **目标**  
  - 回到与原生 sglang 一致的策略：使用 **全量 detokenizer** 裁剪 prompt，仅输出新增 completion。  
  - 保留必要的调试日志，方便确认发送 token 的协议是否正确。  
  - 尽量降低对性能的影响并提供可选开关，以便进行纯性能压测。

---

## 2. 关键改动一览

| 文件 | 主要修改 | 目的 |
|------|----------|------|
| `python/sglang/srt/managers/schedule_batch.py` | `reset_for_retract()` 中补充 `self.last_full_decode_len = len(self.origin_input_ids_unpadded)` | 确保撤回后的请求重置流式基准，避免 offset 混乱 |
| `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | 多处调整 detokenizer 协议，默认采用 **full 模式**；对多模态不再依赖 `send_decode_id_offset` 的增量切片；禁用多模态流式输出 | 始终向 detokenizer 发送完整 `origin_input_ids_unpadded + output_ids`，由 `read_offset` 裁剪 prompt，避免流式导致的重复 |
| `python/sglang/srt/managers/semi_pd_scheduler.py` | 启动阶段强制 `SGLANG_MM_DETOKENIZER_MODE=full`（无条件覆盖旧值） | 防止环境变量残留导致 decode 端继续走增量模式 |
| `python/sglang/srt/managers/detokenizer_manager.py` | 默认读取 `SGLANG_MM_DETOKENIZER_MODE=full` | 与 decode 端保持一致，确保 detokenizer 裁剪逻辑正确 |
| `BUG_ANALYSIS_REPORT.md` | 追加 detokenizer 模式调整记录与验证建议 | 保留排查过程和操作指南 |

> ⚠️ 说明：原先自动设置 `incremental` 的逻辑已移除，如需重新做增量实验，可手动 `export SGLANG_MM_DETOKENIZER_MODE=incremental`，但需自行承担输出重复风险。

---

## 3. 行为对齐与原因说明

1. **撤回请求的状态重置**  
   - SemiPD 会在 OOM 或调度需要时撤回请求再加入 Prefill，若 `last_full_decode_len` 未重置，后续 full 模式会把旧的 offset 当作新请求的基准，导致 detokenizer 误判。  
   - 在 SemiPD decode 路径的 `reset_for_retract()` 中补齐该字段后，撤回请求再次进入 decode 时，detokenizer 的 baseline 与 prompt 长度一致。

2. **多模态始终使用全量 detokenizer**  
   - 原生 sglang 并未针对多模态启用增量流式，而是将整段 token 交给 detokenizer 剪裁，再一次性返回文本。  
   - 当前实现中，我们保持单次 send（多模态默认 `stream=False`），但将发送内容从“增量切片”改为“完整序列”，使 detokenizer 可以利用 `read_offset` 去除 prompt 和模板，输出仅包含新增 completion。

3. **环境变量强制同步**  
   - 多轮测试发现 decode 进程常因旧环境变量继续使用 `incremental` 模式，即便代码已更新。  
   - 通过在 scheduler 初始化阶段直接设定 `os.environ["SGLANG_MM_DETOKENIZER_MODE"] = "full"`，确保 Prefill/Decode 进程重启后即使用新协议。

4. **删除多模态流式输出逻辑**  
   - 现阶段业务不需要多模态 `stream=True`，我们保留强制 `should_output` 在完成时触发的行为，并通过全量 detokenizer 处理文本；必要时仍可开启增量模式自行验证。

---

## 4. 性能影响与评估

| 影响项 | 描述 | 预期影响 |
|--------|------|----------|
| Python 层拷贝 | full 模式需构造完整 token 序列（一次性 8k+ token） | 单次请求额外数十 KB，通常只在完成时进行一次，影响较小 |
| IPC 传输负载 | 传输数据从 128 token 增至数千 token | 消息包更大，提升 CPU 与 ZMQ 压力；QPS 升高时吞吐会略降，需要压测确认 |
| detokenizer 状态 | `decode_status` 仍只保留尾部 tokens，内存可控 | 与原生一致，无额外增长 |
| 可选开关 | 仍可通过 `SGLANG_MM_DETOKENIZER_MODE=off` 或 `incremental` 回到旧模式 | 建议仅在纯性能压测时临时使用，质量测试请保持 `full`（默认对多模态关闭流式，结束时统一解码） |

**建议压测手段：**
- 使用 `auto_qps.py` 在 `full` 模式下跑与之前相同的 QPS，统计 token/s 与 CPU 利用率。  
- 若性能仍不满足需求，可在压测场景临时 `export SGLANG_MM_DETOKENIZER_MODE=off`，但输出需忽略。
- 之后评估是否要针对 detokenizer 的拷贝流程做进一步优化（例如使用 numpy buffer）。

---

## 5. 验证步骤

1. **重启服务前确认环境变量**  
   ```bash
   unset SGLANG_MM_DETOKENIZER_MODE  # 或 export SGLANG_MM_DETOKENIZER_MODE=full
   ```
2. **重启 SemiPD Prefill / Decode 实例**  
   - 启动日志应出现：`SGLANG_MM_DETOKENIZER_MODE=full`。  
3. **观察调度日志**  
   - `semipd_tp.log` 中的 `[DBG_SCHEDULER]` send_len 不再固定 133，而是接近完整 prompt 长度。  
4. **验证输出**  
   - `qps_semipd_tp.log` 等响应文件不再出现大量 `"A "` 或模板重复段落。  
5. **性能压测（可选）**  
   - `auto_qps.py --start_qps ...` 对比修改前后吞吐；若回退 `off` 模式，复现旧行为以确认差异。

---

## 6. 后续建议与开关说明

- **质量优先**：保持 `SGLANG_MM_DETOKENIZER_MODE=full`，即可获得与原生 sglang 接近的输出体验。  
- **性能压测**：快速评估吞吐时可设为 `off` 或 `incremental`，但要意识到输出会重复或包含 prompt，勿用于对外服务。  
- **增量流式研究**：若未来确有流式需求，可在 full 模式稳定的前提下，仔细评估并改造 `init_incremental_detokenize()` 与 detokenizer 的 offset 处理，再开启 `incremental` 进行灰度实验。  
- **代码维护**：后续若合并 upstream，需关注原生仓库在 detokenizer 协议上的更新，保持逻辑一致。

---

## 7. 相关文件一览

- `python/sglang/srt/managers/schedule_batch.py`  
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`  
- `python/sglang/srt/managers/semi_pd_scheduler.py`  
- `python/sglang/srt/managers/detokenizer_manager.py`  
- `BUG_ANALYSIS_REPORT.md`（排查日志与修复过程）  
- `README_SEMIPD_CHANGES.md`（本文件）

---

如需进一步优化或扩展，请结合本 README 中记录的初衷与约束进行评估。欢迎继续补充测试数据与性能结果，以完善整个 SemiPD 多模态工作流。💡
