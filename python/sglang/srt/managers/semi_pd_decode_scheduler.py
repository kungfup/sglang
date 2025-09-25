# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A scheduler that manages a tensor parallel GPU worker."""

import logging
from collections import deque
import threading
import time
from types import SimpleNamespace
from typing import List, Optional
import os
import numpy as np
import torch
import torch.distributed as dist
import zmq

from sglang.semi_pd.utils import InstanceRole
from sglang.srt.managers.io_struct import (
    BatchProcessPrefillResultReq,
    GetNextPrefillBatchInput,
    GetNextPrefillBatchOutput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.semi_pd_scheduler import SemiPDScheduler
from sglang.srt.server_args import PortArgs, SemiPDPortArgs, ServerArgs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import (
    broadcast_pyobj,
    get_bool_env_var,
    get_zmq_socket,
    point_to_point_pyobj,
    semi_pd_log_info_throttle,
    semi_pd_log_every,
)
from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
from sglang.srt.managers.copy_audit import CopyAudit


logger = logging.getLogger(__name__)


# Test retract decode for debugging purposes
TEST_RETRACT = get_bool_env_var("SGLANG_TEST_RETRACT")


class SemiPDDecodeScheduler(SemiPDScheduler):
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,  # 🔧 添加pp_rank参数
        dp_rank: Optional[int],
        bypass_load_weight: bool = False,
    ):
        super().__init__(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            pp_rank,  # 🔧 传递pp_rank
            dp_rank,
            False,
            InstanceRole.DECODE,
        )

        self.pp_rank = pp_rank  # 🔧 保存pp_rank

                # 🔧 从环境变量获取pp_size
        self.pp_size = int(os.environ.get("SGLANG_PP_SIZE", 1))  # 🔧 保存pp_size

        # 🔧 PP模式下的DECODE进程间通信初始化
        # 注意：不要覆盖父类设置的self.pp_group (GroupCoordinator对象)
        # self.pp_group 已经在父类中通过 get_pp_group() 正确设置
        if self.pp_size > 1:
            # SGLang原生PP组已经在父类中初始化，直接使用
            if hasattr(self, 'pp_group') and self.pp_group is not None:
                logger.info(f"🔗 [PP_DECODE] PP{pp_rank}: Using SGLang native PP group with ranks {self.pp_group.ranks}")
                logger.info(f"🔗 [PP_DECODE] PP{pp_rank}: PP group type: {type(self.pp_group)}")
                logger.info(f"🔗 [PP_DECODE] PP{pp_rank}: is_first_rank={self.pp_group.is_first_rank}, is_last_rank={self.pp_group.is_last_rank}")
            else:
                logger.warning(f"⚠️ [PP_DECODE] PP{pp_rank}: No PP group available, PP communication disabled")

        self._request_dispatcher._mapping.extend(
            [
                (GetNextPrefillBatchInput, self.get_next_prefill_batch),
                (BatchProcessPrefillResultReq, self.process_prefill_result),
            ]
        )

        # For requests that has been sent to the prefill scheduler but not yet finished.
        self.scheduled_prefill_batches: List[ScheduleBatch] = []

        # Pending tokens produced by last PP stage PREFILL and handed to DECODE via IPC.
        # They will be sent cross-stage by the native PP event loop when this DECODE
        # stage reaches the last-rank send point, without doing extra GPU compute.
        self._pending_token_ids = deque()
        self._pending_token_logits = deque()
        # Ready-to-run decode batches produced from prefill results via IPC.
        # We fetch from this queue in get_next_batch_to_run to drive event_loop_pp.
        self._ready_decode_batches = deque()

        # Semi-PD D-side queues for candidate/prealloc/authorize stages
        self._spd_candidate_set = set()
        self._spd_prealloc_queue = deque()
        # Lightweight copy audit controls (PP0 DECODE only; enabled via env)
        self._copy_audit_enabled = os.environ.get("SGLANG_COPY_AUDIT", "0").lower() in ("1", "true", "yes")
        self._copy_audit_steps_remaining = int(os.environ.get("SGLANG_COPY_AUDIT_STEPS", "2")) if self._copy_audit_enabled else 0

        self._spd_authorize_outbox = deque()
        self._spd_max_auth_rids = int(os.environ.get("SGLANG_SEMIPD_AUTH_MAX_RIDS", "64"))

        # 🔧 PP stage间通信：DECODE进程需要与下一个stage的DECODE进程通信
        if self.attn_tp_rank == 0:
            context = zmq.Context(2)

            assert isinstance(port_args, SemiPDPortArgs)

            # 🔧 同PP stage内的IPC通信（与PREFILL进程）
            self.bridge_socket = get_zmq_socket(
                context, zmq.PUSH, self.port_args.bridge_ipc_name, False
            )
            logger.info(
                f"[DECODE-PP{pp_rank}] IPC endpoints: bridge(connect)={self.port_args.bridge_ipc_name}, p_scheduler={self.port_args.p_scheduler_input_ipc_name}, d_scheduler(bind)={self.port_args.d_scheduler_input_ipc_name}"
            )
            self.send_to_p_instance = get_zmq_socket(
                context, zmq.PUSH, self.port_args.p_scheduler_input_ipc_name, False
            )
            # 所有PP段的DECODE都需要接收本stage的PREFILL消息（GetNextPrefillBatchInput / BatchProcessPrefillResultReq）
            # For non-PP0 DECODE stages, base Scheduler does not bind
            # d_scheduler_input_ipc_name. Create a dedicated PULL to receive
            # PREFILL→DECODE messages on this stage.
            try:
                if getattr(self, 'pp_rank', 0) != 0 or getattr(self, 'recv_from_tokenizer', None) is None:
                    self.recv_from_p_instance = get_zmq_socket(
                        context, zmq.PULL, self.port_args.d_scheduler_input_ipc_name, True
                    )
                else:
                    self.recv_from_p_instance = None
            except Exception:
                self.recv_from_p_instance = None

            # 🔧 PP stage间通信：与下一个stage的DECODE进程通信
            # 使用SGLang原生的NCCL通信机制，不需要额外的socket
            if hasattr(self.port_args, 'next_stage_decode_port') and self.port_args.next_stage_decode_port:
                logger.info(f"🔗 PP{pp_rank} DECODE: 将使用SGLang原生NCCL与下一个stage的DECODE进程通信")
            else:
                logger.info(f"🔗 PP{pp_rank} DECODE: 这是最后一个stage，无需连接下一个stage")
            # Lightweight bootstrap server per PP stage (health + endpoint alignment)
            try:
                self._bootstrap_port = self.server_args.disaggregation_bootstrap_port + pp_rank
                self._bootstrap = CommonKVBootstrapServer(self._bootstrap_port)
                logger.info(
                    f"[DECODE-PP{pp_rank}] Bootstrap server started at 127.0.0.1:{self._bootstrap_port}"
                )
            except Exception as e:
                logger.warning(f"[DECODE-PP{pp_rank}] Bootstrap server init failed: {e}")

            # HELLO/ACK handshake to ensure P is ready before flowing candidates
            self._handshake_done = os.environ.get("SGLANG_SEMIPD_DISABLE_HANDSHAKE", "0").lower() in ("1","true","yes")
            def _hello_worker():
                while not self._handshake_done:
                    try:
                        self.bridge_socket.send_pyobj({"type": "HELLO", "pp": pp_rank})
                    except Exception:
                        pass
                    time.sleep(0.5)
            self._hello_thread = threading.Thread(target=_hello_worker, daemon=True)
            self._hello_thread.start()
        else:
            self.bridge_socket = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_p_instance = SimpleNamespace(send_pyobj=lambda x: None)
            self._handshake_done = True

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """
        Semi-PD changes:
          - add the retracted requests to the prefill scheduler
          - add EOS token detection for decode phase
        """
        initial_bs = batch.batch_size()

        # Semi-PD: Remove extra EOS detection from update_running_batch
        # Let process_batch_result_decode handle EOS detection properly like original SGLang

        # If this micro-batch is still in EXTEND (first pass on non-first PP stage),
        # keep it as EXTEND until one pass completes, then switch to DECODE.
        if batch.forward_mode is not None and batch.forward_mode.is_extend():
            if not getattr(batch, "first_extend_done", False):
                return batch
            # EXTEND finished once; now switch to DECODE
            batch.forward_mode = ForwardMode.DECODE

        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # 🔧 CRITICAL: D-Scheduler作为KV Cache唯一管理者，执行动态OOM管理
        # 这是Semi-PD调度机制中最精巧的部分，完全由D-Scheduler主导
        if not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier) or (
            TEST_RETRACT and batch.batch_size() > 10
        ):
            old_ratio = self.new_token_ratio

            # D-Scheduler决定撤回哪些请求
            retracted_reqs, new_token_ratio = batch.retract_decode(self.server_args)
            self.new_token_ratio = new_token_ratio

            logger.info(
                f"[DECODE-PP{self.pp_rank}] 🧠 D-Scheduler: OOM detected, executing request retraction. "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )

            # 🔧 MIGRATION: 原版Semi-PD的请求撤回逻辑
            # 注意：原版Semi-PD在撤回时不需要手动释放资源，retract_decode已经处理了
            for req in retracted_reqs:
                req: Req

                # 重新打包请求并发送给P-Scheduler（携带多模态输入）
                mm_inputs_dict = None
                if getattr(req, "multimodal_inputs", None) is not None:
                    mm = req.multimodal_inputs
                    # Build a minimal dict expected by MultimodalInputs.from_dict
                    mm_inputs_dict = {"mm_items": mm.mm_items}
                    # Optional IDs used by different VLMs
                    for _k in (
                        "image_pad_len",
                        "im_token_id",
                        "im_start_id",
                        "im_end_id",
                        "slice_start_id",
                        "slice_end_id",
                        "audio_start_id",
                        "audio_end_id",
                        "audio_token_id",
                        "video_token_id",
                    ):
                        _v = getattr(mm, _k, None)
                        if _v is not None:
                            mm_inputs_dict[_k] = _v

                message = TokenizedGenerateReqInput(
                    rid=req.rid,
                    input_text=(req.origin_input_text or "") + req.decoded_text,
                    input_ids=req.origin_input_ids + req.output_ids,
                    mm_inputs=mm_inputs_dict,
                    sampling_params=req.sampling_params,
                    return_logprob=req.return_logprob,
                    logprob_start_len=req.extend_logprob_start_len,
                    top_logprobs_num=req.top_logprobs_num,
                    token_ids_logprob=req.token_ids_logprob,
                    stream=req.stream,
                    lora_path=req.lora_path,
                    input_embeds=req.input_embeds,
                    custom_logit_processor=req.custom_logit_processor,
                    return_hidden_states=req.return_hidden_states,
                    is_retracted=True,
                )

                self.waiting_queue.insert(0, req)
                self.send_to_p_instance.send_pyobj(message)
                logger.info(f"[DECODE-PP{self.pp_rank}] 🧠 D-Scheduler: Sent retracted request {req.rid} back to P-Scheduler")
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        # Update batch tensors (now it's safe to run decode)
        batch.prepare_for_decode()
        return batch

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        if not self.running_batch.is_empty():
            self.running_batch = self.update_running_batch(self.running_batch)
            ret = self.running_batch if not self.running_batch.is_empty() else None
        else:
            ret = None

        # Handle DP attention
        if self.server_args.enable_dp_attention:
            ret, _ = self.prepare_dp_attn_batch(ret)

        return ret

    def get_new_batch_prefill(self, rids: Optional[List[str]] = None) -> Optional[ScheduleBatch]:
        """
        Semi-PD changes:
          - keep scheduled prefill batches in scheduled_prefill_batches
          - disable mixed-style chunked prefill
          - skip requests that not in rids
        """
        # Check if the grammar is ready in the grammar queue
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        # Handle the cases where prefill is not allowed
        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)
        if running_bs >= self.max_running_requests:
            self.running_batch.batch_is_full = True
            return None

        if self.enable_hierarchical_cache:
            # check for completion of hierarchical cache activities to release memory
            self.tree_cache.writing_check()
            self.tree_cache.loading_check()

        # Get priority queue
        prefix_computed = self.policy.calc_priority(self.waiting_queue)

        # Prefill policy
        adder = PrefillAdder(
            self.page_size,  # v0.4.8 requires page_size as first parameter
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.lora_paths:
            lora_set = set([req.lora_path for req in self.running_batch.reqs])

        # Get requests from the waiting queue to a new prefill batch
        if get_bool_env_var("SGLANG_LOG_WAITQUEUE", default="0"):
            logger.info(
                f"[DECODE-PP{self.pp_rank}] Processing waiting queue, rids={rids}, waiting_queue_size={len(self.waiting_queue)}"
            )
        for req in self.waiting_queue:
            # Semi-PD
            if rids is not None and req.rid not in rids:
                logger.debug(f"[DECODE-PP{self.pp_rank}] Skipping req.rid={req.rid} (not in rids)")
                continue

            if (
                self.lora_paths
                and len(
                    lora_set
                    | set([req.lora_path for req in adder.can_run_list])
                    | set([req.lora_path])
                )
                > self.max_loras_per_batch
            ):
                self.running_batch.batch_is_full = True
                break

            if running_bs + len(adder.can_run_list) >= self.max_running_requests:
                self.running_batch.batch_is_full = True
                break

            req.init_next_round_input(
                None if prefix_computed else self.tree_cache,
                # v0.4.8 removed enable_hierarchical_cache parameter
            )

            res = adder.add_one_req(
                req, self.chunked_req is not None
                # v0.4.8 removed enable_hierarchical_cache parameter
            )
            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (
                            self.running_batch is not None
                            and not self.running_batch.is_empty()
                        )
                    else:
                        self.running_batch.batch_is_full = True
                break

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None
        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]

        if self.enable_hierarchical_cache:
            self.tree_cache.read_to_load_cache()

        if adder.new_chunked_req is not None:
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req:
            self.chunked_req.is_chunked += 1

        # Print stats
        if self.attn_tp_rank == 0:
            self.log_prefill_stats(adder, can_run_list, running_bs)

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
        )
        new_batch.prepare_for_extend()
        # Semi-PD
        self.scheduled_prefill_batches.append(new_batch)

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
        ):
            # Semi-PD
            raise NotImplementedError(
                "Mixed chunked prefill is not supported in Semi-PD mode"
            )
        else:
            new_batch.decoding_reqs = None

        return new_batch

    def get_next_prefill_batch(self, recv_req: GetNextPrefillBatchInput):
        """
        Handle PREFILL→DECODE prefill-batch authorization.

        Match the working semipd_nopp behavior: reply synchronously with
        GetNextPrefillBatchOutput so PREFILL does not hang waiting for D.

        Notes:
        - This still uses get_new_batch_prefill(...) which appends the created
          ScheduleBatch into self.scheduled_prefill_batches for later
          process_prefill_result handling, keeping internal invariants intact.
        - If no capacity or no matching requests are available, send an empty
          authorization to let PREFILL clear its awaiting flag and retry later.
        """
        # quiet

        # Release unfinished chunk if any, mirroring semipd_nopp behavior
        if self.chunked_req:
            try:
                self.tree_cache.cache_unfinished_req(self.chunked_req)
                self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
            except Exception:
                pass

        # Try to allocate a new prefill batch immediately from the candidate rids
        batch = self.get_new_batch_prefill(recv_req.rids)

        try:
            if batch is None:
                # No capacity now: send empty reply to unblock P
                empty = GetNextPrefillBatchOutput(
                    rids=[], chunked_rid=None, req_pool_indices=[], prefix_lens=[], extend_input_lens=[]
                )
                self.bridge_socket.send_pyobj(empty)
            else:
                approved_rids = [r.rid for r in batch.reqs]
                req_pool_indices = [r.req_pool_idx for r in batch.reqs]
                prefix_lens = [len(r.prefix_indices) for r in batch.reqs]
                extend_input_lens = [r.extend_input_len for r in batch.reqs]

                msg = GetNextPrefillBatchOutput(
                    rids=approved_rids,
                    chunked_rid=(self.chunked_req.rid if self.chunked_req else None),
                    req_pool_indices=req_pool_indices,
                    prefix_lens=prefix_lens,
                    extend_input_lens=extend_input_lens,
                )
                self.bridge_socket.send_pyobj(msg)
        except Exception:
            # Fall back silently; PREFILL will retry via resend logic.
            pass

        # Synchronous path fully handles the authorization; no dispatcher output
        return None

    def _maybe_authorize_prefill(self):
        # First try to send any queued authorizations
        if getattr(self, "_handshake_done", True) and self._spd_authorize_outbox:
            try:
                msg = self._spd_authorize_outbox[0]
                self.bridge_socket.send_pyobj(msg)
                self._spd_authorize_outbox.popleft()
                semi_pd_log_info_throttle(
                    logger,
                    key=f"pp{self.pp_rank}.p2d.gnp.send",
                    msg=(
                        f"[DECODE-PP{self.pp_rank}] →P GetNextPrefillBatchOutput: #rids={len(msg.rids)}"
                    ),
                )
            except Exception:
                pass
        # Then build new authorization
        if os.environ.get("SGLANG_SEMIPD_D_IGNORE_CANDIDATES", "0").lower() in ("1","true","yes"):
            # Ignore candidates; authorize from D waiting queue directly
            if self.chunked_req:
                self.tree_cache.cache_unfinished_req(self.chunked_req)
                self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
            batch = self.get_new_batch_prefill(None)
            if batch is None:
                # nudge P
                self._spd_authorize_outbox.append(GetNextPrefillBatchOutput(rids=[], chunked_rid=None, req_pool_indices=[], prefix_lens=[], extend_input_lens=[]))
                return
            approved_rids = [r.rid for r in batch.reqs]
            req_pool_indices = [r.req_pool_idx for r in batch.reqs]
            prefix_lens = [len(r.prefix_indices) for r in batch.reqs]
            extend_input_lens = [r.extend_input_len for r in batch.reqs]
            self._spd_authorize_outbox.append(GetNextPrefillBatchOutput(
                rids=approved_rids,
                chunked_rid=(self.chunked_req.rid if self.chunked_req else None),
                req_pool_indices=req_pool_indices,
                prefix_lens=prefix_lens,
                extend_input_lens=extend_input_lens,
            ))
            return
        # candidate-driven path
        if not self._spd_prealloc_queue:
            return
        # Release unfinished chunk if any
        if self.chunked_req:
            self.tree_cache.cache_unfinished_req(self.chunked_req)
            self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
        # Take a slice of candidates
        slice_rids = []
        while self._spd_prealloc_queue and len(slice_rids) < self._spd_max_auth_rids:
            slice_rids.append(self._spd_prealloc_queue.popleft())
        batch = self.get_new_batch_prefill(slice_rids)
        if batch is None:
            # No capacity now: push back candidates and send an empty auth to let P clear awaiting flag
            for rid in reversed(slice_rids):
                self._spd_prealloc_queue.appendleft(rid)
            empty = GetNextPrefillBatchOutput(
                rids=[], chunked_rid=None, req_pool_indices=[], prefix_lens=[], extend_input_lens=[]
            )
            self._spd_authorize_outbox.append(empty)
            return
        approved_rids = [r.rid for r in batch.reqs]
        for rid in approved_rids:
            self._spd_candidate_set.discard(rid)
        req_pool_indices = [r.req_pool_idx for r in batch.reqs]
        prefix_lens = [len(r.prefix_indices) for r in batch.reqs]
        extend_input_lens = [r.extend_input_len for r in batch.reqs]
        semi_pd_log_info_throttle(
            logger,
            key=f"pp{self.pp_rank}.p2d.alloc",
            msg=(
                f"[DECODE-PP{self.pp_rank}] 🧠 Allocated: #rids={len(approved_rids)}, "
                f"indices={req_pool_indices}, prefix={prefix_lens}, extend={extend_input_lens}"
            ),
        )
        msg = GetNextPrefillBatchOutput(
            rids=approved_rids,
            chunked_rid=(self.chunked_req.rid if self.chunked_req else None),
            req_pool_indices=req_pool_indices,
            prefix_lens=prefix_lens,
            extend_input_lens=extend_input_lens,
        )
        self._spd_authorize_outbox.append(msg)

    def process_prefill_result(self, recv_req: BatchProcessPrefillResultReq):
        """
        PREFILL→DECODE（同 stage）最小交接：
        - 不做跨 stage 的 NCCL 发送
        - 仅将 next_token_ids 合并回本地 decode 队列
        - 由原生 event_loop_pp 统一进行 GPU/PP 的 send/recv
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        import numpy as np
        import torch

        num_tokens = len(recv_req.next_token_ids) if recv_req.next_token_ids else 0
        logger.info(f"[DECODE-PP{getattr(self,'pp_rank','?')}] ←P tokens: {num_tokens}")

        batch = self.scheduled_prefill_batches.pop(0)

        logits_processor_output = None
        if recv_req.next_token_logits is not None:
            logits_processor_output = LogitsProcessorOutput(
                next_token_logits=torch.from_numpy(recv_req.next_token_logits).to(
                    self.device, dtype=torch.float16, non_blocking=True
                ),
                hidden_states=None,
            )

        result = GenerationBatchResult(
            next_token_ids=recv_req.next_token_ids,
            logits_output=logits_processor_output,
            pp_hidden_states_proxy_tensors=None,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            bid=-1,
            can_run_cuda_graph=False,
        )

        if recv_req.next_token_ids:
            # Build tensor directly from list to avoid extra numpy hop
            batch.output_ids = torch.tensor(
                recv_req.next_token_ids, device=self.device, dtype=torch.int64
            )

        self.process_batch_result_prefill(batch, result)
        batch.filter_batch(chunked_req_to_exclude=self.chunked_req)
        # Instead of merging into running_batch (which gets overwritten by event_loop_pp's slot),
        # enqueue this batch so get_next_batch_to_run can surface it for the current microbatch.
        if not batch.is_empty():
            self._ready_decode_batches.append(batch)
            # Minimal signal to confirm queueing
            logger.info(f"[DECODE-PP{getattr(self,'pp_rank','?')}] queued decode batch")

        # Semi-PD: 在最后一个PP段，记录token，交由event_loop_pp的“最后段发送”统一回传给PP0。
        if (
            getattr(self.server_args, 'enable_semi_pd', False)
            and hasattr(self, 'pp_rank')
            and hasattr(self, 'pp_size')
            and self.pp_rank == self.pp_size - 1
        ):
            self._pending_token_ids.append(recv_req.next_token_ids or [])
            self._pending_token_logits.append(recv_req.next_token_logits)

    def run_batch(self, batch: ScheduleBatch):
        """Override: for last PP stage DECODE in Semi-PD, reuse pending tokens.

        This avoids re-running GPU decode at the last stage. The native event loop
        will then send these tokens via PP group to stage 0 as usual.
        """
        try:
            if (
                self.is_generation
                and getattr(self.server_args, 'enable_semi_pd', False)
                and self.pp_group is not None
                and self.pp_group.is_last_rank
                and len(self._pending_token_ids) > 0
            ):
                from sglang.srt.managers.scheduler import GenerationBatchResult
                from sglang.srt.layers.logits_processor import LogitsProcessorOutput
                import numpy as np
                import torch

                next_token_ids = self._pending_token_ids.popleft()
                next_token_logits = self._pending_token_logits.popleft()

                logits_processor_output = None
                if next_token_logits is not None:
                    logits_processor_output = LogitsProcessorOutput(
                        next_token_logits=torch.from_numpy(next_token_logits).to(
                            self.device, dtype=torch.float16, non_blocking=True
                        ),
                        hidden_states=None,
                    )

                if next_token_ids:
                    batch.output_ids = torch.tensor(
                        next_token_ids, device=self.device, dtype=torch.int64
                    )

                return GenerationBatchResult(
                    logits_output=logits_processor_output,
                    pp_hidden_states_proxy_tensors=None,
                    next_token_ids=next_token_ids,
                    extend_input_len_per_req=None,
                    extend_logprob_start_len_per_req=None,
                    bid=-1,
                    can_run_cuda_graph=False,
                )
        except Exception:
            logger.exception("[PP_DECODE] pending-token fastpath failed; fallback to default run")
        # fallback to the default run
        if (
            getattr(self, "pp_rank", 0) == 0
            and getattr(self, "_copy_audit_enabled", False)
            and getattr(self, "_copy_audit_steps_remaining", 0) > 0
        ):
            # Only audit a few decode batches on PP0 to keep overhead small
            with CopyAudit(scope=f"DECODE-PP{self.pp_rank}", log_fn=logger.info):
                result = super().run_batch(batch)
            CopyAudit.dump_summary(top_k=10, log_fn=logger.info)
            self._copy_audit_steps_remaining -= 1
            if self._copy_audit_steps_remaining <= 0:
                CopyAudit.reset()
            return result
        else:
            return super().run_batch(batch)

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        # D-driven: proactively authorize prefill when candidates exist
        self._maybe_authorize_prefill()
        # If event_loop_pp reset running_batch to an empty slot, but we have a
        # ready decode batch prepared by process_prefill_result, move it in.
        if getattr(self, "_ready_decode_batches", None) and self.running_batch.is_empty():
            if len(self._ready_decode_batches) > 0:
                self.running_batch = self._ready_decode_batches.popleft()
        return super().get_next_batch_to_run()

    def recv_requests(self):
        """Extend parent's recv to also poll local PULL from PREFILL on non-PP0 stages."""
        recv_reqs = super().recv_requests()
        # Forward newly arrived generate requests to same-stage PREFILL to ensure its waiting_queue is populated
        try:
            if self.attn_tp_rank == 0 and hasattr(self, 'send_to_p_instance') and self.send_to_p_instance is not None:
                from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
                fwd_ct = 0
                for _obj in list(recv_reqs or []):
                    if isinstance(_obj, TokenizedGenerateReqInput):
                        try:
                            self.send_to_p_instance.send_pyobj(_obj)
                            fwd_ct += 1
                        except Exception:
                            pass
                if fwd_ct:
                    pass
        except Exception:
            pass
        if getattr(self, 'recv_from_p_instance', None) is not None and self.attn_tp_rank == 0:
            while True:
                try:
                    obj = self.recv_from_p_instance.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                if recv_reqs is None:
                    recv_reqs = []
                # Accept only work messages
                if isinstance(obj, (GetNextPrefillBatchInput, BatchProcessPrefillResultReq)):
                    recv_reqs.append(obj)
        return recv_reqs

    def process_input_requests(self, recv_reqs):
        """Filter out Semi-PD control dicts (e.g., HELLO/ACK) before dispatch."""
        try:
            filtered = []
            for obj in (recv_reqs or []):
                if isinstance(obj, dict):
                    if obj.get("type") == "HELLO_ACK":
                        self._handshake_done = True
                        continue
                    # drop other control dicts silently
                    continue
                filtered.append(obj)
            return super().process_input_requests(filtered)
        except Exception:
            return super().process_input_requests(recv_reqs)
        # 其余跨 stage 的张量/令牌交接全部回归 event_loop_pp 处理
