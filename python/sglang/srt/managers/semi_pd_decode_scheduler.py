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
import traceback
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
    StepTag,
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

        # Prefer server_args.pp_size when available; fallback to env if missing
        try:
            self.pp_size = getattr(server_args, "pp_size", self.pp_size)
        except Exception:
            pass

        # 🔧 PP模式下的DECODE进程间通信初始化
        # 注意：不要覆盖父类设置的self.pp_group (GroupCoordinator对象)
        # self.pp_group 已经在父类中通过 get_pp_group() 正确设置
        if self.pp_size > 1:
            # SGLang原生PP组已经在父类中初始化，直接使用
            if hasattr(self, 'pp_group') and self.pp_group is not None:
                logger.info(f"🔗 [PP_DECODE] PP{pp_rank}: Using SGLang native PP group with ranks {self.pp_group.ranks}")
                logger.info(f"🔗 [PP_DECODE] PP{pp_rank}: PP group type: {type(self.pp_group)}")

                # 🔧 CRITICAL: 在Semi-PD+PP模式下，只有last_rank (PP1)负责detokenization和输出
                # 这符合SGLang原生PP设计哲学
                is_last_rank = getattr(self.pp_group, 'is_last_rank', False)
                if is_last_rank:
                    logger.info(f"🎯 [PP_DECODE] PP{pp_rank}: This is LAST_RANK, responsible for:")
                    logger.info(f"   - Receiving hidden states from PP{pp_rank-1}")
                    logger.info(f"   - Generating logits and sampling tokens")
                    logger.info(f"   - Detokenization (via detokenizer process)")
                    logger.info(f"   - Stream output to client")
                    logger.info(f"   - EOS detection and request completion")
                else:
                    logger.info(f"🔗 [PP_DECODE] PP{pp_rank}: This is NOT last_rank, responsible for:")
                    logger.info(f"   - Receiving tokens from tokenizer")
                    logger.info(f"   - Processing hidden states through layers")
                    logger.info(f"   - Sending hidden states to PP{pp_rank+1}")
                    logger.info(f"   - NO detokenization or output")
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
        self._spd_authorize_outbox = deque()
        self._spd_max_auth_rids = int(os.environ.get("SGLANG_SEMIPD_AUTH_MAX_RIDS", "64"))

        # Rate limit for empty authorization nudges (seconds)
        self._last_empty_auth_ts = 0.0

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
            # 🔧 CRITICAL: Delay socket connection until event loop starts
            # Save address and context for later connection
            self._p_sched_addr = self.port_args.p_scheduler_input_ipc_name
            self._zmq_context = context
            self.send_to_p_instance = None  # Will be created in event loop after P_SOCKET_READY
            # 所有PP段的DECODE都需要接收本stage的PREFILL消息（GetNextPrefillBatchInput / BatchProcessPrefillResultReq）
            # For non-PP0 DECODE stages, base Scheduler does not bind
            # d_scheduler_input_ipc_name. Create a dedicated PULL to receive
            # PREFILL→DECODE messages on this stage.
            try:
                # Always bind a PULL socket to receive PREFILL→DECODE work/control on this stage
                self.recv_from_p_instance = get_zmq_socket(
                    context, zmq.PULL, self.port_args.d_scheduler_input_ipc_name, True
                )
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
            self._p_socket_ready = False  # Track if PREFILL socket is ready

    def _create_p_instance_socket(self):
        """Create socket connection to PREFILL after receiving P_SOCKET_READY signal."""
        if self.send_to_p_instance is not None:
            return  # Already created

        try:
            from sglang.srt.utils import get_zmq_socket
            import zmq
            self.send_to_p_instance = get_zmq_socket(
                self._zmq_context, zmq.PUSH, self._p_sched_addr, False
            )
            logger.info(f"[DECODE-PP{self.pp_rank}] Created socket connection to PREFILL: {self._p_sched_addr}")
        except Exception as e:
            logger.error(f"[DECODE-PP{self.pp_rank}] Failed to create socket to PREFILL: {e}")
            # Create a dummy socket to avoid None errors
            from types import SimpleNamespace
            self.send_to_p_instance = SimpleNamespace(send_pyobj=lambda x: None)

    def _wait_for_prefill_socket_ready(self):
        """Wait for PREFILL to signal that its socket is ready to receive messages."""
        if self._p_socket_ready or self._handshake_done:
            return  # Already ready or handshake disabled

        logger.info(f"[DECODE-PP{self.pp_rank}] Waiting for PREFILL socket READY signal...")
        try:
            import time
            timeout = 30.0  # 30 seconds timeout
            start_time = time.time()
            while not self._p_socket_ready and (time.time() - start_time) < timeout:
                try:
                    # 🔧 CRITICAL: P_SOCKET_READY comes from recv_from_p_instance (PULL socket)
                    # NOT from bridge_socket (PUSH socket)
                    if self.recv_from_p_instance:
                        msg = self.recv_from_p_instance.recv_pyobj(zmq.NOBLOCK)
                        if isinstance(msg, dict) and msg.get("type") == "P_SOCKET_READY":
                            self._p_socket_ready = True
                            logger.info(f"[DECODE-PP{self.pp_rank}] Received P_SOCKET_READY signal")
                            # 🔧 CRITICAL: Now create the socket connection to PREFILL
                            self._create_p_instance_socket()
                            logger.info(f"[DECODE-PP{self.pp_rank}] PREFILL socket ready, proceeding with event loop")
                            return
                    else:
                        # No recv socket, skip waiting
                        logger.warning(f"[DECODE-PP{self.pp_rank}] No recv_from_p_instance socket, skipping P_SOCKET_READY wait")
                        self._p_socket_ready = True
                        self._create_p_instance_socket()
                        return
                except zmq.Again:
                    time.sleep(0.01)  # 10ms polling interval
                except Exception as e:
                    logger.warning(f"[DECODE-PP{self.pp_rank}] Error waiting for P_SOCKET_READY: {e}")
                    break
            if not self._p_socket_ready:
                logger.warning(f"[DECODE-PP{self.pp_rank}] Timeout waiting for P_SOCKET_READY, proceeding anyway")
                self._p_socket_ready = True  # Proceed anyway after timeout
                self._create_p_instance_socket()  # Create socket anyway
        except Exception as e:
            logger.warning(f"[DECODE-PP{self.pp_rank}] Failed to wait for P_SOCKET_READY: {e}")
            self._p_socket_ready = True  # Proceed anyway on error
            self._create_p_instance_socket()  # Create socket anyway

            def _hello_worker():
                while not self._handshake_done:
                    try:
                        self.bridge_socket.send_pyobj({"type": "HELLO", "pp": pp_rank})
                    except Exception:
                        pass
                    time.sleep(0.5)
            self._hello_thread = threading.Thread(target=_hello_worker, daemon=True)
            self._hello_thread.start()

            # 🔧 CRITICAL: Wait for PREFILL socket to be ready before proceeding
            # This ensures the first request forwarding doesn't lose messages
            logger.info(f"[DECODE-PP{pp_rank}] About to wait for PREFILL socket ready (attn_tp_rank={self.attn_tp_rank})")
            self._wait_for_prefill_socket_ready()
        else:
            self.bridge_socket = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_p_instance = SimpleNamespace(send_pyobj=lambda x: None)
            self._handshake_done = True

            # Semi-PD unified clock: local micro-batch sequence counter (best-effort)
            try:
                self._mb_seq = 0
            except Exception:
                self._mb_seq = 0

        # 🔧 CRITICAL: Wait for PREFILL socket to be ready at the end of __init__
        # This ensures the first request forwarding doesn't lose messages
        if self.attn_tp_rank == 0 and hasattr(self, '_wait_for_prefill_socket_ready'):
            logger.info(f"[DECODE-PP{self.pp_rank}] __init__ complete, waiting for PREFILL socket ready")
            self._wait_for_prefill_socket_ready()

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


    def get_new_batch_prefill(self, rids: Optional[List[str]] = None) -> Optional[ScheduleBatch]:
        """
        Semi-PD changes:
          - keep scheduled prefill batches in scheduled_prefill_batches
          - disable mixed-style chunked prefill
          - skip requests that not in rids
        """
        # When rids is None, build from the full waiting_queue (used by PP>0 async authorization)
        # If rids is a list, restrict to that subset (used by PP0 sync authorization)
        # Check if the grammar is ready in the grammar queue
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        # Handle the cases where prefill is not allowed
        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            # removed noisy TRACE gnbp.early_exit log
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
        # 🔧 DEBUG: Log new_token_ratio being used
        if self.pp_rank > 0 and os.environ.get("SGLANG_SEMIPD_TRACE","0").lower() in ("1","true","yes"):
            logger.info(f"[DECODE-PP{self.pp_rank}] 🔍 PrefillAdder init: new_token_ratio={self.new_token_ratio:.3f}, waiting_queue={len(self.waiting_queue)}, running={len(self.running_batch.reqs)}")
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
        if os.environ.get("SGLANG_SEMIPD_TRACE","0").lower() in ("1","true","yes"):
            # 🔧 节流：每10000次打印一次
            if not hasattr(self, '_gnbp_begin_count'):
                self._gnbp_begin_count = 0
            self._gnbp_begin_count += 1
            if self._gnbp_begin_count % 10000 == 1:
                try:
                    logger.info(f"[DECODE-PP{self.pp_rank}] TRACE gnbp.begin waiting={len(self.waiting_queue)} rids={'ALL' if rids is None else len(rids)} (count={self._gnbp_begin_count})")
                except Exception:
                    pass
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
                # 🔧 DEBUG: Log why add_one_req failed
                if self.pp_rank > 0 and os.environ.get("SGLANG_SEMIPD_TRACE","0").lower() in ("1","true","yes"):
                    if not hasattr(self, '_add_req_fail_count'):
                        self._add_req_fail_count = 0
                    self._add_req_fail_count += 1
                    if self._add_req_fail_count % 10000 == 1:
                        logger.info(f"[DECODE-PP{self.pp_rank}] 🔍 add_one_req failed: res={res}, rid={req.rid}, can_run_list_sz={len(adder.can_run_list)} (count={self._add_req_fail_count})")
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
            if os.environ.get("SGLANG_SEMIPD_TRACE","0").lower() in ("1","true","yes"):
                # 🔧 节流：每10000次打印一次
                if not hasattr(self, '_gnbp_no_can_run_count'):
                    self._gnbp_no_can_run_count = 0
                self._gnbp_no_can_run_count += 1
                if self._gnbp_no_can_run_count % 10000 == 1:
                    try:
                        logger.info(f"[DECODE-PP{self.pp_rank}] TRACE gnbp.no_can_run waiting={len(self.waiting_queue)} running={len(self.running_batch.reqs)} new_token_ratio={self.new_token_ratio:.3f} (count={self._gnbp_no_can_run_count})")
                    except Exception:
                        pass
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

        # Print stats (optional): avoid confusing DECODE logs as if executing prefill
        if self.attn_tp_rank == 0 and get_bool_env_var("SGLANG_SEMIPD_D_LOG_PREFILL_STATS", default="0"):
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
        # Semi-PD: D-side pre-allocation and metadata setup (does NOT execute model)
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
        # 🔧 CRITICAL: Wait for PREFILL socket to be ready before sending
        # This ensures messages are not lost due to socket not being bound yet
        if self.attn_tp_rank == 0:
            self._wait_for_prefill_socket_ready()

        # Optional trace: who requested authorization and with which candidate rids
        # keep a concise trace; drop heavy stack to reduce log noise
        if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1", "true", "yes"):
            # 🔧 节流：每10000次打印一次
            if not hasattr(self, '_get_next_prefill_batch_count'):
                self._get_next_prefill_batch_count = 0
            self._get_next_prefill_batch_count += 1
            if self._get_next_prefill_batch_count % 10000 == 1:
                try:
                    logger.info(f"[DECODE-PP{self.pp_rank}] TRACE get_next_prefill_batch rids={recv_req.rids} (count={self._get_next_prefill_batch_count})")
                except Exception:
                    pass

        # quiet


        # 🔧 CRITICAL: PP>0 now uses synchronous authorization path aligned with semipd_nopp
        # PREFILL blocks waiting for authorization, so DECODE must respond synchronously
        # No longer use async path for PP>0

        # Decide phase for this authorization (P-only EXTEND on PP0; PRIME on last stage)
        try:
            phase = "EXTEND" if getattr(self, 'pp_rank', 0) == 0 else "PRIME_DECODE"
            # Best-effort StepTag to help P align logs; not used for control gating
            try:
                self.bridge_socket.send_pyobj(StepTag(mb_id=None, phase=phase, pp_rank=getattr(self, 'pp_rank', None), req_ids=list(recv_req.rids or [])))
                # 🔧 节流：每10000次打印一次
                if not hasattr(self, '_auth_begin_count'):
                    self._auth_begin_count = 0
                self._auth_begin_count += 1
                if self._auth_begin_count % 10000 == 1:
                    logger.info(f"[IPC][role=D→P][pp_rank={getattr(self,'pp_rank','?')}][mb_id=-][phase={phase}] SEND AUTH_BEGIN (count={self._auth_begin_count})")
            except Exception:
                pass
        except Exception:
            phase = "EXTEND"

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
                # Reply via work-plane channel to align with semipd_nopp
                self.send_to_p_instance.send_pyobj(empty)
                try:
                    semi_pd_log_info_throttle(
                        logger,
                        key=f"pp{self.pp_rank}.p2d.gnp.send",
                        msg=f"[DECODE-PP{self.pp_rank}] →P GetNextPrefillBatchOutput(sync via p_scheduler_input): #rids=0",
                    )
                except Exception:
                    pass
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
                # Reply via work-plane channel to align with semipd_nopp
                self.send_to_p_instance.send_pyobj(msg)
                try:
                    semi_pd_log_info_throttle(
                        logger,
                        key=f"pp{self.pp_rank}.p2d.gnp.send",
                        msg=f"[DECODE-PP{self.pp_rank}] →P GetNextPrefillBatchOutput(sync via p_scheduler_input): #rids={len(approved_rids)}",
                    )
                except Exception:
                    pass
        except Exception:
            # Fall back silently; PREFILL will retry via resend logic.
            pass

        # Bump local tick for observability
        try:
            ps = int(getattr(self, 'pp_size', 1)) or 1
            self._mb_seq = (getattr(self, '_mb_seq', 0) + 1) % ps
        except Exception:
            pass

        # Synchronous path fully handles the authorization; no dispatcher output
        return None

    def _maybe_authorize_prefill(self):
        # Semi-PD unified clock:
        # - PP0: handles authorization and communicates with PREFILL-PP0
        # - PP>0: DOES NOT authorize PREFILL (PP1 only receives hidden states and generates tokens)

        # 🔧 CRITICAL FIX: In PP mode, only PP0 should authorize PREFILL
        # PP>0 stages should NOT call this method - they only receive hidden states and generate tokens
        if getattr(self, 'pp_rank', 0) > 0:
            # PP>0: Skip authorization entirely
            # PP1 will receive hidden states from PP0 via NCCL and generate tokens
            # No need to authorize PREFILL on PP>0
            return

        try:
            # 🔍 TRACE: probe auth state (throttled) - PP0 only
            try:
                if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1","true","yes"):
                    if not hasattr(self, "_auth_probe_count"):
                        self._auth_probe_count = 0
                    self._auth_probe_count += 1
                    if self._auth_probe_count % 5000 == 1:
                        logger.info(
                            f"[DECODE-PP{getattr(self,'pp_rank','?')}] TRACE auth.probe "
                            f"wq={len(getattr(self,'waiting_queue',[]))} "
                            f"running={len(self.running_batch.reqs) if getattr(self,'running_batch',None) is not None else 0} "
                            f"prealloc={len(getattr(self,'_spd_prealloc_queue',[]))} "
                            f"outbox={len(getattr(self,'_spd_authorize_outbox',[]))} "
                            f"p_ready={getattr(self,'_p_socket_ready',False)} "
                            f"ratio={getattr(self,'new_token_ratio',-1):.3f}"
                        )
            except Exception:
                pass

            if getattr(self.server_args, 'enable_semi_pd', False):
                # PP0 strict sync: by default, do NOT push non-empty async authorizations to PP0.
                # Only nudge with empty auths to clear awaiting flags.
                if getattr(self, 'pp_rank', 0) == 0:
                    try:
                        strict = os.environ.get("SGLANG_SEMIPD_PP0_STRICT_SYNC", "0").lower() in ("1","true","yes")
                        # If proactive PP0 auth is explicitly enabled, override strict to False
                        if os.environ.get("SGLANG_SEMIPD_PP0_PROACTIVE_AUTH", "0").lower() in ("1","true","yes"):
                            strict = False
                    except Exception:
                        strict = True
                    try:
                        if strict:
                            # Strict PP0: do NOT send empty nudges; rely solely on sync get_next_prefill_batch via bridge.
                            return
                        # Non-strict mode: keep previous PP0 async fallback
                        if self.chunked_req:
                            self.tree_cache.cache_unfinished_req(self.chunked_req)
                            self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
                        batch = self.get_new_batch_prefill(None)
                        if batch is None:
                            # Empty nudge
                            import time as _t
                            _min_int = float(os.environ.get("SGLANG_SEMIPD_EMPTY_AUTH_MIN_S", "0.2"))
                            _now = _t.time()
                            if _now - getattr(self, "_last_empty_auth_ts", 0.0) >= _min_int:
                                empty = GetNextPrefillBatchOutput(
                                    rids=[], chunked_rid=None, req_pool_indices=[], prefix_lens=[], extend_input_lens=[]
                                )
                                try:
                                    self.bridge_socket.send_pyobj(empty)
                                except Exception:
                                    pass
                                self._last_empty_auth_ts = _now
                            return
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
                        # StepTag for observability (EXTEND on PP0)
                        try:
                            self.bridge_socket.send_pyobj(
                                StepTag(mb_id=None, phase="EXTEND", pp_rank=getattr(self, 'pp_rank', None), req_ids=list(approved_rids))
                            )
                            # 🔧 节流：每10000次打印一次
                            if not hasattr(self, '_auth_begin_extend_count'):
                                self._auth_begin_extend_count = 0
                            self._auth_begin_extend_count += 1
                            if self._auth_begin_extend_count % 10000 == 1:
                                logger.info(
                                    f"[IPC][role=D→P][pp_rank={getattr(self,'pp_rank','?')}][mb_id=-][phase=EXTEND] SEND AUTH_BEGIN bridge={getattr(self.port_args,'bridge_ipc_name','?')} (count={self._auth_begin_extend_count})"
                                )
                        except Exception:
                            pass
                        # Actually deliver the authorization to PREFILL via the same work-plane channel as PP>0 (p_scheduler_input)
                        try:
                            self.send_to_p_instance.send_pyobj(msg)
                            semi_pd_log_info_throttle(
                                logger,
                                key=f"pp{self.pp_rank}.p2d.gnp.send",
                                msg=f"[DECODE-PP{self.pp_rank}] (PP0) →P GetNextPrefillBatchOutput via p_scheduler_input: #rids={len(approved_rids)}",
                            )
                        except Exception:
                            pass
                        return
                    except Exception:
                        return

                # 🔧 NOTE: PP>0 authorization code removed - PP>0 should NOT authorize PREFILL
                # PP>0 only receives hidden states from PP0 and generates tokens
                # Authorization is handled by PP0 only
        except Exception:
            pass
        # First try to send any queued authorizations
        if self._spd_authorize_outbox:
            try:
                msg = self._spd_authorize_outbox[0]
                self.bridge_socket.send_pyobj(msg)
                if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1", "true", "yes"):
                    # 🔧 节流：每10000次打印一次，移除 traceback
                    if not hasattr(self, '_maybe_authorize_count'):
                        self._maybe_authorize_count = 0
                    self._maybe_authorize_count += 1
                    if self._maybe_authorize_count % 10000 == 1:
                        try:
                            logger.info(f"[DECODE-PP{self.pp_rank}] TRACE _maybe_authorize_prefill send queued msg(#rids={len(msg.rids)}) (count={self._maybe_authorize_count})")
                        except Exception:
                            pass
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

        # Proactive authorization for PP0: authorize PREFILL from local waiting queue
        # This avoids relying on PP0-P to "pull" authorization and keeps P0/P1 in the same tick.
        try:
            if getattr(self, 'pp_rank', 0) == 0:
                # Release unfinished chunk if any
                if self.chunked_req:
                    self.tree_cache.cache_unfinished_req(self.chunked_req)
                    self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
                batch = self.get_new_batch_prefill(None)
                if batch is None:
                    # Nudge P to clear awaiting flag; throttle via outbox
                    self._spd_authorize_outbox.append(
                        GetNextPrefillBatchOutput(
                            rids=[], chunked_rid=None,
                            req_pool_indices=[], prefix_lens=[], extend_input_lens=[]
                        )
                    )
                    return
                approved_rids = [r.rid for r in batch.reqs]
                req_pool_indices = [r.req_pool_idx for r in batch.reqs]
                prefix_lens = [len(r.prefix_indices) for r in batch.reqs]
                extend_input_lens = [r.extend_input_len for r in batch.reqs]
                # Enqueue to outbox; it will be sent at the top on next call
                self._spd_authorize_outbox.append(
                    GetNextPrefillBatchOutput(
                        rids=approved_rids,
                        chunked_rid=(self.chunked_req.rid if self.chunked_req else None),
                        req_pool_indices=req_pool_indices,
                        prefix_lens=prefix_lens,
                        extend_input_lens=extend_input_lens,
                    )
                )
                # Best-effort observability only
                try:
                    phase = "EXTEND"
                    self.bridge_socket.send_pyobj(
                        StepTag(
                            mb_id=None, phase=phase,
                            pp_rank=getattr(self, 'pp_rank', None),
                            req_ids=list(approved_rids),
                        )
                    )
                    logger.info(
                        f"[IPC][role=D→P][pp_rank={getattr(self,'pp_rank','?')}][mb_id=-][phase={phase}] QUEUE AUTH_BEGIN bridge={getattr(self.port_args,'bridge_ipc_name','?')}"
                    )
                except Exception:
                    pass
                return
        except Exception:
            # Fall through to other paths on any error
            pass

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
            msg = GetNextPrefillBatchOutput(
                rids=approved_rids,
                chunked_rid=(self.chunked_req.rid if self.chunked_req else None),
                req_pool_indices=req_pool_indices,
                prefix_lens=prefix_lens,
                extend_input_lens=extend_input_lens,
            )
            self._spd_authorize_outbox.append(msg)
            try:
                semi_pd_log_info_throttle(
                    logger,
                    key=f"pp{self.pp_rank}.d.send.gnpo.pp0",
                    msg=(
                        f"[DECODE-PP{self.pp_rank}] →P GetNextPrefillBatchOutput(async, PP0): #rids={len(approved_rids)} "
                        f"pools={req_pool_indices} pre_lens={prefix_lens} ext_lens={extend_input_lens}"
                    ),
                )
            except Exception:
                pass
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
            try:
                semi_pd_log_info_throttle(
                    logger,
                    key=f"pp{self.pp_rank}.d.auth.queue",
                    msg=f"[DECODE-PP{self.pp_rank}] queue empty GetNextPrefillBatchOutput to unblock P",
                )
            except Exception:
                pass
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
        try:
            semi_pd_log_info_throttle(
                logger,
                key=f"pp{self.pp_rank}.d.auth.queue",
                msg=f"[DECODE-PP{self.pp_rank}] queue GetNextPrefillBatchOutput: #rids={len(approved_rids)}",
            )
        except Exception:
            pass


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
        try:
            # Throttled receive log with rids for visibility
            semi_pd_log_info_throttle(
                logger,
                key=f"pp{getattr(self,'pp_rank','?')}.d.recv.presult",
                msg=f"[DECODE-PP{getattr(self,'pp_rank','?')}] ←P prefill_result: tokens={num_tokens} rids={list(getattr(recv_req,'rids',[]) or [])}",
                interval_ms=1000,
            )
        except Exception:
            pass
        logger.info(f"[DECODE-PP{getattr(self,'pp_rank','?')}] ←P tokens: {num_tokens}")

        # Correlate the correct authorized batch: prefer matching by rids when provided
        batch = None
        try:
            if getattr(recv_req, 'rids', None):
                want = tuple(getattr(recv_req, 'rids'))
                for i, b in enumerate(self.scheduled_prefill_batches):
                    try:
                        have = tuple([r.rid for r in b.reqs])
                    except Exception:
                        have = None
                    if have == want:
                        batch = self.scheduled_prefill_batches.pop(i)
                        break
        except Exception:
            batch = None
        if batch is None:
            # Fallback to FIFO for backward compatibility
            batch = self.scheduled_prefill_batches.pop(0)

        logits_processor_output = None
        if recv_req.next_token_logits is not None:
            logits_processor_output = LogitsProcessorOutput(
                next_token_logits=torch.from_numpy(recv_req.next_token_logits).to(
                    self.device, dtype=torch.float16, non_blocking=True
                ),
                hidden_states=None,
            )

        # 🔧 移除 traceback 打印，只保留简单日志
        if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1", "true", "yes"):
            # 🔧 节流：每10000次打印一次
            if not hasattr(self, '_process_prefill_result_count'):
                self._process_prefill_result_count = 0
            self._process_prefill_result_count += 1
            if self._process_prefill_result_count % 10000 == 1:
                try:
                    logger.info(f"[DECODE-PP{getattr(self,'pp_rank','?')}] TRACE process_prefill_result tokens={num_tokens} (count={self._process_prefill_result_count})")
                except Exception:
                    pass


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
        return super().run_batch(batch)

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        # Semi-PD decode-only loop:
        # - PP0: proactively authorize PREFILL (main coordinator)
        # - PP>0: NO authorization, just receive hidden states and process
        # - never call super().get_next_batch_to_run() to avoid PREFILL execution on DECODE

        # 🔧 CRITICAL: Only PP0 should authorize PREFILL
        # PP>0 stages just receive hidden states and process them
        if self.pp_rank == 0:
            self._maybe_authorize_prefill()

        # 🔧 CRITICAL FIX: In PP mode, DECODE-PP>0 should NOT manually create batches
        # The native PP flow will automatically create dummy batches to receive hidden states
        # We just need to keep the requests in waiting_queue for tracking
        # The actual batch creation and processing is handled by the native PP event loop

        # If we have a ready decode batch queued from PREFILL result, adopt it (PP0 only)
        if getattr(self, "_ready_decode_batches", None) and self.running_batch.is_empty():
            if len(self._ready_decode_batches) > 0:
                self.running_batch = self._ready_decode_batches.popleft()

        # Update/return decode batch only
        ret = None
        if not self.running_batch.is_empty():
            self.running_batch = self.update_running_batch(self.running_batch)
            if not self.running_batch.is_empty():
                ret = self.running_batch

        # 🔧 CRITICAL: In Semi-PD+PP, ALL DECODE stages need to receive hidden states from previous stage
        # even when they have no running_batch. Return an IDLE batch to trigger NCCL recv.
        # - DECODE-PP0 receives tokens from DECODE-PP1 (backward flow)
        # - DECODE-PP>0 receives hidden states from DECODE-PP(rank-1) (forward flow)
        if ret is None and self.pp_size > 1:
            # ALL PP stages: Return IDLE batch to trigger NCCL recv
            from sglang.srt.managers.schedule_batch import ScheduleBatch
            ret = ScheduleBatch.init_new(
                reqs=[],
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tree_cache=self.tree_cache,
                model_config=self.model_config,
                enable_overlap=getattr(self, 'enable_overlap', False),
                spec_algorithm=getattr(self, 'spec_algorithm', None),
                enable_custom_logit_processor=getattr(self, 'enable_custom_logit_processor', False),
            )
            ret.prepare_for_idle()
            try:
                if self.pp_rank == 0:
                    semi_pd_log_info_throttle(
                        logger,
                        key=f"pp{self.pp_rank}.d.idle.recv",
                        msg=f"[DECODE-PP{self.pp_rank}] return IDLE batch to recv tokens from PP1",
                        interval_ms=1000,
                    )
                else:
                    semi_pd_log_info_throttle(
                        logger,
                        key=f"pp{self.pp_rank}.d.idle.recv",
                        msg=f"[DECODE-PP{self.pp_rank}] return IDLE batch to recv hidden states from PP{self.pp_rank-1}",
                        interval_ms=1000,
                    )
            except Exception:
                pass

        # Handle DP attention if enabled
        if self.server_args.enable_dp_attention and ret is not None:
            ret, _ = self.prepare_dp_attn_batch(ret)
        return ret

    def recv_requests(self):
        """Extend parent's recv to also poll local PULL from PREFILL on non-PP0 stages."""
        fwd_ct = 0  # ensure defined even if forwarding block is skipped due to exceptions

        recv_reqs = super().recv_requests()

        # 🔧 CRITICAL: For Semi-PD PP mode, DECODE-PP>0 receives requests from DECODE-PP0 via point_to_point_pyobj
        # These requests need to be forwarded to same-stage PREFILL
        # Don't drop TokenizedGenerateReqInput on PP>0, forward them to PREFILL instead

        # Forward newly arrived generate requests to same-stage PREFILL to ensure its waiting_queue is populated (ALL PP stages)
        try:
            if (
                self.attn_tp_rank == 0
                and getattr(self, 'send_to_p_instance', None) is not None
            ):
                from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
                fwd_ct = 0
                for _obj in list(recv_reqs or []):
                    if isinstance(_obj, TokenizedGenerateReqInput):
                        try:
                            # Wait for socket readiness before sending
                            if not getattr(self, '_p_socket_ready', True):
                                import time
                                timeout = 5.0
                                start_time = time.time()
                                while not self._p_socket_ready and (time.time() - start_time) < timeout:
                                    time.sleep(0.01)
                                if not self._p_socket_ready:
                                    logger.warning(f"[DECODE-PP{self.pp_rank}] Socket not ready, sending anyway")
                            self.send_to_p_instance.send_pyobj(_obj)
                            fwd_ct += 1
                        except Exception:
                            pass
                if fwd_ct:
                    logger.info(f"[DECODE-PP{self.pp_rank}] fwd_to_P: {fwd_ct} TokenizedGenerateReqInput")
        except Exception:
            pass
        try:
            if fwd_ct:
                semi_pd_log_info_throttle(
                    logger,
                    key=f"pp{self.pp_rank}.d.fwd.gen",
                    msg=(
                        f"[DECODE-PP{self.pp_rank}] fwd_to_P via p_scheduler={getattr(self.port_args,'p_scheduler_input_ipc_name','?')} count={fwd_ct}"
                    ),
                    interval_ms=1000,
                )
        except Exception:
            pass
        if getattr(self, 'recv_from_p_instance', None) is not None and self.attn_tp_rank == 0:
            while True:
                try:
                    obj = self.recv_from_p_instance.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                if isinstance(obj, dict):
                    # Handle control plane messages inline (e.g., HELLO/ACK)
                    if obj.get("type") == "HELLO_ACK":
                        self._handshake_done = True
                    # drop other control dicts silently
                    continue
                if recv_reqs is None:
                    recv_reqs = []
                # Accept only work messages
                # 🔧 Semi-PD + PP: Only DECODE-PP0 receives BatchProcessPrefillResultReq (tokens from PREFILL-PP0)
                # DECODE-PP1 does not receive tokens via IPC anymore (uses standard PP pipeline instead)
                if isinstance(obj, GetNextPrefillBatchInput):
                    # All DECODE PP stages receive GetNextPrefillBatchInput (authorization requests)
                    try:
                        semi_pd_log_info_throttle(
                            logger,
                            key=f"pp{self.pp_rank}.d.recv.gnpi",
                            msg=f"[DECODE-PP{self.pp_rank}] <-P GetNextPrefillBatchInput: #rids={len(getattr(obj,'rids',[]) or [])} sample={list(getattr(obj,'rids',[]) or [])[:1]}",
                            interval_ms=1000,
                        )
                    except Exception:
                        pass
                    recv_reqs.append(obj)
                elif isinstance(obj, BatchProcessPrefillResultReq):
                    # Only DECODE-PP0 receives BatchProcessPrefillResultReq (tokens from PREFILL-PP0)
                    if self.pp_rank == 0:
                        try:
                            semi_pd_log_info_throttle(
                                logger,
                                key=f"pp{self.pp_rank}.d.zmq.recv.presult",
                                msg=f"[DECODE-PP0] ZMQ recv BatchProcessPrefillResultReq rids={list(getattr(obj,'rids',[]) or [])}",
                                interval_ms=1000,
                            )
                        except Exception:
                            pass
                        recv_reqs.append(obj)
                    else:
                        # DECODE-PP1 ignores BatchProcessPrefillResultReq (uses standard PP pipeline)
                        try:
                            semi_pd_log_info_throttle(
                                logger,
                                key=f"pp{self.pp_rank}.d.zmq.ignore.presult",
                                msg=f"[DECODE-PP{self.pp_rank}] Ignoring BatchProcessPrefillResultReq (not PP0)",
                                interval_ms=5000,
                            )
                        except Exception:
                            pass
        return recv_reqs

    def process_input_requests(self, recv_reqs):
        """Filter out Semi-PD control dicts (e.g., HELLO/ACK) before dispatch."""
        try:
            filtered = []
            for obj in (recv_reqs or []):
                if isinstance(obj, dict):
                    # 🔧 CRITICAL FIX: Only filter Semi-PD control messages, not all dicts!
                    # Regular requests (TokenizedGenerateReqInput, etc.) should be processed
                    if obj.get("type") == "HELLO_ACK":
                        self._handshake_done = True
                        continue
                    # 🔧 FIX: Don't drop all dicts! Only drop Semi-PD control messages
                    # Check if this is a Semi-PD control message (has "type" field)
                    if "type" in obj:
                        # This is a control message, drop it
                        logger.debug(f"[DECODE-PP{self.pp_rank}] Dropping Semi-PD control message: type={obj.get('type')}")
                        continue
                    # Otherwise, this is a regular request dict, keep it
                    logger.info(f"[DECODE-PP{self.pp_rank}] Received request dict: keys={list(obj.keys())}")
                filtered.append(obj)
            return super().process_input_requests(filtered)
        except Exception:
            return super().process_input_requests(recv_reqs)
        # 其余跨 stage 的张量/令牌交接全部回归 event_loop_pp 处理
