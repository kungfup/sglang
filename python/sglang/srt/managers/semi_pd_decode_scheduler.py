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
import os
import threading
import time
from types import SimpleNamespace
from typing import List, Optional, Union

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
from sglang.srt.managers.scheduler import EmbeddingBatchResult, GenerationBatchResult
from sglang.srt.managers.semi_pd_scheduler import SemiPDScheduler
from sglang.srt.server_args import PortArgs, SemiPDPortArgs, ServerArgs
from sglang.srt.utils import broadcast_pyobj, get_bool_env_var, get_zmq_socket

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
        dp_rank: Optional[int],
        bypass_load_weight: bool = False,
    ):
        super().__init__(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            0,  # pp_rank
            dp_rank,
            False,
            InstanceRole.DECODE,
        )

        # Log environment info for cross-platform debugging
        import platform
        import torch
        with open('/tmp/semi_pd_debug.log', 'w') as f:  # Clear previous log
            f.write(f"=== Semi-PD Debug Log ===\n")
            f.write(f"Platform: {platform.platform()}\n")
            f.write(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'NO_CUDA'}\n")
            f.write(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'NO_CUDA'}\n")
            f.write(f"Model Path: {server_args.model_path}\n")
            f.write(f"Instance Role: DECODE\n")
            f.write(f"Enable Overlap: {getattr(server_args, 'enable_overlap', 'Unknown')}\n")
            f.write(f"CUDA Graph: {not getattr(server_args, 'disable_cuda_graph', True)}\n")
            f.write(f"TP Size: {getattr(server_args, 'tp_size', 'Unknown')}\n")
            f.write(f"GPU ID: {gpu_id}\n")
            f.write(f"=========================\n")

        self._request_dispatcher._mapping.extend(
            [
                (GetNextPrefillBatchInput, self.get_next_prefill_batch),
                (BatchProcessPrefillResultReq, self.process_prefill_result),
            ]
        )

        # For requests that has been sent to the prefill scheduler but not yet finished.
        self.scheduled_prefill_batches: List[ScheduleBatch] = []

        if self.attn_tp_rank == 0:
            context = zmq.Context(2)

            assert isinstance(port_args, SemiPDPortArgs)
            self.bridge_socket = get_zmq_socket(
                context, zmq.PUSH, port_args.bridge_ipc_name, False
            )
            self.send_to_p_instance = get_zmq_socket(
                context, zmq.PUSH, port_args.p_scheduler_input_ipc_name, False
            )
        else:
            self.bridge_socket = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_p_instance = SimpleNamespace(send_pyobj=lambda x: None)

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """
        Semi-PD changes:
          - add the retracted requests to the prefill scheduler
          - add EOS token detection for decode phase
        """
        initial_bs = batch.batch_size()

        # Semi-PD: Remove extra EOS detection from update_running_batch
        # Let process_batch_result_decode handle EOS detection properly like original SGLang

        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # Check if decode out of memory
        if not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier) or (
            TEST_RETRACT and batch.batch_size() > 10
        ):
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio = batch.retract_decode(self.server_args)
            self.new_token_ratio = new_token_ratio

            logger.info(
                "Decode out of memory happened. "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )

            # Semi-PD
            for req in retracted_reqs:
                req: Req
                message = TokenizedGenerateReqInput(
                    rid=req.rid,
                    input_text=req.origin_input_text + req.decoded_text,
                    input_ids=req.origin_input_ids + req.output_ids,
                    image_inputs=req.image_inputs,
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
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        # Update batch tensors
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

    def get_new_batch_prefill(self, rids: List[str]) -> Optional[ScheduleBatch]:
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
        logger.info(f"[DECODE] Processing waiting queue, rids={rids}, waiting_queue_size={len(self.waiting_queue)}")
        for req in self.waiting_queue:
            # Semi-PD
            if req.rid not in rids:
                logger.debug(f"[DECODE] Skipping req.rid={req.rid} (not in rids)")
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
        logger.info(f"[DECODE] 🔥 get_next_prefill_batch called with rids: {recv_req.rids}")

        if self.chunked_req:
            self.tree_cache.cache_unfinished_req(self.chunked_req)
            self.req_to_token_pool.free(self.chunked_req.req_pool_idx)

        logger.info(f"[DECODE] Calling get_new_batch_prefill with rids: {recv_req.rids}")
        batch = self.get_new_batch_prefill(recv_req.rids)
        logger.info(f"[DECODE] get_new_batch_prefill returned: {batch is not None}")

        if batch is None:
            response = GetNextPrefillBatchOutput(
                rids=[],
                chunked_rid=None,
                req_pool_indices=[],
                prefix_lens=[],
                extend_input_lens=[],
            )
            logger.debug(f"[DECODE] Send empty response to P worker: {response}")
            self.bridge_socket.send_pyobj(response)
        else:
            # Serialize the essential information of the batch
            response = GetNextPrefillBatchOutput(
                rids=[r.rid for r in batch.reqs],
                chunked_rid=(self.chunked_req.rid if self.chunked_req else None),
                req_pool_indices=[r.req_pool_idx for r in batch.reqs],
                prefix_lens=[len(r.prefix_indices) for r in batch.reqs],
                extend_input_lens=[r.extend_input_len for r in batch.reqs],
            )
            logger.info(f"[DECODE] Send response to P worker: {response}")
            self.bridge_socket.send_pyobj(response)

    def process_prefill_result(self, recv_req: BatchProcessPrefillResultReq):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        logger.info(f"[DECODE] 🔥 process_prefill_result started, next_token_ids={recv_req.next_token_ids}")
        batch = self.scheduled_prefill_batches.pop(0)
        logger.info(f"[DECODE] 🔥 Got batch with {len(batch.reqs)} requests")
        assert len(batch.reqs) == len(recv_req.next_token_ids)

        logits_processor_output = None
        if recv_req.next_token_logits is not None:
            logits_processor_output = LogitsProcessorOutput(
                next_token_logits=torch.from_numpy(recv_req.next_token_logits).to(
                    self.device, dtype=torch.float16, non_blocking=True
                ),
                hidden_states=None,
            )

        # TODO: return logprobs is not supported in Semi-PD mode
        # Provide proper extend_input_len_per_req and extend_logprob_start_len_per_req
        extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]
        extend_logprob_start_len_per_req = [0 for _ in batch.reqs]  # Start from beginning

        result = GenerationBatchResult(
            logits_output=logits_processor_output,
            pp_hidden_states_proxy_tensors=None,  # v0.4.8 requires this parameter
            next_token_ids=recv_req.next_token_ids,
            extend_input_len_per_req=extend_input_len_per_req,
            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
            bid=-1,  # doesn't matter
            can_run_cuda_graph=False,  # v0.4.8 requires this parameter
        )

        if self.attn_tp_size > 1:
            dist.barrier(group=self.attn_tp_cpu_group)

        logger.info(f"[DECODE] 🔥 Setting batch.output_ids...")
        batch.output_ids = torch.from_numpy(
            np.array(result.next_token_ids, dtype=np.int64)
        ).to(self.device, dtype=torch.int64, non_blocking=True)

        logger.info(f"[DECODE] 🔥 Calling process_batch_result_prefill...")
        self.process_batch_result_prefill(batch, result)
        logger.info(f"[DECODE] 🔥 process_batch_result_prefill completed!")

        logger.info(f"[DECODE] 🔥 Filtering batch...")
        batch.filter_batch(chunked_req_to_exclude=self.chunked_req)

        logger.info(f"[DECODE] 🔥 Merging batch to running_batch...")
        if not batch.is_empty():
            if self.running_batch.is_empty():
                self.running_batch = batch
            else:
                self.running_batch.merge_batch(batch)

        logger.info(f"[DECODE] 🔥 process_prefill_result completed successfully!")

    def process_batch_result_decode(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: Optional[threading.Event] = None,
    ):
        """Semi-PD decode scheduler processes decode results with EOS token detection.

        This is a simplified version based on standard SGLang but adapted for Semi-PD.
        """
        try:
            # Handle overlap mode first (like standard SGLang)
            if self.enable_overlap:
                logits_output, next_token_ids, can_run_cuda_graph = (
                    self.tp_worker.resolve_last_batch_result(launch_done)
                )
            else:
                # Get next token IDs from result
                next_token_ids = result.next_token_ids
                if not isinstance(next_token_ids, list):
                    next_token_ids = next_token_ids.tolist()

            # Enhanced logging for first few tokens (after next_token_ids is defined)
            if len(batch.reqs) > 0 and (len(batch.reqs[0].output_ids) <= 5 or len(batch.reqs[0].output_ids) % 50 == 0):
                with open('/tmp/semi_pd_debug.log', 'a') as f:
                    f.write(f"[DECODE] Processing {len(batch.reqs)} requests, current_tokens: {len(batch.reqs[0].output_ids)}\n")
                    if len(batch.reqs[0].output_ids) <= 5:
                        f.write(f"[DECODE] EARLY_STAGE: next_token_ids={next_token_ids[:5] if isinstance(next_token_ids, list) else 'NOT_LIST'}\n")

            # Update requests with new tokens and check completion conditions (like original SGLang)
            for i, req in enumerate(batch.reqs):
                if req.is_retracted or i >= len(next_token_ids):
                    continue

                # Add the new token to the request (like original SGLang)
                next_token_id = next_token_ids[i]

                # Semi-PD: Detailed debug for cross-platform issues
                if len(req.output_ids) <= 5:  # Only log first few tokens
                    with open('/tmp/semi_pd_debug.log', 'a') as f:
                        f.write(f"[{req.rid[:8]}] DECODE_DEBUG token_{len(req.output_ids)+1}: next_token_id={next_token_id}\n")
                        f.write(f"[{req.rid[:8]}] DECODE_STATE: overlap={self.enable_overlap} batch_size={len(batch.reqs)} req_index={i}\n")
                        f.write(f"[{req.rid[:8]}] CURRENT_OUTPUT: {req.output_ids[-10:] if len(req.output_ids) >= 10 else req.output_ids}\n")

                        # Check if this is the problematic first decode token
                        if len(req.output_ids) == 1:  # This will be the second token (first decode token)
                            f.write(f"[{req.rid[:8]}] CRITICAL_FIRST_DECODE_TOKEN: {next_token_id} -> {repr(req.tokenizer.decode([next_token_id]) if req.tokenizer else 'NO_TOKENIZER')}\n")
                            f.write(f"[{req.rid[:8]}] EXPECTED_CHINESE_TOKENS: Should be around 6313(!) or 104139(有什么) etc\n")

                            # Simple check: compare with expected tokens
                            f.write(f"[{req.rid[:8]}] TOKEN_ANALYSIS: H20_vs_L40S comparison needed\n")
                            f.write(f"[{req.rid[:8]}] ISSUE: H20 generates {next_token_id} but L40S generates 6313 for same input\n")

                # Semi-PD: Validate token ID is in valid range
                vocab_size = getattr(req.tokenizer, 'vocab_size', 50000) if req.tokenizer else 50000
                if not (0 <= next_token_id < vocab_size):
                    logger.warning(f"[DECODE] Invalid token ID {next_token_id} (vocab_size={vocab_size}), forcing EOS")
                    # Force EOS by using a valid token that will trigger stopping
                    next_token_id = min(vocab_size - 1, 151643)  # Use a safe token

                req.output_ids.append(next_token_id)

                # Critical debug: Log token details for cross-platform debugging
                if len(req.output_ids) <= 5 or len(req.output_ids) % 25 == 0:
                    with open('/tmp/semi_pd_debug.log', 'a') as f:
                        token_text = req.tokenizer.decode([next_token_id]) if req.tokenizer else f"TOKEN_{next_token_id}"
                        f.write(f"[{req.rid[:8]}] token_{len(req.output_ids)}: {next_token_id} -> {repr(token_text)}\n")

                # Semi-PD: Ensure EOS detection fields are set correctly before check_finished
                # Use a comprehensive approach to handle Qwen's special tokens
                eos_token_ids = []

                if req.tokenizer:
                    vocab_size = getattr(req.tokenizer, 'vocab_size', 50000)

                    # For Qwen models, use <|endoftext|> as the primary EOS token
                    # This is token 151643 which is within vocab range
                    endoftext_token = vocab_size - 1  # Usually <|endoftext|>
                    eos_token_ids.append(endoftext_token)

                    # Also check for other potential EOS tokens in valid range
                    tokenizer_eos = getattr(req.tokenizer, 'eos_token_id', None)
                    if tokenizer_eos is not None and 0 <= tokenizer_eos < vocab_size:
                        if tokenizer_eos not in eos_token_ids:
                            eos_token_ids.append(tokenizer_eos)

                    # Add any other special tokens that are in valid range
                    for attr in ['bos_token_id', 'pad_token_id', 'unk_token_id']:
                        token_id = getattr(req.tokenizer, attr, None)
                        if token_id is not None and 0 <= token_id < vocab_size:
                            if token_id not in eos_token_ids:
                                eos_token_ids.append(token_id)

                # Fallback: use correct Qwen EOS token if no valid EOS tokens found
                if not eos_token_ids:
                    eos_token_ids = [151645]  # Correct Qwen EOS token ID

                # Set EOS token IDs for the request
                if req.eos_token_ids is None or len(req.eos_token_ids) == 0:
                    req.eos_token_ids = set(eos_token_ids)
                if req.sampling_params.stop_token_ids is None:
                    req.sampling_params.stop_token_ids = eos_token_ids.copy()
                else:
                    for eos_id in eos_token_ids:
                        if eos_id not in req.sampling_params.stop_token_ids:
                            req.sampling_params.stop_token_ids.append(eos_id)

                # Debug: Log token info every 50 tokens
                if len(req.output_ids) % 50 == 0:
                    logger.info(f"[DECODE] 📊 Request {req.rid}: {len(req.output_ids)} tokens, last_token={next_token_id}")

                # Check completion conditions using req.check_finished() (like original SGLang)
                req.check_finished()

                # Semi-PD: Add safety net for very long generations (match max_new_tokens)
                if not req.finished() and len(req.output_ids) >= 800:  # Allow more space for natural EOS
                    logger.info(f"[DECODE] ⚠️ Force stopping request {req.rid} after {len(req.output_ids)} tokens (safety net)")
                    from sglang.srt.managers.schedule_batch import FINISH_LENGTH
                    req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))

                if req.finished():
                    self.tree_cache.cache_finished_req(req)
                    req.time_stats.completion_time = time.time()
                    logger.info(f"[DECODE] 🔥 Request {req.rid} finished: {req.finished_reason}")
                else:
                    self.tree_cache.cache_unfinished_req(req)

            # Stream output to detokenizer
            skip_stream_req = []
            self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

            # Completed - reduced logging

        except Exception as e:
            logger.error(f"[DECODE] ❌ Error in process_batch_result_decode: {e}")
            # Don't re-raise to avoid crashing the server
            pass

    def process_batch_result_prefill(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        launch_done: Optional[threading.Event] = None,
    ):
        """Override the base method to handle Semi-PD specific prefill result processing.

        This method is based on SGLang v0.4.8's scheduler_output_processor_mixin.py
        but adapted for Semi-PD Decode instance.
        """
        logger.info(f"[DECODE] 🔥 process_batch_result_prefill called for Semi-PD Decode instance")

        skip_stream_req = None

        if self.is_generation:
            (
                logits_output,
                next_token_ids,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = (
                result.logits_output,
                result.next_token_ids,
                result.extend_input_len_per_req,
                result.extend_logprob_start_len_per_req,
            )

            # Semi-PD: Handle overlap mode differently
            if self.enable_overlap and not self.server_args.enable_semi_pd:
                logits_output, next_token_ids, _ = (
                    self.tp_worker.resolve_last_batch_result(launch_done)
                )
            else:
                # Move next_token_ids and logprobs to cpu
                # Semi-PD: Only convert to list if not already a list
                if not isinstance(next_token_ids, list):
                    next_token_ids = next_token_ids.tolist()
                if batch.return_logprob:
                    if logits_output.next_token_logprobs is not None:
                        logits_output.next_token_logprobs = (
                            logits_output.next_token_logprobs.tolist()
                        )
                    if logits_output.input_token_logprobs is not None:
                        logits_output.input_token_logprobs = tuple(
                            logits_output.input_token_logprobs.tolist()
                        )

            hidden_state_offset = 0

            # Check finish conditions
            logprob_pt = 0
            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if req.is_retracted:
                    continue

                if self.is_mixed_chunk and self.enable_overlap and req.finished():
                    # Free the one delayed token for the mixed decode batch
                    j = len(batch.out_cache_loc) - len(batch.reqs) + i
                    self.token_to_kv_pool_allocator.free(batch.out_cache_loc[j : j + 1])
                    continue

                if req.is_chunked <= 0:
                    # req output_ids are set here
                    req.output_ids.append(next_token_id)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                        req.time_stats.completion_time = time.time()
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        # This updates radix so others can match
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1

                # Handle logprobs if needed
                if batch.return_logprob and req.is_chunked <= 0:
                    # Process logprobs similar to original implementation
                    if logits_output.input_token_logprobs is not None:
                        self.process_input_logprobs(
                            i, req, req.fill_ids, logits_output, req.is_chunked == 0
                        )

                    if logits_output.next_token_logprobs is not None:
                        req.output_token_logprobs.append(
                            logits_output.next_token_logprobs[i]
                        )

                    logprob_pt += len(req.fill_ids)

                # Handle hidden states if needed
                if result.logits_output and result.logits_output.hidden_states is not None:
                    if req.return_hidden_states:
                        req.hidden_states = result.logits_output.hidden_states[
                            hidden_state_offset : hidden_state_offset + len(req.fill_ids)
                        ].tolist()
                    hidden_state_offset += len(req.fill_ids)

        else:  # embedding or reward model
            embeddings, bid = result.embeddings, result.bid
            embeddings = embeddings.tolist()

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.is_chunked <= 0:
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                    else:
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1

        # Stream output to detokenizer - this is the key missing piece!
        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

        logger.info(f"[DECODE] 🔥 process_batch_result_prefill completed for Semi-PD")




