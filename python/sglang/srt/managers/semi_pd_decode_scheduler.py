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

        # CRITICAL: Verify weight sharing after initialization
        logger.info(f"[DECODE] 🔧 CRITICAL: Verifying weight sharing after initialization...")
        from sglang.srt.managers.semi_pd_scheduler import _verify_weight_sharing
        _verify_weight_sharing(self, InstanceRole.DECODE)

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

        # DEBUG: Check KV cache allocation status BEFORE processing
        avail_size_before = self.token_to_kv_pool_allocator.available_size()
        expected_size = self.token_to_kv_pool_allocator.size
        logger.info(f"[DECODE] 🔧 KV Cache Status BEFORE process_batch_result_prefill: avail={avail_size_before}, expected={expected_size}, diff={avail_size_before - expected_size}")

        logger.info(f"[DECODE] 🔥 Calling process_batch_result_prefill...")
        self.process_batch_result_prefill(batch, result)

        # DEBUG: Check KV cache allocation status AFTER processing
        avail_size_after = self.token_to_kv_pool_allocator.available_size()
        logger.info(f"[DECODE] 🔧 KV Cache Status AFTER process_batch_result_prefill: avail={avail_size_after}, expected={expected_size}, diff={avail_size_after - expected_size}")

        if avail_size_after != avail_size_before:
            logger.info(f"[DECODE] 🔧 KV Cache changed by: {avail_size_after - avail_size_before} tokens")
        else:
            logger.warning(f"[DECODE] ⚠️ KV Cache unchanged - potential memory leak!")

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
        """Semi-PD decode scheduler processes decode results.

        Based on original Semi-PD implementation - keep it simple and clean.
        """
        # Handle overlap mode (like original Semi-PD)
        if self.enable_overlap:
            logits_output, next_token_ids, can_run_cuda_graph = self.tp_worker.resolve_last_batch_result(None)
            next_token_logprobs = logits_output.next_token_logprobs
        elif batch.spec_algorithm.is_none():
            # spec decoding handles output logprobs inside verify process.
            next_token_ids = result.next_token_ids.tolist()
            can_run_cuda_graph = True  # Default for non-overlap mode
            if batch.return_logprob:
                next_token_logprobs = result.logits_output.next_token_logprobs.tolist()
        else:
            can_run_cuda_graph = True  # Default fallback

        self.token_to_kv_pool_allocator.free_group_begin()

        # Check finish condition (like original Semi-PD)
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            if req.is_retracted:
                continue

            if self.enable_overlap and req.finished():
                # Free the one extra delayed token
                if self.page_size == 1:
                    self.token_to_kv_pool_allocator.free(batch.out_cache_loc[i : i + 1])
                else:
                    # Only free when the extra token is in a new page
                    if (
                        len(req.origin_input_ids) + len(req.output_ids) - 1
                    ) % self.page_size == 0:
                        self.token_to_kv_pool_allocator.free(
                            batch.out_cache_loc[i : i + 1]
                        )
                continue

            if batch.spec_algorithm.is_none():
                # speculative worker will solve the output_ids in speculative decoding
                req.output_ids.append(next_token_id)

            req.check_finished()
            if req.finished():
                self.tree_cache.cache_finished_req(req)

            if req.return_logprob and batch.spec_algorithm.is_none():
                # speculative worker handles logprob in speculative decoding
                req.output_token_logprobs_val.append(next_token_logprobs[i])
                req.output_token_logprobs_idx.append(next_token_id)
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs_val.append(
                        result.logits_output.next_token_top_logprobs_val[i]
                    )
                    req.output_top_logprobs_idx.append(
                        result.logits_output.next_token_top_logprobs_idx[i]
                    )
                if req.token_ids_logprob is not None:
                    req.output_token_ids_logprobs_val.append(
                        result.logits_output.next_token_token_ids_logprobs_val[i]
                    )
                    req.output_token_ids_logprobs_idx.append(
                        result.logits_output.next_token_token_ids_logprobs_idx[i]
                    )

            if req.return_hidden_states and result.logits_output.hidden_states is not None:
                req.hidden_states.append(
                    result.logits_output.hidden_states[i].cpu().clone().tolist()
                )

            if req.grammar is not None and batch.spec_algorithm.is_none():
                req.grammar.accept_token(next_token_id)
                req.grammar.finished = req.finished()

        # TEMPORARY PATCH: Free KV cache for each generated token to fix memory leak
        # This is the critical fix mentioned in the KV Cache Bug Guide!
        logger.info(f"[DECODE] 🔧 TEMPORARY PATCH: Freeing KV cache for {len(batch.reqs)} generated tokens")
        if hasattr(batch, 'out_cache_loc') and batch.out_cache_loc is not None:
            # Free the KV cache allocated for the generated tokens
            for i in range(len(batch.reqs)):
                if not batch.reqs[i].is_retracted:
                    # Free one token worth of KV cache per request
                    start_idx = i
                    end_idx = i + 1
                    if start_idx < len(batch.out_cache_loc):
                        self.token_to_kv_pool_allocator.free(batch.out_cache_loc[start_idx:end_idx])
                        logger.info(f"[DECODE] 🔧 TEMPORARY PATCH: Freed KV cache for req {i}, cache_loc={batch.out_cache_loc[start_idx:end_idx]}")

        if batch.next_batch_sampling_info:
            batch.next_batch_sampling_info.update_regex_vocab_mask()
            self.current_stream.synchronize()
            batch.next_batch_sampling_info.sampling_info_done.set()

        self.stream_output(batch.reqs, batch.return_logprob)

        self.token_to_kv_pool_allocator.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        if (
            self.attn_tp_rank == 0
            and self.forward_ct_decode % self.server_args.decode_log_interval == 0
        ):
            self.log_decode_stats(can_run_cuda_graph)

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

        # CRITICAL: Check weight sharing in Decode instance
        logger.info(f"[DECODE] 🚨 CRITICAL: Checking weight sharing in Decode instance...")
        try:
            # Access model through tp_worker
            model = self.tp_worker.model_runner.model
            embed_weight = model.embed_tokens.weight
            embed_checksum = torch.sum(embed_weight.data).item()
            embed_ptr = embed_weight.data_ptr()
            logger.info(f"[DECODE] 🚨 EMBEDDING: checksum={embed_checksum:.6f}, ptr=0x{embed_ptr:x}")

            # Check specific token embeddings
            if embed_weight.shape[0] > 16:
                token_15_embedding = embed_weight[15, :5].tolist()
                token_16_embedding = embed_weight[16, :5].tolist()
                logger.info(f"[DECODE] 🚨 TOKEN 15 EMBEDDING: {token_15_embedding}")
                logger.info(f"[DECODE] 🚨 TOKEN 16 EMBEDDING: {token_16_embedding}")
        except Exception as e:
            logger.error(f"[DECODE] ❌ Failed to check weight sharing: {e}")

        # CRITICAL FIX: Add missing free_group_begin() call for KV Cache management
        # This is the key fix mentioned in the KV Cache Bug Guide!
        self.token_to_kv_pool_allocator.free_group_begin()
        logger.info(f"[DECODE] 🔧 CRITICAL FIX: Called free_group_begin() for KV Cache management")

        # DEBUG: Check if Decode instance also has KV cache to manage
        if hasattr(batch, 'out_cache_loc') and batch.out_cache_loc is not None:
            logger.info(f"[DECODE] 🔧 DEBUG: Decode batch.out_cache_loc exists: {batch.out_cache_loc}")
        else:
            logger.info(f"[DECODE] 🔧 DEBUG: Decode batch.out_cache_loc is None")

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
                    # req output_ids are set here (like original Semi-PD)
                    req.output_ids.append(next_token_id)

                    # Semi-PD: Don't check finished in Prefill stage - let Decode stage handle it
                    # This ensures requests continue to Decode stage for further token generation

                    # Always cache as unfinished since this is Prefill stage
                    if not batch.decoding_reqs or req not in batch.decoding_reqs:
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

        # CRITICAL FIX: Add missing free_group_end() call to complete KV Cache management
        # This pairs with the free_group_begin() call above
        self.token_to_kv_pool_allocator.free_group_end()
        logger.info(f"[DECODE] 🔧 CRITICAL FIX: Called free_group_end() to complete KV Cache management")

        logger.info(f"[DECODE] 🔥 process_batch_result_prefill completed for Semi-PD")