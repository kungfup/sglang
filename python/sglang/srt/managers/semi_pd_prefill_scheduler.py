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
from types import SimpleNamespace
from typing import Optional, Union

import zmq
import torch

from sglang.semi_pd.utils import InstanceRole
from sglang.srt.managers.io_struct import (
    BatchProcessPrefillResultReq,
    FlushCacheReq,
    GetNextPrefillBatchInput,
    GetNextPrefillBatchOutput,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import EmbeddingBatchResult, GenerationBatchResult
from sglang.srt.managers.semi_pd_scheduler import SemiPDScheduler
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import broadcast_pyobj, get_zmq_socket

logger = logging.getLogger(__name__)


class SemiPDPrefillScheduler(SemiPDScheduler):
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        bypass_load_weight: bool = False,
    ):
        # CRITICAL FIX: Disable overlap mode for Semi-PD Prefill instance BEFORE super().__init__()
        # Overlap mode causes future token IDs (negative values) instead of real token IDs
        # Semi-PD Prefill must generate real token IDs for sampling the first token
        # NOTE: Must modify server_args BEFORE super().__init__() to affect tp_worker selection
        original_disable_overlap = server_args.disable_overlap_schedule
        server_args.disable_overlap_schedule = True
        logger.info(f"[PREFILL] 🔥 CRITICAL: Forcing disable_overlap_schedule=True for Semi-PD Prefill (was {original_disable_overlap})")

        super().__init__(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            pp_rank,
            dp_rank,
            bypass_load_weight,
            InstanceRole.PREFILL,
        )

        # Restore original setting (though it doesn't matter after tp_worker is created)
        server_args.disable_overlap_schedule = original_disable_overlap
        logger.info(f"[PREFILL] 🔥 Semi-PD Prefill scheduler initialized with enable_overlap={self.enable_overlap}")

        # CRITICAL: Verify weight sharing after initialization
        logger.info(f"[PREFILL] 🔧 CRITICAL: Verifying weight sharing after initialization...")
        from sglang.srt.managers.semi_pd_scheduler import _verify_weight_sharing
        _verify_weight_sharing(self, InstanceRole.PREFILL)

        self.chunked_rid = None

        if self.attn_tp_rank == 0:
            context = zmq.Context(2)
            self.send_to_d_instance = get_zmq_socket(
                context, zmq.PUSH, port_args.d_scheduler_input_ipc_name, False
            )
            self.bridge_socket = get_zmq_socket(
                context, zmq.PULL, port_args.bridge_ipc_name, True
            )
        else:
            self.send_to_d_instance = SimpleNamespace(send_pyobj=lambda x: None)
            self.bridge_socket = SimpleNamespace(recv_pyobj=lambda: None)

    def to_extend_batch(self, resp):  # TODO: Update for v0.4.8
        can_run_list = [r for r in self.waiting_queue if r.rid in resp.rids]
        # Sort by the order of resp.rids
        can_run_list.sort(key=lambda r: resp.rids.index(r.rid))

        if self.chunked_rid != resp.chunked_rid:
            # Last chunked req has finished prefilling, remove it from waiting queue
            new_waiting_queue = []
            for r in self.waiting_queue:
                if r.rid == self.chunked_rid:
                    continue
                if r.rid in resp.rids and r.rid != resp.chunked_rid:
                    continue
                new_waiting_queue.append(r)
            self.waiting_queue = new_waiting_queue
            self.chunked_rid = resp.chunked_rid
        else:
            self.waiting_queue = [
                r
                for r in self.waiting_queue
                if r.rid not in resp.rids or r.rid == resp.chunked_rid
            ]

        for i, r in enumerate(can_run_list):
            assert r.rid == resp.rids[i]
            # Safe handling of v0.4.8 fields
            r.extend_input_len = resp.extend_input_lens[i] if resp.extend_input_lens else 0
            req_pool_idx = resp.req_pool_indices[i] if resp.req_pool_indices else 0
            pre_len = resp.prefix_lens[i] if resp.prefix_lens else 0

            # Handle prefix indices safely
            if self.req_to_token_pool and self.req_to_token_pool.req_to_token is not None:
                r.prefix_indices = self.req_to_token_pool.req_to_token[
                    req_pool_idx, :pre_len
                ]
            else:
                r.prefix_indices = []

            r.fill_ids = r.origin_input_ids[: pre_len + r.extend_input_len]

        # Ensure req_to_token_pool is available for Semi-PD Prefill instance
        if self.req_to_token_pool is None:
            logger.error("req_to_token_pool is None in Semi-PD Prefill scheduler, updating from tp_worker")
            self.req_to_token_pool, _ = self.tp_worker.get_memory_pool()

        batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
        )
        batch.prepare_for_extend(pre_allocated_req_pool_indices=resp.req_pool_indices)

        # Debug sampling parameters
        for i, req in enumerate(can_run_list):
            logger.info(f"[PREFILL] 🔧 Request {req.rid} sampling params: temperature={req.sampling_params.temperature}, top_k={req.sampling_params.top_k}, top_p={req.sampling_params.top_p}")

        if batch.sampling_info:
            logger.info(f"[PREFILL] 🔧 Batch sampling info: temperatures={batch.sampling_info.temperatures}, is_all_greedy={batch.sampling_info.is_all_greedy}")
        else:
            logger.error(f"[PREFILL] 🔧 ERROR: batch.sampling_info is None!")

        return batch

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        # logger.info(f"[PREFILL] get_next_batch_to_run called, waiting_queue_size={len(self.waiting_queue)}, attn_tp_rank={self.attn_tp_rank}")

        # Add debugging for request reception (removed verbose logging)

        resp = None
        if self.waiting_queue and self.attn_tp_rank == 0:
            n_prefill_tokens = 0
            candidates = []
            for r in self.waiting_queue:
                if n_prefill_tokens > self.server_args.chunked_prefill_size:
                    break
                n_prefill_tokens += len(r.origin_input_ids)
                candidates.append(r.rid)

            req = GetNextPrefillBatchInput(rids=candidates)
            logger.info(f"[PREFILL] Send request to D worker: {req}")
            self.send_to_d_instance.send_pyobj(req)
            resp = self.bridge_socket.recv_pyobj()
            logger.info(f"[PREFILL] Recv response from D worker: {resp}")
            assert isinstance(
                resp, GetNextPrefillBatchOutput
            ), f"Expected GetNextPrefillBatchOutput, but got {type(resp)}"

        if self.attn_tp_size > 1:
            attn_tp_rank_0 = self.pp_rank * self.attn_tp_size
            resp = broadcast_pyobj(
                [resp],
                self.attn_tp_rank,
                self.attn_tp_cpu_group,
                src=attn_tp_rank_0,
            )[0]

        ret = None
        if resp and len(resp.rids) > 0:
            logger.info(f"[PREFILL] Creating batch from response: {resp.rids}")
            ret = self.to_extend_batch(resp)
            if ret:
                logger.info(f"[PREFILL] Successfully created batch with {len(ret.reqs)} requests")
            else:
                logger.error(f"[PREFILL] Failed to create batch from response")

        # Handle DP attention
        if self.server_args.enable_dp_attention:
            ret, _ = self.prepare_dp_attn_batch(ret)

        # Only log when there's actually a batch to process (avoid spam)
        if ret is not None:
            logger.info(f"[PREFILL] get_next_batch_to_run returning batch with {len(ret.reqs)} requests")
        return ret

    # Original process_batch_result_prefill method removed
    # Now implemented in the overridden method below

    def run_batch(
        self, batch: ScheduleBatch
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Override run_batch to ensure Semi-PD Prefill instance generates logits.

        CRITICAL FIX: Semi-PD doesn't use pipeline parallel, so we bypass the
        pp_group.is_last_rank check and always generate logits for Prefill instance.
        This is necessary because Prefill instance needs to sample the first token.
        """
        import time

        self.forward_ct += 1

        # Whether to run the profiler
        self._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # Run forward - ALWAYS generate logits for Semi-PD Prefill (no pipeline parallel)
        if self.is_generation:
            model_worker_batch = batch.get_model_worker_batch()

            # Semi-PD Prefill: Always generate logits and token IDs (bypass pipeline parallel logic)
            logger.info(f"[PREFILL] 🔥 Semi-PD Prefill: Generating logits (no pipeline parallel)")

            # CRITICAL: Check weight sharing before generation
            logger.info(f"[PREFILL] 🚨 CRITICAL: Checking weight sharing before generation...")
            embed_weight = self.model.embed_tokens.weight
            embed_checksum = torch.sum(embed_weight.data).item()
            embed_ptr = embed_weight.data_ptr()
            logger.info(f"[PREFILL] 🚨 EMBEDDING: checksum={embed_checksum:.6f}, ptr=0x{embed_ptr:x}")

            # Check specific token embeddings
            if embed_weight.shape[0] > 16:
                token_15_embedding = embed_weight[15, :5].tolist()
                token_16_embedding = embed_weight[16, :5].tolist()
                logger.info(f"[PREFILL] 🚨 TOKEN 15 EMBEDDING: {token_15_embedding}")
                logger.info(f"[PREFILL] 🚨 TOKEN 16 EMBEDDING: {token_16_embedding}")

            # DEBUG: Log sampling info before generation
            logger.info(f"[PREFILL] 🔧 DEBUG: batch.sampling_info.temperatures = {batch.sampling_info.temperatures}")
            logger.info(f"[PREFILL] 🔧 DEBUG: batch.sampling_info.top_ks = {batch.sampling_info.top_ks}")
            logger.info(f"[PREFILL] 🔧 DEBUG: batch.sampling_info.top_ps = {batch.sampling_info.top_ps}")
            logger.info(f"[PREFILL] 🔧 DEBUG: batch.sampling_info.is_all_greedy = {batch.sampling_info.is_all_greedy}")

            logits_output, next_token_ids, can_run_cuda_graph = (
                self.tp_worker.forward_batch_generation(model_worker_batch)
            )

            # DEBUG: Log logits and token generation details
            if logits_output and logits_output.next_token_logits is not None:
                logits = logits_output.next_token_logits
                logger.info(f"[PREFILL] 🔧 DEBUG: logits shape = {logits.shape}")
                logger.info(f"[PREFILL] 🔧 DEBUG: logits dtype = {logits.dtype}")
                logger.info(f"[PREFILL] 🔧 DEBUG: logits device = {logits.device}")

                # Log top-5 logits for debugging
                top_logits, top_indices = torch.topk(logits[0], k=5)
                logger.info(f"[PREFILL] 🔧 DEBUG: top-5 logits = {top_logits.tolist()}")
                logger.info(f"[PREFILL] 🔧 DEBUG: top-5 indices = {top_indices.tolist()}")

                # Check if token 16 is in top candidates
                token_16_logit = logits[0, 16].item()
                logger.info(f"[PREFILL] 🔧 DEBUG: token 16 logit value = {token_16_logit}")

            logger.info(f"[PREFILL] 🔧 DEBUG: Generated next_token_ids = {next_token_ids}")
            logger.info(f"[PREFILL] 🔧 DEBUG: next_token_ids type = {type(next_token_ids)}")
            batch.output_ids = next_token_ids
            bid = model_worker_batch.bid

            # These 2 values are needed for processing the output
            if batch.return_logprob:
                extend_input_len_per_req = [req.extend_input_len for req in batch.reqs]
                extend_logprob_start_len_per_req = [
                    req.extend_logprob_start_len for req in batch.reqs
                ]
            else:
                extend_input_len_per_req = None
                extend_logprob_start_len_per_req = None

            ret = GenerationBatchResult(
                logits_output=logits_output,  # ALWAYS include logits for Semi-PD Prefill
                pp_hidden_states_proxy_tensors=None,  # No pipeline parallel
                next_token_ids=next_token_ids,  # ALWAYS include token IDs for Semi-PD Prefill
                extend_input_len_per_req=extend_input_len_per_req,
                extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
                bid=bid,
                can_run_cuda_graph=can_run_cuda_graph,
            )
            logger.info(f"[PREFILL] 🔥 Generated logits and token IDs: {next_token_ids}")
            return ret
        else:  # embedding or reward model
            model_worker_batch = batch.get_model_worker_batch()
            embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
            ret = EmbeddingBatchResult(
                embeddings=embeddings, bid=model_worker_batch.bid
            )
            return ret

    def process_batch_result_prefill(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        launch_done=None,
    ):
        """Override the base method to handle Semi-PD specific prefill result processing.

        This method must process the output_ids for requests in the Prefill instance
        before sending results to the Decode instance.
        """
        logger.info(f"[PREFILL] 🔥 process_batch_result_prefill called with result.next_token_ids={result.next_token_ids}")
        logger.info(f"[PREFILL] 🔥 result type: {type(result)}")
        logger.info(f"[PREFILL] 🔥 result.logits_output: {result.logits_output is not None}")

        # CRITICAL: Process output_ids for requests in Prefill instance
        # This is based on SGLang's native scheduler_output_processor_mixin.py
        if self.is_generation:
            next_token_ids = result.next_token_ids.tolist()

            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                next_token_id = next_token_ids[i]
                logger.info(f"[PREFILL] 🔥 Processing req {req.rid}, next_token_id={next_token_id}")

                if req.is_chunked <= 0:
                    # CRITICAL: Add token to output_ids - this was missing!
                    req.output_ids.append(next_token_id)
                    # Semi-PD: DON'T check finished in Prefill stage - let Decode stage handle it
                    logger.info(f"[PREFILL] 🔥 Added token {next_token_id} to req {req.rid}, output_ids={req.output_ids}")

                    # NOTE: In Semi-PD Prefill instance, we don't manage KV Cache
                    # so we skip cache operations that would fail with NoneType errors
                    # Always continue to Decode stage
                    logger.info(f"[PREFILL] 🔥 Request {req.rid} continuing (skipping cache operations)")
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1

        # Process the prefill result and send to D instance
        next_token_logits = None
        if result.logits_output is not None:
            next_token_logits = result.logits_output.next_token_logits.cpu().numpy()
            logger.info(f"[PREFILL] 🔥 next_token_logits shape: {next_token_logits.shape}")

        # DEBUG: Check KV cache allocation status BEFORE sending to Decode
        avail_size = self.token_to_kv_pool_allocator.available_size()
        expected_size = self.token_to_kv_pool_allocator.size
        logger.info(f"[PREFILL] 🔧 KV Cache Status BEFORE: avail={avail_size}, expected={expected_size}, diff={avail_size - expected_size}")

        # CRITICAL FIX: Free KV Cache allocated by Prefill instance
        if hasattr(batch, 'out_cache_loc') and batch.out_cache_loc is not None:
            logger.info(f"[PREFILL] 🔧 batch.out_cache_loc exists: {batch.out_cache_loc}")
            logger.info(f"[PREFILL] 🔧 CRITICAL FIX: Freeing KV cache allocated by Prefill instance")

            # Free the KV cache allocated by Prefill - this is the key fix!
            self.token_to_kv_pool_allocator.free_group_begin()
            self.token_to_kv_pool_allocator.free(batch.out_cache_loc)
            self.token_to_kv_pool_allocator.free_group_end()

            logger.info(f"[PREFILL] ✅ CRITICAL FIX: Successfully freed KV cache in Prefill instance")
        else:
            logger.info(f"[PREFILL] 🔧 batch.out_cache_loc is None (no KV cache to free)")

        logger.info(f"[PREFILL] 🔥 Creating BatchProcessPrefillResultReq with next_token_ids={result.next_token_ids.tolist()}")
        req = BatchProcessPrefillResultReq(
            next_token_ids=result.next_token_ids.tolist(),
            next_token_logits=next_token_logits,
        )

        logger.info(f"[PREFILL] 🔥 Send response to D worker")
        self.send_to_d_instance.send_pyobj(req)

        # Handle overlap mode if enabled
        if self.enable_overlap and launch_done:
            self.tp_worker.resolve_last_batch_result(launch_done)
            self.set_next_batch_sampling_info_done(batch)

    def flush_cache_wrapped(self, recv_req: FlushCacheReq):
        logger.info("Ignore flush cache request")
