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
from typing import List, Optional, Union

import zmq
import torch

from sglang.semi_pd.utils import InstanceRole
from sglang.srt.managers.io_struct import (
    BatchProcessPrefillResultReq,
    FlushCacheReqInput,
    GetNextPrefillBatchInput,
    GetNextPrefillBatchOutput,
    TokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, ForwardMode
from sglang.srt.managers.scheduler import EmbeddingBatchResult, GenerationBatchResult
from sglang.srt.managers.semi_pd_scheduler import SemiPDScheduler
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import broadcast_pyobj, get_zmq_socket, point_to_point_pyobj

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

        self.enable_overlap = False
        self.chunked_rid = None
        
        # （移除临时PP通信测试导入，避免环境缺模块导致噪声告警）

        # 🔧 PP并行修复：每个PP stage都需要独立的IPC连接
        if self.attn_tp_rank == 0:
            context = zmq.Context(2)
            self.send_to_d_instance = get_zmq_socket(
                context, zmq.PUSH, port_args.d_scheduler_input_ipc_name, False
            )
            self.bridge_socket = get_zmq_socket(
                context, zmq.PULL, port_args.bridge_ipc_name, True
            )
            logger.info(f"🔧 [PREFILL-PP{pp_rank}] IPC连接已建立: d_scheduler={port_args.d_scheduler_input_ipc_name}, bridge={port_args.bridge_ipc_name}")
        else:
            self.send_to_d_instance = SimpleNamespace(send_pyobj=lambda x: None)
            self.bridge_socket = SimpleNamespace(recv_pyobj=lambda: None)
            logger.info(f"🔧 [PREFILL-PP{pp_rank}] 非主TP rank，跳过IPC连接")

    def to_extend_batch(self, resp: GetNextPrefillBatchOutput):
        """
        原版Semi-PD的核心设计：P-Scheduler使用D-Scheduler预分配的资源

        关键原理：
        1. D-Scheduler预先分配所有KV Cache资源
        2. P-Scheduler通过resp.req_pool_indices使用这些预分配的资源
        3. 通过pre_allocated_req_pool_indices参数控制ScheduleBatch的行为
        """
        can_run_list = [r for r in self.waiting_queue if r.rid in resp.rids]
        # Sort by the order of resp.rids
        can_run_list.sort(key=lambda r: resp.rids.index(r.rid))

        # 🔧 MIGRATION: 原版Semi-PD的等待队列管理逻辑
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

        # 🔧 MIGRATION: 原版Semi-PD的关键设计 - 使用D-Scheduler预分配的资源
        for i, r in enumerate(can_run_list):
            assert r.rid == resp.rids[i]
            r.extend_input_len = resp.extend_input_lens[i]
            req_pool_idx = resp.req_pool_indices[i]
            pre_len = resp.prefix_lens[i]
            # 🔑 关键：P-Scheduler直接读取D-Scheduler分配的token pool
            r.prefix_indices = self.req_to_token_pool.req_to_token[
                req_pool_idx, :pre_len
            ]
            r.fill_ids = r.origin_input_ids[: pre_len + r.extend_input_len]

        # 🔧 MIGRATION: 原版Semi-PD的ScheduleBatch创建
        # P-Scheduler有资源引用，但通过pre_allocated_req_pool_indices控制分配行为
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
        # 🔑 关键：通过pre_allocated_req_pool_indices告诉ScheduleBatch使用预分配的资源
        batch.prepare_for_extend(pre_allocated_req_pool_indices=resp.req_pool_indices)
        return batch

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        """
        🔧 使用SGLang原生Pipeline并行机制
        
        简化逻辑：
        1. 非PP0 stage需要先接收来自前一个stage的隐藏状态
        2. 如果没有等待请求，返回None让SGLang原生PP处理
        3. 有请求时使用原版Semi-PD逻辑
        4. 让SGLang原生PP机制自动处理stage间同步
        """
        
        # 🔧 INSIGHT: PP通信应该由SGLang原生机制处理，不在调度器层面手动处理
        # tp_worker.forward_batch_generation 会自动处理PP通信的发送和接收
        
        # 🔧 简化：当没有等待请求时，直接返回None
        # SGLang原生PP机制会自动处理stage间的同步和idle状态
        if not self.waiting_queue:
            logger.debug(f"[PREFILL-PP{self.pp_rank}] No waiting requests, returning None for native PP handling")
            return None

        # PP0: 继续原版Semi-PD逻辑；非PP0: 直接使用基类的原生PP批处理逻辑，避免阻塞
        # 这样可确保 PP1 PREFILL 能调度 microbatch，接收来自 PP0 的隐藏状态并产生 next_token_ids
        if self.pp_rank == 0:
            resp = None
            if self.waiting_queue and self.attn_tp_rank == 0:
                # 🔧 MIGRATION: 原版Semi-PD的候选请求选择逻辑
                n_prefill_tokens = 0
                candidates = []
                for r in self.waiting_queue:
                    if n_prefill_tokens > self.server_args.chunked_prefill_size:
                        break
                    n_prefill_tokens += len(r.origin_input_ids)
                    candidates.append(r.rid)

                req = GetNextPrefillBatchInput(rids=candidates)
                logger.debug(f"Send request to D worker: {req}")
                self.send_to_d_instance.send_pyobj(req)
                resp = self.bridge_socket.recv_pyobj()
                logger.debug(f"Recv response from D worker: {resp}")
                assert isinstance(
                    resp, GetNextPrefillBatchOutput
                ), f"Expected GetNextPrefillBatchOutput, but got {type(resp)}"

            # 🔧 MIGRATION: 原版Semi-PD的多GPU广播逻辑
            # 修复v0.4.8兼容性：使用attn_dp_rank而不是dp_rank
            if self.attn_tp_size > 1:
                attn_tp_rank_0 = self.attn_dp_rank * self.attn_tp_size
                resp = broadcast_pyobj(
                    [resp],
                    self.attn_tp_rank,
                    self.attn_tp_cpu_group,
                    src=attn_tp_rank_0,
                )[0]

            ret = None
            if resp and len(resp.rids) > 0:
                ret = self.to_extend_batch(resp)

            # Handle DP attention
            if self.server_args.enable_dp_attention:
                ret, _ = self.prepare_dp_attn_batch(ret)

            return ret
        else:
            # 非 PP0：直接沿用原生 Scheduler 的批构建逻辑，确保本 stage 能及时运行
            return super().get_next_batch_to_run()

    def process_batch_result_prefill(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        launch_done=None,
    ):
        # 🔍 [DEBUG] 添加详细的调试日志
        logger.info(f"🔍 [PREFILL_DEBUG] process_batch_result_prefill called")
        logger.info(f"🔍 [PREFILL_DEBUG] batch type: {type(batch)}")
        logger.info(f"🔍 [PREFILL_DEBUG] batch info: reqs={len(batch.reqs) if hasattr(batch, 'reqs') else 'N/A'}")
        logger.info(f"🔍 [PREFILL_DEBUG] result type: {type(result)}")
        logger.info(f"🔍 [PREFILL_DEBUG] result is None: {result is None}")
        
        if result is not None:
            logger.info(f"🔍 [PREFILL_DEBUG] result attributes: {dir(result)}")
            logger.info(f"🔍 [PREFILL_DEBUG] result.next_token_ids type: {type(getattr(result, 'next_token_ids', None))}")
            logger.info(f"🔍 [PREFILL_DEBUG] result.next_token_ids is None: {getattr(result, 'next_token_ids', None) is None}")
            logger.info(f"🔍 [PREFILL_DEBUG] result.logits_output: {getattr(result, 'logits_output', None)}")
            
            # 检查是否有其他可能的token ID字段
            for attr in dir(result):
                if 'token' in attr.lower():
                    logger.info(f"🔍 [PREFILL_DEBUG] found token-related attr: {attr} = {getattr(result, attr, None)}")
        else:
            logger.error(f"❌ [PREFILL_DEBUG] result is None! This is the root cause of the error")
        
        next_token_logits = None
        if result.logits_output is not None:
            next_token_logits = result.logits_output.next_token_logits.cpu().numpy()

        # 🔍 [DEBUG] 检查 PP rank 和 next_token_ids
        import os
        pp_rank = int(os.environ.get('SGLANG_PP_RANK', 0))
        pp_size = int(os.environ.get('SGLANG_PP_SIZE', 1))
        is_last_pp_stage = (pp_rank == pp_size - 1)
        
        # 🔧 CRITICAL INSIGHT: 不要在调度器层面手动发送隐藏状态！
        # SGLang的tp_worker.forward_batch_generation已经自动处理了PP通信
        # 调度器只需要处理最后一个PP stage的token输出
        
        if pp_size > 1 and not is_last_pp_stage:
            # 非最后一个PP stage不发送token给DECODE，隐藏状态由SGLang原生PP机制处理
            logger.info(f"[PREFILL-PP{pp_rank}] Non-last PP stage, NOT sending tokens to DECODE")
            logger.info(f"[PREFILL-PP{pp_rank}] Hidden states already handled by SGLang native PP mechanism")
            return
            
        # 最后一个PP stage处理next_token_ids
        if result.next_token_ids is None:
            logger.warning(f"[PREFILL-PP{pp_rank}] Last PP stage but next_token_ids is None, using empty list")
            next_token_ids_list = []
        else:
            next_token_ids_list = result.next_token_ids.tolist()
            logger.info(f"[PREFILL-PP{pp_rank}] Last PP stage sending {len(next_token_ids_list)} tokens to decode")

        req = BatchProcessPrefillResultReq(
            next_token_ids=next_token_ids_list,
            next_token_logits=next_token_logits,
        )

        logger.info(f"🔍 [PREFILL_DEBUG] sending request to D instance: {req}")
        self.send_to_d_instance.send_pyobj(req)
        logger.info(f"✅ [PREFILL_DEBUG] request sent successfully")



    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        logger.info("Ignore flush cache request")
