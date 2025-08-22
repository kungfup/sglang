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

from sglang.semi_pd.utils import InstanceRole
from sglang.srt.managers.io_struct import (
    BatchProcessPrefillResultReq,
    FlushCacheReqInput,
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
        dp_rank: Optional[int],
        pp_rank: int = 0,  # 🚀 新增：支持pipeline rank 
        bypass_load_weight: bool = True,  # prefill进程默认不加载权重
    ):
        print(f"🔥 [SemiPD-PREFILL] 启动prefill scheduler - pp_rank={pp_rank}")
        super().__init__(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            dp_rank,
            pp_rank,  # 🔥 传递pp_rank给父类
            bypass_load_weight,  # prefill进程通过IPC获取权重
            InstanceRole.PREFILL,
        )

        self.enable_overlap = False
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
        原版Semi-PD的P-D握手协议

        流程：
        1. P-Scheduler提议候选请求（有token数量限制）
        2. D-Scheduler检查资源并批准部分请求
        3. P-Scheduler使用D-Scheduler预分配的资源执行计算

        🔧 CRITICAL FIX: 当没有等待队列时，直接进入Idle模式
        """
        # 🔧 SEMI-PD IDLE MODE: 当没有等待请求时，直接返回None进入Idle模式
        if not self.waiting_queue:
            logger.debug("[PREFILL] No waiting requests, entering Idle mode")
            return None

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

    def process_batch_result_prefill(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        launch_done=None,
    ):
        # 🚀 PIPELINE FIX: 在pipeline并行模式下，只有最后一个stage生成next_token_ids
        if result.next_token_ids is None:
            # 中间pipeline stage不生成tokens，跳过处理
            logger.debug(f"[Pipeline-Stage-{getattr(self, 'pp_rank', 'unknown')}] "
                        f"Skipping token processing - intermediate stage")
            return
        
        next_token_logits = None
        if result.logits_output is not None:
            next_token_logits = result.logits_output.next_token_logits.cpu().numpy()

        next_token_ids_list = result.next_token_ids.tolist()
        req = BatchProcessPrefillResultReq(
            next_token_ids=next_token_ids_list,
            next_token_logits=next_token_logits,
        )

        logger.debug(f"[Pipeline-Stage-{getattr(self, 'pp_rank', 'unknown')}] "
                    f"Sending tokens to decode instance: {len(next_token_ids_list)} tokens")
        self.send_to_d_instance.send_pyobj(req)

    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        logger.info("Ignore flush cache request")
