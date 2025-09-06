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

        # Do not force-disable overlap; follow server args
        self.chunked_rid = None
        
        # （移除临时PP通信测试导入，避免环境缺模块导致噪声告警）

        # 🔧 PP并行修复：每个PP stage都需要独立的IPC连接
        # 🔑 关键修复：在PP模式下，每个PP stage都需要处理请求！
        if self.attn_tp_rank == 0:
            context = zmq.Context(2)
            self.send_to_d_instance = get_zmq_socket(
                context, zmq.PUSH, port_args.d_scheduler_input_ipc_name, False
            )
            self.bridge_socket = get_zmq_socket(
                context, zmq.PULL, port_args.bridge_ipc_name, True
            )
            logger.debug(f"🔧 [PREFILL-PP{pp_rank}] IPC连接已建立: d_scheduler={port_args.d_scheduler_input_ipc_name}, bridge={port_args.bridge_ipc_name}")
            # 可选：为直连PP0 DECODE准备常驻socket（仅当提供了环境变量且当前非PP0）
            self.send_to_pp0_d_instance = None
            try:
                pp0_ipc = os.environ.get("SGLANG_PP0_D_SCHEDULER_IPC")
                if pp0_ipc and pp_rank != 0:
                    self.send_to_pp0_d_instance = get_zmq_socket(
                        context, zmq.PUSH, pp0_ipc, False
                    )
                    logger.info(
                        f"🔧 [PREFILL-PP{pp_rank}] 建立到PP0 DECODE的直连IPC: {pp0_ipc}"
                    )
            except Exception as e:
                logger.warning(f"⚠️ [PREFILL-PP{pp_rank}] 建立PP0直连IPC失败: {e}")
        else:
            # 🔑 关键修复：非主TP rank也需要IPC连接（在PP模式下）
            # 但只有在PP模式下才这样做，避免影响纯TP模式
            pp_size = int(os.environ.get('SGLANG_PP_SIZE', 1))
            if pp_size > 1:
                # PP模式：所有PP stages都需要处理请求
                context = zmq.Context(2)
                self.send_to_d_instance = get_zmq_socket(
                    context, zmq.PUSH, port_args.d_scheduler_input_ipc_name, False
                )
                self.bridge_socket = get_zmq_socket(
                    context, zmq.PULL, port_args.bridge_ipc_name, True
                )
                logger.debug(f"🔧 [PREFILL-PP{pp_rank}] PP模式：非主TP rank也建立IPC连接: d_scheduler={port_args.d_scheduler_input_ipc_name}, bridge={port_args.bridge_ipc_name}")
                self.send_to_pp0_d_instance = None
            else:
                # 纯TP模式：只有主TP rank需要IPC连接
                self.send_to_d_instance = SimpleNamespace(send_pyobj=lambda x: None)
                self.bridge_socket = SimpleNamespace(recv_pyobj=lambda: None)
                logger.info(f"🔧 [PREFILL-PP{pp_rank}] 纯TP模式：非主TP rank，跳过IPC连接")

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
        Use the standard Semi-PD prefill batching, and let the native
        SGLang PP event loop handle cross-stage send/recv.
        This method should not fabricate dummy batches for downstream PP stages.
        """
        if not self.waiting_queue:
            logger.debug(
                f"[PREFILL-PP{self.pp_rank}] No waiting requests, returning None"
            )
            return None

        # 🔑 关键修复：在PP模式下，所有PP stages都需要处理相同的请求！
        # PP0 和 PP1 都需要主动获取工作，而不是只有PP0获取
        resp = None
        
        if self.waiting_queue and self.attn_tp_rank == 0:
            # 由 DECODE 预分配资源（Semi-PD 的核心约束）
            n_prefill_tokens = 0
            candidates = []
            for r in self.waiting_queue:
                if n_prefill_tokens > self.server_args.chunked_prefill_size:
                    break
                n_prefill_tokens += len(r.origin_input_ids)
                candidates.append(r.rid)

            req = GetNextPrefillBatchInput(rids=candidates)
            logger.debug(
                f"[PREFILL-PP{self.pp_rank}] Send request to D worker: {req}"
            )
            self.send_to_d_instance.send_pyobj(req)
            resp = self.bridge_socket.recv_pyobj()
            logger.debug(
                f"[PREFILL-PP{self.pp_rank}] Recv response from D worker: {resp}"
            )
            assert isinstance(
                resp, GetNextPrefillBatchOutput
            ), f"Expected GetNextPrefillBatchOutput, but got {type(resp)}"

            # 多TP广播
            if self.attn_tp_size > 1:
                attn_tp_rank_0 = self.attn_dp_rank * self.attn_tp_size
                resp = broadcast_pyobj(
                    [resp],
                    self.attn_tp_rank,
                    self.attn_tp_cpu_group,
                    src=attn_tp_rank_0,
                )[0]
        else:
            resp = None

        ret = None
        if resp and len(resp.rids) > 0:
            ret = self.to_extend_batch(resp)
        # 不再为下游PP stage创建虚拟batch，交由 event_loop_pp 统一驱动

        # Handle DP attention
        if self.server_args.enable_dp_attention:
            ret, _ = self.prepare_dp_attn_batch(ret)

        logger.debug(
            f"[PREFILL-PP{self.pp_rank}] Returning batch with {len(ret.reqs) if ret else 0} requests"
        )
        return ret

    def process_batch_result_prefill(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        launch_done=None,
    ):
        """
        🔧 PP模式下的Semi-PD PREFILL处理逻辑
        
        关键修改：
        1. PP0 PREFILL: 不发送token给DECODE，但必须调用父类方法触发PP通信
        2. PP1 PREFILL: 产生next_token_ids，发送给PP1 DECODE
        """
        import os
        
        # 获取PP配置
        pp_rank = int(os.environ.get('SGLANG_PP_RANK', 0))
        pp_size = int(os.environ.get('SGLANG_PP_SIZE', 1))
        is_last_pp_stage = (pp_rank == pp_size - 1)
        
        logger.debug(f"🔧 [PP_PREFILL] PP{pp_rank}/{pp_size-1}: process_batch_result_prefill called")
        logger.debug(f"🔧 [PP_PREFILL] result type: {type(result)}, batch reqs: {len(batch.reqs) if hasattr(batch, 'reqs') else 'N/A'}")
        
        # 🔑 核心逻辑：根据PP stage决定处理方式
        if pp_size > 1:  # PP模式
            if is_last_pp_stage:
                # PP1 PREFILL: 产生next_token_ids，发送给PP1 DECODE
                if result.next_token_ids is None:
                    logger.debug(f"❌ [PP_PREFILL] PP{pp_rank}: Last PP stage but next_token_ids is None!")
                    next_token_ids_list = []
                else:
                    next_token_ids_list = result.next_token_ids.tolist()
                    logger.debug(f"✅ [PP_PREFILL] PP{pp_rank}: Sending {len(next_token_ids_list)} tokens to DECODE")
                
                # 处理logits
                next_token_logits = None
                # 仅在需要返回logprob时传输logits，减少不必要拷贝
                try:
                    if getattr(batch, 'return_logprob', False) and result.logits_output is not None:
                        next_token_logits = result.logits_output.next_token_logits.cpu().numpy()
                except Exception:
                    pass
                
                # 发送结果给同stage的DECODE进程
                req = BatchProcessPrefillResultReq(
                    next_token_ids=next_token_ids_list,
                    next_token_logits=next_token_logits,
                )
                logger.debug(f"🔧 [PP_PREFILL] PP{pp_rank}: Sending tokens to PP{pp_rank} DECODE")
                self.send_to_d_instance.send_pyobj(req)
                
            else:
                # PP0 PREFILL: 不发送token给DECODE，但必须执行父类逻辑以推进状态机
                logger.debug(
                    f"✅ [PP_PREFILL] PP{pp_rank}: Non-last PP stage, delegate to parent processor"
                )
                super().process_batch_result_prefill(batch, result, launch_done)
                return
                
        else:  # 非PP模式，使用原来的逻辑
            if result.next_token_ids is None:
                next_token_ids_list = []
            else:
                next_token_ids_list = result.next_token_ids.tolist()
            
            # 处理logits
            next_token_logits = None
            if result.logits_output is not None:
                next_token_logits = result.logits_output.next_token_logits.cpu().numpy()
            
            req = BatchProcessPrefillResultReq(
                next_token_ids=next_token_ids_list,
                next_token_logits=next_token_logits,
            )
            logger.debug(f"🔧 [PP_PREFILL] Non-PP mode: Sending tokens to DECODE")
            self.send_to_d_instance.send_pyobj(req)



    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        logger.info("Ignore flush cache request")

    def run_batch(self, batch: ScheduleBatch):
        """Ensure PREFILL(last PP stage)立即触发与同 stage DECODE 的交接。

        原生 event_loop_pp 只在“接收下一microbatch输出”时调用process_batch_result，
        对于被解耦的 PREFILL 进程，在最后一段需要立刻把token通过IPC交付DECODE，
        所以这里在父类run_batch返回后调用process_batch_result_prefill。
        """
        ret = super().run_batch(batch)
        try:
            from sglang.semi_pd.utils import InstanceRole as _IR
            if (
                getattr(self.server_args, 'enable_semi_pd', False)
                and getattr(self, 'instance_role', None) == _IR.PREFILL
                and batch is not None
                and hasattr(batch, 'forward_mode')
                and batch.forward_mode.is_extend()
            ):
                # 判断是否最后PP段
                is_last = False
                if hasattr(self, 'pp_group') and self.pp_group is not None:
                    is_last = self.pp_group.is_last_rank
                else:
                    pp_rank = int(os.environ.get('SGLANG_PP_RANK', 0))
                    pp_size = int(os.environ.get('SGLANG_PP_SIZE', 1))
                    is_last = (pp_rank == pp_size - 1)

                if is_last:
                    logger.debug("🧩 [PREFILL_HOOK] last PP stage detected, triggering IPC handoff")
                    self.process_batch_result_prefill(batch, ret)
        except Exception:
            logger.exception("[PREFILL_HOOK] failed; fallback without IPC handoff")
        return ret
