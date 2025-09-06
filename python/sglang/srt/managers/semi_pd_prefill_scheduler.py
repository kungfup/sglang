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

        self.enable_overlap = False
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
            logger.info(f"🔧 [PREFILL-PP{pp_rank}] IPC连接已建立: d_scheduler={port_args.d_scheduler_input_ipc_name}, bridge={port_args.bridge_ipc_name}")
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
                logger.info(f"🔧 [PREFILL-PP{pp_rank}] PP模式：非主TP rank也建立IPC连接: d_scheduler={port_args.d_scheduler_input_ipc_name}, bridge={port_args.bridge_ipc_name}")
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
        🔧 使用SGLang原生Pipeline并行机制
        
        简化逻辑：
        1. 非PP0 stage需要先接收来自前一个stage的隐藏状态
        2. 如果没有等待请求，返回None让SGLang原生PP处理
        3. 有请求时使用原版Semi-PD逻辑
        4. 让SGLang原生PP机制自动处理stage间同步
        """
        
        # 🔧 INSIGHT: PP通信应该由SGLang原生机制处理，不在调度器层面手动处理
        # tp_worker.forward_batch_generation 会自动处理PP通信的发送和接收
        
        # 🔧 PP模式关键修复：确保所有PP stages都同步处理请求
        pp_size = int(os.environ.get('SGLANG_PP_SIZE', 1))
        
        if pp_size > 1:
            # PP模式：所有PP stages必须同步处理相同的请求
            logger.info(f"🔧 [PP_PREFILL] PP{self.pp_rank}: Checking for work in PP mode (pp_size={pp_size})")
            
            # 🔑 关键洞察：在PP模式下，即使PP1没有等待队列中的请求，
            # 它也需要进入工作循环来接收PP0发送的隐藏状态
            if self.pp_rank == 0:
                # PP0: 检查等待队列并获取工作
                if not self.waiting_queue:
                    logger.debug(f"[PREFILL-PP{self.pp_rank}] No waiting requests, returning None")
                    return None
                    
                logger.info(f"🔧 [PP_PREFILL] PP0: Processing {len(self.waiting_queue)} waiting requests")
            else:
                # PP1: 即使没有等待队列，也需要准备接收PP0的隐藏状态
                # 创建一个虚拟的工作batch来触发PP通信
                logger.info(f"🔧 [PP_PREFILL] PP{self.pp_rank}: Preparing to receive hidden states from PP0")
                
                # 检查是否有需要同步处理的工作
                # 通过检查PP0是否有工作来决定PP1是否需要参与
                if not hasattr(self, '_pp_sync_signal'):
                    # 简化：假设如果进入这个逻辑，说明有工作要做
                    # 实际的工作内容会通过PP通信从PP0传递过来
                    logger.info(f"🔧 [PP_PREFILL] PP{self.pp_rank}: Creating dummy batch for PP sync")
                    
                    # 返回None让SGLang原生PP机制处理同步
                    # 重要：不返回None，而是让代码继续执行，确保进入forward逻辑
                    pass
        else:
            # 非PP模式：原有逻辑
            if not self.waiting_queue:
                logger.debug(f"[PREFILL-PP{self.pp_rank}] No waiting requests, returning None for native PP handling")
                return None

        # 🔑 关键修复：在PP模式下，所有PP stages都需要处理相同的请求！
        # PP0 和 PP1 都需要主动获取工作，而不是只有PP0获取
        resp = None
        
        if pp_size > 1:
            # PP模式：所有stages都需要参与，即使没有本地请求
            if self.pp_rank == 0:
                # PP0: 正常处理等待队列中的请求
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
                    logger.debug(f"[PREFILL-PP{self.pp_rank}] Send request to D worker: {req}")
                    self.send_to_d_instance.send_pyobj(req)
                    resp = self.bridge_socket.recv_pyobj()
                    logger.debug(f"[PREFILL-PP{self.pp_rank}] Recv response from D worker: {resp}")
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
                else:
                    # PP0没有请求，返回None
                    return None
            else:
                # PP1: 不直接获取请求，而是通过PP通信接收工作
                # 创建一个空的响应，让forward逻辑处理PP通信
                logger.info(f"🔧 [PP_PREFILL] PP{self.pp_rank}: No local requests, but may receive work from PP0")
                
                # 创建一个虚拟的响应，让代码继续执行到forward阶段
                # 在forward阶段，SGLang的PP机制会处理从PP0接收隐藏状态
                resp = GetNextPrefillBatchOutput(
                    rids=[],
                    chunked_rid=None,
                    req_pool_indices=[],
                    prefix_lens=[],
                    extend_input_lens=[],
                )
                logger.info(f"🔧 [PP_PREFILL] PP{self.pp_rank}: Created dummy response for PP sync")
        else:
            # 非PP模式：原有逻辑
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
                logger.debug(f"[PREFILL-PP{self.pp_rank}] Send request to D worker: {req}")
                self.send_to_d_instance.send_pyobj(req)
                resp = self.bridge_socket.recv_pyobj()
                logger.debug(f"[PREFILL-PP{self.pp_rank}] Recv response from D worker: {resp}")
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
        elif pp_size > 1 and self.pp_rank > 0 and resp is not None:
            # 🔑 关键修复：PP1即使没有本地请求，也需要返回一个dummy batch来进入forward逻辑
            # 这样PP1可以接收PP0发送的hidden states
            logger.info(f"🔧 [PP_PREFILL] PP{self.pp_rank}: Creating dummy batch for PP sync to enable forward pass")
            ret = ScheduleBatch(
                reqs=[],  # 空请求列表，但batch不为None
                forward_mode=ForwardMode.IDLE,  # 标记为空闲模式，但参与PP通信
                batch_is_full=False,
                seq_lens=torch.tensor([], dtype=torch.int64),  # 空的seq_lens tensor
                input_ids=torch.tensor([], dtype=torch.int64),  # 空的input_ids tensor
                req_pool_indices=torch.tensor([], dtype=torch.int64),  # 空的pool indices
                seq_lens_sum=0,  # 序列长度总和为0
            )

        # Handle DP attention
        if self.server_args.enable_dp_attention:
            ret, _ = self.prepare_dp_attn_batch(ret)

        logger.debug(f"[PREFILL-PP{self.pp_rank}] Returning batch with {len(ret.reqs) if ret else 0} requests")
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
        
        logger.info(f"🔧 [PP_PREFILL] PP{pp_rank}/{pp_size-1}: process_batch_result_prefill called")
        logger.info(f"🔧 [PP_PREFILL] result type: {type(result)}, batch reqs: {len(batch.reqs) if hasattr(batch, 'reqs') else 'N/A'}")
        
        # 🔑 核心逻辑：根据PP stage决定处理方式
        if pp_size > 1:  # PP模式
            if is_last_pp_stage:
                # PP1 PREFILL: 产生next_token_ids，发送给PP1 DECODE
                if result.next_token_ids is None:
                    logger.error(f"❌ [PP_PREFILL] PP{pp_rank}: Last PP stage but next_token_ids is None!")
                    next_token_ids_list = []
                else:
                    next_token_ids_list = result.next_token_ids.tolist()
                    logger.info(f"✅ [PP_PREFILL] PP{pp_rank}: Sending {len(next_token_ids_list)} tokens to DECODE")
                
                # 处理logits
                next_token_logits = None
                if result.logits_output is not None:
                    next_token_logits = result.logits_output.next_token_logits.cpu().numpy()
                
                # 发送结果给同stage的DECODE进程
                req = BatchProcessPrefillResultReq(
                    next_token_ids=next_token_ids_list,
                    next_token_logits=next_token_logits,
                )
                logger.info(f"🔧 [PP_PREFILL] PP{pp_rank}: Sending tokens to PP{pp_rank} DECODE")
                self.send_to_d_instance.send_pyobj(req)
                
            else:
                # PP0 PREFILL: 不发送token给DECODE，隐藏状态已在forward中发送
                logger.info(f"✅ [PP_PREFILL] PP{pp_rank}: Non-last PP stage, NOT sending tokens to DECODE")
                logger.info(f"✅ [PP_PREFILL] PP{pp_rank}: Hidden states already sent via SGLang native PP mechanism")
                logger.info(f"✅ [PP_PREFILL] PP{pp_rank}: Completing prefill process for non-last PP stage")
                
                # 🔑 关键修复：PP0 PREFILL 直接结束，不需要调用父类方法
                # 隐藏状态已经在 tp_worker.forward_batch_generation() 中自动发送给PP1
                # 调用父类方法是错误的，因为：
                # 1. PP通信已经完成
                # 2. PP0不应该处理next_token_ids 
                # 3. PP0不应该进行stream_output
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
            logger.info(f"🔧 [PP_PREFILL] Non-PP mode: Sending tokens to DECODE")
            self.send_to_d_instance.send_pyobj(req)



    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        logger.info("Ignore flush cache request")
