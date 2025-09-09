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
import time
import requests
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
from sglang.srt.utils import (
    broadcast_pyobj,
    get_zmq_socket,
    point_to_point_pyobj,
    semi_pd_log_info_throttle,
    semi_pd_log_every,
)

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
        # P→D candidate send gating
        self._awaiting_auth = False
        self._last_candidates_ts = 0.0
        self._last_candidates = []
        
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
            # PP>0: bind its own p_scheduler_input to receive DECODE-forwarded requests
            self.recv_from_decode_forwarded = None
            try:
                if getattr(self, 'pp_rank', 0) != 0:
                    self.recv_from_decode_forwarded = get_zmq_socket(
                        context, zmq.PULL, port_args.p_scheduler_input_ipc_name, True
                    )
                    logger.info(
                        f"[PREFILL-PP{pp_rank}] bind p_scheduler_input: {port_args.p_scheduler_input_ipc_name}"
                    )
            except Exception as _e:
                logger.warning(f"[PREFILL-PP{pp_rank}] bind p_scheduler_input failed: {_e}")
            try:
                if hasattr(self, 'pp_group') and self.pp_group is not None:
                    logger.info(
                        f"[PREFILL-PP{pp_rank}] PP group: is_last_rank={self.pp_group.is_last_rank}"
                    )
            except Exception:
                pass
            logger.info(
                f"[PREFILL-PP{pp_rank}] IPC endpoints: d_scheduler={port_args.d_scheduler_input_ipc_name}, bridge(bind)={port_args.bridge_ipc_name}"
            )
            try:
                # Add a receive timeout so PREFILL won't hang forever if DECODE doesn't reply
                timeout_ms = int(os.environ.get("SEMI_PD_P2D_REQ_TIMEOUT_MS", "200"))
                self.bridge_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
                logger.info(
                    f"[PREFILL-PP{pp_rank}] Configured bridge RCVTIMEO={timeout_ms}ms for P→D handshake"
                )
            except Exception:
                # Fallback silently if the platform/pyzmq doesn't support this option
                logger.warning(
                    f"[PREFILL-PP{pp_rank}] Failed to set RCVTIMEO on bridge socket; will block on recv_pyobj()"
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

        # Bootstrap registration (health + endpoint alignment)
        self._handshake_done = os.environ.get("SGLANG_SEMIPD_DISABLE_HANDSHAKE", "0").lower() in ("1","true","yes")
        try:
            bootstrap_port = self.server_args.disaggregation_bootstrap_port + pp_rank
            url = f"http://127.0.0.1:{bootstrap_port}/route"
            payload = {
                "role": "Prefill",
                "tp_size": self.server_args.tp_size,
                "dp_size": self.server_args.dp_size,
                "rank_ip": "127.0.0.1",
                "rank_port": 0,
                "engine_rank": self.tp_rank,
            }
            requests.put(url, json=payload, timeout=1.0)
            logger.info(f"[PREFILL-PP{pp_rank}] Registered to bootstrap at 127.0.0.1:{bootstrap_port}")
        except Exception as e:
            logger.warning(f"[PREFILL-PP{pp_rank}] Bootstrap registration failed: {e}")

        # One-shot HELLO handshake before entering event loop (best-effort)
        if self.attn_tp_rank == 0 and not getattr(self, "_handshake_done", False):
            deadline = time.time() + float(os.environ.get("SGLANG_SEMIPD_HELLO_WAIT_S", "2.0"))
            while time.time() < deadline and not self._handshake_done:
                try:
                    obj = self.bridge_socket.recv_pyobj(zmq.NOBLOCK)
                    if isinstance(obj, dict) and obj.get("type") == "HELLO":
                        self._handshake_done = True
                        self.send_to_d_instance.send_pyobj({"type": "HELLO_ACK", "pp": self.pp_rank})
                        semi_pd_log_info_throttle(
                            logger,
                            key=f"pp{self.pp_rank}.hello.ack.send.init",
                            msg=f"[PREFILL-PP{self.pp_rank}] (init) HELLO received; ACK sent",
                        )
                        break
                except zmq.Again:
                    pass
                except Exception:
                    break
                time.sleep(0.05)

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
        # Poll forwarded generate requests from same-stage DECODE (PP>0)
        if getattr(self, 'recv_from_decode_forwarded', None) is not None:
            while True:
                try:
                    obj = self.recv_from_decode_forwarded.recv_pyobj(zmq.NOBLOCK)
                except Exception:
                    break
                # Append to waiting queue directly
                try:
                    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
                    if isinstance(obj, TokenizedGenerateReqInput):
                        self.handle_generate_request(obj)
                except Exception:
                    pass

        # For non-PP0 stages: build local prefill batch without D-authorization.
        # Let upstream PP deliver hidden-state proxies; do not send candidates from here.
        try:
            if getattr(self, 'pp_rank', 0) != 0:
                # Local builder (mirrors decode's get_new_batch_prefill, without D interaction)
                if self.grammar_queue:
                    self.move_ready_grammar_requests()
                if (self.running_batch.batch_is_full or len(self.waiting_queue) == 0) and self.chunked_req is None:
                    return None
                running_bs = len(self.running_batch.reqs)
                if running_bs >= self.max_running_requests:
                    self.running_batch.batch_is_full = True
                    return None
                if self.enable_hierarchical_cache:
                    self.tree_cache.writing_check(); self.tree_cache.loading_check()
                prefix_computed = self.policy.calc_priority(self.waiting_queue)
                adder = PrefillAdder(
                    self.page_size,
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
                for req in self.waiting_queue:
                    if (
                        self.lora_paths and len(
                            lora_set | set([req.lora_path for req in adder.can_run_list]) | set([req.lora_path])
                        ) > self.max_loras_per_batch
                    ):
                        self.running_batch.batch_is_full = True
                        break
                    if running_bs + len(adder.can_run_list) >= self.max_running_requests:
                        self.running_batch.batch_is_full = True
                        break
                    req.init_next_round_input(None if prefix_computed else self.tree_cache)
                    res = adder.add_one_req(req, self.chunked_req is not None)
                    if res != AddReqResult.CONTINUE:
                        if res == AddReqResult.NO_TOKEN:
                            if self.enable_hierarchical_cache:
                                self.running_batch.batch_is_full = len(adder.can_run_list) > 0 or (
                                    self.running_batch is not None and not self.running_batch.is_empty()
                                )
                            else:
                                self.running_batch.batch_is_full = True
                        break
                can_run_list = adder.can_run_list
                if len(can_run_list) == 0:
                    return None
                self.waiting_queue = [x for x in self.waiting_queue if x not in set(can_run_list)]
                if self.enable_hierarchical_cache:
                    self.tree_cache.read_to_load_cache()
                if adder.new_chunked_req is not None:
                    assert self.chunked_req is None
                    self.chunked_req = adder.new_chunked_req
                if self.chunked_req:
                    self.chunked_req.is_chunked += 1
                if self.attn_tp_rank == 0:
                    self.log_prefill_stats(adder, can_run_list, running_bs)
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
                return new_batch
        except Exception:
            pass

        if not self.waiting_queue:
            logger.debug(
                f"[PREFILL-PP{self.pp_rank}] No waiting requests, returning None"
            )
            return None

        # 仅 PP0 主动向 DECODE 请求授权；其他 PP 段不从 DECODE 拉批次
        resp = None
        
        if self.waiting_queue and self.attn_tp_rank == 0 and getattr(self, 'pp_rank', 0) == 0:
            # Do not gate on HELLO; proceed to send candidates

            # 发送候选（节流/重发），由D授权
            now = time.time()
            resend_interval = float(os.environ.get("SGLANG_SEMIPD_AUTH_RESEND_S", "0.2"))
            candidates = self._last_candidates if (self._awaiting_auth and (now - self._last_candidates_ts) < resend_interval) else None
            if candidates is None:
                n_prefill_tokens = 0
                candidates = []
                for r in self.waiting_queue:
                    if n_prefill_tokens > self.server_args.chunked_prefill_size:
                        break
                    n_prefill_tokens += len(r.origin_input_ids)
                    candidates.append(r.rid)
            if candidates:
                self.send_to_d_instance.send_pyobj(GetNextPrefillBatchInput(rids=candidates))
                self._awaiting_auth = True
                self._last_candidates_ts = now
                self._last_candidates = candidates
                semi_pd_log_every(
                    logger,
                    key=f"pp{self.pp_rank}.p2d.wait",
                    msg=f"[PREFILL-PP{self.pp_rank}] waiting for D reply on bridge...",
                )
            # Blocking recv with socket-level RCVTIMEO; simpler and more robust than NOBLOCK loop
            resp = None
            try:
                obj = self.bridge_socket.recv_pyobj()
                if isinstance(obj, GetNextPrefillBatchOutput):
                    resp = obj
                # ignore non-auth dicts silently
            except Exception:
                resp = None
            if isinstance(resp, GetNextPrefillBatchOutput):
                semi_pd_log_info_throttle(
                    logger,
                    key=f"pp{self.pp_rank}.p2d.resp",
                    msg=(
                        f"[PREFILL-PP{self.pp_rank}] ←D GetNextPrefillBatchOutput: #rids={len(resp.rids)}"
                    ),
                )
                self._awaiting_auth = False
            else:
                # No authorization; skip this round
                return None

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
        
        # 获取PP配置（优先使用已建立的pp_group/self.pp_rank/self.pp_size）
        try:
            if hasattr(self, 'pp_group') and self.pp_group is not None:
                is_last_pp_stage = self.pp_group.is_last_rank
                pp_rank = getattr(self, 'pp_rank', None)
                pp_size = getattr(self, 'pp_size', None)
            else:
                pp_rank = int(os.environ.get('SGLANG_PP_RANK', 0))
                pp_size = int(os.environ.get('SGLANG_PP_SIZE', 1))
                is_last_pp_stage = (pp_rank == pp_size - 1)
        except Exception:
            pp_rank = int(os.environ.get('SGLANG_PP_RANK', 0))
            pp_size = int(os.environ.get('SGLANG_PP_SIZE', 1))
            is_last_pp_stage = (pp_rank == pp_size - 1)
        
        # keep logs minimal
        
        # 简化：如果本轮产生了token（仅最后PP段会有），则直接发给同段DECODE；否则走父类逻辑
        if result.next_token_ids is not None:
            try:
                next_token_ids_list = result.next_token_ids.tolist()
            except Exception:
                next_token_ids_list = list(result.next_token_ids)
            next_token_logits = None
            try:
                if getattr(batch, 'return_logprob', False) and result.logits_output is not None:
                    next_token_logits = result.logits_output.next_token_logits.cpu().numpy()
            except Exception:
                pass
            req = BatchProcessPrefillResultReq(
                next_token_ids=next_token_ids_list,
                next_token_logits=next_token_logits,
            )
            logger.info(f"[PREFILL-PP{pp_rank}] →D tokens: {len(next_token_ids_list)}")
            self.send_to_d_instance.send_pyobj(req)
            return
        # 非最后段（无token）：保持父类行为，维持原生PP推进
        super().process_batch_result_prefill(batch, result, launch_done)



    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        logger.info("Ignore flush cache request")

    def run_batch(self, batch: ScheduleBatch):
        return super().run_batch(batch)
