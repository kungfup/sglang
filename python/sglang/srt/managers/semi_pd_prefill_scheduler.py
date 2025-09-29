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
import threading

from typing import List, Optional, Union
from collections import deque
import traceback

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
    StepTag,
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
        # BOOT visibility: mark PREFILL-PP init enter for early crash diagnosis
        try:
            logger.info(f"[PREFILL-PP{pp_rank}] BOOT init enter (tp={tp_rank})")
        except Exception:
            pass


        # Do not force-disable overlap; follow server args
        self.chunked_rid = None
        # P→D candidate send gating
        self._awaiting_auth = False
        self._last_candidates_ts = 0.0
        # Inbox to store pending GetNextPrefillBatchOutput from DECODE (pre-drained)
        self._auth_inbox = deque()

        self._last_candidates = []

        # Glue-only: let DECODE be the single source of streaming.
        # Prevent PREFILL from calling stream_output in scheduler_output_processor_mixin.
        self.skip_stream_for_pp = True

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
            # Bind p_scheduler_input to receive DECODE-forwarded requests (all PP ranks)
            self.recv_from_decode_forwarded = None
            try:
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
            # Disable multi-inbox: each PREFILL only binds its own p_scheduler_input.
            self.recv_from_decode_forwarded_all = []
            logger.info(
                f"[PREFILL-PP{pp_rank}] IPC endpoints: d_scheduler={port_args.d_scheduler_input_ipc_name}, bridge(bind)={port_args.bridge_ipc_name}"
            )
            try:
                # Add a receive timeout so PREFILL won't hang forever if DECODE doesn't reply
                # Align default with design doc (README): default 200ms
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
            # 移除未使用的直连PP0通道（SGLANG_PP0_D_SCHEDULER_IPC）
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
        # Front-side short polling (no background drain thread)
        self._auth_inbox_lock = threading.Lock()
        self._drain_thread_enabled = False
        try:
            logger.info(f"[PREFILL-PP{pp_rank}] Using front-side NOBLOCK polling for control/work (no drain thread)")
        except Exception:
            pass

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

        # BOOT visibility: mark PREFILL-PP init done
        try:
            logger.info(f"[PREFILL-PP{pp_rank}] BOOT init done (tp={tp_rank})")
        except Exception:
            pass

    def _drain_with_poller_pp0(self, max_iters: int = 32, timeout_ms: int = 5):
        """Drain p_scheduler_input and bridge with a ZMQ Poller for PP0.

        This reduces the chance of missing initial control/work messages
        and helps align the first EXTEND tick.
        """
        try:
            use_poller = os.environ.get("SGLANG_SEMIPD_PP0_POLLER", "1").lower() in ("1","true","yes")
            if not use_poller:
                return (0, 0)
            if getattr(self, 'pp_rank', 0) != 0 or self.attn_tp_rank != 0:
                return (0, 0)
            poller = zmq.Poller()
            sock_map = {}
            if getattr(self, 'recv_from_decode_forwarded', None) is not None:
                poller.register(self.recv_from_decode_forwarded, zmq.POLLIN)
                sock_map[self.recv_from_decode_forwarded] = 'p_scheduler_input'
            if getattr(self, 'bridge_socket', None) is not None:
                poller.register(self.bridge_socket, zmq.POLLIN)
                sock_map[self.bridge_socket] = 'bridge'
            ps_gen = 0
            ps_auth = 0
            it = 0
            while it < max_iters:
                it += 1
                evts = dict(poller.poll(timeout_ms))
                if not evts:
                    break
                # Prioritize p_scheduler_input over bridge to avoid starving Generate
                ps_list = []
                br_list = []
                for s, _ in evts.items():
                    (_ps_list := ps_list if sock_map.get(s) == 'p_scheduler_input' else br_list).append(s)
                for s in ps_list + br_list:
                    src = sock_map.get(s, 'unknown')
                    try:
                        obj = s.recv_pyobj(zmq.NOBLOCK)
                    except Exception:
                        continue
                    # Control dicts: HELLO
                    if isinstance(obj, dict):
                        try:
                            if obj.get("type") == "HELLO":
                                self._handshake_done = True
                                try:
                                    self.send_to_d_instance.send_pyobj({"type": "HELLO_ACK", "pp": self.pp_rank})
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        continue
                    # StepTag control
                    if isinstance(obj, StepTag):
                        try:
                            self._last_step_tag = obj
                            if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1","true","yes"):
                                logger.info(
                                    f"[IPC][role=D→P][pp_rank={getattr(self,'pp_rank','?')}] RECV STEP on {src} phase={getattr(obj,'phase','?')}"
                                )
                        except Exception:
                            pass
                        continue
                    # Work-plane messages
                    from sglang.srt.managers.io_struct import (
                        TokenizedGenerateReqInput,
                        GetNextPrefillBatchOutput,
                    )
                    if isinstance(obj, TokenizedGenerateReqInput):
                        try:
                            # Pre-log receive (always throttled)
                            try:
                                semi_pd_log_info_throttle(
                                    logger,
                                    key=f"pp{self.pp_rank}.recv.gen",
                                    msg=f"[PREFILL-PP{self.pp_rank}] RECV {src} TokenizedGenerateReqInput rid={getattr(obj,'rid','?')}"
                                )
                            except Exception:
                                pass
                            self.handle_generate_request(obj)
                            ps_gen += 1
                            if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1","true","yes"):
                                logger.info(
                                    f"[PREFILL-PP{self.pp_rank}] TRACE handled {src} TokenizedGenerateReqInput rid={getattr(obj,'rid','?')}"
                                )
                        except Exception as _e:
                            try:
                                semi_pd_log_info_throttle(
                                    logger,
                                    key=f"pp{self.pp_rank}.recv.gen.error",
                                    msg=f"[PREFILL-PP{self.pp_rank}] handle_generate_request error: {_e}"
                                )
                            except Exception:
                                pass
                        continue
                    if isinstance(obj, GetNextPrefillBatchOutput):
                        try:
                            # Drop empty authorizations on PP0 to avoid inbox flooding
                            if not (getattr(self, 'pp_rank', 0) == 0 and len(getattr(obj, 'rids', []) or []) == 0):
                                if hasattr(self, '_auth_inbox_lock'):
                                    with self._auth_inbox_lock:
                                        self._auth_inbox.append(obj)
                                else:
                                    self._auth_inbox.append(obj)
                                ps_auth += 1
                                if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1","true","yes") and len(getattr(obj, 'rids', []) or []) > 0:
                                    logger.info(
                                        f"[PREFILL-PP{self.pp_rank}] inbox+=auth(#rids={len(obj.rids)}) from {src}"
                                    )
                        except Exception:
                            pass
                        continue
            # Throttled poller summary
            try:
                semi_pd_log_info_throttle(
                    logger,
                    key=f"pp{self.pp_rank}.poller.summary",
                    msg=(
                        f"[PREFILL-PP{self.pp_rank}] TRACE poller ps.gen={ps_gen} ps.auth={ps_auth} wq={len(self.waiting_queue)} inbox={len(self._auth_inbox)}"
                    ),
                    interval_ms=1000,
                )
            except Exception:
                pass
            return (ps_gen, ps_auth)
        except Exception:
            return (0, 0)

    def maybe_sleep_on_idle(self):
        """
        PREFILL override: be responsive to Semi-PD control/work sockets when idle.
        Lightly poll p_scheduler_input and bridge to wake up on incoming events,
        without draining here (actual draining happens in get_next_batch_to_run).
        """
        try:
            poller = zmq.Poller()
            any_reg = False
            if getattr(self, 'recv_from_decode_forwarded', None) is not None:
                poller.register(self.recv_from_decode_forwarded, zmq.POLLIN)
                any_reg = True
            if getattr(self, 'bridge_socket', None) is not None:
                poller.register(self.bridge_socket, zmq.POLLIN)
                any_reg = True
            if any_reg:
                import os as _os
                to_ms = int(_os.environ.get("SEMI_PD_IDLE_POLL_TIMEOUT_MS", "200"))
                # Wait only; do not recv here to centralize consumption in GNBTR
                poller.poll(to_ms)
            else:
                super().maybe_sleep_on_idle()
        except Exception:
            try:
                super().maybe_sleep_on_idle()
            except Exception:
                pass


    def to_extend_batch(self, resp: GetNextPrefillBatchOutput):
        """
        原版Semi-PD的核心设计：P-Scheduler使用D-Scheduler预分配的资源

        # Optional trace: show who triggered EXTEND and the authorized rids
        if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1", "true", "yes"):
            try:
                logger.info(f"[PREFILL-PP{self.pp_rank}] TRACE to_extend_batch rids={resp.rids}")
            except Exception:
                pass

        关键原理：
        1. D-Scheduler预先分配所有KV Cache资源
        2. P-Scheduler通过resp.req_pool_indices使用这些预分配的资源
        3. 通过pre_allocated_req_pool_indices参数控制ScheduleBatch的行为
        """
        can_run_list = [r for r in self.waiting_queue if r.rid in resp.rids]
        # Sort by the order of resp.rids
        can_run_list.sort(key=lambda r: resp.rids.index(r.rid))

        # For PP>0 stages: fabricate lightweight placeholder reqs for authorized rids
        # that are not yet in waiting_queue, so downstream PREFILL can build EXTEND
        # batches without having seen local Generate requests. This follows the
        # unified-clock design where only PP0 ingests Generate.

        # Optional: allow PP0 to synthesize placeholders too when env enabled to
        # quickly align the first EXTEND tick if generate arrival races with auth.
        try:
            _pp0_synth = os.environ.get("SGLANG_SEMIPD_PP0_SYNTH_PLACEHOLDER", "1").lower() in ("1", "true", "yes")
            if getattr(self, 'pp_rank', 0) > 0 or _pp0_synth:
                from sglang.srt.managers.schedule_batch import Req
                from sglang.srt.sampling.sampling_params import SamplingParams
                existing = {r.rid for r in can_run_list}
                for i, rid in enumerate(resp.rids):
                    if rid in existing:
                        continue
                    pre_len = int(resp.prefix_lens[i]) if i < len(resp.prefix_lens) else 0
                    ext_len = int(resp.extend_input_lens[i]) if i < len(resp.extend_input_lens) else 0
                    # Minimal synthetic req: sizes only; data come from PP hidden states
                    synth = Req(
                        rid=rid,
                        origin_input_text="",
                        origin_input_ids=[0] * pre_len,
                        sampling_params=SamplingParams(max_new_tokens=max(1, ext_len)),
                    )
                    synth.prefix_indices = [0] * pre_len
                    synth.extend_input_len = ext_len
                    synth.fill_ids = [0] * (pre_len + ext_len)
                    self.waiting_queue.append(synth)
                    can_run_list.append(synth)
                    try:
                        import os as _os
                        if _os.getenv("SGLANG_SEMIPD_TRACE", "0").lower() in ("1","true","yes"):
                            logger.info(
                                f"[PREFILL-PP{self.pp_rank}] synth.placeholder rid={rid} pre={pre_len} ext={ext_len} (pp0={getattr(self,'pp_rank',0)==0})"
                            )
                    except Exception:
                        pass
        except Exception:
            pass

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

        # 🔧 MIGRATION: 优先使用本地分配；仅当DECODE有效预分配时才消费其索引
        use_prealloc = (
            hasattr(resp, 'req_pool_indices')
            and resp.req_pool_indices is not None
            and len(resp.req_pool_indices) == len(can_run_list)
            and all(x is not None for x in resp.req_pool_indices)
        )
        for i, r in enumerate(can_run_list):
            assert r.rid == resp.rids[i]
            r.extend_input_len = resp.extend_input_lens[i]
            pre_len = resp.prefix_lens[i]
            if use_prealloc:
                req_pool_idx = resp.req_pool_indices[i]
                # 直接引用已存在的prefix映射
                r.prefix_indices = self.req_to_token_pool.req_to_token[req_pool_idx, :pre_len]
            else:
                # 尚未分配：只需占位长度，真实映射由prepare_for_extend分配后再建立
                r.prefix_indices = [0] * pre_len
            # If origin_input_ids does not contain extend tokens (synthetic PP>0 case), fabricate zeros for extend.
            if len(r.origin_input_ids) >= pre_len + r.extend_input_len:
                r.fill_ids = r.origin_input_ids[: pre_len + r.extend_input_len]
            else:
                r.fill_ids = (r.origin_input_ids[:pre_len]) + [0] * int(r.extend_input_len)

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
        # 🔑 关键：如果DECODE未有效预分配，则本地分配
        if use_prealloc:
            batch.prepare_for_extend(pre_allocated_req_pool_indices=resp.req_pool_indices)
        else:
            batch.prepare_for_extend()

        return batch

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        """
        Use the standard Semi-PD prefill batching, and let the native
        SGLang PP event loop handle cross-stage send/recv.
        This method should not fabricate dummy batches for downstream PP stages.
        """
        # Entry log (throttled) to confirm loop is running on PP0
        try:
            if getattr(self, 'pp_rank', 0) == 0 and getattr(self, 'attn_tp_rank', 0) == 0:
                semi_pd_log_info_throttle(
                    logger,
                    key=f"pp{self.pp_rank}.gnbtr.enter",
                    msg=f"[PREFILL-PP{self.pp_rank}] ENTER get_next_batch_to_run",
                    interval_ms=1000,
                )
        except Exception:
            pass


        # Poll forwarded generate requests and async authorizations from all PP ranks (PP>0)
        # For PP0, prefer poller-based unified drain to avoid missing initial messages.
        try:
            self._drain_with_poller_pp0(max_iters=256, timeout_ms=10)
        except Exception:
            pass
        # Throttled socket status log to confirm which endpoints are active (no env gating)
        try:
            semi_pd_log_info_throttle(
                logger,
                key=f"pp{self.pp_rank}.sockets",
                msg=(
                    f"[PREFILL-PP{self.pp_rank}] sockets: p_scheduler={'on' if getattr(self, 'recv_from_decode_forwarded', None) is not None else 'off'} "
                    f"addr={getattr(self.port_args, 'p_scheduler_input_ipc_name', '?')} bridge={getattr(self.port_args, 'bridge_ipc_name', '?')}"
                ),
                interval_ms=1000,
            )
        except Exception:
            pass


        sockets = []
        if getattr(self, 'recv_from_decode_forwarded', None) is not None and not (
            getattr(self, 'pp_rank', 0) == 0 and getattr(self, 'attn_tp_rank', 0) == 0
        ):
            sockets = [self.recv_from_decode_forwarded]
        # diag counters for p_scheduler_input poll
        diag_gen = 0
        diag_auth_ps = 0
        if not getattr(self, '_drain_thread_enabled', False):
            for s in sockets:
                while True:
                    try:
                        obj = s.recv_pyobj(zmq.NOBLOCK)
                    except Exception:
                        break
                    try:
                        if isinstance(obj, TokenizedGenerateReqInput):
                            # Pre-log the receive (always throttled), then enqueue request
                            try:
                                semi_pd_log_info_throttle(
                                    logger,
                                    key=f"pp{self.pp_rank}.recv.gen",
                                    msg=(f"[PREFILL-PP{self.pp_rank}] RECV p_scheduler_input TokenizedGenerateReqInput rid={getattr(obj,'rid','?')}")
                                )
                            except Exception:
                                pass
                            try:
                                diag_gen += 1
                                self.handle_generate_request(obj)
                            except Exception as _e:
                                try:
                                    semi_pd_log_info_throttle(
                                        logger,
                                        key=f"pp{self.pp_rank}.recv.gen.error",
                                        msg=f"[PREFILL-PP{self.pp_rank}] handle_generate_request error: {_e}"
                                    )
                                except Exception:
                                    pass
                            continue
                        if isinstance(obj, GetNextPrefillBatchOutput):
                            # Drop empty authorizations on PP0 to avoid inbox flooding
                            if not (getattr(self, 'pp_rank', 0) == 0 and len(getattr(obj, 'rids', []) or []) == 0):
                                if hasattr(self, '_auth_inbox_lock'):
                                    with self._auth_inbox_lock:
                                        self._auth_inbox.append(obj)
                                else:
                                    self._auth_inbox.append(obj)
                                diag_auth_ps += 1
                                if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1", "true", "yes") and len(getattr(obj, 'rids', []) or []) > 0:
                                    try:
                                        logger.info(f"[PREFILL-PP{self.pp_rank}] inbox+=auth(#rids={len(obj.rids)}) from p_scheduler_input")
                                    except Exception:
                                        pass
                            continue
                    except Exception:
                        pass
        # Always-throttled PP0 poll summary (even if TRACE not enabled)
        try:
            if getattr(self, 'pp_rank', 0) == 0:
                inbox_len = len(self._auth_inbox) if hasattr(self, '_auth_inbox') else -1
                wq_len = len(self.waiting_queue) if hasattr(self, 'waiting_queue') else -1
                semi_pd_log_info_throttle(
                    logger,
                    key=f"pp{self.pp_rank}.poll.summary",
                    msg=(f"[PREFILL-PP{self.pp_rank}] poll-summary wq={wq_len} inbox={inbox_len}"),
                    interval_ms=1000,
                )
        except Exception:
            pass

        # TRACE: summarize poll stats for PP0
        try:
            if getattr(self, 'pp_rank', 0) == 0 and os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1", "true", "yes"):
                inbox_len = len(self._auth_inbox) if hasattr(self, '_auth_inbox') else -1
                wq_len = len(self.waiting_queue) if hasattr(self, 'waiting_queue') else -1
                logger.info(
                    f"[PREFILL-PP{self.pp_rank}] TRACE poll ps.gen={diag_gen} ps.auth={diag_auth_ps} wq={wq_len} inbox={inbox_len}"
                )
        except Exception:
            pass


        # Unified clock: PREFILL on all PP ranks only builds batches from explicit authorization
        # issued by same-stage DECODE. Do NOT self-schedule via parent's get_next_batch_to_run.

        # Do not early-return on empty queue; let idle fallback keep PP send/recv aligned
        # if not self.waiting_queue:
        #     logger.debug(
        #         f"[PREFILL-PP{self.pp_rank}] No waiting requests, returning None"
        #     )
        #     return None

        # 仅 PP0 主动向 DECODE 请求授权；其他 PP 段不从 DECODE 拉批次
        resp = None

        if self.waiting_queue and self.attn_tp_rank == 0 and getattr(self, 'pp_rank', 0) == 0:
            # Do not gate on HELLO; proceed to send candidates

            # 发送候选（节流/重发），由D授权
            now = time.time()
            import os
            resend_interval = float(os.environ.get("SGLANG_SEMIPD_AUTH_RESEND_S", "0.05"))
            candidates = self._last_candidates if (self._awaiting_auth and (now - self._last_candidates_ts) < resend_interval) else None
            if candidates is None:
                n_prefill_tokens = 0
                candidates = []
                for r in self.waiting_queue:
                    if n_prefill_tokens > self.server_args.chunked_prefill_size:
                        break
                    n_prefill_tokens += len(r.origin_input_ids)
                    candidates.append(r.rid)
            # TRACE: show PP0-P waiting_queue and planned candidates
            try:
                if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1", "true", "yes"):
                    logger.info(
                        f"[PREFILL-PP{self.pp_rank}] TRACE cand.prepare waiting={len(self.waiting_queue)} awaiting={self._awaiting_auth} cand={len(candidates)}"
                    )
            except Exception:
                pass
            if candidates:
                self.send_to_d_instance.send_pyobj(GetNextPrefillBatchInput(rids=candidates))
                try:
                    semi_pd_log_info_throttle(
                        logger,
                        key=f"pp{self.pp_rank}.p2d.gnp.req",
                        msg=f"[PREFILL-PP{self.pp_rank}] →D GetNextPrefillBatchInput: #rids={len(candidates)}",
                    )
                except Exception:
                    pass
                # TRACE: explicitly record the rids we sent as candidates
                try:
                    if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1", "true", "yes"):
                        logger.info(
                            f"[PREFILL-PP{self.pp_rank}] TRACE cand.send rids={list(candidates)}"
                        )
                except Exception:
                    pass
                self._awaiting_auth = True
                self._last_candidates_ts = now
                self._last_candidates = candidates
            # Robust recv: drain control dicts (e.g., HELLO) and wait up to timeout for auth
            # Prefer a short NOBLOCK loop to avoid control messages causing spurious timeouts
            import os
            resp = None
            timeout_s = float(int(os.environ.get("SEMI_PD_P2D_REQ_TIMEOUT_MS", "200"))) / 1000.0
            deadline = time.time() + timeout_s
            drained_ctrl = 0
            while time.time() < deadline:
                try:
                    obj = self.bridge_socket.recv_pyobj(zmq.NOBLOCK)
                except Exception:
                    obj = None
                if obj is None:
                    time.sleep(0.003)
                    continue
                # Handle control dicts (HELLO) and continue draining
                if isinstance(obj, dict):
                    try:
                        if obj.get("type") == "HELLO":
                            self._handshake_done = True
                            try:
                                self.send_to_d_instance.send_pyobj({"type": "HELLO_ACK", "pp": self.pp_rank})
                            except Exception:
                                pass
                        drained_ctrl += 1
                    except Exception:
                        pass
                    continue
                # Optional StepTag control
                if isinstance(obj, StepTag):
                    try:
                        self._last_step_tag = obj
                        logger.info(f"[IPC][role=D→P][pp_rank={getattr(self,'pp_rank','?')}][mb_id={getattr(obj,'mb_id','-')}][phase={getattr(obj,'phase','?')}] RECV STEP")
                    except Exception:
                        pass
                    drained_ctrl += 1
                    continue
                if isinstance(obj, GetNextPrefillBatchOutput):
                    resp = obj
                    break
                # Ignore other types silently
            if isinstance(resp, GetNextPrefillBatchOutput):
                self._awaiting_auth = False
                try:
                    semi_pd_log_info_throttle(
                        logger,
                        key=f"pp{self.pp_rank}.p2d.gnp.recv",
                        msg=f"[PREFILL-PP{self.pp_rank}] ←D GetNextPrefillBatchOutput: #rids={len(resp.rids)} (ctrl_drained={drained_ctrl})",
                    )
                except Exception:
                    pass
            else:
                # No authorization this round. Do not return early; fall through to idle micro-batch to keep PP aligned.
                try:
                    semi_pd_log_info_throttle(
                        logger,
                        key=f"pp{self.pp_rank}.p2d.gnp.timeout",
                        msg=f"[PREFILL-PP{self.pp_rank}] auth wait timeout; drained_ctrl={drained_ctrl}; will retry",
                    )
                except Exception:
                    pass
                # continue; ret remains None and idle fallback (if pp_size>1) will be used

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

        # Strict gating: only build EXTEND batch from explicitly authorized rids
        ret = None
        # Enqueue received authorization (if any) into inbox for strict gating
        if isinstance(resp, GetNextPrefillBatchOutput):
            # Only enqueue non-empty authorizations to build EXTEND; still clear awaiting flag via resp above
            if len(getattr(resp, 'rids', []) or []) > 0:
                self._auth_inbox.append(resp)
            resp = None

        auth = None
        if self._auth_inbox:
            auth = self._auth_inbox.popleft()
        if auth and len(auth.rids) > 0:
            # Build EXTEND; if failed to form a batch this round, do NOT drop the auth — requeue it.
            ret = self.to_extend_batch(auth)
            if (ret is None) or (not getattr(ret, 'reqs', None)):
                try:
                    self._auth_inbox.appendleft(auth)
                except Exception:
                    pass
                ret = None
            else:
                # TRACE: built EXTEND from authorization
                try:
                    import os as _os
                    if _os.getenv("SGLANG_SEMIPD_TRACE") == "1":
                        logger.info(f"[PREFILL-PP{self.pp_rank}] EXTEND from auth rids={list(auth.rids)} size={len(ret.reqs) if ret else 0}")
                except Exception:
                    pass


        # 不再为下游PP stage创建虚拟batch，交由 event_loop_pp 统一驱动

        # Handle DP attention
        if self.server_args.enable_dp_attention:
            ret, _ = self.prepare_dp_attn_batch(ret)

        # If no authorized work yet on PREFILL under PP, do NOT fabricate an idle batch.
        # Unified clock rule: when idle, both sides must avoid NCCL; return None to skip PP recv/send.
        if ret is None and getattr(self, 'pp_size', 1) > 1:
            try:
                import os as _os
                if _os.getenv("SGLANG_SEMIPD_TRACE") == "1":
                    semi_pd_log_info_throttle(
                        logger,
                        key=f"pp{self.pp_rank}.idle",
                        msg=f"[PREFILL-PP{self.pp_rank}] return IDLE(no NCCL)",
                        interval_ms=int(os.environ.get("SEMI_PD_IDLE_LOG_INTERVAL_MS", "1000")),
                    )
            except Exception:
                pass
            # keep ret=None
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
        # diag: enter hook (always-throttled, no env gating)
        try:
            mode = getattr(batch, 'forward_mode', None)
            is_ext = bool(mode and mode.is_extend())
            from sglang.srt.utils import semi_pd_log_info_throttle as _th
            _th(
                logger,
                key=f"pp{getattr(self,'pp_rank','?')}.p_result.enter",
                msg=f"[PREFILL-PP{getattr(self,'pp_rank','?')}] ENTER process_batch_result_prefill is_extend={is_ext}",
                interval_ms=1000,
            )
        except Exception:
            pass


        # 获取PP配置：优先使用对象属性，其次使用pp_group，最后才退回环境变量
        try:
            pp_rank = getattr(self, 'pp_rank', None)
            pp_size = getattr(self, 'pp_size', None)
            if pp_rank is not None and pp_size is not None:
                is_last_pp_stage = (pp_rank == pp_size - 1)
            elif hasattr(self, 'pp_group') and self.pp_group is not None:
                is_last_pp_stage = self.pp_group.is_last_rank
                pp_rank = getattr(self, 'pp_rank', pp_rank)
                pp_size = getattr(self, 'pp_size', pp_size)
            else:
                pp_rank = int(os.environ.get('SGLANG_PP_RANK', 0))
                pp_size = int(os.environ.get('SGLANG_PP_SIZE', 1))
                is_last_pp_stage = (pp_rank == pp_size - 1)
        except Exception:
            pp_rank = getattr(self, 'pp_rank', None)
            pp_size = getattr(self, 'pp_size', None)
            if pp_rank is None or pp_size is None:
                pp_rank = int(os.environ.get('SGLANG_PP_RANK', 0))
                pp_size = int(os.environ.get('SGLANG_PP_SIZE', 1))
            is_last_pp_stage = (pp_rank == pp_size - 1)

        # keep logs minimal

        # 在最后PP段，无论是否产生token，都通知同段DECODE继续（空列表表示仅完成EXTEND）
        if is_last_pp_stage:
            # Be robust if result is None
            next_token_ids_list = []
            next_token_logits = None
            if result is not None and getattr(result, 'next_token_ids', None) is not None:
                try:
                    next_token_ids_list = result.next_token_ids.tolist()
                except Exception:
                    next_token_ids_list = list(result.next_token_ids)
            try:
                if (
                    result is not None
                    and getattr(batch, 'return_logprob', False)
                    and getattr(result, 'logits_output', None) is not None
                ):
                    next_token_logits = result.logits_output.next_token_logits.cpu().numpy()
            except Exception:
                pass
            rids = [r.rid for r in batch.reqs]
            # diag: will send (always-throttled)
            try:
                semi_pd_log_info_throttle(
                    logger,
                    key=f"pp{pp_rank}.p2d.send.will",
                    msg=f"[PREFILL-PP{pp_rank}] →D will send prefill_result: tokens={len(next_token_ids_list)} rids={rids}",
                    interval_ms=1000,
                )
            except Exception:
                pass
            req = BatchProcessPrefillResultReq(
                rids=rids,
                next_token_ids=next_token_ids_list,
                next_token_logits=next_token_logits,
            )
            self.send_to_d_instance.send_pyobj(req)
            # diag: sent (always-throttled)
            try:
                semi_pd_log_info_throttle(
                    logger,
                    key=f"pp{pp_rank}.p2d.send.done",
                    msg=f"[PREFILL-PP{pp_rank}] →D sent prefill_result",
                    interval_ms=1000,
                )
            except Exception:
                pass
            return
        # 非最后段：不处理tokens，也不调用父类；让 event_loop_pp 根据
        # result.pp_hidden_states_proxy_tensors 自动触发原生PP跨段发送。
        # 这里直接返回，避免父类按 next_token_ids 逻辑进行 zip(...) 而报错。
        return



    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        logger.info("Ignore flush cache request")

    def run_batch(self, batch: ScheduleBatch):
        # Optional trace: who actually triggers model execution on PREFILL
        if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1", "true", "yes"):
            try:
                logger.info(f"[PREFILL-PP{self.pp_rank}] TRACE run_batch(reqlen={len(batch.reqs) if batch else 0})")
            except Exception:
                pass
        return super().run_batch(batch)
