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
                # Send READY signal to DECODE to indicate socket is ready
                try:
                    self.send_to_d_instance.send_pyobj({"type": "P_SOCKET_READY", "pp": pp_rank, "socket": "p_scheduler_input"})
                    logger.info(f"[PREFILL-PP{pp_rank}] sent P_SOCKET_READY signal to DECODE")
                except Exception as e:
                    logger.warning(f"[PREFILL-PP{pp_rank}] failed to send P_SOCKET_READY: {e}")
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
        """Drain p_scheduler_input for PP0 using a simple non-blocking loop.

        Align with semipd_nopp: treat p_scheduler_input as the single work-plane
        for both TokenizedGenerate and GetNextPrefillBatchOutput. Bridge remains
        control-only (StepTag/HELLO), and is not required for progressing work.
        """
        try:
            # Only PP0/TP0 actively drains here
            if getattr(self, 'pp_rank', 0) != 0 or self.attn_tp_rank != 0:
                return (0, 0)
            if getattr(self, 'recv_from_decode_forwarded', None) is None:
                logger.error(f"[PREFILL-PP{getattr(self,'pp_rank',0)}] _drain_with_poller_pp0: recv_from_decode_forwarded is None!")
                return (0, 0)

            ps_gen = 0
            ps_auth = 0
            it = 0
            consecutive_again = 0  # 🔧 FIX: Track consecutive zmq.Again
            max_consecutive_again = 5  # 🔧 FIX: Allow up to 5 consecutive zmq.Again before giving up

            # 🔧 节流：每10000次打印一次
            try:
                import os as _os
                if _os.getenv("SGLANG_SEMIPD_TRACE") == "1":
                    if not hasattr(self, '_drain_start_count'):
                        self._drain_start_count = 0
                    self._drain_start_count += 1
                    if self._drain_start_count % 10000 == 1:
                        sock_fd = getattr(self.recv_from_decode_forwarded, 'FD', None) if self.recv_from_decode_forwarded else None
                        logger.info(f"[PREFILL-PP{self.pp_rank}] _drain START: socket={self.recv_from_decode_forwarded is not None} fd={sock_fd} max_iters={max_iters} (count={self._drain_start_count})")
            except Exception:
                pass
            while it < max_iters:
                it += 1
                try:
                    obj = self.recv_from_decode_forwarded.recv_pyobj(zmq.NOBLOCK)
                    consecutive_again = 0  # 🔧 FIX: Reset counter on successful recv
                    # Debug: log successful recv
                    try:
                        import os as _os
                        if _os.getenv("SGLANG_SEMIPD_TRACE") == "1":
                            logger.info(f"[PREFILL-PP{self.pp_rank}] _drain RECV: type={type(obj).__name__} it={it}")
                    except Exception:
                        pass
                except zmq.Again:
                    consecutive_again += 1  # 🔧 FIX: Increment counter
                    # 🔧 节流：每10000次打印一次
                    if not hasattr(self, '_drain_again_count'):
                        self._drain_again_count = 0
                    self._drain_again_count += 1
                    if self._drain_again_count % 10000 == 1:
                        if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1", "true", "yes"):
                            logger.info(f"[PREFILL-PP{self.pp_rank}] _drain zmq.Again at it={it} consecutive={consecutive_again} ps_gen={ps_gen} ps_auth={ps_auth} (count={self._drain_again_count})")
                    # 🔧 FIX: Only break after multiple consecutive zmq.Again
                    if consecutive_again >= max_consecutive_again:
                        if not hasattr(self, '_drain_giving_up_count'):
                            self._drain_giving_up_count = 0
                        self._drain_giving_up_count += 1
                        if self._drain_giving_up_count % 10000 == 1:
                            if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1", "true", "yes"):
                                logger.info(f"[PREFILL-PP{self.pp_rank}] _drain giving up after {consecutive_again} consecutive zmq.Again (count={self._drain_giving_up_count})")
                        break
                    continue  # 🔧 FIX: Continue trying instead of breaking immediately
                except Exception as e:
                    try:
                        logger.error(f"[PREFILL-PP{self.pp_rank}] _drain recv exception: {e}")
                    except Exception:
                        pass
                    break

                # Handle control dicts (HELLO) best-effort
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

                # Handle StepTag (diagnostic only)
                if isinstance(obj, StepTag):
                    try:
                        self._last_step_tag = obj
                        if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1","true","yes"):
                            logger.info(
                                f"[IPC][role=D→P][pp_rank={getattr(self,'pp_rank','?')}] RECV STEP phase={getattr(obj,'phase','?')}"
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
                        semi_pd_log_info_throttle(
                            logger,
                            key=f"pp{self.pp_rank}.recv.gen",
                            msg=f"[PREFILL-PP{self.pp_rank}] RECV p_scheduler_input TokenizedGenerateReqInput rid={getattr(obj,'rid','?')}"
                        )
                    except Exception:
                        pass
                    try:
                        self.handle_generate_request(obj)
                        ps_gen += 1
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
                                    f"[PREFILL-PP{self.pp_rank}] inbox+=auth(#rids={len(obj.rids)}) from p_scheduler_input"
                                )
                    except Exception:
                        pass
                    continue

            # Summary (throttled)
            try:
                # Additional count-based guard to avoid screen flooding
                if not hasattr(self, '_poller_summary_count'):
                    self._poller_summary_count = 0
                self._poller_summary_count += 1
                if self._poller_summary_count % 100000 == 1:
                    semi_pd_log_info_throttle(
                        logger,
                        key=f"pp{self.pp_rank}.poller.summary",
                        msg=(
                            f"[PREFILL-PP{self.pp_rank}] TRACE poller ps.gen={ps_gen} ps.auth={ps_auth} wq={len(self.waiting_queue)} inbox={len(self._auth_inbox)}"
                        ),
                        interval_ms=60000,
                    )
            except Exception:
                pass

            # 🔧 DEBUG: Log function exit (only if we received something)
            if ps_gen > 0 or ps_auth > 0:
                logger.info(f"[PREFILL-PP{self.pp_rank}] _drain_with_poller_pp0 EXIT: ps_gen={ps_gen} ps_auth={ps_auth} inbox={len(self._auth_inbox)}")

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
            # 🔧 节流：每10000次打印一次
            if not hasattr(self, '_to_extend_batch_count'):
                self._to_extend_batch_count = 0
            self._to_extend_batch_count += 1
            if self._to_extend_batch_count % 10000 == 1:
                try:
                    logger.info(f"[PREFILL-PP{self.pp_rank}] TRACE to_extend_batch rids={resp.rids} (count={self._to_extend_batch_count})")
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
        Semi-PD PREFILL scheduler for PP mode.

        Key principle: PREFILL is PASSIVE and COMMAND-DRIVEN, but NON-BLOCKING in PP mode.
        - PP0: Proposes candidates and NON-BLOCKING checks for authorization
        - PP>0: NON-BLOCKING checks for authorization forwarded by DECODE

        This ensures PP event loop doesn't block while maintaining Semi-PD authorization flow.
        """
        # 🔧 DEBUG: Log function entry (throttled to avoid log flood)
        # 🔧 节流：每500次打印一次
        if not hasattr(self, '_get_next_batch_count'):
            self._get_next_batch_count = 0
        self._get_next_batch_count += 1
        if self._get_next_batch_count % 100000 == 1:
            if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1", "true", "yes"):
                wq_len = len(self.waiting_queue)
                inbox_len = len(getattr(self, '_auth_inbox', []))
                if wq_len > 0 or inbox_len > 0:
                    logger.info(f"[PREFILL-PP{self.pp_rank}] get_next_batch_to_run ENTER: wq={wq_len} inbox={inbox_len} (count={self._get_next_batch_count})")

        # 🔧 CRITICAL: Drain forwarded requests and authorizations using the proper drain function
        # This must happen in every call to ensure we don't miss messages
        if self.attn_tp_rank == 0 and self.pp_rank == 0:
            # PP0: Use the dedicated drain function
            try:
                self._drain_with_poller_pp0(max_iters=100, timeout_ms=5)  # 🔧 增加到100次
            except Exception as e:
                logger.warning(f"[PREFILL-PP{self.pp_rank}] Drain exception: {e}")
        elif self.attn_tp_rank == 0:
            # PP>0: Use simple drain loop
            try:
                while True:
                    try:
                        obj = self.recv_from_decode_forwarded.recv_pyobj(zmq.NOBLOCK)
                        if isinstance(obj, TokenizedGenerateReqInput):
                            self.handle_generate_request(obj)
                            logger.info(f"[PREFILL-PP{self.pp_rank}] Received forwarded request: {obj.rid}")
                        elif isinstance(obj, GetNextPrefillBatchOutput):
                            # Store authorization for later use (skip empty authorizations)
                            if not hasattr(self, '_auth_inbox'):
                                from collections import deque
                                self._auth_inbox = deque()
                            # 🔧 CRITICAL: Only store non-empty authorizations to avoid blocking
                            if len(obj.rids) > 0:
                                self._auth_inbox.append(obj)
                                logger.info(f"[PREFILL-PP{self.pp_rank}] Received authorization: #rids={len(obj.rids)}")
                            else:
                                logger.debug(f"[PREFILL-PP{self.pp_rank}] Skipped empty authorization: #rids=0")
                    except zmq.Again:
                        break
                    except Exception as e:
                        logger.warning(f"[PREFILL-PP{self.pp_rank}] Error draining messages: {e}")
                        break
            except Exception as e:
                logger.warning(f"[PREFILL-PP{self.pp_rank}] Drain exception: {e}")

        # 🔧 CRITICAL: Check waiting_queue for new requests
        # In Semi-PD PP mode, ALL PP stages should have requests in waiting_queue
        # - PP0: receives requests from DECODE-PP0 via IPC
        # - PP>0: receives requests from DECODE-PP>0 via IPC (forwarded from DECODE-PP0 via point_to_point_pyobj)

        # 🔧 If no requests in waiting_queue, check if we have authorization in inbox
        # If we have authorization, it means DECODE sent it but we haven't received the request yet
        # In this case, we should NOT return IDLE batch, but wait for the request
        has_pending_auth = (
            self.attn_tp_rank == 0
            and hasattr(self, '_auth_inbox')
            and self._auth_inbox
        )

        if not self.waiting_queue and not has_pending_auth:
            # No requests and no pending authorization: return IDLE batch for PP warmup
            # This prevents CPU spinning and allows PREFILL to complete warmup
            return self.get_idle_batch()

        # 🔧 ALL PP stages: Propose candidates to same-stage DECODE (if not already waiting for authorization)
        if self.attn_tp_rank == 0:
            if not hasattr(self, '_awaiting_auth'):
                self._awaiting_auth = False

            if not self._awaiting_auth:
                # Propose candidates
                n_prefill_tokens = 0
                candidates = []
                for r in self.waiting_queue:
                    if n_prefill_tokens > self.server_args.chunked_prefill_size:
                        break
                    n_prefill_tokens += len(r.origin_input_ids)
                    candidates.append(r.rid)

                if candidates:
                    req = GetNextPrefillBatchInput(rids=candidates)
                    logger.debug(f"[PREFILL-PP{self.pp_rank}] Send request to D worker: {req}")
                    self.send_to_d_instance.send_pyobj(req)
                    self._awaiting_auth = True

        # 🔧 Check if we have authorization in inbox
        resp = None
        if self.attn_tp_rank == 0:
            if hasattr(self, '_auth_inbox') and self._auth_inbox:
                resp = self._auth_inbox.popleft()
                if hasattr(self, '_awaiting_auth'):
                    self._awaiting_auth = False
                logger.debug(f"[PREFILL-PP{self.pp_rank}] Using authorization from inbox: #rids={len(resp.rids)}")

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
            # Non-rank-0 TP workers: will receive resp via broadcast above
            resp = None

        # 🔧 Build EXTEND batch from authorization
        # ALL PP stages: Build batch from waiting_queue using authorization from same-stage DECODE
        ret = None
        if resp and len(resp.rids) > 0:
            ret = self.to_extend_batch(resp)
            logger.debug(f"[PREFILL-PP{self.pp_rank}] Built EXTEND batch with {len(ret.reqs) if ret else 0} requests")

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

        # DEBUG: Log PP configuration
        try:
            semi_pd_log_info_throttle(
                logger,
                key=f"pp{pp_rank}.p_result.pp_config",
                msg=f"[PREFILL-PP{pp_rank}] PP config: pp_rank={pp_rank}, pp_size={pp_size}, is_last={is_last_pp_stage}",
                interval_ms=5000,
            )
        except Exception:
            pass

        # 🔧 方案 1：让 PREFILL-PP0 保持 batch 直到接收到 token

        # 对于 PREFILL-PP0：
        # - EXTEND 完成后，不立即清空 batch
        # - 标记 batch 为 "等待 PP1 的 token"
        # - batch 会在 event_loop_pp 的下一个 microbatch 中接收 token
        # - 接收到 token 后，通过 IPC 转发给 DECODE-PP0，然后清空 batch

        if self.pp_rank == 0:
            # PREFILL-PP0: 标记 batch 为 "等待 token"
            # 不调用父类的清空逻辑，让 batch 保持在 mbs 数组中
            if batch is not None:
                batch._waiting_for_pp1_token = True
            # 不调用父类逻辑，直接返回
            return

        # 对于 PREFILL-PP1：
        # - EXTEND 完成后，token 会通过标准 PP 逻辑发送给 PP0
        # - 调用父类逻辑处理结果
        # - 不需要清空 batch，因为 event_loop_pp 会在下一个循环中调用 get_next_batch_to_run()

        # 调用父类逻辑处理结果
        super().process_batch_result_prefill(batch, result, launch_done)

    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result,
        launch_done = None,
    ):
        """
        🔧 方案 1：PREFILL-PP0 接收到来自 PREFILL-PP1 的 token 后，转发给 DECODE-PP0

        流程：
        1. PREFILL-PP0 在 mb_id=0 时处理 EXTEND，标记 batch 为 "等待 token"
        2. PREFILL-PP1 在 mb_id=0 时生成 token，通过 NCCL 发送给 PREFILL-PP0
        3. PREFILL-PP0 在 mb_id=1 时接收 token，调用此方法
        4. 此方法将 token 通过 IPC 转发给 DECODE-PP0
        5. 清空 batch，完成流水线
        """
        # 检查是否是 PREFILL-PP0 接收到 token
        if (
            self.pp_rank == 0
            and batch is not None
            and getattr(batch, '_waiting_for_pp1_token', False)
            and result is not None
            and hasattr(result, 'next_token_ids')
            and result.next_token_ids is not None
        ):
            # PREFILL-PP0 接收到来自 PREFILL-PP1 的 token
            # 通过 IPC 转发给 DECODE-PP0
            try:
                next_token_ids_list = []
                try:
                    next_token_ids_list = result.next_token_ids.tolist()
                except Exception:
                    next_token_ids_list = list(result.next_token_ids)

                next_token_logits = None
                try:
                    if (
                        getattr(batch, 'return_logprob', False)
                        and hasattr(result, 'logits_output')
                        and result.logits_output is not None
                    ):
                        next_token_logits = result.logits_output.next_token_logits.cpu().numpy()
                except Exception:
                    pass

                rids = [r.rid for r in batch.reqs]

                # IPC 发送给 DECODE-PP0
                from sglang.srt.managers.io_struct import BatchProcessPrefillResultReq
                req = BatchProcessPrefillResultReq(
                    rids=rids,
                    next_token_ids=next_token_ids_list,
                    next_token_logits=next_token_logits,
                )
                self.send_to_d_instance.send_pyobj(req)

                # 清除标记，batch 可以被清空了
                batch._waiting_for_pp1_token = False

                # 🔧 重要：手动清空 batch
                # 因为 process_batch_result_prefill() 直接返回，不会清空 batch
                # 所以我们需要在这里手动清空
                # 注意：不调用父类逻辑，因为 batch 已经处理完成
                return

            except Exception as e:
                logger.error(f"[PREFILL-PP0] Failed to forward tokens to DECODE-PP0: {e}")

        # 调用父类逻辑处理其他情况
        super().process_batch_result(batch, result, launch_done)

    def flush_cache_wrapped(self, recv_req: FlushCacheReqInput):
        logger.info("Ignore flush cache request")

    def run_batch(self, batch: ScheduleBatch):
        # Optional trace: who actually triggers model execution on PREFILL
        if os.environ.get("SGLANG_SEMIPD_TRACE", "0").lower() in ("1", "true", "yes"):
            # 🔧 节流：每10000次打印一次
            if not hasattr(self, '_run_batch_count'):
                self._run_batch_count = 0
            self._run_batch_count += 1
            if self._run_batch_count % 10000 == 1:
                try:
                    logger.info(f"[PREFILL-PP{self.pp_rank}] TRACE run_batch(reqlen={len(batch.reqs) if batch else 0}) (count={self._run_batch_count})")
                except Exception:
                    pass
        return super().run_batch(batch)
