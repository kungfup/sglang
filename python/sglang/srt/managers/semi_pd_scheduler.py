import faulthandler
import logging
import multiprocessing
import os
import signal
import time
from http import HTTPStatus
from typing import Optional

import psutil
import setproctitle

from sglang.semi_pd.utils import InstanceRole
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req

# Compatibility layer for ImageInputs (v0.4.8 uses MultimodalInputs)
try:
    from sglang.srt.managers.schedule_batch import ImageInputs
except ImportError:
    # Create a compatibility class for v0.4.8
    import dataclasses
    from typing import List, Optional, Union
    import torch
    import numpy as np

    @dataclasses.dataclass
    class ImageInputs:
        """Compatibility class for ImageInputs in v0.4.8"""
        pixel_values: Union[torch.Tensor, np.array]
        image_hashes: Optional[list] = None
        image_sizes: Optional[list] = None
        image_offsets: Optional[list] = None
        image_pad_len: Optional[list] = None
        pad_values: Optional[list] = None
        modalities: Optional[list] = None
        num_image_tokens: Optional[int] = None

        @staticmethod
        def from_dict(obj: dict):
            """Create ImageInputs from dictionary for compatibility"""
            ret = ImageInputs(
                pixel_values=obj["pixel_values"],
                image_hashes=obj.get("image_hashes"),
            )
            # Use image hash as fake token_ids for prefix matching
            if ret.image_hashes:
                ret.pad_values = [x % (1 << 30) for x in ret.image_hashes]
            return ret
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import validate_input_length
from sglang.srt.server_args import PortArgs, SemiPDPortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class SemiPDScheduler(Scheduler):
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        bypass_load_weight: bool = False,
        instance_role: InstanceRole = InstanceRole.OTHER,
    ):
        # 🔧 CRITICAL FIX: 原始Semi-PD不使用pp_rank参数
        # 直接调用原始Scheduler构造函数，跳过pp_rank
        import torch.distributed as dist
        from sglang.srt.managers.scheduler import Scheduler

        # 手动初始化，绕过v0.4.8的pp_rank检查
        self.server_args = server_args
        self.port_args = port_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.bypass_load_weight = bypass_load_weight
        self.instance_role = instance_role

        # 调用原始Semi-PD兼容的初始化逻辑
        self._init_semi_pd_compatible()

    def _init_semi_pd_compatible(self):
        """
        Semi-PD兼容的初始化逻辑，绕过v0.4.8的pp_rank检查
        """
        # 直接调用原始Scheduler的__init__，但跳过pp_rank参数
        from sglang.srt.managers.scheduler import Scheduler

        # 临时设置pp_rank=0以满足v0.4.8的要求
        original_init = Scheduler.__init__

        def patched_init(self_inner, *args, **kwargs):
            # 在args中插入pp_rank=0
            if len(args) >= 4:  # server_args, port_args, gpu_id, tp_rank
                new_args = args[:4] + (0,) + args[4:]  # 插入pp_rank=0
                return original_init(self_inner, *new_args, **kwargs)
            return original_init(self_inner, *args, **kwargs)

        # 临时替换__init__方法
        Scheduler.__init__ = patched_init
        try:
            Scheduler.__init__(
                self,
                self.server_args,
                self.port_args,
                self.gpu_id,
                self.tp_rank,
                self.dp_rank,
                self.bypass_load_weight,
                self.instance_role,
            )
        finally:
            # 恢复原始__init__方法
            Scheduler.__init__ = original_init

    def add_to_waiting_queue(self, req: Req):
        """
        原版Semi-PD的请求队列管理逻辑
        撤回的请求具有高优先级，插入队列头部
        """
        if req.is_retracted:
            self.waiting_queue.insert(0, req)
        else:
            self.waiting_queue.append(req)

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        """
        原版Semi-PD的请求处理逻辑
        主要变化：
          - 禁用grammar功能
          - 处理撤回的请求
          - 使用add_to_waiting_queue管理队列
        """
        logger.info(f"New request {recv_req.rid}, #tokens: {len(recv_req.input_ids)}")

        # Create a new request
        if (
            recv_req.session_params is None
            or recv_req.session_params.id is None
            or recv_req.session_params.id not in self.sessions
        ):

            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                fake_input_ids = [1] * seq_length
                recv_req.input_ids = fake_input_ids

            # Handle custom logit processor passed to the request
            custom_logit_processor = recv_req.custom_logit_processor
            if (
                not self.server_args.enable_custom_logit_processor
                and custom_logit_processor is not None
            ):
                logger.warning(
                    "The SGLang server is not configured to enable custom logit processor."
                    "The custom logit processor passed in will be ignored."
                    "Please set --enable-custom-logits-processor to enable this feature."
                )
                custom_logit_processor = None

            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                stream=recv_req.stream,
                lora_path=recv_req.lora_path,
                input_embeds=recv_req.input_embeds,
                custom_logit_processor=custom_logit_processor,
                return_hidden_states=recv_req.return_hidden_states,
                eos_token_ids=self.model_config.hf_eos_token_id,
            )
            req.tokenizer = self.tokenizer

            if (
                recv_req.session_params is not None
                and recv_req.session_params.id is not None
            ):
                req.finished_reason = FINISH_ABORT(
                    f"Invalid request: session id {recv_req.session_params.id} does not exist"
                )
                # 🔧 MIGRATION: 使用原版Semi-PD的队列管理
                self.add_to_waiting_queue(req)
                return
        else:
            # Create a new request from a previous session
            session = self.sessions[recv_req.session_params.id]
            req = session.create_req(recv_req, self.tokenizer)
            if isinstance(req.finished_reason, FINISH_ABORT):
                # 🔧 MIGRATION: 使用原版Semi-PD的队列管理
                self.add_to_waiting_queue(req)
                return

        # Handle multimodal inputs
        # 🔧 v0.4.8 COMPATIBILITY: image_inputs -> mm_inputs
        if recv_req.mm_inputs is not None:
            # 🔧 v0.4.8: For now, skip complex multimodal processing in Semi-PD
            # This maintains compatibility while avoiding complex multimodal logic
            logger.warning("Multimodal inputs detected but skipped in Semi-PD mode for v0.4.8 compatibility")

            # Basic validation to prevent oversized inputs
            if len(req.origin_input_ids) >= self.max_req_input_len:
                error_msg = (
                    "Multimodal prompt is too long. "
                    f"Input length {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                )
                logger.error(error_msg)
                req.origin_input_ids = [0]
                req.mm_inputs = None
                req.sampling_params.max_new_tokens = 0
                req.finished_reason = FINISH_ABORT(
                    error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
                )
                # 🔧 MIGRATION: 使用原版Semi-PD的队列管理
                self.add_to_waiting_queue(req)
                return

        # Validate prompts length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            req.origin_input_ids = [0]
            req.sampling_params.max_new_tokens = 0
            # 🔧 MIGRATION: 使用原版Semi-PD的队列管理
            self.add_to_waiting_queue(req)
            return

        # Copy more attributes
        if recv_req.logprob_start_len == -1:
            # By default, only return the logprobs for output tokens
            req.logprob_start_len = len(req.origin_input_ids) - 1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )

        # 🔧 MIGRATION: 原版Semi-PD禁用grammar功能以避免复杂性
        # 注释掉grammar相关逻辑，直接添加到等待队列
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            assert self.grammar_backend is not None
            if req.sampling_params.json_schema is not None:
                key = ("json", req.sampling_params.json_schema)
            elif req.sampling_params.regex is not None:
                key = ("regex", req.sampling_params.regex)
            elif req.sampling_params.ebnf is not None:
                key = ("ebnf", req.sampling_params.ebnf)
            elif req.sampling_params.structural_tag:
                key = ("structural_tag", req.sampling_params.structural_tag)

            req.grammar = self.grammar_backend.get_cached_value(key)
            if not req.grammar:
                req.grammar = self.grammar_backend.get_future_value(key)
                add_to_grammar_queue = True

        if add_to_grammar_queue:
            # SemiPD
            raise NotImplementedError("Grammar is not supported in SemiPD mode")
        else:
            # SemiPD
            self.add_to_waiting_queue(req)
    
    def get_ipc_info(self):
        return self.tp_worker.get_ipc_info()



class SemiPDStandaloneScheduler:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: SemiPDPortArgs,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
    ):
        nccl_port = port_args.s_nccl_port
        self.tp_worker = TpModelWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            bypass_load_weight=False,
            instance_role=InstanceRole.OTHER,
        )

        self.max_total_num_tokens = self.tp_worker.max_total_num_tokens
        self.max_req_input_len = self.tp_worker.max_req_input_len

    def get_ipc_info(self):
        return self.tp_worker.get_ipc_info()

    def event_loop(self):
        """
        保持存活 (event_loop): 进入一个无限 time.sleep(1) 循环，
        它的唯一目的就是"占着"GPU 显存，确保模型权重不被释放，始终可供其他进程使用。
        """
        while True:
            time.sleep(1)


class MemoryCachingContext:
    """
    Disable tensor reuse cache.

    This is used for avoiding memory caching in model loading, some of the model parameters
    which get relative small size, will reuse memory from cache pool. This will cause the IPC
    memory panic, so we disable the memory caching for real model loading.
    """

    def __init__(self, enable_caching: bool = True):
        self.enable_caching = enable_caching

    def __enter__(self):
        if not self.enable_caching:
            os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enable_caching:
            del os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"]


def run_standalone_scheduler_process(
    server_args: ServerArgs,
    port_args: SemiPDPortArgs,
    gpu_id: int,
    tp_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    bypass_load_weight: bool = False,
    p_ipc_info_queue: multiprocessing.Queue = None,
    d_ipc_info_queue: multiprocessing.Queue = None,
):
    setproctitle.setproctitle("sglang::semi_pd_standalone_scheduler")
    faulthandler.enable()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    role = "Standalone"
    # Configure the logger
    if dp_rank is None:
        configure_logger(server_args, prefix=f" {role} TP{tp_rank}")
    else:
        configure_logger(server_args, prefix=f" {role} DP{dp_rank} TP{tp_rank}")
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    # Create a scheduler and run the event loop
    try:
        with MemoryCachingContext(enable_caching=False):
            scheduler = SemiPDStandaloneScheduler(
                server_args,
                port_args,
                gpu_id,
                tp_rank,
                pp_rank,
                dp_rank,
            )
        ipc_info = scheduler.get_ipc_info()
        p_ipc_info_queue.put(ipc_info)
        d_ipc_info_queue.put(ipc_info)

        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )

        scheduler.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")


# Removed duplicate function - using the correct one below


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    ipc_info_queue: multiprocessing.Queue = None,
    bypass_load_weight: bool = False,
    instance_role: InstanceRole = InstanceRole.OTHER,
):
    """Semi-PD specific scheduler process runner"""
    # Generate the prefix
    if dp_rank is None:
        prefix = f" {instance_role.name} TP{tp_rank}"
    else:
        prefix = f" {instance_role.name} DP{dp_rank} TP{tp_rank}"

    # Config the process
    setproctitle.setproctitle(f"sglang::semi_pd_scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    parent_process = psutil.Process().parent()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # For Prefill instances, get IPC info from Decode instance first
    ipc_info = None
    if bypass_load_weight:
        logger.info(f"🔥 Receiving IPC handles from Decode instance... (tp_rank={tp_rank}, queue={ipc_info_queue})")
        try:
            logger.info(f"🔍 Queue empty status: {ipc_info_queue.empty()}")
            logger.info(f"🔍 About to call ipc_info_queue.get() with 300s timeout...")
            ipc_info = ipc_info_queue.get()  # 300 second timeout (5 minutes) for large models
            logger.info(f"✅ Successfully received IPC handles from Decode instance! (type={type(ipc_info)})")
        except Exception as e:
            logger.error(f"❌ Failed to receive IPC handles: {e}")
            raise

    # Configure the logger
    if dp_rank is None:
        configure_logger(server_args, prefix=f" {instance_role.name} TP{tp_rank}")
    else:
        configure_logger(
            server_args, prefix=f" {instance_role.name} DP{dp_rank} TP{tp_rank}"
        )
    suppress_other_loggers()

    from sglang.semi_pd.utils import get_device_sm_count

    real_sm = get_device_sm_count(gpu_id)
    mps_percentage = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "100")
    logger.info(f"🔥 Available SMs: {real_sm}, MPS allocation: {mps_percentage}%")
    logger.info(f"✅ CUDA MPS successfully configured for {instance_role.name} instance")

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    # IPC info already received above for Prefill instances

    # Create a scheduler and run the event loop
    try:
        if instance_role == InstanceRole.DECODE:
            from sglang.srt.managers.semi_pd_decode_scheduler import (
                SemiPDDecodeScheduler,
            )

            scheduler = SemiPDDecodeScheduler(
                server_args,
                port_args,
                gpu_id,
                tp_rank,
                dp_rank,
                bypass_load_weight,
            )

            ipc_info = scheduler.get_ipc_info()
            ipc_info_queue.put(ipc_info)
        elif instance_role == InstanceRole.PREFILL:
            from sglang.srt.managers.semi_pd_prefill_scheduler import (
                SemiPDPrefillScheduler,
            )

            scheduler = SemiPDPrefillScheduler(
                server_args,
                port_args,
                gpu_id,
                tp_rank,
                dp_rank,
                bypass_load_weight,
            )
        else:
            raise ValueError(f"Invalid instance role: {instance_role}")

        if bypass_load_weight:
            scheduler.share_params_from_ipc(ipc_info)
            logger.info("✅ Successfully shared parameters via IPC (zero-copy)!")



        scheduler.init_attention_backend()
        if instance_role == InstanceRole.DECODE:
            scheduler.init_cuda_graphs()

        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )

        logger.info("Scheduler initialized")
        logger.info(f"Scheduler disaggregation_mode: {scheduler.disaggregation_mode}")
        logger.info(f"Scheduler enable_overlap: {scheduler.enable_overlap}")
        logger.info(f"Instance role: {instance_role}")

        if scheduler.enable_overlap and instance_role == InstanceRole.DECODE:
            logger.debug("Scheduler running in overlap mode")
            scheduler.event_loop_overlap()
        else:
            logger.debug("Scheduler running in normal mode")
            scheduler.event_loop_normal()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
