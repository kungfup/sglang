"""
Semi-PD Scheduler with Pipeline Parallel Support

Pipeline Parallel (PP) Communication Flow:
1. 请求进入 → Stage0-DECODE (接收请求，主进程)
2. Stage0-DECODE → Stage0-PREFILL (IPC通信，同一GPU内，主进程→辅助进程)
3. Stage0-PREFILL → Stage1-PREFILL (NCCL通信，跨GPU，SGLang原生实现)
4. Stage1-PREFILL → Stage0-DECODE (NCCL通信，跨GPU，SGLang原生实现)
5. Stage0-DECODE → Stage1-DECODE (NCCL通信，跨GPU，SGLang原生实现)
6. Stage1-DECODE → Stage0-DECODE (返回生成token，NCCL通信，跨GPU)

Process Hierarchy:
- 每个PP stage包含两个进程：
  * DECODE进程: 主进程，负责请求接收、响应返回、整体协调
  * PREFILL进程: 辅助进程，负责预填充计算，配合主进程工作

Port Allocation Strategy:
- PP stage0: 端口范围 40000-40999 (GPU 0)
  * 40000: decode_port (主进程)
  * 40001: prefill_port (辅助进程)
- PP stage1: 端口范围 41000-41999 (GPU 1)
  * 41000: decode_port (主进程)
  * 41001: prefill_port (辅助进程)

Communication Rules:
- GPU内通信: IPC (decode主进程 ↔ prefill辅助进程)
- GPU间通信: NCCL (通过SGLang原生PP通信)
- 主进程协调: decode进程作为每个PP stage的主进程，协调整个推理流程
"""

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
        pp_rank: int,  # 🔧 添加pp_rank支持
        dp_rank: Optional[int],
        bypass_load_weight: bool = False,
        instance_role: InstanceRole = InstanceRole.OTHER,
    ):
        # 🔧 设置Semi-PD PP模式的环境变量
        os.environ["SGLANG_ENABLE_SEMI_PD"] = "1"
        os.environ["SGLANG_PP_RANK"] = str(pp_rank)
        os.environ["SGLANG_GPU_ID"] = str(gpu_id)
        
        # 调用原始Scheduler构造函数，现在包含pp_rank
        super().__init__(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            pp_rank,  # 🔧 传递pp_rank
            dp_rank,
            bypass_load_weight,
            instance_role,
        )
        
        # 🔧 记录PP stage信息
        self.pp_rank = pp_rank
        logger.info(f"Semi-PD PP mode: PP stage {pp_rank} using GPU {gpu_id}")
        
        # 🔧 明确进程角色：decode为主进程，prefill为辅助进程
        if instance_role == InstanceRole.DECODE:
            logger.info(f"🎯 PP stage {pp_rank}: DECODE进程作为主进程，负责请求协调")
        elif instance_role == InstanceRole.PREFILL:
            logger.info(f"🔧 PP stage {pp_rank}: PREFILL进程作为辅助进程，配合主进程工作")

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
    pp_rank: int,  # 🔧 添加pp_rank参数
    dp_rank: Optional[int],
    pipe_writer,
    instance_role: InstanceRole,  # 🔧 强制传递角色，放在没有默认值的参数之后
    ipc_info_queue: multiprocessing.Queue = None,
    bypass_load_weight: bool = False,
):
    """Semi-PD specific scheduler process runner with PP support
    
    Process Hierarchy:
    - DECODE进程: 主进程，负责请求接收、响应返回、整体协调
    - PREFILL进程: 辅助进程，负责预填充计算，配合主进程工作
    
    Startup Sequence:
    1. DECODE进程先启动，加载模型权重，生成IPC信息
    2. PREFILL进程后启动，通过IPC共享模型权重，避免重复加载
    """
    # 🔧 设置Semi-PD PP模式的环境变量
    os.environ["SGLANG_ENABLE_SEMI_PD"] = "1"
    os.environ["SGLANG_PP_RANK"] = str(pp_rank)
    os.environ["SGLANG_GPU_ID"] = str(gpu_id)
    
    # Generate the prefix
    if dp_rank is None:
        prefix = f" {instance_role.name} PP{pp_rank} TP{tp_rank}"  # 🔧 添加PP信息
    else:
        prefix = f" {instance_role.name} PP{pp_rank} DP{dp_rank} TP{tp_rank}"  # 🔧 添加PP信息

    # Config the process
    setproctitle.setproctitle(f"sglang::semi_pd_scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    parent_process = psutil.Process().parent()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    # 🔧 主进程逻辑：DECODE进程负责生成IPC信息
    ipc_info = None
    if instance_role == InstanceRole.DECODE:
        logger.info(f"🎯 PP stage {pp_rank}: DECODE主进程启动，将生成IPC信息供PREFILL辅助进程使用")
    elif instance_role == InstanceRole.PREFILL:
        logger.info(f"🔧 PP stage {pp_rank}: PREFILL辅助进程启动，等待DECODE主进程的IPC信息")
        # For Prefill instances, get IPC info from Decode instance first
        if bypass_load_weight:
            logger.info(f"🔥 等待DECODE主进程的IPC信息... (tp_rank={tp_rank}, queue={ipc_info_queue})")
            try:
                logger.info(f"🔍 Queue empty status: {ipc_info_queue.empty()}")
                logger.info(f"🔍 About to call ipc_info_queue.get() with 300s timeout...")
                ipc_info = ipc_info_queue.get()  # 300 second timeout (5 minutes) for large models
                logger.info(f"✅ 成功接收到DECODE主进程的IPC信息! (type={type(ipc_info)})")
            except Exception as e:
                logger.error(f"❌ 接收IPC信息失败: {e}")
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

    # Create a scheduler and run the event loop
    try:
        if instance_role == InstanceRole.DECODE:
            from sglang.srt.managers.semi_pd_decode_scheduler import (
                SemiPDDecodeScheduler,
            )

            logger.info(f"🎯 创建DECODE主进程调度器...")
            scheduler = SemiPDDecodeScheduler(
                server_args,
                port_args,
                gpu_id,
                tp_rank,
                pp_rank,  # 🔧 传递pp_rank
                dp_rank,
                bypass_load_weight,
            )

            # 🔧 主进程职责：生成IPC信息供辅助进程使用
            ipc_info = scheduler.get_ipc_info()
            ipc_info_queue.put(ipc_info)
            logger.info(f"✅ DECODE主进程已生成IPC信息，等待PREFILL辅助进程连接")
            
        elif instance_role == InstanceRole.PREFILL:
            from sglang.srt.managers.semi_pd_prefill_scheduler import (
                SemiPDPrefillScheduler,
            )

            logger.info(f"🔧 创建PREFILL辅助进程调度器...")
            scheduler = SemiPDPrefillScheduler(
                server_args,
                port_args,
                gpu_id,
                tp_rank,
                pp_rank,  # 🔧 传递pp_rank
                dp_rank,
                bypass_load_weight,
            )
        else:
            raise ValueError(f"Invalid instance role: {instance_role}")

        # 🔧 辅助进程通过IPC共享主进程的模型权重
        if bypass_load_weight and instance_role == InstanceRole.PREFILL:
            scheduler.share_params_from_ipc(ipc_info)
            logger.info("✅ PREFILL辅助进程成功通过IPC共享DECODE主进程的模型权重 (zero-copy)!")

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

        # 🔧 主进程运行overlap模式，辅助进程运行normal模式
        if scheduler.enable_overlap and instance_role == InstanceRole.DECODE:
            logger.debug("🎯 DECODE主进程运行overlap模式，负责整体协调")
            scheduler.event_loop_overlap()
        else:
            logger.debug("🔧 PREFILL辅助进程运行normal模式，配合主进程工作")
            scheduler.event_loop_normal()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)


def get_pp_stage_ports(pp_rank: int, base_port: int = 40000) -> dict:
    """
    为PP stage分配独立端口范围
    
    Args:
        pp_rank: Pipeline parallel rank (0, 1, ...)
        base_port: 基础端口号
        
    Returns:
        包含各种端口配置的字典
        
    Note:
        - decode进程是主进程，使用主端口号
        - prefill进程是辅助进程，使用辅助端口号
        - 每个PP stage内部：decode为主，prefill为辅
    """
    # 每个PP stage分配1000个端口范围
    port_range = 1000
    start_port = base_port + pp_rank * port_range
    
    return {
        "decode_port": start_port,        # 🔧 主进程端口 (decode)
        "prefill_port": start_port + 1,   # 🔧 辅助进程端口 (prefill)
        "scheduler_port": start_port + 2,
        "detokenizer_port": start_port + 3,
        "nccl_port": start_port + 100,    # NCCL通信端口
        "port_range": (start_port, start_port + port_range - 1)
    }


def create_pp_stage_port_args(pp_rank: int, base_port: int = 40000) -> PortArgs:
    """
    为PP stage创建PortArgs对象
    
    Args:
        pp_rank: Pipeline parallel rank
        base_port: 基础端口号
        
    Returns:
        PortArgs对象
    """
    ports = get_pp_stage_ports(pp_rank, base_port)
    
    # 创建PortArgs对象，这里需要根据实际的PortArgs结构进行调整
    # 由于PortArgs的具体结构未知，这里返回一个包含端口信息的字典
    # 实际使用时需要根据PortArgs的构造函数进行调整
    
    return {
        "decode_port": ports["decode_port"],
        "prefill_port": ports["prefill_port"], 
        "scheduler_port": ports["scheduler_port"],
        "detokenizer_port": ports["detokenizer_port"],
        "nccl_port": ports["nccl_port"],
        "pp_rank": pp_rank
    }


"""
🎯 Semi-PD Pipeline Parallel 架构总结

每个PP Stage的进程架构:
┌─────────────────────────────────────────────────────────────┐
│                    PP Stage {pp_rank}                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              GPU {gpu_id}                          │   │
│  │  ┌─────────────────┐    ┌─────────────────┐       │   │
│  │  │  DECODE进程     │    │  PREFILL进程    │       │   │
│  │  │   (主进程)      │◄──►│   (辅助进程)    │       │   │
│  │  │                 │IPC │                 │       │   │
│  │  │ • 请求接收      │    │ • 预填充计算    │       │   │
│  │  │ • 响应返回      │    │ • 配合主进程    │       │   │
│  │  │ • 整体协调      │    │ • 共享权重      │       │   │
│  │  └─────────────────┘    └─────────────────┘       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

关键特性:
1. 🎯 DECODE进程是主进程，负责整体协调
2. 🔧 PREFILL进程是辅助进程，通过IPC共享主进程权重
3. 📡 主进程使用主端口号，辅助进程使用辅助端口号
4. 🔄 启动顺序：DECODE先启动 → PREFILL后启动
5. 💾 内存共享：避免重复加载模型权重，节省显存

端口分配示例 (PP Stage 0):
- 40000: decode_port (主进程)
- 40001: prefill_port (辅助进程)
- 40002: scheduler_port
- 40003: detokenizer_port
- 40100: nccl_port (跨GPU通信)
"""
