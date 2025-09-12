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
"""
The entry point of inference server. (SRT = SGLang Runtime)

This file implements python APIs for the inference engine.
"""

import asyncio
import atexit
import dataclasses
import logging
import multiprocessing as mp
import os
import random
import signal
import threading
from typing import AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import zmq
import zmq.asyncio
from PIL.Image import Image

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

import torch
import uvloop

from sglang.semi_pd.utils import (
    DECODE_ENGINE_SM_PERCENTILE,
    PREFILL_ENGINE_SM_PERCENTILE,
    InstanceRole,
)
from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    GetWeightsByNameReqInput,
    ImageDataItem,
    InitWeightsUpdateGroupReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.scheduler import run_scheduler_process as run_original_scheduler_process
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, SemiPDPortArgs, ServerArgs
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    MultiprocessingSerializer,
    assert_pkg_version,
    configure_logger,
    get_zmq_socket,
    is_cuda,
    kill_process_tree,
    launch_dummy_health_check_server,
    maybe_set_triton_cache_manager,
    prepare_model_and_tokenizer,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.version import __version__

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# NOTE: Avoid eager CUDA checks at import time to prevent initializing CUDA before MM pool is created.
# Defer is_cuda() checks to runtime where necessary.


class Engine(EngineBase):
    """
    The entry point to the inference engine.

    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager both run in the main process.
    2. Inter-process communication is done through ICP (each process uses a different port) via the ZMQ library.
    """

    def __init__(self, **kwargs):
        """
        The arguments of this function is the same as `sglang/srt/server_args.py::ServerArgs`.
        Please refer to `ServerArgs` for the documentation.
        """
        if "server_args" in kwargs:
            # Directly load server_args
            server_args = kwargs["server_args"]
        else:
            # Construct server_args from kwargs
            if "log_level" not in kwargs:
                # Do not print logs by default
                kwargs["log_level"] = "error"
            server_args = ServerArgs(**kwargs)

        if server_args.enable_semi_pd:
            raise NotImplementedError("Engine API does not support Semi-PD yet.")

        # Shutdown the subprocesses automatically when the program exits
        atexit.register(self.shutdown)

        # Allocate ports for inter-process communications
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

        # Launch subprocesses
        tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses(
            server_args=server_args,
            port_args=port_args,
        )
        self.server_args = server_args
        self.tokenizer_manager = tokenizer_manager
        self.template_manager = template_manager
        self.scheduler_info = scheduler_info

        context = zmq.Context(2)
        self.send_to_rpc = get_zmq_socket(
            context, zmq.DEALER, port_args.rpc_ipc_name, True
        )

    def generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[
            Union[
                List[List[ImageDataItem]],
                List[ImageDataItem],
                ImageDataItem,
            ]
        ] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: bool = False,
        stream: bool = False,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        data_parallel_rank: Optional[int] = None,
    ) -> Union[Dict, Iterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        if self.server_args.enable_dp_attention:
            if data_parallel_rank is None:
                logger.debug("data_parallel_rank not provided, using default dispatch")
            elif data_parallel_rank < 0:
                raise ValueError("data_parallel_rank must be non-negative")
            elif data_parallel_rank >= self.server_args.dp_size:
                raise ValueError(
                    f"data_parallel_rank must be less than dp_size: {self.server_args.dp_size}"
                )

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
            return_hidden_states=return_hidden_states,
            stream=stream,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            data_parallel_rank=data_parallel_rank,
        )
        loop = asyncio.get_event_loop()
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream:

            def generator_wrapper():
                while True:
                    try:
                        chunk = loop.run_until_complete(generator.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break

            return generator_wrapper()
        else:
            ret = loop.run_until_complete(generator.__anext__())
            return ret

    async def async_generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[
            Union[
                List[List[ImageDataItem]],
                List[ImageDataItem],
                ImageDataItem,
            ]
        ] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        stream: bool = False,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        data_parallel_rank: Optional[int] = None,
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """

        if self.server_args.enable_dp_attention:
            if data_parallel_rank is None:
                logger.debug("data_parallel_rank not provided, using default dispatch")
            elif data_parallel_rank < 0:
                raise ValueError("data_parallel_rank must be non-negative")
            elif data_parallel_rank >= self.server_args.dp_size:
                raise ValueError(
                    f"data_parallel_rank must be in range [0, {self.server_args.dp_size-1}]"
                )

        logger.info(f"data_parallel_rank: {data_parallel_rank}")
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            stream=stream,
            custom_logit_processor=custom_logit_processor,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            data_parallel_rank=data_parallel_rank,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream is True:
            return generator
        else:
            return await generator.__anext__()

    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
        image_data: Optional[
            Union[
                List[List[Union[Image, str]]],
                List[Union[Image, str]],
                Union[Image, str],
            ]
        ] = None,
    ) -> Dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(text=prompt, image_data=image_data)
        loop = asyncio.get_event_loop()
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = loop.run_until_complete(generator.__anext__())
        return ret

    async def async_encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
        image_data: Optional[Union[List[str], str]] = None,
    ) -> Dict:
        """
        Asynchronous version of encode method.

        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(text=prompt, image_data=image_data)
        generator = self.tokenizer_manager.generate_request(obj, None)
        return await generator.__anext__()

    def rerank(
        self,
        prompt: Union[List[List[str]]],
    ) -> Dict:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::EmbeddingReqInput`.
        Please refer to `EmbeddingReqInput` for the documentation.
        """
        obj = EmbeddingReqInput(text=prompt, is_cross_encoder_request=True)
        loop = asyncio.get_event_loop()
        generator = self.tokenizer_manager.generate_request(obj, None)
        ret = loop.run_until_complete(generator.__anext__())
        return ret

    def shutdown(self):
        """Shutdown the engine"""
        kill_process_tree(os.getpid(), include_parent=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    def flush_cache(self):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.tokenizer_manager.flush_cache())

    def start_profile(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.tokenizer_manager.start_profile())

    def stop_profile(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.tokenizer_manager.stop_profile())

    def start_expert_distribution_record(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.tokenizer_manager.start_expert_distribution_record()
        )

    def stop_expert_distribution_record(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.tokenizer_manager.stop_expert_distribution_record()
        )

    def dump_expert_distribution_record(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.tokenizer_manager.dump_expert_distribution_record()
        )

    def get_server_info(self):
        loop = asyncio.get_event_loop()
        internal_states = loop.run_until_complete(
            self.tokenizer_manager.get_internal_state()
        )
        return {
            **dataclasses.asdict(self.tokenizer_manager.server_args),
            **self.scheduler_info,
            "internal_states": internal_states,
            "version": __version__,
        }

    def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ):
        """Initialize parameter update group."""
        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.init_weights_update_group(obj, None)
        )

    def update_weights_from_distributed(self, name: str, dtype, shape):
        """Update weights from distributed source."""
        obj = UpdateWeightsFromDistributedReqInput(
            name=name,
            dtype=dtype,
            shape=shape,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_distributed(obj, None)
        )

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update weights from distributed source. If there are going to be more updates, set `flush_cache` to be false
        to avoid duplicated cache cleaning operation."""
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(self.server_args.tp_size)
            ],
            load_format=load_format,
            flush_cache=flush_cache,
        )
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_tensor(obj, None)
        )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: Optional[str] = None,
    ):
        """Update the weights from disk inplace without re-launching the engine.

        This method allows updating the model weights from disk without restarting
        the engine. It can be used to load a different model or update weights with
        new training.
        """
        obj = UpdateWeightFromDiskReqInput(
            model_path=model_path,
            load_format=load_format,
        )

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.update_weights_from_disk(obj, None)
        )

    def get_weights_by_name(self, name: str, truncate_size: int = 100):
        """Get weights by parameter name."""
        obj = GetWeightsByNameReqInput(name=name, truncate_size=truncate_size)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.get_weights_by_name(obj, None)
        )

    def release_memory_occupation(self, tags: Optional[List[str]] = None):
        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.release_memory_occupation(obj, None)
        )

    def resume_memory_occupation(self, tags: Optional[List[str]] = None):
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.resume_memory_occupation(obj, None)
        )

    """
    Execute an RPC call on all scheduler processes.
    """

    def collective_rpc(self, method: str, **kwargs):
        obj = RpcReqInput(method=method, parameters=kwargs)
        self.send_to_rpc.send_pyobj(obj)
        recv_req = self.send_to_rpc.recv_pyobj(zmq.BLOCKY)
        assert isinstance(recv_req, RpcReqOutput)
        assert recv_req.success, recv_req.message

    def save_remote_model(self, **kwargs):
        self.collective_rpc("save_remote_model", **kwargs)

    def save_sharded_model(self, **kwargs):
        self.collective_rpc("save_sharded_model", **kwargs)

    def score(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> List[List[float]]:
        """
        Score the probability of specified token IDs appearing after the given (query + item) pair. For example:
        query = "<|user|>Is the following city the capital of France? "
        items = ["Paris <|assistant|>", "London <|assistant|>", "Berlin <|assistant|>"]
        label_token_ids = [2332, 1223] # Token IDs for "Yes" and "No"
        item_first = False

        This would pass the following prompts to the model:
        "<|user|>Is the following city the capital of France? Paris <|assistant|>"
        "<|user|>Is the following city the capital of France? London <|assistant|>"
        "<|user|>Is the following city the capital of France? Berlin <|assistant|>"
        The api would then return the probabilities of the model producing "Yes" and "No" as the next token.
        The output would look like:
        [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]]


        Args:
            query: The query text or pre-tokenized query token IDs. Must be provided.
            items: The item text(s) or pre-tokenized item token IDs. Must be provided.
            label_token_ids: List of token IDs to compute probabilities for. If None, no token probabilities will be computed.
            apply_softmax: Whether to normalize probabilities using softmax.
            item_first: If True, prepend items to query. Otherwise append items to query.

        Returns:
            List of dictionaries mapping token IDs to their probabilities for each item.
            Each dictionary in the list corresponds to one item input.

        Raises:
            ValueError: If query is not provided, or if items is not provided,
                      or if token IDs are out of vocabulary, or if logprobs are not available for the specified tokens.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.score_request(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=item_first,
                request=None,
            )
        )

    async def async_score(
        self,
        query: Optional[Union[str, List[int]]] = None,
        items: Optional[Union[str, List[str], List[List[int]]]] = None,
        label_token_ids: Optional[List[int]] = None,
        apply_softmax: bool = False,
        item_first: bool = False,
    ) -> List[List[float]]:
        """
        Asynchronous version of score method.

        See score() for detailed documentation.
        """
        return await self.tokenizer_manager.score_request(
            query=query,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
            item_first=item_first,
            request=None,
        )


def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.6.post1",
            "Please uninstall the old version and "
            "reinstall the latest version by following the instructions "
            "at https://docs.flashinfer.ai/installation.html.",
        )
    # Avoid eager CUDA checks in parent; allow deferring via env
    defer_cuda_check = os.environ.get("SGLANG_DEFER_CUDA_CHECK", "0").lower() in ("1", "true")
    if not defer_cuda_check and is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.9",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    def sigchld_handler(signum, frame):
        pid, exitcode = os.waitpid(0, os.WNOHANG)
        if exitcode != 0:
            logger.warning(
                f"Child process unexpectedly failed with {exitcode=}. {pid=}"
            )

    signal.signal(signal.SIGCHLD, sigchld_handler)

    # Register the signal handler.
    # The child processes will send SIGQUIT to this process when any error happens
    # This process then clean up the whole process tree
    def sigquit_handler(signum, frame):
        logger.error(
            "Received sigquit from a child process. It usually means the child failed."
        )
        kill_process_tree(os.getpid())

    signal.signal(signal.SIGQUIT, sigquit_handler)

    # Set mp start method
    mp.set_start_method("spawn", force=True)


def _launch_subprocesses(
    server_args: ServerArgs, port_args: Optional[PortArgs] = None
) -> Tuple[TokenizerManager, TemplateManager, Dict]:
    """
    Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
    """
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)


    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []
    if server_args.dp_size == 1:
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        scheduler_pipe_readers = []

        # 🔧 SEMI-PD FIX: 修复分布式环境初始化
        if server_args.enable_semi_pd:
            # Semi-PD模式：PP0和PP1的DECODE进程共享同一个分布式环境
            # 关键：所有PP stage的DECODE进程使用相同的world_size和rank
            nnodes_per_tp_group = max(server_args.nnodes // server_args.pp_size, 1)
            tp_size_per_node = server_args.tp_size // nnodes_per_tp_group

            # 🔑 关键修复：DECODE进程的分布式环境
            # PP0-DECODE: rank=0, world_size=pp_size
            # PP1-DECODE: rank=1, world_size=pp_size
            # 这样它们可以通信，而不是每个PP stage独立
            for pp_rank in range(server_args.pp_size):
                for tp_rank in range(tp_size_per_node):
                    reader, writer = mp.Pipe(duplex=False)
                    phys_gpu_id = (
                        server_args.base_gpu_id
                        + (pp_rank * tp_size_per_node)
                        + (tp_rank * server_args.gpu_id_step)
                    )
                    isolate_child = os.environ.get("SGLANG_ISOLATE_CHILD_VISIBLE", "0").lower() in ("1", "true")
                    if isolate_child:
                        # Limit child to a single physical GPU via CUDA_VISIBLE_DEVICES
                        prev_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(phys_gpu_id)
                        # Optional diagnostic: disable NCCL P2P for this child when enabled
                        prev_nccl_p2p = os.environ.get("NCCL_P2P_DISABLE")
                        if os.environ.get("SGLANG_DEBUG_NCCL_P2P_DISABLE") in ("1", "true", "True"):
                            os.environ["NCCL_P2P_DISABLE"] = "1"
                        logger.info(
                            f"[LAUNCH] PP{pp_rank} TP{tp_rank} set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
                        )

                    try:
                        proc = mp.Process(
                            target=run_original_scheduler_process,
                            args=(
                                server_args,
                                port_args,
                                0 if isolate_child else phys_gpu_id,
                                tp_rank,
                                pp_rank,
                                None,
                                writer,
                            ),
                        )
                        with memory_saver_adapter.configure_subprocess():
                            proc.start()
                    finally:
                        if isolate_child:
                            # Restore parent's env to avoid leaking settings
                            if prev_cuda_visible is None:
                                del os.environ["CUDA_VISIBLE_DEVICES"]
                            else:
                                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible
                            if prev_nccl_p2p is None and "NCCL_P2P_DISABLE" in os.environ:
                                del os.environ["NCCL_P2P_DISABLE"]
                            elif prev_nccl_p2p is not None:
                                os.environ["NCCL_P2P_DISABLE"] = prev_nccl_p2p

                    scheduler_procs.append(proc)
                    scheduler_pipe_readers.append(reader)
        else:
            # 原版SGLang的pipeline并行逻辑
            nnodes_per_tp_group = max(server_args.nnodes // server_args.pp_size, 1)
            tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
            tp_rank_range = range(
                tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
                tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
            )

            pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
            pp_rank_range = range(
                pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group),
                pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group + 1),
            )

            for pp_rank in pp_rank_range:
                for tp_rank in tp_rank_range:
                    reader, writer = mp.Pipe(duplex=False)
                    gpu_id = (
                        server_args.base_gpu_id
                        + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                        + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                    )
                    proc = mp.Process(
                        target=run_original_scheduler_process,
                        args=(
                            server_args,
                            port_args,
                            gpu_id,
                            tp_rank,
                            pp_rank,
                            None,
                            writer,
                        ),
                    )

                    with memory_saver_adapter.configure_subprocess():
                        proc.start()
                    scheduler_procs.append(proc)
                    scheduler_pipe_readers.append(reader)
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()
        scheduler_procs.append(proc)

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        for reader in scheduler_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None, None

        launch_dummy_health_check_server(server_args.host, server_args.port)

        for proc in scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
        return None, None, None

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(
            server_args,
            port_args,
        ),
    )
    detoken_proc.start()

    # Launch tokenizer process
    tokenizer_manager = TokenizerManager(server_args, port_args)

    # Initialize templates
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )

    # Wait for the model to finish loading
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            scheduler_procs[i].join()
            logger.error(f"Exit code: {scheduler_procs[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)

    # Assume all schedulers have the same scheduler_info
    scheduler_info = scheduler_infos[0]
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    return tokenizer_manager, template_manager, scheduler_info


def _launch_semi_pd_subprocesses(
    server_args: ServerArgs,
) -> Tuple[TokenizerManager, TemplateManager, Dict]:
    # Defer any CUDA checks in the parent process to avoid initializing CUDA before MM pool creation
    os.environ.setdefault("SGLANG_DEFER_CUDA_CHECK", "1")

    # Locals for optional early tokenizer init
    early_tokenizer_initialized = False
    tokenizer_manager: Optional[TokenizerManager] = None
    template_manager: Optional[TemplateManager] = None

    from sglang.srt.managers.semi_pd_scheduler import run_scheduler_process

    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)


    logger.info(f"🚀 [SEMI-PD] Starting Semi-PD subprocesses with config: {server_args=}")
    logger.info(f"🔧 [SEMI-PD] TP_SIZE={server_args.tp_size}, PP_SIZE={server_args.pp_size}, DP_SIZE={server_args.dp_size}, NNODES={server_args.nnodes}")

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []
    scheduler_infos = []
    if server_args.dp_size == 1:
        # Allocate ports for inter-process communications
        logger.info("🔧 [SEMI-PD] Allocating Semi-PD specific ports...")

        # 🔧 PP + TP Configuration (参考原生逻辑)
        nnodes_per_tp_group = max(server_args.nnodes // server_args.pp_size, 1)
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group),
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group + 1),
        )

        logger.info(f"🔧 [SEMI-PD] PP+TP configuration:")
        logger.info(f"  📊 nnodes_per_tp_group={nnodes_per_tp_group}")
        logger.info(f"  📊 tp_size_per_node={tp_size_per_node}, tp_rank_range={list(tp_rank_range)}")
        logger.info(f"  📊 pp_size_per_node={pp_size_per_node}, pp_rank_range={list(pp_rank_range)}")

        # 🔧 为每个PP stage创建独立的端口配置
        port_args_per_pp: List[SemiPDPortArgs] = []
        for pp_rank in pp_rank_range:
            port_args = SemiPDPortArgs.init_new(server_args, pp_rank=pp_rank)
            port_args_per_pp.append(port_args)
            logger.info(f"🔧 [SEMI-PD] PP{pp_rank} port allocation:")
            logger.info(f"  📡 Tokenizer IPC: {port_args.tokenizer_ipc_name}")
            logger.info(f"  📡 Detokenizer IPC: {port_args.detokenizer_ipc_name}")
            logger.info(f"  📡 Standalone Scheduler IPC: {port_args.s_scheduler_input_ipc_name}")
            logger.info(f"  📡 Prefill Scheduler IPC: {port_args.p_scheduler_input_ipc_name}")
            logger.info(f"  📡 Decode Scheduler IPC: {port_args.d_scheduler_input_ipc_name}")
            logger.info(f"  📡 Bridge IPC: {port_args.bridge_ipc_name}")

        # Early init Tokenizer and Templates BEFORE launching any CUDA/NCCL children
        # This ensures MM process pool (fork) happens when parent has not touched CUDA
        logger.info("🔧 [SEMI-PD] Early-initializing Tokenizer and Templates before schedulers")
        tokenizer_manager = TokenizerManager(server_args, port_args_per_pp[0])
        template_manager = TemplateManager()
        try:
            template_manager.initialize_templates(
                tokenizer_manager=tokenizer_manager,
                model_path=server_args.model_path,
                chat_template=server_args.chat_template,
                completion_template=server_args.completion_template,
            )
            logger.info("✅ [SEMI-PD] Early templates initialized (chat/completion)")
        except Exception:
            logger.exception("[SEMI-PD] Early template initialization failed; proceeding with HF default")
        logger.info("✅ [SEMI-PD] Early Tokenizer and Template managers initialized")
        early_tokenizer_initialized = True

        logger.info(f"🔧 [SEMI-PD] Port allocation completed for {len(port_args_per_pp)} PP stages")
        # 移除未使用的直连PP0 IPC注入，避免路径分叉与困惑

        # 🔧 关键修复：为Semi-PD + TP模式分配独立的分布式初始化端口
        # DECODE和PREFILL进程使用不同的分布式初始化端口，但共享相同的NCCL端口
        if server_args.enable_semi_pd:
            import random
            # 🔧 修复：Semi-PD + PP模式下的分布式环境配置
            if server_args.pp_size > 1:
                # PP模式：DECODE和PREFILL使用不同端口，但PP stage间使用相同NCCL端口
                base_dist_port = 40000 + random.randint(100, 199)

                # DECODE和PREFILL使用不同的分布式初始化端口，避免端口冲突
                decode_dist_port = base_dist_port
                prefill_dist_port = base_dist_port + 100  # 错开100个端口

                # 每个PP stage使用独立的NCCL端口
                base_nccl_port = 40000 + random.randint(200, 299)
                for pp_rank in range(server_args.pp_size):
                    pp_nccl_port = base_nccl_port + pp_rank * 10
                    logger.info(f"🔧 [SEMI-PD+PP] PP{pp_rank} NCCL端口: {pp_nccl_port}")

                logger.info(f"🔧 [SEMI-PD+PP] DECODE进程分布式端口: {decode_dist_port}")
                logger.info(f"🔧 [SEMI-PD+PP] PREFILL进程分布式端口: {prefill_dist_port}")
                logger.info(f"🔧 [SEMI-PD+PP] PP模式：DECODE和PREFILL使用独立端口，PP stage间使用相同NCCL端口")
            else:
                # TP模式：DECODE和PREFILL进程使用不同的分布式初始化端口
                decode_dist_port = 40000 + random.randint(100, 199)
                prefill_dist_port = 40000 + random.randint(200, 299)
                logger.info(f"🔧 [SEMI-PD] DECODE和PREFILL进程使用独立的分布式初始化端口")

            os.environ["SGLANG_DECODE_DIST_PORT"] = str(decode_dist_port)
            os.environ["SGLANG_PREFILL_DIST_PORT"] = str(prefill_dist_port)
            logger.info(f"🔧 [SEMI-PD] DECODE进程分布式端口: {decode_dist_port}")
            logger.info(f"🔧 [SEMI-PD] PREFILL进程分布式端口: {prefill_dist_port}")
            logger.info(f"🔧 [SEMI-PD] DECODE分布式环境: tcp://127.0.0.1:{decode_dist_port}")
            logger.info(f"🔧 [SEMI-PD] PREFILL分布式环境: tcp://127.0.0.1:{prefill_dist_port}")

        # Launch tensor parallel scheduler processes
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        p_scheduler_pipe_readers = []
        d_scheduler_pipe_readers = []

        # 为每个PP stage创建IPC队列
        total_pp_stages = len(pp_rank_range)
        p_ipc_info_queues: List[List[mp.Queue]] = [
            [mp.Queue() for _ in range(tp_size_per_node)]
            for _ in range(total_pp_stages)
        ]

        # 🔧 PHASE 1: Launch Decode (D) instances first - they load model weights
        logger.info("🚀 [SEMI-PD] PHASE 1: Launching Decode instances (model weight loaders)...")
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                queue_idx = tp_rank % tp_size_per_node
                p_ipc_info_queue = p_ipc_info_queues[pp_rank - pp_rank_range.start][queue_idx]

                # 使用对应PP stage的端口配置
                port_args = port_args_per_pp[pp_rank - pp_rank_range.start]

                # GPU ID计算 (参考原生逻辑)
                phys_gpu_id = (
                    server_args.base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )

                # Set CUDA MPS for Decode instance (100% SMs)
                os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
                    DECODE_ENGINE_SM_PERCENTILE
                )
                logger.info(
                    f"🚀 [SEMI-PD] Launching D instance PP{pp_rank} TP{tp_rank} on GPU{phys_gpu_id} with "
                    f"{os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']}% SMs, "
                    f"NCCL port: {port_args.d_nccl_port}"
                )

                d_reader, d_writer = mp.Pipe(duplex=False)
                isolate_child = os.environ.get("SGLANG_ISOLATE_CHILD_VISIBLE", "0").lower() in ("1", "true")
                if isolate_child:
                    # Per-process GPU visibility isolation for DECODE
                    prev_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(phys_gpu_id)
                    prev_nccl_p2p = os.environ.get("NCCL_P2P_DISABLE")
                    if os.environ.get("SGLANG_DEBUG_NCCL_P2P_DISABLE") in ("1", "true", "True"):
                        os.environ["NCCL_P2P_DISABLE"] = "1"
                    logger.info(
                        f"[LAUNCH] D PP{pp_rank} TP{tp_rank} set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
                    )
                try:
                    d_proc = mp.Process(
                        target=run_scheduler_process,
                        args=(
                            server_args,
                            port_args,
                            0 if isolate_child else phys_gpu_id,  # child sees only one GPU when isolated
                            tp_rank,
                            None,  # dp_rank=None for DECODE instances
                            d_writer,
                            p_ipc_info_queue,
                            False,  # bypass_load_weight=False - D instances load weights
                            InstanceRole.DECODE,
                            pp_rank,  # 🔧 正确传递pp_rank参数
                        ),
                    )
                    with memory_saver_adapter.configure_subprocess():
                        d_proc.start()
                finally:
                    if isolate_child:
                        if prev_cuda_visible is None:
                            del os.environ["CUDA_VISIBLE_DEVICES"]
                        else:
                            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible
                        if prev_nccl_p2p is None and "NCCL_P2P_DISABLE" in os.environ:
                            del os.environ["NCCL_P2P_DISABLE"]
                        elif prev_nccl_p2p is not None:
                            os.environ["NCCL_P2P_DISABLE"] = prev_nccl_p2p
                scheduler_procs.append(d_proc)
                d_scheduler_pipe_readers.append(d_reader)
                logger.info(f"✅ [SEMI-PD] D instance PP{pp_rank} TP{tp_rank} started with PID: {d_proc.pid}")

        # Wait for all Decode instances to be ready and share IPC info
        logger.info("⏳ [SEMI-PD] Waiting for all Decode instances to be ready...")
        for i, reader in enumerate(d_scheduler_pipe_readers):
            pp_rank = pp_rank_range[i // len(tp_rank_range)]
            tp_rank = tp_rank_range[i % len(tp_rank_range)]
            logger.info(f"⏳ [SEMI-PD] Waiting for D instance PP{pp_rank} TP{tp_rank} to be ready...")
            data = reader.recv()
            assert data["status"] == "ready"
            scheduler_infos.append(data)
            server_args.max_total_tokens = data["max_total_num_tokens"]
            logger.info(f"✅ [SEMI-PD] D instance PP{pp_rank} TP{tp_rank} ready, max_total_tokens: {data['max_total_num_tokens']}")

            # 验证同一PP stage内的max_total_tokens一致性
            if i % len(tp_rank_range) > 0:
                assert (
                    server_args.max_total_tokens
                    == data["max_total_num_tokens"]
                )
                logger.info(f"✅ [SEMI-PD] D instance PP{pp_rank} TP{tp_rank} max_total_tokens validation passed")

        # 🔧 PHASE 2: Launch Prefill (P) instances - they share weights via IPC
        logger.info("🚀 [SEMI-PD] PHASE 2: Launching Prefill instances (weight sharers)...")
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                queue_idx = tp_rank % tp_size_per_node
                p_ipc_info_queue = p_ipc_info_queues[pp_rank - pp_rank_range.start][queue_idx]

                # 使用对应PP stage的端口配置
                port_args = port_args_per_pp[pp_rank - pp_rank_range.start]

                # GPU ID计算 (与DECODE实例相同)
                phys_gpu_id = (
                    server_args.base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )

                # Set CUDA MPS for Prefill instance (80% SMs)
                os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
                    PREFILL_ENGINE_SM_PERCENTILE
                )

                logger.info(
                    f"🚀 [SEMI-PD] Launching P instance PP{pp_rank} TP{tp_rank} on GPU{phys_gpu_id} with "
                    f"{os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']}% SMs, "
                    f"NCCL port: {port_args.p_nccl_port}, "
                    f"IPC queue: {queue_idx}"
                )

                p_reader, p_writer = mp.Pipe(duplex=False)
                isolate_child = os.environ.get("SGLANG_ISOLATE_CHILD_VISIBLE", "0").lower() in ("1", "true")
                if isolate_child:
                    # Per-process GPU visibility isolation for PREFILL
                    prev_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(phys_gpu_id)
                    prev_nccl_p2p = os.environ.get("NCCL_P2P_DISABLE")
                    if os.environ.get("SGLANG_DEBUG_NCCL_P2P_DISABLE") in ("1", "true", "True"):
                        os.environ["NCCL_P2P_DISABLE"] = "1"
                    logger.info(
                        f"[LAUNCH] P PP{pp_rank} TP{tp_rank} set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
                    )
                try:
                    p_proc = mp.Process(
                        target=run_scheduler_process,
                        args=(
                            server_args,
                            port_args,
                            0 if isolate_child else phys_gpu_id,
                            tp_rank,
                            None,  # dp_rank=None for PREFILL instances
                            p_writer,
                            p_ipc_info_queue,
                            True,  # bypass_load_weight=True - P instances share weights
                            InstanceRole.PREFILL,
                            pp_rank,  # 🔧 正确传递pp_rank参数
                        ),
                    )
                    with memory_saver_adapter.configure_subprocess():
                        p_proc.start()
                finally:
                    if isolate_child:
                        if prev_cuda_visible is None:
                            del os.environ["CUDA_VISIBLE_DEVICES"]
                        else:
                            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible
                        if prev_nccl_p2p is None and "NCCL_P2P_DISABLE" in os.environ:
                            del os.environ["NCCL_P2P_DISABLE"]
                        elif prev_nccl_p2p is not None:
                            os.environ["NCCL_P2P_DISABLE"] = prev_nccl_p2p
                scheduler_procs.append(p_proc)
                p_scheduler_pipe_readers.append(p_reader)
                logger.info(f"✅ [SEMI-PD] P instance PP{pp_rank} TP{tp_rank} started with PID: {p_proc.pid}")

        assert len(p_scheduler_pipe_readers) == len(d_scheduler_pipe_readers)

        # Wait for all Prefill instances to be ready
        logger.info("⏳ [SEMI-PD] Waiting for all Prefill instances to be ready...")
        for i, reader in enumerate(p_scheduler_pipe_readers):
            pp_rank = pp_rank_range[i // len(tp_rank_range)]
            tp_rank = tp_rank_range[i % len(tp_rank_range)]
            logger.info(f"⏳ [SEMI-PD] Waiting for P instance PP{pp_rank} TP{tp_rank} to be ready...")
            data = reader.recv()
            assert data["status"] == "ready"
            scheduler_infos.append(data)
            logger.info(f"✅ [SEMI-PD] P instance PP{pp_rank} TP{tp_rank} ready")

        logger.info("🎉 [SEMI-PD] All schedulers are ready! Semi-PD initialization completed successfully!")
        logger.info(f"📊 [SEMI-PD] Final stats: {len(scheduler_procs)} total processes, {len(scheduler_infos)} scheduler infos")

        # Log final process mapping
        for i, proc in enumerate(scheduler_procs):
            if i < len(d_scheduler_pipe_readers):
                # DECODE processes
                pp_rank = pp_rank_range[i // len(tp_rank_range)]
                tp_rank = tp_rank_range[i % len(tp_rank_range)]
                logger.info(f"📋 [SEMI-PD] Process {i}: D instance PP{pp_rank} TP{tp_rank}, PID: {proc.pid}")
            else:
                # PREFILL processes
                p_idx = i - len(d_scheduler_pipe_readers)
                pp_rank = pp_rank_range[p_idx // len(tp_rank_range)]
                tp_rank = tp_rank_range[p_idx % len(tp_rank_range)]
                logger.info(f"📋 [SEMI-PD] Process {i}: P instance PP{pp_rank} TP{tp_rank}, PID: {proc.pid}")
    else:
        # Allocate ports for inter-process communications
        port_args = PortArgs.init_new(server_args)

        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()
        scheduler_procs.append(proc)

        for i, reader in enumerate(scheduler_pipe_readers):
            data = reader.recv()
            assert data["status"] == "ready"
            scheduler_infos.append(data)

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None, None

        launch_dummy_health_check_server(server_args.host, server_args.port)

        for proc in scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
        return None, None, None

    # Launch detokenizer process with pipe for ready signal
    logger.info("🚀 [SEMI-PD] Launching Detokenizer process...")
    detoken_reader, detoken_writer = mp.Pipe(duplex=False)

    # 🔧 在PP模式下，使用第一个PP stage的端口配置（回退以匹配你的 Semi-PD+PP IPC 假设）
    if server_args.dp_size == 1 and server_args.pp_size > 1:
        port_args = port_args_per_pp[0]
        logger.info(f"🔧 [SEMI-PD] Using PP0 port configuration for detokenizer")

    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(
            server_args,
            port_args,
            detoken_writer,  # Pass pipe_writer for ready signal
        ),
    )
    detoken_proc.start()
    logger.info(f"✅ [SEMI-PD] Detokenizer process started with PID: {detoken_proc.pid}")

    # Wait for detokenizer to be ready
    logger.info("⏳ [SEMI-PD] Waiting for Detokenizer to be ready...")
    try:
        if detoken_reader.poll(60):  # 60 seconds timeout for L20
            detoken_data = detoken_reader.recv()
            logger.info(f"📨 [SEMI-PD] Received data from detokenizer: {detoken_data}")
            if detoken_data["status"] == "ready":
                logger.info("✅ [SEMI-PD] Detokenizer is ready")
            else:
                logger.error(f"❌ [SEMI-PD] Detokenizer failed to start: {detoken_data}")
                raise RuntimeError("Detokenizer initialization failed")
        else:
            logger.error("❌ [SEMI-PD] Timeout waiting for Detokenizer ready signal after 60 seconds")
            raise RuntimeError("Detokenizer ready timeout")
    except Exception as e:
        logger.error(f"❌ [SEMI-PD] Error waiting for Detokenizer: {e}")
        raise

    # Launch tokenizer process (if not created early)
    if not early_tokenizer_initialized:
        logger.info("🚀 [SEMI-PD] Launching Tokenizer process...")

        # 🔧 在PP模式下，使用第一个PP stage的端口配置（回退以匹配你的 Semi-PD+PP IPC 假设）
        if server_args.dp_size == 1 and server_args.pp_size > 1:
            port_args = port_args_per_pp[0]
            logger.info(f"🔧 [SEMI-PD] Using PP0 port configuration for tokenizer")

        tokenizer_manager = TokenizerManager(server_args, port_args)
        template_manager = TemplateManager()
        # Align Semi-PD path with standard init: load chat/completion templates
        try:
            template_manager.initialize_templates(
                tokenizer_manager=tokenizer_manager,
                model_path=server_args.model_path,
                chat_template=server_args.chat_template,
                completion_template=server_args.completion_template,
            )
            logger.info("✅ [SEMI-PD] Templates initialized (chat/completion)")
        except Exception:
            logger.exception("[SEMI-PD] Template initialization failed; proceeding with HF default")
        logger.info("✅ [SEMI-PD] Tokenizer and Template managers initialized")

    # Assume all schedulers have the same scheduler_info
    scheduler_info = scheduler_infos[0]
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    logger.info(f"✅ [SEMI-PD] Final configuration: max_req_input_len={scheduler_info['max_req_input_len']}")

    return tokenizer_manager, template_manager, scheduler_info
