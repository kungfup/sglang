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

from sglang.srt.code_completion_parser import load_completion_template_for_openai_api
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
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api.adapter import (
    guess_chat_template_name_from_model_path,
    load_chat_template_for_openai_api,
)
from sglang.srt.server_args import PortArgs, ServerArgs
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

_is_cuda = is_cuda()


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

        # Shutdown the subprocesses automatically when the program exits
        atexit.register(self.shutdown)

        # Allocate ports for inter-process communications
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

        # Launch subprocesses
        tokenizer_manager, scheduler_info = _launch_subprocesses(
            server_args=server_args,
            port_args=port_args,
        )
        self.server_args = server_args
        self.tokenizer_manager = tokenizer_manager
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
                logger.info("data_parallel_rank not provided, using default dispatch")
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
                logger.info("data_parallel_rank not provided, using default dispatch")
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

    def release_memory_occupation(self):
        """Release GPU occupation temporarily."""
        obj = ReleaseMemoryOccupationReqInput()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.release_memory_occupation(obj, None)
        )

    def resume_memory_occupation(self):
        """Resume GPU occupation."""
        obj = ResumeMemoryOccupationReqInput()
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
    if _is_cuda:
        assert_pkg_version(
            "sgl-kernel",
            "0.1.7",
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
) -> Tuple[TokenizerManager, Dict]:
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

    # Launch VIT Scheduler if enabled
    vit_scheduler_procs = []
    vit_cache_server_proc = None
    if server_args.enable_vit_scheduler and server_args.node_rank == 0:
        from sglang.srt.configs.model_config import ModelConfig

        # Check if model is multimodal
        model_config = ModelConfig(
            model_path=server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            model_override_args="{}",
        )

        if model_config.is_multimodal:
            # 🔧 动态计算 VIT Scheduler 端口（基于服务器端口）
            if server_args.vit_scheduler_port is None:
                # 默认: server_port + 1000
                # 例如: --port 30019 => vit_scheduler_port = 31019
                server_args.vit_scheduler_port = server_args.port + 1000
                logger.info(f"VIT Scheduler port auto-generated: {server_args.vit_scheduler_port} (server_port + 1000)")

            # 🔧 设置 VIT 缓存禁用环境变量
            if server_args.disable_vit_cache:
                os.environ["SGLANG_VIT_DISABLE_CACHE"] = "1"
                logger.info("[VIT CACHE] ViT embedding cache is DISABLED (--disable-vit-cache)")
            else:
                os.environ["SGLANG_VIT_DISABLE_CACHE"] = "0"

            # 🔧 新架构: 启动 CacheServer 进程 (仅当缓存启用时)
            use_new_arch = os.environ.get("SGLANG_VIT_NEW_ARCH", "1") == "1"
            if use_new_arch and not server_args.disable_vit_cache:
                from sglang.srt.managers.vit_cache_server import start_cache_server

                cache_rpc_port = int(os.environ.get("SGLANG_VIT_CACHE_RPC_PORT", "18888"))
                cache_size_mb = server_args.vit_scheduler_cache_size_mb

                logger.info(f"[NEW ARCH] Launching VIT CacheServer on port {cache_rpc_port}, cache_size={cache_size_mb}MB...")

                cache_reader, cache_writer = mp.Pipe(duplex=False)
                vit_cache_server_proc = mp.Process(
                    target=start_cache_server,
                    args=(cache_rpc_port, cache_size_mb * 1024 * 1024, cache_writer),
                    daemon=True,
                )
                vit_cache_server_proc.start()

                # 等待 CacheServer 就绪
                if cache_reader.poll(timeout=10):
                    data = cache_reader.recv()
                    if data != "ready":
                        raise RuntimeError(f"VIT CacheServer failed to start: {data}")
                    logger.info(f"[NEW ARCH] VIT CacheServer is ready on port {cache_rpc_port}, PID={vit_cache_server_proc.pid}")
                else:
                    raise RuntimeError("VIT CacheServer startup timeout")

                # 设置环境变量供 VITScheduler 使用
                os.environ["SGLANG_VIT_CACHE_RPC_PORT"] = str(cache_rpc_port)
            elif use_new_arch and server_args.disable_vit_cache:
                logger.info("[NEW ARCH] VIT CacheServer NOT started (cache disabled)")
            else:
                logger.info("[LEGACY ARCH] Using in-process CacheServer")

            # 默认开启 Worker Pool（除非用户显式关闭）
            if (
                use_new_arch
                and "SGLANG_VIT_USE_WORKER_POOL" not in os.environ
                and os.environ.get("SGLANG_VIT_NEW_ARCH", "1") == "1"
            ):
                os.environ["SGLANG_VIT_USE_WORKER_POOL"] = "1"
                logger.info("[WORKER POOL] Default enable SGLANG_VIT_USE_WORKER_POOL=1 for batched ViT processing")

            # 🔧 Worker Pool: 启动 Worker Pool（如果启用）
            vit_worker_procs = []
            use_worker_pool = os.environ.get("SGLANG_VIT_USE_WORKER_POOL", "0") == "1"
            if use_worker_pool:
                vit_dp = int(os.environ.get("SGLANG_VIT_DP", "1"))
                vit_tp_size = server_args.vit_tp_size
                worker_rpc_port_start = int(os.environ.get("SGLANG_VIT_WORKER_RPC_PORT_START", "19000"))

                logger.info(f"[WORKER POOL] Launching {vit_dp} Worker(s) with TP={vit_tp_size}, port_start={worker_rpc_port_start}...")

                for worker_id in range(vit_dp):
                    # 每个 Worker 启动 vit_tp_size 个进程（TP ranks）
                    worker_procs_for_this_worker = []

                    for tp_rank in range(vit_tp_size):
                        reader, writer = mp.Pipe(duplex=False)

                        # Worker RPC 端口（每个 Worker 一个端口，所有 TP ranks 共享）
                        worker_rpc_port = worker_rpc_port_start + worker_id

                        proc = mp.Process(
                            target=run_vit_worker_process,
                            args=(server_args, worker_id, tp_rank, vit_tp_size, worker_rpc_port, writer),
                            daemon=True,
                        )
                        proc.start()
                        worker_procs_for_this_worker.append((proc, reader, tp_rank))

                        logger.info(
                            f"[WORKER POOL] Started Worker {worker_id} TP rank {tp_rank}/{vit_tp_size}, "
                            f"PID={proc.pid}, port={worker_rpc_port}"
                        )

                    # 等待该 Worker 的所有 TP ranks 就绪
                    for proc, reader, tp_rank in worker_procs_for_this_worker:
                        if reader.poll(timeout=30):
                            data = reader.recv()
                            if data != "ready":
                                raise RuntimeError(f"Worker {worker_id} TP rank {tp_rank} failed to start: {data}")
                            logger.info(f"[WORKER POOL] Worker {worker_id} TP rank {tp_rank} is ready")
                        else:
                            raise RuntimeError(f"Worker {worker_id} TP rank {tp_rank} startup timeout")

                    vit_worker_procs.extend(worker_procs_for_this_worker)

                logger.info(f"[WORKER POOL] All {vit_dp} Worker(s) are ready")

            # 🔧 VIT TP: 启动多个 VIT Scheduler 进程（如果 vit_tp_size > 1）
            vit_tp_size = server_args.vit_tp_size
            if vit_tp_size > 1:
                logger.info(f"Launching {vit_tp_size} VIT Scheduler processes (TP={vit_tp_size})...")

                # 启动多个 VIT Scheduler 进程
                for tp_rank in range(vit_tp_size):
                    # 每个 TP rank 使用相同的 ZMQ 端口（但只有 rank 0 会监听）
                    # 其他 ranks 只参与计算
                    reader, writer = mp.Pipe(duplex=False)

                    # 设置环境变量
                    env_vars = {
                        "SGLANG_VIT_TP_RANK": str(tp_rank),
                        "SGLANG_VIT_TP_SIZE": str(vit_tp_size),
                        "SGLANG_VIT_TP_PORT": str(server_args.vit_tp_port),
                    }

                    proc = mp.Process(
                        target=run_vit_scheduler_process,
                        args=(server_args, writer, env_vars),
                    )
                    proc.start()
                    vit_scheduler_procs.append((proc, reader, tp_rank))

                    logger.info(
                        f"Started VIT Scheduler process: TP rank {tp_rank}/{vit_tp_size}, "
                        f"PID={proc.pid}"
                    )

                # 等待所有 VIT Scheduler 进程准备好
                for proc, reader, tp_rank in vit_scheduler_procs:
                    data = reader.recv()
                    if data != "ready":
                        raise RuntimeError(f"VIT Scheduler TP rank {tp_rank} failed to start: {data}")
                    logger.info(f"VIT Scheduler TP rank {tp_rank} is ready")

                logger.info(f"All {vit_tp_size} VIT Scheduler processes are ready")
            else:
                # 单个 VIT Scheduler 进程（原有逻辑）
                logger.info(f"Launching VIT Scheduler process on port {server_args.vit_scheduler_port}...")

                reader, writer = mp.Pipe(duplex=False)
                vit_scheduler_proc = mp.Process(
                    target=run_vit_scheduler_process,
                    args=(server_args, writer, {}),
                )
                vit_scheduler_proc.start()
                vit_scheduler_procs.append((vit_scheduler_proc, reader, 0))

                # Wait for VIT Scheduler to be ready
                data = reader.recv()
                if data != "ready":
                    raise RuntimeError(f"VIT Scheduler failed to start: {data}")

                logger.info(f"VIT Scheduler is ready on port {server_args.vit_scheduler_port}")

            # Set environment variables for main Scheduler
            os.environ["SGLANG_VIT_SCHEDULER_ENABLED"] = "1"
            os.environ["SGLANG_VIT_SCHEDULER_HOST"] = "localhost"
            os.environ["SGLANG_VIT_SCHEDULER_PORT"] = str(server_args.vit_scheduler_port)
            os.environ["SGLANG_VIT_SCHEDULER_TIMEOUT_MS"] = "15000"  # 增加到 15 秒，避免首次计算超时
        else:
            logger.info("Model is not multimodal, VIT Scheduler disabled")

    scheduler_procs = []
    if server_args.dp_size == 1:
        # Launch tensor parallel scheduler processes
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []

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
                    target=run_scheduler_process,
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
            return None, None

        launch_dummy_health_check_server(server_args.host, server_args.port)

        for proc in scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
        return None, None

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
    if server_args.chat_template:
        load_chat_template_for_openai_api(
            tokenizer_manager, server_args.chat_template, server_args.model_path
        )
    else:
        guess_chat_template_name_from_model_path(server_args.model_path)

    if server_args.completion_template:
        load_completion_template_for_openai_api(server_args.completion_template)

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
    return tokenizer_manager, scheduler_info


def run_vit_worker_process(
    server_args: ServerArgs,
    worker_id: int,
    tp_rank: int,
    tp_size: int,
    rpc_port: int,
    pipe_writer,
):
    """Run VIT Worker process

    Args:
        server_args: Server arguments
        worker_id: Worker ID (for DP)
        tp_rank: TP rank (0 to tp_size-1)
        tp_size: TP size
        rpc_port: Worker RPC port
        pipe_writer: Pipe writer for sending ready signal
    """
    try:
        # 设置环境变量
        os.environ["SGLANG_VIT_TP_RANK"] = str(tp_rank)
        os.environ["SGLANG_VIT_TP_SIZE"] = str(tp_size)
        os.environ["SGLANG_VIT_TP_PORT"] = str(server_args.vit_tp_port + worker_id)  # 每个 Worker 独立的 NCCL 端口

        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.managers.vit_worker_rpc import start_vit_worker_process

        # Load model config
        model_config = ModelConfig(
            model_path=server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            model_override_args="{}",
        )

        # 获取 CacheServer 配置
        cache_rpc_host = "localhost"
        cache_rpc_port = int(os.environ.get("SGLANG_VIT_CACHE_RPC_PORT", "18888"))

        # 通知父进程当前 rank 就绪，必须所有 rank 都发送
        pipe_writer.send("ready")
        logger.info(
            f"[Worker {worker_id}] TP rank {tp_rank} sent ready signal to parent process"
        )

        # 启动 Worker（这会阻塞）
        start_vit_worker_process(
            worker_id=worker_id,
            model_config=model_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            rpc_port=rpc_port,
            cache_rpc_host=cache_rpc_host,
            cache_rpc_port=cache_rpc_port,
        )
    except Exception as e:
        logger.error(f"[Worker {worker_id}] Worker process failed: {e}")
        import traceback
        traceback.print_exc()
        pipe_writer.send(f"error: {e}")
        raise


def run_vit_scheduler_process(server_args: ServerArgs, pipe_writer, env_vars: dict = None):
    """Run VIT Scheduler process

    Args:
        server_args: Server arguments
        pipe_writer: Pipe writer for sending ready signal
        env_vars: Environment variables to set (for TP support)
    """
    try:
        # 🔧 VIT TP: 设置环境变量
        if env_vars:
            for key, value in env_vars.items():
                os.environ[key] = value

        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.managers.vit_scheduler import start_vit_scheduler

        # Load model config
        model_config = ModelConfig(
            model_path=server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            model_override_args="{}",
        )

        # 🔧 新架构: 获取 cache_rpc_port
        cache_rpc_port = None
        use_new_arch = os.environ.get("SGLANG_VIT_NEW_ARCH", "1") == "1"
        if use_new_arch:
            cache_rpc_port = int(os.environ.get("SGLANG_VIT_CACHE_RPC_PORT", "18888"))

        # Start VIT Scheduler (this will block)
        # NOTE: start_vit_scheduler will send "ready" signal via pipe_writer when ZMQ is listening
        start_vit_scheduler(
            model_config=model_config,
            device=server_args.vit_scheduler_device,
            zmq_port=server_args.vit_scheduler_port,
            batch_size=server_args.vit_scheduler_batch_size,
            batch_timeout_ms=server_args.vit_scheduler_batch_timeout_ms,
            cache_size_mb=server_args.vit_scheduler_cache_size_mb,
            cache_rpc_port=cache_rpc_port,
            pipe_writer=pipe_writer,
        )
    except Exception as e:
        logger.error(f"VIT Scheduler process failed: {e}")
        import traceback
        traceback.print_exc()
        pipe_writer.send(f"error: {e}")
        raise
