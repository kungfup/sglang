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
"""DetokenizerManager is a process that detokenizes the token ids."""

import dataclasses
import logging
import os
import signal
from collections import OrderedDict
from typing import Dict, List, Union

import psutil
import setproctitle
import zmq

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOut,
    BatchMultimodalDecodeReq,
    BatchMultimodalOut,
    BatchStrOut,
    BatchTokenIDOut,
)
from sglang.srt.server_args import PortArgs, SemiPDPortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    get_zmq_socket,
    kill_itself_when_parent_died,
)
from sglang.utils import (
    TypeBasedDispatcher,
    find_printable_text,
    get_exception_traceback,
)

logger = logging.getLogger(__name__)

# Maximum number of request states that detokenizer can hold. When exceeded,
# oldest request states will be evicted. Default: 65536 (1<<16).
# For more details, see: https://github.com/sgl-project/sglang/issues/2812
# Use power of 2 values for better memory allocation.
DETOKENIZER_MAX_STATES = int(os.environ.get("SGLANG_DETOKENIZER_MAX_STATES", 1 << 16))


@dataclasses.dataclass
class DecodeStatus:
    """Store the status of incremental decoding."""

    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int
    # Offset that's sent to tokenizer for incremental update.
    sent_offset: int = 0


class DetokenizerManager:
    """DetokenizerManager is a process that detokenizes the token ids."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: Union[PortArgs, SemiPDPortArgs],
    ):
        logger.info("🔧 [DETOKENIZER] DetokenizerManager.__init__ started")

        # Init inter-process communication
        logger.info("🔧 [DETOKENIZER] Creating ZMQ context...")
        context = zmq.Context(2)
        logger.info("🔧 [DETOKENIZER] ZMQ context created successfully")

        logger.info(f"🔧 [DETOKENIZER] Setting up recv_from_scheduler socket: {port_args.detokenizer_ipc_name}")
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )
        logger.info("🔧 [DETOKENIZER] recv_from_scheduler socket created successfully")

        logger.info(f"🔧 [DETOKENIZER] Setting up send_to_tokenizer socket: {port_args.tokenizer_ipc_name}")
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_ipc_name, False
        )
        logger.info("🔧 [DETOKENIZER] send_to_tokenizer socket created successfully")

        logger.info(f"🔧 [DETOKENIZER] skip_tokenizer_init: {server_args.skip_tokenizer_init}")
        if server_args.skip_tokenizer_init:
            logger.info("🔧 [DETOKENIZER] Skipping tokenizer initialization")
            self.tokenizer = None
        else:
            logger.info(f"🔧 [DETOKENIZER] Loading tokenizer from: {server_args.tokenizer_path}")
            logger.info(f"🔧 [DETOKENIZER] Tokenizer mode: {server_args.tokenizer_mode}")
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
            logger.info("🔧 [DETOKENIZER] Tokenizer loaded successfully")

        logger.info("🔧 [DETOKENIZER] Creating decode_status dictionary...")
        self.decode_status = LimitedCapacityDict(capacity=DETOKENIZER_MAX_STATES)
        self.is_dummy = server_args.load_format == "dummy"
        logger.info(f"🔧 [DETOKENIZER] is_dummy: {self.is_dummy}")

        logger.info("🔧 [DETOKENIZER] Setting up request dispatcher...")
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchEmbeddingOut, self.handle_batch_embedding_out),
                (BatchTokenIDOut, self.handle_batch_token_id_out),
                (BatchMultimodalDecodeReq, self.handle_multimodal_decode_req),
            ]
        )
        logger.info("🔧 [DETOKENIZER] Request dispatcher created successfully")
        logger.info("🔧 [DETOKENIZER] DetokenizerManager.__init__ completed successfully")

    def event_loop(self):
        """The event loop that handles requests"""
        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()
            output = self._request_dispatcher(recv_obj)
            self.send_to_tokenizer.send_pyobj(output)

    def trim_matched_stop(
        self, output: Union[str, List[int]], finished_reason: Dict, no_stop_trim: bool
    ):
        if no_stop_trim or not finished_reason:
            return output

        matched = finished_reason.get("matched", None)
        if not matched:
            return output

        # TODO(lmzheng): handle the case where multiple stop strs are hit

        # Trim stop str.
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            return output[:pos] if pos != -1 else output

        # Trim stop token.
        if isinstance(matched, int) and isinstance(output, list):
            assert len(output) > 0
            return output[:-1]
        return output

    def handle_batch_embedding_out(self, recv_obj: BatchEmbeddingOut):
        # If it is embedding model, no detokenization is needed.
        return recv_obj

    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOut):
        bs = len(recv_obj.rids)

        # Initialize decode status
        read_ids, surr_ids = [], []
        for i in range(bs):
            rid = recv_obj.rids[i]
            if rid not in self.decode_status:
                s = DecodeStatus(
                    decoded_text=recv_obj.decoded_texts[i],
                    decode_ids=recv_obj.decode_ids[i],
                    surr_offset=0,
                    read_offset=recv_obj.read_offsets[i],
                )
                self.decode_status[rid] = s
            else:
                s = self.decode_status[rid]
                s.decode_ids.extend(recv_obj.decode_ids[i])

            read_ids.append(
                self.trim_matched_stop(
                    s.decode_ids[s.surr_offset :],
                    recv_obj.finished_reasons[i],
                    recv_obj.no_stop_trim[i],
                )
            )
            surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])

        # TODO(lmzheng): handle skip_special_tokens/spaces_between_special_tokens per request
        surr_texts = self.tokenizer.batch_decode(
            surr_ids,
            skip_special_tokens=recv_obj.skip_special_tokens[0],
            spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
        )
        read_texts = self.tokenizer.batch_decode(
            read_ids,
            skip_special_tokens=recv_obj.skip_special_tokens[0],
            spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
        )

        # Incremental decoding
        output_strs = []
        for i in range(bs):
            try:
                s = self.decode_status[recv_obj.rids[i]]
            except KeyError:
                raise RuntimeError(
                    f"Decode status not found for request {recv_obj.rids[i]}. "
                    "It may be due to the request being evicted from the decode status due to memory pressure. "
                    "Please increase the maximum number of requests by setting "
                    "the SGLANG_DETOKENIZER_MAX_STATES environment variable to a bigger value than the default value. "
                    f"The current value is {DETOKENIZER_MAX_STATES}. "
                    "For more details, see: https://github.com/sgl-project/sglang/issues/2812"
                )
            new_text = read_texts[i][len(surr_texts[i]) :]
            if recv_obj.finished_reasons[i] is None:
                # Streaming chunk: update the decode status
                if len(new_text) > 0 and not new_text.endswith("�"):
                    s.decoded_text = s.decoded_text + new_text
                    s.surr_offset = s.read_offset
                    s.read_offset = len(s.decode_ids)
                    new_text = ""
                else:
                    new_text = find_printable_text(new_text)

            output_str = self.trim_matched_stop(
                s.decoded_text + new_text,
                recv_obj.finished_reasons[i],
                recv_obj.no_stop_trim[i],
            )
            # Incrementally send text.
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            output_strs.append(incremental_output)

        return BatchStrOut(
            rids=recv_obj.rids,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
            output_ids=None,
            prompt_tokens=recv_obj.prompt_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
            spec_verify_ct=recv_obj.spec_verify_ct,
            input_token_logprobs_val=recv_obj.input_token_logprobs_val,
            input_token_logprobs_idx=recv_obj.input_token_logprobs_idx,
            output_token_logprobs_val=recv_obj.output_token_logprobs_val,
            output_token_logprobs_idx=recv_obj.output_token_logprobs_idx,
            input_top_logprobs_val=recv_obj.input_top_logprobs_val,
            input_top_logprobs_idx=recv_obj.input_top_logprobs_idx,
            output_top_logprobs_val=recv_obj.output_top_logprobs_val,
            output_top_logprobs_idx=recv_obj.output_top_logprobs_idx,
            input_token_ids_logprobs_val=recv_obj.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=recv_obj.input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=recv_obj.output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=recv_obj.output_token_ids_logprobs_idx,
            output_hidden_states=recv_obj.output_hidden_states,
        )

    def handle_multimodal_decode_req(self, recv_obj: BatchMultimodalDecodeReq):
        outputs = self.tokenizer.detokenize(recv_obj)
        return BatchMultimodalOut(
            rids=recv_obj.rids,
            finished_reasons=recv_obj.finished_reasons,
            outputs=outputs,
            prompt_tokens=recv_obj.prompt_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
        )


class LimitedCapacityDict(OrderedDict):
    def __init__(self, capacity: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            # Remove the oldest element (first item in the dict)
            self.popitem(last=False)
        # Set the new item
        super().__setitem__(key, value)


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: Union[PortArgs, SemiPDPortArgs],
    pipe_writer=None,
):
    # Write to a file immediately to confirm function is called
    try:
        with open("/tmp/detokenizer_debug.log", "w") as f:
            f.write("🔧 [DETOKENIZER] Function called - starting initialization\n")
            f.flush()
    except Exception as e:
        pass

    # Add immediate debug output to stdout
    print("🔧 [DETOKENIZER] Function called - starting initialization", flush=True)

    try:
        kill_itself_when_parent_died()
        print("🔧 [DETOKENIZER] kill_itself_when_parent_died completed", flush=True)

        setproctitle.setproctitle("sglang::detokenizer")
        print("🔧 [DETOKENIZER] setproctitle completed", flush=True)

        configure_logger(server_args)
        print("🔧 [DETOKENIZER] configure_logger completed", flush=True)

        parent_process = psutil.Process().parent()
        print("🔧 [DETOKENIZER] parent_process obtained", flush=True)
    except Exception as early_error:
        print(f"🔧 [DETOKENIZER] EARLY ERROR: {early_error}", flush=True)
        import traceback
        print(f"🔧 [DETOKENIZER] EARLY TRACEBACK: {traceback.format_exc()}", flush=True)
        return

    try:
        print("🔧 [DETOKENIZER] Starting DetokenizerManager initialization...", flush=True)
        logger.info("🔧 [DETOKENIZER] Starting DetokenizerManager initialization...")
        logger.info(f"🔧 [DETOKENIZER] server_args type: {type(server_args)}")
        logger.info(f"🔧 [DETOKENIZER] port_args type: {type(port_args)}")

        print("🔧 [DETOKENIZER] About to create DetokenizerManager instance", flush=True)
        logger.info("🔧 [DETOKENIZER] About to create DetokenizerManager instance")
        manager = DetokenizerManager(server_args, port_args)
        print("🔧 [DETOKENIZER] DetokenizerManager initialization completed successfully", flush=True)
        logger.info("🔧 [DETOKENIZER] DetokenizerManager initialization completed successfully")

        # Send ready signal to parent process
        if pipe_writer is not None:
            logger.info("🔧 [DETOKENIZER] Sending ready signal to parent process")
            pipe_writer.send({"status": "ready"})
            logger.info("🔧 [DETOKENIZER] Ready signal sent successfully")
        else:
            logger.warning("🔧 [DETOKENIZER] No pipe_writer provided, cannot send ready signal")

        logger.info("🔧 [DETOKENIZER] Starting Detokenizer event loop...")
        manager.event_loop()
    except Exception as e:
        import traceback as tb
        error_msg = tb.format_exc()
        logger.error(f"🔧 [DETOKENIZER] DetokenizerManager hit an exception: {error_msg}")
        print(f"🔧 [DETOKENIZER] CRITICAL ERROR: {error_msg}", flush=True)

        if pipe_writer is not None:
            try:
                logger.info("🔧 [DETOKENIZER] Sending error signal to parent process")
                pipe_writer.send({"status": "error", "error": error_msg})
                logger.info("🔧 [DETOKENIZER] Error signal sent successfully")
            except Exception as send_error:
                logger.error(f"🔧 [DETOKENIZER] Failed to send error signal: {send_error}")
                print(f"🔧 [DETOKENIZER] Failed to send error signal: {send_error}", flush=True)

        try:
            parent_process.send_signal(signal.SIGQUIT)
        except Exception as signal_error:
            logger.error(f"🔧 [DETOKENIZER] Failed to send SIGQUIT: {signal_error}")
            print(f"🔧 [DETOKENIZER] Failed to send SIGQUIT: {signal_error}", flush=True)
