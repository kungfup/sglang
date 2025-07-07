import os
from dataclasses import dataclass
from enum import Enum
from typing import List

import torch
import zmq

# Force fallback mode for now (can be changed later)
# To use IPC mode, set FORCE_FALLBACK = False after compiling semi_pd_ipc
FORCE_FALLBACK = True

if FORCE_FALLBACK:
    print("🔧 Forcing Semi-PD fallback mode (FORCE_FALLBACK=True)")
    HAS_IPC = False
    semi_pd_ipc = None
else:
    # Try to import semi_pd_ipc, but allow fallback if not available
    try:
        import semi_pd_ipc
        # Test if semi_pd_ipc actually works by creating a small test tensor
        import torch
        if torch.cuda.is_available():
            test_tensor = torch.randn(10, 10).cuda()
            try:
                # Test if IPC functions work
                test_handle = semi_pd_ipc.get_ipc_handle(test_tensor)
                sm_count = semi_pd_ipc.get_device_sm_count(0)
                HAS_IPC = True
                print("✅ Semi-PD IPC extension loaded and functional")
            except Exception as e:
                print(f"⚠️ Semi-PD IPC extension loaded but not functional: {e}")
                print("   Using fallback mode instead")
                HAS_IPC = False
                semi_pd_ipc = None
            finally:
                del test_tensor
        else:
            print("⚠️ CUDA not available, using Semi-PD fallback mode")
            HAS_IPC = False
            semi_pd_ipc = None
    except ImportError as e:
        print(f"ℹ️ Semi-PD IPC extension not available: {e}")
        print("   Using fallback mode (this is normal and expected)")
        HAS_IPC = False
        semi_pd_ipc = None

PREFILL_ENGINE_SM_PERCENTILE = int(os.getenv("SEMI_PD_PREFILL_SM_PERCENTILE", 80))
DECODE_ENGINE_SM_PERCENTILE = int(os.getenv("SEMI_PD_DECODE_SM_PERCENTILE", 100))


@dataclass
class IPCInfo:
    params_info: dict
    weight_handles: dict
    register_buffer_handles: dict
    kv_cache_handles: list[list]
    kvcache_info: dict
    req_to_token_handle: list
    req_to_token_info: dict


class InstanceRole(Enum):
    PREFILL = 0
    DECODE = 1
    OTHER = 2


class AggregatedSocket:
    def __init__(self, sockets: List[zmq.Socket]):
        self.sockets = sockets

    def send_pyobj(self, obj):
        for socket in self.sockets:
            socket.send_pyobj(obj)


DTYPE_TO_ATEN = {
    torch.float32: "at::kFloat",
    torch.float64: "at::kDouble",
    torch.float16: "at::kHalf",
    torch.int64: "at::kLong",
    torch.int32: "at::kInt",
    torch.int16: "at::kShort",
    torch.int8: "at::kChar",
    torch.uint64: "at::kUInt64",
    torch.uint32: "at::kUInt32",
    torch.uint16: "at::kUInt16",
    torch.uint8: "at::kByte",
    torch.uint32: "at::kUInt32",
    torch.uint64: "at::kUInt64",
    torch.bool: "at::kBool",
    torch.bfloat16: "at::kBFloat16",
    torch.complex32: "at::kComplexHalf",
    torch.complex64: "at::kComplexFloat",
    torch.complex128: "at::kComplexDouble",
    torch.float8_e4m3fn: "at::kFloat8_e4m3fn",
    torch.float8_e5m2: "at::kFloat8_e5m2",
    torch.float8_e4m3fnuz: "at::kFloat8_e4m3fnuz",
    torch.float8_e5m2fnuz: "at::kFloat8_e5m2fnuz",
}


def get_ipc_handle(tensor: torch.Tensor):
    if not HAS_IPC:
        raise RuntimeError("semi_pd_ipc not available, cannot get IPC handle")

    # https://github.com/pytorch/pytorch/blob/cbcc03c2ad11fbf1080f6a1025cc3f7aee0c858d/torch/multiprocessing/reductions.py#L371
    (
        device,
        handle,
        storage_size_bytes,  # size(in bytes) of the storage
        storage_offset_bytes,  # offset(in bytes) of the storage in the CUDA allocation
    ) = tensor.storage()._share_cuda_()[:4]
    assert storage_size_bytes == tensor.numel() * tensor.element_size()

    return semi_pd_ipc.get_ipc_handle(tensor), storage_offset_bytes


def convert_ipc_handle_to_tensor(ipc_handle, size, dtype, device):
    if not HAS_IPC:
        raise RuntimeError("semi_pd_ipc not available, cannot convert IPC handle")

    dtype_str = DTYPE_TO_ATEN[dtype]
    return semi_pd_ipc.convert_ipc_handle_to_tensor(ipc_handle, size, dtype_str, device)


def get_device_sm_count(rank: int = 0):
    if not HAS_IPC:
        # Fallback: return a reasonable default SM count
        return 108  # Common for many GPUs, can be overridden

    return semi_pd_ipc.get_device_sm_count(rank)
