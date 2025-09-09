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
"""Fused operators for normalization layers."""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from sglang.srt.custom_op import CustomOp
from sglang.srt.two_batch_overlap import global_server_args_dict
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
)

try:
    from sgl_kernel.utils import is_hopper_arch
except ImportError:
    is_hopper_arch = lambda: False

import os

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()

if _is_cuda:
    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )

if _use_aiter:
    from aiter import rmsnorm2d_fwd as rms_norm
    from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm
elif _is_hip:
    from vllm._custom_ops import fused_add_rms_norm, rms_norm

logger = logging.getLogger(__name__)

if is_npu():
    import torch_npu


class RMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        if _use_aiter:
            self._forward_method = self.forward_aiter

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        _dbg = bool(int(os.environ.get("SGLANG_TBO_DEBUG", "0")))
        if _dbg:
            print(f"[TBO][rmsnorm] forward_cuda enter: x.shape={tuple(x.shape)}, x.dtype={x.dtype}, device={x.device}, residual={'Y' if residual is not None else 'N'}", flush=True)
        
        # 移除过于宽泛的 TBO context 检测，只在确实需要时才使用 forward_native
        if self.variance_size_override is not None:
            if _dbg:
                print("[TBO][rmsnorm] using forward_native due to variance_size_override", flush=True)
            return self.forward_native(x, residual)
        
        # 对于 Hopper 架构或常规路径，尝试使用融合核；若失败则回退到 native
        try:
            if residual is not None:
                if _dbg:
                    print("[TBO][rmsnorm] calling fused_add_rmsnorm", flush=True)
                fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
                if _dbg:
                    print("[TBO][rmsnorm] fused_add_rmsnorm done", flush=True)
                return x, residual
            if _dbg:
                print("[TBO][rmsnorm] calling rmsnorm", flush=True)
            out = rmsnorm(x, self.weight.data, self.variance_epsilon)
            if _dbg:
                print("[TBO][rmsnorm] rmsnorm done", flush=True)
            return out
        except RuntimeError as e:
            # 某些形状（例如很小的 batch 维度）可能导致 CUDA kernel 配置非法，回退到 native
            if _is_cuda:
                logger.warning(
                    f"RMSNorm fused kernel failed: {e}. Falling back to native implementation."
                )
                if _dbg:
                    print(f"[TBO][rmsnorm] fused kernel failed: {e}, using forward_native", flush=True)
            return self.forward_native(x, residual)

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            out, _, residual_out = torch_npu.npu_add_rms_norm(
                residual, x, self.weight.data, self.variance_epsilon
            )
            return out, residual_out
        return torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]

    def forward_aiter(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            residual_out = torch.empty_like(x)
            output = torch.empty_like(x)
            fused_add_rms_norm(
                output,
                x,
                residual,
                residual_out,
                self.weight.data,
                self.variance_epsilon,
            )
            return output, residual_out
        return rms_norm(x, self.weight.data, self.variance_epsilon)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            # NOTE: Remove this if aiter kernel supports discontinuous input
            x = x.contiguous()
        if residual is not None:
            fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = torch.empty_like(x)
        rms_norm(out, x, self.weight.data, self.variance_epsilon)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        _dbg = bool(int(os.environ.get("SGLANG_TBO_DEBUG", "0")))
        if _dbg:
            print(f"[TBO][rmsnorm] forward_native enter: x.shape={tuple(x.shape)}, x.dtype={x.dtype}, device={x.device}, residual={'Y' if residual is not None else 'N'}", flush=True)
        
        if not x.is_contiguous():
            x = x.contiguous()
        
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Expected hidden_size to be "
                f"{self.hidden_size}, but found: {hidden_size}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[..., : self.variance_size_override]

        if _dbg:
            print(f"[TBO][rmsnorm] computing variance, x_var.shape={x_var.shape}", flush=True)
        
        # 简化 variance 计算，使用更直接的方式
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = (x * self.weight).to(orig_dtype)
        
        if _dbg:
            print("[TBO][rmsnorm] forward_native exit", flush=True)
        
        if residual is None:
            return x
        else:
            return x, residual

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if _is_cpu_amx_available:
            if residual is not None:
                torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return x, residual
            return torch.ops.sgl_kernel.rmsnorm_cpu(
                x, self.weight.data, self.variance_epsilon
            )
        else:
            return self.forward_native(x, residual)

    def forward_with_allreduce_fusion(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward method with allreduce fusion, prioritizing flashinfer fused operations
        """
        if residual is not None:
            from sglang.srt.distributed import get_tensor_model_parallel_world_size
            from sglang.srt.layers.flashinfer_comm_fusion import (
                flashinfer_allreduce_residual_rmsnorm,
            )

            if get_tensor_model_parallel_world_size() > 1:
                fused_result = flashinfer_allreduce_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=self.weight,
                    eps=self.variance_epsilon,
                )
                if fused_result[0] is not None:
                    return fused_result

        return self.forward(x, residual)


class GemmaRMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        # Re-dispatch
        if _is_hip:
            self._forward_method = self.forward_native

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            gemma_fused_add_rmsnorm(
                x, residual, self.weight.data, self.variance_epsilon
            )
            return x, residual
        out = gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


if not (_is_cuda or _is_hip or _is_npu or (_is_cpu and _is_cpu_amx_available)):
    logger.info(
        "sgl-kernel layernorm implementation is not available on current platform. Fallback to other kernel libraries."
    )
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
