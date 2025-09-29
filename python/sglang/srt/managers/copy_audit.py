# Lightweight to/copy audit for pinpointing small-copy storms in decode path
# Usage:
#   export SGLANG_COPY_AUDIT=1
#   export SGLANG_COPY_AUDIT_STEPS=2   # number of decode batches to audit on PP0
# Hooks monkey‑patch torch.Tensor.to and torch.Tensor.copy_ to count calls/bytes and
# attribute them to call sites in our code.

from __future__ import annotations

import inspect
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch


@dataclass
class Stat:
    calls: int = 0
    bytes: int = 0
    cpu_to_cuda: int = 0
    cuda_to_cpu: int = 0
    dtype_changes: int = 0
    h2d_bytes: int = 0
    d2h_bytes: int = 0


class CopyAudit:
    _lock = threading.Lock()
    _enabled_tls = threading.local()
    _orig_to = None
    _orig_copy_ = None
    _stats: Dict[str, Stat] = defaultdict(Stat)

    def __init__(self, scope: str = "", log_fn: Optional[Callable[[str], None]] = None):
        self.scope = scope
        self.log_fn = log_fn or (lambda s: None)

    def __enter__(self):
        with CopyAudit._lock:
            if CopyAudit._orig_to is None:
                CopyAudit._orig_to = torch.Tensor.to
            if CopyAudit._orig_copy_ is None:
                CopyAudit._orig_copy_ = torch.Tensor.copy_
            torch.Tensor.to = _make_to_patch(CopyAudit._orig_to)
            torch.Tensor.copy_ = _make_copy_patch(CopyAudit._orig_copy_)
        setattr(CopyAudit._enabled_tls, "enabled", True)
        self.log_fn(f"[CopyAudit] start scope={self.scope}")
        return self

    def __exit__(self, exc_type, exc, tb):
        with CopyAudit._lock:
            if CopyAudit._orig_to is not None:
                torch.Tensor.to = CopyAudit._orig_to
            if CopyAudit._orig_copy_ is not None:
                torch.Tensor.copy_ = CopyAudit._orig_copy_
        setattr(CopyAudit._enabled_tls, "enabled", False)
        self.log_fn(f"[CopyAudit] end scope={self.scope}")

    @staticmethod
    def _is_enabled() -> bool:
        return getattr(CopyAudit._enabled_tls, "enabled", False)

    @staticmethod
    def _record(kind: str, t: torch.Tensor, dst_device: Optional[torch.device], dst_dtype: Optional[torch.dtype]):
        try:
            if not CopyAudit._is_enabled():
                return
            # find first user frame in sglang or callsite otherwise
            site = _find_site()
            key = f"{kind}|{site}"
            b = int(t.numel() * t.element_size()) if hasattr(t, "numel") else 0
            stat = CopyAudit._stats[key]
            stat.calls += 1
            stat.bytes += b
            # device direction
            try:
                src_dev = t.device.type if hasattr(t, "device") else "cpu"
                dst_dev = (dst_device.type if isinstance(dst_device, torch.device) else (dst_device if isinstance(dst_device, str) else None))
                if dst_dev is None and hasattr(t, "device"):
                    dst_dev = t.device.type
                if src_dev == "cpu" and dst_dev == "cuda":
                    stat.cpu_to_cuda += 1
                    stat.h2d_bytes += b
                elif src_dev == "cuda" and dst_dev == "cpu":
                    stat.cuda_to_cpu += 1
                    stat.d2h_bytes += b
            except Exception:
                pass
            # dtype change
            try:
                if dst_dtype is not None and hasattr(t, "dtype") and dst_dtype != t.dtype:
                    stat.dtype_changes += 1
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def dump_summary(top_k: int = 10, log_fn: Optional[Callable[[str], None]] = None):
        log = log_fn or (lambda s: None)
        items = []
        for k, v in CopyAudit._stats.items():
            items.append((v.bytes, v.calls, v.cpu_to_cuda, v.cuda_to_cpu, v.dtype_changes, k))
        items.sort(reverse=True)
        log("[CopyAudit] Top sources by bytes (desc):")
        for i, (by, calls, h2d, d2h, dchg, k) in enumerate(items[:top_k]):
            log(f"  {i+1:02d}. bytes={by/1e6:.3f}MB calls={calls} h2d={h2d} d2h={d2h} dtype_changes={dchg} site={k}")
        if not items:
            log("  (no events captured)")

    @staticmethod
    def reset():
        CopyAudit._stats.clear()

    @staticmethod
    def totals():
        t_calls = t_bytes = h2d_calls = d2h_calls = h2d_bytes = d2h_bytes = 0
        for v in CopyAudit._stats.values():
            t_calls += v.calls
            t_bytes += v.bytes
            h2d_calls += v.cpu_to_cuda
            d2h_calls += v.cuda_to_cpu
            h2d_bytes += v.h2d_bytes
            d2h_bytes += v.d2h_bytes
        return {
            "calls": t_calls,
            "bytes": t_bytes,
            "h2d_calls": h2d_calls,
            "h2d_bytes": h2d_bytes,
            "d2h_calls": d2h_calls,
            "d2h_bytes": d2h_bytes,
        }


def _find_site() -> str:
    try:
        for frame in inspect.stack()[2:10]:  # skip our wrappers
            fn = frame.filename
            if "/sglang/" in fn and "copy_audit.py" not in fn:
                return f"{fn.split('/sglang/')[-1]}:{frame.lineno}"
        fr = inspect.stack()[2]
        return f"{fr.filename}:{fr.lineno}"
    except Exception:
        return "<unknown>"


def _parse_to_args(args, kwargs, self_tensor: torch.Tensor) -> Tuple[Optional[torch.device], Optional[torch.dtype]]:
    device = kwargs.get("device")
    dtype = kwargs.get("dtype")
    if device is None and len(args) >= 1:
        a0 = args[0]
        if isinstance(a0, torch.device) or isinstance(a0, str):
            device = a0
        elif isinstance(a0, torch.Tensor):
            device = a0.device
            if dtype is None:
                dtype = a0.dtype
    if dtype is None and len(args) >= 2:
        a1 = args[1]
        if isinstance(a1, torch.dtype):
            dtype = a1
    return device, dtype


def _make_to_patch(orig_to):
    def to_patch(self: torch.Tensor, *args, **kwargs):
        try:
            device, dtype = _parse_to_args(args, kwargs, self)
            CopyAudit._record("aten::to", self, device, dtype)
        except Exception:
            pass
        return orig_to(self, *args, **kwargs)
    return to_patch


def _make_copy_patch(orig_copy_):
    def copy_patch(self: torch.Tensor, other: torch.Tensor, *args, **kwargs):
        try:
            ddev = self.device if hasattr(self, "device") else None
            CopyAudit._record("aten::copy_", other, ddev, getattr(other, "dtype", None))
        except Exception:
            pass
        return orig_copy_(self, other, *args, **kwargs)
    return copy_patch

