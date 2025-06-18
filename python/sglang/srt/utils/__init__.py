# flake8: noqa
"""兼容层：保持旧路径 `sglang.srt.utils` 的完整功能。

此目录仅用于重定向到真正的实现文件
`sglang/srt/utils.py`（位于父目录）。我们在此动态加载该文件
并将其所有公开符号透传。这样即可在保留 `profiler` 子模块的同
时，继续支持诸如 `from sglang.srt.utils import get_ip` 的写法。
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from types import ModuleType

# ---------------------------------------------------------------------------
# 动态加载真正的 utils.py 模块
# ---------------------------------------------------------------------------

_parent_dir = pathlib.Path(__file__).resolve().parent.parent  # .../srt
_legacy_path = _parent_dir / "utils.py"
_module_name = "sglang.srt._legacy_utils"

if _module_name in sys.modules:
    _legacy_mod: ModuleType = sys.modules[_module_name]
else:
    _spec = importlib.util.spec_from_file_location(_module_name, _legacy_path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Cannot find legacy utils module at {_legacy_path}")
    _legacy_mod = importlib.util.module_from_spec(_spec)
    sys.modules[_module_name] = _legacy_mod  # register before exec to handle recursive imports
    _spec.loader.exec_module(_legacy_mod)

# ---------------------------------------------------------------------------
# 将符号暴露到当前命名空间
# ---------------------------------------------------------------------------
for _name in dir(_legacy_mod):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_legacy_mod, _name)

__all__ = [n for n in globals().keys() if not n.startswith("__")]

# ---------------------------------------------------------------------------
# 再暴露 profiler 相关符号并注册子模块 `sglang.srt.utils.profiler`
# ---------------------------------------------------------------------------
from sglang.srt.utils_profiler import LayerProfiler, layer_profiler  # type: ignore

_profiler_mod = types.ModuleType(__name__ + ".profiler")
_profiler_mod.LayerProfiler = LayerProfiler  # type: ignore
_profiler_mod.layer_profiler = layer_profiler  # type: ignore
sys.modules[_profiler_mod.__name__] = _profiler_mod 