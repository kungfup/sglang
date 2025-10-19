"""Simplified shared-memory helpers for VIT Scheduler.

设计目标:
1. **简单**: 只有几个纯函数,无后台线程,无引用计数。
2. **统一**: 完整使用 `multiprocessing.shared_memory`, 不混用 `posix_ipc`。
3. **鲁棒**: 写入后立即 flush, 读取支持重试, 清理函数安全幂等。

该模块提供两类能力:
- 嵌入共享内存: `write_embedding_to_shm` / `read_embedding_from_shm` / `cleanup_embedding_shm`
- 通用 Tensor 共享内存: `read_tensor_from_shared_memory` / `cleanup_shared_memory`

参考实现: LightLLM `server/embed_cache/utils.py`
"""

from __future__ import annotations

import glob
import os
import time
import logging
from io import BytesIO
from typing import Optional, Sequence, Tuple

import torch
import multiprocessing.shared_memory as shm

logger = logging.getLogger(__name__)

# 统一前缀,便于遍历 /dev/shm
EMBED_PREFIX = "vit_embed"


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """序列化 tensor (自动搬到 CPU, 保留 dtype/shape)."""
    tensor = tensor.detach().cpu()
    dest = torch.empty_like(tensor)
    dest.copy_(tensor)
    buffer = BytesIO()
    torch.save(dest, buffer, _use_new_zipfile_serialization=False, pickle_protocol=4)
    buffer.seek(0)
    return buffer.read()


def _bytes_to_tensor(data: bytes) -> torch.Tensor:
    """反序列化 tensor (返回 CPU Tensor)."""
    return torch.load(BytesIO(data), weights_only=False)


def _create_shm(name: str, data: bytes) -> bool:
    """创建名为 name 的共享内存并写入 data."""
    try:
        shared_memory = shm.SharedMemory(name=name, create=True, size=len(data))
        shared_memory.buf[: len(data)] = data
        shared_memory.close()
        logger.debug(
            "[VIT SHM] created shm=%s size=%.2fMB", name, len(data) / (1024 ** 2)
        )
        return True
    except FileExistsError:
        logger.warning("[VIT SHM] shm already exists: %s", name)
        return False
    except Exception as exc:
        logger.error("[VIT SHM] create shm failed: %s (%s)", name, exc)
        return False


def _read_shm(name: str) -> Optional[bytes]:
    """读取名为 name 的共享内存块.

    ✅ 修复: 读取后 close() handle,避免泄漏
    注意: 虽然 LightLLM 也不 close(),但我们没有 CacheServer 统一管理
    """
    try:
        shared_memory = shm.SharedMemory(name=name)
        data = shared_memory.buf.tobytes()
        shared_memory.close()  # ✅ 读取后 close() handle
        return data
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.error("[VIT SHM] read shm failed: %s (%s)", name, exc)
        return None


def cleanup_shared_memory(name: str) -> None:
    """关闭并 unlink 指定共享内存 (若不存在则忽略)."""
    try:
        shared_memory = shm.SharedMemory(name=name)
        shared_memory.close()
        shared_memory.unlink()
        logger.debug("[VIT SHM] cleanup shm=%s", name)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("[VIT SHM] cleanup shm failed: %s (%s)", name, exc)


# ---------------------------------------------------------------------------
# 嵌入共享内存 (统一使用 vit_embed_{request_id})
# ---------------------------------------------------------------------------

def _embed_name(request_id: str) -> str:
    return f"{EMBED_PREFIX}_{request_id}"


def write_embedding_to_shm(request_id: str, embedding: torch.Tensor) -> bool:
    """将 embedding 写入共享内存."""
    try:
        payload = _tensor_to_bytes(embedding)
        return _create_shm(_embed_name(request_id), payload)
    except Exception as exc:
        logger.error(
            "[VIT SHM] write embedding failed: request=%s error=%s", request_id, exc
        )
        return False


def read_embedding_from_shm(
    request_id: str,
    max_retries: int = 5,
    retry_delay_ms: float = 5.0,
) -> Optional[torch.Tensor]:
    """读取 embedding (带重试)."""
    name = _embed_name(request_id)
    for attempt in range(max_retries):
        data = _read_shm(name)
        if data is not None:
            tensor = _bytes_to_tensor(data)
            logger.debug(
                "[VIT SHM] read embedding: request=%s shape=%s dtype=%s",
                request_id,
                tuple(tensor.shape),
                tensor.dtype,
            )
            return tensor
        if attempt < max_retries - 1:
            time.sleep(retry_delay_ms / 1000.0)
    logger.error("[VIT SHM] embedding not ready: request=%s", request_id)
    return None


def cleanup_embedding_shm(request_id: str) -> None:
    """清理 embedding 共享内存."""
    cleanup_shared_memory(_embed_name(request_id))


# ---------------------------------------------------------------------------
# Raw SHM 辅助函数 (用于 CacheServer，不添加前缀)
# ---------------------------------------------------------------------------

def write_embedding_to_shm_raw(shm_name: str, embedding: torch.Tensor) -> bool:
    """将 embedding 写入共享内存 (使用原始名称，不添加前缀)."""
    try:
        payload = _tensor_to_bytes(embedding)
        return _create_shm(shm_name, payload)
    except Exception as exc:
        logger.error(
            "[VIT SHM] write embedding (raw) failed: name=%s error=%s", shm_name, exc
        )
        return False


def read_embedding_from_shm_raw(
    shm_name: str,
    max_retries: int = 5,
    retry_delay_ms: float = 5.0,
) -> Optional[torch.Tensor]:
    """读取 embedding (使用原始名称，不添加前缀，带重试)."""
    for attempt in range(max_retries):
        data = _read_shm(shm_name)
        if data is not None:
            tensor = _bytes_to_tensor(data)
            logger.debug(
                "[VIT SHM] read embedding (raw): name=%s shape=%s dtype=%s",
                shm_name,
                tuple(tensor.shape),
                tensor.dtype,
            )
            return tensor
        if attempt < max_retries - 1:
            time.sleep(retry_delay_ms / 1000.0)
    logger.warning("[VIT SHM] read embedding (raw) failed after retries: name=%s", shm_name)
    return None


def cleanup_all_vit_shm(prefix: str = EMBED_PREFIX) -> int:
    """遍历 /dev/shm 下所有 `prefix_*` 并清理."""
    shm_dir = "/dev/shm"
    pattern = os.path.join(shm_dir, f"{prefix}_*")
    count = 0
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        cleanup_shared_memory(name)
        count += 1
    if count:
        logger.info("[VIT SHM] cleaned %d lingering shm objects", count)
    return count


# ---------------------------------------------------------------------------
# 通用 tensor 共享内存 (用于 pixel/grid)
# ---------------------------------------------------------------------------

def read_tensor_from_shared_memory(
    shm_name: str,
    shape: Sequence[int],
    dtype_str: str,
) -> torch.Tensor:
    """从共享内存读取 tensor 并克隆到新的 CPU Tensor."""
    dtype = getattr(torch, dtype_str)
    shared_memory = shm.SharedMemory(name=shm_name)
    try:
        tensor = torch.frombuffer(shared_memory.buf, dtype=dtype).reshape(shape).clone()
    finally:
        shared_memory.close()
    return tensor


def cleanup_tensor_shared_memory(shm_name: str) -> None:
    """清理客户端创建的 pixel/grid 共享内存."""
    cleanup_shared_memory(shm_name)
