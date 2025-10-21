"""
ViT Cache Server - 独立进程管理嵌入缓存

参考: LightLLM/lightllm/server/embed_cache/impl/naive_memory_cache.py

核心设计:
1. 独立进程运行，通过 RPyC 提供 RPC 服务
2. 双索引: cache_id → Record, content_hash → cache_id
3. 引用计数: alloc() 时 ref += 1, release() 时 ref -= 1
4. LRU 驱逐: 优先驱逐 ref_count == 0 且最久未访问的条目
5. 统一 SHM 清理: 驱逐时调用 cleanup_embedding_shm()
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import rpyc
from rpyc.utils.server import ThreadedServer

from sglang.srt.managers.vit_shm_utils import (
    cleanup_shared_memory,
    read_embedding_from_shm_raw,
    write_embedding_to_shm_raw,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CacheRecord:
    """缓存记录 (参考 LightLLM Record)"""
    cache_id: int           # UUID (唯一标识)
    content_hash: int       # 内容哈希 (用于去重)
    shm_key: str            # SHM 名称
    size_bytes: int         # 占用字节数
    ref_count: int          # 引用计数
    create_time: float      # 创建时间
    last_access: float      # 最后访问时间


class VITCacheManager:
    """ViT 缓存管理器 (进程内实现)"""
    
    def __init__(self, max_cache_bytes: int, expired_secs: int = 3600):
        self._max_cache_bytes = max_cache_bytes
        self._expired_secs = expired_secs
        self._total_bytes = 0

        # 双索引: cache_id → Record, content_hash → cache_id
        self._records: Dict[int, CacheRecord] = {}
        self._hash_to_id: Dict[int, int] = {}

        # LRU 顺序 (按访问时间排序)
        self._lru_order: OrderedDict[int, None] = OrderedDict()

        # 🔧 自增 ID 生成器 (64-bit)
        self._next_cache_id = 0

        self._lock = threading.Lock()

        # 统计信息
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
        }

    def _generate_cache_id(self) -> int:
        """生成 64-bit 缓存 ID (线程安全)"""
        cache_id = self._next_cache_id
        self._next_cache_id += 1
        return cache_id

    @staticmethod
    def _generate_shm_key(cache_id: int) -> str:
        """生成 SHM 名称 (确保 < 31 字符，兼容 macOS)

        格式: vit_cache_<16位十六进制> (最多 26 字符)
        """
        return f"vit_cache_{cache_id:016x}"

    def alloc(self, content_hash: int, size_bytes: int) -> Optional[Tuple[int, bool]]:
        """
        分配缓存槽位 (参考 LightLLM alloc)

        Returns:
            (cache_id, is_new): cache_id 为 None 表示失败，is_new 表示是否新分配
        """
        with self._lock:
            self._stats["total_requests"] += 1

            # 命中缓存
            if content_hash in self._hash_to_id:
                cache_id = self._hash_to_id[content_hash]
                record = self._records[cache_id]
                record.ref_count += 1
                record.last_access = time.time()

                # 更新 LRU
                self._lru_order.move_to_end(cache_id)

                self._stats["cache_hits"] += 1
                logger.debug(f"[Cache] Hit: hash={content_hash:x}, id={cache_id}, ref={record.ref_count}")
                return (cache_id, False)  # 🔧 返回 (cache_id, is_new=False)

            # 未命中，需要分配新槽位
            self._stats["cache_misses"] += 1

            # 驱逐旧条目
            if not self._evict_if_needed(size_bytes):
                logger.warning(f"[Cache] Failed to evict for {size_bytes} bytes")
                return None

            # 🔧 使用自增 ID 代替 UUID
            cache_id = self._generate_cache_id()
            shm_key = self._generate_shm_key(cache_id)

            record = CacheRecord(
                cache_id=cache_id,
                content_hash=content_hash,
                shm_key=shm_key,
                size_bytes=size_bytes,
                ref_count=1,
                create_time=time.time(),
                last_access=time.time(),
            )

            self._records[cache_id] = record
            self._hash_to_id[content_hash] = cache_id
            self._lru_order[cache_id] = None
            self._total_bytes += size_bytes

            logger.debug(f"[Cache] Alloc: hash={content_hash:x}, id={cache_id}, size={size_bytes}, shm_key={shm_key}")
            return (cache_id, True)  # 🔧 返回 (cache_id, is_new=True)
    
    def release(self, cache_id: int) -> None:
        """减少引用计数 (参考 LightLLM release)"""
        with self._lock:
            record = self._records.get(cache_id)
            if record is None:
                return
            
            if record.ref_count > 0:
                record.ref_count -= 1
            record.last_access = time.time()
            
            logger.debug(f"[Cache] Release: id={cache_id}, ref={record.ref_count}")
            
            # 如果引用计数为 0，尝试驱逐
            if record.ref_count == 0:
                self._evict_if_needed(0)
    
    def get_shm_key(self, cache_id: int) -> Optional[str]:
        """获取 SHM key"""
        with self._lock:
            record = self._records.get(cache_id)
            if record is None:
                return None
            record.last_access = time.time()
            self._lru_order.move_to_end(cache_id)
            return record.shm_key
    
    def get_cache_id(self, content_hash: int) -> Optional[int]:
        """根据 content_hash 获取 cache_id

        🔑 Phase 3.1: 新增接口，用于 Scheduler 端检查缓存

        Returns:
            cache_id: 如果命中返回 cache_id，否则返回 None
        """
        with self._lock:
            cache_id = self._hash_to_id.get(content_hash)
            if cache_id is not None:
                record = self._records.get(cache_id)
                if record is not None:
                    # 更新访问时间
                    record.last_access = time.time()
                    self._lru_order.move_to_end(cache_id)
                    # 增加引用计数 (立即发送时会使用)
                    record.ref_count += 1
                    logger.debug(f"[Cache] get_cache_id: hash={content_hash}, id={cache_id}, ref={record.ref_count}")
            return cache_id

    def contains(self, content_hash: int) -> bool:
        """检查缓存是否存在"""
        with self._lock:
            return content_hash in self._hash_to_id

    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            hit_rate = self._stats["cache_hits"] / max(1, self._stats["total_requests"])
            return {
                **self._stats,
                "hit_rate": hit_rate,
                "num_entries": len(self._records),
                "total_bytes": self._total_bytes,
                "max_bytes": self._max_cache_bytes,
            }
    
    def _evict_if_needed(self, required_bytes: int) -> bool:
        """
        驱逐策略: LRU + ref_count (参考 LightLLM _clear)

        🔧 Phase 3 补充: 严格保护 ref_count > 0 的条目

        优先驱逐:
        1. ref_count == 0 (必须条件)
        2. 过期 (last_access > expired_secs) (可选条件)
        3. 按 LRU 顺序
        """
        while self._total_bytes + required_bytes > self._max_cache_bytes:
            if not self._lru_order:
                return False

            # 找到最老的 ref_count=0 的条目
            evicted = False
            for cache_id in list(self._lru_order.keys()):
                record = self._records[cache_id]

                # 🔧 Phase 3 补充: 严格检查 ref_count
                # ref_count > 0 的条目不得驱逐（即使过期）
                if record.ref_count > 0:
                    continue

                # 检查是否可驱逐 (ref_count == 0)
                is_expired = time.time() - record.last_access > self._expired_secs

                # 🔧 清理 SHM (使用 raw cleanup，不添加前缀)
                cleanup_shared_memory(record.shm_key)

                # 删除记录
                del self._records[cache_id]
                del self._hash_to_id[record.content_hash]
                del self._lru_order[cache_id]
                self._total_bytes -= record.size_bytes
                self._stats["evictions"] += 1

                logger.debug(f"[Cache] Evict: id={cache_id}, size={record.size_bytes}, "
                            f"ref={record.ref_count}, expired={is_expired}")
                evicted = True
                break

            if not evicted:
                # 没有可驱逐的条目 (所有条目 ref_count > 0)
                logger.warning(f"[Cache] Cannot evict: all entries have ref_count > 0, "
                              f"total_bytes={self._total_bytes}, required={required_bytes}")
                return False

        return True


class VITCacheService(rpyc.Service):
    """RPyC 服务接口 (参考 LightLLM CacheServer)"""

    def __init__(self, max_cache_bytes: int):
        super().__init__()
        self._manager = VITCacheManager(max_cache_bytes)

    def exposed_alloc(self, content_hash: int, size_bytes: int) -> Optional[Tuple[int, bool]]:
        """分配缓存槽位

        Returns:
            (cache_id, is_new): cache_id 为 None 表示失败，is_new 表示是否新分配
        """
        return self._manager.alloc(content_hash, size_bytes)
    
    def exposed_release(self, cache_id: int) -> None:
        """释放缓存"""
        self._manager.release(cache_id)
    
    def exposed_get_shm_key(self, cache_id: int) -> Optional[str]:
        """获取 SHM key"""
        return self._manager.get_shm_key(cache_id)
    
    def exposed_get_cache_id(self, content_hash: int) -> Optional[int]:
        """根据 content_hash 获取 cache_id

        🔑 Phase 3.1: 新增 RPC 接口，用于 Scheduler 端检查缓存
        """
        return self._manager.get_cache_id(content_hash)

    def exposed_contains(self, content_hash: int) -> bool:
        """检查缓存是否存在"""
        return self._manager.contains(content_hash)

    def exposed_get_stats(self) -> dict:
        """获取统计信息"""
        return self._manager.get_stats()


def start_cache_server(port: int, max_cache_bytes: int, pipe_writer):
    """启动 CacheServer 进程 (参考 LightLLM start_cache_manager)"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [VIT Cache] %(message)s',
    )
    
    logger.info(f"Starting VIT CacheServer on port {port}, max_cache={max_cache_bytes / (1024**2):.1f}MB")
    
    service = VITCacheService(max_cache_bytes)
    server = ThreadedServer(service, port=port, protocol_config={"allow_pickle": True})
    
    # 通知父进程已就绪
    if pipe_writer is not None:
        pipe_writer.send("ready")
        logger.info("CacheServer sent ready signal to parent process")
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("CacheServer shutting down")
        server.close()

