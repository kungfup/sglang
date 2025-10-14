#!/usr/bin/env python3
"""
VIT Shared Memory Cleanup Script

清理 VIT Scheduler 使用的 POSIX 共享内存

Usage:
    python scripts/cleanup_vit_shm.py [--prefix vit_embed] [--dry-run]
"""

import argparse
import logging
import sys

try:
    import posix_ipc
    POSIX_IPC_AVAILABLE = True
except ImportError:
    POSIX_IPC_AVAILABLE = False
    print("ERROR: posix_ipc not available. Install with: pip install posix-ipc")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_shm_objects(prefix: str = "vit_embed") -> list:
    """列出所有匹配前缀的共享内存对象
    
    Args:
        prefix: 共享内存名称前缀
        
    Returns:
        匹配的共享内存名称列表
    """
    # POSIX IPC 没有提供列出所有对象的 API
    # 需要通过 /dev/shm 目录查找
    import os
    shm_dir = "/dev/shm"
    
    if not os.path.exists(shm_dir):
        logger.warning(f"SHM directory not found: {shm_dir}")
        return []
    
    shm_objects = []
    for name in os.listdir(shm_dir):
        if name.startswith(prefix):
            shm_objects.append(name)
    
    return shm_objects


def cleanup_shm(name: str, dry_run: bool = False) -> bool:
    """清理单个共享内存对象
    
    Args:
        name: 共享内存名称
        dry_run: 是否只是模拟运行
        
    Returns:
        是否成功清理
    """
    try:
        if dry_run:
            logger.info(f"[DRY RUN] Would unlink: {name}")
            return True
        
        # 尝试打开并删除
        try:
            shm = posix_ipc.SharedMemory(name, flags=0)
            size = shm.size
            shm.close_fd()
            shm.unlink()
            logger.info(f"✅ Unlinked: {name} (size={size / 1024**2:.2f} MB)")
            return True
        except posix_ipc.ExistentialError:
            logger.warning(f"⚠️  Not found: {name}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to unlink {name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup VIT Scheduler POSIX shared memory"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="vit_embed",
        help="Shared memory name prefix (default: vit_embed)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be cleaned, don't actually clean"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force cleanup without confirmation"
    )
    
    args = parser.parse_args()
    
    logger.info(f"🔍 Searching for shared memory objects with prefix: {args.prefix}")
    
    # 列出所有匹配的共享内存对象
    shm_objects = list_shm_objects(args.prefix)
    
    if not shm_objects:
        logger.info("✅ No shared memory objects found")
        return 0
    
    logger.info(f"📋 Found {len(shm_objects)} shared memory objects:")
    for name in shm_objects:
        logger.info(f"  - {name}")
    
    # 确认清理
    if not args.force and not args.dry_run:
        response = input(f"\n⚠️  Clean up {len(shm_objects)} objects? [y/N]: ")
        if response.lower() != 'y':
            logger.info("❌ Cancelled")
            return 1
    
    # 清理
    success_count = 0
    fail_count = 0
    
    for name in shm_objects:
        if cleanup_shm(name, dry_run=args.dry_run):
            success_count += 1
        else:
            fail_count += 1
    
    # 总结
    logger.info(f"\n📊 Summary:")
    logger.info(f"  ✅ Success: {success_count}")
    logger.info(f"  ❌ Failed: {fail_count}")
    
    if args.dry_run:
        logger.info("\n💡 Run without --dry-run to actually clean up")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

