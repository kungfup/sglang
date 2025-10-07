#!/usr/bin/env python3
"""
测试迁移的功能是否正常工作
"""

import os
import sys

# 添加当前目录的 python 路径到 sys.path 最前面
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.join(current_dir, "python")
sys.path.insert(0, python_dir)


def test_imports():
    """测试所有修改的模块是否能正常导入"""
    print("Testing imports...")
    
    try:
        # Test parallel_state imports
        from sglang.srt.distributed.parallel_state import (
            get_pipeline_model_parallel_layer_split,
            get_pp_indices,
            initialize_model_parallel,
        )
        print("✓ parallel_state imports successful")
    except Exception as e:
        print(f"✗ parallel_state imports failed: {e}")
        return False
    
    try:
        # Test mm_utils imports
        from sglang.srt.managers.mm_utils import hash_feature, tensor_hash
        print("✓ mm_utils imports successful")
    except Exception as e:
        print(f"✗ mm_utils imports failed: {e}")
        return False
    
    try:
        # Test schedule_batch imports
        from sglang.srt.managers.schedule_batch import MultimodalDataItem
        print("✓ schedule_batch imports successful")
    except Exception as e:
        print(f"✗ schedule_batch imports failed: {e}")
        return False
    
    try:
        # Test vit_worker imports
        from sglang.srt.managers.vit_worker import ViTWorkerManager, ViTWorkerThread
        print("✓ vit_worker imports successful")
    except Exception as e:
        print(f"✗ vit_worker imports failed: {e}")
        return False
    
    return True


def test_get_pp_indices():
    """测试 get_pp_indices 函数"""
    print("\nTesting get_pp_indices...")
    
    try:
        from sglang.srt.distributed.parallel_state import get_pp_indices
        
        # Test case 1: 均匀分割
        num_layers = 32
        pp_size = 4
        
        for pp_rank in range(pp_size):
            start, end = get_pp_indices(num_layers, pp_rank, pp_size)
            print(f"  PP rank {pp_rank}: layers [{start}, {end})")
            
            # 验证范围
            assert 0 <= start < end <= num_layers, f"Invalid range: [{start}, {end})"
        
        # Test case 2: 自定义分割
        # 这个测试需要先设置 layer_split，暂时跳过
        
        print("✓ get_pp_indices tests passed")
        return True
    except Exception as e:
        print(f"✗ get_pp_indices tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hash_feature():
    """测试 hash_feature 函数"""
    print("\nTesting hash_feature...")
    
    try:
        import torch
        import numpy as np
        from sglang.srt.managers.mm_utils import hash_feature
        
        # Test case 1: torch.Tensor
        tensor = torch.randn(10, 20)
        hash1 = hash_feature(tensor)
        assert isinstance(hash1, int), "Hash should be an integer"
        print(f"  Tensor hash: {hash1}")
        
        # Test case 2: numpy array
        array = np.random.randn(10, 20)
        hash2 = hash_feature(array)
        assert isinstance(hash2, int), "Hash should be an integer"
        print(f"  Numpy array hash: {hash2}")
        
        # Test case 3: list of tensors
        tensor_list = [torch.randn(5, 10), torch.randn(5, 10)]
        hash3 = hash_feature(tensor_list)
        assert isinstance(hash3, int), "Hash should be an integer"
        print(f"  Tensor list hash: {hash3}")
        
        print("✓ hash_feature tests passed")
        return True
    except Exception as e:
        print(f"✗ hash_feature tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vit_worker_basic():
    """测试 ViT Worker 基本功能"""
    print("\nTesting ViT Worker basic functionality...")
    
    try:
        from sglang.srt.managers.vit_worker import ViTWorkerManager
        
        # 创建一个禁用的 worker（不需要实际的 ViT 模型）
        worker = ViTWorkerManager(vit_model=None, device="cuda:0", enable=False)
        
        # 测试统计信息
        stats = worker.get_stats()
        assert stats["submitted"] == 0, "Initial submitted count should be 0"
        assert stats["completed"] == 0, "Initial completed count should be 0"
        print(f"  Initial stats: {stats}")
        
        # 测试禁用状态下的操作
        result = worker.submit_task("test_id", None, None)
        assert result is False, "Disabled worker should not accept tasks"
        
        result = worker.get_result("test_id", timeout=0.1)
        assert result is None, "Disabled worker should return None"
        
        print("✓ ViT Worker basic tests passed")
        return True
    except Exception as e:
        print(f"✗ ViT Worker basic tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("SGLang Migration Tests")
    print("=" * 60)
    
    all_passed = True
    
    # 运行测试
    all_passed &= test_imports()
    all_passed &= test_get_pp_indices()
    all_passed &= test_hash_feature()
    all_passed &= test_vit_worker_basic()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    else:
        print("Some tests failed! ✗")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

