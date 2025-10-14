"""
Unit tests for VIT Shared Memory components

Tests:
1. SharedMemoryHelper - tensor serialization and SHM read/write
2. SharedMemoryManager - lifecycle management, reference counting, LRU eviction
3. DynamicMemoryEstimator - memory estimation and batch_size calculation
"""

import os
import time
import unittest
import torch

# 设置环境变量启用 SHM 模式
os.environ["SGLANG_VIT_USE_SHM"] = "true"

try:
    import posix_ipc
    POSIX_IPC_AVAILABLE = True
except ImportError:
    POSIX_IPC_AVAILABLE = False

# 只在 posix_ipc 可用时运行测试
@unittest.skipIf(not POSIX_IPC_AVAILABLE, "posix_ipc not available")
class TestSharedMemoryHelper(unittest.TestCase):
    """测试 SharedMemoryHelper 工具类"""
    
    def setUp(self):
        """测试前准备"""
        from sglang.srt.managers.vit_scheduler import SharedMemoryHelper
        self.helper = SharedMemoryHelper
    
    def test_tensor2bytes_float16(self):
        """测试 float16 tensor 序列化"""
        tensor = torch.randn(10, 20, dtype=torch.float16)
        data = self.helper.tensor2bytes(tensor)
        
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)
    
    def test_tensor2bytes_bfloat16(self):
        """测试 bfloat16 tensor 序列化"""
        tensor = torch.randn(10, 20, dtype=torch.bfloat16)
        data = self.helper.tensor2bytes(tensor)
        
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)
    
    def test_bytes2tensor_roundtrip(self):
        """测试序列化和反序列化往返"""
        original = torch.randn(10, 20, dtype=torch.float16)
        data = self.helper.tensor2bytes(original)
        restored = self.helper.bytes2tensor(data)
        
        self.assertEqual(original.shape, restored.shape)
        self.assertEqual(original.dtype, restored.dtype)
        torch.testing.assert_close(original, restored)
    
    def test_write_read_shm(self):
        """测试 SHM 写入和读取"""
        request_id = "test_request_001"
        tensor = torch.randn(5, 10, dtype=torch.float16)
        
        # 写入
        success = self.helper.write_to_shm(request_id, tensor)
        self.assertTrue(success)
        
        # 读取
        restored = self.helper.read_from_shm(request_id)
        self.assertIsNotNone(restored)
        
        # 验证
        self.assertEqual(tensor.shape, restored.shape)
        self.assertEqual(tensor.dtype, restored.dtype)
        torch.testing.assert_close(tensor, restored)
        
        # 清理
        self.helper.cleanup_shm(request_id)
    
    def tearDown(self):
        """测试后清理"""
        # 清理可能残留的 SHM
        try:
            self.helper.cleanup_shm("test_request_001")
        except:
            pass


@unittest.skipIf(not POSIX_IPC_AVAILABLE, "posix_ipc not available")
class TestSharedMemoryManager(unittest.TestCase):
    """测试 SharedMemoryManager"""
    
    def setUp(self):
        """测试前准备"""
        from sglang.srt.managers.vit_scheduler import SharedMemoryManager
        self.manager = SharedMemoryManager(
            max_shm_size_gb=1.0,  # 1 GB for testing
            cleanup_interval_sec=1.0,
            expired_timeout_sec=5.0,
        )
    
    def test_write_embedding(self):
        """测试写入 embedding"""
        request_id = "test_req_001"
        embedding = torch.randn(100, 200, dtype=torch.float16)
        
        success = self.manager.write_embedding(request_id, embedding)
        self.assertTrue(success)
        
        # 验证注册
        self.assertIn(request_id, self.manager.shm_registry)
    
    def test_acquire_release(self):
        """测试引用计数"""
        request_id = "test_req_002"
        embedding = torch.randn(100, 200, dtype=torch.float16)
        
        # 写入
        self.manager.write_embedding(request_id, embedding)
        
        # 获取引用
        self.manager.acquire(request_id)
        entry = self.manager.shm_registry[request_id]
        self.assertEqual(entry.ref_count, 2)  # write + acquire
        
        # 释放引用
        self.manager.release(request_id)
        entry = self.manager.shm_registry[request_id]
        self.assertEqual(entry.ref_count, 1)
    
    def test_lru_eviction(self):
        """测试 LRU 驱逐"""
        # 写入多个 embedding,超过容量
        embeddings = []
        for i in range(10):
            request_id = f"test_req_{i:03d}"
            # 每个 embedding 约 100 MB
            embedding = torch.randn(25000, 1000, dtype=torch.float16)
            embeddings.append((request_id, embedding))
        
        # 写入前几个应该成功
        for request_id, embedding in embeddings[:5]:
            success = self.manager.write_embedding(request_id, embedding)
            self.assertTrue(success)
        
        # 继续写入会触发 LRU 驱逐
        for request_id, embedding in embeddings[5:]:
            self.manager.write_embedding(request_id, embedding)
        
        # 验证总大小不超过限制
        total_size = sum(e.size_bytes for e in self.manager.shm_registry.values())
        self.assertLessEqual(total_size, self.manager.max_shm_size_bytes)
    
    def tearDown(self):
        """测试后清理"""
        self.manager.cleanup_all()


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestDynamicMemoryEstimator(unittest.TestCase):
    """测试 DynamicMemoryEstimator"""
    
    def setUp(self):
        """测试前准备"""
        from sglang.srt.managers.vit_scheduler import DynamicMemoryEstimator
        self.estimator = DynamicMemoryEstimator(
            device="cuda:0",
            safety_margin_gb=0.5,
            overhead_ratio=4.0,
        )
    
    def test_get_available_memory(self):
        """测试获取可用显存"""
        available = self.estimator.get_available_memory()
        self.assertGreater(available, 0)
    
    def test_estimate_request_memory(self):
        """测试估算请求显存"""
        pixel_values = torch.randn(4, 3, 224, 224, dtype=torch.float16)
        estimated = self.estimator.estimate_request_memory(pixel_values)
        
        self.assertGreater(estimated, 0)
        # 估算应该是输入大小的 overhead_ratio 倍
        input_size = pixel_values.numel() * pixel_values.element_size()
        expected = int(input_size * 4.0)
        self.assertAlmostEqual(estimated, expected, delta=1000)
    
    def test_get_max_batch_size(self):
        """测试计算最大 batch_size"""
        pixel_values_list = [
            torch.randn(1, 3, 224, 224, dtype=torch.float16)
            for _ in range(10)
        ]
        
        max_batch_size = self.estimator.get_max_batch_size(pixel_values_list)
        
        self.assertGreater(max_batch_size, 0)
        self.assertLessEqual(max_batch_size, len(pixel_values_list))


if __name__ == "__main__":
    unittest.main()

