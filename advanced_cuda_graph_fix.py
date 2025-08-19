#!/usr/bin/env python3
"""
高级修复：针对32B模型的CUDA Graph完全优化
"""

import torch
import os

class OptimizedCudaGraphRunner32B:
    """专门为32B模型优化的CUDA Graph Runner"""
    
    def __init__(self, model_runner):
        self.model_runner = model_runner
        
        # 32B模型特定配置
        self.use_graph_pool = True  # 使用graph池
        self.graph_pool = {}  # bs -> [graphs]
        self.max_graphs_per_bs = 3  # 每个batch size最多缓存3个graph
        
        # Stream管理
        self.capture_stream = torch.cuda.Stream()
        self.replay_streams = [torch.cuda.Stream() for _ in range(2)]  # 双buffer
        self.current_stream_idx = 0
        
        # 内存优化
        self.enable_memory_pool = True
        if self.enable_memory_pool:
            self._setup_memory_pool()
    
    def _setup_memory_pool(self):
        """设置内存池优化"""
        # 为32B模型预分配大块内存
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024,expandable_segments:True"
        
        # 预热内存分配器
        dummy = torch.zeros((16*1024*1024*1024//4,), dtype=torch.float16, device='cuda')
        del dummy
        torch.cuda.empty_cache()
    
    def capture_optimized(self, bs):
        """优化的capture流程"""
        # 使用独立的capture stream
        with torch.cuda.stream(self.capture_stream):
            # 禁用梯度计算
            with torch.no_grad():
                # 禁用cudnn benchmark避免额外开销
                with torch.backends.cudnn.flags(benchmark=False):
                    graph = torch.cuda.CUDAGraph()
                    
                    # 只capture一次，不做warmup
                    with torch.cuda.graph(graph, stream=self.capture_stream):
                        output = self.model_runner.forward(...)
                    
                    # 缓存到graph池
                    if bs not in self.graph_pool:
                        self.graph_pool[bs] = []
                    
                    if len(self.graph_pool[bs]) < self.max_graphs_per_bs:
                        self.graph_pool[bs].append(graph)
                    
                    return graph, output
    
    def replay_optimized(self, forward_batch):
        """优化的replay流程"""
        bs = self.get_padded_batch_size(forward_batch.batch_size)
        
        # 选择replay stream（双buffer）
        stream = self.replay_streams[self.current_stream_idx]
        self.current_stream_idx = (self.current_stream_idx + 1) % 2
        
        # 在选定的stream上执行所有操作
        with torch.cuda.stream(stream):
            # 1. 异步内存拷贝
            self._async_copy_inputs(forward_batch)
            
            # 2. 选择合适的graph
            graph = self._select_best_graph(bs)
            
            # 3. Replay without sync
            graph.replay()
            
            # 4. 对于32B模型，使用event同步而非stream同步
            if self.needs_sync:
                event = torch.cuda.Event()
                event.record(stream)
                # 稍后在需要结果时才wait
                return self.output_buffers[bs], event
            else:
                return self.output_buffers[bs], None
    
    def _async_copy_inputs(self, forward_batch):
        """异步拷贝输入数据"""
        # 使用non_blocking=True加速
        self.input_ids.copy_(forward_batch.input_ids, non_blocking=True)
        self.positions.copy_(forward_batch.positions, non_blocking=True)
        # ... 其他拷贝
    
    def _select_best_graph(self, bs):
        """选择最佳的graph"""
        if bs in self.graph_pool and self.graph_pool[bs]:
            # 轮询使用，避免单个graph过热
            graph = self.graph_pool[bs][0]
            self.graph_pool[bs].rotate(1)  # 轮转
            return graph
        else:
            # 需要新capture
            graph, _ = self.capture_optimized(bs)
            return graph
