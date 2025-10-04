# ViT 解耦到独立进程 - 完整技术方案

## 📋 目录
1. [当前 ViT 调用流程分析](#当前-vit-调用流程分析)
2. [核心问题](#核心问题)
3. [解耦方案设计](#解耦方案设计)
4. [技术实现路线](#技术实现路线)
5. [关键技术点](#关键技术点)
6. [实现步骤](#实现步骤)

---

## 当前 ViT 调用流程分析

### 1. 请求到达 → Scheduler

```
用户请求 (image + text)
    ↓
Tokenizer 进程 (process_input_requests)
    ↓
Scheduler.process_input_requests(recv_reqs)
    ↓
创建 Req 对象，包含:
    - origin_input_ids: [text_tokens + <image_placeholder_tokens>]
    - mm_inputs: MultimodalInputs
        - mm_items: List[MultimodalDataItem]
            - pixel_values: torch.Tensor (原始图片数据)
            - image_grid_thw: torch.Tensor (图片网格信息)
    ↓
加入 waiting_queue
```

### 2. Scheduler → Batch 构建

```
event_loop_pp() 主循环
    ↓
get_next_batch_to_run()
    ↓
构建 ScheduleBatch:
    - reqs: List[Req]
    - forward_mode: EXTEND (prefill)
    - mm_inputs: List[MultimodalInput] (从 Req 中提取)
    ↓
run_batch(batch)
```

### 3. Batch → Model Forward

```
run_batch(batch)
    ↓
tp_worker.forward_batch_generation(batch)
    ↓
model_runner.forward(forward_batch)
    ↓
model.forward(input_ids, positions, forward_batch)
    ↓
[PP0] _prepare_initial_embeddings(input_ids, forward_batch)
    ↓
embed_mm_inputs(
    mm_inputs_list,
    extend_prefix_lens,
    extend_seq_lens,
    input_ids,
    input_embedding,
    image_data_embedding_func=self.get_image_feature,  # ← ViT 在这里被调用！
)
```

### 4. ViT 计算流程

```
embed_mm_inputs()
    ↓
_get_chunked_prefill_embedding()
    ↓
检查 embedding_cache (基于 hash)
    ↓
如果未命中:
    embedding = image_data_embedding_func(embedding_items)  # ← 调用 get_image_feature
        ↓
        self.get_image_feature(items)
            ↓
            pixel_values = torch.cat([item.pixel_values for item in items])
            image_grid_thw = torch.cat([item.image_grid_thw for item in items])
            ↓
            [ViT 异步优化] with torch.cuda.stream(self.vit_stream):
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)  # ← ViT 计算！
            ↓
            返回 image_embeds: [num_image_tokens, hidden_dim]  # 例如 [6480, 3584]
    ↓
    存入 embedding_cache
    ↓
返回 embedding_chunk
    ↓
_replace_multimodal_embeddings()
    ↓
将 image_embeds 替换到 inputs_embeds 中对应的 placeholder 位置
    ↓
返回完整的 inputs_embeds: [total_tokens, hidden_dim]
```

### 5. 关键数据结构

```python
# MultimodalDataItem (在 schedule_batch.py 中定义)
@dataclasses.dataclass
class MultimodalDataItem:
    modality: Modality  # IMAGE, VIDEO, AUDIO
    hash: int  # 用于缓存查找
    pixel_values: torch.Tensor  # [1, num_patches] 或 [batch, num_patches]
    image_grid_thw: torch.Tensor  # [1, 3] 或 [batch, 3]，表示 [T, H, W]
    
# MultimodalInput (在 schedule_batch.py 中定义)
@dataclasses.dataclass
class MultimodalInput:
    mm_items: List[MultimodalDataItem]
    mrope_positions: torch.Tensor  # mRoPE 位置编码
    
# ForwardBatch (在 forward_batch_info.py 中定义)
@dataclasses.dataclass
class ForwardBatch:
    mm_inputs: List[Optional[MultimodalInput]]  # 每个请求一个
    extend_prefix_lens_cpu: List[int]  # 每个请求的 prefix 长度
    extend_seq_lens_cpu: List[int]  # 每个请求的序列长度
```

---

## 核心问题

### 1. ViT 计算在关键路径上

```
时间轴（当前）:
┌─────────────────────────────────────────────────────────────┐
│ Scheduler 主线程                                            │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐             │
│  │ 构建     │  │ ViT 计算     │  │ LLM      │             │
│  │ Batch    │→ │ (540ms)      │→ │ Forward  │             │
│  └──────────┘  └──────────────┘  └──────────┘             │
│                     ↑ 阻塞整个 Pipeline！                   │
└─────────────────────────────────────────────────────────────┘
```

**问题**：
- ViT 计算在 `run_batch()` 中同步执行
- 即使使用了 CUDA stream 异步，仍然在 Scheduler 主线程中
- 阻塞了 Batch 的构建和调度
- PP0 和 PP1 串行执行，无法流水线并行

### 2. ViT 计算时间占比过高

```
单个请求的 Prefill 时间分解:
- ViT 计算: 540ms (93%)
- LLM Prefill (PP0): 20ms (3%)
- LLM Prefill (PP1): 20ms (3%)
- 总时间: 580ms

瓶颈: ViT 计算时间 >> LLM 计算时间
```

### 3. GPU 利用率低

```
PP0 (GPU0):
  ViT 计算 (540ms) → GPU 利用率 100%
  等待 PP1 (20ms) → GPU 利用率 0%

PP1 (GPU1):
  等待 PP0 (540ms) → GPU 利用率 0%
  LLM 计算 (20ms) → GPU 利用率 100%

平均 GPU 利用率: ~50%
```

---

## 解耦方案设计

### 核心思想

**将 ViT 计算从 Scheduler 主线程中解耦到独立进程，完全异步化**

```
理想时间轴:
┌─────────────────────────────────────────────────────────────┐
│ Scheduler 主线程                                            │
│  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐                 │
│  │Batch1││Batch2││Batch3││Batch4││Batch5│                 │
│  └──────┘└──────┘└──────┘└──────┘└──────┘                 │
└─────────────────────────────────────────────────────────────┘
         ↓       ↓       ↓       ↓       ↓
┌─────────────────────────────────────────────────────────────┐
│ ViT 独立进程 (后台计算)                                     │
│  ┌──────────────┐┌──────────────┐┌──────────────┐         │
│  │ViT(Req1)     ││ViT(Req2)     ││ViT(Req3)     │         │
│  │(540ms)       ││(540ms)       ││(540ms)       │         │
│  └──────────────┘└──────────────┘└──────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│ 用户请求                                                    │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Tokenizer 进程                                              │
│  - 处理图片预处理                                           │
│  - 生成 pixel_values, image_grid_thw                        │
└─────────────────────────────────────────────────────────────┘
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
┌──────────────────┐    ┌──────────────────────────────────┐
│ Scheduler 进程   │    │ ViT Worker 进程 (新增)          │
│                  │    │                                  │
│ 1. 接收请求      │    │ 1. 独立的 GPU 进程               │
│ 2. 提交 ViT 任务 │───→│ 2. 维护 ViT 模型                 │
│    (非阻塞)      │    │ 3. 接收 ViT 计算任务             │
│ 3. 构建 Batch    │    │ 4. 批量计算 ViT                  │
│ 4. 查询 ViT 结果 │←───│ 5. 返回 embedding 结果           │
│    (非阻塞查询)  │    │                                  │
│ 5. 如果就绪:     │    │ 特点:                            │
│    - 使用缓存    │    │ - 完全异步                       │
│    - 执行 LLM    │    │ - 支持 batch 计算                │
│    如果未就绪:   │    │ - 独立的 CUDA context            │
│    - 跳过该请求  │    │ - 不阻塞 Scheduler               │
│    - 处理其他    │    │                                  │
└──────────────────┘    └──────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│ Model Worker 进程 (PP0, PP1)                                │
│  - 只负责 LLM 计算                                          │
│  - 使用预计算的 image embeddings                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 技术实现路线

### 方案 A：基于共享内存 + 多进程（推荐）⭐⭐⭐⭐⭐

**优点**：
- 完全解耦，ViT 进程独立运行
- 支持 batch 计算（多个请求的 ViT 一起算）
- 可以使用独立的 GPU（如果有多卡）
- 零拷贝通信（共享内存）

**缺点**：
- 实现复杂度中等
- 需要管理进程生命周期

**技术栈**：
- `torch.multiprocessing` (进程管理)
- `torch.cuda.IpcMemoryHandle` (GPU 共享内存)
- `multiprocessing.Queue` (任务队列)
- `multiprocessing.shared_memory` (CPU 共享内存，用于元数据)

### 方案 B：基于 gRPC/ZMQ + 独立服务（备选）⭐⭐⭐

**优点**：
- 可以部署在不同机器上
- 支持多个 ViT Worker（负载均衡）
- 容错性好

**缺点**：
- 网络通信开销
- 需要序列化/反序列化
- 实现复杂度高

**技术栈**：
- `grpc` 或 `zmq` (通信)
- `torch.save/load` (序列化)

### 方案 C：基于线程池 + 异步队列（最简单，但效果有限）⭐⭐

**优点**：
- 实现简单
- 无需进程管理

**缺点**：
- 受 GIL 限制（Python）
- 无法真正并行（CPU 部分）
- 无法使用独立 GPU

---

## 关键技术点

### 1. ViT 任务提交时机

**策略**：在请求到达 Scheduler 时立即提交

```python
# 在 Scheduler.process_input_requests() 中
def process_input_requests(self, recv_reqs):
    for recv_req in recv_reqs:
        # 创建 Req 对象
        req = self._create_request(recv_req)
        
        # 如果有图片，立即提交 ViT 任务（非阻塞）
        if req.mm_inputs is not None:
            self.vit_worker.submit_task(
                request_id=req.rid,
                pixel_values=req.mm_inputs.mm_items[0].pixel_values,
                image_grid_thw=req.mm_inputs.mm_items[0].image_grid_thw,
            )
        
        # 加入等待队列
        self.waiting_queue.append(req)
```

### 2. ViT 结果查询时机

**策略**：在构建 Batch 时非阻塞查询

```python
# 在 Scheduler.get_next_batch_to_run() 中
def get_next_batch_to_run(self):
    # 遍历 waiting_queue
    for req in self.waiting_queue:
        if req.mm_inputs is not None:
            # 非阻塞查询 ViT 结果
            embedding = self.vit_worker.try_get_result(req.rid)
            if embedding is None:
                # ViT 还没算完，跳过这个请求
                continue
            else:
                # ViT 已完成，缓存结果
                req.mm_inputs.precomputed_embedding = embedding
        
        # 将请求加入 batch
        batch.add_request(req)
```

### 3. Embedding 缓存机制

**两级缓存**：
1. **ViT Worker 内部缓存**：避免重复计算相同图片
2. **Scheduler 端缓存**：存储已完成的 embedding

```python
# ViT Worker 端
class ViTWorker:
    def __init__(self):
        self.embedding_cache = {}  # hash -> embedding
    
    def compute_vit(self, pixel_values, image_grid_thw):
        # 计算 hash
        hash_val = self._compute_hash(pixel_values, image_grid_thw)
        
        # 查询缓存
        if hash_val in self.embedding_cache:
            return self.embedding_cache[hash_val]
        
        # 计算 ViT
        embedding = self.vit_model(pixel_values, grid_thw=image_grid_thw)
        
        # 存入缓存
        self.embedding_cache[hash_val] = embedding
        return embedding

# Scheduler 端
class Scheduler:
    def process_vit_result(self, req, embedding):
        # 将 embedding 存入 MultimodalDataItem
        req.mm_inputs.mm_items[0].precomputed_features = embedding
```

### 4. Batch ViT 计算

**策略**：累积多个请求，批量计算 ViT

```python
class ViTWorker:
    def __init__(self, batch_size=4, batch_timeout=0.01):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_tasks = []
    
    def worker_loop(self):
        while True:
            # 累积任务
            task = self.task_queue.get(timeout=self.batch_timeout)
            self.pending_tasks.append(task)
            
            # 如果凑够 batch 或超时，执行计算
            if len(self.pending_tasks) >= self.batch_size or self._is_timeout():
                self._batch_compute()
    
    def _batch_compute(self):
        # 批量拼接
        pixel_values_list = [t['pixel_values'] for t in self.pending_tasks]
        grid_thw_list = [t['grid_thw'] for t in self.pending_tasks]
        
        pixel_values_batch = torch.cat(pixel_values_list, dim=0)
        grid_thw_batch = torch.cat(grid_thw_list, dim=0)
        
        # 批量计算
        embeddings_batch = self.vit_model(pixel_values_batch, grid_thw=grid_thw_batch)
        
        # 拆分结果
        split_sizes = [pv.shape[0] for pv in pixel_values_list]
        embeddings_list = torch.split(embeddings_batch, split_sizes, dim=0)
        
        # 返回结果
        for task, embedding in zip(self.pending_tasks, embeddings_list):
            self.result_cache[task['request_id']] = embedding
        
        self.pending_tasks = []
```

### 5. 共享内存通信

**GPU 共享内存**（零拷贝）：

```python
# ViT Worker 端（发送）
def send_embedding_via_shared_memory(embedding: torch.Tensor, request_id: str):
    # 创建 IPC handle
    ipc_handle = embedding.cuda().share_memory_()
    
    # 发送元数据（通过 Queue）
    metadata = {
        'request_id': request_id,
        'shape': embedding.shape,
        'dtype': str(embedding.dtype),
        'device': str(embedding.device),
        'ipc_handle': ipc_handle,
    }
    result_queue.put(metadata)

# Scheduler 端（接收）
def receive_embedding_via_shared_memory(metadata):
    # 从 IPC handle 重建 tensor
    embedding = torch.cuda.FloatTensor(
        storage=torch.cuda.UntypedStorage._new_shared_cuda(
            metadata['ipc_handle'],
            size=torch.prod(torch.tensor(metadata['shape'])).item() * 4,  # float32 = 4 bytes
            device=metadata['device'],
        )
    ).view(metadata['shape'])
    
    return embedding
```

---

## 实现步骤

### Phase 1: ViT Worker 进程（核心）

**文件**：`sglang/python/sglang/srt/managers/vit_worker_process.py`

```python
import torch
import torch.multiprocessing as mp
from queue import Empty
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ViTWorkerProcess:
    """ViT 独立进程 Worker"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        batch_size: int = 4,
        batch_timeout: float = 0.01,
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # 创建进程间通信队列
        self.task_queue = mp.Queue(maxsize=100)
        self.result_queue = mp.Queue(maxsize=100)
        
        # 启动 Worker 进程
        self.process = mp.Process(
            target=self._worker_main,
            args=(self.task_queue, self.result_queue),
            daemon=True,
        )
        self.process.start()
        logger.info(f"[ViT Worker] Started process PID={self.process.pid}")
    
    def _worker_main(self, task_queue, result_queue):
        """Worker 进程主函数"""
        # 加载 ViT 模型
        from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer
        
        logger.info(f"[ViT Worker] Loading ViT model on {self.device}")
        # TODO: 加载模型配置
        self.vit_model = Qwen2_5_VisionTransformer(...)
        self.vit_model.to(self.device)
        self.vit_model.eval()
        
        # 缓存
        self.embedding_cache = {}
        
        # 批处理缓冲区
        self.pending_tasks = []
        self.last_batch_time = time.time()
        
        logger.info("[ViT Worker] Ready to process tasks")
        
        # 主循环
        while True:
            try:
                # 获取任务（带超时）
                task = task_queue.get(timeout=self.batch_timeout)
                self.pending_tasks.append(task)
                
                # 判断是否执行 batch 计算
                should_compute = (
                    len(self.pending_tasks) >= self.batch_size or
                    time.time() - self.last_batch_time > self.batch_timeout
                )
                
                if should_compute:
                    self._batch_compute_and_send(result_queue)
                    
            except Empty:
                # 超时，检查是否有待处理任务
                if len(self.pending_tasks) > 0:
                    self._batch_compute_and_send(result_queue)
    
    def _batch_compute_and_send(self, result_queue):
        """批量计算并发送结果"""
        # TODO: 实现批量计算逻辑
        pass
    
    def submit_task(self, request_id: str, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        """提交 ViT 计算任务（非阻塞）"""
        task = {
            'request_id': request_id,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
        }
        try:
            self.task_queue.put_nowait(task)
            return True
        except:
            logger.warning(f"[ViT Worker] Task queue full, dropping task {request_id}")
            return False
    
    def try_get_result(self, request_id: str) -> Optional[torch.Tensor]:
        """非阻塞查询结果"""
        try:
            result = self.result_queue.get_nowait()
            if result['request_id'] == request_id:
                return result['embedding']
        except Empty:
            return None
    
    def shutdown(self):
        """关闭 Worker 进程"""
        self.process.terminate()
        self.process.join()
```

### Phase 2: Scheduler 集成

**修改文件**：`sglang/python/sglang/srt/managers/scheduler.py`

1. **初始化 ViT Worker**
2. **提交任务**（在 `process_input_requests` 中）
3. **查询结果**（在 `get_next_batch_to_run` 中）

### Phase 3: 修改 embed_mm_inputs

**修改文件**：`sglang/python/sglang/srt/managers/mm_utils.py`

支持使用 `precomputed_features`，跳过 ViT 计算。

### Phase 4: 测试和优化

1. **功能测试**：单请求、多请求、并发请求
2. **性能测试**：吞吐量、延迟、GPU 利用率
3. **稳定性测试**：长时间运行、异常处理

---

## 预期效果

### 吞吐量提升

```
当前（串行）:
  单请求延迟: 580ms
  吞吐量: 1.72 req/s

解耦后（并行）:
  ViT 计算: 540ms (后台)
  LLM 计算: 40ms (前台)
  
  如果有 N 个并发请求:
    吞吐量 ≈ N / 540ms = 1.85N req/s
  
  例如 N=10:
    吞吐量 ≈ 18.5 req/s
    加速比 ≈ 10.7x
```

### GPU 利用率提升

```
当前:
  PP0: 50% (ViT 时 100%, 等待时 0%)
  PP1: 50% (计算时 100%, 等待时 0%)

解耦后:
  PP0: 90%+ (持续处理 LLM)
  PP1: 90%+ (持续处理 LLM)
  ViT GPU: 90%+ (持续处理 ViT)
```

---

## 总结

这个方案的核心是：
1. **完全解耦**：ViT 计算不阻塞 Scheduler
2. **批量计算**：多个请求的 ViT 一起算
3. **零拷贝通信**：使用 GPU 共享内存
4. **异步查询**：Scheduler 非阻塞查询结果

这样可以让 LLM Pipeline 持续运行，ViT 在后台并行计算，充分利用 GPU 资源！

