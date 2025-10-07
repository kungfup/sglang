# 关键修改对比

本文档展示了迁移过程中的关键代码修改对比。

## 1. parallel_state.py - Pipeline Parallel 增强

### 添加全局配置

```python
# 新增
_PIPELINE_GLOBAL_CONFIG = {
    "layer_split": None,
}

def get_pipeline_model_parallel_layer_split():
    """Get the pipeline model parallel layer split."""
    return _PIPELINE_GLOBAL_CONFIG["layer_split"]
```

### initialize_model_parallel 函数签名

```python
# 旧版本
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
    duplicate_tp_group: bool = False,
) -> None:

# 新版本
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_model_parallel_layer_split: Optional[List[int]] = None,  # 新增
    backend: Optional[str] = None,
    duplicate_tp_group: bool = False,
) -> None:
```

### TP/PP 组构建逻辑

```python
# 旧版本 - TP 组
num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
group_ranks = []
for i in range(num_tensor_model_parallel_groups):
    ranks = list(
        range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
    )
    group_ranks.append(ranks)

# 新版本 - TP 组（更清晰）
all_tp_groups = []
for i in range(pipeline_model_parallel_size):
    ranks = range(
        i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size
    )
    all_tp_groups.append(list(ranks))
```

```python
# 旧版本 - PP 组
num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
group_ranks = []
for i in range(num_pipeline_model_parallel_groups):
    ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
    group_ranks.append(ranks)

# 新版本 - PP 组（Megatron 风格）
all_pp_groups = []
for i in range(tensor_model_parallel_size):
    ranks = range(i, world_size, tensor_model_parallel_size)
    all_pp_groups.append(list(ranks))

# 存储层分割配置
_PIPELINE_GLOBAL_CONFIG["layer_split"] = pipeline_model_parallel_layer_split
```

### 新增 get_pp_indices 函数

```python
# 新增函数
def get_pp_indices(
    num_layers: int,
    pp_rank: Optional[int] = None,
    pp_size: Optional[int] = None,
) -> Tuple[int, int]:
    """Get the start and end layer indices for a given pipeline rank."""
    if pp_rank is None:
        pp_rank = get_pipeline_model_parallel_rank()
    if pp_size is None:
        pp_size = get_pipeline_model_parallel_world_size()

    layer_split = get_pipeline_model_parallel_layer_split()
    if layer_split:
        # 自定义分割
        if len(layer_split) != pp_size - 1:
            raise ValueError(
                "The number of layer splits must be equal to pp_size - 1."
            )
        start_layer = 0 if pp_rank == 0 else layer_split[pp_rank - 1]
        end_layer = (
            layer_split[pp_rank] if pp_rank < pp_size - 1 else num_layers
        )
        return start_layer, end_layer

    # 均匀分割
    layers_per_stage = num_layers // pp_size
    start_layer = pp_rank * layers_per_stage
    end_layer = (
        (pp_rank + 1) * layers_per_stage if pp_rank != pp_size - 1 else num_layers
    )
    return start_layer, end_layer
```

### destroy_model_parallel 改进

```python
# 旧版本
def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _TP
    if _TP:
        _TP.destroy()
    _TP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None

# 新版本
def destroy_model_parallel():
    """Destroy all model parallel groups."""
    global _TP
    if _TP:
        _TP.destroy()
    _TP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None

    monkey_patch_vllm_parallel_state(reverse=True)  # 新增
```

## 2. mm_utils.py - Tensor Hashing 优化

### tensor_hash 函数重构

```python
# 旧版本
def tensor_hash(tensor_list) -> int:
    """
    hash a tensor or a tensor list
    """
    tensor = tensor_list
    if isinstance(tensor_list, list):
        tensor_list = flatten_nested_list(tensor_list)
        tensor_list = [
            x.flatten() if isinstance(x, torch.Tensor) else x for x in tensor_list
        ]
        tensor = torch.concat(tensor_list)
    if tensor.is_cuda:
        return gpu_tensor_hash(tensor.cuda())
    tensor = tensor.detach().contiguous()

    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()

    assert isinstance(tensor, torch.Tensor)
    tensor_cpu = tensor.cpu()

    mv = memoryview(tensor_cpu.numpy())
    return data_hash(mv.tobytes())

# 新版本
def tensor_hash(tensor_list) -> int:
    """
    Hash a tensor or a list of tensors.
    It prioritizes using the GPU if available, and falls back to the CPU on failure.
    """
    tensor = tensor_list
    if isinstance(tensor_list, list):
        tensor_list = flatten_nested_list(tensor_list)
        tensor_list = [
            x.flatten() if isinstance(x, torch.Tensor) else x for x in tensor_list
        ]
        tensor = torch.concat(tensor_list)

    # Define a CPU fallback hash function
    def cpu_hash(t: torch.Tensor) -> int:
        if t.dtype == torch.bfloat16:
            t = t.float()
        # Ensure tensor is on CPU
        hash_bytes = hashlib.sha256(t.cpu().numpy().tobytes()).digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big", signed=False)

    # Prioritize GPU hash
    if torch.cuda.is_available():
        try:
            gpu_tensor = tensor.to("cuda", non_blocking=True)
            if gpu_tensor.dtype == torch.bfloat16:
                gpu_tensor = gpu_tensor.float()
            return gpu_tensor_hash(gpu_tensor)
        except Exception as e:
            logger.warning(f"GPU tensor hashing failed: {e}. Falling back to CPU hash.")
            return cpu_hash(tensor)
    else:
        # If no CUDA device is available, use CPU hash directly
        return cpu_hash(tensor)
```

## 3. schedule_batch.py - Pad Value 计算

### set_pad_value 方法

```python
# 旧版本
def set_pad_value(self):
    """
    Set the pad value after first hashing the data
    """
    from sglang.srt.managers.mm_utils import hash_feature

    if self.hash is None:
        if self.feature is not None:
            hashed_feature = self.feature
        else:
            hashed_feature = self.precomputed_embeddings
        self.hash = hash_feature(hashed_feature)
    assert self.hash is not None
    self.pad_value = self.hash % (1 << 30)  # 旧计算方式

# 新版本
def set_pad_value(self):
    """
    Set the pad value after first hashing the data.
    """
    from sglang.srt.managers.mm_utils import hash_feature

    if self.hash is None:
        if self.feature is not None:
            hashed_feature = self.feature
        else:
            hashed_feature = self.precomputed_embeddings
        self.hash = hash_feature(hashed_feature)
    assert self.hash is not None
    self.pad_value = self.hash % (2**31 - 1)  # 新计算方式
```

## 4. scheduler.py - PP 组清理

### run_scheduler_process 函数

```python
# 旧版本
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
# 函数结束

# 新版本
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
    finally:
        # 新增：Flush & 销毁 PP 组，防止 gloo 长尾
        try:
            from sglang.srt.distributed.parallel_state import get_pp_group

            pg = None
            try:
                pg = get_pp_group()
            except Exception:
                pg = None  # 未初始化 PP 组

            if pg is not None:
                try:
                    pg.barrier()  # 确保前一轮 Send/Recv 全部完成
                finally:
                    pg.destroy()  # 彻底关闭 device_group & cpu_group
        except Exception as _e:
            logger.warning(f"Error when flushing/destroying pp_group: {_e}")
```

## 5. vit_worker.py - 新增模块

这是一个全新的模块，提供 ViT 异步计算功能。

### 核心类

```python
class ViTWorkerThread:
    """ViT 工作线程（在后台处理 ViT 计算）"""
    
    def __init__(self, vit_model, device, result_cache, task_queue):
        self.vit_stream = torch.cuda.Stream()  # 独立 CUDA Stream
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

class ViTWorkerManager:
    """ViT Worker 管理器（主线程侧）"""
    
    def submit_task(self, request_id, pixel_values, grid_thw) -> bool:
        """非阻塞提交任务"""
        
    def get_result(self, request_id, timeout=10.0) -> Optional[torch.Tensor]:
        """阻塞等待结果"""
        
    def try_get_result(self, request_id) -> Optional[torch.Tensor]:
        """非阻塞获取结果"""
```

---

**说明**: 以上对比展示了主要的代码修改。完整的修改细节请参考各个文件的实际代码。

