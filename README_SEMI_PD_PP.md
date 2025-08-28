# 🎯 Semi-PD Pipeline Parallel 架构设计

## 概述

Semi-PD Pipeline Parallel (Semi-PD PP) 是SGLang中Semi-PD调度器的Pipeline Parallel扩展，支持多GPU的流水线并行推理。

## 🏗️ 架构设计

### 进程层次结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PP Stage 0 (GPU 0)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ┌─────────────────┐    ┌─────────────────┐                       │   │
│  │  │  DECODE_PP0     │◄──►│  PREFILL_PP0    │                       │   │
│  │  │   (主进程)      │IPC │   (辅助进程)    │                       │   │
│  │  │                 │    │                 │                       │   │
│  │  │ • 请求接收      │    │ • 预填充计算    │                       │   │
│  │  │ • 响应返回      │    │ • 配合主进程    │                       │   │
│  │  │ • 整体协调      │    │ • 共享权重      │                       │   │
│  │  └─────────────────┘    └─────────────────┘                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ NCCL通信
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PP Stage 1 (GPU 1)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ┌─────────────────┐    ┌─────────────────┐                       │   │
│  │  │  DECODE_PP1     │◄──►│  PREFILL_PP1    │                       │   │
│  │  │   (主进程)      │IPC │   (辅助进程)    │                       │   │
│  │  │                 │    │                 │                       │   │
│  │  │ • 请求接收      │    │ • 预填充计算    │                       │   │
│  │  │ • 响应返回      │    │ • 配合主进程    │                       │   │
│  │  │ • 整体协调      │    │ • 共享权重      │                       │   │
│  │  └─────────────────┘    └─────────────────┘                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 通信机制

#### 1. 同PP Stage内通信 (IPC)
- **DECODE进程 ↔ PREFILL进程**: 通过ZMQ IPC通信
- **共享模型权重**: PREFILL进程通过IPC共享DECODE进程的模型权重
- **请求协调**: DECODE进程作为主进程，协调PREFILL进程的工作

#### 2. PP Stage间通信 (NCCL)
- **DECODE_PP0 ↔ DECODE_PP1**: 通过SGLang原生的NCCL通信
- **PREFILL_PP0 ↔ PREFILL_PP1**: 通过SGLang原生的NCCL通信
- **中间隐藏状态传递**: 在推理过程中传递中间计算结果

## 🔧 关键特性

### 1. 主进程设计
- **DECODE进程是主进程**: 负责请求接收、响应返回、整体协调
- **PREFILL进程是辅助进程**: 负责预填充计算，配合主进程工作
- **启动顺序**: 所有PP stage的DECODE进程同时启动 → 等待就绪 → 所有PP stage的PREFILL进程同时启动

### 2. 内存共享
- **避免重复加载**: PREFILL进程通过IPC共享DECODE进程的模型权重
- **显存节省**: 每个PP stage只需要加载一次模型权重
- **零拷贝**: 使用IPC机制实现高效的内存共享

### 3. 设备隔离
- **不同PP stage使用不同GPU**: 确保NCCL通信组隔离
- **独立通信组**: 每个进程可以创建不同的通信组
- **GroupCoordinator使用local_rank**: `self.device = torch.device(f"cuda:{local_rank}")`

## 📡 端口分配策略

### 端口范围
- **PP Stage 0**: 40000-40099 (GPU 0)
- **PP Stage 1**: 40100-40199 (GPU 1)
- **PP Stage 2**: 40200-40299 (GPU 2)

### 端口分配
每个PP Stage内部：
```
Stage Base Port + 0:  DECODE进程 (主进程)
Stage Base Port + 1:  PREFILL进程 (辅助进程)
Stage Base Port + 2:  调度器
Stage Base Port + 3:  去tokenizer
Stage Base Port + 100: NCCL通信
```

## 🚀 请求流程

### 完整请求处理流程
```
1. 请求进入 → Stage0-DECODE (接收请求)
2. Stage0-DECODE → Stage0-PREFILL (IPC通信)
3. Stage0-PREFILL → Stage1-PREFILL (NCCL传递中间隐藏状态)
4. Stage1-PREFILL → Stage0-DECODE (跨stage传输)
5. Stage0-DECODE → Stage1-DECODE (NCCL传递中间隐藏状态)
6. Stage1-DECODE → Stage0-DECODE (返回生成token，NCCL传递中间隐藏状态)
```

### 关键步骤说明
1. **请求接收**: Stage0的DECODE进程接收用户请求
2. **预填充计算**: Stage0的PREFILL进程执行预填充计算
3. **跨stage传输**: 通过NCCL将中间结果传递给下一个stage
4. **流水线处理**: 每个stage处理模型的不同层
5. **结果聚合**: 最终结果通过NCCL返回给Stage0

## 🛠️ 实现细节

### 1. 调度器类
- **SemiPDDecodeScheduler**: DECODE进程调度器，主进程
- **SemiPDPrefillScheduler**: PREFILL进程调度器，辅助进程
- **SemiPDScheduler**: 基础调度器类

### 2. 端口管理
- **PPStagePortManager**: 管理所有PP stage的端口配置
- **create_pp_stage_port_args**: 为特定PP stage创建端口参数

### 3. 环境变量
```bash
SGLANG_ENABLE_SEMI_PD=1          # 启用Semi-PD模式
SGLANG_PP_RANK=<pp_rank>         # 设置PP rank
SGLANG_GPU_ID=<gpu_id>           # 设置GPU ID
```

## 🔍 调试和监控

### 日志标识
- **PP Stage标识**: `[DECODE-PP0]`, `[PREFILL-PP1]`
- **进程角色标识**: `🎯 DECODE主进程`, `🔧 PREFILL辅助进程`
- **通信标识**: `🔗 PP stage间通信`, `📡 同Stage通信`

### 关键日志点
1. **启动阶段**: 进程启动、IPC连接、NCCL初始化
2. **请求处理**: 请求接收、资源分配、计算执行
3. **通信阶段**: IPC通信、NCCL传输、结果返回

## ⚠️ 注意事项

### 1. 资源管理
- 确保不同PP stage使用不同GPU
- 正确配置NCCL通信组
- 避免端口冲突

### 2. 性能优化
- 合理设置batch size和chunk size
- 优化NCCL通信频率
- 监控GPU显存使用

### 3. 错误处理
- 处理IPC连接失败
- 处理NCCL通信错误
- 实现优雅降级机制

## 🧪 测试和验证

### 测试场景
1. **单PP stage测试**: 验证IPC通信和权重共享
2. **多PP stage测试**: 验证NCCL通信和流水线处理
3. **错误恢复测试**: 验证通信失败时的恢复机制

### 性能指标
- **吞吐量**: 每秒处理的请求数
- **延迟**: 请求处理时间
- **显存使用**: GPU显存占用情况
- **通信开销**: NCCL通信时间占比

## 📚 相关文件

- `semi_pd_scheduler.py`: 主调度器入口
- `semi_pd_decode_scheduler.py`: DECODE进程调度器
- `semi_pd_prefill_scheduler.py`: PREFILL进程调度器
- `semi_pd_pp_port_config.py`: 端口配置管理
- `README_SEMI_PD_PP.md`: 本文档

## 🔮 未来扩展

1. **动态PP stage**: 支持运行时动态调整PP stage数量
2. **负载均衡**: 实现智能的请求分配策略
3. **容错机制**: 增强错误恢复和容错能力
4. **性能优化**: 进一步优化通信和计算性能

## 🔍 原生sglang分布式启动机制分析

### 1. **启动流程**

原生sglang的启动流程是这样的：

```
主进程启动 → 解析命令行参数 → 启动所有PP stage进程 → 每个进程自动初始化分布式 → 所有组件就绪
```

### 2. **分布式初始化时机**

**关键发现**：原生sglang的分布式初始化是在`ModelRunner.__init__()`中进行的，而不是在进程启动后！

```python
# 在 TpModelWorker.__init__() 中
self.model_runner = ModelRunner(
    model_path=server_args.model_path,
    # ... 其他参数
)

# 在 ModelRunner.__init__() 中
min_per_gpu_memory = self.init_torch_distributed()  # 这里初始化分布式！
```

### 3. **分布式初始化参数**

从代码可以看到，原生sglang的分布式初始化参数是：

```python
init_distributed_environment(
    backend=backend,
    world_size=self.tp_size * self.pp_size,  # 总进程数 = TP × PP
    rank=self.tp_size * self.pp_rank + self.tp_rank,  # 全局rank计算
    local_rank=self.gpu_id,
    distributed_init_method=dist_init_method,
    timeout=self.server_args.dist_timeout,  # 超时设置
)
```

### 4. **关键差异分析**

**原生sglang vs 我们的Semi-PD**：

| 方面 | 原生sglang | 我们的Semi-PD |
|------|------------|---------------|
| **分布式初始化时机** | 在ModelRunner初始化时 | 在进程启动后 |
| **进程启动顺序** | 先启动所有进程，再初始化分布式 | 先启动PP0，等待就绪，再启动PP1 |
| **分布式超时** | 使用`server_args.dist_timeout` | 没有设置超时 |
| **进程同步** | 通过torch.distributed自动同步 | 通过pipe手动同步 |

### 5. **问题根源**

我们的问题在于：

1. **时机不对**：我们在进程启动后等待就绪，但原生sglang是在ModelRunner初始化时就进行分布式初始化
2. **同步机制不同**：原生sglang依赖torch.distributed的自动同步，我们依赖手动pipe同步
3. **超时缺失**：我们没有设置分布式初始化超时，可能导致无限等待

### 6. **正确的修复方向**

应该参考原生sglang的方式：

1. **先启动所有进程**：让所有PP进程都启动
2. **让每个进程自己初始化分布式**：在ModelRunner初始化时自动进行
3. **设置合理的超时**：避免无限等待
4. **依赖torch.distributed的同步**：而不是手动pipe同步

这就是为什么我们的修复没有完全解决问题的原因！我们需要重新设计启动逻辑，让它更接近原生sglang的方式。

## 🔍 Semi-PD TP=2 分布式启动机制详解

### 1. **启动流程概览**

```
主进程 → DECODE Schedulers (TP0, TP1) → PREFILL Schedulers (TP0, TP1)
   ↓              ↓                                    ↓
启动控制      模型加载+KV Cache管理              推理服务
```

### 2. **TP=2时的具体启动时序**

#### **第一阶段：DECODE Schedulers启动**
```python
# 设置CUDA MPS为100% SM
os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(DECODE_ENGINE_SM_PERCENTILE)

for tp_rank in tp_rank_range:  # tp_rank = 0, 1
    d_proc = mp.Process(
        target=run_scheduler_process,
        args=(
            server_args, port_args, gpu_id, tp_rank, None,
            d_writer, p_ipc_info_queue, False, InstanceRole.DECODE
        )
    )
    d_proc.start()
```

**关键点：**
- 每个TP rank启动一个**DECODE进程**
- 使用**100%的GPU SM资源**
- `bypass_load_weight=False`：**完整加载模型权重**
- 初始化完整的KV Cache和推理环境

#### **第二阶段：等待DECODE就绪**
```python
for i, reader in enumerate(d_scheduler_pipe_readers):
    logger.info(f"Waiting for D instance {tp_rank_base + i} to be ready")
    data = reader.recv()
    assert data["status"] == "ready"
    scheduler_infos.append(data)
    server_args.max_total_tokens = data["max_total_num_tokens"]
```

**关键点：**
- 主进程等待所有DECODE实例就绪
- 获取`max_total_num_tokens`等关键信息
- 确保所有TP rank的配置一致

#### **第三阶段：PREFILL Schedulers启动**
```python
# 设置CUDA MPS为80% SM  
os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(PREFILL_ENGINE_SM_PERCENTILE)

for tp_rank in tp_rank_range:  # tp_rank = 0, 1
    p_proc = mp.Process(
        target=run_scheduler_process,
        args=(
            server_args, port_args, gpu_id, tp_rank, None,
            p_writer, p_ipc_info_queue, True, InstanceRole.PREFILL
        )
    )
    p_proc.start()
```

**关键点：**
- 每个TP rank启动一个**PREFILL进程**
- 使用**80%的GPU SM资源**
- `bypass_load_weight=True`：**跳过权重加载**
- 通过IPC队列获取DECODE实例的资源信息

### 3. **分布式初始化机制**

#### **NCCL端口分配**
```python
# 在 SemiPDPortArgs.init_new() 中
s_port = SemiPDPortArgs.get_nccl_port(server_args)  
p_port = SemiPDPortArgs.get_nccl_port(server_args) 
d_port = SemiPDPortArgs.get_nccl_port(server_args) 
```

**关键点：**
- 每个TP rank分配**独立的NCCL端口**
- 支持**多节点扩展**：`tp_size_per_node = tp_size // nnodes`
- 每个节点内的TP rank使用连续的GPU ID

#### **IPC队列同步机制**
```python
# 为每个TP rank创建独立的IPC队列
p_ipc_info_queues: List[mp.Queue] = [
    mp.Queue() for _ in range(tp_size_per_node)
]

# DECODE实例将资源信息放入队列
d_proc = mp.Process(
    target=run_scheduler_process,
    args=(..., p_ipc_info_queue, ...)  # 传递队列引用
)

# PREFILL实例从队列获取资源信息
p_proc = mp.Process(
    target=run_scheduler_process,
    args=(..., p_ipc_info_queue, True, ...)  # bypass_load_weight=True
)
```

**关键点：**
- 每个TP rank有**独立的IPC队列**
- DECODE实例将**模型权重和KV Cache信息**放入队列
- PREFILL实例通过`bypass_load_weight=True`从队列获取资源

### 4. **Torch Distributed初始化**

#### **在semi_pd_scheduler.py中的初始化**
```python
# 创建scheduler时会自动初始化torch distributed
if instance_role == InstanceRole.DECODE:
    scheduler = SemiPDDecodeScheduler(
        server_args, port_args, gpu_id, tp_rank, dp_rank, bypass_load_weight
    )
    ipc_info = scheduler.get_ipc_info()
    ipc_info_queue.put(ipc_info)
```

**关键点：**
- 每个DECODE实例在创建时自动初始化**torch.distributed**
- 使用分配的NCCL端口建立**进程间通信**
- 支持**tensor parallel**和**data parallel**的混合并行

### 5. **详细启动时序**

```
T0: 主进程启动
T1: 配置日志和环境变量
T2: 准备模型和tokenizer路径
T3: 分配SemiPDPortArgs (NCCL端口等)
T4: 创建IPC队列数组 (每个TP rank一个)
T5: 启动DECODE Schedulers (TP0, TP1) - 100% SM
T6: 等待所有DECODE实例就绪
T7: 获取max_total_num_tokens等配置
T8: 启动PREFILL Schedulers (TP0, TP1) - 80% SM
T9: 等待所有PREFILL实例就绪
T10: 启动detokenizer进程
T11: 启动tokenizer进程
T12: 所有组件就绪，开始服务请求
```

### 6. **为什么P和D能在同一GPU上运行？**

#### **CUDA MPS资源隔离**
- **DECODE实例**：100% SM资源，负责模型推理和KV Cache管理
- **PREFILL实例**：80% SM资源，负责请求预处理和token生成
- 通过**CUDA Multi-Process Service**实现SM级别的资源分配

#### **内存共享机制**
- **模型权重**：DECODE实例加载，PREFILL实例通过IPC共享
- **KV Cache**：DECODE实例作为唯一管理者
- **请求队列**：通过ZMQ IPC实现进程间通信

这种设计的**核心优势**是：
1. **资源隔离**：通过CUDA MPS实现精确的SM资源分配
2. **内存效率**：避免重复加载模型权重，节省显存
3. **并发执行**：P和D实例可以同时处理不同的请求
4. **扩展性**：支持任意TP size的分布式训练

## 🔍 SGLang原生Pipeline并行分布式初始化机制详解

### 1. **Pipeline并行架构概览**

```
PP Stage 0 (GPU 0) → PP Stage 1 (GPU 1) → ... → PP Stage N (GPU N)
     ↓                    ↓                        ↓
  模型层0-10          模型层11-20              模型层21-30
```

### 2. **分布式初始化时机**

#### **第一阶段：World Group初始化**
```python
def init_world_group(ranks: List[int], local_rank: int, backend: str) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="world",
    )
```

**关键点：**
- 在**torch.distributed.init_process_group()**之后立即调用
- 建立**全局进程组**，包含所有参与训练的进程
- 使用**gloo后端**进行CPU协调，**NCCL后端**进行GPU通信

#### **第二阶段：Model Parallel Groups初始化**
```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    # 验证world_size = tp_size * pp_size
    if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )
```

**关键点：**
- 在**模型加载之前**调用
- 确保`world_size = tp_size × pp_size`
- 同时初始化**tensor parallel**和**pipeline parallel**组

### 3. **端口号分配机制**

#### **NCCL端口分配**
```python
# 在GroupCoordinator.__init__中
for ranks in group_ranks:
    device_group = torch.distributed.new_group(
        ranks, backend=torch_distributed_backend  # NCCL后端
    )
    # CPU协调组使用gloo后端
    cpu_group = torch.distributed.new_group(ranks, backend="gloo")
```

**端口分配策略：**
- **NCCL端口**：由torch.distributed自动分配，通常从**29500**开始
- **ZMQ端口**：在`SemiPDPortArgs`中定义，支持自定义偏移
- **IPC名称**：每个PP stage有独立的IPC通信名称

#### **Semi-PD Pipeline并行端口配置**
```python
@dataclasses.dataclass
class SemiPDPortArgs:
    """Port arguments for Semi-PD (Semi-Prefill-Decode) disaggregation"""
    tokenizer_ipc_name: str
    s_scheduler_input_ipc_name: str
    p_scheduler_input_ipc_name: str
    d_scheduler_input_ipc_name: str
    detokenizer_ipc_name: str
    bridge_ipc_name: str
    rpc_ipc_name: str
    weight_share_ipc_name: str
    
    s_nccl_port: int
    p_nccl_port: int
    d_nccl_port: int
```

**端口分配示例（PP=2, TP=2）：**
```
PP0 TP0: s_nccl_port=29500, p_nccl_port=29501, d_nccl_port=29502
PP0 TP1: s_nccl_port=29503, p_nccl_port=29504, d_nccl_port=29505
PP1 TP0: s_nccl_port=29506, p_nccl_port=29507, d_nccl_port=29508
PP1 TP1: s_nccl_port=29509, p_nccl_port=29510, d_nccl_port=29511
```

### 4. **Pipeline并行组构建**

#### **Pipeline Parallel Groups构建**
```python
# Build the pipeline model-parallel groups.
num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
global _PP
assert _PP is None, "pipeline model parallel group is already initialized"
group_ranks = []
for i in range(num_pipeline_model_parallel_groups):
    ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
    group_ranks.append(ranks)

_PP = init_model_parallel_group(
    group_ranks,
    get_world_group().local_rank,
    backend,
    use_custom_allreduce=False,  # Pipeline parallel不需要custom allreduce
    group_name="pp",
)
```

**组构建逻辑：**
- **PP=2, TP=2**时，总进程数=4
- **Pipeline组0**：[rank0, rank2] (PP0的TP0和TP1)
- **Pipeline组1**：[rank1, rank3] (PP1的TP0和TP1)
- **Tensor组0**：[rank0, rank1] (TP0的PP0和PP1)
- **Tensor组1**：[rank2, rank3] (TP1的PP0和PP1)

### 5. **详细启动时序**

```
T0: 主进程启动
T1: 解析命令行参数 (--pipeline-parallel-size, --tensor-parallel-size)
T2: 调用torch.distributed.init_process_group()
T3: 初始化World Group (所有进程的全局组)
T4: 调用initialize_model_parallel(tp_size, pp_size)
T5: 构建Tensor Parallel Groups
T6: 构建Pipeline Parallel Groups
T7: 分配NCCL端口和IPC名称
T8: 启动各个PP stage的进程
T9: 每个PP stage加载对应的模型层
T10: 建立PP stage间的通信连接
T11: 所有组件就绪，开始服务请求
```

### 6. **Pipeline并行的关键特性**

#### **通信模式**
- **Forward Pass**：数据从PP0流向PP1，最后流向PPN
- **Backward Pass**：梯度从PPN流向PP1，最后流向PP0
- **Micro-batching**：支持micro-batch来隐藏通信延迟

#### **约束条件**
```python
if self.pp_size > 1:
    assert (
        self.disable_overlap_schedule
        and self.speculative_algorithm is None
        and not self.enable_mixed_chunk
    ), "Pipeline parallelism is not compatible with overlap schedule, speculative decoding, mixed chunked prefill."
```

**关键限制：**
- **不支持overlap schedule**
- **不支持speculative decoding**
- **不支持mixed chunked prefill**
- 这些限制是为了确保pipeline并行的正确性

### 7. **Semi-PD适配Pipeline并行的关键点**

#### **IPC队列扩展**
```python
# 需要为每个PP stage创建独立的IPC队列
ipc_queues = {}
for pp_rank in range(pp_size):
    for tp_rank in range(tp_size):
        ipc_queues[(pp_rank, tp_rank)] = mp.Queue()
```

#### **权重共享机制**
- 每个PP stage的DECODE进程加载对应的模型层
- 通过IPC在不同PP stage间共享权重信息
- 支持跨PP stage的零拷贝参数传递

## 🚀 Semi-PD Pipeline并行实现方案

### 1. **修正后的启动时序**

```
T0: 主进程启动
T1: 配置日志和环境变量
T2: 准备模型和tokenizer路径
T3: 分配Pipeline并行端口 (每个PP stage独立端口范围)
T4: 创建IPC队列矩阵 (每个(PP, TP)组合一个队列)
T5: 同时启动所有PP stage的DECODE进程
T6: 等待所有DECODE进程就绪 (设置超时)
T7: 同时启动所有PP stage的PREFILL进程
T8: 等待所有PREFILL进程就绪 (设置超时)
T9: 启动其他辅助进程
T10: 所有组件就绪，开始服务请求
```

### 2. **分布式初始化修复**

#### **在ModelRunner初始化时进行分布式初始化**
```python
class SemiPDModelRunner:
    def __init__(self, server_args, port_args, pp_rank, tp_rank, pp_size, tp_size):
        # 关键：在模型加载前初始化分布式
        self.init_torch_distributed(pp_rank, tp_rank, pp_size, tp_size)
        
        # 然后加载模型
        self.load_model()
    
    def init_torch_distributed(self, pp_rank, tp_rank, pp_size, tp_size):
        # 计算全局rank
        world_size = pp_size * tp_size
        global_rank = pp_rank * tp_size + tp_rank
        
        # 设置环境变量
        os.environ['RANK'] = str(global_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(tp_rank)
        os.environ['PP_RANK'] = str(pp_rank)
        os.environ['TP_RANK'] = str(tp_rank)
        
        # 初始化torch.distributed
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=global_rank,
            init_method='env://',
            timeout=timedelta(seconds=300)  # 设置超时
        )
        
        # 初始化SGLang的并行组
        from sglang.srt.distributed.parallel_state import initialize_model_parallel
        initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size
        )
```

### 3. **IPC队列矩阵设计**

#### **为每个(PP, TP)组合创建独立队列**
```python
class SemiPDPipelinePortManager:
    def __init__(self, pp_size: int, tp_size: int):
        self.pp_size = pp_size
        self.tp_size = tp_size
        
        # 为每个(PP, TP)组合创建独立的IPC队列
        self.ipc_queues = {}
        for pp_rank in range(pp_size):
            for tp_rank in range(tp_size):
                self.ipc_queues[(pp_rank, tp_rank)] = {
                    'decode_to_prefill': mp.Queue(),
                    'prefill_to_decode': mp.Queue(),
                    'weight_share': mp.Queue(),
                    'cross_stage': mp.Queue()
                }
        
        # 为每个PP stage分配端口范围
        self.pp_stage_ports = {}
        for pp_rank in range(pp_size):
            base_port = 40000 + pp_rank * 100
            self.pp_stage_ports[pp_rank] = {
                'decode': base_port + 0,
                'prefill': base_port + 1,
                'scheduler': base_port + 2,
                'detokenizer': base_port + 3,
                'nccl': base_port + 100
            }
```

### 4. **Pipeline并行通信实现**

#### **跨PP stage的通信机制**
```python
class SemiPDPipelineScheduler:
    def __init__(self, pp_rank, tp_rank, pp_size, tp_size):
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        self.pp_size = pp_size
        self.tp_size = tp_size
        
        # 获取下一个PP stage的rank
        self.next_pp_rank = (pp_rank + 1) % pp_size
        self.prev_pp_rank = (pp_rank - 1) % pp_size
    
    def forward_pass(self, input_tensor):
        """Pipeline并行的前向传播"""
        if self.pp_rank == 0:
            # 第一个stage：处理输入
            output = self.process_input(input_tensor)
        else:
            # 中间stage：接收上一个stage的输出
            output = self.receive_from_prev_stage()
            output = self.process_intermediate(output)
        
        if self.pp_rank < self.pp_size - 1:
            # 不是最后一个stage：发送给下一个stage
            self.send_to_next_stage(output)
        
        return output
    
    def backward_pass(self, grad_tensor):
        """Pipeline并行的反向传播"""
        if self.pp_rank == self.pp_size - 1:
            # 最后一个stage：处理梯度
            grad = self.process_gradient(grad_tensor)
        else:
            # 中间stage：接收下一个stage的梯度
            grad = self.receive_from_next_stage()
            grad = self.process_intermediate_gradient(grad)
        
        if self.pp_rank > 0:
            # 不是第一个stage：发送给上一个stage
            self.send_to_prev_stage(grad)
        
        return grad
```

### 5. **启动脚本实现**

#### **Pipeline并行的启动脚本**
```python
# launch_semipd_pipeline.py
def main():
    pp_size = 2
    tp_size = 2
    
    # 分配端口
    port_manager = SemiPDPipelinePortManager(pp_size, tp_size)
    
    # 启动所有PP stage的进程
    launch_all_pp_stages(pp_size, tp_size, port_manager)
    
    # 等待所有进程就绪
    wait_for_all_processes_ready(pp_size, tp_size)
    
    print("🎉 Semi-PD Pipeline Parallel 启动完成！")

def launch_all_pp_stages(pp_size: int, tp_size: int, port_manager):
    """启动所有PP stage的进程"""
    # 第一阶段：启动所有DECODE进程
    decode_processes = {}
    for pp_rank in range(pp_size):
        for tp_rank in range(tp_size):
            proc = mp.Process(
                target=run_decode_process,
                args=(pp_rank, tp_rank, pp_size, tp_size, port_manager)
            )
            proc.start()
            decode_processes[(pp_rank, tp_rank)] = proc
    
    # 等待所有DECODE进程就绪
    wait_for_decode_processes_ready(decode_processes, timeout=300)
    
    # 第二阶段：启动所有PREFILL进程
    prefill_processes = {}
    for pp_rank in range(pp_size):
        for tp_rank in range(tp_size):
            proc = mp.Process(
                target=run_prefill_process,
                args=(pp_rank, tp_rank, pp_size, tp_size, port_manager)
            )
            proc.start()
            prefill_processes[(pp_rank, tp_rank)] = proc
    
    # 等待所有PREFILL进程就绪
    wait_for_prefill_processes_ready(prefill_processes, timeout=300)
```

## 🎯 总结

通过以上分析和修复方案，Semi-PD Pipeline并行的关键改进点包括：

1. **启动时序修正**：从串行启动改为并行启动所有PP stage
2. **分布式初始化时机**：在ModelRunner初始化时进行，而不是进程启动后
3. **超时机制**：设置合理的分布式初始化超时，避免无限等待
4. **IPC队列扩展**：为每个(PP, TP)组合创建独立的通信队列
5. **通信机制**：实现跨PP stage的NCCL通信和同PP stage的IPC通信

这样修改后，Semi-PD就能正确支持Pipeline并行，实现与SGLang原生Pipeline并行相同的功能和性能！

