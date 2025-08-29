# Semi-PD TP=2 分布式初始化详细流程

## 概述

本文档详细描述了SGLang在Semi-PD模式下使用TP=2（Tensor Parallel=2）时的分布式初始化流程。通过添加的详细日志，我们可以跟踪每个进程的启动时序、端口分配、通信机制等关键信息。

## 架构概览

### 进程架构
```
┌─────────────────────────────────────────────────────────────────┐
│                    Semi-PD TP=2 架构                            │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   PP Stage 0    │    │   PP Stage 1    │                    │
│  │                 │    │                 │                    │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │                    │
│  │ │ DECODE TP0  │ │    │ │ DECODE TP0  │ │                    │
│  │ │ (主进程)    │ │    │ │ (主进程)    │ │                    │
│  │ │ GPU 0       │ │    │ │ GPU 1       │ │                    │
│  │ └─────────────┘ │    │ └─────────────┘ │                    │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │                    │
│  │ │ PREFILL TP0 │ │    │ │ PREFILL TP0 │ │                    │
│  │ │ (辅助进程)  │ │    │ │ (辅助进程)  │ │                    │
│  │ │ GPU 0       │ │    │ │ GPU 1       │ │                    │
│  │ └─────────────┘ │    │ └─────────────┘ │                    │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │                    │
│  │ │ DECODE TP1  │ │    │ │ DECODE TP1  │ │                    │
│  │ │ (主进程)    │ │    │ │ (主进程)    │ │                    │
│  │ │ GPU 0       │ │    │ │ GPU 1       │ │                    │
│  │ └─────────────┘ │    │ └─────────────┘ │                    │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │                    │
│  │ │ PREFILL TP1 │ │    │ │ PREFILL TP1 │ │                    │
│  │ │ (辅助进程)  │ │    │ │ (辅助进程)  │ │                    │
│  │ │ GPU 0       │ │    │ │ GPU 1       │ │                    │
│  │ └─────────────┘ │    │ └─────────────┘ │                    │
│  └─────────────────┘    └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### 配置参数
- **TP大小**: 2 (Tensor Parallel)
- **PP大小**: 2 (Pipeline Parallel)  
- **总进程数**: 4 (2 TP × 2 PP)
- **每个PP Stage**: 2个TP ranks (TP0, TP1)
- **每个TP rank**: 2个进程 (DECODE主进程, PREFILL辅助进程)

## 详细初始化流程

### 1. 进程启动时序

#### 1.1 主进程启动
```
[2025-XX-XX XX:XX:XX] INFO: Started server process [PID]
```

#### 1.2 调度器进程并行启动
每个PP Stage启动4个进程（2个TP ranks × 2个进程类型）：

**PP Stage 0 (GPU 0)**:
- DECODE TP0 (主进程)
- PREFILL TP0 (辅助进程)  
- DECODE TP1 (主进程)
- PREFILL TP1 (辅助进程)

**PP Stage 1 (GPU 1)**:
- DECODE TP0 (主进程)
- PREFILL TP0 (辅助进程)
- DECODE TP1 (主进程) 
- PREFILL TP1 (辅助进程)

### 2. 详细日志跟踪

#### 2.1 进程启动日志
```
🚀 [SEMI_PD_TP2] ========== Semi-PD TP=2 进程启动 ==========
🚀 [SEMI_PD_TP2] 进程信息: instance_role=DECODE, pp_rank=0, tp_rank=0, gpu_id=0
🚀 [SEMI_PD_TP2] 配置信息: tp_size=2, pp_size=2, dp_size=1
🚀 [SEMI_PD_TP2] 环境变量设置: SGLANG_ENABLE_SEMI_PD=1, SGLANG_PP_RANK=0, SGLANG_GPU_ID=0
🚀 [SEMI_PD_TP2] 进程标题设置: sglang::semi_pd_scheduler_DECODE_PP0_TP0
🚀 [SEMI_PD_TP2] 父进程PID: [父进程PID]
```

#### 2.2 调度器初始化日志
```
🔧 [SEMI_PD_TP2] PP0 TP0: SemiPDScheduler初始化开始
🔧 [SEMI_PD_TP2] PP0 TP0: 环境变量设置完成
🔧 [SEMI_PD_TP2] PP0 TP0: SGLANG_ENABLE_SEMI_PD=1, SGLANG_PP_RANK=0, SGLANG_GPU_ID=0
🔧 [SEMI_PD_TP2] PP0 TP0: 调用父类Scheduler构造函数...
🔧 [SEMI_PD_TP2] PP0 TP0: Semi-PD PP模式: PP stage 0 using GPU 0
🎯 [SEMI_PD_TP2] PP0 TP0: DECODE进程作为主进程，负责请求协调
🎯 [SEMI_PD_TP2] PP0 TP0: 职责包括: 请求接收、响应返回、整体协调、KV Cache管理
✅ [SEMI_PD_TP2] PP0 TP0: SemiPDScheduler初始化完成
```

#### 2.3 DECODE调度器初始化日志
```
🎯 [SEMI_PD_TP2] PP0 TP0: SemiPDDecodeScheduler初始化开始
🎯 [SEMI_PD_TP2] PP0 TP0: 保存pp_rank=0
🎯 [SEMI_PD_TP2] PP0 TP0: 设置请求分发器映射...
🎯 [SEMI_PD_TP2] PP0 TP0: 初始化预填充批次队列
🎯 [SEMI_PD_TP2] PP0 TP0: 注意力TP rank 0，初始化IPC通信socket...
🎯 [SEMI_PD_TP2] PP0 TP0: 端口参数类型确认: SemiPDPortArgs
🎯 [SEMI_PD_TP2] PP0 TP0: 创建bridge socket: [bridge_ipc_name]
🎯 [SEMI_PD_TP2] PP0 TP0: 创建send_to_p_instance socket: [p_scheduler_input_ipc_name]
🔗 [SEMI_PD_TP2] PP0 TP0: 将使用SGLang原生NCCL与下一个stage的DECODE进程通信
🔗 [SEMI_PD_TP2] PP0 TP0: 下一个stage端口: [next_stage_decode_port]
✅ [SEMI_PD_TP2] PP0 TP0: IPC通信socket初始化完成
✅ [SEMI_PD_TP2] PP0 TP0: SemiPDDecodeScheduler初始化完成
```

#### 2.4 PREFILL调度器初始化日志
```
🔧 [SEMI_PD_TP2] PP0 TP0: SemiPDPrefillScheduler初始化开始
🔧 [SEMI_PD_TP2] PP0 TP0: 保存pp_rank=0
🔧 [SEMI_PD_TP2] PP0 TP0: enable_overlap=False
🔧 [SEMI_PD_TP2] PP0 TP0: 使用GPU设备: cuda:0
🔧 [SEMI_PD_TP2] PP0 TP0: 注意力TP rank 0，初始化IPC通信socket...
🔧 [SEMI_PD_TP2] PP0 TP0: 创建send_to_d_instance socket: [p_scheduler_input_ipc_name]
🔧 [SEMI_PD_TP2] PP0 TP0: 创建bridge socket: [bridge_ipc_name]
🔗 [SEMI_PD_TP2] PP0 TP0: 将使用SGLang原生NCCL与下一个stage的PREFILL进程通信
🔗 [SEMI_PD_TP2] PP0 TP0: 下一个stage端口: [next_stage_prefill_port]
✅ [SEMI_PD_TP2] PP0 TP0: IPC通信socket初始化完成
✅ [SEMI_PD_TP2] PP0 TP0: SemiPDPrefillScheduler初始化完成
```

### 3. 分布式初始化流程

#### 3.1 PyTorch分布式初始化
```
[TORCH_DIST] ========== PyTorch分布式初始化开始 ==========
[TORCH_DIST] 设备=cuda, gpu_id=0, tp_rank=0, pp_rank=0
[TORCH_DIST] tp_size=2, pp_size=2, dp_size=1
[TORCH_DIST] 总进程数: 4, 当前rank: 0
[TORCH_DIST] 成功设置设备: cuda:0
[TORCH_DIST] 选择后端: nccl
[TORCH_DIST] 初始化前可用GPU内存: 43.94 GB
[TORCH_DIST] 禁用P2P访问检查
[TORCH_DIST] 分布式初始化方法: tcp://127.0.0.1:30828
[TORCH_DIST] NCCL端口: 30828
[TORCH_DIST] Semi-PD模式: 1
[TORCH_DIST] 自定义allreduce: True
[TORCH_DIST] MSCclpp allreduce: False
```

#### 3.2 分布式环境初始化
```
[DIST_INIT] ========== 分布式环境初始化开始 ==========
[DIST_INIT] world_size=4, rank=0, local_rank=0
[DIST_INIT] distributed_init_method=tcp://127.0.0.1:30828, backend=nccl
[DIST_INIT] timeout=1800
[DIST_INIT] 当前进程PID=1550760
[DIST_INIT] 环境变量CUDA_VISIBLE_DEVICES=N/A
[DIST_INIT] 环境变量SGLANG_ENABLE_SEMI_PD=1
[DIST_INIT] 环境变量SGLANG_PP_RANK=0
[DIST_INIT] 环境变量SGLANG_GPU_ID=0
[DIST_INIT] PyTorch分布式环境尚未初始化，开始初始化...
[DIST_INIT] 设置超时时间: 0:30:00
[DIST_INIT] ========== 调用torch.distributed.init_process_group ==========
[DIST_INIT] 参数: backend=nccl, init_method=tcp://127.0.0.1:30828, world_size=4, rank=0
[DIST_INIT] ✅ PyTorch分布式环境初始化完成
[DIST_INIT] torch.distributed.is_initialized() = True
[DIST_INIT] torch.distributed.get_world_size() = 4
[DIST_INIT] torch.distributed.get_rank() = 0
[DIST_INIT] 使用传入的local_rank=0
[DIST_INIT] ========== 创建WORLD组 ==========
[DIST_INIT] WORLD组ranks=[0, 1, 2, 3]
[DIST_INIT] ✅ WORLD组创建完成: world_size=4, rank=0, local_rank=0
[DIST_INIT] WORLD组设备: cuda:0
[DIST_INIT] ========== 分布式环境初始化完成 ==========
```

#### 3.3 模型并行初始化
```
[MODEL_PARALLEL] ========== 模型并行初始化开始 ==========
[MODEL_PARALLEL] tensor_model_parallel_size=2
[MODEL_PARALLEL] pipeline_model_parallel_size=2
[MODEL_PARALLEL] Semi-PD模式: 1
[MODEL_PARALLEL] world_size=4, backend=nccl
[MODEL_PARALLEL] 当前rank=0
[MODEL_PARALLEL] ========== 创建TP组 ==========
[MODEL_PARALLEL] 将创建2个TP组，每组2个进程
[MODEL_PARALLEL] 创建TP组 0: ranks=[0, 1]
[MODEL_PARALLEL] 创建TP组 1: ranks=[2, 3]
[MODEL_PARALLEL] 初始化TP组，使用message_queue_broadcaster=true
[MODEL_PARALLEL] ✅ TP组初始化完成: world_size=2, rank_in_group=0
[MODEL_PARALLEL] ========== 创建PP组 ==========
[MODEL_PARALLEL] 将创建2个PP组，每组2个进程
[MODEL_PARALLEL] 创建PP组 0: ranks=[0, 1]
[MODEL_PARALLEL] 创建PP组 1: ranks=[2, 3]
[MODEL_PARALLEL] ✅ PP组初始化完成: world_size=2, rank_in_group=0
[MODEL_PARALLEL] ========== 模型并行初始化完成 ==========
```

#### 3.4 组协调器初始化
```
[GROUP_COORD] ========== 组协调器初始化开始 ==========
[GROUP_COORD] 组名称: world:0
[GROUP_COORD] group_ranks=[[0, 1, 2, 3]], local_rank=0
[GROUP_COORD] backend=nccl, group_name=world
[GROUP_COORD] Semi-PD模式: 1
[GROUP_COORD] 当前进程rank=0, local_rank=0
[GROUP_COORD] 为ranks=[0, 1, 2, 3]创建设备组和CPU组
[GROUP_COORD] 当前进程属于组ranks=[0, 1, 2, 3], rank_in_group=0, world_size=4
[GROUP_COORD] Semi-PD PP模式: PP stage 0 使用GPU 0
[GROUP_COORD] 设备分配: cuda:0
[GROUP_COORD] ========== 通信配置 ==========
[GROUP_COORD] use_pynccl=True, use_pymscclpp=False, use_custom_allreduce=True
[GROUP_COORD] use_hpu_communicator=True, use_xpu_communicator=True, use_npu_communicator=True
[GROUP_COORD] use_message_queue_broadcaster=False
[GROUP_COORD] 初始化PyNcclCommunicator
[GROUP_COORD] 初始化HpuCommunicator
[GROUP_COORD] 初始化XpuCommunicator
[GROUP_COORD] 初始化NpuCommunicator
[GROUP_COORD] ========== 组协调器 world:0 初始化完成 ==========
```

### 4. 端口分配

#### 4.1 基础端口分配
每个PP Stage分配独立的端口范围：

**PP Stage 0 (GPU 0)**:
- 40000: decode_port (主进程)
- 40001: prefill_port (辅助进程)
- 40002: scheduler_port
- 40003: detokenizer_port
- 40100: nccl_port (跨GPU通信)

**PP Stage 1 (GPU 1)**:
- 41000: decode_port (主进程)
- 41001: prefill_port (辅助进程)
- 41002: scheduler_port
- 41003: detokenizer_port
- 41100: nccl_port (跨GPU通信)

#### 4.2 NCCL端口分配
```
[TORCH_DIST] NCCL端口: 30828
[DIST_INIT] distributed_init_method=tcp://127.0.0.1:30828
```

### 5. 通信机制

#### 5.1 同PP Stage内通信
- **DECODE ↔ PREFILL**: 通过ZMQ IPC socket
- **Bridge Socket**: 用于进程间协调
- **Send Socket**: 用于数据传输

#### 5.2 跨PP Stage通信
- **DECODE ↔ DECODE**: 使用SGLang原生NCCL
- **PREFILL ↔ PREFILL**: 使用SGLang原生NCCL
- **TP组内通信**: 使用NCCL进行tensor并行

#### 5.3 分布式组通信
- **WORLD组**: 所有进程的全局组
- **TP组**: 同PP stage内的tensor并行组
- **PP组**: 跨PP stage的pipeline并行组

### 6. 启动时序

#### 6.1 进程启动顺序
1. **主进程启动** (PID: 1550592)
2. **PP Stage 0 DECODE TP0** (PID: 1550760) - 主进程
3. **PP Stage 0 PREFILL TP0** (PID: 1550761) - 辅助进程
4. **PP Stage 0 DECODE TP1** (PID: 1550762) - 主进程
5. **PP Stage 0 PREFILL TP1** (PID: 1550763) - 辅助进程
6. **PP Stage 1 DECODE TP0** (PID: 1550764) - 主进程
7. **PP Stage 1 PREFILL TP0** (PID: 1550765) - 辅助进程
8. **PP Stage 1 DECODE TP1** (PID: 1550766) - 主进程
9. **PP Stage 1 PREFILL TP1** (PID: 1550767) - 辅助进程

#### 6.2 初始化时序
1. **环境变量设置**: SGLANG_ENABLE_SEMI_PD=1, SGLANG_PP_RANK, SGLANG_GPU_ID
2. **PyTorch分布式初始化**: NCCL后端，TCP初始化
3. **WORLD组创建**: 全局进程组
4. **TP组创建**: tensor并行组
5. **PP组创建**: pipeline并行组
6. **IPC Socket创建**: 进程间通信
7. **调度器初始化**: DECODE/PREFILL调度器
8. **模型加载**: 权重加载和CUDA图捕获

### 7. 关键特性

#### 7.1 Semi-PD特性
- **DECODE主进程**: 负责请求接收、响应返回、整体协调、KV Cache管理
- **PREFILL辅助进程**: 负责预填充计算、共享主进程权重、配合主进程
- **IPC权重共享**: 避免重复加载模型权重，节省显存
- **异步执行**: DECODE和PREFILL进程异步工作

#### 7.2 TP=2特性
- **Tensor并行**: 模型层在2个TP ranks间分割
- **注意力并行**: 注意力机制在TP组内并行计算
- **通信优化**: 使用NCCL进行高效的GPU间通信

#### 7.3 PP=2特性
- **Pipeline并行**: 模型在2个PP stages间分割
- **Stage间通信**: 使用NCCL进行跨stage通信
- **流水线调度**: 实现请求的流水线处理

### 8. 监控和调试

#### 8.1 关键日志标识
- `[SEMI_PD_TP2]`: Semi-PD TP=2相关日志
- `[TORCH_DIST]`: PyTorch分布式初始化日志
- `[DIST_INIT]`: 分布式环境初始化日志
- `[MODEL_PARALLEL]`: 模型并行初始化日志
- `[GROUP_COORD]`: 组协调器初始化日志

#### 8.2 关键检查点
1. **进程启动**: 检查所有8个进程是否正常启动
2. **端口分配**: 检查端口是否被正确分配和使用
3. **分布式初始化**: 检查NCCL通信是否正常建立
4. **组创建**: 检查WORLD、TP、PP组是否正确创建
5. **IPC通信**: 检查进程间通信是否正常
6. **模型加载**: 检查权重加载和CUDA图捕获是否成功

### 9. 故障排除

#### 9.1 常见问题
1. **端口冲突**: 检查端口是否被其他进程占用
2. **NCCL初始化失败**: 检查网络配置和GPU可见性
3. **IPC通信失败**: 检查ZMQ socket创建和连接
4. **内存不足**: 检查GPU内存是否足够加载模型
5. **进程启动失败**: 检查环境变量和依赖库

#### 9.2 调试方法
1. **查看详细日志**: 使用添加的详细日志跟踪初始化流程
2. **检查进程状态**: 使用`ps aux | grep sglang`查看进程状态
3. **检查端口使用**: 使用`netstat -tulpn | grep [port]`检查端口使用
4. **检查GPU状态**: 使用`nvidia-smi`检查GPU使用情况
5. **检查网络连接**: 使用`ping`和`telnet`检查网络连通性

## 总结

通过添加的详细日志，我们可以完整跟踪Semi-PD TP=2模式下的分布式初始化流程。这个流程确保了：

1. **正确的进程启动**: 8个进程按正确顺序启动
2. **正确的端口分配**: 每个进程使用独立的端口范围
3. **正确的通信建立**: NCCL和IPC通信正常建立
4. **正确的组创建**: WORLD、TP、PP组正确创建
5. **正确的设备分配**: 每个进程使用正确的GPU设备
6. **正确的角色分配**: DECODE主进程和PREFILL辅助进程职责明确

这些日志为调试和监控Semi-PD TP=2模式提供了强有力的工具。 