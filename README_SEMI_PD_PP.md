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
- **启动顺序**: DECODE先启动 → PREFILL后启动

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