# 🔧 Semi-PD Pipeline并行架构修复总结

## 🎯 修复目标

将Semi-PD从错误的"分离式架构"修复为正确的"统一分布式组架构"，对齐SGLang原生Pipeline并行。

## ❌ 修复前的错误架构

### 错误1：分离式分布式组
```python
# 错误：每个PP stage独立初始化
if self.server_args.enable_semi_pd and self.pp_size > 1:
    world_size = self.tp_size  # 只考虑当前PP stage
    rank = self.tp_rank        # 在当前stage内的rank
    
    initialize_model_parallel(
        tensor_model_parallel_size=self.tp_size,
        pipeline_model_parallel_size=1,  # 错误：设为1
    )
```

### 错误2：错误的进程架构
```
GPU 0: PP0-DECODE(主), PP0-PREFILL(辅)  (完整模型 + 预填充)
GPU 1: PP1-DECODE(主), PP1-PREFILL(辅)  (完整模型 + 预填充)
```

**问题**：每个GPU都有完整模型，无法实现真正的Pipeline并行！

## ✅ 修复后的正确架构

### 修复1：统一分布式组
```python
# 正确：所有PP stage共享一个分布式组
if self.server_args.enable_semi_pd and self.pp_size > 1:
    # Semi-PD模式：所有PP stage共享一个分布式组，这样才能实现跨stage的NCCL通信
    world_size = self.tp_size * self.pp_size  # 例如：2 * 2 = 4
    rank = self.tp_size * self.pp_rank + self.tp_rank  # 例如：2 * 1 + 0 = 2
    
    initialize_model_parallel(
        tensor_model_parallel_size=self.tp_size,      # 2
        pipeline_model_parallel_size=self.pp_size,    # 2 (使用完整的PP size)
    )
```

### 修复2：正确的进程架构
```
GPU 0: PP0-TP0, PP0-TP1  (处理前几层)
GPU 1: PP1-TP0, PP1-TP1  (处理后几层)
```

**优势**：真正的Pipeline并行，模型按层分割到不同GPU！

### 修复3：正确的PP组创建策略
```python
# Semi-PD模式：使用跨步分组策略，对齐SGLang原生
if os.environ.get('SGLANG_ENABLE_SEMI_PD', 'false').lower() in ('1', 'true'):
    # 跨步分组：PP组0=[0,2,4,6], PP组1=[1,3,5,7]
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
else:
    # 标准模式：连续分组
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i * pipeline_model_parallel_size, (i + 1) * pipeline_model_parallel_size))
        group_ranks.append(ranks)
```

## 🔄 通信流程

### 修复后的正确流程
```
1. 请求进入 → PP0-TP0 (接收请求，处理前几层)
2. PP0-TP0 → PP0-TP1 (TP并行，同一GPU内)
3. PP0-TP1 → PP1-TP0 (PP并行，跨GPU，通过NCCL)
4. PP1-TP0 → PP1-TP1 (TP并行，同一GPU内)
5. PP1-TP1 → PP0-TP0 (返回结果，跨GPU，通过NCCL)
```

### Semi-PD特色保留
- **权重共享**：DECODE进程加载权重，PREFILL进程通过IPC共享
- **异步协调**：通过主进程(DECODE)协调整个推理流程
- **混合通信**：GPU内用IPC，GPU间用NCCL

## 📊 配置示例

### 2 GPU, TP=2, PP=2 配置
```
world_size = 4 (2 * 2)
ranks = [0, 1, 2, 3]

TP组：
- TP组0: [0, 1] (GPU 0)
- TP组1: [2, 3] (GPU 1)

PP组：
- PP组0: [0, 2] (跨GPU，处理前几层)
- PP组1: [1, 3] (跨GPU，处理后几层)

进程映射：
- GPU 0: rank 0 (PP0-TP0), rank 1 (PP0-TP1)
- GPU 1: rank 2 (PP1-TP0), rank 3 (PP1-TP1)
```

## 🚀 关键修复点

1. **分布式初始化**：统一world_size和rank计算
2. **模型并行**：使用完整的pipeline_model_parallel_size
3. **PP组创建**：使用跨步分组策略
4. **设备分配**：通过环境变量正确分配GPU
5. **广播逻辑**：使用全局rank进行跨stage通信

## ✅ 验证方法

1. 检查日志中的分布式组信息
2. 确认world_size = tp_size * pp_size
3. 验证PP组使用跨步分组策略
4. 测试跨GPU的NCCL通信是否正常

## 🎉 修复效果

- **对齐SGLang原生**：完全兼容SGLang的Pipeline并行机制
- **保留Semi-PD特色**：权重共享、异步协调等优势
- **正确Pipeline并行**：模型按层分割，真正的流水线执行
- **稳定跨stage通信**：通过NCCL实现高效的GPU间通信

## 🚨 最新修复：PP组创建逻辑问题

### **问题描述**
在TP=2, PP=1的配置下，Semi-PD错误地创建了2个PP组，每个组只有1个进程，导致无法进行PP并行通信而卡住。

### **错误日志**
```
[MODEL_PARALLEL] 将创建2个PP组，每组1个进程
[MODEL_PARALLEL] 🔧 Semi-PD模式：使用跨步分组策略创建PP组
[MODEL_PARALLEL] 创建PP组 0: ranks=[0] (跨步分组)
[MODEL_PARALLEL] 创建PP组 1: ranks=[1] (跨步分组)
```

### **修复内容**

#### **1. parallel_state.py - PP组创建逻辑**
- **修复前**：当`pp_size=1`时，错误地创建了`world_size // pp_size = 2`个PP组
- **修复后**：当`pp_size=1`时，只创建1个PP组包含所有进程

```python
# 修复前：错误的逻辑
if os.environ.get('SGLANG_ENABLE_SEMI_PD', 'false').lower() in ('1', 'true'):
    # 强制使用跨步分组，导致单PP stage时创建多个组
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)

# 修复后：正确的逻辑
if pipeline_model_parallel_size == 1:
    # 单PP stage模式：所有进程属于同一个PP组
    num_pipeline_model_parallel_groups = 1
    ranks = list(range(world_size))
    group_ranks.append(ranks)
else:
    # 多PP stage模式：使用分组策略
    if os.environ.get('SGLANG_ENABLE_SEMI_PD', 'false').lower() in ('1', 'true'):
        # Semi-PD模式：使用跨步分组策略
        for i in range(num_pipeline_model_parallel_groups):
            ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
            group_ranks.append(ranks)
```

#### **2. model_runner.py - 分布式初始化逻辑**
- **修复前**：单PP模式下仍然使用`world_size = tp_size * pp_size`
- **修复后**：单PP模式下使用`world_size = tp_size`，只进行TP并行

```python
# 修复前：错误的初始化
if self.server_args.enable_semi_pd and self.pp_size > 1:
    world_size = self.tp_size * self.pp_size
    rank = self.tp_size * self.pp_rank + self.tp_rank
else:
    # 标准模式：仍然使用PP size
    world_size = self.tp_size * self.pp_size
    rank = self.tp_size * self.pp_rank + self.tp_rank

# 修复后：正确的初始化
if self.server_args.enable_semi_pd and self.pp_size > 1:
    # Semi-PD多PP模式：使用完整分布式组
    world_size = self.tp_size * self.pp_size
    rank = self.tp_size * self.pp_rank + self.tp_rank
else:
    # 单PP模式：只使用TP并行
    world_size = self.tp_size
    rank = self.tp_rank
```

### **修复效果**
- **单PP模式**：正确创建1个PP组，所有进程可以正常通信
- **多PP模式**：保持跨步分组策略，实现真正的Pipeline并行
- **兼容性**：完全兼容SGLang原生的Pipeline并行机制

### **验证方法**
重新运行TP=2, PP=1的配置，应该看到：
```
[MODEL_PARALLEL] 🔧 单PP stage模式：创建1个PP组包含所有进程
[MODEL_PARALLEL] 创建PP组 0: ranks=[0, 1] (单PP stage)
```

而不是之前的错误：
```
[MODEL_PARALLEL] 将创建2个PP组，每组1个进程
[MODEL_PARALLEL] 创建PP组 0: ranks=[0] (跨步分组)
[MODEL_PARALLEL] 创建PP组 1: ranks=[1] (跨步分组)
``` 