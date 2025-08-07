
# MPS环境对CUDA Graph影响测试

## 测试1: 禁用MPS测试
```bash
# 停止MPS服务
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
# 启动Semi-PD (单进程模式)
# 观察CUDA Graph性能变化
```

## 测试2: 对比不同MPS配置
```bash
# 配置1: 50%-50%分割
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50

# 配置2: 80%-20%分割  
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80

# 观察CUDA Graph replay时间变化
```

## 测试3: 非Semi-PD环境对比
```bash
# 启动普通SGLang 0.4.8 (非Semi-PD)
# 对比CUDA Graph性能
# 确认是否Semi-PD特有问题
```

## 预期结果:
- 如果禁用MPS后性能正常 → MPS兼容性问题
- 如果不同配置有差异 → MPS资源分割影响
- 如果非Semi-PD正常 → Semi-PD架构特殊问题
