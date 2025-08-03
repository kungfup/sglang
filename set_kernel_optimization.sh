#!/bin/bash
# Semi-PD 内核优化环境变量

# 强制使用高效内核
export SGLANG_FORCE_GEMVX_KERNEL=true
export SGLANG_DISABLE_CUTLASS_OPTIMIZATION=false
export SGLANG_PREFER_GEMVX_OVER_CUTLASS=true

# FlashInfer优化
export FLASHINFER_ENABLE_OPTIMIZATION=true
export FLASHINFER_USE_LEGACY_KERNEL=false

# 禁用可能冲突的0.4.8特性
export SGLANG_DISABLE_DISAGGREGATION=true
export SGLANG_DISABLE_TORCH_COMPILE=true
export SEMI_PD_MEMORY_COMPAT=true

echo "✅ Semi-PD内核优化环境变量已设置"
