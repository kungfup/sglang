"""
Semi-PD 内存管理优化配置
"""

import os

def apply_semi_pd_memory_optimizations():
    """应用Semi-PD内存优化配置"""
    
    # 禁用可能冲突的内存池
    os.environ["SGLANG_MOONCAKE_CUSTOM_MEM_POOL"] = "false"
    os.environ["SEMI_PD_DISABLE_CUSTOM_MEM_POOL"] = "true"
    
    # 强制连续内存格式
    os.environ["PYTORCH_MEMORY_FORMAT"] = "contiguous"
    os.environ["SEMI_PD_FORCE_CONTIGUOUS"] = "true"
    
    # 优化CUDA内存管理
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # 允许异步启动
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    print("✅ Semi-PD内存优化配置已应用")

if __name__ == "__main__":
    apply_semi_pd_memory_optimizations()
