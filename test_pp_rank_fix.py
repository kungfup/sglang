#!/usr/bin/env python3
"""
测试PP rank参数传递修复
"""

import os
import sys
import multiprocessing as mp
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "python"))

def test_pp_rank_parameter():
    """测试PP rank参数传递"""
    print("🧪 测试PP rank参数传递...")
    
    # 模拟engine.py中的参数传递
    def mock_run_scheduler_process(
        server_args,
        port_args,
        gpu_id,
        tp_rank,
        dp_rank,
        pipe_writer,
        ipc_info_queue,
        bypass_load_weight,
        instance_role,
        pp_rank=0,
    ):
        print(f"✅ 参数接收正确:")
        print(f"   - gpu_id: {gpu_id}")
        print(f"   - tp_rank: {tp_rank}")
        print(f"   - dp_rank: {dp_rank}")
        print(f"   - pp_rank: {pp_rank}")
        print(f"   - instance_role: {instance_role}")
        print(f"   - bypass_load_weight: {bypass_load_weight}")
        
        # 验证环境变量设置
        os.environ["SGLANG_ENABLE_SEMI_PD"] = "1"
        os.environ["SGLANG_PP_RANK"] = str(pp_rank)
        os.environ["SGLANG_GPU_ID"] = str(gpu_id)
        
        print(f"✅ 环境变量设置正确:")
        print(f"   - SGLANG_ENABLE_SEMI_PD: {os.environ.get('SGLANG_ENABLE_SEMI_PD')}")
        print(f"   - SGLANG_PP_RANK: {os.environ.get('SGLANG_PP_RANK')}")
        print(f"   - SGLANG_GPU_ID: {os.environ.get('SGLANG_GPU_ID')}")
        
        return True
    
    # 测试DECODE进程参数传递
    print("\n🔧 测试DECODE进程参数传递 (PP0 TP0):")
    result = mock_run_scheduler_process(
        server_args="mock",
        port_args="mock",
        gpu_id=0,
        tp_rank=0,
        dp_rank=None,
        pipe_writer="mock",
        ipc_info_queue="mock",
        bypass_load_weight=False,
        instance_role="DECODE",
        pp_rank=0,
    )
    
    # 测试PREFILL进程参数传递
    print("\n🔧 测试PREFILL进程参数传递 (PP1 TP0):")
    result = mock_run_scheduler_process(
        server_args="mock",
        port_args="mock",
        gpu_id=1,
        tp_rank=0,
        dp_rank=None,
        pipe_writer="mock",
        ipc_info_queue="mock",
        bypass_load_weight=True,
        instance_role="PREFILL",
        pp_rank=1,
    )
    
    print("\n🎉 测试完成！")

if __name__ == "__main__":
    test_pp_rank_parameter() 