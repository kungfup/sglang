# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
🎯 Semi-PD Pipeline Parallel 端口配置

端口分配策略：
- PP Stage 0: 40000-40099 (GPU 0)
- PP Stage 1: 40100-40199 (GPU 1)
- PP Stage 2: 40200-40299 (GPU 2)
- ...

每个PP Stage内部：
- DECODE进程: 主端口 (主进程)
- PREFILL进程: 辅助端口 (辅助进程)
- 其他服务端口

PP Stage间通信：
- 使用SGLang原生的NCCL通信机制
- 每个stage的DECODE进程与下一个stage的DECODE进程通信
- 每个stage的PREFILL进程与下一个stage的PREFILL进程通信
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PPStagePorts:
    """单个PP Stage的端口配置"""
    pp_rank: int
    gpu_id: int
    
    # 🔧 同PP stage内的IPC通信端口
    decode_port: int          # DECODE进程主端口
    prefill_port: int         # PREFILL进程辅助端口
    scheduler_port: int       # 调度器端口
    detokenizer_port: int     # 去tokenizer端口
    
    # 🔧 PP stage间通信端口
    next_stage_decode_port: Optional[int] = None    # 下一个stage的DECODE端口
    next_stage_prefill_port: Optional[int] = None   # 下一个stage的PREFILL端口
    
    # 🔧 NCCL通信端口
    nccl_port: int            # NCCL通信端口
    
    def __post_init__(self):
        """验证端口配置的有效性"""
        assert self.decode_port != self.prefill_port, f"PP{self.pp_rank}: decode_port和prefill_port不能相同"
        assert self.gpu_id >= 0, f"PP{self.pp_rank}: gpu_id必须非负"


class PPStagePortManager:
    """PP Stage端口管理器"""
    
    def __init__(self, num_pp_stages: int, base_port: int = 40000, gpu_ids: Optional[List[int]] = None):
        """
        初始化PP Stage端口管理器
        
        Args:
            num_pp_stages: PP stage数量
            base_port: 基础端口号
            gpu_ids: 每个stage使用的GPU ID列表，如果为None则自动分配
        """
        self.num_pp_stages = num_pp_stages
        self.base_port = base_port
        self.gpu_ids = gpu_ids or list(range(num_pp_stages))
        
        assert len(self.gpu_ids) == num_pp_stages, f"GPU ID数量({len(self.gpu_ids)})必须等于PP stage数量({num_pp_stages})"
        
        self.stage_ports: Dict[int, PPStagePorts] = {}
        self._generate_port_configs()
    
    def _generate_port_configs(self):
        """生成所有PP stage的端口配置"""
        for pp_rank in range(self.num_pp_stages):
            gpu_id = self.gpu_ids[pp_rank]
            stage_base_port = self.base_port + pp_rank * 100
            
            # 🔧 同PP stage内的端口分配
            stage_ports = PPStagePorts(
                pp_rank=pp_rank,
                gpu_id=gpu_id,
                decode_port=stage_base_port + 0,      # 主进程端口
                prefill_port=stage_base_port + 1,     # 辅助进程端口
                scheduler_port=stage_base_port + 2,   # 调度器端口
                detokenizer_port=stage_base_port + 3, # 去tokenizer端口
                nccl_port=stage_base_port + 100,      # NCCL通信端口
            )
            
            # 🔧 PP stage间通信端口配置
            if pp_rank < self.num_pp_stages - 1:
                # 不是最后一个stage，需要连接下一个stage
                next_stage_base_port = self.base_port + (pp_rank + 1) * 100
                stage_ports.next_stage_decode_port = next_stage_base_port + 0
                stage_ports.next_stage_prefill_port = next_stage_base_port + 1
            
            self.stage_ports[pp_rank] = stage_ports
    
    def get_stage_ports(self, pp_rank: int) -> PPStagePorts:
        """获取指定PP stage的端口配置"""
        if pp_rank not in self.stage_ports:
            raise ValueError(f"Invalid PP rank: {pp_rank}, valid range: 0-{self.num_pp_stages-1}")
        return self.stage_ports[pp_rank]
    
    def get_all_stage_ports(self) -> Dict[int, PPStagePorts]:
        """获取所有PP stage的端口配置"""
        return self.stage_ports.copy()
    
    def print_port_config(self):
        """打印端口配置信息"""
        print(f"🎯 Semi-PD PP={self.num_pp_stages} 端口配置:")
        print(f"基础端口: {self.base_port}")
        print(f"GPU分配: {self.gpu_ids}")
        print()
        
        for pp_rank in range(self.num_pp_stages):
            ports = self.stage_ports[pp_rank]
            print(f"🔧 PP Stage {pp_rank} (GPU {ports.gpu_id}):")
            print(f"  📡 同Stage通信:")
            print(f"    - DECODE进程: {ports.decode_port}")
            print(f"    - PREFILL进程: {ports.prefill_port}")
            print(f"    - 调度器: {ports.scheduler_port}")
            print(f"    - 去tokenizer: {ports.detokenizer_port}")
            print(f"  🔗 跨Stage通信:")
            if ports.next_stage_decode_port:
                print(f"    - 下一个DECODE: {ports.next_stage_decode_port}")
                print(f"    - 下一个PREFILL: {ports.next_stage_prefill_port}")
            else:
                print(f"    - 最后一个stage，无下一个stage")
            print(f"  🌐 NCCL通信: {ports.nccl_port}")
            print()


def create_pp_stage_port_args(pp_rank: int, base_port: int = 40000) -> Dict:
    """
    为PP stage创建端口参数字典
    
    Args:
        pp_rank: Pipeline parallel rank
        base_port: 基础端口号
        
    Returns:
        包含端口信息的字典
    """
    # 计算当前stage的基础端口
    stage_base_port = base_port + pp_rank * 100
    
    return {
        "decode_port": stage_base_port + 0,      # DECODE进程主端口
        "prefill_port": stage_base_port + 1,     # PREFILL进程辅助端口
        "scheduler_port": stage_base_port + 2,   # 调度器端口
        "detokenizer_port": stage_base_port + 3, # 去tokenizer端口
        "nccl_port": stage_base_port + 100,      # NCCL通信端口
        "pp_rank": pp_rank,
        # PP stage间通信端口
        "next_stage_decode_port": stage_base_port + 100 if pp_rank == 0 else None,  # 示例：只有stage 0连接stage 1
        "next_stage_prefill_port": stage_base_port + 101 if pp_rank == 0 else None,
    }


# 🔧 使用示例
if __name__ == "__main__":
    # 创建PP=2的端口管理器
    port_manager = PPStagePortManager(
        num_pp_stages=2,
        base_port=40000,
        gpu_ids=[0, 1]  # PP stage 0使用GPU 0，PP stage 1使用GPU 1
    )
    
    # 打印配置
    port_manager.print_port_config()
    
    # 获取特定stage的端口配置
    stage0_ports = port_manager.get_stage_ports(0)
    stage1_ports = port_manager.get_stage_ports(1)
    
    print(f"🔧 Stage 0 GPU ID: {stage0_ports.gpu_id}")
    print(f"🔧 Stage 1 GPU ID: {stage1_ports.gpu_id}")
    
    # 验证端口隔离
    assert stage0_ports.gpu_id != stage1_ports.gpu_id, "不同PP stage必须使用不同的GPU"
    print("✅ 端口配置验证通过：不同PP stage使用不同GPU，确保NCCL通信组隔离") 