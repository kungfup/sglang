import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DeviceConfig:
    device: Optional[torch.device]

    def __init__(self, device: str = "cuda") -> None:
        """
        Semi-PD:
        - Allow meta device for P & D instances.
        - Support pipeline parallel with different GPU devices per stage.
        """
        if device in ["cuda", "xpu", "hpu", "cpu", "npu", "meta"]:
            self.device_type = device
        else:
            raise RuntimeError(f"Not supported device type: {device}")
        
        # 在Semi-PD PP模式下，设备会在GroupCoordinator中正确设置
        # 这里只设置设备类型，具体设备ID由parallel_state.py处理
        if device == "meta":
            self.device = None  # meta设备不需要具体设备ID
        else:
            # 🔧 修复：在Semi-PD模式下，根据环境变量设置设备
            if device == "meta":
                self.device = None  # meta设备不需要具体设备ID
            else:
                # 尝试从环境变量获取GPU ID
                import os
                gpu_id = os.environ.get('SGLANG_GPU_ID')
                if gpu_id is not None:
                    self.device = torch.device(f"{self.device_type}:{gpu_id}")
                    logger.info(f"[DEVICE_CONFIG] 从环境变量获取设备: {self.device}")
                else:
                    self.device = torch.device(self.device_type)
                    logger.info(f"[DEVICE_CONFIG] 使用默认设备类型: {self.device}")
