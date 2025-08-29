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
            # 尝试从parallel_state获取正确的设备信息
            try:
                from sglang.srt.distributed.parallel_state import get_group_coordinator
                coordinator = get_group_coordinator()
                if coordinator and hasattr(coordinator, 'device'):
                    self.device = coordinator.device
                    logger.info(f"[DEVICE_CONFIG] 从GroupCoordinator获取设备: {self.device}")
                else:
                    self.device = torch.device(self.device_type)
                    logger.info(f"[DEVICE_CONFIG] 使用默认设备类型: {self.device}")
            except Exception as e:
                # 如果无法获取GroupCoordinator，使用默认设备类型
                self.device = torch.device(self.device_type)
                logger.info(f"[DEVICE_CONFIG] 无法获取GroupCoordinator，使用默认设备: {self.device}, 错误: {e}")
