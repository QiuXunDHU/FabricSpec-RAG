"""
Backbone模块基类
"""

import torch.nn as nn
from abc import ABC, abstractmethod
from .backbones import DenseBackbone, ResidualBackbone, MultiCovBackbone

__all__ = ["DenseBackbone", "ResidualBackbone", "MultiCovBackbone"]

class BackboneBase(nn.Module, ABC):
    """
    特征提取backbone的基类
    
    所有backbone实现必须继承此类并实现forward方法
    """
    
    def __init__(self, in_channels, out_channels):
        """
        初始化backbone
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    @abstractmethod
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, in_channels, sequence_length]
            
        Returns:
            输出张量 [batch_size, out_channels, sequence_length]
        """
        pass
