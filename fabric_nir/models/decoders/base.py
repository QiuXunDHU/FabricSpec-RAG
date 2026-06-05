"""
解码器基类
"""

import torch.nn as nn
from abc import ABC, abstractmethod

class DecoderBase(nn.Module, ABC):
    """
    解码器的基类
    
    所有解码器实现必须继承此类并实现forward和decode方法
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化解码器
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    @abstractmethod
    def forward(self, x, encoder_outputs, hidden=None):
        """
        前向传播
        
        Args:
            x: 输入张量
            encoder_outputs: 编码器输出
            hidden: 隐藏状态
            
        Returns:
            输出张量和新的隐藏状态
        """
        pass
    
    @abstractmethod
    def decode(self, encoder_outputs, max_length, device):
        """
        解码生成序列
        
        Args:
            encoder_outputs: 编码器输出
            max_length: 最大生成长度
            device: 计算设备
            
        Returns:
            生成的序列
        """
        pass
