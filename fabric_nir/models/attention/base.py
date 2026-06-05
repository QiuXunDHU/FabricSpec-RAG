"""
注意力机制基类
"""

import torch.nn as nn
from abc import ABC, abstractmethod

class AttentionBase(nn.Module, ABC):
    """
    注意力机制的基类
    
    所有注意力机制实现必须继承此类并实现forward方法
    """
    
    def __init__(self):
        """
        初始化注意力机制
        """
        super().__init__()
    
    @abstractmethod
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            应用注意力后的输出张量
        """
        pass
