"""
Seq2Seq模型基类
"""

import torch.nn as nn
from abc import ABC, abstractmethod

class Seq2SeqBase(nn.Module, ABC):
    """
    Seq2Seq模型的基类
    
    所有Seq2Seq模型实现必须继承此类并实现forward和generate方法
    """
    
    def __init__(self, backbone, decoder, tokenizer):
        """
        初始化Seq2Seq模型
        
        Args:
            backbone: 特征提取backbone
            decoder: 解码器
            tokenizer: 分词器
        """
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.tokenizer = tokenizer
    
    @abstractmethod
    def forward(self, spectrals, decoder_inputs, labels=None):
        """
        前向传播
        
        Args:
            spectrals: 光谱输入
            decoder_inputs: 解码器输入
            labels: 标签（训练时使用）
            
        Returns:
            损失和预测结果
        """
        pass
    
    @abstractmethod
    def generate(self, spectrals, max_length=16):
        """
        生成序列
        
        Args:
            spectrals: 光谱输入
            max_length: 最大生成长度
            
        Returns:
            生成的序列
        """
        pass
