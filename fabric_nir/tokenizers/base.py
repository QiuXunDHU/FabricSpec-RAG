"""
Tokenizer基类
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class TokenizerBase(ABC):
    """
    分词器的基类
    
    所有分词器实现必须继承此类并实现相关方法
    """
    
    def __init__(self):
        """
        初始化分词器
        """
        self.vocab_size = 0
        self.pad_token_id = 0
        self.bos_token_id = 0
        self.eos_token_id = 0
    
    @abstractmethod
    def encode(self, text, max_length: int, percentages: Optional[List[float]] = None) -> Tuple[List[int], Optional[List[float]]]:
        """
        将文本编码为token ID
        
        Args:
            text: 输入文本或组件列表
            max_length: 最大序列长度
            percentages: 比例列表（可选）
            
        Returns:
            token ID列表和比例列表（如果提供）
        """
        pass
    
    @abstractmethod
    def decode(self, ids: List[int], percentages: Optional[List[float]] = None) -> str:
        """
        将token ID解码为文本
        
        Args:
            ids: token ID列表
            percentages: 比例列表（可选）
            
        Returns:
            解码后的文本
        """
        pass
    
    @abstractmethod
    def parse_label(self, label: str) -> Tuple[List[str], List[float]]:
        """
        解析标签为成分和比例
        
        Args:
            label: 标签文本
            
        Returns:
            成分列表和比例列表
        """
        pass
    
    def batch_decode(self, token_ids_batch: List[List[int]], percentages_batch: Optional[List[List[float]]] = None) -> List[str]:
        """
        批量解码token IDs为标签字符串
        
        Args:
            token_ids_batch: token ID批次
            percentages_batch: 比例批次（可选）
            
        Returns:
            解码后的文本列表
        """
        if percentages_batch is None:
            percentages_batch = [None] * len(token_ids_batch)
        return [
            self.decode(tokens, percentages)
            for tokens, percentages in zip(token_ids_batch, percentages_batch)
        ]
