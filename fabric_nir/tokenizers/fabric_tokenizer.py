"""
纺织品分词器
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .base import TokenizerBase


class FabricTokenizer(TokenizerBase):
    """
    纺织品成分分词器
    用于将纺织品成分字符串转换为token序列
    """
    
    def __init__(self, vocab_path: str):
        """
        初始化纺织品分词器
        
        Args:
            vocab_path: 词表文件路径
        """
        super().__init__()
        
        # 加载词表
        self.vocab = self._load_vocab(vocab_path)
        self.vocab_path = vocab_path
        
        # 加载特殊标记映射文件
        tokenizer_dir = Path(vocab_path).parent
        special_tokens_map_path = tokenizer_dir / "special_tokens_map.json"
        with open(special_tokens_map_path, 'r', encoding='utf-8') as f:
            self.special_tokens_map = json.load(f)
        
        # 提取所有特殊标记内容并确保添加到词表
        self.special_tokens = [v["content"] for v in self.special_tokens_map.values()]
        self._add_special_tokens()
        
        # 初始化特殊标记属性
        self.bos_token = self.special_tokens_map["bos_token"]["content"]
        self.eos_token = self.special_tokens_map["eos_token"]["content"]
        self.unk_token = self.special_tokens_map["unk_token"]["content"]
        self.pad_token = self.special_tokens_map["pad_token"]["content"]
        self.sep_token = self.special_tokens_map["sep_token"]["content"]
        self.cls_token = self.special_tokens_map["cls_token"]["content"]
        self.mask_token = self.special_tokens_map["mask_token"]["content"]
        
        # 特殊token ID
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        
        # ID到token的映射
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # 成分和比例的正则表达式
        self.pattern = re.compile(r'([A-Za-z]+)(\d+\.?\d*)')
        
        # 词表大小
        self.vocab_size = len(self.vocab)
    
    def _load_vocab(self, path: str) -> Dict[str, int]:
        """
        加载词表文件
        
        Args:
            path: 词表文件路径
            
        Returns:
            词表字典
        """
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _add_special_tokens(self):
        """
        确保特殊标记存在于词表中
        """
        max_id = max(self.vocab.values(), default=-1)
        for idx, token in enumerate(self.special_tokens):
            if token not in self.vocab:
                self.vocab[token] = max_id + idx + 1
    
    def tokenize(self, text: str) -> List[str]:
        """
        将纺织品成分字符串转换为token序列
        
        示例:
        "W80P20" -> [bos_token, "W", "80", "P", "20", eos_token]
        
        Args:
            text: 纺织品成分字符串
            
        Returns:
            token序列
        """
        # 拆分成分
        parts = []
        remaining_text = text
        
        while remaining_text:
            match = self.pattern.match(remaining_text)
            if match:
                component, value = match.groups()
                parts.extend([component, value])
                remaining_text = remaining_text[match.end():]
            else:
                # 处理无法匹配的部分
                if remaining_text[0].isdigit():
                    parts.append(remaining_text)
                else:
                    parts.append(remaining_text[0])
                remaining_text = remaining_text[1:]
        
        # 添加特殊标记
        return [self.bos_token] + parts + [self.eos_token]
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        将token序列转换为ID序列
        
        Args:
            tokens: token序列
            
        Returns:
            ID序列
        """
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        将ID序列转换为token序列
        
        Args:
            ids: ID序列
            
        Returns:
            token序列
        """
        return [self.inverse_vocab.get(id_, self.unk_token) for id_ in ids if id_ in self.inverse_vocab]
    
    def parse_label(self, label: str) -> Tuple[List[str], List[float]]:
        """
        解析标签为成分和比例
        
        Args:
            label: 标签文本
            
        Returns:
            成分列表和比例列表
        """
        components = []
        percentages = []
        
        # 使用finditer处理连续匹配
        matches = list(self.pattern.finditer(label))
        
        # 提取所有有效成分和比例
        for match in matches:
            comp = match.group(1)
            perc = float(match.group(2))
            
            components.append(comp)
            percentages.append(perc / 100)  # 转换为比例
        
        return components, percentages
    
    def encode(self, text: str, max_length: int, percentages: Optional[List[float]] = None) -> Tuple[List[int], Optional[List[float]]]:
        """
        将文本编码为token ID
        
        Args:
            text: 输入文本
            max_length: 最大序列长度
            percentages: 比例列表（可选）
            
        Returns:
            token ID列表和比例列表（如果提供）
        """
        # 分词
        tokens = self.tokenize(text)
        
        # 转换为ID
        token_ids = self.convert_tokens_to_ids(tokens)
        
        # 截断或填充
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        elif len(token_ids) < max_length:
            padding = [self.pad_token_id] * (max_length - len(token_ids))
            token_ids += padding
        
        # 处理比例（如果提供）
        if percentages:
            if len(percentages) > max_length:
                percentages = percentages[:max_length]
            elif len(percentages) < max_length:
                percentages += [0.0] * (max_length - len(percentages))
        
        return token_ids, percentages
    
    def decode(self, token_ids: List[int], percentages: Optional[List[float]] = None) -> str:
        """
        将token ID解码为文本
        
        Args:
            token_ids: token ID列表
            percentages: 比例列表（可选）
            
        Returns:
            解码后的文本
        """
        # 转换为token
        tokens = self.convert_ids_to_tokens(token_ids)
        
        # 去除特殊标记
        special_ids = {self.vocab[t] for t in self.special_tokens}
        tokens = [t for t in tokens if t not in self.special_tokens and self.vocab.get(t, self.unk_token_id) not in special_ids]
        
        # 合并成分
        components = []
        current_component = ""
        
        for token in tokens:
            if token.isalpha():
                if current_component:
                    components.append(current_component)
                current_component = token
            else:
                current_component += token
        
        if current_component:
            components.append(current_component)
        
        return "".join(components)
    
    def save_vocab(self, save_path: str):
        """
        保存词表
        
        Args:
            save_path: 保存路径
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
