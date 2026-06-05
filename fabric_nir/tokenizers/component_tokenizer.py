"""
成分分词器
"""

import json
import re
from typing import List, Tuple, Optional, Union

from .base import TokenizerBase


class FabricComponentTokenizer(TokenizerBase):
    """
    废旧纺织品成分分词器
    用于将成分标签解析为token ID和比例
    """
    
    def __init__(self, vocab_path: str):
        """
        初始化成分分词器
        
        Args:
            vocab_path: 词表文件路径
        """
        super().__init__()
        
        # 加载词表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        self.vocab_path = vocab_path
        self.vocab_size = len(self.vocab)
        
        # 特殊token
        self.pad_token_id = self.vocab.get("<pad>", 1)
        self.bos_token_id = self.vocab.get("<s>", 0)
        self.eos_token_id = self.vocab.get("</s>", 2)
        self.mask_token_id = self.vocab.get("<mask>", 4)  # 添加掩码token ID
        self.unk_token_id = self.vocab.get("<unk>", 3)
        
        # 特殊token名称
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.mask_token = "<mask>"
        self.unk_token = "<unk>"
        
        # 添加其他可能需要的特殊token别名
        self.sos_token_id = self.bos_token_id  # 句子开始符号
        self.cls_token_id = self.bos_token_id  # 分类符号
        self.sep_token_id = self.eos_token_id  # 分隔符号
        
        # 特殊token名称别名
        self.sos_token = self.bos_token
        self.cls_token = self.bos_token
        self.sep_token = self.eos_token
        
        # ID到token的映射
        self.id2token = {v: k for k, v in self.vocab.items()}
        
        # 成分和比例的正则表达式
        self.pattern = re.compile(r'([A-Za-z]+)(\d+\.?\d*)')
    
    def parse_label(self, label: str) -> Tuple[List[str], List[float]]:
        """
        解析无分隔符的标签格式，例如: L50P50 → ['L','P'], [0.5, 0.5]
        
        Args:
            label: 标签文本，如 "L50P50"
            
        Returns:
            成分列表和比例列表
        """
        components = []
        percentages = []
        
        # 确保标签是字符串类型
        if not isinstance(label, str):
            print(f"警告: 标签不是字符串类型，当前类型: {type(label)}，将转换为字符串")
            label = str(label)
        
        # 使用finditer处理连续匹配
        try:
            matches = list(self.pattern.finditer(label))
            
            # 提取所有有效成分和比例
            total = 0.0
            for match in matches:
                comp = match.group(1)
                perc = float(match.group(2))
                
                # 验证成分有效性
                if comp not in self.vocab:
                    print(f"警告: 未知成分 '{comp}' 在标签 {label} 中，将使用<unk>替代")
                    comp = "<unk>"
                
                # 累积百分比
                total += perc
                
                components.append(comp)
                percentages.append(perc / 100)  # 转换为比例
            
            # 验证百分比总和（允许 ±0.1% 浮点误差）
            # if abs(total - 100) > 0.1:
            #     raise ValueError(f"Percentage sum {total}% != 100% in label: {label}")
            
            # 如果没有匹配到任何成分，返回默认值
            if not components:
                print(f"警告: 标签 '{label}' 未匹配到任何成分，将使用默认值")
                return ["<unk>"], [1.0]
                
            return components, percentages
            
        except Exception as e:
            print(f"解析标签失败: {str(e)}，标签: {label}，将使用默认值")
            return ["<unk>"], [1.0]
    
    def encode(self, input_data: Union[str, List[str]], max_length: int,
               percentages: Optional[List[float]] = None) -> Tuple[List[int], Optional[List[float]]]:
        """
        编码为多任务序列
        
        Args:
            input_data: 标签字符串或成分列表
            max_length: 最大序列长度
            percentages: 比例列表（可选）
            
        Returns:
            token ID列表和比例列表（如果提供）
        """
        try:
            # 确保输入数据类型正确
            if input_data is None:
                raise ValueError("输入数据不能为None")
                
            # 如果输入是数字，转换为字符串
            if isinstance(input_data, (int, float)):
                input_data = str(input_data)
                
            # 如果输入是字符串，先解析成成分和比例
            if isinstance(input_data, str):
                components, parsed_percentages = self.parse_label(input_data)
                if percentages is None:
                    percentages = parsed_percentages
            else:
                components = input_data
                
                # 确保components是列表
                if not isinstance(components, list):
                    raise ValueError(f"成分必须是列表类型，当前类型: {type(components)}")
            
            # 成分编码
            token_ids = []
            for c in components:
                # 确保成分是字符串
                if not isinstance(c, str):
                    c = str(c)
                    
                if c not in self.vocab:
                    print(f"警告: 未知成分 '{c}'，不在词表中，将使用<unk>替代")
                    token_ids.append(self.vocab.get("<unk>", 3))
                else:
                    token_ids.append(self.vocab[c])
            
            # 添加起始符
            token_ids = [self.bos_token_id] + token_ids
            
            # 填充处理
            if len(token_ids) < max_length:
                token_ids += [self.eos_token_id]
                padding = [self.pad_token_id] * (max_length - len(token_ids))
                token_ids += padding
                
                if percentages:
                    percentages = [0.0] + percentages  # 为起始符添加比例0
                    percentages += [0.0] * (max_length - len(percentages))
            
            return token_ids[:max_length], percentages[:max_length] if percentages else None
        
        except Exception as e:
            print(f"编码失败: {str(e)}，输入: {input_data}，将返回默认编码")
            # 返回默认编码，避免程序崩溃
            default_ids = [self.bos_token_id, self.eos_token_id] + [self.pad_token_id] * (max_length - 2)
            default_percs = [0.0] * max_length if percentages is not None else None
            return default_ids, default_percs
    
    def decode(self, token_ids: List[int], percentages: Optional[List[float]] = None) -> str:
        """
        解码为标签字符串
        
        Args:
            token_ids: token ID列表
            percentages: 比例列表（可选）
            
        Returns:
            解码后的标签字符串
        """
        try:
            def flatten(values):
                if hasattr(values, "detach"):
                    values = values.detach().cpu().tolist()
                elif hasattr(values, "tolist"):
                    values = values.tolist()

                if isinstance(values, (list, tuple)):
                    out = []
                    for value in values:
                        out.extend(flatten(value))
                    return out
                return [values]

            token_ids = [int(i) for i in flatten(token_ids)]
            if percentages is not None:
                percentages = [float(p) for p in flatten(percentages)]

            # 过滤掉特殊token
            components = []
            for i in token_ids:
                if i != self.pad_token_id and i != self.eos_token_id and i != self.bos_token_id:
                    # 确保id在id2token中
                    if i in self.id2token:
                        components.append(self.id2token[i])
                    else:
                        print(f"警告: 未知token ID {i}，将使用<unk>替代")
                        components.append("<unk>")
            
            if percentages is not None:
                # 过滤掉特殊token对应的比例
                valid_percs = []
                for p, token_id in zip(percentages, token_ids):
                    if token_id != self.pad_token_id and token_id != self.eos_token_id and token_id != self.bos_token_id:
                        valid_percs.append(p)
                
                # 确保components和valid_percs长度一致
                if len(components) != len(valid_percs):
                    print(f"警告: 成分数量 {len(components)} 与比例数量 {len(valid_percs)} 不一致，将使用默认比例")
                    valid_percs = [1.0/len(components)] * len(components) if components else []
                
                return "".join(f"{c}{format(p * 100, '.1f')}" for c, p in zip(components, valid_percs))
            
            return "".join(components)
        except Exception as e:
            print(f"解码失败: {str(e)}，token_ids: {token_ids}，将返回空字符串")
            return ""
