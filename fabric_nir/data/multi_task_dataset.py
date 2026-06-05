"""
多任务数据集模块
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import re


class MultiTaskFabricDataset(Dataset):
    """
    废旧纺织品近红外光谱数据集
    用于多任务Seq2Seq模型
    """
    
    def __init__(self, file_path, tokenizer, max_length=16):
        """
        初始化数据集
        
        Args:
            file_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        self.data = pd.read_excel(file_path)
        
        # 预处理数据
        self._preprocess()
    
    def _preprocess(self):
        """
        预处理数据
        """
        # 提取光谱数据 - 修正特征列筛选逻辑
        # 排除非数值列和标签列
        label_col = 'Labels'  # 假设标签列名为'Labels'
        spectral_cols = []
        
        # 更健壮的特征列筛选
        for col in self.data.columns:
            if col == label_col:
                continue
                
            # 检查列名是否为数值或可转换为数值
            is_numeric_col = False
            if isinstance(col, (int, float)):
                is_numeric_col = True
            elif isinstance(col, str):
                try:
                    float(col)  # 尝试转换为数值
                    is_numeric_col = True
                except ValueError:
                    pass
            
            if is_numeric_col:
                spectral_cols.append(col)
        
        # 检查特征列是否为空
        if not spectral_cols:
            print(f"警告: 未找到有效的光谱特征列！数据文件中的列名: {self.data.columns.tolist()[:10]}...")
            # 尝试使用所有非标签列作为特征列
            spectral_cols = [col for col in self.data.columns if col != label_col]
            if not spectral_cols:
                raise ValueError("无法找到任何可用的特征列！")
        
        print(f"找到 {len(spectral_cols)} 个光谱特征列")
        self.spectral_data = self.data[spectral_cols].values
        
        # 检查光谱数据是否为空
        if self.spectral_data.size == 0:
            raise ValueError("光谱数据为空，无法进行归一化处理")
        
        # 归一化光谱数据
        try:
            min_val = np.min(self.spectral_data)
            max_val = np.max(self.spectral_data)
            
            # 检查是否存在除零风险
            if max_val == min_val:
                print("警告: 光谱数据最大值等于最小值，将使用零填充")
                self.spectral_data = np.zeros_like(self.spectral_data)
            else:
                self.spectral_data = (self.spectral_data - min_val) / (max_val - min_val)
        except Exception as e:
            print(f"警告: 光谱数据归一化失败: {str(e)}，光谱数据形状: {self.spectral_data.shape}")
            # 使用零填充
            self.spectral_data = np.zeros_like(self.spectral_data)
        
        # 提取标签数据
        self.labels = self.data[label_col].values
        
        # 解析标签，提取成分和比例
        self.components = []
        self.ratios = []
        
        for label in self.labels:
            components, ratios = self._parse_label(label)
            self.components.append(components)
            self.ratios.append(ratios)
        
        # 编码标签
        self.encoded_labels = []
        self.encoded_ratios = []
        
        for i, (components, ratios) in enumerate(zip(self.components, self.ratios)):
            # 编码成分
            encoded, encoded_ratio = self.tokenizer.encode(self.labels[i], self.max_length)
            self.encoded_labels.append(encoded)
            
            # 如果tokenizer没有返回比例，则手动创建
            if encoded_ratio is None:
                encoded_ratio = np.zeros(self.max_length)
                for j, ratio in enumerate(ratios):
                    if j + 1 < self.max_length:  # +1 是因为第一个token是起始符
                        encoded_ratio[j + 1] = ratio / 100.0  # 归一化比例
            
            self.encoded_ratios.append(encoded_ratio)
    
    def _parse_label(self, label):
        """
        解析标签，提取成分和比例
        
        Args:
            label: 标签字符串，如 "C65.0P35.0"
            
        Returns:
            components: 成分列表
            ratios: 比例列表
        """
        components = []
        ratios = []
        
        # 确保标签是字符串类型
        if not isinstance(label, str):
            print(f"警告: 标签不是字符串类型，当前类型: {type(label)}，将转换为字符串")
            label = str(label)
        
        try:
            # 使用正则表达式匹配成分和比例
            pattern = r'([A-Za-z]+)(\d+\.\d+|\d+)'
            matches = re.findall(pattern, label)
            
            for match in matches:
                component = match[0]
                ratio = float(match[1])
                components.append(component)
                ratios.append(ratio)
            
            # 如果没有匹配到任何成分，返回默认值
            if not components:
                print(f"警告: 标签 '{label}' 未匹配到任何成分，将使用默认值")
                return ["<unk>"], [100.0]
                
            return components, ratios
            
        except Exception as e:
            print(f"解析标签失败: {str(e)}，标签: {label}，将使用默认值")
            return ["<unk>"], [100.0]
    
    def __len__(self):
        """
        返回数据集长度
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            数据项字典
        """
        # 获取光谱数据
        spectral = self.spectral_data[idx]
        spectral = torch.tensor(spectral, dtype=torch.float32).unsqueeze(0)  # 添加通道维度
        
        # 获取标签
        label = self.labels[idx]
        label_ids = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        
        # 获取比例
        ratios = torch.tensor(self.encoded_ratios[idx], dtype=torch.float32)
        
        # 确保label_ids和ratios是一维张量
        if len(label_ids.shape) > 1:
            print(f"警告: label_ids是{len(label_ids.shape)}D张量，将展平为1D")
            label_ids = label_ids.view(-1)
        
        if len(ratios.shape) > 1:
            print(f"警告: ratios是{len(ratios.shape)}D张量，将展平为1D")
            ratios = ratios.view(-1)
        
        # 确保label_ids和ratios长度一致且为max_length
        if len(label_ids) != self.max_length:
            print(f"警告: label_ids长度{len(label_ids)}与max_length{self.max_length}不一致，将调整")
            if len(label_ids) < self.max_length:
                # 填充到max_length
                padding = torch.full((self.max_length - len(label_ids),), self.tokenizer.pad_token_id, dtype=torch.long)
                label_ids = torch.cat([label_ids, padding])
            else:
                # 截断到max_length
                label_ids = label_ids[:self.max_length]
        
        if len(ratios) != self.max_length:
            print(f"警告: ratios长度{len(ratios)}与max_length{self.max_length}不一致，将调整")
            if len(ratios) < self.max_length:
                # 填充到max_length
                padding = torch.zeros(self.max_length - len(ratios), dtype=torch.float32)
                ratios = torch.cat([ratios, padding])
            else:
                # 截断到max_length
                ratios = ratios[:self.max_length]
        
        return {
            "spectral": spectral,
            "label": label,
            "label_ids": label_ids,
            "ratios": ratios
        }
