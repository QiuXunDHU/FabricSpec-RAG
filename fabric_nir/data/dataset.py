"""
数据集模块
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class FabricDataset(Dataset):
    """
    废旧纺织品近红外光谱数据集
    用于单任务Seq2Seq模型
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
        
        # 编码标签
        self.encoded_labels = []
        for label in self.labels:
            encoded = self.tokenizer.encode(label, max_length=self.max_length)
            self.encoded_labels.append(encoded)
    
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
        
        return {
            "spectral": spectral,
            "label": label,
            "label_ids": label_ids
        }
