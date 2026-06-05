"""
Beam Search生成算法模块
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class BeamSearchGenerator:
    """
    Beam Search生成算法
    支持单任务和多任务模型
    """
    
    def __init__(self, model, tokenizer, config=None):
        """
        初始化Beam Search生成器
        
        Args:
            model: 模型
            tokenizer: 分词器
            config: 配置
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        
        # 获取配置
        self.beam_size = self.config.get("beam_size", 3)
        self.max_length = self.config.get("max_length", 16)
        self.temperature = self.config.get("temperature", 1.0)
        self.top_k = self.config.get("top_k", 5)
    
    def generate(self, x, task_type="single_task"):
        """
        生成序列
        
        Args:
            x: 输入张量，形状为 [batch_size, in_channels, seq_len]
            task_type: 任务类型，"single_task" 或 "multi_task"
            
        Returns:
            results: 生成结果列表，每个元素为 (components, ratios)
        """
        # 设置模型为评估模式
        self.model.eval()
        
        # 获取批次大小
        batch_size = x.size(0)
        
        # 结果列表
        results = []
        
        # 逐个样本生成
        for i in range(batch_size):
            # 提取单个样本
            sample = x[i:i+1]
            
            # 生成序列
            if task_type == "single_task":
                components, ratios = self._generate_single_task(sample)
            else:
                components, ratios = self._generate_multi_task(sample)
            
            # 添加到结果列表
            results.append((components, ratios))
        
        return results
    
    def _generate_single_task(self, x):
        """
        单任务模型生成序列
        
        Args:
            x: 输入张量，形状为 [1, in_channels, seq_len]
            
        Returns:
            components: 成分列表
            ratios: 比例列表
        """
        # 获取设备
        device = x.device
        
        # 提取特征
        with torch.no_grad():
            features = self.model.get_embeddings(x)
        
        # 初始化beam
        beam = [(
            torch.tensor([[self.tokenizer.bos_token_id]], device=device),  # 序列
            0.0,  # 分数
            None  # 隐藏状态
        )]
        
        # 逐步生成
        for _ in range(self.max_length - 1):
            # 候选列表
            candidates = []
            
            # 处理每个beam
            for seq, score, hidden in beam:
                # 检查是否已经生成了结束符
                if seq[0, -1].item() == self.tokenizer.eos_token_id:
                    candidates.append((seq, score, hidden))
                    continue
                
                # 解码一步
                with torch.no_grad():
                    decoder_result = self.model.decoder(seq, hidden, features)
                    if isinstance(decoder_result, tuple):
                        output, hidden = decoder_result
                    else:
                        output = decoder_result
                        hidden = None
                    logits = output[0, -1] / self.temperature
                    
                    # Top-k采样
                    if self.top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, self.top_k)
                        probs = F.softmax(top_k_logits, dim=-1)
                        
                        # 添加候选
                        for i, prob in enumerate(probs):
                            next_token = top_k_indices[i].unsqueeze(0).unsqueeze(0)
                            next_seq = torch.cat([seq, next_token], dim=1)
                            next_score = score - torch.log(prob).item()
                            candidates.append((next_seq, next_score, hidden))
                    else:
                        # 全词表采样
                        probs = F.softmax(logits, dim=-1)
                        
                        # 添加候选
                        for i, prob in enumerate(probs):
                            if prob > 0:
                                next_token = torch.tensor([[i]], device=device)
                                next_seq = torch.cat([seq, next_token], dim=1)
                                next_score = score - torch.log(prob).item()
                                candidates.append((next_seq, next_score, hidden))
            
            # 如果所有beam都已经生成了结束符，提前结束
            if all(seq[0, -1].item() == self.tokenizer.eos_token_id for seq, _, _ in candidates):
                beam = candidates[:self.beam_size]
                break
            
            # 按分数排序并选择前beam_size个
            beam = sorted(candidates, key=lambda x: x[1])[:self.beam_size]
        
        # 选择最佳序列
        best_seq = beam[0][0].squeeze().cpu().numpy()
        
        # 解码序列
        decoded = self.tokenizer.decode(best_seq)
        
        # 解析成分和比例
        try:
            components, ratios = self.tokenizer.parse_label(decoded)
        except ValueError:
            components, ratios = [], []
        
        return components, ratios
    
    def _generate_multi_task(self, x):
        """
        多任务模型生成序列
        
        Args:
            x: 输入张量，形状为 [1, in_channels, seq_len]
            
        Returns:
            components: 成分列表
            ratios: 比例列表
        """
        # 获取设备
        device = x.device
        
        # 提取特征
        with torch.no_grad():
            features = self.model.get_embeddings(x)
        
        # 初始化beam
        beam = [(
            torch.tensor([[self.tokenizer.bos_token_id]], device=device),  # 序列
            0.0,  # 分数
            None,  # 隐藏状态
            []  # 比例列表
        )]
        
        # 逐步生成
        for _ in range(self.max_length - 1):
            # 候选列表
            candidates = []
            
            # 处理每个beam
            for seq, score, hidden, ratios in beam:
                # 检查是否已经生成了结束符
                if seq[0, -1].item() == self.tokenizer.eos_token_id:
                    candidates.append((seq, score, hidden, ratios))
                    continue
                
                # 解码一步
                with torch.no_grad():
                    cls_output, reg_output = self.model(x, seq, None)
                    logits = cls_output[0, -1] / self.temperature
                    ratio = torch.sigmoid(reg_output[0, -1]).item()  # 使用sigmoid确保比例在0-1之间
                    
                    # Top-k采样
                    if self.top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, self.top_k)
                        probs = F.softmax(top_k_logits, dim=-1)
                        
                        # 添加候选
                        for i, prob in enumerate(probs):
                            next_token = top_k_indices[i].unsqueeze(0).unsqueeze(0)
                            next_seq = torch.cat([seq, next_token], dim=1)
                            next_score = score - torch.log(prob).item()
                            next_ratios = ratios + [ratio]
                            candidates.append((next_seq, next_score, hidden, next_ratios))
                    else:
                        # 全词表采样
                        probs = F.softmax(logits, dim=-1)
                        
                        # 添加候选
                        for i, prob in enumerate(probs):
                            if prob > 0:
                                next_token = torch.tensor([[i]], device=device)
                                next_seq = torch.cat([seq, next_token], dim=1)
                                next_score = score - torch.log(prob).item()
                                next_ratios = ratios + [ratio]
                                candidates.append((next_seq, next_score, hidden, next_ratios))
            
            # 如果所有beam都已经生成了结束符，提前结束
            if all(seq[0, -1].item() == self.tokenizer.eos_token_id for seq, _, _, _ in candidates):
                beam = candidates[:self.beam_size]
                break
            
            # 按分数排序并选择前beam_size个
            beam = sorted(candidates, key=lambda x: x[1])[:self.beam_size]
        
        # 选择最佳序列
        best_seq = beam[0][0].squeeze().cpu().numpy()
        best_ratios = beam[0][3]
        
        # 解码序列
        tokens = []
        for token_id in best_seq:
            if token_id == self.tokenizer.eos_token_id:
                break
            if token_id != self.tokenizer.bos_token_id:
                token = self.tokenizer.id2token.get(token_id, "<unk>")
                tokens.append(token)
        
        # 确保比例列表长度与token列表相同
        ratios = best_ratios[:len(tokens)]
        if len(ratios) < len(tokens):
            ratios.extend([0.0] * (len(tokens) - len(ratios)))
        
        return tokens, ratios
