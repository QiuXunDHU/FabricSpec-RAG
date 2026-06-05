"""
自监督预训练模块
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random

from fabric_nir.data.dataset import FabricDataset
from fabric_nir.models.seq2seq.single_task_seq2seq import SingleTaskSeq2Seq
from fabric_nir.tokenizers.component_tokenizer import FabricComponentTokenizer
from fabric_nir.metrics.metrics_logger import MetricsLogger


class SelfSupervisedPretrainer:
    """
    自监督预训练器 - 使用MLM风格的预训练方法
    """
    
    def __init__(self, config_manager, experiment_id=None):
        """
        初始化自监督预训练器
        
        Args:
            config_manager: 配置管理器
            experiment_id: 实验ID（可选）
        """
        self.config = config_manager.config
        self.experiment_id = experiment_id or "self_supervised"
        
        # 创建结果目录
        self.results_dir = self.config.get("results_dir", "results")
        self.pretrain_dir = os.path.join(self.results_dir, "pretrained")
        self.metrics_dir = os.path.join(self.results_dir, "metrics")
        
        os.makedirs(self.pretrain_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # 初始化分词器
        vocab_path = os.path.join("fabric_nir", "tokenizers", "vocab", "component_vocab.json")
        self.tokenizer = FabricComponentTokenizer(vocab_path)
        
        # 初始化数据集和数据加载器
        self._init_datasets()
        
        # 初始化模型
        self._init_model()
        
        # 初始化评价指标记录器
        self.metrics_logger = MetricsLogger(self.metrics_dir, f"{self.experiment_id}_pretrain")
        
        # MLM相关参数
        self.mlm_config = self.config.get("pretrain", {}).get("mlm", {})
        self.mask_prob = self.mlm_config.get("mask_prob", 0.15)
        self.random_prob = self.mlm_config.get("random_prob", 0.1)
        self.keep_prob = self.mlm_config.get("keep_prob", 0.1)
        self.mask_token_id = self.tokenizer.mask_token_id
    
    def _init_datasets(self):
        """
        初始化数据集和数据加载器
        """
        # 获取数据配置
        data_config = self.config.get("data", {})
        train_file = data_config.get("train_file", "data/train.xlsx")
        valid_file = data_config.get("valid_file", "data/valid.xlsx")
        batch_size = data_config.get("batch_size", 32)
        max_length = data_config.get("max_length", 16)
        
        # 创建数据集
        self.train_dataset = FabricDataset(
            file_path=train_file,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        self.valid_dataset = FabricDataset(
            file_path=valid_file,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def _init_model(self):
        """
        初始化模型
        """
        # 获取模型配置
        model_config = self.config.get("model", {})
        
        # 创建模型
        self.model = SingleTaskSeq2Seq(
            config=model_config,
            vocab_size=self.tokenizer.vocab_size
        )
        
        # 初始化损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    
    def _mask_tokens(self, inputs):
        """
        对输入序列进行MLM风格的掩码
        
        Args:
            inputs: 输入序列，形状为 [batch_size, seq_len]
            
        Returns:
            masked_inputs: 掩码后的输入序列
            labels: 原始序列（用于计算损失）
        """
        labels = inputs.clone()
        
        # 创建掩码概率矩阵
        probability_matrix = torch.full(labels.shape, self.mask_prob, device=labels.device)
        
        # 不掩码特殊token（如PAD、SOS、EOS等）
        special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool, device=inputs.device)
        for token_id in [self.tokenizer.pad_token_id, self.tokenizer.sos_token_id, self.tokenizer.eos_token_id]:
            special_tokens_mask = special_tokens_mask | (inputs == token_id)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # 确定要掩码的位置
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 将不需要预测的位置设为-100（CrossEntropyLoss会忽略这些位置）
        labels[~masked_indices] = -100
        
        # 80%的时间用mask token替换
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=labels.device)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_id
        
        # 10%的时间用随机token替换
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=labels.device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.tokenizer.vocab_size, labels.shape, dtype=torch.long, device=labels.device)
        inputs[indices_random] = random_words[indices_random]
        
        # 10%的时间保持不变
        # 剩余的位置已经在上面的操作中保持不变
        
        return inputs, labels
    
    def _mask_spectral_features(self, spectrals):
        """
        对光谱特征进行掩码，模拟MLM风格的预训练
        
        Args:
            spectrals: 光谱特征，形状为 [batch_size, channels, features]
            
        Returns:
            masked_spectrals: 掩码后的光谱特征
            mask: 掩码矩阵，1表示被掩码，0表示未被掩码
        """
        batch_size, channels, features = spectrals.shape
        
        # 创建掩码矩阵
        mask = torch.zeros((batch_size, channels), device=spectrals.device)
        
        # 随机选择要掩码的通道
        for i in range(batch_size):
            # 随机确定要掩码的通道数量（10%-30%）
            num_to_mask = max(1, int(channels * random.uniform(0.1, 0.3)))
            
            # 随机选择要掩码的通道
            mask_indices = random.sample(range(channels), num_to_mask)
            mask[i, mask_indices] = 1
        
        # 扩展掩码维度以匹配光谱特征
        mask = mask.unsqueeze(-1).expand(-1, -1, features)
        
        # 创建掩码后的光谱特征（将被掩码的通道置零）
        masked_spectrals = spectrals.clone()
        masked_spectrals[mask.bool()] = 0.0
        
        return masked_spectrals, mask
    
    def pretrain(self, epochs=10, device="cuda"):
        """
        自监督预训练
        
        Args:
            epochs: 训练轮数
            device: 计算设备
        """
        # 获取训练配置
        pretrain_config = self.config.get("pretrain", {})
        lr = pretrain_config.get("learning_rate", 0.001)
        weight_decay = pretrain_config.get("weight_decay", 0.0001)
        clip_grad_norm = pretrain_config.get("clip_grad_norm", 1.0)
        
        # 设置设备
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # 初始化优化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 训练循环
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            
            # 训练一个epoch
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Pretrain]")
            for batch_idx, batch in enumerate(train_pbar):
                # 获取数据
                spectrals = batch["spectral"].to(device)
                labels = batch["label_ids"].to(device)
                
                # 对光谱特征进行掩码
                masked_spectrals, spectral_mask = self._mask_spectral_features(spectrals)
                
                # 对标签序列进行掩码（MLM风格）
                masked_labels, mlm_targets = self._mask_tokens(labels[:, :-1])
                
                # 前向传播
                outputs = self.model(masked_spectrals, masked_labels)
                
                # 计算损失
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    mlm_targets.reshape(-1)
                )
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                
                # 更新参数
                optimizer.step()
                
                # 更新损失
                train_loss += loss.item()
                train_pbar.set_postfix({"loss": loss.item()})
            
            # 计算平均损失
            train_loss /= len(self.train_loader)
            
            # 验证模式
            self.model.eval()
            valid_loss = 0.0
            
            # 验证一个epoch
            valid_pbar = tqdm(self.valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Pretrain Valid]")
            with torch.no_grad():
                for batch_idx, batch in enumerate(valid_pbar):
                    # 获取数据
                    spectrals = batch["spectral"].to(device)
                    labels = batch["label_ids"].to(device)
                    
                    # 对光谱特征进行掩码
                    masked_spectrals, spectral_mask = self._mask_spectral_features(spectrals)
                    
                    # 对标签序列进行掩码（MLM风格）
                    masked_labels, mlm_targets = self._mask_tokens(labels[:, :-1])
                    
                    # 前向传播
                    outputs = self.model(masked_spectrals, masked_labels)
                    
                    # 计算损失
                    loss = self.criterion(
                        outputs.reshape(-1, outputs.size(-1)),
                        mlm_targets.reshape(-1)
                    )
                    
                    # 更新损失
                    valid_loss += loss.item()
                    valid_pbar.set_postfix({"loss": loss.item()})
            
            # 计算平均损失
            valid_loss /= len(self.valid_loader)
            
            # 更新评价指标记录器
            metrics = {
                "train_loss": train_loss,
                "valid_loss": valid_loss
            }
            self.metrics_logger.update(metrics, phase="pretrain", epoch=epoch)
            
            # 打印指标
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Pretrain Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
            
            # 保存模型
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                model_path = os.path.join(self.pretrain_dir, f"{self.experiment_id}_pretrain_epoch_{epoch+1}.pt")
                torch.save(self.model.state_dict(), model_path)
                print(f"Saved pretrained model to {model_path}")
        
        # 保存最终预训练模型
        model_path = os.path.join(self.pretrain_dir, f"{self.experiment_id}_pretrain_final.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved final pretrained model to {model_path}")
        
        # 保存评价指标历史
        self.metrics_logger.save()
        
        return model_path
