"""
多任务Seq2Seq模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fabric_nir.models.backbones.backbones import DenseBackbone, ResidualBackbone, MultiCovBackbone
from fabric_nir.models.attention.cbam import CBAM
from fabric_nir.models.decoders.decoders import GRUDecoder, LSTMDecoder, TransformerDecoder


class MultiTaskSeq2Seq(nn.Module):
    """
    多任务Seq2Seq模型
    同时处理分类和回归任务
    """
    
    def __init__(self, config, vocab_size):
        """
        初始化多任务Seq2Seq模型
        
        Args:
            config: 模型配置
            vocab_size: 词表大小
        """
        super(MultiTaskSeq2Seq, self).__init__()
        
        # 获取配置
        backbone_config = config.get("backbone", {})
        attention_config = config.get("attention", {})
        decoder_config = config.get("decoder", {})
        
        # 初始化backbone
        backbone_type = backbone_config.get("type", "dense")
        in_channels = backbone_config.get("in_channels", 1)
        out_channels = backbone_config.get("out_channels", 128)
        
        if backbone_type == "dense":
            kernel_sizes = backbone_config.get("kernel_sizes", [3, 5, 7])
            self.backbone = DenseBackbone(in_channels, out_channels, kernel_sizes)
        elif backbone_type == "residual":
            num_blocks = backbone_config.get("num_blocks", 3)
            self.backbone = ResidualBackbone(in_channels, out_channels, num_blocks)
        elif backbone_type == "multicov":
            kernel_sizes = backbone_config.get("kernel_sizes", [3, 5, 7, 9])
            self.backbone = MultiCovBackbone(in_channels, out_channels, kernel_sizes)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
        
        # 初始化注意力机制
        use_cbam = attention_config.get("use_cbam", True)
        if use_cbam:
            reduction_ratio = attention_config.get("reduction_ratio", 16)
            feature_map_dim = backbone_config.get("feature_map_dim", 42)
            self.attention = CBAM(feature_map_dim, reduction_ratio)
        else:
            self.attention = nn.Identity()
        
        # 初始化解码器
        decoder_type = decoder_config.get("type", "transformer")
        input_size = decoder_config.get("input_size", 42)  # 与backbone输出的feature_map_dim一致
        hidden_size = decoder_config.get("hidden_size", 128)  # 与backbone的out_channels一致
        num_layers = decoder_config.get("num_layers", 2)
        dropout = decoder_config.get("dropout", 0.1)
        
        if decoder_type == "gru":
            self.decoder = GRUDecoder(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=vocab_size,
                num_layers=num_layers,
                dropout=dropout
            )
        elif decoder_type == "lstm":
            self.decoder = LSTMDecoder(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=vocab_size,
                num_layers=num_layers,
                dropout=dropout
            )
        elif decoder_type == "transformer":
            nhead = decoder_config.get("nhead", 8)
            self.decoder = TransformerDecoder(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=vocab_size,
                num_layers=num_layers,
                nhead=nhead,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        
        # 回归头。当前decoder返回词表logits，因此比例回归头接收vocab维度。
        self.regression_head = nn.Linear(vocab_size, 1)
        
        # 保存配置
        self.config = config
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.decoder_type = decoder_type
        
        # 初始化注意力权重
        self.attention_weights = None
    
    def forward(self, x, target=None, target_ratios=None):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, in_channels, seq_len]
            target: 目标张量，形状为 [batch_size, target_len]
            target_ratios: 目标比例张量，形状为 [batch_size, target_len]
            
        Returns:
            cls_output: 分类输出张量，形状为 [batch_size, target_len, vocab_size]
            reg_output: 回归输出张量，形状为 [batch_size, target_len, 1]
        """
        # 提取特征
        features = self.backbone(x)
        
        # 应用注意力机制
        if hasattr(self.attention, "forward"):
            # 修改注意力机制的应用方式，直接处理[batch, num_channel, feature_map_dim]格式
            features = self.attention(features)
        
        # 解码
        if target is None:
            # 推理模式
            return self._inference(features)
        else:
            # 训练模式
            return self._train(features, target, target_ratios)
    
    def _train(self, features, target, target_ratios):
        """
        训练模式前向传播
        
        Args:
            features: 特征张量，形状为 [batch_size, hidden_size]
            target: 目标张量，形状为 [batch_size, target_len]
            target_ratios: 目标比例张量，形状为 [batch_size, target_len]
            
        Returns:
            cls_output: 分类输出张量，形状为 [batch_size, target_len, vocab_size]
            reg_output: 回归输出张量，形状为 [batch_size, target_len-1]
        """
        # 调整特征形状以匹配目标序列长度
        # 修复target.size()返回3维的问题
        if len(target.shape) > 2:
            print(f"警告: target维度过高，当前为{len(target.shape)}D，将降维")
            # 如果target是3D，取第一个维度作为序列
            if len(target.shape) == 3:
                target = target[:, 0, :]
            else:
                # 如果是更高维度，展平到2D
                target = target.reshape(target.size(0), -1)
        
        # 将特征转换为解码器期望的形状 [batch_size, feature_len, hidden_size]
        # 对于卷积特征，需要调整为序列形式
        batch_size = features.size(0)
        if len(features.shape) == 2:  # [batch_size, hidden_size]
            # 将特征扩展为序列形式
            encoder_outputs = features.unsqueeze(1)  # [batch_size, 1, hidden_size]
        else:
            encoder_outputs = features
        
        # 解码
        decoder_output = self.decoder(target, None, encoder_outputs)
        
        # 分类输出
        cls_output = decoder_output
        
        # 回归输出 - 使用decoder_output而不是hidden_states
        # 由于解码器现在只返回outputs，我们直接使用decoder_output作为回归输入
        reg_input = decoder_output  # [batch_size, seq_len, hidden_size]
        
        # 应用回归头
        reg_output = self.regression_head(reg_input)
        
        # 确保reg_output是2D张量 [batch_size, seq_len]
        reg_output = reg_output.squeeze(-1)
        
        return cls_output, reg_output
    
    def _inference(self, features):
        """
        推理模式前向传播
        
        Args:
            features: 特征张量，形状为 [batch_size, hidden_size]
            
        Returns:
            cls_output: 分类输出张量，形状为 [batch_size, max_len, vocab_size]
            reg_output: 回归输出张量，形状为 [batch_size, max_len, 1]
        """
        # 获取配置
        batch_size = features.size(0)
        device = features.device
        max_len = 16  # 最大生成长度
        
        # 将特征转换为解码器期望的形状 [batch_size, feature_len, hidden_size]
        if len(features.shape) == 2:  # [batch_size, hidden_size]
            # 将特征扩展为序列形式
            encoder_outputs = features.unsqueeze(1)  # [batch_size, 1, hidden_size]
        else:
            encoder_outputs = features
        
        # 初始化输入
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=device)  # 起始符
        
        # 初始化输出
        cls_outputs = []
        reg_outputs = []
        
        # 初始化隐藏状态 - 不再需要，因为解码器现在只返回outputs
        # hidden = None
        
        # 逐步解码
        for _ in range(max_len):
            # 解码一步 - 传递encoder_outputs
            decoder_output = self.decoder(decoder_input, None, encoder_outputs)
            
            # 分类输出
            cls_output = decoder_output
            cls_outputs.append(cls_output)
            
            # 回归输出 - 直接使用decoder_output
            reg_input = decoder_output
            
            # 应用回归头
            reg_output = self.regression_head(reg_input)
            reg_outputs.append(reg_output)
            
            # 更新输入
            decoder_input = torch.argmax(cls_output[:, -1:], dim=-1)
            
            # 不再需要更新hidden状态，因为解码器现在只返回outputs
            # hidden = hidden_states
        
        # 拼接输出
        cls_outputs = torch.cat(cls_outputs, dim=1)
        reg_outputs = torch.cat(reg_outputs, dim=1)
        
        return cls_outputs, reg_outputs
    
    def get_embeddings(self, x):
        """
        获取嵌入向量
        
        Args:
            x: 输入张量，形状为 [batch_size, in_channels, seq_len]
            
        Returns:
            embeddings: 嵌入向量，形状为 [batch_size, hidden_size]
        """
        # 提取特征
        features = self.backbone(x)
        
        # 应用注意力机制
        if hasattr(self.attention, "forward"):
            features = self.attention(features)
        
        return features
    
    def get_attention_weights(self, x):
        """
        获取注意力权重
        
        Args:
            x: 输入张量，形状为 [batch_size, in_channels, seq_len]
            
        Returns:
            attention_weights: 注意力权重
        """
        # 提取特征
        features = self.backbone(x)
        
        # 应用注意力机制并保存权重
        if hasattr(self.attention, "forward"):
            features = self.attention(features)
            
            # 注意：这里假设CBAM模块会保存注意力权重
            # 实际实现中可能需要修改CBAM模块以返回注意力权重
            if hasattr(self.attention, "channel_attention") and hasattr(self.attention.channel_attention, "attention_weights"):
                return self.attention.channel_attention.attention_weights
        
        # 如果没有注意力权重，返回None
        return None
