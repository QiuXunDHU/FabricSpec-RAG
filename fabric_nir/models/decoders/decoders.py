"""
解码器模块 - 修复版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GRUDecoder(nn.Module):
    """
    GRU解码器 - 包含编码GRU和解码GRU，使用多头自注意力机制
    """
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1, nhead=8):
        """
        初始化GRU解码器
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出大小（词表大小）
            num_layers: GRU层数
            dropout: Dropout比例
            nhead: 多头注意力的头数
        """
        super(GRUDecoder, self).__init__()
        
        # 确保embedding的输出维度与hidden_size一致
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 编码GRU - 处理backbone输出的特征序列 [batch, num_channel, feature_map_dim]
        # 确保input_size与backbone输出的feature_map_dim一致
        self.encoder_gru = nn.GRU(
            input_size=input_size,  # 与backbone输出的feature_map_dim一致
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 使用双向GRU增强特征提取能力
        )
        
        # 编码GRU的多头自注意力机制
        # 确保embed_dim与双向GRU的输出维度一致
        encoder_dim = hidden_size * 2  # 双向GRU的输出维度是hidden_size*2
        self.encoder_attention = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 将编码GRU的输出映射到解码GRU的输入维度
        # 确保输入维度是双向GRU的输出维度(hidden_size*2)，输出维度与hidden_size一致
        self.encoder_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # 解码GRU - 用于自回归生成
        self.decoder_gru = nn.GRU(
            input_size=hidden_size,  # 输入是embedding的维度
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 解码GRU的多头自注意力机制
        self.decoder_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, features):
        """
        编码阶段 - 处理backbone输出的特征
        
        Args:
            features: backbone输出的特征，形状为 [batch_size, num_channel, feature_map_dim]
            
        Returns:
            nir_embedding: 编码后的NIR特征嵌入，形状为 [batch_size, hidden_size]
            encoder_hidden: 编码器最终隐藏状态，形状为 [num_layers, batch_size, hidden_size]
        """
        if features is None:
            # 防止NoneType错误
            raise ValueError("Encoder features cannot be None")
            
        batch_size = features.size(0)
        
        # 将每个channel的特征图作为时间步输入编码GRU
        # features: [batch_size, num_channel, feature_map_dim]
        encoder_outputs, encoder_hidden = self.encoder_gru(features)
        # encoder_outputs: [batch_size, num_channel, hidden_size*2]
        # encoder_hidden: [num_layers*2, batch_size, hidden_size]
        
        # 应用多头自注意力机制增强特征表示
        attn_output, _ = self.encoder_attention(
            encoder_outputs, encoder_outputs, encoder_outputs
        )
        # attn_output: [batch_size, num_channel, hidden_size*2]
        
        # 获取最终的NIR嵌入表示（使用注意力增强后的最后一个时间步）
        nir_embedding = attn_output[:, -1, :]  # [batch_size, hidden_size*2]
        
        # 投影到解码器维度
        nir_embedding = self.encoder_proj(nir_embedding)  # [batch_size, hidden_size]
        
        # 重组隐藏状态用于解码器初始化
        # 取双向GRU的前向和后向最后一层的隐藏状态
        num_layers = self.decoder_gru.num_layers
        forward_hidden = encoder_hidden[:num_layers, :, :]
        backward_hidden = encoder_hidden[num_layers:, :, :]
        # 合并前向和后向隐藏状态
        decoder_hidden = (forward_hidden + backward_hidden) / 2
        
        return nir_embedding, decoder_hidden
    
    def decode_step(self, input_token, hidden, nir_embedding):
        """
        单步解码 - 用于自回归生成
        
        Args:
            input_token: 输入token，形状为 [batch_size, 1]
            hidden: 上一步的隐藏状态，形状为 [num_layers, batch_size, hidden_size]
            nir_embedding: NIR特征嵌入，形状为 [batch_size, hidden_size]
            
        Returns:
            output: 输出概率分布，形状为 [batch_size, 1, output_size]
            hidden: 更新后的隐藏状态，形状为 [num_layers, batch_size, hidden_size]
        """
        # 嵌入输入token
        embedded = self.embedding(input_token)  # [batch_size, 1, hidden_size]
        
        # 应用dropout
        embedded = self.dropout(embedded)
        
        # 解码GRU前向传播
        decoder_output, hidden = self.decoder_gru(embedded, hidden)
        # decoder_output: [batch_size, 1, hidden_size]
        
        # 将NIR嵌入与当前解码状态结合，用于注意力计算
        # 扩展NIR嵌入维度以匹配decoder_output
        nir_embedding_expanded = nir_embedding.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # 应用多头自注意力机制，将解码状态与NIR嵌入进行交互
        attn_output, _ = self.decoder_attention(
            decoder_output, nir_embedding_expanded, nir_embedding_expanded
        )
        # attn_output: [batch_size, 1, hidden_size]
        
        # 生成输出概率分布
        output = self.fc(attn_output)  # [batch_size, 1, output_size]
        
        return output, hidden
    
    def forward(self, x, hidden=None, encoder_outputs=None, teacher_forcing_ratio=0.5):
        """
        前向传播 - 训练阶段
        
        Args:
            x: 目标序列，形状为 [batch_size, seq_len]
            hidden: 初始隐藏状态，默认为None
            encoder_outputs: backbone输出的特征，形状为 [batch_size, num_channel, feature_map_dim]
            teacher_forcing_ratio: 教师强制比例
            
        Returns:
            outputs: 输出序列，形状为 [batch_size, seq_len, output_size]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 编码阶段
        nir_embedding, hidden = self.encode(encoder_outputs)
        
        # 准备解码阶段的输出容器
        outputs = torch.zeros(batch_size, seq_len, self.output_size).to(x.device)
        
        # 初始输入为起始符号（通常是索引为0的token）
        input_token = x[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # 逐步解码
        for t in range(1, seq_len):
            # 单步解码
            output, hidden = self.decode_step(input_token, hidden, nir_embedding)
            
            # 保存输出
            outputs[:, t, :] = output.squeeze(1)
            
            # 决定下一步的输入：教师强制或使用模型预测
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            if teacher_force:
                # 使用真实目标作为下一步输入
                input_token = x[:, t].unsqueeze(1)
            else:
                # 使用模型预测作为下一步输入
                top1 = output.argmax(2)
                input_token = top1
        
        # 只返回outputs，不返回hidden状态
        return outputs


class LSTMDecoder(nn.Module):
    """
    LSTM解码器 - 包含编码LSTM和解码LSTM，使用多头自注意力机制
    """
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1, nhead=8):
        """
        初始化LSTM解码器
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出大小（词表大小）
            num_layers: LSTM层数
            dropout: Dropout比例
            nhead: 多头注意力的头数
        """
        super(LSTMDecoder, self).__init__()
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 编码LSTM - 处理backbone输出的特征序列 [batch, num_channel, feature_map_dim]
        # 确保input_size与backbone输出的feature_map_dim一致
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,  # 与backbone输出的feature_map_dim一致
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 使用双向LSTM增强特征提取能力
        )
        
        # 编码LSTM的多头自注意力机制
        # 确保embed_dim与双向LSTM的输出维度一致
        encoder_dim = hidden_size * 2  # 双向LSTM的输出维度是hidden_size*2
        self.encoder_attention = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 将编码LSTM的输出映射到解码LSTM的输入维度
        # 确保输入维度是双向LSTM的输出维度(hidden_size*2)，输出维度与hidden_size一致
        self.encoder_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # 解码LSTM - 用于自回归生成
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,  # 输入是embedding的维度
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 解码LSTM的多头自注意力机制
        self.decoder_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, features):
        """
        编码阶段 - 处理backbone输出的特征
        
        Args:
            features: backbone输出的特征，形状为 [batch_size, num_channel, feature_map_dim]
            
        Returns:
            nir_embedding: 编码后的NIR特征嵌入，形状为 [batch_size, hidden_size]
            encoder_hidden: 编码器最终隐藏状态，形状为 ([num_layers, batch_size, hidden_size], [num_layers, batch_size, hidden_size])
        """
        if features is None:
            # 防止NoneType错误
            raise ValueError("Encoder features cannot be None")
            
        batch_size = features.size(0)
        
        # 将每个channel的特征图作为时间步输入编码LSTM
        # features: [batch_size, num_channel, feature_map_dim]
        encoder_outputs, (h_n, c_n) = self.encoder_lstm(features)
        # encoder_outputs: [batch_size, num_channel, hidden_size*2]
        # h_n, c_n: [num_layers*2, batch_size, hidden_size]
        
        # 应用多头自注意力机制增强特征表示
        attn_output, _ = self.encoder_attention(
            encoder_outputs, encoder_outputs, encoder_outputs
        )
        # attn_output: [batch_size, num_channel, hidden_size*2]
        
        # 获取最终的NIR嵌入表示（使用注意力增强后的最后一个时间步）
        nir_embedding = attn_output[:, -1, :]  # [batch_size, hidden_size*2]
        
        # 投影到解码器维度
        nir_embedding = self.encoder_proj(nir_embedding)  # [batch_size, hidden_size]
        
        # 重组隐藏状态用于解码器初始化
        # 取双向LSTM的前向和后向最后一层的隐藏状态
        num_layers = self.decoder_lstm.num_layers
        
        # 处理h_n
        forward_h = h_n[:num_layers, :, :]
        backward_h = h_n[num_layers:, :, :]
        h_0 = (forward_h + backward_h) / 2
        
        # 处理c_n
        forward_c = c_n[:num_layers, :, :]
        backward_c = c_n[num_layers:, :, :]
        c_0 = (forward_c + backward_c) / 2
        
        decoder_hidden = (h_0, c_0)
        
        return nir_embedding, decoder_hidden
    
    def decode_step(self, input_token, hidden, nir_embedding):
        """
        单步解码 - 用于自回归生成
        
        Args:
            input_token: 输入token，形状为 [batch_size, 1]
            hidden: 上一步的隐藏状态，形状为 ([num_layers, batch_size, hidden_size], [num_layers, batch_size, hidden_size])
            nir_embedding: NIR特征嵌入，形状为 [batch_size, hidden_size]
            
        Returns:
            output: 输出概率分布，形状为 [batch_size, 1, output_size]
            hidden: 更新后的隐藏状态，形状为 ([num_layers, batch_size, hidden_size], [num_layers, batch_size, hidden_size])
        """
        # 嵌入输入token
        embedded = self.embedding(input_token)  # [batch_size, 1, hidden_size]
        
        # 应用dropout
        embedded = self.dropout(embedded)
        
        # 解码LSTM前向传播
        decoder_output, hidden = self.decoder_lstm(embedded, hidden)
        # decoder_output: [batch_size, 1, hidden_size]
        
        # 将NIR嵌入与当前解码状态结合，用于注意力计算
        # 扩展NIR嵌入维度以匹配decoder_output
        nir_embedding_expanded = nir_embedding.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # 应用多头自注意力机制，将解码状态与NIR嵌入进行交互
        attn_output, _ = self.decoder_attention(
            decoder_output, nir_embedding_expanded, nir_embedding_expanded
        )
        # attn_output: [batch_size, 1, hidden_size]
        
        # 生成输出概率分布
        output = self.fc(attn_output)  # [batch_size, 1, output_size]
        
        return output, hidden
    
    def forward(self, x, hidden=None, encoder_outputs=None, teacher_forcing_ratio=0.5):
        """
        前向传播 - 训练阶段
        
        Args:
            x: 目标序列，形状为 [batch_size, seq_len]
            hidden: 初始隐藏状态，默认为None
            encoder_outputs: backbone输出的特征，形状为 [batch_size, num_channel, feature_map_dim]
            teacher_forcing_ratio: 教师强制比例
            
        Returns:
            outputs: 输出序列，形状为 [batch_size, seq_len, output_size]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 编码阶段
        nir_embedding, hidden = self.encode(encoder_outputs)
        
        # 准备解码阶段的输出容器
        outputs = torch.zeros(batch_size, seq_len, self.output_size).to(x.device)
        
        # 初始输入为起始符号（通常是索引为0的token）
        input_token = x[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # 逐步解码
        for t in range(1, seq_len):
            # 单步解码
            output, hidden = self.decode_step(input_token, hidden, nir_embedding)
            
            # 保存输出
            outputs[:, t, :] = output.squeeze(1)
            
            # 决定下一步的输入：教师强制或使用模型预测
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            if teacher_force:
                # 使用真实目标作为下一步输入
                input_token = x[:, t].unsqueeze(1)
            else:
                # 使用模型预测作为下一步输入
                top1 = output.argmax(2)
                input_token = top1
        
        # 只返回outputs，不返回hidden状态
class TransformerDecoder(nn.Module):
    """
    Transformer解码器
    """
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, nhead=8, dropout=0.1):
        """
        初始化Transformer解码器
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            output_size: 输出大小（词表大小）
            num_layers: Transformer层数
            nhead: 多头注意力的头数
            dropout: Dropout比例
        """
        super(TransformerDecoder, self).__init__()
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        # 输入特征映射层 - 将backbone输出的特征映射到Transformer期望的维度
        # 确保input_size与backbone输出的feature_map_dim一致
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, features):
        """
        编码阶段 - 处理backbone输出的特征
        
        Args:
            features: backbone输出的特征，形状为 [batch_size, num_channel, feature_map_dim]
            
        Returns:
            memory: 编码后的特征，形状为 [batch_size, num_channel, hidden_size]
        """
        if features is None:
            # 防止NoneType错误
            raise ValueError("Encoder features cannot be None")
            
        # Project [batch_size, num_channel, feature_map_dim] to the
        # Transformer hidden size before entering the batch-first encoder.
        memory = self.input_proj(features)
        memory = self.transformer_encoder(memory)
        
        return memory
    
    def forward(self, x, hidden=None, encoder_outputs=None, teacher_forcing_ratio=0.5):
        """
        前向传播
        
        Args:
            x: 目标序列，形状为 [batch_size, seq_len]
            hidden: 初始隐藏状态，默认为None（Transformer不使用）
            encoder_outputs: backbone输出的特征，形状为 [batch_size, num_channel, feature_map_dim]
            teacher_forcing_ratio: 教师强制比例（Transformer不使用）
            
        Returns:
            outputs: 输出序列，形状为 [batch_size, seq_len, output_size]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 编码阶段 - 获取memory
        memory = self.encode(encoder_outputs)
        
        # 嵌入目标序列
        embedded = self.embedding(x) * math.sqrt(self.hidden_size)
        embedded = self.pos_encoder(embedded)
        
        # 创建目标掩码（防止看到未来信息）
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # 应用Transformer解码器
        output = self.transformer_decoder(
            tgt=embedded,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # 生成输出概率分布
        output = self.fc(output)
        
        # 只返回outputs
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """
        生成方形后续掩码（用于防止解码器看到未来信息）
        
        Args:
            sz: 序列长度
            
        Returns:
            mask: 掩码张量，形状为 [sz, sz]
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            dropout: Dropout比例
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]
            
        Returns:
            输出张量，形状为 [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
