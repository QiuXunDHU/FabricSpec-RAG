"""
CBAM attention modules for one-dimensional spectral features.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention block."""

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()

        hidden_channels = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x).squeeze(-1)

        if avg_out.size(1) != self.fc[0].in_features:
            device = avg_out.device
            in_channels = avg_out.size(1)
            hidden_channels = max(1, in_channels // 16)
            self.fc = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, in_channels),
            ).to(device)

        avg_out = self.fc(avg_out).unsqueeze(-1)
        max_out = self.fc(self.max_pool(x).squeeze(-1)).unsqueeze(-1)
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention block."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(out))


class CBAM(nn.Module):
    """Convolutional Block Attention Module for [batch, channels, length]."""

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x_reshaped = x.permute(0, 2, 1)
        x_reshaped = self.channel_attention(x_reshaped)
        x_reshaped = self.spatial_attention(x_reshaped)
        return x_reshaped.permute(0, 2, 1)
