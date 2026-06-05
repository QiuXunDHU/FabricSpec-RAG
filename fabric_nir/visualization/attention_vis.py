"""
注意力可视化模块
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Optional, Union, Tuple


def visualize_attention(
    attention_weights: Union[np.ndarray, torch.Tensor],
    input_tokens: Optional[list] = None,
    output_tokens: Optional[list] = None,
    save_path: str = "attention_visualization.png",
    title: str = "Attention Weights Visualization",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    cmap: str = "viridis"
):
    """
    可视化注意力权重
    
    Args:
        attention_weights: 注意力权重矩阵，形状为 [output_len, input_len]
        input_tokens: 输入token列表（可选）
        output_tokens: 输出token列表（可选）
        save_path: 保存路径
        title: 图表标题
        figsize: 图表大小
        dpi: 图表分辨率
        cmap: 颜色映射
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # 转换为numpy数组
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # 设置matplotlib样式
    plt.style.use('default')
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制热力图
    plt.imshow(attention_weights, cmap=cmap)
    
    # 设置标题
    plt.title(title)
    
    # 设置坐标轴标签
    if input_tokens:
        plt.xticks(np.arange(len(input_tokens)), input_tokens, rotation=45, ha='right')
    else:
        plt.xlabel("Input Position")
    
    if output_tokens:
        plt.yticks(np.arange(len(output_tokens)), output_tokens)
    else:
        plt.ylabel("Output Position")
    
    # 添加颜色条
    plt.colorbar(label='Attention Weight')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    
    return save_path
