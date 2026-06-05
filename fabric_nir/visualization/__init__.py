"""
可视化模块 - 集成接口
"""

import os
from typing import List, Optional, Union, Tuple, Dict, Any

from .embedding_vis import visualize_embedding_tsne
from .attention_vis import visualize_attention
from .shap_analysis import shap_analysis
from .grad_cam import visualize_gradcam


class Visualizer:
    """
    可视化工具集成接口
    """
    
    def __init__(self, save_dir="results/visualizations"):
        """
        初始化可视化工具
        
        Args:
            save_dir: 保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_embedding(self, embeddings, labels=None, name="embedding", **kwargs):
        """
        可视化嵌入向量
        
        Args:
            embeddings: 嵌入向量
            labels: 标签
            name: 保存文件名前缀
            **kwargs: 其他参数
            
        Returns:
            save_path: 保存路径
        """
        save_path = os.path.join(self.save_dir, f"{name}_tsne.png")
        return visualize_embedding_tsne(
            embeddings=embeddings,
            labels=labels,
            save_path=save_path,
            **kwargs
        )
    
    def visualize_attention(self, attention_weights, input_tokens=None, output_tokens=None, name="attention", **kwargs):
        """
        可视化注意力权重
        
        Args:
            attention_weights: 注意力权重
            input_tokens: 输入token
            output_tokens: 输出token
            name: 保存文件名前缀
            **kwargs: 其他参数
            
        Returns:
            save_path: 保存路径
        """
        save_path = os.path.join(self.save_dir, f"{name}_attention.png")
        return visualize_attention(
            attention_weights=attention_weights,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            save_path=save_path,
            **kwargs
        )
    
    def visualize_shap(self, model, data, feature_names=None, name="shap", **kwargs):
        """
        可视化SHAP分析
        
        Args:
            model: 模型
            data: 数据
            feature_names: 特征名称
            name: 保存文件名前缀
            **kwargs: 其他参数
            
        Returns:
            save_path: 保存路径
        """
        save_path = os.path.join(self.save_dir, f"{name}_shap.png")
        return shap_analysis(
            model=model,
            data=data,
            feature_names=feature_names,
            save_path=save_path,
            **kwargs
        )
    
    def visualize_gradcam(self, model, data, target_layer, class_idx=None, name="gradcam", **kwargs):
        """
        可视化Grad-CAM分析
        
        Args:
            model: 模型
            data: 数据
            target_layer: 目标层
            class_idx: 类别索引
            name: 保存文件名前缀
            **kwargs: 其他参数
            
        Returns:
            save_path: 保存路径
        """
        save_path = os.path.join(self.save_dir, f"{name}_gradcam.png")
        return visualize_gradcam(
            model=model,
            data=data,
            target_layer=target_layer,
            class_idx=class_idx,
            save_path=save_path,
            **kwargs
        )


# 导出接口
__all__ = [
    "Visualizer",
    "visualize_embedding_tsne",
    "visualize_attention",
    "shap_analysis",
    "visualize_gradcam"
]
