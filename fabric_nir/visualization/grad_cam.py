"""
Grad-CAM分析模块
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple, Callable


class GradCAM:
    """
    Grad-CAM实现
    """
    
    def __init__(self, model, target_layer):
        """
        初始化Grad-CAM
        
        Args:
            model: PyTorch模型
            target_layer: 目标层
        """
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        
        # 注册钩子
        self.register_hooks()
        
        # 存储特征和梯度
        self.feature_maps = None
        self.gradients = None
    
    def register_hooks(self):
        """
        注册前向和反向钩子
        """
        # 前向钩子
        def forward_hook(module, input, output):
            self.feature_maps = output
        
        # 反向钩子
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # 注册钩子
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_backward_hook(backward_hook)
        
        # 保存钩子句柄
        self.hooks = [forward_handle, backward_handle]
    
    def remove_hooks(self):
        """
        移除钩子
        """
        for hook in self.hooks:
            hook.remove()
    
    def __call__(self, x, class_idx=None):
        """
        计算Grad-CAM
        
        Args:
            x: 输入张量，形状为 [batch_size, channels, height, width]
            class_idx: 类别索引（可选）
            
        Returns:
            cam: Grad-CAM热力图，形状为 [batch_size, height, width]
        """
        # 确保模型处于评估模式
        self.model.eval()
        
        # 前向传播
        output = self.model(x)
        
        # 如果是元组，取第一个元素
        if isinstance(output, tuple):
            output = output[0]
        
        # 如果没有指定类别索引，使用预测类别
        batch_size = x.size(0)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        
        # 初始化梯度
        self.model.zero_grad()
        
        # 计算梯度
        one_hot = torch.zeros_like(output)
        for i in range(batch_size):
            one_hot[i, class_idx[i]] = 1
        
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # 计算Grad-CAM
        cam = torch.sum(weights * self.feature_maps, dim=1)
        cam = F.relu(cam)
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam


def visualize_gradcam(
    model: torch.nn.Module,
    data: Union[np.ndarray, torch.Tensor],
    target_layer: torch.nn.Module,
    class_idx: Optional[Union[int, List[int]]] = None,
    save_path: str = "gradcam_analysis.png",
    title: str = "Grad-CAM Analysis",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300
):
    """
    可视化Grad-CAM结果
    
    Args:
        model: PyTorch模型
        data: 输入数据，形状为 [batch_size, channels, height, width]
        target_layer: 目标层
        class_idx: 类别索引（可选）
        save_path: 保存路径
        title: 图表标题
        figsize: 图表大小
        dpi: 图表分辨率
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # 设置matplotlib样式
    plt.style.use('default')
    
    # 转换为PyTorch张量
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    
    # 创建Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # 计算Grad-CAM
    cam = grad_cam(data, class_idx)
    
    # 移除钩子
    grad_cam.remove_hooks()
    
    # 转换为numpy数组
    cam = cam.detach().cpu().numpy()
    
    # 创建图表
    batch_size = data.size(0)
    fig, axes = plt.subplots(1, batch_size, figsize=figsize)
    
    # 如果只有一个样本，确保axes是列表
    if batch_size == 1:
        axes = [axes]
    
    # 绘制每个样本的Grad-CAM
    for i in range(batch_size):
        # 获取输入数据
        input_data = data[i].detach().cpu().numpy()
        
        # 如果是1D数据，转换为2D
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        # 绘制输入数据
        axes[i].imshow(input_data, cmap='gray')
        
        # 叠加Grad-CAM
        axes[i].imshow(cam[i], cmap='jet', alpha=0.5)
        
        # 设置标题
        if class_idx is not None and isinstance(class_idx, list):
            axes[i].set_title(f"Class: {class_idx[i]}")
        elif class_idx is not None:
            axes[i].set_title(f"Class: {class_idx}")
        else:
            axes[i].set_title(f"Sample {i+1}")
        
        # 移除坐标轴
        axes[i].axis('off')
    
    # 设置总标题
    fig.suptitle(title)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    
    return save_path
