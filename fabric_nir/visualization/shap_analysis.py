"""
SHAP分析模块
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap
from typing import List, Optional, Union, Tuple, Callable


def shap_analysis(
    model: torch.nn.Module,
    data: Union[np.ndarray, torch.Tensor],
    feature_names: Optional[List[str]] = None,
    save_path: str = "shap_analysis.png",
    title: str = "SHAP Analysis",
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    n_samples: int = 100,
    model_output_transform: Optional[Callable] = None
):
    """
    使用SHAP分析模型特征重要性
    
    Args:
        model: PyTorch模型
        data: 输入数据，形状为 [n_samples, n_features]
        feature_names: 特征名称列表（可选）
        save_path: 保存路径
        title: 图表标题
        figsize: 图表大小
        dpi: 图表分辨率
        n_samples: SHAP分析样本数量
        model_output_transform: 模型输出转换函数（可选）
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # 设置matplotlib样式
    plt.style.use('default')
    
    # 转换为numpy数组
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    # 限制样本数量
    if data.shape[0] > n_samples:
        indices = np.random.choice(data.shape[0], n_samples, replace=False)
        data = data[indices]
    
    # 创建模型包装器
    def model_wrapper(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        if len(x_tensor.shape) == 2:
            x_tensor = x_tensor.unsqueeze(1)  # 添加通道维度
        
        with torch.no_grad():
            output = model(x_tensor)
            
            if model_output_transform is not None:
                output = model_output_transform(output)
            
            if isinstance(output, tuple):
                output = output[0]  # 取第一个输出
            
            if isinstance(output, torch.Tensor):
                output = output.detach().cpu().numpy()
        
        return output
    
    # 创建SHAP解释器
    explainer = shap.KernelExplainer(model_wrapper, data)
    
    # 计算SHAP值
    shap_values = explainer.shap_values(data)
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制SHAP摘要图
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(data.shape[1])]
    
    shap.summary_plot(
        shap_values,
        data,
        feature_names=feature_names,
        show=False
    )
    
    # 设置标题
    plt.title(title)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    
    return save_path
