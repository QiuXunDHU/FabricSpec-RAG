"""
可视化模块 - 嵌入可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import List, Optional, Union, Tuple


def visualize_embedding_tsne(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: str = "embedding_tsne.png",
    title: str = "Embedding TSNE Visualization",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42
):
    """
    使用TSNE可视化嵌入向量
    
    Args:
        embeddings: 嵌入向量数组，形状为 [n_samples, n_features]
        labels: 样本标签列表（可选）
        save_path: 保存路径
        title: 图表标题
        figsize: 图表大小
        dpi: 图表分辨率
        perplexity: TSNE参数
        n_iter: TSNE迭代次数
        random_state: 随机种子
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # 设置matplotlib样式
    plt.style.use('default')
    
    # 使用TSNE降维
    # 自适应调整perplexity参数，确保小于样本数量
    n_samples = embeddings.shape[0]
    adaptive_perplexity = min(perplexity, max(5, n_samples // 2 - 1))  # 确保perplexity小于n_samples
    print(f"自适应调整TSNE perplexity参数: {adaptive_perplexity} (原始值: {perplexity}, 样本数: {n_samples})")
    
    # 检查并处理embeddings维度
    if len(embeddings.shape) > 2:
        print(f"原始embeddings形状: {embeddings.shape}，进行降维处理")
        if len(embeddings.shape) == 3:
            # 对第二维取均值，保留batch维度和特征维度
            embeddings = np.mean(embeddings, axis=1)
        else:
            # 如果维度更高，先展平到2D
            batch_size = embeddings.shape[0]
            embeddings = embeddings.reshape(batch_size, -1)
        print(f"处理后embeddings形状: {embeddings.shape}")
    
    tsne = TSNE(n_components=2, perplexity=adaptive_perplexity, n_iter=n_iter, random_state=random_state)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制散点图
    if labels is not None:
        # 获取唯一标签
        unique_labels = list(set(labels))
        
        # 为每个标签分配颜色
        for label in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(
                embeddings_2d[indices, 0],
                embeddings_2d[indices, 1],
                label=label,
                alpha=0.7
            )
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    # 设置标题和轴标签
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    
    return save_path
