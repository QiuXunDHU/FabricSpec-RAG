"""
评价指标记录器
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional


class MetricsLogger:
    """
    评价指标记录器，用于记录训练过程中的评价指标
    """
    
    def __init__(self, save_dir: str, experiment_id: Optional[str] = None):
        """
        初始化评价指标记录器
        
        Args:
            save_dir: 保存目录
            experiment_id: 实验ID，如果不提供则使用时间戳
        """
        self.save_dir = save_dir
        self.experiment_id = experiment_id or f"metrics_{time.strftime('%Y%m%d-%H%M%S')}"
        self.metrics_history = {
            "train": {},
            "valid": {}
        }
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
    def update(self, metrics: Dict[str, float], phase: str = "train", epoch: int = 0) -> None:
        """
        更新评价指标
        
        Args:
            metrics: 评价指标字典
            phase: 阶段，train或valid
            epoch: 当前epoch
        """
        if phase not in self.metrics_history:
            self.metrics_history[phase] = {}
            
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history[phase]:
                self.metrics_history[phase][metric_name] = []
                
            # 确保列表长度与epoch匹配
            while len(self.metrics_history[phase][metric_name]) <= epoch:
                self.metrics_history[phase][metric_name].append(None)
                
            self.metrics_history[phase][metric_name][epoch] = value
    
    def save(self) -> str:
        """
        保存评价指标历史
        
        Returns:
            保存路径
        """
        save_path = os.path.join(self.save_dir, f"{self.experiment_id}.json")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
            
        return save_path
    
    def load(self, file_path: str) -> None:
        """
        加载评价指标历史
        
        Args:
            file_path: 文件路径
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"评价指标历史文件不存在: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            self.metrics_history = json.load(f)
    
    def plot(self, metric_names: List[str], save_dir: Optional[str] = None) -> List[str]:
        """
        绘制评价指标历史曲线
        
        Args:
            metric_names: 要绘制的指标名称列表
            save_dir: 保存目录，默认为self.save_dir
            
        Returns:
            保存的图像文件路径列表
        """
        save_dir = save_dir or self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        saved_paths = []
        
        # 设置matplotlib样式
        plt.style.use('default')
        
        for metric_name in metric_names:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for phase in ['train', 'valid']:
                if phase in self.metrics_history and metric_name in self.metrics_history[phase]:
                    values = self.metrics_history[phase][metric_name]
                    epochs = np.arange(len(values))
                    
                    # 过滤None值
                    valid_indices = [i for i, v in enumerate(values) if v is not None]
                    valid_epochs = [epochs[i] for i in valid_indices]
                    valid_values = [values[i] for i in valid_indices]
                    
                    if valid_values:
                        ax.plot(valid_epochs, valid_values, marker='o', linestyle='-', 
                                label=f"{phase.capitalize()}")
            
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f"{metric_name.replace('_', ' ').title()} vs. Epoch")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 保存图像
            save_path = os.path.join(save_dir, f"{self.experiment_id}_{metric_name}.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
            
            saved_paths.append(save_path)
            
        return saved_paths
    
    def get_best_epoch(self, metric_name: str, phase: str = "valid", higher_better: bool = True) -> int:
        """
        获取最佳epoch
        
        Args:
            metric_name: 指标名称
            phase: 阶段，train或valid
            higher_better: 是否越高越好
            
        Returns:
            最佳epoch
        """
        if phase not in self.metrics_history or metric_name not in self.metrics_history[phase]:
            return 0
            
        values = self.metrics_history[phase][metric_name]
        valid_values = [(i, v) for i, v in enumerate(values) if v is not None]
        
        if not valid_values:
            return 0
            
        if higher_better:
            return max(valid_values, key=lambda x: x[1])[0]
        else:
            return min(valid_values, key=lambda x: x[1])[0]
    
    def get_latest_metrics(self, phase: str = "valid") -> Dict[str, float]:
        """
        获取最新的评价指标
        
        Args:
            phase: 阶段，train或valid
            
        Returns:
            最新的评价指标字典
        """
        if phase not in self.metrics_history:
            return {}
            
        latest_metrics = {}
        for metric_name, values in self.metrics_history[phase].items():
            valid_values = [v for v in values if v is not None]
            if valid_values:
                latest_metrics[metric_name] = valid_values[-1]
                
        return latest_metrics
