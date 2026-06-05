"""
实验管理模块
"""

import os
import yaml
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional


class ExperimentManager:
    """
    实验管理器
    管理消融实验的配置、运行和结果记录
    """
    
    def __init__(self, matrix_config_path: str, results_dir: str = "results"):
        """
        初始化实验管理器
        
        Args:
            matrix_config_path: 实验矩阵配置文件路径
            results_dir: 结果保存目录
        """
        self.matrix_config_path = matrix_config_path
        self.results_dir = results_dir
        
        # 加载实验矩阵配置
        with open(matrix_config_path, 'r') as f:
            self.matrix_config = yaml.safe_load(f)
        
        # 创建结果目录
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "visualizations"), exist_ok=True)
        
        # 初始化实验记录
        self.experiments = []
        self.load_experiments()
    
    def load_experiments(self) -> None:
        """
        加载已有实验记录
        """
        record_path = os.path.join(self.results_dir, "experiment_records.json")
        if os.path.exists(record_path):
            with open(record_path, 'r') as f:
                self.experiments = json.load(f)
    
    def save_experiments(self) -> None:
        """
        保存实验记录
        """
        record_path = os.path.join(self.results_dir, "experiment_records.json")
        with open(record_path, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def generate_experiment_id(self) -> str:
        """
        生成实验ID
        
        Returns:
            experiment_id: 实验ID
        """
        # 使用时间戳和实验数量生成ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        count = len(self.experiments) + 1
        return f"exp_{timestamp}_{count:03d}"
    
    def register_experiment(self, config: Dict[str, Any], experiment_id: Optional[str] = None) -> str:
        """
        注册实验
        
        Args:
            config: 实验配置
            experiment_id: 实验ID（可选）
            
        Returns:
            experiment_id: 实验ID
        """
        if experiment_id is None:
            experiment_id = self.generate_experiment_id()
        
        # 创建实验记录
        experiment = {
            "id": experiment_id,
            "config": config,
            "status": "registered",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metrics": {}
        }
        
        # 添加到实验列表
        self.experiments.append(experiment)
        
        # 保存实验记录
        self.save_experiments()
        
        return experiment_id
    
    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        """
        更新实验状态
        
        Args:
            experiment_id: 实验ID
            status: 实验状态
        """
        for experiment in self.experiments:
            if experiment["id"] == experiment_id:
                experiment["status"] = status
                experiment["updated_at"] = datetime.now().isoformat()
                break
        
        # 保存实验记录
        self.save_experiments()
    
    def update_experiment_metrics(self, experiment_id: str, metrics: Dict[str, Any]) -> None:
        """
        更新实验指标
        
        Args:
            experiment_id: 实验ID
            metrics: 实验指标
        """
        for experiment in self.experiments:
            if experiment["id"] == experiment_id:
                experiment["metrics"] = metrics
                experiment["updated_at"] = datetime.now().isoformat()
                break
        
        # 保存实验记录
        self.save_experiments()
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        获取实验记录
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            experiment: 实验记录
        """
        for experiment in self.experiments:
            if experiment["id"] == experiment_id:
                return experiment
        
        return None
    
    def get_experiments_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        获取指定状态的实验记录
        
        Args:
            status: 实验状态
            
        Returns:
            experiments: 实验记录列表
        """
        return [experiment for experiment in self.experiments if experiment["status"] == status]
    
    def get_best_experiment(self, metric: str = "overall_score") -> Optional[Dict[str, Any]]:
        """
        获取最佳实验记录
        
        Args:
            metric: 评价指标
            
        Returns:
            experiment: 实验记录
        """
        completed_experiments = self.get_experiments_by_status("completed")
        
        if not completed_experiments:
            return None
        
        # 按指标排序
        sorted_experiments = sorted(
            completed_experiments,
            key=lambda x: x["metrics"].get(metric, 0),
            reverse=True
        )
        
        return sorted_experiments[0] if sorted_experiments else None
    
    def export_results(self, path: Optional[str] = None) -> str:
        """
        导出实验结果
        
        Args:
            path: 导出路径（可选）
            
        Returns:
            path: 导出路径
        """
        if path is None:
            path = os.path.join(self.results_dir, "experiment_results.csv")
        
        # 提取实验结果
        results = []
        for experiment in self.experiments:
            result = {
                "id": experiment["id"],
                "status": experiment["status"],
                "created_at": experiment["created_at"],
                "updated_at": experiment["updated_at"]
            }
            
            # 添加配置
            for key, value in experiment["config"].items():
                result[f"config_{key}"] = value
            
            # 添加指标
            for key, value in experiment["metrics"].items():
                result[f"metric_{key}"] = value
            
            results.append(result)
        
        # 创建DataFrame
        df = pd.DataFrame(results)
        
        # 保存CSV
        df.to_csv(path, index=False)
        
        return path
