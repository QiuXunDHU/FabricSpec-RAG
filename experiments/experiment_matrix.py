"""
实验矩阵管理与日志记录模块
"""

import os
import json
import yaml
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ExperimentMatrixLogger:
    """
    实验矩阵管理与日志记录器
    用于管理消融实验矩阵、记录实验结果并生成比较报告
    """
    
    def __init__(self, matrix_config_path: str, results_dir: str):
        """
        初始化实验矩阵管理器
        
        Args:
            matrix_config_path: 实验矩阵配置文件路径
            results_dir: 结果保存目录
        """
        self.matrix_config_path = matrix_config_path
        self.results_dir = results_dir
        self.matrix_config = self._load_matrix_config(matrix_config_path)
        
        # 确保结果目录存在
        os.makedirs(results_dir, exist_ok=True)
        
    def _load_matrix_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载实验矩阵配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"实验矩阵配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def get_experiment_results(self) -> List[Dict[str, Any]]:
        """
        获取所有实验结果
        
        Returns:
            实验结果列表
        """
        results = []
        
        # 遍历结果目录中的所有JSON文件
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.results_dir, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    results.append(result)
                    
        return results
    
    def generate_comparison_table(self, metric_keys: Optional[List[str]] = None) -> pd.DataFrame:
        """
        生成实验比较表格
        
        Args:
            metric_keys: 要比较的指标键列表
            
        Returns:
            比较表格DataFrame
        """
        # 默认比较指标
        if metric_keys is None:
            metric_keys = ["overall_score", "component_f1", "ratio_r2", "joint_accuracy", "bleu"]
            
        # 获取所有实验结果
        results = self.get_experiment_results()
        
        if not results:
            return pd.DataFrame()
            
        # 构建比较表格
        data = []
        
        for result in results:
            row = {}
            
            # 添加实验配置
            for key, value in result.get("config", {}).items():
                if key != "experiment_id":
                    row[key] = value
                    
            # 添加实验ID
            row["experiment_id"] = result.get("experiment_id", "unknown")
            
            # 添加指标
            for key in metric_keys:
                if "metrics" in result and key in result["metrics"]:
                    row[key] = result["metrics"][key]
                else:
                    row[key] = None
                    
            data.append(row)
            
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 按overall_score排序（如果存在）
        if "overall_score" in df.columns:
            df = df.sort_values("overall_score", ascending=False)
            
        return df
    
    def save_comparison_table(self, output_path: str, metric_keys: Optional[List[str]] = None) -> None:
        """
        保存实验比较表格
        
        Args:
            output_path: 输出文件路径
            metric_keys: 要比较的指标键列表
        """
        df = self.generate_comparison_table(metric_keys)
        
        if df.empty:
            print("No experiment results found")
            return
            
        # 保存为CSV
        df.to_csv(output_path, index=False)
        print(f"Comparison table saved to {output_path}")
    
    def plot_comparison_heatmap(self, output_path: str, metric_key: str = "overall_score") -> None:
        """
        绘制实验比较热力图
        
        Args:
            output_path: 输出文件路径
            metric_key: 要比较的指标键
        """
        # 获取所有实验结果
        results = self.get_experiment_results()
        
        if not results:
            print("No experiment results found")
            return
            
        # 提取维度和值
        dimensions = self.matrix_config.get("dimensions", {})
        if len(dimensions) < 2:
            print("Need at least 2 dimensions for heatmap")
            return
            
        # 选择前两个维度作为热力图的x和y轴
        dim_keys = list(dimensions.keys())
        x_dim, y_dim = dim_keys[0], dim_keys[1]
        
        # 构建热力图数据
        heatmap_data = {}
        for result in results:
            config = result.get("config", {})
            metrics = result.get("metrics", {})
            
            if x_dim in config and y_dim in config and metric_key in metrics:
                x_val = config[x_dim]
                y_val = config[y_dim]
                
                if y_val not in heatmap_data:
                    heatmap_data[y_val] = {}
                    
                heatmap_data[y_val][x_val] = metrics[metric_key]
                
        # 转换为DataFrame
        df = pd.DataFrame(heatmap_data).T
        
        # 绘制热力图
        plt.style.use('default')
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(df, annot=True, cmap="viridis", fmt=".3f", linewidths=.5)
        
        plt.title(f"Comparison of {metric_key}")
        plt.xlabel(x_dim.replace('_', ' ').title())
        plt.ylabel(y_dim.replace('_', ' ').title())
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Comparison heatmap saved to {output_path}")
    
    def generate_summary_report(self, output_path: str) -> None:
        """
        生成实验总结报告
        
        Args:
            output_path: 输出文件路径
        """
        # 获取所有实验结果
        results = self.get_experiment_results()
        
        if not results:
            print("No experiment results found")
            return
            
        # 生成比较表格
        df = self.generate_comparison_table()
        
        # 找出每个指标的最佳实验
        metric_keys = ["overall_score", "component_f1", "ratio_r2", "joint_accuracy", "bleu"]
        best_experiments = {}
        
        for metric in metric_keys:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                if best_idx is not None:
                    best_experiments[metric] = df.loc[best_idx].to_dict()
        
        # 生成报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Ablation Experiments Summary Report\n\n")
            
            f.write("## Experiment Matrix\n\n")
            f.write(f"Total experiments: {len(results)}\n\n")
            
            f.write("## Best Configurations\n\n")
            
            for metric, exp in best_experiments.items():
                f.write(f"### Best for {metric}: {exp.get(metric, 'N/A')}\n\n")
                f.write("Configuration:\n")
                
                for key, value in exp.items():
                    if key not in metric_keys and key != "experiment_id":
                        f.write(f"- {key}: {value}\n")
                        
                f.write("\n")
            
            f.write("## Dimension Analysis\n\n")
            
            # 分析各维度的影响
            dimensions = self.matrix_config.get("dimensions", {})
            
            for dim_name, dim_values in dimensions.items():
                f.write(f"### Impact of {dim_name}\n\n")
                
                # 计算每个维度值的平均性能
                dim_performance = {}
                
                for value in dim_values:
                    filtered_df = df[df[dim_name] == value]
                    
                    if not filtered_df.empty:
                        for metric in metric_keys:
                            if metric in filtered_df.columns:
                                key = f"{dim_name}_{value}_{metric}"
                                dim_performance[key] = filtered_df[metric].mean()
                                
                                f.write(f"- {value}: Average {metric} = {dim_performance[key]:.4f}\n")
                                
                f.write("\n")
            
            f.write("## Conclusion\n\n")
            f.write("Based on the experimental results, the recommended configuration is:\n\n")
            
            # 使用overall_score最高的配置作为推荐
            if "overall_score" in best_experiments:
                best_overall = best_experiments["overall_score"]
                
                for key, value in best_overall.items():
                    if key in dimensions:
                        f.write(f"- {key}: {value}\n")
                        
            f.write("\n")
            
        print(f"Summary report saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate experiment matrix reports")
    parser.add_argument("--matrix", type=str, default="configs/experiment_matrix.yaml", 
                        help="Path to experiment matrix config")
    parser.add_argument("--results", type=str, default="results/metrics", 
                        help="Directory with experiment results")
    parser.add_argument("--output", type=str, default="results", 
                        help="Output directory for reports")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 初始化实验矩阵管理器
    logger = ExperimentMatrixLogger(args.matrix, args.results)
    
    # 生成比较表格
    logger.save_comparison_table(os.path.join(args.output, "experiment_comparison.csv"))
    
    # 生成热力图
    logger.plot_comparison_heatmap(os.path.join(args.output, "comparison_heatmap.png"))
    
    # 生成总结报告
    logger.generate_summary_report(os.path.join(args.output, "experiment_summary.md"))
