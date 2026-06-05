"""
实验运行模块 - 消融实验自动化系统
"""

import os
import yaml
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from pathlib import Path

from fabric_nir.utils.config import ConfigManager
from fabric_nir.train.single_task_trainer import SingleTaskTrainer
from fabric_nir.train.multi_task_trainer import MultiTaskTrainer

import numpy as np
def run_ablation_experiments(matrix_config_path, results_dir="results/metrics", epochs=2, device="cuda"):
    """
    运行所有消融实验
    
    Args:
        matrix_config_path: 实验矩阵配置文件路径
        results_dir: 结果保存目录
        epochs: 训练轮数
        device: 计算设备
    """
    # 加载实验矩阵配置
    with open(matrix_config_path, 'r',encoding='utf-8') as f:
        matrix_config = yaml.safe_load(f)
    
    # 获取实验维度
    dimensions = matrix_config.get("dimensions", {})
    
    # 创建实验组合
    experiment_combinations = []
    dimension_names = []
    dimension_values = []
    
    for dim_name, dim_values in dimensions.items():
        dimension_names.append(dim_name)
        dimension_values.append(dim_values)
    
    # 生成所有组合
    for combination in itertools.product(*dimension_values):
        experiment = dict(zip(dimension_names, combination))
        experiment_combinations.append(experiment)
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 运行每个实验
    results = []
    for i, experiment in enumerate(experiment_combinations):
        # 创建实验ID
        experiment_id = f"experiment_{i+1:03d}"
        
        # 打印实验信息
        print(f"\n{'='*80}")
        print(f"Running experiment {experiment_id}:")
        for dim_name, dim_value in experiment.items():
            print(f"  {dim_name}: {dim_value}")
        print(f"{'='*80}\n")
        
        # 运行实验
        try:
            metrics = run_single_experiment(
                experiment_id=experiment_id,
                experiment=experiment,
                matrix_config=matrix_config,
                results_dir=results_dir,
                epochs=epochs,
                device=device
            )
            # 添加实验结果
            result = {
                "experiment_id": experiment_id,
                **experiment,
                **metrics
            }
            results.append(result)
        except Exception as e:
            print(metrics)
            print(f"Error in experiment {experiment_id}: {e}")
    
    # 保存所有实验结果
    results_df = pd.DataFrame(results)
    results_path = os.path.join(results_dir, "ablation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved all experiment results to {results_path}")
    
    # 比较实验结果
    compare_experiments(matrix_config_path, results_dir)


def run_single_experiment(experiment_id, experiment=None, matrix_config_path=None, matrix_config=None, results_dir="results/metrics", epochs=2, device="cuda"):
    """
    运行单个实验
    
    Args:
        experiment_id: 实验ID
        experiment: 实验配置字典（可选）
        matrix_config_path: 实验矩阵配置文件路径（可选）
        matrix_config: 实验矩阵配置（可选）
        results_dir: 结果保存目录
        epochs: 训练轮数
        device: 计算设备
        
    Returns:
        metrics: 评价指标字典
    """
    # 加载实验矩阵配置
    if matrix_config is None:
        if matrix_config_path is None:
            raise ValueError("Either matrix_config or matrix_config_path must be provided")
        
        with open(matrix_config_path, 'r',encoding="utf-8") as f:
            matrix_config = yaml.safe_load(f)
    
    # 如果没有提供实验配置，使用默认实验
    if experiment is None:
        experiment = matrix_config.get("default_experiment", {})
    
    # 获取基础配置路径
    base_config_path = matrix_config.get("base_config", "configs/base_config.yaml")
    # 确保路径正确，避免重复添加configs前缀
    if not os.path.exists(base_config_path) and not os.path.isabs(base_config_path):
        # 如果路径不存在且不是绝对路径，尝试修正
        if "configs/" in base_config_path:
            # 已经包含configs前缀，不需要再添加
            pass
        else:
            # 添加configs前缀
            base_config_path = os.path.join("configs", base_config_path)
    
    print(f"使用基础配置文件: {base_config_path}")
    
    # 创建配置管理器
    config_manager = ConfigManager(base_config_path)
    
    # 应用实验配置
    for dim_name, dim_value in experiment.items():
        if dim_name == "decoder_type":
            config_manager.config["model"]["decoder"]["type"] = dim_value
        elif dim_name == "task_architecture":
            config_manager.config["training"]["task_type"] = dim_value
        elif dim_name == "initialization_method":
            if dim_value == "pretrained":
                config_manager.config["model"]["pretrained"]["use_pretrained"] = True
            else:
                config_manager.config["model"]["pretrained"]["use_pretrained"] = False
        elif dim_name == "attention_mechanism":
            if dim_value == "with_cbam":
                config_manager.config["model"]["attention"]["use_cbam"] = True
            else:
                config_manager.config["model"]["attention"]["use_cbam"] = False
        elif dim_name == "backbone_type":
            config_manager.config["model"]["backbone"]["type"] = dim_value
    
    # 设置结果目录
    config_manager.config["results_dir"] = results_dir
    
    # 获取任务类型
    task_type = config_manager.config["training"]["task_type"]
    
    # 创建训练器
    if task_type == "single_task":
        trainer = SingleTaskTrainer(config_manager, experiment_id)
    else:
        trainer = MultiTaskTrainer(config_manager, experiment_id)
    
    # 训练模型
    metrics = trainer.train(epochs=epochs, device=device)
    
    # 测试模型
    test_metrics = trainer.test(device=device)
    
    # 合并指标
    metrics.update(test_metrics)
    
    return metrics


def compare_experiments(matrix_config_path, results_dir="results/metrics"):
    """
    比较实验结果
    
    Args:
        matrix_config_path: 实验矩阵配置文件路径
        results_dir: 结果保存目录
    """
    # 加载实验矩阵配置
    with open(matrix_config_path, 'r',encoding="utf-8") as f:
        matrix_config = yaml.safe_load(f)
    
    # 获取实验维度
    dimensions = matrix_config.get("dimensions", {})
    
    # 加载实验结果
    results_path = os.path.join(results_dir, "ablation_results.csv")
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    results_df = pd.read_csv(results_path)
    
    # 创建可视化目录
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 设置matplotlib样式
    plt.style.use('default')
    
    # 比较每个维度
    for dim_name, dim_values in dimensions.items():
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 分组并计算平均值
        grouped = results_df.groupby(dim_name)
        
        # 绘制条形图
        metrics = ["component_f1", "ratio_r2", "joint_accuracy", "overall_score"]
        x = np.arange(len(dim_values))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [grouped.get_group(value)[metric].mean() for value in dim_values]
            plt.bar(x + i*width, values, width, label=metric)
        
        # 设置图表属性
        plt.xlabel(dim_name)
        plt.ylabel("Score")
        plt.title(f"Comparison of {dim_name}")
        plt.xticks(x + width*1.5, dim_values)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图表
        save_path = os.path.join(vis_dir, f"compare_{dim_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"Saved comparison of {dim_name} to {save_path}")
    
    # 创建交互效应图
    for dim1, dim2 in itertools.combinations(dimensions.keys(), 2):
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 分组并计算平均值
        pivot = pd.pivot_table(
            results_df,
            values="overall_score",
            index=dim1,
            columns=dim2,
            aggfunc="mean"
        )
        
        # 绘制热力图
        plt.imshow(pivot, cmap="viridis")
        
        # 设置图表属性
        plt.xlabel(dim2)
        plt.ylabel(dim1)
        plt.title(f"Interaction between {dim1} and {dim2}")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.colorbar(label="Overall Score")
        
        # 添加数值标签
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                plt.text(j, i, f"{pivot.iloc[i, j]:.3f}",
                        ha="center", va="center", color="white")
        
        # 保存图表
        save_path = os.path.join(vis_dir, f"interaction_{dim1}_{dim2}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"Saved interaction between {dim1} and {dim2} to {save_path}")
    
    # 创建最佳实验报告
    best_exp = results_df.loc[results_df["overall_score"].idxmax()]
    
    print("\nBest Experiment:")
    print(f"  Experiment ID: {best_exp['experiment_id']}")
    for dim_name in dimensions.keys():
        print(f"  {dim_name}: {best_exp[dim_name]}")
    print(f"  Overall Score: {best_exp['overall_score']:.4f}")
    print(f"  Component F1: {best_exp['component_f1']:.4f}")
    print(f"  Ratio R2: {best_exp['ratio_r2']:.4f}")
    print(f"  Joint Accuracy: {best_exp['joint_accuracy']:.4f}")
    
    # 保存最佳实验报告
    report_path = os.path.join(results_dir, "best_experiment.txt")
    with open(report_path, "w",encoding='utf-8') as f:
        f.write("Best Experiment:\n")
        f.write(f"  Experiment ID: {best_exp['experiment_id']}\n")
        for dim_name in dimensions.keys():
            f.write(f"  {dim_name}: {best_exp[dim_name]}\n")
        f.write(f"  Overall Score: {best_exp['overall_score']:.4f}\n")
        f.write(f"  Component F1: {best_exp['component_f1']:.4f}\n")
        f.write(f"  Ratio R2: {best_exp['ratio_r2']:.4f}\n")
        f.write(f"  Joint Accuracy: {best_exp['joint_accuracy']:.4f}\n")
    
    print(f"Saved best experiment report to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--matrix", type=str, default="configs/experiment_matrix.yaml", help="Experiment matrix configuration file")
    parser.add_argument("--results_dir", type=str, default="results/metrics", help="Results directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--experiment_id", type=str, default=None, help="Run a specific experiment")
    
    args = parser.parse_args()
    
    if args.experiment_id:
        run_single_experiment(
            experiment_id=args.experiment_id,
            matrix_config_path=args.matrix,
            results_dir=args.results_dir,
            epochs=args.epochs,
            device=args.device
        )
    else:
        run_ablation_experiments(
            matrix_config_path=args.matrix,
            results_dir=args.results_dir,
            epochs=args.epochs,
            device=args.device
        )
