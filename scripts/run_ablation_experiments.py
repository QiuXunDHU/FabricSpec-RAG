"""
运行所有消融实验的脚本
"""

import os
import sys
import yaml
import torch
import argparse
from tqdm import tqdm
import time

from fabric_nir.utils.config import ConfigManager
from fabric_nir.train.single_task_trainer import SingleTaskTrainer
from fabric_nir.train.multi_task_trainer import MultiTaskTrainer
from fabric_nir.train.self_supervised_pretrainer import SelfSupervisedPretrainer


def run_ablation_experiments(matrix_config_path, results_dir="results", epochs=2, device="cuda"):
    """
    运行所有消融实验
    
    Args:
        matrix_config_path: 实验矩阵配置文件路径
        results_dir: 结果保存目录
        epochs: 训练轮数
        device: 计算设备
    """
    # 加载实验矩阵配置
    with open(matrix_config_path, "r") as f:
        matrix_config = yaml.safe_load(f)
    
    # 获取所有实验维度
    dimensions = matrix_config.get("dimensions", {})
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 记录所有实验结果
    all_results = {}
    
    # 遍历所有实验维度
    for dimension_name, dimension_values in dimensions.items():
        print(f"\n{'='*50}")
        print(f"运行 {dimension_name} 维度的消融实验")
        print(f"{'='*50}")
        
        # 遍历该维度的所有值
        for value in dimension_values:
            # 构建实验ID
            experiment_id = f"{dimension_name}_{value}"
            print(f"\n{'-'*50}")
            print(f"运行实验: {experiment_id}")
            print(f"{'-'*50}")
            
            # 运行单个实验
            try:
                metrics = run_single_experiment(
                    experiment_id=experiment_id,
                    matrix_config_path=matrix_config_path,
                    results_dir=results_dir,
                    epochs=epochs,
                    device=device
                )
                
                # 记录结果
                all_results[experiment_id] = metrics
                
                # 等待一段时间，避免GPU内存问题
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(2)
            except Exception as e:
                print(f"实验 {experiment_id} 运行失败: {str(e)}")
                continue
    
    # 保存所有实验结果
    results_path = os.path.join(results_dir, "all_ablation_results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(all_results, f)
    
    print(f"\n所有消融实验完成，结果保存在 {results_path}")
    
    return all_results


def run_single_experiment(experiment_id, matrix_config_path, results_dir="results", epochs=2, device="cuda"):
    """
    运行单个实验
    
    Args:
        experiment_id: 实验ID
        matrix_config_path: 实验矩阵配置文件路径
        results_dir: 结果保存目录
        epochs: 训练轮数
        device: 计算设备
        
    Returns:
        metrics: 实验结果指标
    """
    # 解析实验ID
    parts = experiment_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"无效的实验ID: {experiment_id}")
    
    # 处理维度名称，如decoder_type_gru中，dimension应为decoder_type
    if len(parts) >= 3 and parts[0] + "_" + parts[1] in ["decoder_type", "task_architecture", 
                                                        "initialization_method", "attention_mechanism", 
                                                        "backbone_type"]:
        dimension = parts[0] + "_" + parts[1]
        value = "_".join(parts[2:])
    else:
        dimension = parts[0]
        value = "_".join(parts[1:])
    
    # 加载实验矩阵配置
    with open(matrix_config_path, "r") as f:
        matrix_config = yaml.safe_load(f)
    
    # 获取基础配置
    base_config_path = matrix_config.get("base_config", "configs/base_config.yaml")
    
    # 获取该维度的配置文件路径
    config_paths = matrix_config.get("config_paths", {})
    if dimension not in config_paths:
        raise ValueError(f"未找到维度 {dimension} 的配置路径")
    
    dimension_path = config_paths[dimension]
    config_path = os.path.join(dimension_path, f"{value}.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    print(f"使用配置文件: {config_path}")
    
    # 创建配置管理器
    config_manager = ConfigManager(config_path)
    
    # 获取任务类型
    task_type = config_manager.config.get("training", {}).get("task_type", "single_task")
    
    # 检查是否需要预训练
    pretrain_config = config_manager.config.get("pretrain", {})
    use_pretrain = pretrain_config.get("use_pretrain", False)
    pretrain_epochs = pretrain_config.get("epochs", 1)
    
    # 如果需要预训练，先进行预训练
    if use_pretrain:
        print(f"开始自监督预训练 ({pretrain_epochs} epochs)...")
        pretrainer = SelfSupervisedPretrainer(config_manager, experiment_id)
        pretrain_model_path = pretrainer.pretrain(epochs=pretrain_epochs, device=device)
        
        # 更新配置，使用预训练模型
        config_manager.config["model"]["pretrained"] = {
            "use_pretrained": True,
            "weights_path": pretrain_model_path
        }
    
    # 创建训练器
    if task_type == "single_task":
        trainer = SingleTaskTrainer(config_manager, experiment_id)
    else:
        trainer = MultiTaskTrainer(config_manager, experiment_id)
    
    # 训练模型
    print(f"开始训练 ({epochs} epochs)...")
    trainer.train(epochs=epochs, device=device)
    
    # 测试模型
    print("开始测试...")
    metrics = trainer.test(device=device)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行消融实验")
    parser.add_argument("--matrix", type=str, default="configs/experiment_matrix.yaml", help="实验矩阵配置文件路径")
    parser.add_argument("--results_dir", type=str, default="results", help="结果保存目录")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--experiment_id", type=str, help="单个实验ID（可选）")
    
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
