"""
主程序 - 运行消融实验
"""

import os
import sys
import argparse
import torch

from experiments.run_ablation import run_ablation_experiments, run_single_experiment


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="废旧纺织品近红外光谱Seq2Seq模型消融实验")
    parser.add_argument("--matrix", type=str, default="configs/experiment_matrix.yaml", help="实验矩阵配置文件路径")
    parser.add_argument("--config", type=str, help="单个实验配置文件路径（可选，与--matrix互斥）")
    parser.add_argument("--results_dir", type=str, default="results", help="结果保存目录")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument('--experiment_id', type=str, default=None, help='实验ID，用于保存结果')
    parser.add_argument('--test_only', action='store_true', help='仅执行测试，不进行训练')
    
    args = parser.parse_args()
    
    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "visualizations"), exist_ok=True)
    
    # 运行实验
    if args.config:
        # 使用单个配置文件运行实验
        print(f"使用配置文件运行实验: {args.config}")
        from fabric_nir.utils.config import ConfigManager
        from fabric_nir.train.single_task_trainer import SingleTaskTrainer
        from fabric_nir.train.multi_task_trainer import MultiTaskTrainer
        
        # 创建配置管理器
        config_manager = ConfigManager(args.config)
        
        # 获取任务类型
        task_type = config_manager.config.get("training", {}).get("task_type", "single_task")
        
        # 创建训练器
        experiment_id = args.experiment_id or "custom_experiment"
        if task_type == "single_task":
            trainer = SingleTaskTrainer(config_manager, experiment_id)
        else:
            trainer = MultiTaskTrainer(config_manager, experiment_id)
        
        # 根据参数决定是否只执行测试
        if args.test_only:
            print(f"仅执行测试模式，不进行训练")
            trainer.test(device=args.device)
        else:
            # 训练模型
            trainer.train(epochs=args.epochs, device=args.device)
            
            # 测试模型
            trainer.test(device=args.device)
    elif args.experiment_id:
        print(f"运行单个实验: {args.experiment_id}")
        run_single_experiment(
            experiment_id=args.experiment_id,
            matrix_config_path=args.matrix,
            results_dir=args.results_dir,
            epochs=args.epochs,
            device=args.device
        )
    else:
        print("运行所有消融实验")
        run_ablation_experiments(
            matrix_config_path=args.matrix,
            results_dir=args.results_dir,
            epochs=args.epochs,
            device=args.device
        )


if __name__ == "__main__":
    main()
