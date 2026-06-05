"""
自监督训练模块 - 用于运行自监督训练并修复可能的bug
"""

import os
import torch
import argparse
from tqdm import tqdm
import numpy as np

from fabric_nir.utils.config import ConfigManager
from fabric_nir.data.dataset import FabricDataset
from fabric_nir.tokenizers.component_tokenizer import FabricComponentTokenizer
from fabric_nir.models.seq2seq.single_task_seq2seq import SingleTaskSeq2Seq
from fabric_nir.train.single_task_trainer import SingleTaskTrainer


def run_self_supervised_training(
    data_file="data/train.xlsx",
    valid_file="data/valid.xlsx",
    output_dir="results/self_supervised",
    device="cpu",
    epochs=5,
    batch_size=32,
    learning_rate=1e-3,
    debug=True
):
    """
    运行自监督训练并修复可能的bug
    
    Args:
        data_file: 训练数据文件路径
        valid_file: 验证数据文件路径
        output_dir: 输出目录
        device: 计算设备
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        debug: 是否开启调试模式
    """
    try:
        print(f"开始自监督训练，将使用{data_file}数据集，结果将保存到{output_dir}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查数据文件是否存在
        if not os.path.exists(data_file):
            print(f"错误: 训练数据文件不存在: {data_file}")
            raise FileNotFoundError(f"训练数据文件不存在: {data_file}")
            
        if not os.path.exists(valid_file):
            print(f"错误: 验证数据文件不存在: {valid_file}")
            raise FileNotFoundError(f"验证数据文件不存在: {valid_file}")
        
        # 加载配置
        config_manager = ConfigManager("configs/base_config.yaml")
        
        # 初始化分词器
        vocab_path = os.path.join("fabric_nir", "tokenizers", "vocab", "component_vocab.json")
        if not os.path.exists(vocab_path):
            print(f"错误: 词汇表文件不存在: {vocab_path}")
            raise FileNotFoundError(f"词汇表文件不存在: {vocab_path}")
            
        # 初始化训练器
        trainer = SingleTaskTrainer(
            config_manager=config_manager,
            experiment_id="self_supervised"
        )
        
        # 修改训练参数
        trainer.config["data"]["train_file"] = data_file
        trainer.config["data"]["valid_file"] = valid_file
        trainer.config["data"]["batch_size"] = batch_size
        trainer.config["training"]["learning_rate"] = learning_rate
        trainer.results_dir = output_dir
        
        # 重新初始化数据集和模型
        trainer._init_datasets()
        trainer._init_model()
        
        # 运行自监督训练
        print(f"开始自监督训练，共{epochs}轮...")
        
        # 直接调用trainer的train方法
        metrics = trainer.train(epochs=epochs, device=device)
        print(f"训练完成，最终指标: {metrics}")
        
        # 模型已在trainer.train()中自动保存
        print(f"模型已保存到: {os.path.join(trainer.models_dir, f'{trainer.experiment_id}_final.pt')}")
        
        print(f"自监督训练完成")
        return final_model_path
    except Exception as e:
        print(f"自监督训练过程中发生未捕获的异常: {str(e)}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自监督训练")
    parser.add_argument("--data_file", type=str, default="data/train.xlsx", help="训练数据文件路径")
    parser.add_argument("--valid_file", type=str, default="data/valid.xlsx", help="验证数据文件路径")
    parser.add_argument("--output_dir", type=str, default="results/self_supervised", help="输出目录")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式")
    
    args = parser.parse_args()
    
    run_self_supervised_training(
        data_file=args.data_file,
        valid_file=args.valid_file,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        debug=args.debug
    )
