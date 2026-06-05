"""
单任务训练器模块
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from fabric_nir.data.dataset import FabricDataset
from fabric_nir.models.seq2seq.single_task_seq2seq import SingleTaskSeq2Seq
from fabric_nir.tokenizers.component_tokenizer import FabricComponentTokenizer
from fabric_nir.metrics.metrics_logger import MetricsLogger
from fabric_nir.metrics.multi_task_metrics import MultiTaskMetrics
from fabric_nir.models.beam_search import BeamSearchGenerator
from fabric_nir.visualization import Visualizer


class SingleTaskTrainer:
    """
    单任务Seq2Seq模型训练器
    """
    
    def __init__(self, config_manager, experiment_id=None):
        """
        初始化单任务训练器
        
        Args:
            config_manager: 配置管理器
            experiment_id: 实验ID（可选）
        """
        self.config = config_manager.config
        self.experiment_id = experiment_id or "single_task"
        
        # 创建结果目录
        self.results_dir = self.config.get("results_dir", "results")
        self.metrics_dir = os.path.join(self.results_dir, "metrics")
        self.models_dir = os.path.join(self.results_dir, "models")
        self.vis_dir = os.path.join(self.results_dir, "visualizations")
        
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # 初始化分词器
        vocab_path = os.path.join("fabric_nir", "tokenizers", "vocab", "component_vocab.json")
        self.tokenizer = FabricComponentTokenizer(vocab_path)
        
        # 初始化数据集和数据加载器
        self._init_datasets()
        
        # 初始化模型
        self._init_model()
        
        # 初始化评价指标记录器
        self.metrics_logger = MetricsLogger(self.metrics_dir, self.experiment_id)
        
        # 初始化可视化管理器
        self.vis_manager = Visualizer(self.vis_dir)
        
        # 初始化评价指标计算器
        self.metrics_calculator = MultiTaskMetrics(self.tokenizer)
        
        # 初始化Beam Search生成器
        self.beam_search = BeamSearchGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config.get("generation", {})
        )
    
    def _init_datasets(self):
        """
        初始化数据集和数据加载器
        """
        # 获取数据配置
        data_config = self.config.get("data", {})
        train_file = data_config.get("train_file", "data/train.xlsx")
        valid_file = data_config.get("valid_file", "data/valid.xlsx")
        batch_size = data_config.get("batch_size", 32)
        max_length = data_config.get("max_length", 16)
        
        # 创建数据集
        self.train_dataset = FabricDataset(
            file_path=train_file,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        self.valid_dataset = FabricDataset(
            file_path=valid_file,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    
    def _init_model(self):
        """
        初始化模型
        """
        # 获取模型配置
        model_config = self.config.get("model", {})
        
        # 创建模型
        self.model = SingleTaskSeq2Seq(
            config=model_config,
            vocab_size=self.tokenizer.vocab_size
        )
        
        # 加载预训练权重（如果有）
        pretrained_config = model_config.get("pretrained", {})
        if pretrained_config.get("use_pretrained", False):
            weights_path = pretrained_config.get("weights_path")
            if weights_path and os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu")
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights from {weights_path}")
        
        # 初始化损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    
    def train(self, epochs=10, device="cuda"):
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            device: 计算设备
        """
        # 获取训练配置
        training_config = self.config.get("training", {})
        lr = training_config.get("learning_rate", 0.001)
        weight_decay = training_config.get("weight_decay", 0.0001)
        clip_grad_norm = training_config.get("clip_grad_norm", 1.0)
        early_stopping_patience = training_config.get("early_stopping_patience", 5)
        
        # 设置设备
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # 初始化优化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 初始化早停
        best_loss = float('inf')
        patience_counter = 0
        
        # 训练循环
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            
            # 训练一个epoch
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch_idx, batch in enumerate(train_pbar):
                # 获取数据
                spectrals = batch["spectral"].to(device)
                labels = batch["label_ids"].to(device)
                
                # 确保只接收一个返回值
                outputs = self.model(spectrals, labels[:, :-1])

                # 确保输出和标签的batch_size匹配
                # 分析标签形状问题
                # 正确的reshape方式应该是将序列维度展平
                seq_len = labels.size(1) - 1  # 减去teacher forcing的第一个token
                batch_size = labels.size(0)
                
                # 重新计算正确的reshape方式
                reshaped_outputs = outputs.view(batch_size * seq_len, -1)
                reshaped_labels = labels[:, 1:].contiguous().view(-1)
                
                # 检查并修正batch_size不匹配问题
                if reshaped_outputs.size(0) != reshaped_labels.size(0):
                    # 使用最小的长度
                    min_len = min(reshaped_outputs.size(0), reshaped_labels.size(0))
                    reshaped_outputs = reshaped_outputs[:min_len]
                    reshaped_labels = reshaped_labels[:min_len]
                
                # 计算损失
                loss = self.criterion(
                    reshaped_outputs,
                    reshaped_labels
                )
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                
                # 更新参数
                optimizer.step()
                
                # 更新损失
                train_loss += loss.item()
                train_pbar.set_postfix({"loss": loss.item()})
            
            # 计算平均损失
            train_loss /= len(self.train_loader)
            
            # 验证模式
            self.model.eval()
            valid_loss = 0.0
            all_preds = []
            all_labels = []
                         # 验证一个epoch
            valid_pbar = tqdm(self.valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
            with torch.no_grad():
                for batch_idx, batch in enumerate(valid_pbar):
                    # 获取数据
                    spectrals = batch["spectral"].to(device)
                    labels = batch["label_ids"].to(device)
                    
                    # 前向传播
                    outputs = self.model(spectrals, labels[:, :-1])
                    
                    # 计算损失 - 使用与训练阶段相同的reshape逻辑
                    seq_len = labels.size(1) - 1
                    batch_size = labels.size(0)
                    
                    reshaped_outputs = outputs.view(batch_size * seq_len, -1)
                    reshaped_labels = labels[:, 1:].contiguous().view(-1)
                    
                    # 检查并修正batch_size不匹配问题
                    if reshaped_outputs.size(0) != reshaped_labels.size(0):
                        min_len = min(reshaped_outputs.size(0), reshaped_labels.size(0))
                        reshaped_outputs = reshaped_outputs[:min_len]
                        reshaped_labels = reshaped_labels[:min_len]
                    
                    loss = self.criterion(
                        reshaped_outputs,
                        reshaped_labels
                    )
                    
                    # 更新损失
                    valid_loss += loss.item()
                    valid_pbar.set_postfix({"loss": loss.item()})
                    
                    # 生成预测
                    preds = torch.argmax(outputs, dim=-1)
                    
                    # 收集预测和标签
                    all_preds.extend(self.tokenizer.batch_decode(preds.cpu().numpy()))
                    all_labels.extend(self.tokenizer.batch_decode(labels[:, 1:].cpu().numpy()))
            
            # 计算平均损失
            valid_loss /= len(self.valid_loader)
            
            # 计算评价指标
            true_labels = all_labels
            pred_labels = all_preds
            
            # 解析标签
            true_components, true_ratios = [], []
            pred_components, pred_ratios = [], []
            
            for label in true_labels:
                try:
                    comps, ratios = self.tokenizer.parse_label(label)
                    true_components.append(comps)
                    true_ratios.append(ratios)
                except ValueError:
                    true_components.append([])
                    true_ratios.append([])
            
            for label in pred_labels:
                try:
                    comps, ratios = self.tokenizer.parse_label(label)
                    pred_components.append(comps)
                    pred_ratios.append(ratios)
                except ValueError:
                    pred_components.append([])
                    pred_ratios.append([])
            
            # 计算指标
            metrics = self.metrics_calculator.calculate_all(
                true_labels=true_labels,
                pred_labels=pred_labels,
                true_ratios=true_ratios,
                pred_ratios=pred_ratios
            )
            
            # 添加损失指标
            metrics["train_loss"] = train_loss
            metrics["valid_loss"] = valid_loss
            
            # 更新评价指标记录器
            self.metrics_logger.update(metrics, phase="valid", epoch=epoch)
            
            # 打印指标
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
            print(f"Component F1: {metrics['component_f1']:.4f}, Ratio R2: {metrics['ratio_r2']:.4f}")
            print(f"Joint Accuracy: {metrics['joint_accuracy']:.4f}, Overall Score: {metrics['overall_score']:.4f}")
            
            # 可视化
            if epoch % 1 == 0:
                # 可视化训练曲线
                self.metrics_logger.plot(
                    ["train_loss", "valid_loss", "component_f1", "ratio_r2", "joint_accuracy", "overall_score"],
                    save_dir=self.vis_dir
                )
                
                # 可视化嵌入
                if hasattr(self.model, "get_embeddings"):
                    embeddings = self.model.get_embeddings(spectrals[:10].to(device)).detach().cpu().numpy()
                    self.vis_manager.visualize_embedding(
                        embeddings=embeddings,
                        name=f"epoch_{epoch+1}_embedding"
                    )
                
                # 可视化注意力
                if hasattr(self.model, "get_attention_weights"):
                    attention_weights = self.model.get_attention_weights(spectrals[:1].to(device)).detach().cpu().numpy()
                    self.vis_manager.visualize_attention(
                        attention_weights=attention_weights[0],
                        name=f"epoch_{epoch+1}_attention"
                    )
            
            # 保存模型
            if valid_loss < best_loss:
                best_loss = valid_loss
                patience_counter = 0
                
                # 保存最佳模型
                model_path = os.path.join(self.models_dir, f"{self.experiment_id}_best.pt")
                torch.save(self.model.state_dict(), model_path)
                print(f"Saved best model to {model_path}")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # 保存最终模型
        model_path = os.path.join(self.models_dir, f"{self.experiment_id}_final.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved final model to {model_path}")
        
        # 保存评价指标历史
        self.metrics_logger.save()
        
        return self.metrics_logger.get_latest_metrics("valid")
    
    def test(self, device="cuda"):
        """
        测试模型
        
        Args:
            device: 计算设备
        """
        # 设置设备
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # 加载最佳模型
        model_path = os.path.join(self.models_dir, f"{self.experiment_id}_best.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            print(f"Loaded best model from {model_path}")
        
        # 测试模式
        self.model.eval()
        all_preds = []
        all_labels = []
        
        # 测试
        test_pbar = tqdm(self.valid_loader, desc="Testing")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_pbar):
                # 获取数据
                spectrals = batch["spectral"].to(device)
                labels = batch["label_ids"].to(device)
                
                # 使用Beam Search生成
                results = self.beam_search.generate(spectrals, task_type="single_task")
                
                # 收集预测和标签
                for i, (comps, percs) in enumerate(results):
                    pred_label = "".join(f"{c}{p*100:.1f}" for c, p in zip(comps, percs))
                    all_preds.append(pred_label)
                
                all_labels.extend(self.tokenizer.batch_decode(labels[:, 1:].cpu().numpy()))
        
        # 解析标签
        true_components, true_ratios = [], []
        pred_components, pred_ratios = [], []
        
        for label in all_labels:
            try:
                comps, ratios = self.tokenizer.parse_label(label)
                true_components.append(comps)
                true_ratios.append(ratios)
            except ValueError:
                true_components.append([])
                true_ratios.append([])
        
        for label in all_preds:
            try:
                comps, ratios = self.tokenizer.parse_label(label)
                pred_components.append(comps)
                pred_ratios.append(ratios)
            except ValueError:
                pred_components.append([])
                pred_ratios.append([])
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_all(
            true_labels=all_labels,
            pred_labels=all_preds,
            true_ratios=true_ratios,
            pred_ratios=pred_ratios
        )
        
        # 打印指标
        print("Test Results:")
        print(f"Component F1: {metrics['component_f1']:.4f}, Ratio R2: {metrics['ratio_r2']:.4f}")
        print(f"Joint Accuracy: {metrics['joint_accuracy']:.4f}, Overall Score: {metrics['overall_score']:.4f}")
        
        # 保存测试结果
        test_results_path = os.path.join(self.metrics_dir, f"{self.experiment_id}_test_results.txt")
        with open(test_results_path, "w") as f:
            f.write("Test Results:\n")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
        
        return metrics
