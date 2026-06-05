"""
多任务训练器模块
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from fabric_nir.data.multi_task_dataset import MultiTaskFabricDataset
from fabric_nir.models.seq2seq.multi_task_seq2seq import MultiTaskSeq2Seq
from fabric_nir.tokenizers.component_tokenizer import FabricComponentTokenizer
from fabric_nir.metrics.metrics_logger import MetricsLogger
from fabric_nir.metrics.multi_task_metrics import MultiTaskMetrics
from fabric_nir.models.beam_search import BeamSearchGenerator
from fabric_nir.visualization import Visualizer


class MultiTaskTrainer:
    """
    多任务Seq2Seq模型训练器
    """
    
    def __init__(self, config_manager, experiment_id=None):
        """
        初始化多任务训练器
        
        Args:
            config_manager: 配置管理器
            experiment_id: 实验ID（可选）
        """
        self.config = config_manager.config
        self.experiment_id = experiment_id or "multi_task"
        
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
        self.train_dataset = MultiTaskFabricDataset(
            file_path=train_file,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        self.valid_dataset = MultiTaskFabricDataset(
            file_path=valid_file,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        # 自定义collate_fn，确保批处理后的张量维度正确
        def collate_fn(batch):
            spectral = torch.stack([item['spectral'] for item in batch])
            label = [item['label'] for item in batch]
            label_ids = torch.stack([item['label_ids'] for item in batch])
            ratios = torch.stack([item['ratios'] for item in batch])
            
            # 确保label_ids和ratios是2D张量 [batch_size, seq_len]
            if len(label_ids.shape) > 2:
                print(f"警告: collate_fn中label_ids维度为{len(label_ids.shape)}D，将降维")
                label_ids = label_ids.view(label_ids.size(0), -1)
            
            if len(ratios.shape) > 2:
                print(f"警告: collate_fn中ratios维度为{len(ratios.shape)}D，将降维")
                ratios = ratios.view(ratios.size(0), -1)
            
            return {
                'spectral': spectral,
                'label': label,
                'label_ids': label_ids,
                'ratios': ratios
            }
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
    
    def _init_model(self):
        """
        初始化模型
        """
        # 获取模型配置
        model_config = self.config.get("model", {})
        
        # 创建模型
        self.model = MultiTaskSeq2Seq(
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
        self.cls_criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.reg_criterion = nn.MSELoss()
    
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
            train_cls_loss = 0.0
            train_reg_loss = 0.0
            train_total_loss = 0.0
            
            # 训练一个epoch
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch_idx, batch in enumerate(train_pbar):
                # 获取数据
                spectrals = batch["spectral"].to(device)
                labels = batch["label_ids"].to(device)
                ratios = batch["ratios"].to(device)
                
                # 确保输入维度正确
                if len(labels.shape) > 2:
                    print(f"警告: labels维度过高，当前为{len(labels.shape)}D，将降维")
                    # 如果labels是3D，取第一个维度作为序列
                    if len(labels.shape) == 3:
                        labels = labels[:, 0, :]
                    else:
                        # 如果是更高维度，展平到2D
                        labels = labels.reshape(labels.size(0), -1)
                
                if len(ratios.shape) > 2:
                    print(f"警告: ratios维度过高，当前为{len(ratios.shape)}D，将降维")
                    # 如果ratios是3D，取第一个维度作为序列
                    if len(ratios.shape) == 3:
                        ratios = ratios[:, 0, :]
                    else:
                        # 如果是更高维度，展平到2D
                        ratios = ratios.reshape(ratios.size(0), -1)
                
                # 前向传播
                cls_outputs, reg_outputs = self.model(spectrals, labels[:, :-1], ratios[:, :-1])
                
                # 计算分类损失
                cls_loss = self.cls_criterion(
                    cls_outputs.reshape(-1, cls_outputs.size(-1)),
                    labels[:, 1:].reshape(-1)
                )
                
                # 计算回归损失
                # 创建掩码，忽略填充位置
                mask = (labels[:, 1:] != self.tokenizer.pad_token_id).float()
                
                # 确保reg_outputs和mask形状一致
                if reg_outputs.shape != mask.shape:
                    print(f"警告: reg_outputs形状{reg_outputs.shape}与mask形状{mask.shape}不一致，将调整")
                    # 如果是序列长度不一致，截断较长的或扩展较短的
                    if reg_outputs.size(1) > mask.size(1):
                        reg_outputs = reg_outputs[:, :mask.size(1)]
                    elif reg_outputs.size(1) < mask.size(1):
                        # 调整mask和ratios以匹配reg_outputs
                        mask = mask[:, :reg_outputs.size(1)]
                        ratios = ratios.clone()  # 创建副本以避免修改原始数据
                        ratios_slice = ratios[:, 1:][:, :reg_outputs.size(1)]
                        
                        # 计算回归损失
                        reg_loss = self.reg_criterion(
                            reg_outputs * mask,
                            ratios_slice * mask
                        )
                    else:
                        # 形状一致，直接计算损失
                        reg_loss = self.reg_criterion(
                            reg_outputs * mask,
                            ratios[:, 1:] * mask
                        )
                else:
                    # 形状一致，直接计算损失
                    reg_loss = self.reg_criterion(
                        reg_outputs * mask,
                        ratios[:, 1:] * mask
                    )
                
                # 总损失
                total_loss = cls_loss + reg_loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                
                # 更新参数
                optimizer.step()
                
                # 更新损失
                train_cls_loss += cls_loss.item()
                train_reg_loss += reg_loss.item()
                train_total_loss += total_loss.item()
                train_pbar.set_postfix({
                    "cls_loss": cls_loss.item(),
                    "reg_loss": reg_loss.item(),
                    "total_loss": total_loss.item()
                })
            
            # 计算平均损失
            train_cls_loss /= len(self.train_loader)
            train_reg_loss /= len(self.train_loader)
            train_total_loss /= len(self.train_loader)
            
            # 验证模式
            self.model.eval()
            valid_cls_loss = 0.0
            valid_reg_loss = 0.0
            valid_total_loss = 0.0
            all_preds = []
            all_labels = []
            all_pred_ratios = []
            all_true_ratios = []
            
            # 验证一个epoch
            valid_pbar = tqdm(self.valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
            with torch.no_grad():
                for batch_idx, batch in enumerate(valid_pbar):
                    # 获取数据
                    spectrals = batch["spectral"].to(device)
                    labels = batch["label_ids"].to(device)
                    ratios = batch["ratios"].to(device)
                    
                    # 前向传播
                    cls_outputs, reg_outputs = self.model(spectrals, labels[:, :-1], ratios[:, :-1])
                    
                    # 计算分类损失
                    cls_loss = self.cls_criterion(
                        cls_outputs.reshape(-1, cls_outputs.size(-1)),
                        labels[:, 1:].reshape(-1)
                    )
                    
                    # 计算回归损失
                    # 创建掩码，忽略填充位置
                    mask = (labels[:, 1:] != self.tokenizer.pad_token_id).float()
                    reg_loss = self.reg_criterion(
                        reg_outputs * mask,
                        ratios[:, 1:] * mask
                    )
                    
                    # 总损失
                    total_loss = cls_loss + reg_loss
                    
                    # 更新损失
                    valid_cls_loss += cls_loss.item()
                    valid_reg_loss += reg_loss.item()
                    valid_total_loss += total_loss.item()
                    valid_pbar.set_postfix({
                        "cls_loss": cls_loss.item(),
                        "reg_loss": reg_loss.item(),
                        "total_loss": total_loss.item()
                    })
                    
                    # 生成预测
                    preds = torch.argmax(cls_outputs, dim=-1)
                    
                    # 收集预测和标签
                    for i in range(preds.size(0)):
                        pred_seq = preds[i].cpu().numpy()
                        label_seq = labels[i, 1:].cpu().numpy()
                        pred_ratio_seq = reg_outputs[i].cpu().numpy()
                        true_ratio_seq = ratios[i, 1:].cpu().numpy()
                        
                        # 过滤掉填充位置
                        valid_indices = label_seq != self.tokenizer.pad_token_id
                        
                        # 解码
                        pred_tokens = [self.tokenizer.id2token.get(idx, "<unk>") for idx in pred_seq[valid_indices]]
                        label_tokens = [self.tokenizer.id2token.get(idx, "<unk>") for idx in label_seq[valid_indices]]
                        
                        # 构建标签字符串
                        pred_label = "".join(f"{t}{r*100:.1f}" for t, r in zip(pred_tokens, pred_ratio_seq[valid_indices]))
                        true_label = "".join(f"{t}{r*100:.1f}" for t, r in zip(label_tokens, true_ratio_seq[valid_indices]))
                        
                        all_preds.append(pred_label)
                        all_labels.append(true_label)
                        all_pred_ratios.append(pred_ratio_seq[valid_indices].tolist())
                        all_true_ratios.append(true_ratio_seq[valid_indices].tolist())
            
            # 计算平均损失
            valid_cls_loss /= len(self.valid_loader)
            valid_reg_loss /= len(self.valid_loader)
            valid_total_loss /= len(self.valid_loader)
            
            # 解析标签
            true_components, _ = [], []
            pred_components, _ = [], []
            
            for label in all_labels:
                try:
                    comps, _ = self.tokenizer.parse_label(label)
                    true_components.append(comps)
                except ValueError:
                    true_components.append([])
            
            for label in all_preds:
                try:
                    comps, _ = self.tokenizer.parse_label(label)
                    pred_components.append(comps)
                except ValueError:
                    pred_components.append([])
            
            # 计算指标
            metrics = self.metrics_calculator.calculate_all(
                true_labels=all_labels,
                pred_labels=all_preds,
                true_ratios=all_true_ratios,
                pred_ratios=all_pred_ratios
            )
            
            # 添加损失指标
            metrics["train_cls_loss"] = train_cls_loss
            metrics["train_reg_loss"] = train_reg_loss
            metrics["train_total_loss"] = train_total_loss
            metrics["valid_cls_loss"] = valid_cls_loss
            metrics["valid_reg_loss"] = valid_reg_loss
            metrics["valid_total_loss"] = valid_total_loss
            
            # 更新评价指标记录器
            self.metrics_logger.update(metrics, phase="valid", epoch=epoch)
            
            # 打印指标
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_total_loss:.4f} (Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss:.4f})")
            print(f"Valid Loss: {valid_total_loss:.4f} (Cls: {valid_cls_loss:.4f}, Reg: {valid_reg_loss:.4f})")
            print(f"Component F1: {metrics['component_f1']:.4f}, Ratio R2: {metrics['ratio_r2']:.4f}")
            print(f"Joint Accuracy: {metrics['joint_accuracy']:.4f}, Overall Score: {metrics['overall_score']:.4f}")
            
            # 可视化
            if epoch % 1 == 0:
                # 可视化训练曲线
                self.metrics_logger.plot(
                    ["train_total_loss", "valid_total_loss", "component_f1", "ratio_r2", "joint_accuracy", "overall_score"],
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
                    try:
                        attention_weights = self.model.get_attention_weights(spectrals[:1].to(device))
                        if attention_weights is not None:
                            attention_weights = attention_weights.detach().cpu().numpy()
                            self.vis_manager.visualize_attention(
                                attention_weights=attention_weights[0],
                                name=f"epoch_{epoch+1}_attention"
                            )
                    except Exception as e:
                        print(f"注意力可视化失败: {e}")
            
            # 保存模型
            if valid_total_loss < best_loss:
                best_loss = valid_total_loss
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
        all_pred_ratios = []
        all_true_ratios = []
        
        # 测试
        test_pbar = tqdm(self.valid_loader, desc="Testing")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_pbar):
                # 获取数据
                spectrals = batch["spectral"].to(device)
                labels = batch["label_ids"].to(device)
                ratios = batch["ratios"].to(device)
                
                # 使用Beam Search生成
                results = self.beam_search.generate(spectrals, task_type="multi_task")
                
                # 收集预测和标签
                for i, (comps, percs) in enumerate(results):
                    pred_label = "".join(f"{c}{p*100:.1f}" for c, p in zip(comps, percs))
                    all_preds.append(pred_label)
                    all_pred_ratios.append(percs)
                
                # 收集真实标签
                for i in range(labels.size(0)):
                    label_seq = labels[i, 1:].cpu().numpy()
                    true_ratio_seq = ratios[i, 1:].cpu().numpy()
                    
                    # 过滤掉填充位置
                    valid_indices = label_seq != self.tokenizer.pad_token_id
                    
                    # 解码
                    label_tokens = [self.tokenizer.id2token.get(idx, "<unk>") for idx in label_seq[valid_indices]]
                    
                    # 构建标签字符串
                    true_label = "".join(f"{t}{r*100:.1f}" for t, r in zip(label_tokens, true_ratio_seq[valid_indices]))
                    
                    all_labels.append(true_label)
                    all_true_ratios.append(true_ratio_seq[valid_indices].tolist())
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_all(
            true_labels=all_labels,
            pred_labels=all_preds,
            true_ratios=all_true_ratios,
            pred_ratios=all_pred_ratios
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
