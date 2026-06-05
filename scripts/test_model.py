"""
测试模型加载和生成功能，保存Top10结果到Excel
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from collections import defaultdict

from fabric_nir.data.dataset import FabricDataset
from fabric_nir.models.seq2seq.single_task_seq2seq import SingleTaskSeq2Seq
from fabric_nir.models.seq2seq.multi_task_seq2seq import MultiTaskSeq2Seq
from fabric_nir.tokenizers.component_tokenizer import FabricComponentTokenizer
from fabric_nir.utils.config import ConfigManager


class ModelTester:
    """
    模型测试器 - 加载模型并生成结果，保存Top10到Excel
    """
    
    def __init__(self, config_path, model_path, experiment_id=None, device="cpu"):
        """
        初始化模型测试器
        
        Args:
            config_path: 配置文件路径
            model_path: 模型文件路径
            experiment_id: 实验ID
            device: 计算设备
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.model_path = model_path
        self.experiment_id = experiment_id or "test"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 创建结果目录
        self.results_dir = self.config.get("results_dir", "results")
        self.test_results_dir = os.path.join(self.results_dir, "test_results")
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        # 初始化分词器
        vocab_path = os.path.join("fabric_nir", "tokenizers", "vocab", "component_vocab.json")
        self.tokenizer = FabricComponentTokenizer(vocab_path)
        
        # 初始化数据集
        self._init_dataset()
        
        # 初始化模型
        self._init_model()
    
    def _init_dataset(self):
        """
        初始化测试数据集
        """
        # 获取数据配置
        data_config = self.config.get("data", {})
        test_file = data_config.get("valid_file", "data/valid.xlsx")  # 使用验证集作为测试集
        max_length = data_config.get("max_length", 16)
        
        # 创建数据集
        self.test_dataset = FabricDataset(
            file_path=test_file,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        # 保存原始数据
        self.raw_data = pd.read_excel(test_file)
    
    def _init_model(self):
        """
        初始化模型
        """
        # 获取模型配置
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "single_task")
        
        # 创建模型
        if model_type == "single_task":
            self.model = SingleTaskSeq2Seq(
                config=model_config,
                vocab_size=self.tokenizer.vocab_size
            )
        else:
            self.model = MultiTaskSeq2Seq(
                config=model_config,
                vocab_size=self.tokenizer.vocab_size
            )
        
        # 加载模型权重
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
            print(f"Loaded model from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # 将模型移至设备
        self.model.to(self.device)
        self.model.eval()
    
    def generate_top_k(self, spectral, k=10):
        """
        生成Top-K预测结果
        
        Args:
            spectral: 光谱特征，形状为 [1, channels, features]
            k: 生成的候选数量
            
        Returns:
            top_k_results: Top-K预测结果列表，每个元素为(成分列表, 比例列表, 得分)
        """
        # 确保模型处于评估模式
        self.model.eval()
        
        # 将输入移至设备
        spectral = spectral.to(self.device)
        
        # 获取模型类型
        model_type = self.config.get("model", {}).get("type", "single_task")
        
        # 生成参数
        beam_width = k * 2  # 使用更大的beam width以获得更多候选
        max_length = 20
        
        with torch.no_grad():
            # 编码器前向传播
            encoder_outputs = self.model.encode(spectral)
            
            # 初始化
            batch_size = spectral.size(0)
            device = spectral.device
            
            # 初始序列：起始符号
            sequences = torch.full((beam_width, 1), self.tokenizer.sos_token_id, dtype=torch.long, device=device)
            sequence_scores = torch.zeros(beam_width, device=device)
            
            # 存储完成的序列
            completed_sequences = []
            completed_scores = []
            
            # 初始输入
            input_token = torch.full((1, 1), self.tokenizer.sos_token_id, dtype=torch.long, device=device)
            
            # 第一步解码
            if model_type == "single_task":
                output, hidden = self.model.decoder(input_token, None, encoder_outputs)
            else:
                output, hidden = self.model.component_decoder(input_token, None, encoder_outputs)
            
            # 获取前beam_width个最高概率
            log_probs = torch.log_softmax(output.squeeze(1), dim=-1)
            topk_log_probs, topk_indices = log_probs.topk(beam_width)
            
            # 更新序列
            sequences = torch.cat([sequences, topk_indices.unsqueeze(1)], dim=1)
            sequence_scores = topk_log_probs.squeeze(0)
            
            # 主循环
            for t in range(2, max_length):
                curr_len = sequences.size(1)
                
                # 展开所有可能的下一个token
                candidates = []
                candidate_scores = []
                
                # 为每个beam执行一步解码
                for i in range(len(sequences)):
                    # 如果序列已经结束，则跳过
                    if sequences[i, -1].item() == self.tokenizer.eos_token_id:
                        candidates.append(sequences[i])
                        candidate_scores.append(sequence_scores[i])
                        continue
                    
                    # 当前序列作为输入
                    curr_seq = sequences[i].unsqueeze(0)  # [1, curr_len]
                    
                    # 单步解码
                    if model_type == "single_task":
                        output, _ = self.model.decoder(curr_seq, None, encoder_outputs)
                    else:
                        output, _ = self.model.component_decoder(curr_seq, None, encoder_outputs)
                    
                    # 只关注最后一个时间步的输出
                    output = output[:, -1:, :]  # [1, 1, vocab_size]
                    
                    # 计算log概率
                    log_probs = torch.log_softmax(output.squeeze(1), dim=-1)
                    
                    # 获取前beam_width个最高概率
                    topk_log_probs, topk_indices = log_probs.topk(min(beam_width, log_probs.size(-1)))
                    
                    # 添加到候选列表
                    for j in range(topk_log_probs.size(-1)):
                        candidate_seq = torch.cat([sequences[i], topk_indices[0, j:j+1]], dim=0)
                        candidates.append(candidate_seq)
                        candidate_scores.append(sequence_scores[i] + topk_log_probs[0, j])
                
                # 如果所有序列都已完成，则退出循环
                if len(candidates) == 0:
                    break
                
                # 将候选列表转换为张量
                candidate_scores = torch.stack(candidate_scores)
                
                # 选择前beam_width个候选
                topk_scores, topk_indices = candidate_scores.topk(min(len(candidate_scores), beam_width))
                
                # 更新序列和分数
                new_sequences = []
                
                for i, idx in enumerate(topk_indices):
                    new_sequences.append(candidates[idx])
                    
                    # 如果序列以EOS结束，则添加到完成列表
                    if candidates[idx][-1].item() == self.tokenizer.eos_token_id:
                        completed_sequences.append(candidates[idx])
                        # 归一化分数，避免长序列优势
                        normalized_score = topk_scores[i] / len(candidates[idx])
                        completed_scores.append(normalized_score)
                
                # 如果所有beam都已完成，则退出循环
                if len(new_sequences) == 0:
                    break
                
                # 更新序列和分数
                sequences = torch.stack(new_sequences)
                sequence_scores = topk_scores
            
            # 如果没有完成的序列，则使用当前序列
            if len(completed_sequences) == 0:
                completed_sequences = sequences
                # 归一化分数
                completed_scores = [score / len(seq) for score, seq in zip(sequence_scores, sequences)]
                completed_scores = torch.stack(completed_scores)
            else:
                completed_scores = torch.stack(completed_scores)
            
            # 选择Top-K序列
            topk_scores, topk_indices = completed_scores.topk(min(k, len(completed_scores)))
            
            # 解码序列
            top_k_results = []
            for i, idx in enumerate(topk_indices):
                seq = completed_sequences[idx]
                score = topk_scores[i].item()
                
                # 解码序列
                decoded = self.tokenizer.decode(seq.cpu().numpy())
                
                # 解析成分和比例
                try:
                    components, ratios = self.tokenizer.parse_label(decoded)
                    top_k_results.append((components, ratios, score))
                except ValueError:
                    # 如果解析失败，则跳过
                    continue
            
            return top_k_results
    
    def test_and_save(self, k=10):
        """
        测试模型并保存Top-K结果到Excel
        
        Args:
            k: 每个样本生成的候选数量
            
        Returns:
            excel_path: 保存的Excel文件路径
        """
        # 准备结果容器
        results = []
        
        # 遍历测试数据集
        for i in tqdm(range(len(self.test_dataset)), desc="Testing"):
            # 获取样本
            sample = self.test_dataset[i]
            spectral = sample["spectral"].unsqueeze(0)  # 添加batch维度
            true_label = sample["label"]
            
            # 生成Top-K预测
            top_k_results = self.generate_top_k(spectral, k=k)
            
            # 解析真实标签
            try:
                true_components, true_ratios = self.tokenizer.parse_label(true_label)
            except ValueError:
                true_components, true_ratios = [], []
            
            # 添加到结果列表
            for rank, (pred_components, pred_ratios, score) in enumerate(top_k_results):
                # 构建预测标签
                pred_label = "".join(f"{c}{r*100:.1f}" for c, r in zip(pred_components, pred_ratios))
                
                # 添加到结果
                results.append({
                    "Sample_ID": i,
                    "True_Label": true_label,
                    "True_Components": ",".join(true_components),
                    "True_Ratios": ",".join([f"{r*100:.1f}" for r in true_ratios]),
                    "Rank": rank + 1,
                    "Predicted_Label": pred_label,
                    "Predicted_Components": ",".join(pred_components),
                    "Predicted_Ratios": ",".join([f"{r*100:.1f}" for r in pred_ratios]),
                    "Score": score
                })
        
        # 创建DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存到Excel
        excel_path = os.path.join(self.test_results_dir, f"{self.experiment_id}_top{k}_results.xlsx")
        results_df.to_excel(excel_path, index=False)
        print(f"Saved Top-{k} results to {excel_path}")
        
        # 计算准确率统计
        self._calculate_accuracy_stats(results_df)
        
        return excel_path
    
    def _calculate_accuracy_stats(self, results_df):
        """
        计算准确率统计
        
        Args:
            results_df: 结果DataFrame
        """
        # 按样本ID分组
        grouped = results_df.groupby("Sample_ID")
        
        # 统计Top-1, Top-3, Top-5, Top-10准确率
        total_samples = len(grouped)
        correct_top = defaultdict(int)
        
        for sample_id, group in grouped:
            true_label = group["True_Label"].iloc[0]
            
            # 检查Top-N中是否有正确预测
            for n in [1, 3, 5, 10]:
                top_n = group[group["Rank"] <= n]
                if any(top_n["Predicted_Label"] == true_label):
                    correct_top[n] += 1
        
        # 计算准确率
        accuracy_stats = {
            f"Top-{n} Accuracy": correct_top[n] / total_samples * 100
            for n in [1, 3, 5, 10]
        }
        
        # 打印统计结果
        print("\nAccuracy Statistics:")
        for metric, value in accuracy_stats.items():
            print(f"{metric}: {value:.2f}%")
        
        # 保存统计结果
        stats_path = os.path.join(self.test_results_dir, f"{self.experiment_id}_accuracy_stats.txt")
        with open(stats_path, "w") as f:
            f.write("Accuracy Statistics:\n")
            for metric, value in accuracy_stats.items():
                f.write(f"{metric}: {value:.2f}%\n")
        
        print(f"Saved accuracy statistics to {stats_path}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="Test model and save Top-K results to Excel")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--experiment_id", type=str, default="test", help="Experiment ID")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--k", type=int, default=10, help="Number of candidates to generate")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ModelTester(
        config_path=args.config,
        model_path=args.model,
        experiment_id=args.experiment_id,
        device=args.device
    )
    
    # 测试并保存结果
    excel_path = tester.test_and_save(k=args.k)
    print(f"Testing completed. Results saved to {excel_path}")


if __name__ == "__main__":
    main()
