"""
多任务评价指标模块
"""

import re
import numpy as np
from typing import List, Dict, Tuple
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    multilabel_confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.preprocessing import MultiLabelBinarizer


class MultiTaskMetrics:
    def __init__(self, tokenizer, ratio_tolerance: float = 0.03):
        """
        多任务指标计算器

        Args:
            tokenizer: 成分分词器
            ratio_tolerance: 比例匹配容忍度（绝对值）
        """
        self.tokenizer = tokenizer
        self.rouge = Rouge(['rouge-1', 'rouge-2', 'rouge-l'])
        self.smooth = SmoothingFunction().method4
        self.ratio_tolerance = ratio_tolerance

    def calculate_all(self,
                      true_labels: List[str],
                      pred_labels: List[str],
                      true_ratios: List[List[float]],
                      pred_ratios: List[List[float]]) -> Dict[str, float]:
        """
        计算所有指标

        Args:
            true_labels: 真实标签列表 (如 ["W80P20", "C100"])
            pred_labels: 预测标签列表
            true_ratios: 真实比例列表（与tokenizer.parse_label的输出对齐）
            pred_ratios: 预测比例列表
            
        Returns:
            包含各项指标的字典
        """
        # 解析成分
        true_components, _ = self._parse_labels(true_labels)
        pred_components, _ = self._parse_labels(pred_labels)

        metrics = {
            **self._component_metrics(true_components, pred_components),
            **self._ratio_metrics(true_ratios, pred_ratios),
            **self._joint_metrics(true_components, pred_components, true_ratios, pred_ratios),
            **self._generation_metrics(true_labels, pred_labels)
        }
        metrics["overall_score"] = self._compute_overall_score(metrics)
        return metrics

    def _parse_labels(self, labels: List[str]) -> Tuple[List[List[str]], List[List[float]]]:
        """
        解析标签到成分和比例
        
        Args:
            labels: 标签列表
            
        Returns:
            成分列表和比例列表
        """
        components, percentages = [], []
        for label in labels:
            try:
                comps, percs = self.tokenizer.parse_label(label)
                components.append(comps)
                percentages.append(percs)
            except ValueError:
                components.append([])
                percentages.append([])
        return components, percentages

    def _component_metrics(self,
                           trues: List[List[str]],
                           preds: List[List[str]]) -> Dict[str, float]:
        """
        成分分类指标（多标签）
        
        Args:
            trues: 真实成分列表
            preds: 预测成分列表
            
        Returns:
            成分分类指标字典
        """
        # Exact Match (顺序无关)
        exact_match = accuracy_score(
            ['/'.join(sorted(t)) for t in trues],
            ['/'.join(sorted(p)) for p in preds]
        )

        # 多标签指标
        mlb = MultiLabelBinarizer()
        try:
            y_true = mlb.fit_transform(trues)
            y_pred = mlb.transform(preds)
        except ValueError:
            # 处理空预测的情况
            y_true = mlb.fit_transform(trues)
            y_pred = np.zeros_like(y_true)

        micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
        micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

        # 混淆矩阵（按类别求和）
        confusion_matrix = multilabel_confusion_matrix(y_true, y_pred).sum(axis=0).tolist()

        return {
            "component_accuracy": exact_match,
            "component_precision": micro_precision,
            "component_recall": micro_recall,
            "component_f1": micro_f1,
            "component_confusion_matrix": confusion_matrix
        }

    def _ratio_metrics(self,
                       trues: List[List[float]],
                       preds: List[List[float]]) -> Dict[str, float]:
        """
        比例回归指标
        
        Args:
            trues: 真实比例列表
            preds: 预测比例列表
            
        Returns:
            比例回归指标字典
        """
        # 展平数据并过滤无效值
        flat_trues, flat_preds = [], []
        for t_list, p_list in zip(trues, preds):
            min_len = min(len(t_list), len(p_list))
            flat_trues.extend(t_list[:min_len])
            flat_preds.extend(p_list[:min_len])

        if not flat_trues:
            return {
                "ratio_r2": 0.0,
                "ratio_mae": 0.0,
                "ratio_mse": 0.0,
                "ratio_rmse": 0.0
            }

        return {
            "ratio_r2": r2_score(flat_trues, flat_preds),
            "ratio_mae": mean_absolute_error(flat_trues, flat_preds),
            "ratio_mse": mean_squared_error(flat_trues, flat_preds),
            "ratio_rmse": np.sqrt(mean_squared_error(flat_trues, flat_preds))
        }

    def _joint_metrics(self,
                       trues_comp: List[List[str]],
                       preds_comp: List[List[str]],
                       trues_ratio: List[List[float]],
                       preds_ratio: List[List[float]]) -> Dict[str, float]:
        """
        联合匹配指标
        
        Args:
            trues_comp: 真实成分列表
            preds_comp: 预测成分列表
            trues_ratio: 真实比例列表
            preds_ratio: 预测比例列表
            
        Returns:
            联合匹配指标字典
        """
        joint_acc = 0
        total = len(trues_comp)

        for tc, pc, tr, pr in zip(trues_comp, preds_comp, trues_ratio, preds_ratio):
            if len(tc) != len(pc):
                continue

            match = all(
                (t_comp == p_comp) and (abs(t_ratio - p_ratio) <= self.ratio_tolerance)
                for t_comp, p_comp, t_ratio, p_ratio in zip(tc, pc, tr, pr)
            )
            joint_acc += int(match)

        return {"joint_accuracy": joint_acc / total if total else 0.0}

    def _generation_metrics(self,
                            trues: List[str],
                            preds: List[str]) -> Dict[str, float]:
        """
        文本生成指标
        
        Args:
            trues: 真实标签列表
            preds: 预测标签列表
            
        Returns:
            文本生成指标字典
        """
        # 数据预处理
        clean_trues, clean_preds = [], []
        pattern = re.compile(r'([A-Za-z]+)(\d+\.?\d*)')

        for t, p in zip(trues, preds):
            t_tokens = [item for pair in pattern.findall(t) for item in pair if pair]
            p_tokens = [item for pair in pattern.findall(p) for item in pair if pair]

            if t_tokens and p_tokens:
                clean_trues.append(' '.join(t_tokens))
                clean_preds.append(' '.join(p_tokens))

        # 处理空数据情况
        if not clean_trues:
            return {
                "bleu": 0.0,
                "rouge_1": 0.0,
                "rouge_2": 0.0,
                "rouge_l": 0.0
            }

        # BLEU计算
        references = [[s.split()] for s in clean_trues]
        hypotheses = [s.split() for s in clean_preds]
        bleu = corpus_bleu(references, hypotheses, smoothing_function=self.smooth)

        # ROUGE计算
        try:
            scores = self.rouge.get_scores(clean_preds, clean_trues, avg=True)
            rouge_1 = scores['rouge-1']['f']
            rouge_2 = scores['rouge-2']['f']
            rouge_l = scores['rouge-l']['f']
        except Exception as e:
            print(f"ROUGE calculation error: {str(e)}")
            rouge_1 = rouge_2 = rouge_l = 0.0

        return {
            "bleu": bleu,
            "rouge_1": rouge_1,
            "rouge_2": rouge_2,
            "rouge_l": rouge_l
        }

    def _compute_overall_score(self, metrics: Dict) -> float:
        """
        综合评分（加权平均）
        
        Args:
            metrics: 指标字典
            
        Returns:
            综合评分
        """
        weights = {
            'component_f1': 0.25,
            'ratio_r2': 0.25,
            'joint_accuracy': 0.25,
            'rouge_l': 0.25
        }
        return sum(metrics[k] * w for k, w in weights.items())
