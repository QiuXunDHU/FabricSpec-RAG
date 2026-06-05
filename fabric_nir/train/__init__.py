"""
训练模块
"""

from .single_task_trainer import SingleTaskTrainer
from .multi_task_trainer import MultiTaskTrainer

__all__ = ["SingleTaskTrainer", "MultiTaskTrainer"]
