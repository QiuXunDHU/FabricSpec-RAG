"""
Seq2Seq模型模块
"""

from .single_task_seq2seq import SingleTaskSeq2Seq
from .multi_task_seq2seq import MultiTaskSeq2Seq

__all__ = ["SingleTaskSeq2Seq", "MultiTaskSeq2Seq"]
