
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """
    Args:
        y_true: 真实标签 [N]
        y_pred: 预测标签 [N]
        y_prob: 正类概率 [N]（可选，用于 AUC）
    Returns:
        dict: accuracy / f1_macro / f1_weighted / precision / recall / auc
    """
    result = {
        'accuracy':     float(accuracy_score(y_true, y_pred)),
        'f1_macro':     float(f1_score(y_true, y_pred, average='macro',    zero_division=0)),
        'f1_weighted':  float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'precision':    float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall':       float(recall_score(y_true, y_pred, average='macro',    zero_division=0)),
        'auc':          -1.0,
    }
    if y_prob is not None:
        try:
            result['auc'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
    return result


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience: int = 10, mode: str = 'max', delta: float = 0.0):
        self.patience = patience
        self.mode     = mode
        self.delta    = delta
        self.counter  = 0
        self.best     = None

    def __call__(self, metric: float) -> bool:
        """返回 True 表示应该停止"""
        improved = (self.best is None or
                    (self.mode == 'max' and metric > self.best + self.delta) or
                    (self.mode == 'min' and metric < self.best - self.delta))
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience