from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_ids: List[int]) -> Dict:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0))
    recalls = recall_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_recall": [float(x) for x in recalls.tolist()],
        "confusion_matrix": cm.tolist(),
    }

