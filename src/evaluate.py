import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def precision_recall_at_k(y, score, k):
    top = np.argsort(-score)[:k]
    tp = y[top].sum()
    return tp / k, tp / y.sum()

def report(y, score, k):
    p, r = precision_recall_at_k(y, score, k)
    return {"PR-AUC": average_precision_score(y, score),
            "ROC-AUC": roc_auc_score(y, score),
            f"precision@{k}": p, f"recall@{k}": r}