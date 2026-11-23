import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _eer(fpr, tpr):
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def threshold_at_fpr(y_true, y_prob, target_fpr=0.01):
    """在验证集上选阈值：满足 FPR<=target 的最大 FPR 点。"""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return float(thr[0])
    return float(thr[valid[-1]])


def threshold_at_best_f1(y_true, y_prob):
    """在给定数据上取使 F1 最大的阈值（只用于验证集/分析，不要用于测试集调参）。"""
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    if len(thr) == 0:
        return 0.5
    return float(thr[int(np.argmax(f1[:-1]))])


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        out["eer"] = _eer(fpr, tpr)
        out["tpr_at_fpr1"] = float(np.interp(0.01, fpr, tpr))
        out["tpr_at_fpr10"] = float(np.interp(0.10, fpr, tpr))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        fpr=float(fp / (fp + tn)) if (fp + tn) else 0.0,
            )
    return out
