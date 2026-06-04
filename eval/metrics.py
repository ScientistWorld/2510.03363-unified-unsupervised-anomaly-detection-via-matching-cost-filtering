from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.uint8).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score) * 100.0)


def ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true.reshape(-1), y_score.reshape(-1)) * 100.0)


def f1_max(y_true: np.ndarray, y_score: np.ndarray, steps: int = 200) -> float:
    y_true = y_true.astype(bool).reshape(-1)
    y_score = y_score.reshape(-1)
    best = 0.0
    for thr in np.linspace(float(y_score.min()), float(y_score.max()), steps):
        pred = y_score >= thr
        tp = np.logical_and(pred, y_true).sum()
        fp = np.logical_and(pred, ~y_true).sum()
        fn = np.logical_and(~pred, y_true).sum()
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
        best = max(best, float(f1))
    return best * 100.0


def aupro(masks: np.ndarray, maps: np.ndarray, max_fpr: float = 0.3, steps: int = 100) -> float:
    masks = masks.astype(bool)
    maps = maps.astype(np.float64)
    thresholds = np.linspace(float(maps.max()), float(maps.min()), steps)
    pros, fprs = [], []
    negatives = ~masks
    neg_count = negatives.sum()
    for thr in thresholds:
        pred = maps >= thr
        fp = np.logical_and(pred, negatives).sum()
        fpr = fp / max(1, neg_count)
        if fpr > max_fpr:
            continue
        per_region = []
        for mask, pm in zip(masks, pred):
            if mask.any():
                per_region.append(np.logical_and(mask, pm).sum() / mask.sum())
        if per_region:
            pros.append(float(np.mean(per_region)))
            fprs.append(float(fpr))
    if len(fprs) < 2:
        return 0.0
    order = np.argsort(fprs)
    fprs = np.asarray(fprs)[order]
    pros = np.asarray(pros)[order]
    if fprs[-1] < max_fpr:
        fprs = np.r_[fprs, max_fpr]
        pros = np.r_[pros, pros[-1]]
    return float(np.trapz(pros, fprs) / max_fpr * 100.0)
