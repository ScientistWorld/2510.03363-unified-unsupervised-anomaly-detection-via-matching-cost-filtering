"""Exact anomaly-detection metrics computed from raw scores (hidden-test copy).

All curve metrics come from one descending sort of the scores, grouped at
distinct score values, so ties are handled exactly (AUROC equals
sklearn.roc_auc_score; AP equals sklearn.average_precision_score).
Values are percentages.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy<2 fallback
_EIGHT = np.ones((3, 3), dtype=bool)


def _curve(y_true: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None):
    s = np.asarray(scores, dtype=np.float32).ravel()
    y = np.asarray(y_true, dtype=bool).ravel()
    order = np.argsort(s)[::-1]
    s_sorted = s[order]
    ends = np.r_[np.flatnonzero(s_sorted[1:] != s_sorted[:-1]), s_sorted.size - 1]
    del s_sorted
    counts = np.cumsum(y[order], dtype=np.int64)
    tps = counts[ends].astype(np.float64)
    del counts
    fps = (ends + 1).astype(np.float64) - tps
    cum_w = None
    if weights is not None:
        w_sorted = np.asarray(weights, dtype=np.float64).ravel()[order]
        np.cumsum(w_sorted, out=w_sorted)
        cum_w = w_sorted[ends]
        del w_sorted
    return tps, fps, cum_w


def _auroc(tps: np.ndarray, fps: np.ndarray) -> float:
    p, n = tps[-1], fps[-1]
    if p == 0 or n == 0:
        return float("nan")
    return float(_trapezoid(np.r_[0.0, tps / p], np.r_[0.0, fps / n]) * 100.0)


def _ap(tps: np.ndarray, fps: np.ndarray) -> float:
    p = tps[-1]
    if p == 0:
        return float("nan")
    precision = tps / np.maximum(tps + fps, 1.0)
    recall = tps / p
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision) * 100.0)


def _f1_max(tps: np.ndarray, fps: np.ndarray) -> float:
    p = tps[-1]
    if p == 0:
        return float("nan")
    return float(np.max(2.0 * tps / (tps + fps + p)) * 100.0)


def _aupro(fps: np.ndarray, cum_pro: np.ndarray, max_fpr: float = 0.3) -> float:
    n = fps[-1]
    if n == 0:
        return float("nan")
    fpr = np.r_[0.0, fps / n]
    pro = np.r_[0.0, cum_pro]
    idx = int(np.searchsorted(fpr, max_fpr, side="right"))
    x, y = fpr[:idx], pro[:idx]
    if idx < fpr.size:
        x0, x1, y0, y1 = fpr[idx - 1], fpr[idx], pro[idx - 1], pro[idx]
        yi = y1 if x1 <= x0 else y0 + (y1 - y0) * (max_fpr - x0) / (x1 - x0)
        x, y = np.r_[x, max_fpr], np.r_[y, yi]
    return float(_trapezoid(y, x) / max_fpr * 100.0)


def region_weights(masks: np.ndarray) -> np.ndarray:
    """Per-pixel weight 1/(|region| * #regions) over 8-connected defect regions."""
    weights = np.zeros(masks.shape, dtype=np.float64)
    sizes_all = []
    labelled = []
    for mask in masks:
        lab, count = ndimage.label(mask, structure=_EIGHT)
        labelled.append((lab, count))
        if count:
            sizes_all.append(np.bincount(lab.ravel())[1:])
    total_regions = int(sum(len(s) for s in sizes_all))
    if total_regions == 0:
        return weights
    k = 0
    for i, (lab, count) in enumerate(labelled):
        if count == 0:
            continue
        sizes = sizes_all[k]
        k += 1
        per_label = np.r_[0.0, 1.0 / (sizes.astype(np.float64) * total_regions)]
        weights[i] = per_label[lab]
    return weights


def image_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    tps, fps, _ = _curve(labels, scores)
    return {"image_auroc": _auroc(tps, fps), "image_ap": _ap(tps, fps), "image_f1_max": _f1_max(tps, fps)}


def pixel_metrics(masks: np.ndarray, maps: np.ndarray) -> dict[str, float]:
    weights = region_weights(masks)
    tps, fps, cum_pro = _curve(masks, maps, weights)
    del weights
    return {
        "pixel_auroc": _auroc(tps, fps),
        "pixel_ap": _ap(tps, fps),
        "pixel_f1_max": _f1_max(tps, fps),
        "pixel_aupro": _aupro(fps, cum_pro, 0.3),
    }


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    tps, fps, _ = _curve(y_true, scores)
    return _auroc(tps, fps)
