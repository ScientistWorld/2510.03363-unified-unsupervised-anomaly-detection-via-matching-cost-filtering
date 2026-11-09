"""Genuine normal-only RGB guide calibration added to trained UCF outputs.

Fit local normal-patch distributions using only normal training RGB. Preserve
the trained UCF maps and blend the calibrated evidence; choose the operating
point with the visible train evaluator's slice only. No test labels enter fit.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from data.synthetic_uad import SyntheticUADDataset
from eval.train.evaluate import split_mask
from redteam.methods import load_artifact, save_artifact, topk, AUDIT


def patches(image, width):
    padded = F.pad(image[None], (width // 2,) * 4, mode='reflect')
    return F.unfold(padded, width)[0].T.numpy().astype(np.float64)


def main(folder):
    arrays = load_artifact(AUDIT / 'reference_seed7/artifact/predictions.npz')
    train = SyntheticUADDataset('train', include_synthetic_train_anomalies=False)
    test = SyntheticUADDataset('test')
    categories = arrays['categories']
    visible = split_mask(categories, arrays['labels'], arrays['indices'])
    train_samples = [train[i] for i in range(len(train))]
    # Test images only. Masks are read separately for the output contract and
    # visible-train validation, never used for distribution fitting/inference.
    images = [test[i]['image'] for i in range(len(test))]
    candidate_records = []
    best = None
    for width in (3, 5):
        dims = 3 * width * width
        raw_maps = np.zeros_like(arrays['fused_maps'], dtype=np.float64)
        learned = []
        for cat in np.unique(categories):
            x = np.concatenate([patches(s['image'], width) for s in train_samples if int(s['category']) == cat])
            mean = x.mean(axis=0)
            centered = x - mean
            covariance = centered.T @ centered / (len(x) - 1)
            covariance += np.eye(dims) * (0.01 * np.trace(covariance) / dims)
            chol = np.linalg.cholesky(covariance)
            inverse = np.linalg.solve(chol, np.eye(dims)).T
            train_distance = np.square(centered @ inverse).sum(axis=1)
            location = float(train_distance.mean())
            scale = float(train_distance.std())
            learned.append(dict(category=int(cat), mean=mean.tolist(), precision_factor=inverse.tolist(),
                                normal_location=location, normal_scale=scale))
            for i in np.flatnonzero(categories == cat):
                z = patches(images[i], width) - mean
                distance = np.square(z @ inverse).sum(axis=1)
                standardized = (distance - location) / max(scale, 1e-6)
                raw_maps[i] = standardized.reshape(raw_maps.shape[-2:])
        for smoothing in (1, 3):
            raw = torch.from_numpy(raw_maps[:, None]).float()
            if smoothing != 1:
                raw = F.avg_pool2d(F.pad(raw, (1,) * 4, mode='reflect'), 3, stride=1)
            calibrated = torch.sigmoid(raw[:, 0].clamp(-15, 15)).numpy()
            # A nonzero UCF contribution remains in every candidate.
            for fraction in (0.25, 0.5, 0.75):
                maps = (1 - fraction) * arrays['fused_maps'] + fraction * calibrated
                pixel_auc = roc_auc_score(arrays['masks'][visible].ravel(), maps[visible].ravel())
                pixel_ap = average_precision_score(arrays['masks'][visible].ravel(), maps[visible].ravel())
                for k in (1, 16, 64, 250):
                    scores = topk(maps, k)
                    image_auc = roc_auc_score(arrays['labels'][visible], scores[visible])
                    image_ap = average_precision_score(arrays['labels'][visible], scores[visible])
                    objective = pixel_auc + 0.8 * image_auc + 0.35 * image_ap + 0.25 * pixel_ap
                    row = dict(patch_width=width, smoothing=smoothing, calibration_fraction=fraction, topk=k,
                               visible_train_objective=objective, visible_train_pixel_auroc=100 * pixel_auc,
                               visible_train_image_auroc=100 * image_auc)
                    candidate_records.append(row)
                    if best is None or objective > best[0]:
                        extra_params = len(learned) * (dims + dims * dims + 2) / 1e6
                        extra_flops = (48 * 48 * (2 * dims * dims + 4 * dims + 20)) / 1e9
                        best = (objective, row, maps.copy(), scores.copy(), learned, extra_params, extra_flops)
    arrays['fused_maps'], arrays['fused_scores'] = best[2], best[3]
    arrays['added_params_M'] = arrays['added_params_M'] + best[5]
    arrays['added_flops_G'] = arrays['added_flops_G'] + best[6]
    save_artifact(folder, arrays)
    (folder / 'normal_model.json').write_text(json.dumps(best[4]))
    (folder.parent / 'selection.json').write_text(json.dumps(dict(selected=best[1], candidates=candidate_records,
        training='108 normal train images only; original RGB guide source and UCF retained',
        extra_params_M=best[5], extra_flops_G=best[6]), indent=2))
