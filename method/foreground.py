"""Unsupervised object-foreground masks for anomaly synthesis on object categories.

GLAD-style synthesis places Perlin-noise anomalies on the object for object
categories (textures use the whole image). The foreground is estimated
without labels from the first principal component of the frozen DINOv2
patch tokens of each category's normal training images (as in AnomalDINO's
masking), oriented so that border patches are background, dilated by one
patch. Images whose estimate covers <5% or >95% of patches fall back to the
whole image.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

OBJECT_CATEGORIES = (
    # MVTec-AD objects
    "bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor",
    # VisA single-object categories (the authors' GLAD data code restricts these to the foreground;
    # capsules and pcb1-4 use the whole image there and here)
    "candle", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pipe_fryum",
)


@torch.no_grad()
def foreground_masks(last_layer: torch.Tensor, categories: np.ndarray, names: tuple[str, ...]) -> tuple[torch.Tensor, dict]:
    """last_layer: (N,C,h,w) patch tokens -> (N,h,w) bool foreground on the same device."""
    n, c, h, w = last_layer.shape
    fg = torch.ones(n, h, w, dtype=torch.bool, device=last_layer.device)
    border = torch.zeros(h, w, dtype=torch.bool, device=last_layer.device)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    stats = {}
    for ci, name in enumerate(names):
        if name not in OBJECT_CATEGORIES:
            continue
        idx = torch.as_tensor(np.flatnonzero(categories == ci), device=last_layer.device)
        if idx.numel() == 0:
            continue
        x = last_layer[idx].float().permute(0, 2, 3, 1).reshape(-1, c)
        mu = x.mean(dim=0, keepdim=True)
        xc = x - mu
        cov = xc.T @ xc / xc.shape[0]
        _, vecs = torch.linalg.eigh(cov)
        proj = (xc @ vecs[:, -1]).view(-1, h, w)
        if proj[:, border].mean() > proj[:, ~border].mean():
            proj = -proj
        m = proj > 0
        m = F.max_pool2d(m.float()[:, None], 3, 1, 1)[:, 0] > 0
        frac = m.float().mean(dim=(1, 2))
        bad = (frac < 0.05) | (frac > 0.95)
        m[bad] = True
        fg[idx] = m
        stats[name] = {"mean_foreground_fraction": float(m.float().mean()), "fallback_images": int(bad.sum())}
    return fg, stats
