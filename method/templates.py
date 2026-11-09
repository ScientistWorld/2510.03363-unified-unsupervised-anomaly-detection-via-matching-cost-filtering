"""Normal-template selection for few-template matching.

Paper protocol (AnomalDF / AnomalDF+UCF): N = 3 templates are sampled at
random from the same-category training set, both in training and testing.
Test-time draws are seeded per test id, so every method that uses this
helper matches each input against the same templates.
"""
from __future__ import annotations

import numpy as np
import torch


def category_index(train_categories: np.ndarray) -> dict[int, np.ndarray]:
    return {int(c): np.flatnonzero(train_categories == c) for c in np.unique(train_categories)}


def train_templates(index: dict[int, np.ndarray], category: int, n: int, rng: np.random.Generator, exclude: int) -> np.ndarray:
    pool = index[int(category)]
    pool = pool[pool != exclude]
    return rng.choice(pool, size=n, replace=False)


def test_templates(index: dict[int, np.ndarray], category: int, n: int, seed: int, draw: int, sample_id: int) -> np.ndarray:
    rng = np.random.default_rng([int(seed), int(draw), int(sample_id)])
    return rng.choice(index[int(category)], size=n, replace=False)


def nearest_templates(
    index: dict[int, np.ndarray], category: int, n: int, query_desc: torch.Tensor, train_desc: torch.Tensor
) -> np.ndarray:
    """Top-n same-category training images by cosine similarity of global descriptors."""
    pool = index[int(category)]
    sims = train_desc[torch.as_tensor(pool, device=train_desc.device)] @ query_desc
    top = torch.topk(sims, k=n).indices.cpu().numpy()
    return pool[top]
