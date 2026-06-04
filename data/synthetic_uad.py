"""Deterministic multi-class anomaly data for compact UCF reproduction.

The original benchmarks are industrial/medical anomaly datasets. This module
provides a small procedural substitute with the same normal-only training
regime, image-level labels, pixel masks, and per-class normal template pool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticConfig:
    image_size: int = 48
    train_per_class: int = 36
    test_normal_per_class: int = 10
    test_anomaly_per_class: int = 18
    template_count: int = 2
    categories: tuple[str, ...] = ("stripes", "rings", "grid")


def _mesh(size: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    return x / size, y / size


def _normal_pattern(category: int, size: int, rng: np.random.Generator) -> np.ndarray:
    x, y = _mesh(size)
    phase = rng.uniform(0, 2 * math.pi)
    freq = rng.uniform(5.0, 8.0)
    if category == 0:
        base = 0.5 + 0.25 * np.sin((x * freq + 0.35 * y) * 2 * math.pi + phase)
        aux = 0.5 + 0.18 * np.sin((y * (freq * 0.6)) * 2 * math.pi - phase)
    elif category == 1:
        r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        base = 0.5 + 0.28 * np.sin(r * freq * 2 * math.pi + phase)
        aux = 0.5 + 0.15 * np.cos((x + y) * freq * math.pi)
    else:
        base = 0.5 + 0.22 * np.sign(np.sin(x * freq * 2 * math.pi + phase))
        aux = 0.5 + 0.22 * np.sign(np.sin(y * (freq * 0.9) * 2 * math.pi - phase))
    img = np.stack([base, aux, 0.55 * base + 0.45 * aux], axis=0)
    img += rng.normal(0.0, 0.035, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _smooth_noise(size: int, rng: np.random.Generator, grid: int = 8) -> np.ndarray:
    coarse = rng.normal(size=(grid, grid)).astype(np.float32)
    t = torch.from_numpy(coarse)[None, None]
    up = torch.nn.functional.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)
    arr = up[0, 0].numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return arr


def _anomaly_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    x, y = _mesh(size)
    cx, cy = rng.uniform(0.2, 0.8, size=2)
    rx, ry = rng.uniform(0.07, 0.18, size=2)
    angle = rng.uniform(0, math.pi)
    ca, sa = math.cos(angle), math.sin(angle)
    xr = (x - cx) * ca + (y - cy) * sa
    yr = -(x - cx) * sa + (y - cy) * ca
    blob = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0
    ragged = _smooth_noise(size, rng, grid=10) > rng.uniform(0.35, 0.55)
    mask = np.logical_and(blob, ragged)
    if mask.sum() < size * size * 0.01:
        mask = blob
    return mask.astype(np.float32)


def add_synthetic_anomaly(img: np.ndarray, category: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    size = img.shape[-1]
    mask = _anomaly_mask(size, rng)
    corrupt = img.copy()
    texture = _normal_pattern((category + 1) % 3, size, rng)
    tint = rng.uniform(0.15, 0.35)
    corrupt = corrupt * (1.0 - mask) + np.clip((1 - tint) * texture + tint * rng.random((3, 1, 1)), 0, 1) * mask
    edge_noise = rng.normal(0, 0.05, size=corrupt.shape).astype(np.float32) * mask
    return np.clip(corrupt + edge_noise, 0.0, 1.0).astype(np.float32), mask.astype(np.float32)


class SyntheticUADDataset(Dataset):
    def __init__(self, split: str, config: SyntheticConfig | None = None, include_synthetic_train_anomalies: bool = False):
        self.config = config or SyntheticConfig()
        self.split = split
        self.include_synthetic_train_anomalies = include_synthetic_train_anomalies
        self.samples: list[tuple[int, int, int]] = []
        for category in range(len(self.config.categories)):
            if split == "train":
                for idx in range(self.config.train_per_class):
                    self.samples.append((category, idx, 0))
            elif split == "test":
                for idx in range(self.config.test_normal_per_class):
                    self.samples.append((category, idx, 0))
                for idx in range(self.config.test_anomaly_per_class):
                    self.samples.append((category, idx, 1))
            else:
                raise ValueError(f"unknown split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def normal_image(self, category: int, idx: int, offset: int = 0) -> np.ndarray:
        seed = 10_000 * category + 97 * idx + offset
        rng = np.random.default_rng(seed)
        return _normal_pattern(category, self.config.image_size, rng)

    def template_bank(self, category: int, anchor_idx: int) -> np.ndarray:
        templates = []
        for k in range(self.config.template_count):
            idx = (anchor_idx * 13 + k * 17 + 5) % self.config.train_per_class
            templates.append(self.normal_image(category, idx, offset=33 + k))
        return np.stack(templates, axis=0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        category, idx, is_anom = self.samples[index]
        image = self.normal_image(category, idx, offset=1000 if self.split == "test" else 0)
        mask = np.zeros((self.config.image_size, self.config.image_size), dtype=np.float32)
        label = is_anom
        if self.split == "train" and self.include_synthetic_train_anomalies and index % 2 == 1:
            image, mask = add_synthetic_anomaly(image, category, 99_000 + index)
            label = 1
        elif self.split == "test" and is_anom:
            image, mask = add_synthetic_anomaly(image, category, 88_000 + 101 * category + idx)
        templates = self.template_bank(category, idx)
        return {
            "image": torch.from_numpy(image),
            "templates": torch.from_numpy(templates),
            "mask": torch.from_numpy(mask).long(),
            "label": torch.tensor(label, dtype=torch.long),
            "category": torch.tensor(category, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


def iter_dataset_summary(config: SyntheticConfig | None = None) -> Iterable[str]:
    cfg = config or SyntheticConfig()
    yield f"image_size={cfg.image_size}"
    yield f"categories={','.join(cfg.categories)}"
    yield f"train={cfg.train_per_class * len(cfg.categories)} normal images"
    yield f"test={(cfg.test_normal_per_class + cfg.test_anomaly_per_class) * len(cfg.categories)} images"
