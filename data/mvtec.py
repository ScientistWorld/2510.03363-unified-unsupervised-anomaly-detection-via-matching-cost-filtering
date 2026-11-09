"""Method-visible MVTec-AD arrays (224x224 RGB, uint8).

  load_train()        -> images (N,224,224,3), categories (N,)       normal images only
  load_test_inputs()  -> images (M,224,224,3), categories (M,), ids (M,)
                         every test image under an anonymous id; no labels or masks
  load_textures()     -> DTD textures (K,224,224,3) for anomaly synthesis

The arrays are produced by scripts/prepare_data.sh from the archives fetched by
scripts/download.sh.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MVTEC_DIR = ROOT / "data" / "downloads" / "mvtec224"
TEXTURES = ROOT / "data" / "downloads" / "dtd" / "dtd224.npy"
IMAGE_SIZE = 224
CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/download.sh then scripts/prepare_data.sh")
    return path


def load_train() -> tuple[np.ndarray, np.ndarray]:
    with np.load(_require(MVTEC_DIR / "train.npz")) as d:
        return d["images"], d["categories"]


def load_test_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(_require(MVTEC_DIR / "test_inputs.npz")) as d:
        return d["images"], d["categories"], d["ids"]


def load_textures() -> np.ndarray:
    return np.load(_require(TEXTURES))


def write_predictions(out_dir: Path, ids: np.ndarray, image_scores: np.ndarray, anomaly_maps: np.ndarray) -> Path:
    """Write the evaluation contract: one row per test input id."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.npz"
    tmp = out_dir / "predictions.tmp.npz"
    np.savez_compressed(
        tmp,
        ids=np.asarray(ids, dtype=np.int64),
        image_scores=np.asarray(image_scores, dtype=np.float32),
        anomaly_maps=np.asarray(anomaly_maps, dtype=np.float32),
    )
    tmp.replace(path)
    return path
