"""Method-visible arrays of the two benchmarks (224x224 RGB, uint8).

Two datasets, one scored experiment each:

  mvtec  MVTec-AD, 15 categories   data/downloads/mvtec224/   -> outputs/<method>/predictions.npz
  visa   VisA, 12 categories       data/downloads/visa224/    -> outputs/<method>/predictions_visa.npz

  load_train(dataset)        -> images (N,224,224,3), categories (N,)       normal images only
  load_test_inputs(dataset)  -> images (M,224,224,3), categories (M,), ids (M,)
                                every test image under an anonymous id; no labels or masks
  category_names(dataset)    -> tuple of category names (index = category id)
  load_textures()            -> DTD textures (K,224,224,3) for anomaly synthesis
  write_predictions(out_dir, ids, scores, maps, dataset)

The arrays are produced by scripts/prepare_data.sh from the archives fetched by
scripts/download.sh.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEXTURES = ROOT / "data" / "downloads" / "dtd" / "dtd224.npy"
IMAGE_SIZE = 224
CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)
VISA_CATEGORIES = (
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
)
DATASETS = {
    "mvtec": {"dir": ROOT / "data" / "downloads" / "mvtec224", "categories": CATEGORIES, "predictions": "predictions.npz"},
    "visa": {"dir": ROOT / "data" / "downloads" / "visa224", "categories": VISA_CATEGORIES, "predictions": "predictions_visa.npz"},
}
MVTEC_DIR = DATASETS["mvtec"]["dir"]


def _spec(dataset: str) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")
    return DATASETS[dataset]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/download.sh then scripts/prepare_data.sh")
    return path


def category_names(dataset: str = "mvtec") -> tuple[str, ...]:
    return _spec(dataset)["categories"]


def predictions_name(dataset: str = "mvtec") -> str:
    return _spec(dataset)["predictions"]


def metadata_name(dataset: str = "mvtec") -> str:
    """Free-form run metadata written next to the predictions (never read by the evaluators)."""
    return "metadata.json" if dataset == "mvtec" else f"metadata_{dataset}.json"


def load_train(dataset: str = "mvtec") -> tuple[np.ndarray, np.ndarray]:
    with np.load(_require(_spec(dataset)["dir"] / "train.npz")) as d:
        return d["images"], d["categories"]


def load_test_inputs(dataset: str = "mvtec") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(_require(_spec(dataset)["dir"] / "test_inputs.npz")) as d:
        return d["images"], d["categories"], d["ids"]


def load_textures() -> np.ndarray:
    return np.load(_require(TEXTURES))


def write_predictions(out_dir: Path, ids: np.ndarray, image_scores: np.ndarray, anomaly_maps: np.ndarray, dataset: str = "mvtec") -> Path:
    """Write the evaluation contract for one dataset: one row per test input id."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / predictions_name(dataset)
    tmp = out_dir / (path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        ids=np.asarray(ids, dtype=np.int64),
        image_scores=np.asarray(image_scores, dtype=np.float32),
        anomaly_maps=np.asarray(anomaly_maps, dtype=np.float32),
    )
    tmp.replace(path)
    return path
