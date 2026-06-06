from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from eval.metrics import ap, auroc, aupro, f1_max


EXPERIMENT_NAME = "rgb_mvtec_multiclass"
SLICE = "train"
SPLIT_SEED = 20261003


def stable_method_name(prediction_path: Path) -> str:
    parent = prediction_path.parent.name
    if parent == "ucf_synthetic":
        return "anomaldf_ucf"
    name = re.sub(r"[^A-Za-z0-9_]+", "_", parent).strip("_").lower()
    return name or "method"


def stable_order_key(category: int, label: int, index: int) -> int:
    raw = f"{SPLIT_SEED}:{category}:{label}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def split_mask(categories: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    categories = categories.reshape(-1)
    labels = labels.reshape(-1)
    indices = indices.reshape(-1)
    keep = np.zeros(labels.shape[0], dtype=bool)
    for category in np.unique(categories):
        for label in np.unique(labels):
            group = np.flatnonzero((categories == category) & (labels == label))
            ordered = sorted(group, key=lambda i: stable_order_key(int(category), int(label), int(indices[i])))
            midpoint = len(ordered) // 2
            keep[np.asarray(ordered[:midpoint], dtype=np.int64)] = True
    return keep


def summarize(
    categories: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    image_scores: np.ndarray,
    maps: np.ndarray,
    params_m: float,
    flops_g: float,
) -> dict[str, float]:
    return {
        "image_auroc": auroc(labels, image_scores),
        "pixel_auroc": auroc(masks, maps),
        "added_params_M": float(params_m),
        "added_flops_G": float(flops_g),
        "image_ap": ap(labels, image_scores),
        "image_f1_max": f1_max(labels, image_scores),
        "pixel_ap": ap(masks, maps),
        "pixel_f1_max": f1_max(masks, maps),
        "pixel_aupro": aupro(masks, maps),
        "pixel_aupro_5pct": aupro(masks, maps, max_fpr=0.05),
        "category_mean_image_auroc": category_mean_metric(categories, labels, image_scores, auroc),
        "category_mean_pixel_auroc": category_mean_metric(categories, masks, maps, auroc),
    }


def category_mean_metric(
    categories: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric,
) -> float:
    categories = categories.reshape(-1)
    values = []
    for category in np.unique(categories):
        keep = categories == category
        value = metric(y_true[keep], y_score[keep])
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def load_params_m(prediction_path: Path, data: np.lib.npyio.NpzFile) -> Optional[float]:
    for key in ("params_m", "added_params_M", "method_params_M", "ucf_params_M"):
        if key in data.files:
            return float(np.asarray(data[key]).reshape(-1)[0])
    metadata_path = prediction_path.parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        for key in ("params_m", "added_params_M", "method_params_M", "ucf_params_M"):
            if key in metadata:
                return float(metadata[key])
    return None


def load_flops_g(prediction_path: Path, data: np.lib.npyio.NpzFile) -> Optional[float]:
    for key in ("flops_g", "added_flops_G", "method_flops_G", "ucf_flops_G"):
        if key in data.files:
            return float(np.asarray(data[key]).reshape(-1)[0])
    metadata_path = prediction_path.parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        for key in ("flops_g", "added_flops_G", "method_flops_G", "ucf_flops_G"):
            if key in metadata:
                return float(metadata[key])
    return None


def unique_name(base: str, used: set[str], prediction_path: Path) -> str:
    if base not in used:
        used.add(base)
        return base
    suffix = hashlib.blake2b(str(prediction_path).encode("utf-8"), digest_size=4).hexdigest()
    name = f"{base}_{suffix}"
    used.add(name)
    return name


def evaluate_predictions(prediction_paths: list[Path]) -> dict:
    if not prediction_paths:
        raise ValueError("at least one prediction artifact is required")

    results: dict[str, dict[str, float]] = {}
    used_names: set[str] = set()
    split_counts: dict[str, dict[str, int | str]] = {}
    included_count = 0

    for prediction_path in prediction_paths:
        method_base = stable_method_name(prediction_path)

        data = np.load(prediction_path)
        required = {
            "labels",
            "categories",
            "indices",
            "masks",
            "baseline_scores",
            "baseline_maps",
            "fused_scores",
            "fused_maps",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{prediction_path} is missing required arrays: {', '.join(missing)}")

        labels = data["labels"]
        categories = data["categories"]
        masks = data["masks"]
        keep = split_mask(categories, labels, data["indices"])
        split_counts[str(prediction_path)] = {
            "slice": SLICE,
            "selected": int(keep.sum()),
            "total": int(labels.shape[0]),
        }

        if included_count == 0:
            results["anomaldf"] = summarize(
                categories[keep],
                labels[keep],
                masks[keep],
                data["baseline_scores"][keep],
                data["baseline_maps"][keep],
                0.0,
                0.0,
            )

        method_name = unique_name(method_base, used_names, prediction_path)
        params_m = load_params_m(prediction_path, data)
        flops_g = load_flops_g(prediction_path, data)
        if params_m is None or flops_g is None:
            raise ValueError(
                f"{prediction_path} must include added_params_M and added_flops_G in the .npz artifact or metadata.json"
            )
        results[method_name] = summarize(
            categories[keep],
            labels[keep],
            masks[keep],
            data["fused_scores"][keep],
            data["fused_maps"][keep],
            params_m,
            flops_g,
        )
        included_count += 1

    if included_count == 0:
        raise ValueError("no prediction artifacts were evaluated")

    return {
        "_slice": SLICE,
        "_split": {
            "style": "deterministic stratified 50/50 over category and anomaly label",
            "seed": SPLIT_SEED,
            "artifacts": split_counts,
        },
        "experiments": {
            EXPERIMENT_NAME: {
                "_slice": SLICE,
                "results": results,
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="One or more pre-generated prediction artifacts to score as visible train-slice methods.",
    )
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()

    scores = evaluate_predictions(args.predictions)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2, sort_keys=True))
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
