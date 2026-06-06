from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from eval.metrics import ap, auroc, aupro, f1_max


EXPERIMENT_NAME = "rgb_mvtec_multiclass"
SLICE = "test"
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
            keep[np.asarray(ordered[midpoint:], dtype=np.int64)] = True
    return keep


def summarize(
    labels: np.ndarray,
    masks: np.ndarray,
    image_scores: np.ndarray,
    maps: np.ndarray,
    params_m: float,
) -> dict[str, float]:
    return {
        "image_auroc": auroc(labels, image_scores),
        "pixel_auroc": auroc(masks, maps),
        "ucf_params_M": float(params_m),
        "image_ap": ap(labels, image_scores),
        "pixel_f1_max": f1_max(masks, maps),
        "pixel_aupro": aupro(masks, maps),
    }


def load_params_m(prediction_path: Path, data: np.lib.npyio.NpzFile) -> float:
    for key in ("params_m", "ucf_params_M", "method_params_M", "added_params_M"):
        if key in data.files:
            return float(np.asarray(data[key]).reshape(-1)[0])
    metadata_path = prediction_path.parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        for key in ("params_m", "ucf_params_M", "method_params_M", "added_params_M"):
            if key in metadata:
                return float(metadata[key])
    return 0.16


def unique_name(base: str, used: set[str], prediction_path: Path) -> str:
    if base not in used:
        used.add(base)
        return base
    suffix = hashlib.blake2b(str(prediction_path).encode("utf-8"), digest_size=4).hexdigest()
    name = f"{base}_{suffix}"
    used.add(name)
    return name


def expected_methods(train_scores_path: Path) -> set[str]:
    if not train_scores_path.exists():
        raise FileNotFoundError(
            f"{train_scores_path} does not exist; run scripts/evaluate_train.sh before scripts/evaluate_test.sh"
        )
    data = json.loads(train_scores_path.read_text())
    experiments = data.get("experiments", {})
    experiment = experiments.get(EXPERIMENT_NAME, {})
    results = experiment.get("results", {})
    return {name for name in results if not name.startswith("_")}


def evaluate_predictions(prediction_paths: list[Path], train_scores_path: Path) -> dict:
    if not prediction_paths:
        raise ValueError("at least one prediction artifact is required")

    results: dict[str, dict[str, float]] = {}
    used_names: set[str] = set()
    split_counts: dict[str, dict[str, int | str]] = {}

    for artifact_index, prediction_path in enumerate(prediction_paths):
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
        masks = data["masks"]
        keep = split_mask(data["categories"], labels, data["indices"])
        split_counts[str(prediction_path)] = {
            "slice": SLICE,
            "selected": int(keep.sum()),
            "total": int(labels.shape[0]),
        }

        if artifact_index == 0:
            results["anomaldf"] = summarize(
                labels[keep],
                masks[keep],
                data["baseline_scores"][keep],
                data["baseline_maps"][keep],
                0.0,
            )

        method_name = unique_name(stable_method_name(prediction_path), used_names, prediction_path)
        results[method_name] = summarize(
            labels[keep],
            masks[keep],
            data["fused_scores"][keep],
            data["fused_maps"][keep],
            load_params_m(prediction_path, data),
        )

    expected = expected_methods(train_scores_path)
    missing = sorted(expected.difference(results))
    extra = sorted(set(results).difference(expected))
    if missing:
        raise ValueError(
            "held-out test slice could not score every method from scores_train.json; "
            f"missing: {', '.join(missing)}"
        )
    if extra:
        results = {name: results[name] for name in sorted(expected)}

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
        help="One or more pre-generated prediction artifacts to score as held-out test-slice methods.",
    )
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_test.json"))
    parser.add_argument("--train-scores", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()

    scores = evaluate_predictions(args.predictions, args.train_scores)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2, sort_keys=True))
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
