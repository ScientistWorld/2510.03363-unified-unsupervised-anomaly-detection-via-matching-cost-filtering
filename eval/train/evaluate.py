"""Development-slice (train) evaluator for the RGB multi-class anomaly-detection gym.

Scores every submitted prediction artifact against the labels and masks of the
visible development half (eval/train/data/dev_answers.npz). The hidden test
uses the same contract, metrics and task gate on the other half, whose defect
types never appear here. A method contributes raw outputs only:

    outputs/<method>/predictions.npz
        ids            (M,)          int    one row per test input id, each exactly once
        image_scores   (M,)          float  higher = more anomalous
        anomaly_maps   (M, 224, 224) float  higher = more anomalous

Everything else - which ids are scored, labels, masks, metric definitions,
the task gate - is fixed here. Any other array or file the method writes
(self-reported costs, labels, masks, scores, flags) is ignored.

A run is scored as failed (every metric = 0, the worst value) when the
artifact is unreadable, does not cover the full id set exactly once, has the
wrong shape, contains non-finite values, or does not solve both halves of
the task (image AUROC or pixel AUROC below TASK_GATE).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from eval.train.metrics import auroc, image_metrics, pixel_metrics

ROOT = Path(__file__).resolve().parents[2]
ANSWERS = ROOT / "eval" / "train" / "data" / "dev_answers.npz"
EXPERIMENT = "rgb_mvtec_multiclass"
SLICE = "train"
MAP_SIZE = 224
TASK_GATE = 60.0  # image AND pixel AUROC must clear this (chance = 50)
METRICS = (
    "image_auroc",
    "pixel_auroc",
    "image_ap",
    "image_f1_max",
    "pixel_ap",
    "pixel_f1_max",
    "pixel_aupro",
)


def method_name(path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", path.parent.name).strip("_").lower()
    return name or "method"


def load_answers(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"{path} missing; run scripts/prepare_data.sh")
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def _real(dtype: np.dtype) -> bool:
    return bool(np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer))


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def failed(reason: str) -> tuple[dict[str, float], str, dict]:
    return {m: 0.0 for m in METRICS}, f"failed: {reason}", {}


def score_artifact(path: Path, answers: dict[str, np.ndarray]) -> tuple[dict[str, float], str, dict]:
    all_ids = answers["all_ids"].astype(np.int64)
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - any unreadable artifact fails
        return failed(f"unreadable artifact ({type(exc).__name__})")
    if not isinstance(data, np.lib.npyio.NpzFile):
        return failed("artifact is not an .npz archive")
    with data:
        missing = [k for k in ("ids", "image_scores", "anomaly_maps") if k not in data.files]
        if missing:
            return failed(f"missing arrays {missing}")
        try:
            ids = np.asarray(data["ids"])
            scores = np.asarray(data["image_scores"])
            maps = data["anomaly_maps"]
        except Exception as exc:  # noqa: BLE001
            return failed(f"unreadable arrays ({type(exc).__name__})")
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        return failed("ids must be a 1-D integer array")
    ids = ids.astype(np.int64)
    if ids.size != all_ids.size or np.unique(ids).size != ids.size or not np.array_equal(np.sort(ids), np.sort(all_ids)):
        return failed("ids must list every test input exactly once")
    if scores.shape != ids.shape or not _real(scores.dtype):
        return failed("image_scores must have one real number per id")
    if maps.shape != (ids.size, MAP_SIZE, MAP_SIZE) or not _real(maps.dtype):
        return failed(f"anomaly_maps must be real numbers of shape (n_ids, {MAP_SIZE}, {MAP_SIZE})")

    position = {int(v): i for i, v in enumerate(ids)}
    rows = np.asarray([position[int(v)] for v in answers["ids"]], dtype=np.int64)
    sel_scores = scores[rows].astype(np.float64)
    sel_maps = np.asarray(maps[rows], dtype=np.float32)
    del maps
    if not np.all(np.isfinite(sel_scores)) or not np.all(np.isfinite(sel_maps)):
        return failed("non-finite image scores or anomaly maps")
    if not np.all(np.isfinite(np.asarray(scores, dtype=np.float64))):
        return failed("non-finite image scores")

    labels = answers["labels"].astype(bool)
    masks = answers["masks"].astype(bool)
    result = image_metrics(labels, sel_scores)
    result.update(pixel_metrics(masks, sel_maps))
    result = {m: float(result[m]) for m in METRICS}
    if not all(np.isfinite(v) for v in result.values()):
        return failed("metric undefined")

    categories = answers["categories"]
    names = [str(n) for n in answers["category_names"]]
    per_category = {}
    for ci in np.unique(categories):
        keep = categories == ci
        per_category[names[int(ci)]] = {
            "image_auroc": _finite_or_none(auroc(labels[keep], sel_scores[keep])),
            "pixel_auroc": _finite_or_none(auroc(masks[keep], sel_maps[keep])),
        }
    if result["image_auroc"] < TASK_GATE or result["pixel_auroc"] < TASK_GATE:
        reason = (
            f"task gate: image_auroc={result['image_auroc']:.2f}, pixel_auroc={result['pixel_auroc']:.2f} "
            f"(both must be >= {TASK_GATE})"
        )
        return {m: 0.0 for m in METRICS}, f"failed: {reason}", {"per_category": per_category}
    return result, "ok", {"per_category": per_category}


def evaluate(paths: list[Path]) -> dict:
    answers = load_answers(ANSWERS)
    results, validity, diagnostics, sources = {}, {}, {}, {}
    for path in sorted(paths):
        name = method_name(path)
        if name in results:
            raise SystemExit(f"two artifacts map to method name '{name}'")
        metrics, status, diag = score_artifact(path, answers)
        results[name] = metrics
        validity[name] = status
        diagnostics[name] = diag
        sources[name] = str(path)
        print(f"[{SLICE}] {name}: {status} " + json.dumps({k: round(v, 3) for k, v in metrics.items()}), flush=True)
    return {
        "_slice": SLICE,
        "_answers": {
            "images": int(answers["ids"].size),
            "anomalous": int(answers["labels"].sum()),
            "normal": int((answers["labels"] == 0).sum()),
            "map_size": MAP_SIZE,
        },
        "_task_gate": f"image_auroc and pixel_auroc must both be >= {TASK_GATE}; otherwise every metric is 0",
        "experiments": {
            EXPERIMENT: {
                "_slice": SLICE,
                "_validity": validity,
                "_sources": sources,
                "_diagnostics": diagnostics,
                "results": results,
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--scores", type=Path, default=ROOT / "scoring" / "scores_train.json")
    args = parser.parse_args()
    scores = evaluate(args.predictions)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
