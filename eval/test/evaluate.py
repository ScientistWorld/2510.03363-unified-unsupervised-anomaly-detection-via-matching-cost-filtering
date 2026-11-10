"""Hidden-test evaluator for the RGB multi-class anomaly-detection gym.

Scores every submitted method directory against labels and masks that only
eval/test/ holds. A method contributes raw outputs only:

    outputs/<method>/predictions.npz
        ids            (M,)          int    one row per test input id, each exactly once
        image_scores   (M,)          float  higher = more anomalous
        anomaly_maps   (M, 224, 224) float  higher = more anomalous

Everything else - which ids are scored, labels, masks, metric definitions,
the averaging protocol, the task gate - is fixed here. Any other array or
file the method writes (self-reported costs, labels, masks, scores, flags) is
ignored.

Protocol (the paper's, Tables I/IV/XXIV): every metric is computed per
category over that category's scored images and then averaged over the 15
categories with equal weight ("class-wise mean"). Rank-based metrics are
therefore unaffected by how scores are scaled from one category to another.

A run is scored as failed - every metric set to 0, its worst value - when the
artifact is missing or unreadable, does not cover the full id set exactly
once, has the wrong shape or dtype, contains non-finite values, or does not
solve both halves of the task (class-mean image AUROC or pixel AUROC below
TASK_GATE; chance is 50).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from eval.test.metrics import auroc, image_metrics, pixel_metrics

ROOT = Path(__file__).resolve().parents[2]
ANSWERS = ROOT / "eval" / "test" / "downloads" / "test_answers.npz"
EXPERIMENT = "rgb_mvtec_multiclass"
SLICE = "test"
MAP_SIZE = 224
TASK_GATE = 70.0  # class-mean image AUROC AND pixel AUROC must clear this (chance = 50)
METRICS = (
    "image_auroc",
    "pixel_auroc",
    "image_ap",
    "image_f1_max",
    "pixel_ap",
    "pixel_f1_max",
    "pixel_aupro",
)


class ArtifactError(Exception):
    pass


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


def failed(reason: str, diag: dict | None = None) -> tuple[dict[str, float], str, dict]:
    return {m: 0.0 for m in METRICS}, f"failed: {reason}", diag or {}


def load_artifact(path: Path, all_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise ArtifactError("no predictions.npz")
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - any unreadable artifact fails
        raise ArtifactError(f"unreadable artifact ({type(exc).__name__})") from exc
    if not isinstance(data, np.lib.npyio.NpzFile):
        raise ArtifactError("artifact is not an .npz archive")
    with data:
        missing = [k for k in ("ids", "image_scores", "anomaly_maps") if k not in data.files]
        if missing:
            raise ArtifactError(f"missing arrays {missing}")
        try:
            ids = np.asarray(data["ids"])
            scores = np.asarray(data["image_scores"])
            maps = data["anomaly_maps"]
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(f"unreadable arrays ({type(exc).__name__})") from exc
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ArtifactError("ids must be a 1-D integer array")
    ids = ids.astype(np.int64)
    if ids.size != all_ids.size or np.unique(ids).size != ids.size or not np.array_equal(np.sort(ids), np.sort(all_ids)):
        raise ArtifactError("ids must list every test input exactly once")
    if scores.shape != ids.shape or not _real(scores.dtype):
        raise ArtifactError("image_scores must have one real number per id")
    if maps.shape != (ids.size, MAP_SIZE, MAP_SIZE) or not _real(maps.dtype):
        raise ArtifactError(f"anomaly_maps must be real numbers of shape (n_ids, {MAP_SIZE}, {MAP_SIZE})")
    if not np.all(np.isfinite(np.asarray(scores, dtype=np.float64))):
        raise ArtifactError("non-finite image scores")
    return ids, scores, maps


def score_artifact(path: Path, answers: dict[str, np.ndarray]) -> tuple[dict[str, float], str, dict]:
    try:
        ids, scores, maps = load_artifact(path, answers["all_ids"].astype(np.int64))
        position = {int(v): i for i, v in enumerate(ids)}
        rows = np.asarray([position[int(v)] for v in answers["ids"]], dtype=np.int64)
        sel_scores = scores[rows].astype(np.float64)
        sel_maps = np.asarray(maps[rows], dtype=np.float32)
        del maps
        if not np.all(np.isfinite(sel_maps)):
            raise ArtifactError("non-finite anomaly maps")
        if not np.all(np.isfinite(sel_scores)):
            raise ArtifactError("non-finite image scores")

        labels = answers["labels"].astype(bool)
        masks = answers["masks"].astype(bool)
        categories = answers["categories"]
        names = [str(n) for n in answers["category_names"]]
        per_category: dict[str, dict[str, float]] = {}
        for ci in np.unique(categories):
            keep = categories == ci
            row = image_metrics(labels[keep], sel_scores[keep])
            row.update(pixel_metrics(masks[keep], sel_maps[keep]))
            per_category[names[int(ci)]] = {m: float(row[m]) for m in METRICS}
        result = {m: float(np.mean([per_category[c][m] for c in per_category])) for m in METRICS}
        diag = {
            "per_category": {c: {m: round(v, 4) for m, v in r.items()} for c, r in per_category.items()},
            "pooled_over_categories": {
                "image_auroc": float(auroc(labels, sel_scores)),
                "pixel_auroc": float(auroc(masks, sel_maps)),
            },
        }
    except ArtifactError as exc:
        return failed(str(exc))
    except MemoryError:
        return failed("artifact too large to score")
    except Exception as exc:  # noqa: BLE001 - never let a malformed artifact crash the scoring of others
        return failed(f"scoring error ({type(exc).__name__})")

    if not all(np.isfinite(v) for v in result.values()):
        return failed("metric undefined", diag)
    if result["image_auroc"] < TASK_GATE or result["pixel_auroc"] < TASK_GATE:
        reason = (
            f"task gate: image_auroc={result['image_auroc']:.2f}, pixel_auroc={result['pixel_auroc']:.2f} "
            f"(class means; both must be >= {TASK_GATE})"
        )
        return failed(reason, diag)
    return result, "ok", diag


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
            "categories": int(np.unique(answers["categories"]).size),
            "map_size": MAP_SIZE,
        },
        "_protocol": "every metric is computed per category, then averaged over categories with equal weight",
        "_task_gate": f"class-mean image_auroc and pixel_auroc must both be >= {TASK_GATE}; otherwise every metric is 0",
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
    parser.add_argument("--predictions", type=Path, nargs="+", required=True,
                        help="outputs/<method>/predictions.npz paths (a missing file scores that method as failed)")
    parser.add_argument("--scores", type=Path, default=ROOT / "scoring" / "scores_test.json")
    args = parser.parse_args()
    scores = evaluate(args.predictions)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.scores.with_suffix(".tmp")
    tmp.write_text(json.dumps(scores, indent=2, sort_keys=True))
    tmp.replace(args.scores)


if __name__ == "__main__":
    main()
