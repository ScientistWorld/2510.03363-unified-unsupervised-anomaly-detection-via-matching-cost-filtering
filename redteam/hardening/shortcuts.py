"""Shortcut payloads re-run against the hardened hidden-test evaluator.

Every payload is built only from method-visible material (the reference
prediction artifact and the public test-input ids); none reads eval/test/.
Payloads are written to redteam/hardening/outputs/shortcut_<name>/ and scored
by run_study.sh with the unchanged evaluator (python -m eval.test.evaluate).
Two cases need the evaluator invoked in a specific way and are scored here:
forged_scores (a forged score file already sits at the output path) and
baseline_replacement (the baseline artifact is replaced next to the
reference artifact).

missing_artifact is a method directory with only a metadata file claiming
perfect results (a crashed or abandoned run): it must score its worst value,
not drop out of the ranking. category_prior shifts every category's image
scores and maps by an offset ordered by that category's anomaly rate on the
visible development slice (the hidden half has the same per-category split
ratio): it exploits metrics pooled over categories and leaves every
within-category ranking unchanged.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "eval" / "train" / "data" / "dev_answers.npz"  # visible development answers
REF = ROOT / "outputs" / "anomaldf_ucf" / "predictions.npz"
OUT = ROOT / "redteam" / "hardening" / "outputs"
AUX = ROOT / "redteam" / "hardening" / "aux_outputs"
SC = ROOT / "redteam" / "hardening" / "scores"


def save(folder: Path, **arrays) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "predictions.npz", **arrays)
    return folder


def evaluate(paths: list[Path], scores: Path, slice_name: str = "test") -> None:
    subprocess.run(
        [sys.executable, "-m", f"eval.{slice_name}.evaluate", "--predictions", *map(str, paths), "--scores", str(scores)],
        check=True,
        cwd=ROOT,
    )


def main() -> None:
    SC.mkdir(parents=True, exist_ok=True)
    with np.load(REF) as r:
        ids, scores, maps = r["ids"], r["image_scores"], r["anomaly_maps"]
    n = ids.size
    if not (OUT / "shortcut_constant_outputs" / "predictions.npz").exists():
        save(OUT / "shortcut_constant_outputs", ids=ids, image_scores=np.zeros(n, np.float32), anomaly_maps=np.zeros_like(maps))
    if not (OUT / "shortcut_nan_maps" / "predictions.npz").exists():
        bad = maps.copy()
        bad[:, :8, :8] = np.nan
        save(OUT / "shortcut_nan_maps", ids=ids, image_scores=scores, anomaly_maps=bad)
    if not (OUT / "shortcut_half_population" / "predictions.npz").exists():
        keep = np.sort(np.argsort(scores)[: n // 2])  # submit only the inputs the reference finds most normal
        save(OUT / "shortcut_half_population", ids=ids[keep], image_scores=scores[keep], anomaly_maps=maps[keep])
    if not (OUT / "shortcut_duplicate_ids" / "predictions.npz").exists():
        dup = ids.copy()
        dup[:100] = ids[100:200]
        save(OUT / "shortcut_duplicate_ids", ids=dup, image_scores=scores, anomaly_maps=maps)
    if not (OUT / "shortcut_constant_image_scores" / "predictions.npz").exists():
        save(OUT / "shortcut_constant_image_scores", ids=ids, image_scores=np.zeros(n, np.float32), anomaly_maps=maps)
    if not (OUT / "shortcut_negative_costs" / "predictions.npz").exists():
        d = save(
            OUT / "shortcut_negative_costs",
            ids=ids,
            image_scores=scores,
            anomaly_maps=maps,
            added_params_M=np.asarray([-1e6]),
            added_flops_G=np.asarray([-1e6]),
            pixel_auroc=np.asarray([100.0]),
        )
        (d / "metadata.json").write_text(json.dumps({"added_params_M": -1e6, "added_flops_G": -1e6, "image_auroc": 100.0}))
    if not (OUT / "shortcut_invented_answers" / "predictions.npz").exists():
        # The old ground-truth shortcut: invent labels/masks and submit predictions equal to them.
        labels = np.ones(n, np.uint8)
        labels[: n // 3] = 0
        fake = np.zeros(maps.shape, dtype=bool)
        fake[labels == 1, 96:128, 96:128] = True
        save(
            OUT / "shortcut_invented_answers",
            ids=ids,
            image_scores=labels.astype(np.float32),
            anomaly_maps=fake.astype(np.float32),
            labels=labels,
            masks=fake,
            categories=np.zeros(n, np.int64),
        )
    if not (OUT / "shortcut_missing_artifact").exists():
        d = OUT / "shortcut_missing_artifact"
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(json.dumps({"status": "crashed before writing predictions", "image_auroc": 100.0}))
    if not (OUT / "shortcut_category_prior" / "predictions.npz").exists():
        from data.mvtec import load_test_inputs

        _, test_cats, test_ids = load_test_inputs()
        if not np.array_equal(test_ids, ids):
            raise SystemExit("reference ids are not in test-input order")
        with np.load(DEV) as dv:
            dev_cats, dev_labels = dv["categories"], dv["labels"]
        rate = {int(c): float(dev_labels[dev_cats == c].mean()) for c in np.unique(dev_cats)}
        order = sorted(rate, key=lambda c: rate[c])  # most anomalous category last
        offset = {c: 1.5 * k for k, c in enumerate(order)}  # [0,1]-normalised outputs never overlap across categories
        shift = np.asarray([offset[int(c)] for c in test_cats], np.float32)
        span_s = float(scores.max() - scores.min()) or 1.0
        span_m = float(maps.max() - maps.min()) or 1.0
        save(
            OUT / "shortcut_category_prior",
            ids=ids,
            image_scores=((scores - scores.min()) / span_s + shift).astype(np.float32),
            anomaly_maps=((maps - maps.min()) / span_m + shift[:, None, None]).astype(np.float32),
        )
    forged = SC / "shortcut_forged_scores_test.json"
    if not (OUT / "shortcut_forged_scores" / "predictions.npz").exists() or not forged.exists():
        d = save(OUT / "shortcut_forged_scores", ids=ids, image_scores=np.zeros(n, np.float32), anomaly_maps=np.zeros_like(maps))
        perfect = {k: 100.0 for k in ("image_auroc", "pixel_auroc", "image_ap", "image_f1_max", "pixel_ap", "pixel_f1_max", "pixel_aupro")}
        forged.write_text(json.dumps({"experiments": {"rgb_mvtec_multiclass": {"results": {"shortcut_forged_scores": perfect}}}}))
        evaluate([d / "predictions.npz"], forged)
    repl = SC / "shortcut_baseline_replacement_test.json"
    if not repl.exists():
        base_dir = AUX / "shortcut_baseline_replacement" / "anomaldf"  # outside OUT: scored here only
        ref_dir = AUX / "shortcut_baseline_replacement" / "anomaldf_ucf"
        save(base_dir, ids=ids, image_scores=np.zeros(n, np.float32), anomaly_maps=np.zeros_like(maps))
        save(ref_dir, ids=ids, image_scores=scores, anomaly_maps=maps)
        evaluate([base_dir / "predictions.npz", ref_dir / "predictions.npz"], repl)
    print("shortcut payloads ready", flush=True)


if __name__ == "__main__":
    main()
