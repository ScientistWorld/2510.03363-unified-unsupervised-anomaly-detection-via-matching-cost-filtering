"""Shortcut payloads re-run against the final hidden-test evaluator (both experiments).

Every payload is built only from method-visible material (the reference
prediction artifacts, the public test-input ids and, for category_prior, the
visible development answers); none reads eval/test/. Payloads are written to
redteam/final/outputs/shortcut_<name>/ and scored by run_study.sh with the
unchanged evaluators. Two cases need the evaluator invoked in a specific way
and are scored here: forged_scores (a forged score file already sits at the
output path) and baseline_replacement (the baseline artifacts are replaced
next to the reference artifacts).

  constant_outputs       zero maps and scores
  nan_maps               reference outputs with a NaN patch in every map
  half_population        only the half of the inputs the reference finds most normal
  duplicate_ids          100 ids listed twice, 100 missing
  constant_image_scores  reference maps, constant image scores
  negative_costs         reference outputs plus self-reported negative costs / perfect metrics
  invented_answers       invented labels and masks, predictions equal to them
  missing_artifact       a method directory without predictions (a crashed run)
  mvtec_only             the reference's MVTec-AD predictions, no VisA file
  oversized_header       MVTec-AD maps whose .npy header claims ~400 GB; valid VisA file
  category_prior         per-category offsets ordered by the visible anomaly rate
  forged_scores          a forged perfect score file, constant predictions
  baseline_replacement   the baseline's predictions replaced by zeros
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

from data.mvtec import load_test_inputs, predictions_name

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("mvtec", "visa")
DEV = {
    "mvtec": ROOT / "eval" / "train" / "data" / "dev_answers.npz",
    "visa": ROOT / "eval" / "train" / "data" / "visa_dev_answers.npz",
}
REF = ROOT / "outputs" / "anomaldf_ucf"
OUT = ROOT / "redteam" / "final" / "outputs"
AUX = ROOT / "redteam" / "final" / "aux_outputs"
SC = ROOT / "redteam" / "final" / "scores"
METRICS = ("image_auroc", "pixel_auroc", "image_ap", "image_f1_max", "pixel_ap", "pixel_f1_max", "pixel_aupro")


def save(folder: Path, dataset: str, **arrays) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / predictions_name(dataset), **arrays)
    return folder


def evaluate(paths: list[Path], scores: Path, slice_name: str = "test") -> None:
    subprocess.run(
        [sys.executable, "-m", f"eval.{slice_name}.evaluate", "--predictions", *map(str, paths), "--scores", str(scores)],
        check=True,
        cwd=ROOT,
    )


def done(name: str) -> bool:
    d = OUT / name
    return all((d / predictions_name(ds)).exists() for ds in DATASETS)


def reference(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(REF / predictions_name(dataset)) as r:
        return r["ids"], r["image_scores"], r["anomaly_maps"]


def npy_bytes(arr: np.ndarray) -> bytes:
    b = io.BytesIO()
    np.save(b, arr)
    return b.getvalue()


def main() -> None:
    SC.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        ids, scores, maps = reference(ds)
        n = ids.size
        if not (OUT / "shortcut_constant_outputs" / predictions_name(ds)).exists():
            save(OUT / "shortcut_constant_outputs", ds, ids=ids, image_scores=np.zeros(n, np.float32), anomaly_maps=np.zeros_like(maps))
        if not (OUT / "shortcut_nan_maps" / predictions_name(ds)).exists():
            bad = maps.copy()
            bad[:, :8, :8] = np.nan
            save(OUT / "shortcut_nan_maps", ds, ids=ids, image_scores=scores, anomaly_maps=bad)
        if not (OUT / "shortcut_half_population" / predictions_name(ds)).exists():
            keep = np.sort(np.argsort(scores)[: n // 2])
            save(OUT / "shortcut_half_population", ds, ids=ids[keep], image_scores=scores[keep], anomaly_maps=maps[keep])
        if not (OUT / "shortcut_duplicate_ids" / predictions_name(ds)).exists():
            dup = ids.copy()
            dup[:100] = ids[100:200]
            save(OUT / "shortcut_duplicate_ids", ds, ids=dup, image_scores=scores, anomaly_maps=maps)
        if not (OUT / "shortcut_constant_image_scores" / predictions_name(ds)).exists():
            save(OUT / "shortcut_constant_image_scores", ds, ids=ids, image_scores=np.zeros(n, np.float32), anomaly_maps=maps)
        if not (OUT / "shortcut_negative_costs" / predictions_name(ds)).exists():
            d = save(
                OUT / "shortcut_negative_costs", ds,
                ids=ids, image_scores=scores, anomaly_maps=maps,
                added_params_M=np.asarray([-1e6]), added_flops_G=np.asarray([-1e6]), pixel_auroc=np.asarray([100.0]),
            )
            (d / "metadata.json").write_text(json.dumps({"added_params_M": -1e6, "added_flops_G": -1e6, "image_auroc": 100.0}))
        if not (OUT / "shortcut_invented_answers" / predictions_name(ds)).exists():
            labels = np.ones(n, np.uint8)
            labels[: n // 3] = 0
            fake = np.zeros(maps.shape, dtype=bool)
            fake[labels == 1, 96:128, 96:128] = True
            save(
                OUT / "shortcut_invented_answers", ds,
                ids=ids, image_scores=labels.astype(np.float32), anomaly_maps=fake.astype(np.float32),
                labels=labels, masks=fake, categories=np.zeros(n, np.int64),
            )
        if not (OUT / "shortcut_category_prior" / predictions_name(ds)).exists():
            _, test_cats, test_ids = load_test_inputs(ds)
            if not np.array_equal(test_ids, ids):
                raise SystemExit("reference ids are not in test-input order")
            with np.load(DEV[ds]) as dv:
                dev_cats, dev_labels = dv["categories"], dv["labels"]
            rate = {int(c): float(dev_labels[dev_cats == c].mean()) for c in np.unique(dev_cats)}
            order = sorted(rate, key=lambda c: rate[c])
            offset = {c: 1.5 * k for k, c in enumerate(order)}
            shift = np.asarray([offset[int(c)] for c in test_cats], np.float32)
            span_s = float(scores.max() - scores.min()) or 1.0
            span_m = float(maps.max() - maps.min()) or 1.0
            save(
                OUT / "shortcut_category_prior", ds,
                ids=ids,
                image_scores=((scores - scores.min()) / span_s + shift).astype(np.float32),
                anomaly_maps=((maps - maps.min()) / span_m + shift[:, None, None]).astype(np.float32),
            )
        del maps

    # A crashed run: a directory with only a metadata file claiming perfect results.
    d = OUT / "shortcut_missing_artifact"
    if not d.exists():
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(json.dumps({"status": "crashed before writing predictions", "image_auroc": 100.0}))

    # Only one of the two experiments submitted.
    d = OUT / "shortcut_mvtec_only"
    if not (d / predictions_name("mvtec")).exists():
        ids, scores, maps = reference("mvtec")
        save(d, "mvtec", ids=ids, image_scores=scores, anomaly_maps=maps)

    # A header claiming ~400 GB of maps (MVTec-AD), next to a valid VisA artifact.
    d = OUT / "shortcut_oversized_header"
    if not done("shortcut_oversized_header"):
        d.mkdir(parents=True, exist_ok=True)
        ids, scores, _ = reference("mvtec")
        hdr = io.BytesIO()
        npy_format.write_array_header_1_0(hdr, {"descr": "<f4", "fortran_order": False, "shape": (2_000_000, 224, 224)})
        with zipfile.ZipFile(d / predictions_name("mvtec"), "w") as z:
            z.writestr("ids.npy", npy_bytes(ids.astype(np.int64)))
            z.writestr("image_scores.npy", npy_bytes(scores.astype(np.float32)))
            z.writestr("anomaly_maps.npy", hdr.getvalue() + b"\0" * 4096)
        vids, vscores, vmaps = reference("visa")
        save(d, "visa", ids=vids, image_scores=vscores, anomaly_maps=vmaps)

    # A forged perfect score file at the output path, constant predictions.
    forged = SC / "shortcut_forged_scores_test.json"
    if not done("shortcut_forged_scores") or not forged.exists():
        d = OUT / "shortcut_forged_scores"
        for ds in DATASETS:
            ids, _, maps = reference(ds)
            save(d, ds, ids=ids, image_scores=np.zeros(ids.size, np.float32), anomaly_maps=np.zeros_like(maps))
        perfect = {k: 100.0 for k in METRICS}
        forged.write_text(json.dumps({"experiments": {e: {"results": {"shortcut_forged_scores": perfect}} for e in ("rgb_mvtec_multiclass", "rgb_visa_multiclass")}}))
        evaluate([d / "predictions.npz"], forged)

    # The fixed comparison replaced: baseline predictions set to zeros next to the reference.
    repl = SC / "shortcut_baseline_replacement_test.json"
    if not repl.exists():
        base_dir = AUX / "shortcut_baseline_replacement" / "anomaldf"
        ref_dir = AUX / "shortcut_baseline_replacement" / "anomaldf_ucf"
        for ds in DATASETS:
            ids, scores, maps = reference(ds)
            save(base_dir, ds, ids=ids, image_scores=np.zeros(ids.size, np.float32), anomaly_maps=np.zeros_like(maps))
            save(ref_dir, ds, ids=ids, image_scores=scores, anomaly_maps=maps)
        evaluate([base_dir / "predictions.npz", ref_dir / "predictions.npz"], repl)
    print("shortcut payloads ready", flush=True)


if __name__ == "__main__":
    main()
