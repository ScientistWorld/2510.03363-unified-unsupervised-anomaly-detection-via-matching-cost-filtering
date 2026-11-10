"""Summarise the hardening study (audit-only; reads hidden answers for the paired bootstrap).

rank(a) = sum_m |coef_m| * dir_m * (a_m - ref_m) / std_m, with dir_m = +1 if
higher_is_better else -1 and std_m the population std of metric m over the
named cohort. Reference = the anomaldf_ucf row of scoring/scores_<slice>.json.

Paired bootstrap (hidden test): images are resampled with replacement within
each category (the same resample for every method), and the class-mean image
AUROC, image AP and pixel AUROC (maps average-pooled 8x8 to 28x28, a cell is
anomalous if it holds any defect pixel)
of each method minus the reference are recorded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from eval.test.metrics import auroc, image_metrics

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "redteam" / "hardening"
SC = H / "scores"
OUT = H / "outputs"
EXP = "rgb_mvtec_multiclass"
SEEDS = ("reference", "ucf_seed1", "ucf_seed2")
GROUPS = {
    "reference": "reference",
    "anomaldf_baseline": "baseline",
    "ucf_seed1": "seed",
    "ucf_seed2": "seed",
    "ucf_nearest": "better",
    "ucf_ensemble3": "better",
    "ucf_ensemble3_nearest": "better",
    "ucf_nearest_seed1": "better (seed)",
    "ucf_nearest_seed2": "better (seed)",
    "ucf_tta4": "over template budget",
    "ucf_fullbank": "over template budget",
    "knn_fullshot": "brute force (over template budget)",
    "ucf_alone": "weaker",
    "ucf_no_guidance": "weaker",
    "ucf_focal_only": "weaker",
    "ucf_tenth_data": "weaker",
    "ucf_quarter_epochs": "weaker",
    "fused_lam0": "lambda sensitivity",
    "fused_lam0p25": "lambda sensitivity",
    "fused_lam0p75": "lambda sensitivity",
    "shortcut_dev_supervised": "residual shortcut (left to grader)",
}
BOOT = 200
POOL = 8


def group_of(name: str) -> str:
    if name in GROUPS:
        return GROUPS[name]
    return "shortcut" if name.startswith("shortcut_") else "other"


def load_rows(slice_name: str) -> tuple[dict, dict, dict]:
    rows, status, sources = {}, {}, {}
    main = json.loads((ROOT / "scoring" / f"scores_{slice_name}.json").read_text())["experiments"][EXP]
    for key, alias in (("anomaldf_ucf", "reference"), ("anomaldf", "anomaldf_baseline")):
        rows[alias] = main["results"][key]
        status[alias] = main.get("_validity", {}).get(key, "ok")
        sources[alias] = ROOT / main.get("_sources", {}).get(key, "")
    for f in sorted(SC.glob(f"*_{slice_name}.json")):
        name = f.name[: -len(f"_{slice_name}.json")]
        exp = json.loads(f.read_text())["experiments"][EXP]
        res, val, src = exp["results"], exp.get("_validity", {}), exp.get("_sources", {})
        if len(res) == 1:
            key = next(iter(res))
            rows[name], status[name], sources[name] = res[key], val.get(key, "?"), ROOT / src.get(key, "")
        else:
            for key in res:
                rows[f"{name}:{key}"], status[f"{name}:{key}"] = res[key], val.get(key, "?")
                sources[f"{name}:{key}"] = ROOT / src.get(key, "")
    return rows, status, sources


def ranks(rows: dict, members: list[str], ref: dict, metrics: dict) -> tuple[dict, dict]:
    std = {m: float(np.std([rows[r][m] for r in members])) for m in metrics}
    out = {}
    for name, row in rows.items():
        total = 0.0
        for m, spec in metrics.items():
            if std[m] > 0:
                direction = 1.0 if spec["higher_is_better"] else -1.0
                total += abs(spec["coefficient"]) * direction * (row[m] - ref[m]) / std[m]
        out[name] = total
    return out, std


def pooled(maps: np.ndarray, f: int = POOL) -> np.ndarray:
    n, h, w = maps.shape
    return maps.reshape(n, h // f, f, w // f, f).mean(axis=(2, 4)).reshape(n, -1)


def load_aligned(path: Path, answer_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as d:
        pos = {int(v): i for i, v in enumerate(d["ids"])}
        rows = np.asarray([pos[int(v)] for v in answer_ids])
        maps = d["anomaly_maps"]
        return d["image_scores"][rows].astype(np.float64), pooled(np.asarray(maps[rows], dtype=np.float32))


def class_mean_metrics(labels, masks, scores, maps, cat_rows, draws) -> np.ndarray:
    """(n_resamples, 3): class-mean image AUROC, image AP, pixel AUROC (pooled) per resample."""
    out = np.zeros((len(draws), 3))
    for b, draw in enumerate(draws):
        vals = []
        for c, rows in cat_rows.items():
            idx = rows[draw[c]]
            im = image_metrics(labels[idx], scores[idx])
            vals.append((im["image_auroc"], im["image_ap"], auroc(masks[idx], maps[idx])))
        out[b] = np.nanmean(np.asarray(vals, dtype=np.float64), axis=0)
    return out


def paired_bootstrap(candidates: dict[str, Path], ref_path: Path) -> dict:
    with np.load(ROOT / "eval" / "test" / "downloads" / "test_answers.npz") as d:
        ids, labels, cats = d["ids"], d["labels"].astype(bool), d["categories"]
        masks = pooled(d["masks"].astype(np.float32)) > 0  # a 28x28 cell is anomalous if it holds any defect pixel
    cat_rows = {int(c): np.flatnonzero(cats == c) for c in np.unique(cats)}
    rng = np.random.default_rng(0)
    draws = []
    while len(draws) < BOOT:
        draw = {c: rng.integers(0, r.size, r.size) for c, r in cat_rows.items()}
        if all(labels[cat_rows[c][draw[c]]].any() and not labels[cat_rows[c][draw[c]]].all() for c in cat_rows):
            draws.append(draw)
    rs, rm = load_aligned(ref_path, ids)
    ref_vals = class_mean_metrics(labels, masks, rs, rm, cat_rows, draws)
    out = {}
    for name, path in candidates.items():
        try:
            s, m = load_aligned(path, ids)
            delta = class_mean_metrics(labels, masks, s, m, cat_rows, draws) - ref_vals
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": repr(exc)}
            continue
        q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
        out[name] = {
            "image_auroc_delta_mean": float(delta[:, 0].mean()),
            "image_auroc_delta_ci95": q(delta[:, 0]),
            "image_ap_delta_mean": float(delta[:, 1].mean()),
            "image_ap_delta_ci95": q(delta[:, 1]),
            "pixel_auroc_28px_delta_mean": float(delta[:, 2].mean()),
            "pixel_auroc_28px_delta_ci95": q(delta[:, 2]),
            "resamples": BOOT,
        }
    return out


def main() -> None:
    ref_json = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"][EXP]
    metrics = ref_json["metrics"]
    summary = {"metrics": {m: {"coefficient": s["coefficient"], "higher_is_better": s["higher_is_better"]} for m, s in metrics.items()}}
    test_sources = {}
    for slice_name in ("test", "train"):
        try:
            rows, status, sources = load_rows(slice_name)
        except FileNotFoundError:
            continue
        if slice_name == "test":
            test_sources = sources
        ref = rows["reference"]
        honest = [r for r in rows if not r.startswith("shortcut_")]
        full = list(rows)
        r_h, std_h = ranks(rows, honest, ref, metrics)
        r_f, _ = ranks(rows, full, ref, metrics)
        seeds = [s for s in SEEDS if s in rows]
        spread = {m: float(max(rows[s][m] for s in seeds) - min(rows[s][m] for s in seeds)) for m in metrics}
        spread["rank_honest"] = float(max(r_h[s] for s in seeds) - min(r_h[s] for s in seeds))
        spread["rank_full"] = float(max(r_f[s] for s in seeds) - min(r_f[s] for s in seeds))
        base = rows["anomaldf_baseline"]
        summary[slice_name] = {
            "rows": {
                name: {
                    "group": group_of(name),
                    "status": status[name],
                    "rank_honest_cohort": r_h[name] if name in honest else None,
                    "rank_full_cohort": r_f[name],
                    **{m: rows[name][m] for m in metrics},
                }
                for name in rows
            },
            "reference_minus_baseline": {m: float(ref[m] - base[m]) for m in metrics},
            "seed_spread_max_minus_min": spread,
            "honest_cohort_std": std_h,
            "cohorts": {"honest": honest, "full": full},
        }

    ref_pred = ROOT / "outputs" / "anomaldf_ucf" / "predictions.npz"
    answers = ROOT / "eval" / "test" / "downloads" / "test_answers.npz"
    paired = {}
    if "--bootstrap" not in sys.argv:  # intermediate summaries skip the (slow) bootstrap
        cached = H / "summary.json"
        if cached.exists():
            paired = json.loads(cached.read_text()).get("paired_bootstrap_vs_reference_test", {})
    elif answers.exists() and ref_pred.exists():
        cands = {}
        for name, src in test_sources.items():
            if name == "reference" or not src.is_file():
                continue
            if name.startswith("shortcut_") and name not in ("shortcut_dev_supervised", "shortcut_category_prior"):
                continue
            if ":" in name:
                continue
            cands[name] = src
        paired = paired_bootstrap(cands, ref_pred)
    summary["paired_bootstrap_vs_reference_test"] = paired
    (H / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = []
    for slice_name in ("test", "train"):
        if slice_name not in summary:
            continue
        s = summary[slice_name]
        cols = list(metrics)
        lines.append(f"## {slice_name} slice (class-mean metrics)\n")
        lines.append("| run | group | status | rank (honest) | rank (full) | " + " | ".join(cols) + " |")
        lines.append("|---|---|---|---:|---:|" + "---:|" * len(cols))
        for name, row in sorted(s["rows"].items(), key=lambda kv: -kv[1]["rank_full_cohort"]):
            rh = "" if row["rank_honest_cohort"] is None else f"{row['rank_honest_cohort']:+.3f}"
            lines.append(
                f"| {name} | {row['group']} | {row['status'][:70]} | {rh} | {row['rank_full_cohort']:+.3f} | "
                + " | ".join(f"{row[m]:.2f}" for m in cols)
                + " |"
            )
        lines.append("\nreference minus AnomalDF baseline: " + json.dumps({k: round(v, 3) for k, v in s["reference_minus_baseline"].items()}))
        lines.append("\nseed spread (max-min over 3 training seeds): " + json.dumps({k: round(v, 3) for k, v in s["seed_spread_max_minus_min"].items()}) + "\n")
    if paired:
        lines.append(f"## paired bootstrap vs reference (hidden test, {BOOT} category-stratified image resamples, class means)\n")
        lines.append("| run | d image AUROC | 95% CI | d image AP | 95% CI | d pixel AUROC (28px) | 95% CI |")
        lines.append("|---|---:|---|---:|---|---:|---|")
        for name, v in paired.items():
            if "error" in v:
                lines.append(f"| {name} | error | | | | | |")
                continue
            ci_i, ci_a, ci_p = v["image_auroc_delta_ci95"], v["image_ap_delta_ci95"], v["pixel_auroc_28px_delta_ci95"]
            lines.append(
                f"| {name} | {v['image_auroc_delta_mean']:+.2f} | [{ci_i[0]:+.2f}, {ci_i[1]:+.2f}] | "
                f"{v['image_ap_delta_mean']:+.2f} | [{ci_a[0]:+.2f}, {ci_a[1]:+.2f}] | "
                f"{v['pixel_auroc_28px_delta_mean']:+.2f} | [{ci_p[0]:+.2f}, {ci_p[1]:+.2f}] |"
            )
    (H / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
