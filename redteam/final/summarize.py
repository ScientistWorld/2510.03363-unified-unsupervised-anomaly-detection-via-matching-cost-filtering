"""Summarise the final hardening study (audit-only; reads hidden answers for the paired bootstrap).

rank(a) = sum_e w_e * sum_m |coef_m| * dir_m * (a_em - ref_em) / std_em, with
dir_m = +1 if higher_is_better else -1, w_e the experiment weight and std_em
the population std of metric m of experiment e over the named cohort.
Reference = the anomaldf_ucf row of scoring/scores_<slice>.json.

Paired bootstrap (hidden test, per experiment): images are resampled with
replacement within each category (the same resample for every method), and
the class-mean image AUROC, image AP and pixel AUROC (maps average-pooled
16x16 to 14x14 cells, a cell is anomalous if it holds any defect pixel) of
each method minus the reference are recorded. "combined" averages the two
experiments' class-mean deltas with equal weight (as the rank weights them),
using the b-th resample of each experiment together.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from eval.test.metrics import auroc, image_metrics

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "redteam" / "final"
SC = F / "scores"
EXPS = {
    "rgb_mvtec_multiclass": {"answers": ROOT / "eval" / "test" / "downloads" / "test_answers.npz", "file": "predictions.npz"},
    "rgb_visa_multiclass": {"answers": ROOT / "eval" / "test" / "downloads" / "visa_test_answers.npz", "file": "predictions_visa.npz"},
}
SEEDS = ("reference", "ucf_seed1", "ucf_seed2")
GROUPS = {
    "reference": "reference (full-shot UCF, seed 0)",
    "anomaldf_baseline": "baseline (full-shot AnomalDF = memory-bank brute force)",
    "ucf_seed1": "seed",
    "ucf_seed2": "seed",
    "ucf_ensemble3": "better candidate",
    "ucf_train_n16": "better candidate",
    "ucf_focal_only": "paper ablation (worse in the paper)",
    "ucf_no_guidance": "paper ablation (worse in the paper)",
    "anomaldf_ucf_e10": "weaker (10 of 40 epochs)",
    "ucf_n3_random": "weaker (paper's N=3 random-template inference)",
    "anomaldf_n3": "weaker (AnomalDF, N=3 random templates)",
    "ucf_alone": "weaker (UCF map alone, lambda=1)",
    "fused_lam0": "weaker (filter switched off, lambda=0)",
    "fused_lam0p25": "lambda sensitivity",
    "fused_lam0p75": "lambda sensitivity",
    "shortcut_dev_supervised": "residual shortcut (left to grader)",
}
BOOT = 300
POOL = 16


def group_of(name: str) -> str:
    if name in GROUPS:
        return GROUPS[name]
    return "shortcut" if name.startswith("shortcut_") else "other"


def load_rows(slice_name: str) -> tuple[dict, dict, dict]:
    """rows[name][exp] = metrics; status[name][exp]; sources[name][exp] = artifact path."""
    rows, status, sources = {}, {}, {}

    def add(alias: str, exp: str, res: dict, val: dict, src: dict, key: str) -> None:
        rows.setdefault(alias, {})[exp] = res[key]
        status.setdefault(alias, {})[exp] = val.get(key, "?")
        sources.setdefault(alias, {})[exp] = ROOT / src.get(key, "")

    main = json.loads((ROOT / "scoring" / f"scores_{slice_name}.json").read_text())["experiments"]
    for exp in EXPS:
        e = main[exp]
        for key, alias in (("anomaldf_ucf", "reference"), ("anomaldf", "anomaldf_baseline")):
            add(alias, exp, e["results"], e.get("_validity", {}), e.get("_sources", {}), key)
    for f in sorted(SC.glob(f"*_{slice_name}.json")):
        name = f.name[: -len(f"_{slice_name}.json")]
        exps = json.loads(f.read_text())["experiments"]
        for exp in EXPS:
            e = exps.get(exp)
            if e is None:
                continue
            res, val, src = e["results"], e.get("_validity", {}), e.get("_sources", {})
            if len(res) == 1:
                add(name, exp, res, val, src, next(iter(res)))
            else:
                for key in res:
                    add(f"{name}:{key}", exp, res, val, src, key)
    rows = {k: v for k, v in rows.items() if all(exp in v for exp in EXPS)}
    return rows, status, sources


def ranks(rows: dict, members: list[str], ref: dict, metrics: dict, weights: dict) -> tuple[dict, dict, dict]:
    std = {e: {m: float(np.std([rows[r][e][m] for r in members])) for m in metrics} for e in EXPS}
    total, per_exp = {}, {}
    for name, row in rows.items():
        per_exp[name] = {}
        for e in EXPS:
            s = 0.0
            for m, spec in metrics.items():
                if std[e][m] > 0:
                    direction = 1.0 if spec["higher_is_better"] else -1.0
                    s += abs(spec["coefficient"]) * direction * (row[e][m] - ref[e][m]) / std[e][m]
            per_exp[name][e] = weights[e] * s
        total[name] = sum(per_exp[name].values())
    return total, per_exp, std


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


def paired_bootstrap(candidates: dict[str, dict[str, Path]], ref_paths: dict[str, Path]) -> dict:
    deltas: dict[str, dict[str, np.ndarray]] = {name: {} for name in candidates}
    errors: dict[str, str] = {}
    for exp, spec in EXPS.items():
        with np.load(spec["answers"]) as d:
            ids, labels, cats = d["ids"], d["labels"].astype(bool), d["categories"]
            masks = pooled(d["masks"].astype(np.float32)) > 0
        cat_rows = {int(c): np.flatnonzero(cats == c) for c in np.unique(cats)}
        rng = np.random.default_rng(0)
        draws = []
        while len(draws) < BOOT:
            draw = {c: rng.integers(0, r.size, r.size) for c, r in cat_rows.items()}
            if all(labels[cat_rows[c][draw[c]]].any() and not labels[cat_rows[c][draw[c]]].all() for c in cat_rows):
                draws.append(draw)
        rs, rm = load_aligned(ref_paths[exp], ids)
        ref_vals = class_mean_metrics(labels, masks, rs, rm, cat_rows, draws)
        for name, paths in candidates.items():
            if name in errors:
                continue
            try:
                s, m = load_aligned(paths[exp], ids)
                deltas[name][exp] = class_mean_metrics(labels, masks, s, m, cat_rows, draws) - ref_vals
            except Exception as exc:  # noqa: BLE001
                errors[name] = repr(exc)
        print(f"bootstrap done for {exp}", flush=True)

    def q(v: np.ndarray) -> list[float]:
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out = {}
    for name in candidates:
        if name in errors:
            out[name] = {"error": errors[name]}
            continue
        entry = {"resamples": BOOT}
        for exp in EXPS:
            dv = deltas[name][exp]
            entry[exp] = {
                "image_auroc": {"mean": float(dv[:, 0].mean()), "ci95": q(dv[:, 0])},
                "image_ap": {"mean": float(dv[:, 1].mean()), "ci95": q(dv[:, 1])},
                "pixel_auroc_14x14": {"mean": float(dv[:, 2].mean()), "ci95": q(dv[:, 2])},
            }
        comb = np.mean([deltas[name][exp] for exp in EXPS], axis=0)
        entry["combined"] = {
            "image_auroc": {"mean": float(comb[:, 0].mean()), "ci95": q(comb[:, 0])},
            "image_ap": {"mean": float(comb[:, 1].mean()), "ci95": q(comb[:, 1])},
            "pixel_auroc_14x14": {"mean": float(comb[:, 2].mean()), "ci95": q(comb[:, 2])},
        }
        out[name] = entry
    return out


def main() -> None:
    ref_json = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    metrics = ref_json["rgb_mvtec_multiclass"]["metrics"]
    weights = {e: float(ref_json[e].get("weight", 1.0)) for e in EXPS}
    summary = {
        "metrics": {m: {"coefficient": s["coefficient"], "higher_is_better": s["higher_is_better"]} for m, s in metrics.items()},
        "weights": weights,
    }
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
        r_h, pe_h, std_h = ranks(rows, honest, ref, metrics, weights)
        r_f, _, _ = ranks(rows, full, ref, metrics, weights)
        seeds = [s for s in SEEDS if s in rows]
        spread = {
            e: {m: float(max(rows[s][e][m] for s in seeds) - min(rows[s][e][m] for s in seeds)) for m in metrics} for e in EXPS
        }
        spread["rank_honest"] = float(max(r_h[s] for s in seeds) - min(r_h[s] for s in seeds))
        spread["rank_full"] = float(max(r_f[s] for s in seeds) - min(r_f[s] for s in seeds))
        base = rows["anomaldf_baseline"]
        summary[slice_name] = {
            "rows": {
                name: {
                    "group": group_of(name),
                    "status": status[name],
                    "rank_honest_cohort": r_h[name] if name in honest else None,
                    "rank_honest_by_experiment": pe_h[name] if name in honest else None,
                    "rank_full_cohort": r_f[name],
                    **{e: rows[name][e] for e in EXPS},
                }
                for name in rows
            },
            "reference_minus_baseline": {e: {m: float(ref[e][m] - base[e][m]) for m in metrics} for e in EXPS},
            "seed_spread_max_minus_min": spread,
            "honest_cohort_std": std_h,
            "cohorts": {"honest": honest, "full": full},
        }

    paired = {}
    if "--bootstrap" not in sys.argv:
        cached = F / "summary.json"
        if cached.exists():
            paired = json.loads(cached.read_text()).get("paired_bootstrap_vs_reference_test", {})
    elif all(spec["answers"].exists() for spec in EXPS.values()):
        cands = {}
        for name, src in test_sources.items():
            if name == "reference" or ":" in name:
                continue
            if name.startswith("shortcut_") and name not in ("shortcut_dev_supervised", "shortcut_category_prior"):
                continue
            if all(Path(src[e]).is_file() for e in EXPS):
                cands[name] = src
        paired = paired_bootstrap(cands, test_sources["reference"])
    summary["paired_bootstrap_vs_reference_test"] = paired
    (F / "summary.json").write_text(json.dumps(summary, indent=2))

    short = {"rgb_mvtec_multiclass": "MVTec", "rgb_visa_multiclass": "VisA"}
    lines = []
    for slice_name in ("test", "train"):
        if slice_name not in summary:
            continue
        s = summary[slice_name]
        lines.append(f"## {slice_name} slice (class-mean metrics)\n")
        cols = ["image_auroc", "pixel_auroc", "pixel_aupro", "pixel_ap", "image_ap", "pixel_f1_max", "image_f1_max"]
        head = "| run | group | status | rank (honest) | MVTec part | VisA part | rank (full) | " + " | ".join(
            f"{short[e]} {c}" for e in EXPS for c in cols
        ) + " |"
        lines.append(head)
        lines.append("|---|---|---|---:|---:|---:|---:|" + "---:|" * (len(cols) * len(EXPS)))
        for name, row in sorted(s["rows"].items(), key=lambda kv: -kv[1]["rank_full_cohort"]):
            rh = "" if row["rank_honest_cohort"] is None else f"{row['rank_honest_cohort']:+.3f}"
            pe = row["rank_honest_by_experiment"] or {}
            parts = " | ".join(f"{pe[e]:+.2f}" if e in pe else "" for e in EXPS)
            st = "; ".join(f"{short[e]}: {row['status'][e][:50]}" for e in EXPS if row["status"][e] != "ok") or "ok"
            vals = " | ".join(f"{row[e][c]:.2f}" for e in EXPS for c in cols)
            lines.append(f"| {name} | {row['group']} | {st} | {rh} | {parts} | {row['rank_full_cohort']:+.3f} | {vals} |")
        for e in EXPS:
            lines.append(f"\n{short[e]} reference minus baseline: " + json.dumps({k: round(v, 3) for k, v in s["reference_minus_baseline"][e].items()}))
        sp = s["seed_spread_max_minus_min"]
        lines.append("\nseed spread (max-min over 3 training seeds): " + json.dumps(
            {**{short[e]: {k: round(v, 3) for k, v in sp[e].items()} for e in EXPS}, "rank_honest": round(sp["rank_honest"], 3), "rank_full": round(sp["rank_full"], 3)}
        ) + "\n")
    if paired:
        lines.append(f"## paired bootstrap vs reference (hidden test, {BOOT} category-stratified image resamples, class means)\n")
        lines.append("| run | MVTec d I-AUROC [95% CI] | VisA d I-AUROC [95% CI] | combined d I-AUROC [95% CI] | combined d I-AP [95% CI] | MVTec d P-AUROC14 [95% CI] | VisA d P-AUROC14 [95% CI] |")
        lines.append("|---|---|---|---|---|---|---|")

        def fmt(v: dict) -> str:
            return f"{v['mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"

        for name, v in sorted(paired.items(), key=lambda kv: -kv[1].get("combined", {}).get("image_auroc", {}).get("mean", -1e9)):
            if "error" in v:
                lines.append(f"| {name} | error: {v['error'][:60]} | | | | | |")
                continue
            m, vi, c = v["rgb_mvtec_multiclass"], v["rgb_visa_multiclass"], v["combined"]
            lines.append(
                f"| {name} | {fmt(m['image_auroc'])} | {fmt(vi['image_auroc'])} | {fmt(c['image_auroc'])} | {fmt(c['image_ap'])} | "
                f"{fmt(m['pixel_auroc_14x14'])} | {fmt(vi['pixel_auroc_14x14'])} |"
            )
    (F / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
