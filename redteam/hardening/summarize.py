"""Summarise the hardening study (audit-only; reads hidden answers for paired bootstrap).

rank(a) = sum_m |coef_m| * dir_m * (a_m - ref_m) / std_m, with dir_m = +1 if
higher_is_better else -1 and std_m the population std of metric m over the
named cohort. Reference = stored scoring/scores_test.json anomaldf_ucf row.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval.test.metrics import auroc

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "redteam" / "hardening"
SC = H / "scores"
OUT = H / "outputs"
EXP = "rgb_mvtec_multiclass"
SEEDS = ("reference", "ucf_seed1", "ucf_seed2")


def load_rows(slice_name: str) -> tuple[dict, dict]:
    rows, status = {}, {}
    main = json.loads((ROOT / "scoring" / f"scores_{slice_name}.json").read_text())["experiments"][EXP]
    rows["reference"] = main["results"]["anomaldf_ucf"]
    rows["anomaldf_baseline"] = main["results"]["anomaldf"]
    status["reference"] = main.get("_validity", {}).get("anomaldf_ucf", "ok")
    status["anomaldf_baseline"] = main.get("_validity", {}).get("anomaldf", "ok")
    for f in sorted(SC.glob(f"*_{slice_name}.json")):
        name = f.name[: -len(f"_{slice_name}.json")]
        exp = json.loads(f.read_text())["experiments"][EXP]
        res, val = exp["results"], exp.get("_validity", {})
        if len(res) == 1:
            key = next(iter(res))
            rows[name], status[name] = res[key], val.get(key, "?")
        else:
            for key in res:
                rows[f"{name}:{key}"], status[f"{name}:{key}"] = res[key], val.get(key, "?")
    return rows, status


def ranks(rows: dict, members: list[str], ref: dict, metrics: dict) -> dict:
    std = {m: float(np.std([rows[r][m] for r in members])) for m in metrics}
    out = {}
    for name, row in rows.items():
        total = 0.0
        for m, spec in metrics.items():
            if std[m] > 0:
                direction = 1.0 if spec["higher_is_better"] else -1.0
                total += abs(spec["coefficient"]) * direction * (row[m] - ref[m]) / std[m]
        out[name] = total
    return out


def pooled(maps: np.ndarray, f: int = 8) -> np.ndarray:
    n, h, w = maps.shape
    return maps.reshape(n, h // f, f, w // f, f).mean(axis=(2, 4))


def bootstrap(answers: dict, ref_path: Path, alt_path: Path, b: int = 200) -> dict:
    def aligned(path):
        with np.load(path) as d:
            pos = {int(v): i for i, v in enumerate(d["ids"])}
            rows = np.asarray([pos[int(v)] for v in answers["ids"]])
            return d["image_scores"][rows].astype(np.float64), pooled(d["anomaly_maps"][rows].astype(np.float32))

    labels = answers["labels"].astype(bool)
    masks = pooled(answers["masks"].astype(np.float32)) >= 0.5
    rs, rm = aligned(ref_path)
    as_, am = aligned(alt_path)
    rng = np.random.default_rng(0)
    n = labels.size
    di, dp = [], []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        if labels[idx].all() or not labels[idx].any():
            continue
        di.append(auroc(labels[idx], as_[idx]) - auroc(labels[idx], rs[idx]))
        dp.append(auroc(masks[idx], am[idx]) - auroc(masks[idx], rm[idx]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {
        "image_auroc_delta_mean": float(np.mean(di)),
        "image_auroc_delta_ci95": q(di),
        "pixel_auroc_28px_delta_mean": float(np.mean(dp)),
        "pixel_auroc_28px_delta_ci95": q(dp),
        "resamples": len(di),
    }


def main() -> None:
    ref_json = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"][EXP]
    metrics = ref_json["metrics"]
    summary = {"metrics": {m: {"coefficient": s["coefficient"], "higher_is_better": s["higher_is_better"]} for m, s in metrics.items()}}
    for slice_name in ("test", "train"):
        try:
            rows, status = load_rows(slice_name)
        except FileNotFoundError:
            continue
        ref = rows["reference"]
        honest = [r for r in rows if not r.startswith("shortcut_")]
        full = list(rows)
        r_h = ranks(rows, honest, ref, metrics)
        r_f = ranks(rows, full, ref, metrics)
        seeds = [s for s in SEEDS if s in rows]
        spread = {m: float(max(rows[s][m] for s in seeds) - min(rows[s][m] for s in seeds)) for m in metrics}
        spread["rank_honest"] = float(max(r_h[s] for s in seeds) - min(r_h[s] for s in seeds))
        spread["rank_full"] = float(max(r_f[s] for s in seeds) - min(r_f[s] for s in seeds))
        summary[slice_name] = {
            "rows": {
                name: {
                    "status": status[name],
                    "rank_honest_cohort": r_h[name] if name in honest else None,
                    "rank_full_cohort": r_f[name],
                    **{m: rows[name][m] for m in metrics},
                }
                for name in rows
            },
            "seed_spread_max_minus_min": spread,
            "cohorts": {"honest": honest, "full": full},
        }
    answers_path = ROOT / "eval" / "test" / "downloads" / "test_answers.npz"
    ref_pred = ROOT / "outputs" / "anomaldf_ucf" / "predictions.npz"
    paired = {}
    if answers_path.exists() and ref_pred.exists():
        with np.load(answers_path) as d:
            answers = {k: d[k] for k in ("ids", "labels", "masks")}
        cands = {"anomaldf_baseline": ROOT / "outputs" / "anomaldf" / "predictions.npz"}
        for d in sorted(OUT.iterdir()) if OUT.exists() else []:
            p = d / "predictions.npz"
            if p.exists() and (not d.name.startswith("shortcut_") or d.name == "shortcut_dev_supervised"):
                cands[d.name] = p
        cache_path = H / "bootstrap_cache.json"
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        ref_stamp = f"{ref_pred.stat().st_size}:{ref_pred.stat().st_mtime_ns}"
        for name, path in cands.items():
            if not path.exists():
                continue
            stamp = f"{ref_stamp}|{path.stat().st_size}:{path.stat().st_mtime_ns}"
            if cache.get(name, {}).get("stamp") == stamp:
                paired[name] = cache[name]["value"]
                continue
            try:
                paired[name] = bootstrap(answers, ref_pred, path)
                cache[name] = {"stamp": stamp, "value": paired[name]}
            except Exception as exc:  # noqa: BLE001
                paired[name] = {"error": repr(exc)}
        cache_path.write_text(json.dumps(cache))
    summary["paired_bootstrap_vs_reference_test"] = paired
    (H / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = []
    for slice_name in ("test", "train"):
        if slice_name not in summary:
            continue
        s = summary[slice_name]
        cols = list(metrics)
        lines.append(f"## {slice_name} slice\n")
        lines.append("| run | status | rank (honest) | rank (full) | " + " | ".join(cols) + " |")
        lines.append("|---|---|---:|---:|" + "---:|" * len(cols))
        for name, row in sorted(s["rows"].items(), key=lambda kv: -kv[1]["rank_full_cohort"]):
            rh = "" if row["rank_honest_cohort"] is None else f"{row['rank_honest_cohort']:+.3f}"
            lines.append(
                f"| {name} | {row['status'][:60]} | {rh} | {row['rank_full_cohort']:+.3f} | "
                + " | ".join(f"{row[m]:.2f}" for m in cols)
                + " |"
            )
        lines.append("\nseed spread (max-min over 3 training seeds): " + json.dumps({k: round(v, 3) for k, v in s["seed_spread_max_minus_min"].items()}) + "\n")
    if paired:
        lines.append("## paired bootstrap vs reference (hidden test, 200 image resamples)\n")
        lines.append("| run | d image AUROC | 95% CI | d pixel AUROC (28px) | 95% CI |")
        lines.append("|---|---:|---|---:|---|")
        for name, v in paired.items():
            if "error" in v:
                lines.append(f"| {name} | error | | | |")
                continue
            ci_i, ci_p = v["image_auroc_delta_ci95"], v["pixel_auroc_28px_delta_ci95"]
            lines.append(
                f"| {name} | {v['image_auroc_delta_mean']:+.2f} | [{ci_i[0]:+.2f}, {ci_i[1]:+.2f}] | "
                f"{v['pixel_auroc_28px_delta_mean']:+.2f} | [{ci_p[0]:+.2f}, {ci_p[1]:+.2f}] |"
            )
    (H / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
