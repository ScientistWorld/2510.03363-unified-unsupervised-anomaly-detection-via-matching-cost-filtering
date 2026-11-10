"""Choose UCF training-setup knobs on the VISIBLE development slice only.

Knobs (setup, not method): number of training epochs, Adam learning rate,
and whether synthetic anomalies are restricted to the object foreground.
Each (lr, foreground) setting is trained once for 40 epochs (seed 0, the
paper's schedule) with snapshots at epochs 1, 2, 4, 8, 16; every snapshot
predicts all test inputs with the paper's inference (UCF fused with the
AnomalDF response, lambda = 0.5) and is scored by the development
evaluator (eval/train). The candidate with the highest coefficient-weighted
development score (coefficients from scoring/reference.json) is written to
method/ucf_setup.json, which method/train_ucf.py uses as its defaults.
The hidden test is never read.

Large prediction files are deleted once scored; the small per-run
components (method/fusion.py) stay under tmp/select/ for later analysis.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEL_DIR = ROOT / "method" / "selection"
PRED_DIR = ROOT / "tmp" / "select"
CKPT_DIR = ROOT / "checkpoints" / "select"
SETUP_FILE = ROOT / "method" / "ucf_setup.json"
EXP = "rgb_mvtec_multiclass"
EPOCHS = 40
SNAPSHOTS = (1, 2, 4, 8, 16)
GRID = (
    {"name": "fg_lr3", "lr": 1e-3, "fg_restrict": True},
    {"name": "fg_lr4", "lr": 1e-4, "fg_restrict": True},
    {"name": "nofg_lr3", "lr": 1e-3, "fg_restrict": False},
    {"name": "nofg_lr4", "lr": 1e-4, "fg_restrict": False},
)


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def dev_score(paths: list[Path], scores_path: Path) -> dict:
    run([sys.executable, "-m", "eval.train.evaluate", "--predictions", *map(str, paths), "--scores", str(scores_path)])
    return json.loads(scores_path.read_text())["experiments"][EXP]


def main() -> None:
    SEL_DIR.mkdir(parents=True, exist_ok=True)
    exp = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"][EXP]
    coefs = {m: abs(s["coefficient"]) for m, s in exp["metrics"].items()}

    def weighted(row: dict) -> float:
        return float(sum(coefs[m] * row[m] for m in coefs))

    base_path = SEL_DIR / "dev_scores_anomaldf.json"
    if not base_path.exists():
        dev_score([ROOT / "outputs" / "anomaldf" / "predictions.npz"], base_path)
    base_row = json.loads(base_path.read_text())["experiments"][EXP]["results"]["anomaldf"]

    table = {}
    for g in GRID:
        scores_path = SEL_DIR / f"dev_scores_{g['name']}.json"
        epochs_of = {g["name"]: EPOCHS, **{f"{g['name']}_e{e}": e for e in SNAPSHOTS}}
        if not scores_path.exists():
            out = PRED_DIR / g["name"]
            if not (out / "predictions.npz").exists():
                run([
                    sys.executable, "-m", "method.train_ucf",
                    "--seed", "0", "--epochs", str(EPOCHS), "--lr", str(g["lr"]),
                    "--fg-restrict" if g["fg_restrict"] else "--no-fg-restrict",
                    "--snapshot-epochs", ",".join(map(str, SNAPSHOTS)),
                    "--checkpoint", str(CKPT_DIR / f"{g['name']}.pt"),
                    "--output", str(out),
                ])
            dev_score([PRED_DIR / name / "predictions.npz" for name in epochs_of], scores_path)
            for name in epochs_of:  # keep components.npz + metadata, drop the 300 MB maps
                (PRED_DIR / name / "predictions.npz").unlink(missing_ok=True)
        res = json.loads(scores_path.read_text())["experiments"][EXP]
        for name, e in epochs_of.items():
            row = res["results"][name]
            table[name] = {
                "dev_weighted_score": weighted(row),
                "status": res["_validity"][name],
                **row,
                "epochs": e,
                "lr": g["lr"],
                "fg_restrict": g["fg_restrict"],
            }

    best = max(table, key=lambda k: table[k]["dev_weighted_score"])
    setup = {
        "epochs": table[best]["epochs"],
        "lr": table[best]["lr"],
        "fg_restrict": table[best]["fg_restrict"],
        "selected": best,
        "selected_on": "visible development slice (eval/train), seed 0, paper inference (lambda = 0.5); hidden test never read",
        "criterion": "sum over scored metrics of |coefficient| x development value (class means)",
        "baseline_dev_weighted_score": weighted(base_row),
        "baseline_dev": base_row,
        "candidates": table,
    }
    SETUP_FILE.write_text(json.dumps(setup, indent=2))
    print(json.dumps({k: setup[k] for k in ("epochs", "lr", "fg_restrict", "selected", "baseline_dev_weighted_score")}, indent=2))
    for name in sorted(table, key=lambda k: -table[k]["dev_weighted_score"]):
        r = table[name]
        print(
            f"{name:14s} S={r['dev_weighted_score']:8.2f} I-AUROC={r['image_auroc']:6.2f} P-AUROC={r['pixel_auroc']:6.2f} "
            f"P-AP={r['pixel_ap']:6.2f} AUPRO={r['pixel_aupro']:6.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
