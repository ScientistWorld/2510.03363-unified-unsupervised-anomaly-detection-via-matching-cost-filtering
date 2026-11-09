"""Choose UCF training-setup knobs on the VISIBLE development slice only.

Knobs (setup, not method): number of epochs, Adam learning rate, and whether
synthetic anomalies are restricted to the object foreground. Each
(lr, foreground) setting is trained once for 40 epochs (seed 0) with
snapshots at epochs 1, 2, 4, 8, 16; every snapshot predicts all test inputs
and is scored by the development evaluator (eval/train). The setting with the
highest coefficient-weighted development score (coefficients from
scoring/reference.json) is written to method/ucf_setup.json, which
method/train_ucf.py uses as its defaults. The hidden test is never read.
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


def main() -> None:
    SEL_DIR.mkdir(parents=True, exist_ok=True)
    candidates = {}
    for g in GRID:
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
        candidates[g["name"]] = {"epochs": EPOCHS, "lr": g["lr"], "fg_restrict": g["fg_restrict"], "path": out}
        for e in SNAPSHOTS:
            candidates[f"{g['name']}_e{e}"] = {"epochs": e, "lr": g["lr"], "fg_restrict": g["fg_restrict"], "path": PRED_DIR / f"{g['name']}_e{e}"}

    scores_path = SEL_DIR / "dev_scores.json"
    paths = [str(c["path"] / "predictions.npz") for c in candidates.values()]
    paths.append(str(ROOT / "outputs" / "anomaldf" / "predictions.npz"))
    run([sys.executable, "-m", "eval.train.evaluate", "--predictions", *paths, "--scores", str(scores_path)])

    exp = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]["rgb_mvtec_multiclass"]
    coefs = {m: abs(s["coefficient"]) for m, s in exp["metrics"].items()}
    results = json.loads(scores_path.read_text())["experiments"]["rgb_mvtec_multiclass"]["results"]

    def weighted(row: dict) -> float:
        return float(sum(coefs[m] * row[m] for m in coefs))

    table = {name: {"dev_weighted_score": weighted(results[name]), **results[name], **{k: v for k, v in c.items() if k != "path"}}
             for name, c in candidates.items()}
    best = max(table, key=lambda k: table[k]["dev_weighted_score"])
    setup = {
        "epochs": table[best]["epochs"],
        "lr": table[best]["lr"],
        "fg_restrict": table[best]["fg_restrict"],
        "selected": best,
        "selected_on": "visible development slice (eval/train), seed 0; hidden test never read",
        "criterion": "sum over scored metrics of |coefficient| x development value",
        "baseline_dev_weighted_score": weighted(results["anomaldf"]),
        "candidates": table,
    }
    SETUP_FILE.write_text(json.dumps(setup, indent=2))
    print(json.dumps({k: setup[k] for k in ("epochs", "lr", "fg_restrict", "selected", "baseline_dev_weighted_score")}, indent=2))
    for name in sorted(table, key=lambda k: -table[k]["dev_weighted_score"]):
        r = table[name]
        print(f"{name:14s} S={r['dev_weighted_score']:8.2f} I-AUROC={r['image_auroc']:6.2f} P-AUROC={r['pixel_auroc']:6.2f} AUPRO={r['pixel_aupro']:6.2f}", flush=True)


if __name__ == "__main__":
    main()
