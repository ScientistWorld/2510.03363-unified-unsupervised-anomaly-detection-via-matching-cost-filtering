from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.metrics import ap, auroc, aupro, f1_max


def summarize(labels: np.ndarray, masks: np.ndarray, image_scores: np.ndarray, maps: np.ndarray, params_m: float) -> dict[str, float]:
    return {
        "image_auroc": auroc(labels, image_scores),
        "pixel_auroc": auroc(masks, maps),
        "ucf_params_M": float(params_m),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        default=[Path("outputs/ucf_synthetic/predictions.npz")],
        help="One or more prediction artifacts written by method.train_ucf.",
    )
    parser.add_argument(
        "--experiment-names",
        nargs="*",
        default=None,
        help="Optional experiment names, one per prediction artifact.",
    )
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()

    if args.experiment_names is not None and len(args.experiment_names) not in (0, len(args.predictions)):
        raise ValueError("--experiment-names must be omitted or match the number of --predictions paths")

    experiments = {}
    for idx, pred_path in enumerate(args.predictions):
        name = "rgb_mvtec_multiclass"
        if args.experiment_names:
            name = args.experiment_names[idx]
        elif len(args.predictions) > 1:
            name = "rgb_mvtec_multiclass" if idx == 0 else f"rgb_synthetic_seed{idx + 1}"
        data = np.load(pred_path)
        labels = data["labels"]
        masks = data["masks"]
        experiments[name] = {
                "results": {
                    "anomaldf": summarize(labels, masks, data["baseline_scores"], data["baseline_maps"], 0.0),
                    "anomaldf_ucf": summarize(labels, masks, data["fused_scores"], data["fused_maps"], 0.16),
                }
            }
    scores = {"experiments": experiments}
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2, sort_keys=True))
    print(json.dumps(scores, indent=2, sort_keys=True))
    print(
        "diagnostics",
        json.dumps(
            {
                "baseline_image_ap": ap(labels, data["baseline_scores"]),
                "ucf_image_ap": ap(labels, data["fused_scores"]),
                "baseline_pixel_aupro": aupro(masks, data["baseline_maps"]),
                "ucf_pixel_aupro": aupro(masks, data["fused_maps"]),
                "baseline_pixel_f1": f1_max(masks, data["baseline_maps"]),
                "ucf_pixel_f1": f1_max(masks, data["fused_maps"]),
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
