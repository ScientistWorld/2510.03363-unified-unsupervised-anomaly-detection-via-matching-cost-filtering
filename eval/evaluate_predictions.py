from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from eval.metrics import ap, auroc, aupro, f1_max


def _category_mean_auroc(categories: np.ndarray, labels: np.ndarray, masks: np.ndarray, image_scores: np.ndarray, maps: np.ndarray) -> tuple[float, float]:
    image_values = []
    pixel_values = []
    for category in np.unique(categories):
        keep = categories == category
        image_values.append(auroc(labels[keep], image_scores[keep]))
        pixel_values.append(auroc(masks[keep], maps[keep]))
    return float(np.nanmean(image_values)), float(np.nanmean(pixel_values))


def summarize(
    labels: np.ndarray,
    categories: np.ndarray,
    masks: np.ndarray,
    image_scores: np.ndarray,
    maps: np.ndarray,
    params_m: float,
    flops_g: float,
) -> dict[str, float]:
    category_image, category_pixel = _category_mean_auroc(categories, labels, masks, image_scores, maps)
    return {
        "image_auroc": auroc(labels, image_scores),
        "pixel_auroc": auroc(masks, maps),
        "image_ap": ap(labels, image_scores),
        "image_f1_max": f1_max(labels, image_scores),
        "pixel_ap": ap(masks, maps),
        "pixel_f1_max": f1_max(masks, maps),
        "pixel_aupro": aupro(masks, maps),
        "pixel_aupro_5pct": aupro(masks, maps, max_fpr=0.05),
        "category_mean_image_auroc": category_image,
        "category_mean_pixel_auroc": category_pixel,
        "added_params_M": float(params_m),
        "added_flops_G": float(flops_g),
    }


def load_metadata_value(prediction_path: Path, data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        if key in data.files:
            return float(np.asarray(data[key]).reshape(-1)[0])
    metadata_path = prediction_path.parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        for key in keys:
            if key in metadata:
                return float(metadata[key])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        default=[Path("outputs/ucf_synthetic/predictions.npz")],
        help="One or more prediction artifacts written by a submitted method.",
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
            parent_name = pred_path.parent.name
            if idx == 0:
                name = "rgb_mvtec_multiclass"
            elif "seed13" in parent_name:
                name = "rgb_synthetic_seed13"
            else:
                name = f"rgb_synthetic_seed{idx + 1}"
        data = np.load(pred_path)
        params_m = load_metadata_value(pred_path, data, ("params_m", "added_params_M", "method_params_M", "ucf_params_M"))
        flops_g = load_metadata_value(pred_path, data, ("flops_g", "added_flops_G", "method_flops_G", "ucf_flops_G"))
        if params_m is None or flops_g is None:
            raise ValueError(
                f"{pred_path} must include added_params_M and added_flops_G in the .npz artifact or metadata.json"
            )
        labels = data["labels"]
        categories = data["categories"]
        masks = data["masks"]
        experiments[name] = {
                "results": {
                    "anomaldf": summarize(labels, categories, masks, data["baseline_scores"], data["baseline_maps"], 0.0, 0.0),
                    "anomaldf_ucf": summarize(labels, categories, masks, data["fused_scores"], data["fused_maps"], params_m, flops_g),
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
