from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.synthetic_uad import SyntheticConfig, SyntheticUADDataset, iter_dataset_summary
from method.ucf_model import CostVolumeBuilder, baseline_map
from method.train_ucf import topk_image_score


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/raw_cost_volume"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SyntheticConfig()
    dataset = SyntheticUADDataset("test", cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    builder = CostVolumeBuilder(template_count=cfg.template_count).to(device)

    labels, categories, indices = [], [], []
    masks, maps, scores = [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        templates = batch["templates"].to(device)
        volume, _, _ = builder(image, templates)
        score_map = baseline_map(volume, image.shape[-2:])
        labels.append(batch["label"].numpy())
        categories.append(batch["category"].numpy())
        indices.append(batch["index"].numpy())
        masks.append(batch["mask"].numpy().astype(np.uint8))
        maps.append(score_map.cpu().numpy().astype(np.float32))
        scores.append(topk_image_score(score_map).cpu().numpy())

    maps_arr = np.concatenate(maps)
    scores_arr = np.concatenate(scores)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "predictions.npz",
        labels=np.concatenate(labels),
        categories=np.concatenate(categories),
        indices=np.concatenate(indices),
        masks=np.concatenate(masks),
        baseline_maps=maps_arr,
        baseline_scores=scores_arr,
        fused_maps=maps_arr,
        fused_scores=scores_arr,
        added_params_M=np.asarray([0.0], dtype=np.float32),
        added_flops_G=np.asarray([0.0], dtype=np.float32),
    )
    (args.output / "metadata.json").write_text(
        json.dumps({"dataset": list(iter_dataset_summary()), "baseline": "raw min-pooled cost volume"}, indent=2)
    )


if __name__ == "__main__":
    main()
