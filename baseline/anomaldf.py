"""AnomalDF baseline: patch matching against normal templates on frozen DINOv2.

Each test patch token (last ViT-S/14 block, 16x16 grid at 224px) is scored by
its cosine distance to the nearest patch token of the templates. Default:
N = 3 random same-category templates per input (the paper's AnomalDF
protocol, identical draws to AnomalDF+UCF). `--templates all` matches against
every training image of the category (full memory bank, the exhaustive
template search the paper's method avoids). Maps are upsampled to 224px and
smoothed with a Gaussian (sigma 4); the image score is the mean of the top 1%
of map values.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.mvtec import IMAGE_SIZE, load_test_inputs, load_train, write_predictions
from method.backbone import LAST_LAYER, DinoFeatures, gaussian_blur, to_float_images
from method.templates import category_index, test_templates
from method.ucf_model import topk_mean


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("outputs/anomaldf"))
    p.add_argument("--templates", default="3", help="number of random templates, or 'all'")
    p.add_argument("--template-seed", type=int, default=2026)
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--top-percent", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DinoFeatures(layers=LAST_LAYER).to(device)
    train_images, train_cats = load_train()
    index = category_index(train_cats)
    bank = []
    for s in range(0, len(train_images), 64):
        x = to_float_images(torch.from_numpy(train_images[s:s + 64]).to(device))
        f = backbone(x)[0]  # (b,384,16,16)
        bank.append(F.normalize(f.flatten(2).transpose(1, 2), dim=-1).half())  # (b,256,384)
    bank = torch.cat(bank)

    images, cats, ids = load_test_inputs()
    m = len(ids)
    maps = np.zeros((m, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    scores = np.zeros(m, dtype=np.float32)
    k_top = max(1, int(math.ceil(args.top_percent / 100.0 * IMAGE_SIZE * IMAGE_SIZE)))
    bs = 32
    for s in range(0, m, bs):
        x = to_float_images(torch.from_numpy(images[s:s + bs]).to(device))
        q = backbone(x)[0]
        b, c, h, w = q.shape
        qn = F.normalize(q.flatten(2).transpose(1, 2), dim=-1).half()  # (b,256,384)
        dist = torch.empty(b, h * w, device=device)
        for j in range(b):
            i = s + j
            if args.templates == "all":
                t_idx = index[int(cats[i])]
            else:
                t_idx = test_templates(index, cats[i], int(args.templates), args.template_seed, 0, ids[i])
            ref = bank[torch.as_tensor(t_idx, device=device)].reshape(-1, c)  # (N*256,384)
            sim = (qn[j] @ ref.T).float()
            dist[j] = 1.0 - sim.amax(dim=1)
        up = F.interpolate(dist.view(b, 1, h, w), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[:, 0]
        if args.sigma > 0:
            up = gaussian_blur(up, args.sigma)
        maps[s:s + bs] = up.cpu().numpy()
        scores[s:s + bs] = topk_mean(up, k_top).cpu().numpy()
    path = write_predictions(args.output, ids, scores, maps)
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "method": "AnomalDF patch matching (no cost filtering)",
                "backbone": "DINOv2 ViT-S/14, last block, 224px",
                "templates": args.templates,
                "template_seed": args.template_seed,
                "gaussian_sigma": args.sigma,
                "image_score": f"mean of top {args.top_percent}% map values",
            },
            indent=2,
        )
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
