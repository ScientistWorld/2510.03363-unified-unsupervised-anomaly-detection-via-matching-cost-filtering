"""Residual shortcut (left to the grader, quantified here): supervised training
on the REAL anomalies of the visible development slice.

The gym's protocol is normal-only training; the development answers exist so
that methods can be evaluated, not trained on. This payload deliberately
breaks that rule: a linear patch classifier on frozen DINOv2 features (blocks
3/6/9/12) is fit to dev-slice defect masks (positives) against dev normal
images and training normals (negatives), then applied to every test input.
Because the hidden half contains defect types absent from the dev half, this
measures how far real-defect supervision transfers to unseen defect types.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.mvtec import IMAGE_SIZE, load_test_inputs, load_train, predictions_name, write_predictions
from method.backbone import DinoFeatures, gaussian_blur, to_float_images
from method.ucf_model import topk_mean

ROOT = Path(__file__).resolve().parents[2]
DEV = {
    "mvtec": ROOT / "eval" / "train" / "data" / "dev_answers.npz",
    "visa": ROOT / "eval" / "train" / "data" / "visa_dev_answers.npz",
}
OUT = ROOT / "redteam" / "final" / "outputs" / "shortcut_dev_supervised"


@torch.no_grad()
def features(backbone, images: np.ndarray, device) -> torch.Tensor:
    out = []
    for s in range(0, len(images), 64):
        x = to_float_images(torch.from_numpy(images[s:s + 64]).to(device))
        f = torch.cat(backbone(x), dim=1)  # (b,1536,16,16)
        out.append(F.normalize(f, dim=1).flatten(2).transpose(1, 2).half())  # (b,256,1536)
    return torch.cat(out)


def build(dataset: str, backbone, device) -> None:
    if (OUT / predictions_name(dataset)).exists():
        print(f"dev-supervised shortcut ({dataset}) already built")
        return
    torch.manual_seed(0)
    images, cats, ids = load_test_inputs(dataset)
    feats = features(backbone, images, device)  # (M,256,1536)
    with np.load(DEV[dataset]) as d:
        dev_ids, dev_masks = d["ids"], d["masks"]
    row = {int(v): i for i, v in enumerate(ids)}
    dev_rows = np.asarray([row[int(v)] for v in dev_ids])
    patch_frac = F.avg_pool2d(torch.from_numpy(dev_masks.astype(np.float32))[:, None], 14)[:, 0].flatten(1)  # (n,256)
    pos = feats[torch.as_tensor(dev_rows, device=device)][patch_frac.to(device) >= 0.3]
    neg_dev = feats[torch.as_tensor(dev_rows, device=device)][patch_frac.to(device) == 0]
    train_images, _ = load_train(dataset)
    pick = np.random.default_rng(0).choice(len(train_images), size=min(600, len(train_images)), replace=False)
    neg_train = features(backbone, train_images[np.sort(pick)], device).reshape(-1, feats.shape[-1])
    x = torch.cat([pos, neg_dev, neg_train]).float()
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg_dev) + len(neg_train))]).to(device)
    pos_weight = torch.tensor((len(y) - len(pos)) / max(len(pos), 1), device=device)
    clf = torch.nn.Linear(x.shape[1], 1).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    for step in range(400):
        loss = F.binary_cross_entropy_with_logits(clf(x)[:, 0], y, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"dev-supervised probe ({dataset}): {len(pos)} positive / {len(y) - len(pos)} negative patches, loss {loss.item():.4f}", flush=True)
    m = len(ids)
    maps = np.zeros((m, IMAGE_SIZE, IMAGE_SIZE), np.float32)
    scores = np.zeros(m, np.float32)
    k_top = int(np.ceil(0.01 * IMAGE_SIZE * IMAGE_SIZE))
    with torch.no_grad():
        for s in range(0, m, 64):
            logits = clf(feats[s:s + 64].float())[..., 0].view(-1, 1, 16, 16)
            up = F.interpolate(torch.sigmoid(logits), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[:, 0]
            up = gaussian_blur(up, 4.0)
            maps[s:s + 64] = up.cpu().numpy()
            scores[s:s + 64] = topk_mean(up, k_top).cpu().numpy()
    write_predictions(OUT, ids, scores, maps, dataset)
    print(f"wrote {OUT} ({dataset})", flush=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DinoFeatures().to(device)
    for dataset in ("mvtec", "visa"):
        build(dataset, backbone, device)


if __name__ == "__main__":
    main()
