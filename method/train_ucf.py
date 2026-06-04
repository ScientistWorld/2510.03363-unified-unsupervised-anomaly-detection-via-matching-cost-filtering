from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from data.synthetic_uad import SyntheticConfig, SyntheticUADDataset, iter_dataset_summary
from method.ucf_model import CostVolumeBuilder, UCFNet, baseline_map


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    g = gamma.reshape(-1, 1, 1).to(logits.device)
    return (((1 - pt) ** g) * ce).mean()


def soft_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = logits.softmax(dim=1)[:, 1]
    tgt = target.float()
    inter = (prob * tgt).sum(dim=(1, 2))
    union = (prob + tgt - prob * tgt).sum(dim=(1, 2))
    return (1 - (inter + 1.0) / (union + 1.0)).mean()


def ssim_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = logits.softmax(dim=1)[:, 1:2]
    tgt = target.float()[:, None]
    mu_x = F.avg_pool2d(prob, 7, stride=1, padding=3)
    mu_y = F.avg_pool2d(tgt, 7, stride=1, padding=3)
    sig_x = F.avg_pool2d(prob * prob, 7, stride=1, padding=3) - mu_x * mu_x
    sig_y = F.avg_pool2d(tgt * tgt, 7, stride=1, padding=3) - mu_y * mu_y
    sig_xy = F.avg_pool2d(prob * tgt, 7, stride=1, padding=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2) + 1e-6)
    return (1 - ssim.clamp(-1, 1)).mean()


def topk_image_score(maps: torch.Tensor, k: int = 250) -> torch.Tensor:
    flat = maps.flatten(1)
    k = min(k, flat.shape[1])
    return flat.topk(k, dim=1).values.mean(dim=1)


def run_epoch(model: UCFNet, builder: CostVolumeBuilder, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device, alpha: float) -> float:
    model.train()
    losses = []
    for batch in loader:
        image = batch["image"].to(device)
        templates = batch["templates"].to(device)
        mask = batch["mask"].to(device)
        category = batch["category"].to(device)
        volume, initial, _ = builder(image, templates)
        logits, cls = model(volume, image, initial)
        with torch.no_grad():
            pred_cls = cls.argmax(dim=1)
            prob_true = cls.softmax(dim=1).gather(1, category[:, None]).squeeze(1)
            gamma = torch.where(pred_cls == category, 3.0 - prob_true, torch.full_like(prob_true, 3.0)).clamp_min(0.5)
        loss = focal_loss(logits, mask, gamma)
        loss = loss + F.cross_entropy(cls, category)
        loss = loss + alpha * (soft_iou_loss(logits, mask) + ssim_loss(logits, mask))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model: UCFNet, builder: CostVolumeBuilder, loader: DataLoader, device: torch.device, out_dir: Path, lam: float) -> None:
    model.eval()
    labels, categories, indices = [], [], []
    masks, ucf_maps, base_maps, fused_maps = [], [], [], []
    ucf_scores, base_scores, fused_scores = [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        templates = batch["templates"].to(device)
        volume, initial, _ = builder(image, templates)
        logits, _ = model(volume, image, initial)
        ucf = logits.softmax(dim=1)[:, 1]
        base = baseline_map(volume, image.shape[-2:])
        fused = lam * ucf + (1.0 - lam) * base
        labels.append(batch["label"].numpy())
        categories.append(batch["category"].numpy())
        indices.append(batch["index"].numpy())
        masks.append(batch["mask"].numpy().astype(np.uint8))
        ucf_maps.append(ucf.cpu().numpy().astype(np.float32))
        base_maps.append(base.cpu().numpy().astype(np.float32))
        fused_maps.append(fused.cpu().numpy().astype(np.float32))
        ucf_scores.append(topk_image_score(ucf).cpu().numpy())
        base_scores.append(topk_image_score(base).cpu().numpy())
        fused_scores.append(topk_image_score(fused).cpu().numpy())
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz",
        labels=np.concatenate(labels),
        categories=np.concatenate(categories),
        indices=np.concatenate(indices),
        masks=np.concatenate(masks),
        ucf_maps=np.concatenate(ucf_maps),
        baseline_maps=np.concatenate(base_maps),
        fused_maps=np.concatenate(fused_maps),
        ucf_scores=np.concatenate(ucf_scores),
        baseline_scores=np.concatenate(base_scores),
        fused_scores=np.concatenate(fused_scores),
    )
    (out_dir / "metadata.json").write_text(
        json.dumps({"dataset": list(iter_dataset_summary()), "lambda": lam, "topk": 250}, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lambda-fusion", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("outputs/ucf_synthetic"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ucf_synthetic.pt"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SyntheticConfig()
    train_ds = SyntheticUADDataset("train", cfg, include_synthetic_train_anomalies=True)
    test_ds = SyntheticUADDataset("test", cfg)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    builder = CostVolumeBuilder().to(device)
    model = UCFNet(categories=len(cfg.categories)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    best = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        loss = run_epoch(model, builder, train_loader, opt, device, args.alpha)
        sched.step(loss)
        if loss < best:
            best = loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg.__dict__}, args.checkpoint)
        print(f"epoch={epoch} loss={loss:.4f} lr={opt.param_groups[0]['lr']:.2e}", flush=True)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    predict(model, builder, test_loader, device, args.output, args.lambda_fusion)


if __name__ == "__main__":
    main()
