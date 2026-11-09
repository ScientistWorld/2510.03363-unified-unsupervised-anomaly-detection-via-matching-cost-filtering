"""Train AnomalDF+UCF on MVTec-AD normal images and write test predictions.

Training (paper Sec. III-G, S4-A/B): 40 epochs, batch 8, Adam with a
ReduceLROnPlateau scheduler, loss
    L = Focal(M, M_s, gamma(sigmoid(y_c))) + CE(y_c, Y) + alpha * (SoftIoU + SSIM),  alpha = 0.1, gamma0 = 3,
on Perlin-noise synthetic anomalies. N = 3 same-category templates are drawn
at random per input (training and testing). The image score is the mean of
the top-250 anomaly-map values.

Usage:
  python -m method.train_ucf --output outputs/anomaldf_ucf --checkpoint checkpoints/anomaldf_ucf_seed0.pt --seed 0
  python -m method.train_ucf --predict-only --checkpoint ... --output ... [--test-draws 4 | --test-templates nearest]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.mvtec import CATEGORIES, IMAGE_SIZE, load_test_inputs, load_textures, load_train, write_predictions
from method.anomaly_synthesis import AnomalySynthesizer
from method.backbone import DinoFeatures, to_float_images
from method.templates import category_index, nearest_templates, test_templates, train_templates
from method.ucf_model import UCFNet, build_cost_volume, topk_mean

GRID = 16  # 224 / 14 patch grid
TOPK = 250
RESUME_KEYS = ("templates", "no_guidance", "seed", "lr", "batch_size", "alpha", "gamma0")


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    logpt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
    pt = logpt.exp()
    # clamp keeps d/dpt (1-pt)^gamma finite when gamma < 1 and pt rounds to 1
    return (-((1.0 - pt).clamp_min(1e-6) ** gamma.view(-1, 1, 1)) * logpt).mean()


def soft_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = logits.softmax(dim=1)[:, 1]
    tgt = target.float()
    inter = (prob * tgt).sum(dim=(1, 2))
    union = (prob + tgt - prob * tgt).sum(dim=(1, 2))
    return (1.0 - (inter + 1.0) / (union + 1.0)).mean()


def ssim_loss(logits: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    x = logits.softmax(dim=1)[:, 1:2]
    y = target.float()[:, None]
    pad = window // 2
    mu_x = F.avg_pool2d(x, window, 1, pad)
    mu_y = F.avg_pool2d(y, window, 1, pad)
    sxx = F.avg_pool2d(x * x, window, 1, pad) - mu_x**2
    syy = F.avg_pool2d(y * y, window, 1, pad) - mu_y**2
    sxy = F.avg_pool2d(x * y, window, 1, pad) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return (1.0 - ssim).mean()


@torch.no_grad()
def feature_cache(backbone: DinoFeatures, images: torch.Tensor, batch: int = 64) -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N, L, C, 16, 16) float16."""
    out = []
    for s in range(0, images.shape[0], batch):
        feats = backbone(to_float_images(images[s:s + batch]))
        out.append(torch.stack(feats, dim=1).half())
    return torch.cat(out)


def gather_templates(cache: torch.Tensor, t_idx: np.ndarray) -> list[torch.Tensor]:
    t = cache[torch.as_tensor(t_idx, device=cache.device)]  # (B,N,L,C,H,W)
    return [t[:, :, layer].float() for layer in range(t.shape[2])]


def build_model(args, device: torch.device) -> UCFNet:
    return UCFNet(
        match_channels=args.templates * GRID * GRID,
        num_classes=len(CATEGORIES),
        use_guidance=not args.no_guidance,
    ).to(device)


def config_of(args) -> dict:
    return {k: getattr(args, k) for k in RESUME_KEYS}


def train(args, device, backbone, cache, train_images, train_cats_np, index) -> UCFNet:
    model = build_model(args, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    rng = np.random.default_rng(args.seed)
    synth = AnomalySynthesizer(torch.from_numpy(load_textures()).to(device), seed=args.seed + 1)
    torch.manual_seed(args.seed)
    train_cats = torch.from_numpy(train_cats_np).long().to(device)
    start = 1
    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if state.get("config") == config_of(args):
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            rng.bit_generator.state = state["rng"]
            synth.load_state(state["synth"])
            start = int(state["epoch"]) + 1
            print(f"resuming from epoch {state['epoch']}", flush=True)
        else:
            print("checkpoint config differs; training from scratch", flush=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    n = len(train_cats_np)
    for epoch in range(start, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = rng.permutation(n)
        total = torch.zeros((), device=device)
        steps = 0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            idx_t = torch.as_tensor(idx, device=device)
            images = to_float_images(train_images[idx_t])
            aug, masks = synth(images)
            q = backbone(aug)
            t_idx = np.stack([train_templates(index, train_cats_np[i], args.templates, rng, exclude=int(i)) for i in idx])
            volume, initial = build_cost_volume(q, gather_templates(cache, t_idx))
            logits, cls = model(volume, initial, q)
            logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
            cats = train_cats[idx_t]
            target = masks.long()
            with torch.no_grad():
                gamma = args.gamma0 * torch.sigmoid(cls.gather(1, cats[:, None]).squeeze(1))
            loss = (
                focal_loss(logits, target, gamma)
                + F.cross_entropy(cls, cats)
                + args.alpha * (soft_iou_loss(logits, target) + ssim_loss(logits, target))
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach()
            steps += 1
        mean_loss = float(total.item() / max(steps, 1))
        sched.step(mean_loss)
        state = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "rng": rng.bit_generator.state,
            "synth": synth.state(),
            "epoch": epoch,
            "config": config_of(args),
        }
        tmp = args.checkpoint.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(args.checkpoint)
        print(f"epoch={epoch} loss={mean_loss:.4f} lr={opt.param_groups[0]['lr']:.2e} time={time.time() - t0:.0f}s", flush=True)
    return model


@torch.no_grad()
def predict(args, device, model, backbone, cache, train_cats_np, index) -> None:
    model.eval()
    images, cats, ids = load_test_inputs()
    m = len(ids)
    maps = np.zeros((m, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    scores = np.zeros(m, dtype=np.float32)
    train_desc = F.normalize(cache[:, -1].float().mean(dim=(2, 3)), dim=1)
    bs = 16
    for s in range(0, m, bs):
        rows = range(s, min(s + bs, m))
        x = to_float_images(torch.from_numpy(images[s:s + bs]).to(device))
        q = backbone(x)
        acc = torch.zeros(x.shape[0], IMAGE_SIZE, IMAGE_SIZE, device=device)
        draws = 1 if args.test_templates == "nearest" else args.test_draws
        for draw in range(draws):
            if args.test_templates == "nearest":
                desc = F.normalize(q[-1].mean(dim=(2, 3)), dim=1)
                t_idx = np.stack([nearest_templates(index, cats[i], args.templates, desc[j], train_desc) for j, i in enumerate(rows)])
            else:
                t_idx = np.stack([test_templates(index, cats[i], args.templates, args.template_seed, draw, ids[i]) for i in rows])
            volume, initial = build_cost_volume(q, gather_templates(cache, t_idx))
            logits, _ = model(volume, initial, q)
            logits = F.interpolate(logits, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
            acc += logits.softmax(dim=1)[:, 1]
        prob = acc / draws
        maps[s:s + bs] = prob.cpu().numpy()
        scores[s:s + bs] = topk_mean(prob, TOPK).cpu().numpy()
    path = write_predictions(args.output, ids, scores, maps)
    meta = {
        "method": "AnomalDF+UCF (cost-volume filtering)",
        "backbone": "DINOv2 ViT-S/14, blocks 3/6/9/12, 224px",
        "templates": args.templates,
        "test_templates": args.test_templates,
        "test_draws": draws,
        "template_seed": args.template_seed,
        "guidance": not args.no_guidance,
        "seed": args.seed,
        "epochs": args.epochs,
        "image_score": f"mean of top-{TOPK} anomaly-map values",
        "trainable_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
    }
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma0", type=float, default=3.0)
    p.add_argument("--templates", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-guidance", action="store_true", help="ablation: drop both guidance streams")
    p.add_argument("--output", type=Path, default=Path("outputs/anomaldf_ucf"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/anomaldf_ucf_seed0.pt"))
    p.add_argument("--test-draws", type=int, default=1, help="random template draws averaged at test time")
    p.add_argument("--test-templates", choices=("random", "nearest"), default="random")
    p.add_argument("--template-seed", type=int, default=2026)
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("CUDA is not available; pass --allow-cpu to run (very slowly) on CPU")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    train_images_np, train_cats_np = load_train()
    train_images = torch.from_numpy(train_images_np).to(device)
    index = category_index(train_cats_np)
    backbone = DinoFeatures().to(device)
    cache = feature_cache(backbone, train_images)

    if args.predict_only:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if int(state["epoch"]) < args.epochs:
            raise SystemExit(f"{args.checkpoint} holds epoch {state['epoch']} < {args.epochs}")
        model = build_model(args, device)
        model.load_state_dict(state["model"])
    else:
        model = train(args, device, backbone, cache, train_images, train_cats_np, index)
    predict(args, device, model, backbone, cache, train_cats_np, index)


if __name__ == "__main__":
    main()
