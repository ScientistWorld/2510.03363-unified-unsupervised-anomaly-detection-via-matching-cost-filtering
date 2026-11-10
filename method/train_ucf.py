"""Train AnomalDF+UCF on MVTec-AD normal images and write test predictions.

Training (paper Sec. III-G, S4-A/B): batch 8, Adam with a ReduceLROnPlateau
scheduler, loss
    L = Focal(M, M_s, gamma) + CE(y_c, Y) + alpha * (SoftIoU + SSIM),  alpha = 0.1,
with the class-aware adaptor setting the focal parameter per sample:
gamma = gamma0 - sigmoid(y_c) when the category is predicted correctly and
gamma = gamma0 otherwise (gamma0 = 3). Synthetic anomalies are Perlin-noise
masks (restricted to the object foreground for object categories,
GLAD-style). N = 3 same-category templates are drawn at random per input
(training and testing).

Inference (Sec. III-F/G, method/fusion.py): the network's anomaly probability
M is fused with the baseline (AnomalDF) response for the same templates,
lambda * M + (1 - lambda) * M_baseline with lambda = 0.5, and the image score
is the mean of the top-250 values of the fused map.

Setup defaults (epochs, learning rate, foreground restriction) come from
method/ucf_setup.json, written by method/select_setup.py, which chooses them
on the visible development slice only. Without that file the paper defaults
apply (40 epochs, lr 1e-3, foreground restriction on).

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
from method.foreground import foreground_masks
from method.fusion import COMPONENTS, LAMBDA, combine, fuse
from method.templates import category_index, nearest_templates, test_templates, train_templates
from method.ucf_model import ARCH_VERSION, UCFNet, build_cost_volume

GRID = 16  # 224 / 14 patch grid
GAMMA_RULE = 2  # 2: gamma = gamma0 - sigmoid(y_c) if the category is predicted correctly, else gamma0 (paper Sec. III-G)
RESUME_KEYS = ("templates", "no_guidance", "seed", "lr", "batch_size", "alpha", "gamma0", "fg_restrict", "train_fraction", "loss")
SETUP_FILE = Path(__file__).resolve().parent / "ucf_setup.json"
PAPER_SETUP = {"epochs": 40, "lr": 1e-3, "fg_restrict": True}


def load_setup() -> dict:
    setup = dict(PAPER_SETUP)
    if SETUP_FILE.exists():
        chosen = json.loads(SETUP_FILE.read_text())
        setup.update({k: chosen[k] for k in PAPER_SETUP if k in chosen})
    return setup


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
    return {"arch": ARCH_VERSION, "gamma_rule": GAMMA_RULE, **{k: getattr(args, k) for k in RESUME_KEYS}}


def adaptor_gamma(cls: torch.Tensor, cats: torch.Tensor, gamma0: float) -> torch.Tensor:
    """Class-aware adaptor: gamma0 - sigmoid(y_c) for correctly classified samples, gamma0 otherwise."""
    true_logit = cls.gather(1, cats[:, None]).squeeze(1)
    correct = cls.argmax(dim=1) == cats
    return torch.where(correct, gamma0 - torch.sigmoid(true_logit), torch.full_like(true_logit, gamma0))


def category_banks(cache_last: torch.Tensor, index: dict[int, np.ndarray]) -> dict[int, torch.Tensor]:
    """L2-normalised last-layer patch tokens of every training image, per category: (n*h*w, C) half."""
    banks = {}
    for c, idx in index.items():
        t = cache_last[torch.as_tensor(idx, device=cache_last.device)].float()  # (n,C,h,w)
        banks[c] = F.normalize(t.flatten(2).transpose(1, 2).reshape(-1, t.shape[1]), dim=1).half()
    return banks


def full_bank_cost(q_last: torch.Tensor, cats: np.ndarray, banks: dict[int, torch.Tensor]) -> torch.Tensor:
    """Min cosine cost of every query token against the whole category memory bank: (B,h,w)."""
    b, _, h, w = q_last.shape
    qn = F.normalize(q_last.float().flatten(2).transpose(1, 2), dim=-1).half()  # (B,hw,C)
    out = torch.empty(b, h * w, device=q_last.device)
    for j in range(b):
        out[j] = 1.0 - (qn[j] @ banks[int(cats[j])].T).float().amax(dim=1)
    return out.view(b, h, w)


def train(args, device, backbone, cache, train_images, train_cats_np, index) -> UCFNet:
    model = build_model(args, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    rng = np.random.default_rng(args.seed)
    synth = AnomalySynthesizer(torch.from_numpy(load_textures()).to(device), seed=args.seed + 1)
    torch.manual_seed(args.seed)
    train_cats = torch.from_numpy(train_cats_np).long().to(device)
    fg_small = None
    if args.fg_restrict:
        fg_small, fg_stats = foreground_masks(cache[:, -1], train_cats_np, CATEGORIES)
        print("foreground masks: " + json.dumps(fg_stats), flush=True)
    queries = np.arange(len(train_cats_np))
    if args.train_fraction < 1.0:
        sub_rng = np.random.default_rng(args.seed + 7)
        keep = []
        for c in np.unique(train_cats_np):
            pool = np.flatnonzero(train_cats_np == c)
            keep.append(sub_rng.choice(pool, size=max(1, int(round(args.train_fraction * len(pool)))), replace=False))
        queries = np.sort(np.concatenate(keep))
        print(f"training queries: {len(queries)} of {len(train_cats_np)} images", flush=True)
    snapshots = {int(e) for e in args.snapshot_epochs.split(",") if e.strip()} if args.snapshot_epochs else set()
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
    n = len(queries)
    for epoch in range(start, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = queries[rng.permutation(n)]
        total = torch.zeros((), device=device)
        steps = 0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            idx_t = torch.as_tensor(idx, device=device)
            images = to_float_images(train_images[idx_t])
            fg = None
            if fg_small is not None:
                fg = F.interpolate(fg_small[idx_t].float()[:, None], size=images.shape[-2:], mode="nearest")[:, 0]
            aug, masks = synth(images, fg)
            q = backbone(aug)
            t_idx = np.stack([train_templates(index, train_cats_np[i], args.templates, rng, exclude=int(i)) for i in idx])
            volume, initial = build_cost_volume(q, gather_templates(cache, t_idx))
            logits, cls = model(volume, initial, q)
            logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
            cats = train_cats[idx_t]
            target = masks.long()
            if args.loss == "focal":  # ablation (paper Table IX): focal loss alone, no adaptor / CE / structural terms
                loss = focal_loss(logits, target, torch.full((logits.shape[0],), args.gamma0, device=device))
            else:
                with torch.no_grad():
                    gamma = adaptor_gamma(cls, cats, args.gamma0)
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
        if epoch in snapshots and epoch < args.epochs:
            snap_ckpt = args.checkpoint.with_name(f"{args.checkpoint.stem}_e{epoch}.pt")
            torch.save(state, snap_ckpt)
            snap_out = args.output.with_name(f"{args.output.name}_e{epoch}")
            if not (snap_out / "predictions.npz").exists():
                predict(args, device, model, backbone, cache, train_cats_np, index, output=snap_out, epoch=epoch)
    return model


@torch.no_grad()
def predict(args, device, model, backbone, cache, train_cats_np, index, output: Path | None = None, epoch: int | None = None) -> None:
    output = args.output if output is None else output
    epoch = args.epochs if epoch is None else epoch
    model.eval()
    images, cats, ids = load_test_inputs()
    m = len(ids)
    maps = np.zeros((m, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    scores = np.zeros(m, dtype=np.float32)
    comp_logits = comp_cost = None
    train_desc = F.normalize(cache[:, -1].float().mean(dim=(2, 3)), dim=1)
    banks = category_banks(cache[:, -1], index) if args.baseline_bank == "all" else None
    draws = 1 if args.test_templates == "nearest" else args.test_draws
    bs = 16
    for s in range(0, m, bs):
        rows = range(s, min(s + bs, m))
        x = to_float_images(torch.from_numpy(images[s:s + bs]).to(device))
        q = backbone(x)
        logits_draws, cost_draws = [], []
        for draw in range(draws):
            if args.test_templates == "nearest":
                desc = F.normalize(q[-1].mean(dim=(2, 3)), dim=1)
                t_idx = np.stack([nearest_templates(index, cats[i], args.templates, desc[j], train_desc) for j, i in enumerate(rows)])
            else:
                t_idx = np.stack([test_templates(index, cats[i], args.templates, args.template_seed, draw, ids[i]) for i in rows])
            volume, initial = build_cost_volume(q, gather_templates(cache, t_idx))
            logits, _ = model(volume, initial, q)  # (B,2,16,16)
            logits_draws.append(logits.float())
            if args.baseline_bank == "templates":
                cost_draws.append(initial[:, -1].float())  # AnomalDF cost: last layer, min over the templates
        if args.baseline_bank == "all":
            cost_draws = [full_bank_cost(q[-1], cats[s:s + bs], banks)]
        prob, base = combine(logits_draws, cost_draws)
        maps[s:s + bs], scores[s:s + bs] = fuse(prob, base, args.lam)
        lg = torch.stack(logits_draws, dim=1).cpu().numpy()
        cs = torch.stack(cost_draws, dim=1).cpu().numpy()
        if comp_logits is None:
            comp_logits = np.zeros((m, *lg.shape[1:]), np.float32)
            comp_cost = np.zeros((m, *cs.shape[1:]), np.float32)
        comp_logits[s:s + bs] = lg
        comp_cost[s:s + bs] = cs
    path = write_predictions(output, ids, scores, maps)
    np.savez_compressed(output / COMPONENTS, ids=np.asarray(ids, np.int64), logits16=comp_logits, cost16=comp_cost)
    meta = {
        "method": "AnomalDF+UCF (cost-volume filtering, fused with the AnomalDF response)",
        "backbone": "DINOv2 ViT-S/14, blocks 3/6/9/12, 224px",
        "templates": args.templates,
        "test_templates": args.test_templates,
        "test_draws": draws,
        "template_seed": args.template_seed,
        "guidance": not args.no_guidance,
        "loss": args.loss,
        "seed": args.seed,
        "epochs": epoch,
        "lr": args.lr,
        "fg_restrict": args.fg_restrict,
        "train_fraction": args.train_fraction,
        "lambda": args.lam,
        "baseline_bank": args.baseline_bank,
        "anomaly_maps": "lambda * UCF anomaly probability + (1 - lambda) * AnomalDF map (same templates unless baseline_bank=all)",
        "image_score": "mean of the top-250 values of the fused map",
        "arch_version": ARCH_VERSION,
        "trainable_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
    }
    (output / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {path}", flush=True)


def main() -> None:
    setup = load_setup()
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=setup["epochs"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=setup["lr"])
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma0", type=float, default=3.0)
    p.add_argument("--templates", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-guidance", action="store_true", help="ablation: drop both guidance streams")
    p.add_argument("--loss", choices=("full", "focal"), default="full", help="ablation: 'focal' trains with the focal loss alone")
    p.add_argument("--lam", type=float, default=LAMBDA, help="inference fusion weight of the UCF map (paper: 0.5)")
    p.add_argument("--baseline-bank", choices=("templates", "all"), default="templates",
                   help="baseline map fused at inference: same N templates (paper) or the whole category memory bank")
    p.add_argument("--output", type=Path, default=Path("outputs/anomaldf_ucf"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/anomaldf_ucf_seed0.pt"))
    p.add_argument("--test-draws", type=int, default=1, help="random template draws averaged at test time")
    p.add_argument("--test-templates", choices=("random", "nearest"), default="random")
    p.add_argument("--template-seed", type=int, default=2026)
    p.add_argument("--fg-restrict", dest="fg_restrict", action="store_true", default=setup["fg_restrict"])
    p.add_argument("--no-fg-restrict", dest="fg_restrict", action="store_false")
    p.add_argument("--train-fraction", type=float, default=1.0, help="fraction of normal training images used as queries")
    p.add_argument("--snapshot-epochs", default="", help="comma list: also save + predict at these epochs (<output>_e<k>)")
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
