"""Train AnomalDF+UCF on one benchmark's normal images and write test predictions.

Datasets: --dataset mvtec (MVTec-AD, 15 categories) or visa (VisA, 12
categories); one unified model per dataset, as in the paper.

Training (paper Sec. III-G, S4-A/B): batch 8, Adam with a ReduceLROnPlateau
scheduler, 40 epochs, loss
    L = Focal(M, M_s, gamma) + CE(y_c, Y) + alpha * (SoftIoU + SSIM),  alpha = 0.1,
with the class-aware adaptor setting the focal parameter per sample:
gamma = gamma0 - sigmoid(y_c) when the category is predicted correctly and
gamma = gamma0 otherwise (gamma0 = 3). Synthetic anomalies are Perlin-noise
masks (restricted to the object foreground for object categories,
GLAD-style). N = 3 same-category templates are drawn at random per input.
The cost volume keeps the K = 256 lowest matching costs per input token
(method/ucf_model.py), so its shape does not depend on the number of
templates.

Inference (Sec. III-F/G, S-E, Table XXIV; method/fusion.py): by default the
trained model is applied, without retraining, to the cost volume against the
WHOLE category memory bank (the paper's full-shot AnomalDF protocol); the
network's anomaly probability M is fused with the baseline (AnomalDF)
response for the same templates, lambda * M + (1 - lambda) * M_baseline with
lambda = 0.5, and the image score is the mean of the top-250 values of the
fused map. --test-templates random reproduces the paper's N = 3 random-template
inference instead.

Setup defaults (epochs, learning rate, foreground restriction) come from
method/ucf_setup.json (chosen on the visible MVTec-AD development slice; the
choice equals the paper's defaults: 40 epochs, lr 1e-3, foreground restriction on).

Usage:
  python -m method.train_ucf --dataset mvtec --output outputs/anomaldf_ucf --checkpoint checkpoints/anomaldf_ucf_seed0.pt --seed 0
  python -m method.train_ucf --dataset visa --output outputs/anomaldf_ucf --checkpoint checkpoints/anomaldf_ucf_visa_seed0.pt --seed 0
  python -m method.train_ucf --predict-only --checkpoint ... --output ... [--test-templates random]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.mvtec import IMAGE_SIZE, category_names, load_test_inputs, load_textures, load_train, metadata_name, predictions_name, write_predictions
from method.anomaly_synthesis import AnomalySynthesizer
from method.backbone import DinoFeatures, to_float_images
from method.foreground import foreground_masks
from method.fusion import LAMBDA, combine, components_name, fuse
from method.templates import category_index, nearest_templates, test_templates, train_templates
from method.ucf_model import ARCH_VERSION, MATCH_K, UCFNet, bank_cost_volume, build_cost_volume

GRID = 16  # 224 / 14 patch grid
GAMMA_RULE = 2  # 2: gamma = gamma0 - sigmoid(y_c) if the category is predicted correctly, else gamma0 (paper Sec. III-G)
RESUME_KEYS = ("dataset", "templates", "match_k", "no_guidance", "seed", "lr", "batch_size", "alpha", "gamma0", "fg_restrict", "train_fraction", "loss")
SETUP_FILE = Path(__file__).resolve().parent / "ucf_setup.json"
PAPER_SETUP = {"epochs": 40, "lr": 1e-3, "fg_restrict": True}
DEFAULT_WIDTHS = "64,96,128"  # compact 3D U-Net (about 4M parameters; the paper's filter adds 43.8M)


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
        match_channels=args.match_k,
        num_classes=len(category_names(args.dataset)),
        widths=tuple(int(v) for v in args.widths.split(",")),
        use_guidance=not args.no_guidance,
    ).to(device)


def config_of(args) -> dict:
    cfg = {"arch": ARCH_VERSION, "gamma_rule": GAMMA_RULE, **{k: getattr(args, k) for k in RESUME_KEYS}}
    if args.widths != DEFAULT_WIDTHS:  # recorded only when changed, so default-width checkpoints keep their config
        cfg["widths"] = args.widths
    return cfg


def adaptor_gamma(cls: torch.Tensor, cats: torch.Tensor, gamma0: float) -> torch.Tensor:
    """Class-aware adaptor: gamma0 - sigmoid(y_c) for correctly classified samples, gamma0 otherwise."""
    true_logit = cls.gather(1, cats[:, None]).squeeze(1)
    correct = cls.argmax(dim=1) == cats
    return torch.where(correct, gamma0 - torch.sigmoid(true_logit), torch.full_like(true_logit, gamma0))


def layer_banks(cache: torch.Tensor, index: dict[int, np.ndarray]) -> dict[int, list[torch.Tensor]]:
    """Whole-category memory banks: per category, per layer, the L2-normalised patch
    tokens of every training image of that category, (n*h*w, C) float16."""
    banks = {}
    for c, idx in index.items():
        t = cache[torch.as_tensor(idx, device=cache.device)]  # (n,L,C,h,w) half
        banks[c] = [
            F.normalize(t[:, layer].float().flatten(2).transpose(1, 2).reshape(-1, t.shape[2]), dim=1).half()
            for layer in range(t.shape[1])
        ]
    return banks


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
        fg_small, fg_stats = foreground_masks(cache[:, -1], train_cats_np, category_names(args.dataset))
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
            volume, initial = build_cost_volume(q, gather_templates(cache, t_idx), args.match_k)
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
            snap_root = args.snapshot_dir if args.snapshot_dir is not None else args.output.parent
            snap_out = snap_root / f"{args.output.name}_e{epoch}"
            if not (snap_out / predictions_name(args.dataset)).exists():
                predict(args, device, model, backbone, cache, train_cats_np, index, output=snap_out, epoch=epoch)
    return model


@torch.no_grad()
def predict(args, device, model, backbone, cache, train_cats_np, index, output: Path | None = None, epoch: int | None = None) -> None:
    output = args.output if output is None else output
    epoch = args.epochs if epoch is None else epoch
    model.eval()
    images, cats, ids = load_test_inputs(args.dataset)
    m = len(ids)
    maps = np.zeros((m, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    scores = np.zeros(m, dtype=np.float32)
    comp_logits = comp_cost = None
    full = args.test_templates == "all"
    banks = layer_banks(cache, index) if full else None
    train_desc = F.normalize(cache[:, -1].float().mean(dim=(2, 3)), dim=1) if args.test_templates == "nearest" else None
    draws = 1 if args.test_templates in ("nearest", "all") else args.test_draws
    bs = 16
    for s in range(0, m, bs):
        rows = range(s, min(s + bs, m))
        x = to_float_images(torch.from_numpy(images[s:s + bs]).to(device))
        q = backbone(x)
        logits_draws, cost_draws = [], []
        for draw in range(draws):
            if full:
                b = x.shape[0]
                volume = initial = None
                for c in np.unique(cats[s:s + bs]):
                    sel = torch.as_tensor(np.flatnonzero(cats[s:s + bs] == c), device=device)
                    v, i0 = bank_cost_volume([f[sel] for f in q], banks[int(c)], args.match_k)
                    if volume is None:
                        volume = v.new_zeros((b, *v.shape[1:]))
                        initial = i0.new_zeros((b, *i0.shape[1:]))
                    volume[sel], initial[sel] = v, i0
            else:
                if args.test_templates == "nearest":
                    desc = F.normalize(q[-1].mean(dim=(2, 3)), dim=1)
                    t_idx = np.stack([nearest_templates(index, cats[i], args.templates, desc[j], train_desc) for j, i in enumerate(rows)])
                else:
                    t_idx = np.stack([test_templates(index, cats[i], args.templates, args.template_seed, draw, ids[i]) for i in rows])
                volume, initial = build_cost_volume(q, gather_templates(cache, t_idx), args.match_k)
            logits, _ = model(volume, initial, q)  # (B,2,16,16)
            logits_draws.append(logits.float())
            cost_draws.append(initial[:, -1].float())  # AnomalDF cost for the same templates: last layer, min over template tokens
        prob, base = combine(logits_draws, cost_draws)
        maps[s:s + bs], scores[s:s + bs] = fuse(prob, base, args.lam)
        lg = torch.stack(logits_draws, dim=1).cpu().numpy()
        cs = torch.stack(cost_draws, dim=1).cpu().numpy()
        if comp_logits is None:
            comp_logits = np.zeros((m, *lg.shape[1:]), np.float32)
            comp_cost = np.zeros((m, *cs.shape[1:]), np.float32)
        comp_logits[s:s + bs] = lg
        comp_cost[s:s + bs] = cs
    path = write_predictions(output, ids, scores, maps, args.dataset)
    np.savez_compressed(output / components_name(args.dataset), ids=np.asarray(ids, np.int64), logits16=comp_logits, cost16=comp_cost)
    meta_path = output / metadata_name(args.dataset)
    meta_all = {
        "method": "AnomalDF+UCF (cost-volume filtering, fused with the AnomalDF response)",
        "dataset": args.dataset,
        "backbone": "DINOv2 ViT-S/14, blocks 3/6/9/12, 224px",
        "train_templates": args.templates,
        "matching_dimension": f"{args.match_k} lowest costs per input token",
        "test_templates": "whole category memory bank (full-shot)" if full else args.test_templates,
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
        "anomaly_maps": "lambda * UCF anomaly probability + (1 - lambda) * AnomalDF map for the same templates",
        "image_score": "mean of the top-250 values of the fused map",
        "arch_version": ARCH_VERSION,
        "widths": args.widths,
        "trainable_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
    }
    meta_path.write_text(json.dumps(meta_all, indent=2))
    print(f"wrote {path}", flush=True)


def main() -> None:
    setup = load_setup()
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("mvtec", "visa"), default="mvtec")
    p.add_argument("--device", default="cuda", help="torch device for this run (e.g. cuda:1)")
    p.add_argument("--epochs", type=int, default=setup["epochs"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=setup["lr"])
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma0", type=float, default=3.0)
    p.add_argument("--templates", type=int, default=3, help="random same-category templates per training input (paper: N = 3)")
    p.add_argument("--match-k", type=int, default=MATCH_K, help="matching dimension: lowest costs kept per input token")
    p.add_argument("--widths", default=DEFAULT_WIDTHS, help="channel widths of the three 3D U-Net levels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-guidance", action="store_true", help="ablation: drop both guidance streams")
    p.add_argument("--loss", choices=("full", "focal"), default="full", help="ablation: 'focal' trains with the focal loss alone")
    p.add_argument("--lam", type=float, default=LAMBDA, help="inference fusion weight of the UCF map (paper: 0.5)")
    p.add_argument("--output", type=Path, default=Path("outputs/anomaldf_ucf"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/anomaldf_ucf_seed0.pt"))
    p.add_argument("--test-draws", type=int, default=1, help="random template draws averaged at test time (--test-templates random)")
    p.add_argument("--test-templates", choices=("all", "random", "nearest"), default="all",
                   help="all: whole category memory bank (paper full-shot, default); random / nearest: N = --templates images")
    p.add_argument("--template-seed", type=int, default=2026)
    p.add_argument("--fg-restrict", dest="fg_restrict", action="store_true", default=setup["fg_restrict"])
    p.add_argument("--no-fg-restrict", dest="fg_restrict", action="store_false")
    p.add_argument("--train-fraction", type=float, default=1.0, help="fraction of normal training images used as queries")
    p.add_argument("--snapshot-epochs", default="", help="comma list: also save + predict at these epochs (<snapshot-dir>/<output name>_e<k>)")
    p.add_argument("--snapshot-dir", type=Path, default=None, help="where snapshot predictions go (default: next to --output)")
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)  # so library code that says "cuda" lands on this run's GPU
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("CUDA is not available; pass --allow-cpu to run (very slowly) on CPU")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    train_images_np, train_cats_np = load_train(args.dataset)
    train_images = torch.from_numpy(train_images_np).to(device)
    index = category_index(train_cats_np)
    backbone = DinoFeatures().to(device)
    cache = feature_cache(backbone, train_images)
    print(f"dataset={args.dataset} train_images={len(train_cats_np)} categories={len(index)} device={device}", flush=True)

    if args.predict_only:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if int(state["epoch"]) < args.epochs:
            raise SystemExit(f"{args.checkpoint} holds epoch {state['epoch']} < {args.epochs}")
        cfg = state.get("config", {})
        for key in ("arch", "dataset", "match_k", "no_guidance", "widths"):
            want = ARCH_VERSION if key == "arch" else getattr(args, key)
            have = cfg.get(key, DEFAULT_WIDTHS) if key == "widths" else cfg.get(key)
            if have != want:
                raise SystemExit(f"{args.checkpoint}: config {key}={cfg.get(key)!r} does not match {want!r}")
        model = build_model(args, device)
        model.load_state_dict(state["model"])
    else:
        model = train(args, device, backbone, cache, train_images, train_cats_np, index)
    predict(args, device, model, backbone, cache, train_cats_np, index)


if __name__ == "__main__":
    main()
