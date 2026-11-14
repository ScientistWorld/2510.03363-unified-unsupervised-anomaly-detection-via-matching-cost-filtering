"""Inference fusion of UCF with the baseline response (paper Sec. III-G).

    M_final = lam * M_ucf + (1 - lam) * M_baseline,      lam = 0.5 (paper default)

M_ucf      softmax anomaly probability of the cost-filtering network, its 16x16
           logits bilinearly upsampled to the 224px input before the softmax.
M_baseline the AnomalDF response for the same templates (the whole category
           memory bank at full-shot inference): the minimum cosine matching
           cost of the last DINOv2 block (the last-layer slice of the initial
           anomaly map), bilinearly upsampled and Gaussian-smoothed (sigma 4)
           exactly as the AnomalDF baseline does.
Image score: mean of the top-250 values of the final map (paper Sec. III-F).

predict() in method/train_ucf.py stores the per-draw components (16x16
network logits and 16x16 baseline costs) next to its predictions, so other
lam values and multi-run ensembles (probabilities averaged over runs) are
rebuilt with the same code, without re-running the network:

    python -m method.fusion --dataset mvtec|visa --runs DIR [DIR ...] --lam 0.5 --output OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.mvtec import IMAGE_SIZE, metadata_name, write_predictions
from method.backbone import gaussian_blur
from method.ucf_model import topk_mean

LAMBDA = 0.5  # paper default (Sec. IV-E6: gains peak for lambda in 0.4-0.6; 0.5 adopted)
TOPK = 250
BASELINE_SIGMA = 4.0
COMPONENTS = "components.npz"


def components_name(dataset: str = "mvtec") -> str:
    return COMPONENTS if dataset == "mvtec" else f"components_{dataset}.npz"


def ucf_probability(logits16: torch.Tensor) -> torch.Tensor:
    """(B,2,h,w) network logits -> (B,224,224) float64 anomaly probability."""
    up = F.interpolate(logits16.float(), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    return up.double().softmax(dim=1)[:, 1]


def baseline_map(cost16: torch.Tensor) -> torch.Tensor:
    """(B,h,w) last-layer min matching cost -> (B,224,224) float64 AnomalDF map."""
    up = F.interpolate(cost16.float()[:, None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[:, 0]
    return gaussian_blur(up, BASELINE_SIGMA).double()


def combine(logits_draws: list[torch.Tensor], cost_draws: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Average the UCF probability over the given (run, draw) logits and the
    baseline map over the given draw costs -> (prob, base), each (B,224,224)."""
    prob = torch.stack([ucf_probability(lg) for lg in logits_draws]).mean(dim=0)
    base = torch.stack([baseline_map(c) for c in cost_draws]).mean(dim=0)
    return prob, base


def fuse(prob: torch.Tensor, base: torch.Tensor, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (maps float32 (B,224,224), image scores float32 (B,))."""
    final = lam * prob + (1.0 - lam) * base
    scores = topk_mean(final, TOPK).float().cpu().numpy()
    if lam >= 1.0:  # UCF alone: store log-probabilities (same ranking, no float32 saturation ties at p = 1)
        final = torch.log(prob.clamp_min(1e-300))
    return final.float().cpu().numpy(), scores


def load_components(run: Path, dataset: str = "mvtec") -> dict[str, np.ndarray]:
    with np.load(run / components_name(dataset)) as d:
        return {k: d[k] for k in d.files}


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="prediction dirs holding components.npz")
    p.add_argument("--lam", type=float, default=LAMBDA)
    p.add_argument("--dataset", choices=("mvtec", "visa"), default="mvtec")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 0.0 <= args.lam <= 1.0:
        raise SystemExit("lam must lie in [0, 1]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comps = [load_components(r, args.dataset) for r in args.runs]
    ids = comps[0]["ids"]
    for r, c in zip(args.runs, comps):
        if not np.array_equal(c["ids"], ids):
            raise SystemExit(f"{r}: ids differ from {args.runs[0]}")
    same_base = all(np.array_equal(c["cost16"], comps[0]["cost16"]) for c in comps)
    m = ids.size
    maps = np.zeros((m, IMAGE_SIZE, IMAGE_SIZE), np.float32)
    scores = np.zeros(m, np.float32)
    base_runs = comps[:1] if same_base else comps  # runs sharing templates share the baseline map
    bs = 32
    for s in range(0, m, bs):
        logits_draws, cost_draws = [], []
        for c in comps:
            logits = torch.from_numpy(c["logits16"][s:s + bs]).to(device)  # (b, draws, 2, h, w)
            logits_draws += [logits[:, d] for d in range(logits.shape[1])]
        for c in base_runs:
            cost = torch.from_numpy(c["cost16"][s:s + bs]).to(device)  # (b, draws, h, w)
            cost_draws += [cost[:, d] for d in range(cost.shape[1])]
        prob, base = combine(logits_draws, cost_draws)
        maps[s:s + bs], scores[s:s + bs] = fuse(prob, base, args.lam)
    path = write_predictions(args.output, ids, scores, maps, args.dataset)
    meta_path = args.output / metadata_name(args.dataset)
    meta_all = {
        "method": "AnomalDF+UCF fused output rebuilt from stored components",
        "runs": [str(r) for r in args.runs],
        "lambda": args.lam,
        "ucf_probability": f"mean over {len(comps)} run(s) and their template draws",
        "baseline_shared_by_runs": bool(same_base),
        "image_score": f"mean of top-{TOPK} values of the fused map",
    }
    meta_path.write_text(json.dumps(meta_all, indent=2))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
