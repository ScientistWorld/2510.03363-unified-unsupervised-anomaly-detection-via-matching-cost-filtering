"""Frozen DINOv2 ViT-S/14 feature extractor (the AnomalDF feature source).

The paper builds AnomalDF+UCF cost volumes from the 3rd, 6th, 9th and 12th
blocks of a pretrained DINO(v2) encoder. At 224x224 input ViT-S/14 yields a
16x16 patch grid with 384-d tokens per block.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DINO_DIR = ROOT / "data" / "downloads" / "dinov2"
UCF_LAYERS = (2, 5, 8, 11)  # 0-based block indices = 3rd, 6th, 9th, 12th blocks
LAST_LAYER = (11,)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _import_vit_small():
    repo = DINO_DIR / "repo"
    if not (repo / "dinov2" / "models" / "vision_transformer.py").exists():
        raise FileNotFoundError(f"{repo} missing; run scripts/download.sh")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if not hasattr(torch.nn.utils, "weight_norm"):  # dino_head import guard for future torch
        stub = types.ModuleType("dinov2.layers.dino_head")
        stub.DINOHead = nn.Identity
        sys.modules.setdefault("dinov2.layers.dino_head", stub)
    from dinov2.models.vision_transformer import vit_small  # noqa: E402

    return vit_small


class DinoFeatures(nn.Module):
    """images in [0,1], (B,3,224,224) -> list of (B,384,16,16) patch-token maps."""

    def __init__(self, layers: tuple[int, ...] = UCF_LAYERS):
        super().__init__()
        vit_small = _import_vit_small()
        model = vit_small(
            patch_size=14,
            img_size=518,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
        )
        state = torch.load(DINO_DIR / "dinov2_vits14_pretrain.pth", map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.layers = list(layers)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):  # always frozen / eval mode
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        x = (images - self.mean) / self.std
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            feats = self.model.get_intermediate_layers(x, n=self.layers, reshape=True, norm=True)
        return [f.float() for f in feats]


def to_float_images(batch_uint8: torch.Tensor) -> torch.Tensor:
    """(B,H,W,3) uint8 -> (B,3,H,W) float in [0,1]."""
    return batch_uint8.permute(0, 3, 1, 2).float().div_(255.0)


def gaussian_blur(maps: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Separable Gaussian smoothing of (B,H,W) maps (AnomalDF post-processing)."""
    radius = int(round(4 * sigma))
    x = torch.arange(-radius, radius + 1, device=maps.device, dtype=maps.dtype)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    k = k / k.sum()
    m = maps[:, None]
    m = F.conv2d(F.pad(m, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    m = F.conv2d(F.pad(m, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1))
    return m[:, 0]
