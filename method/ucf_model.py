from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FixedFeaturePyramid(nn.Module):
    """Small frozen hierarchical feature extractor for RGB template matching."""

    def __init__(self):
        super().__init__()
        kernels = []
        base = torch.tensor(
            [
                [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            ],
            dtype=torch.float32,
        )
        for c in range(3):
            k = torch.zeros(4, 3, 3, 3)
            k[:, c] = base
            kernels.append(k)
        weight = torch.cat(kernels, dim=0)
        weight[-3:] /= weight[-3:].sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for size in (32, 16, 8):
            z = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            f = F.conv2d(z, self.weight, padding=1)
            feats.append(torch.cat([z, f], dim=1))
        return feats


def _cosine_cost(query: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
    b, c, h, w = query.shape
    _, k, _, ht, wt = templates.shape
    q = F.normalize(query, dim=1)
    t = F.normalize(templates.reshape(b, k, c, ht * wt), dim=2)
    sim = torch.einsum("bchw,bkcm->bkmhw", q, t).reshape(b, k * ht * wt, h, w)
    return 1.0 - sim


class CostVolumeBuilder(nn.Module):
    def __init__(self, base_size: int = 16):
        super().__init__()
        self.features = FixedFeaturePyramid()
        self.base_size = base_size

    @torch.no_grad()
    def forward(self, image: torch.Tensor, templates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        image_feats = self.features(image)
        b, k, c, h, w = templates.shape
        template_feats_flat = self.features(templates.reshape(b * k, c, h, w))
        cost_layers = []
        for q, tf in zip(image_feats, template_feats_flat):
            _, cf, ht, wt = tf.shape
            t = tf.reshape(b, k, cf, ht, wt)
            cost = _cosine_cost(q, t)
            cost = F.interpolate(cost, size=(self.base_size, self.base_size), mode="bilinear", align_corners=False)
            cost_layers.append(cost)
        volume = torch.cat(cost_layers, dim=1)
        initial = volume.min(dim=1, keepdim=True).values
        return volume, initial, image_feats


class RCSA3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(4, channels // 4)
        self.channel = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1),
        )
        self.spatial = nn.Conv3d(2, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool3d(x, 1)
        mx = F.adaptive_max_pool3d(x, 1)
        x = x + x * torch.sigmoid(self.channel(avg) + self.channel(mx))
        mean = x.mean(dim=1, keepdim=True)
        maxv = x.max(dim=1, keepdim=True).values
        x = x + x * torch.sigmoid(self.spatial(torch.cat([mean, maxv], dim=1)))
        return x


class UCFNet(nn.Module):
    """Compact UCF: 3D cost-volume filtering with dual image/map guidance."""

    def __init__(self, categories: int = 3, guide_channels: int = 16):
        super().__init__()
        self.guide = nn.Sequential(
            nn.Conv2d(4, guide_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(guide_channels, guide_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc1 = nn.Sequential(nn.Conv3d(1 + guide_channels, 16, 3, padding=1), nn.ReLU(inplace=True), RCSA3D(16))
        self.enc2 = nn.Sequential(nn.Conv3d(16, 32, 3, stride=(2, 1, 1), padding=1), nn.ReLU(inplace=True), RCSA3D(32))
        self.enc3 = nn.Sequential(nn.Conv3d(32, 48, 3, stride=(2, 1, 1), padding=1), nn.ReLU(inplace=True), RCSA3D(48))
        self.dec2 = nn.Sequential(nn.Conv3d(48 + 32, 32, 3, padding=1), nn.ReLU(inplace=True), RCSA3D(32))
        self.dec1 = nn.Sequential(nn.Conv3d(32 + 16, 24, 3, padding=1), nn.ReLU(inplace=True), RCSA3D(24))
        self.seg = nn.Conv2d(24, 2, 3, padding=1)
        self.classifier = nn.Linear(48, categories)

    def forward(self, volume: torch.Tensor, image: torch.Tensor, initial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, d, h, w = volume.shape
        img = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)
        guide2d = self.guide(torch.cat([img, initial], dim=1))
        guide3d = guide2d[:, :, None].expand(-1, -1, d, -1, -1)
        x0 = torch.cat([volume[:, None], guide3d], dim=1)
        e1 = self.enc1(x0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        cls = self.classifier(e3.mean(dim=(2, 3, 4)))
        u2 = F.interpolate(e3, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        u2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(u2, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        u1 = self.dec1(torch.cat([u1, e1], dim=1))
        filtered = u1.min(dim=2).values
        logits = self.seg(filtered)
        logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return logits, cls


def baseline_map(volume: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
    score = volume.min(dim=1, keepdim=True).values
    score = F.interpolate(score, size=out_size, mode="bilinear", align_corners=False)
    lo = score.amin(dim=(2, 3), keepdim=True)
    hi = score.amax(dim=(2, 3), keepdim=True)
    return ((score - lo) / (hi - lo + 1e-6))[:, 0]
