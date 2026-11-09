"""Unified Cost Filtering (UCF) for unimodal RGB anomaly detection.

Pipeline (paper Sec. III-D..F):
  1. anomaly cost volume: global cosine matching of every input patch token
     against every patch token of N normal templates, per feature layer
     (Eq. 2, 6):  C(j, n, l, i) = 1 - cos(f_I^{i,l}, f_T^{n,j,l})
     -> volume (B, D*N, L, H', W') with D = H'W' matching positions as
     channels and the L feature layers as 3D depth;
  2. initial multi-layer anomaly map M = global min over the D*N axis;
  3. 3D U-Net filtering with residual channel-spatial attention (RCSA, Eq. 7)
     guided by two streams: projected input features h(f_I^l) and the
     projected initial map h(M);
  4. class-aware adaptor: pooled deepest features -> category logits that
     modulate the focal-loss focusing parameter;
  5. the last decoder layer is a 3D conv that reduces depth L -> 1 and
     returns the filtered cost volume in the matching dimension (D*N
     channels), as a residual correction of the input (last-layer) costs;
     global min pooling along the matching dimension, a conv and a softmax
     give the normal/anomaly map: M = softmax(conv(min(x))).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

ARCH_VERSION = 2  # 2: filtered volume keeps the D*N matching channels (residual on the input costs)


def build_cost_volume(query_feats: list[torch.Tensor], template_feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """query_feats: L x (B,C,H,W); template_feats: L x (B,N,C,H,W).

    Returns volume (B, N*H*W, L, H, W) and initial map (B, L, H, W).
    """
    layers = []
    for q, t in zip(query_feats, template_feats):
        b, _, h, w = q.shape
        n = t.shape[1]
        qn = F.normalize(q.flatten(2), dim=1)  # (B,C,HW)
        tn = F.normalize(t.flatten(3), dim=2)  # (B,N,C,D)
        sim = torch.einsum("bci,bncj->bnji", qn, tn)  # (B,N,D,HW)
        layers.append((1.0 - sim).reshape(b, n * tn.shape[-1], h, w))
    volume = torch.stack(layers, dim=2)
    initial = volume.amin(dim=1)
    return volume, initial


def _norm(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if c % 8 == 0 else 1, c)


class ConvBlock3d(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel=3, stride=1):
        k = (kernel,) * 3 if isinstance(kernel, int) else tuple(kernel)
        pad = tuple(v // 2 for v in k)
        super().__init__(nn.Conv3d(cin, cout, k, stride=stride, padding=pad, bias=False), _norm(cout), nn.ReLU(inplace=True))


class RCSA(nn.Module):
    """Residual channel-spatial attention with dual-stream guidance (Eq. 7)."""

    def __init__(self, channels: int, guide_channels: int, out_channels: int):
        super().__init__()
        cin = channels + 2 * guide_channels
        self.ca_max = nn.Conv3d(cin, cin, 1)
        self.ca_avg = nn.Conv3d(cin, cin, 1)
        self.sa = nn.Conv3d(2, 1, kernel_size=(3, 7, 7), padding=(1, 3, 3))
        self.fuse = ConvBlock3d(cin, out_channels)

    def forward(self, x: torch.Tensor, g_map: torch.Tensor | None, g_feat: torch.Tensor | None) -> torch.Tensor:
        if g_map is not None:
            x = torch.cat([x, g_map, g_feat], dim=1)
        ca = torch.sigmoid(self.ca_max(F.adaptive_max_pool3d(x, 1)) + self.ca_avg(F.adaptive_avg_pool3d(x, 1)))
        x = ca * x + x
        sa = torch.sigmoid(self.sa(torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)))
        x = sa * x + x
        return self.fuse(x)


class UCFNet(nn.Module):
    def __init__(
        self,
        match_channels: int,
        num_classes: int,
        feat_dim: int = 384,
        depth: int = 4,
        widths: tuple[int, int, int] = (64, 96, 128),
        guide_channels: int = 16,
        use_guidance: bool = True,
    ):
        super().__init__()
        w0, w1, w2 = widths
        self.use_guidance = use_guidance
        g = guide_channels if use_guidance else 0
        if use_guidance:
            self.h_map = ConvBlock3d(1, guide_channels)  # matching guidance (MG) projector
            self.h_feat = ConvBlock3d(feat_dim, guide_channels, kernel=1)  # spatial guidance (SG) projector
        self.stem = ConvBlock3d(match_channels, w0)
        self.att0 = RCSA(w0, g, w0)
        self.down1 = ConvBlock3d(w0, w1, stride=(1, 2, 2))
        self.att1 = RCSA(w1, g, w1)
        self.down2 = ConvBlock3d(w1, w2, stride=(1, 2, 2))
        self.att2 = RCSA(w2, g, w2)
        self.classifier = nn.Linear(w2, num_classes)  # class-aware adaptor
        self.up1 = ConvBlock3d(w2 + w1, w1)
        self.up0 = ConvBlock3d(w1 + w0, w0)
        # L -> 1, back to the D*N matching channels; zero init = start from the raw costs
        self.refine = nn.Conv3d(w0, match_channels, kernel_size=(depth, 1, 1))
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)
        self.head = nn.Conv2d(1, 2, kernel_size=3, padding=1)

    def _guides(self, gm, gf, size):
        if gm is None:
            return None, None
        return F.adaptive_avg_pool3d(gm, size), F.adaptive_avg_pool3d(gf, size)

    def forward(self, volume: torch.Tensor, initial: torch.Tensor, image_feats: list[torch.Tensor]):
        gm = gf = None
        if self.use_guidance:
            gm = self.h_map(initial[:, None])
            gf = self.h_feat(torch.stack(image_feats, dim=2))
        x0 = self.stem(volume)
        x0 = self.att0(x0, *self._guides(gm, gf, x0.shape[-3:]))
        x1 = self.down1(x0)
        x1 = self.att1(x1, *self._guides(gm, gf, x1.shape[-3:]))
        x2 = self.down2(x1)
        x2 = self.att2(x2, *self._guides(gm, gf, x2.shape[-3:]))
        cls = self.classifier(x2.mean(dim=(2, 3, 4)))
        u1 = F.interpolate(x2, size=x1.shape[-3:], mode="trilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, x1], dim=1))
        u0 = F.interpolate(u1, size=x0.shape[-3:], mode="trilinear", align_corners=False)
        u0 = self.up0(torch.cat([u0, x0], dim=1))
        filtered = volume[:, :, -1] + self.refine(u0)[:, :, 0]  # (B, D*N, H, W) filtered cost volume
        pooled = filtered.amin(dim=1, keepdim=True)  # global min pooling along the matching dimension
        logits = self.head(pooled)
        return logits, cls


def topk_mean(maps: torch.Tensor, k: int) -> torch.Tensor:
    flat = maps.flatten(1)
    return flat.topk(min(k, flat.shape[1]), dim=1).values.mean(dim=1)
