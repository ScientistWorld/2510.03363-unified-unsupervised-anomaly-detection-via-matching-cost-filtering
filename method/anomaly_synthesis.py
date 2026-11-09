"""Perlin-noise guided anomaly synthesis (GLAD / DRAEM protocol), on GPU.

For half of the training images a Perlin-noise mask selects a region that is
replaced by either an augmented DTD texture or a grid-shuffled copy of the
image itself (structural perturbation), blended with factor
beta ~ U(0, 0.8). The binary mask supervises the filtering network.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

PERLIN_SIZE = 256
PERLIN_MAX_SCALE = 6  # resolutions 2^0 .. 2^5, as in DRAEM
PERLIN_THRESHOLD = 0.5
ANOMALY_PROB = 0.5
BETA_MAX = 0.8


def _fade(t: torch.Tensor) -> torch.Tensor:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def rand_perlin_2d(res: tuple[int, int], gen: torch.Generator, device: torch.device, size: int = PERLIN_SIZE) -> torch.Tensor:
    delta = (res[0] / size, res[1] / size)
    d = (size // res[0], size // res[1])
    gy = torch.arange(0, res[0], delta[0], device=device)[:size]
    gx = torch.arange(0, res[1], delta[1], device=device)[:size]
    grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij"), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1, generator=gen, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def tile(s1, s2):
        g = gradients[s1[0]:s1[1], s2[0]:s2[1]]
        return g.repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)

    def dot(grad, shift):
        shifted = torch.stack((grid[..., 0] + shift[0], grid[..., 1] + shift[1]), dim=-1)
        return (shifted * grad).sum(dim=-1)

    n00 = dot(tile((0, -1), (0, -1)), (0, 0))
    n10 = dot(tile((1, None), (0, -1)), (-1, 0))
    n01 = dot(tile((0, -1), (1, None)), (0, -1))
    n11 = dot(tile((1, None), (1, None)), (-1, -1))
    t = _fade(grid)
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def _rotate(x: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate (C,H,W) about its centre, zero fill."""
    theta = math.radians(degrees)
    mat = torch.tensor(
        [[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0]],
        device=x.device,
        dtype=x.dtype,
    )[None]
    grid = F.affine_grid(mat, (1, *x.shape), align_corners=False)
    return F.grid_sample(x[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]


def _augment_texture(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Three random photometric/geometric ops (DRAEM-style augmenter pool)."""
    ops = rng.choice(8, size=3, replace=False)
    for op in ops:
        if op == 0:  # gamma contrast
            x = x.clamp_min(1e-4) ** float(rng.uniform(0.5, 2.0))
        elif op == 1:  # multiply-and-add brightness
            x = x * float(rng.uniform(0.8, 1.2)) + float(rng.uniform(-30, 30)) / 255.0
        elif op == 2:  # saturation (hue/saturation stand-in) + channel permutation
            gray = x.mean(dim=0, keepdim=True)
            x = gray + float(rng.uniform(0.5, 1.5)) * (x - gray)
            x = x[torch.as_tensor(rng.permutation(3), device=x.device)]
        elif op == 3:  # solarize
            thr = float(rng.uniform(32, 128)) / 255.0
            x = torch.where(x >= thr, 1.0 - x, x)
        elif op == 4:  # posterize to 4 bits
            x = torch.floor(x.clamp(0, 1) * 15.0) / 15.0
        elif op == 5:  # invert
            x = 1.0 - x
        elif op == 6:  # autocontrast
            lo = x.amin(dim=(1, 2), keepdim=True)
            hi = x.amax(dim=(1, 2), keepdim=True)
            x = (x - lo) / (hi - lo).clamp_min(1e-4)
        else:  # rotation
            x = _rotate(x, float(rng.uniform(-45, 45)))
        x = x.clamp(0.0, 1.0)
    return x


def _grid_shuffle(img: torch.Tensor, rng: np.random.Generator, cells: int = 8) -> torch.Tensor:
    c, h, w = img.shape
    ch, cw = h // cells, w // cells
    blocks = img[:, : ch * cells, : cw * cells].reshape(c, cells, ch, cells, cw).permute(1, 3, 0, 2, 4).reshape(cells * cells, c, ch, cw)
    blocks = blocks[torch.as_tensor(rng.permutation(cells * cells), device=img.device)]
    out = blocks.reshape(cells, cells, c, ch, cw).permute(2, 0, 3, 1, 4).reshape(c, ch * cells, cw * cells)
    result = img.clone()
    result[:, : ch * cells, : cw * cells] = out
    return result


class AnomalySynthesizer:
    def __init__(self, textures_uint8: torch.Tensor, seed: int):
        """textures_uint8: (K,H,W,3) uint8 tensor on the target device."""
        self.textures = textures_uint8
        self.device = textures_uint8.device
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "gen": self.gen.get_state()}

    def load_state(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]
        self.gen.set_state(state["gen"])

    def _mask(self, size: int) -> torch.Tensor:
        res = (2 ** int(self.rng.integers(0, PERLIN_MAX_SCALE)), 2 ** int(self.rng.integers(0, PERLIN_MAX_SCALE)))
        noise = rand_perlin_2d(res, self.gen, self.device)
        noise = _rotate(noise[None], float(self.rng.uniform(-90, 90)))
        noise = F.interpolate(noise[None], size=(size, size), mode="bilinear", align_corners=False)[0, 0]
        return (noise > PERLIN_THRESHOLD).float()

    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """images (B,3,H,W) in [0,1] -> (augmented images, masks (B,H,W) in {0,1})."""
        out = images.clone()
        b, _, h, w = images.shape
        masks = torch.zeros(b, h, w, device=images.device)
        for i in range(b):
            if self.rng.random() >= ANOMALY_PROB:
                continue
            mask = self._mask(h)
            if mask.sum() < 1:
                continue
            if self.rng.random() < 0.5:
                k = int(self.rng.integers(0, self.textures.shape[0]))
                src = self.textures[k].permute(2, 0, 1).float() / 255.0
                if src.shape[-2:] != (h, w):
                    src = F.interpolate(src[None], size=(h, w), mode="bilinear", align_corners=False)[0]
                src = _augment_texture(src, self.rng)
            else:
                src = _grid_shuffle(images[i], self.rng)
            beta = float(self.rng.uniform(0.0, BETA_MAX))
            m = mask[None]
            out[i] = images[i] * (1 - m) + (1 - beta) * src * m + beta * images[i] * m
            masks[i] = mask
        return out.clamp(0.0, 1.0), masks
