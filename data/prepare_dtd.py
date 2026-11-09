"""Pack a deterministic subset of the Describable Textures Dataset (DTD) at 224px.

DTD is only an anomaly-synthesis source for training (Perlin-masked texture
pasting); it never appears in evaluation. Writes data/downloads/dtd/dtd224.npy.

Run inside the container:  python -m data.prepare_dtd
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "downloads" / "dtd" / "dtd-r1.0.1.tar.gz"
OUT = ROOT / "data" / "downloads" / "dtd" / "dtd224.npy"
SIZE = 224
EVERY = 4  # keep every 4th image of the sorted list (~1410 of 5640)


def main() -> None:
    if OUT.exists():
        print(f"{OUT} exists; nothing to do")
        return
    if not ARCHIVE.exists():
        raise SystemExit(f"missing {ARCHIVE}; run scripts/download.sh")
    with tarfile.open(ARCHIVE, mode="r:gz") as tf:
        members = sorted(
            (m for m in tf.getmembers() if m.isfile() and "/images/" in m.name and m.name.lower().endswith(".jpg")),
            key=lambda m: m.name,
        )
        chosen = members[::EVERY]
        arrays: list[np.ndarray | None] = [None] * len(chosen)
        # read in archive order: seeking backwards in a .tar.gz restarts decompression
        for i in sorted(range(len(chosen)), key=lambda j: chosen[j].offset_data):
            fh = tf.extractfile(chosen[i])
            if fh is None:
                continue
            img = Image.open(io.BytesIO(fh.read())).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
            arrays[i] = np.asarray(img, dtype=np.uint8)
        arrays = [a for a in arrays if a is not None]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.stem + ".tmp.npy")
    np.save(tmp, np.stack(arrays))
    tmp.rename(OUT)
    print(f"wrote {len(arrays)} textures to {OUT}")


if __name__ == "__main__":
    main()
