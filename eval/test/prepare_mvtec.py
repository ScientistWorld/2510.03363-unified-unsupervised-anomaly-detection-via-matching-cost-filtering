"""Build the gym's 224px MVTec-AD arrays from the official archive.

This is the ONLY code that can map anonymised test inputs back to MVTec-AD
file names (and therefore to labels and masks). It lives under eval/test/ on
purpose: the raw archive (eval/test/downloads/mvtec_anomaly_detection.tar.xz)
and the hidden answers never leave eval/test/.

Streams the archive once (no extraction to disk) and writes

  data/downloads/mvtec224/train.npz        normal training images (method-visible)
  data/downloads/mvtec224/test_inputs.npz  every test image, shuffled under random
                                           ids, with its category only (method-visible)
  eval/train/data/dev_answers.npz          labels + masks of the visible development half
  eval/test/downloads/test_answers.npz     labels + masks of the hidden half

The two halves split every category's normal test images 50/50 and assign
whole defect TYPES to one half (categories with a single defect type split
their defective images 50/50), so the hidden half contains defect types that
never appear in the development half.

Run inside the container:  python -m eval.test.prepare_mvtec
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "eval" / "test" / "downloads" / "mvtec_anomaly_detection.tar.xz"
TRAIN_OUT = ROOT / "data" / "downloads" / "mvtec224" / "train.npz"
INPUTS_OUT = ROOT / "data" / "downloads" / "mvtec224" / "test_inputs.npz"
DEV_OUT = ROOT / "eval" / "train" / "data" / "dev_answers.npz"
HIDDEN_OUT = ROOT / "eval" / "test" / "downloads" / "test_answers.npz"
MANIFEST = ROOT / "eval" / "test" / "downloads" / "prepare_manifest.json"

FORMAT_VERSION = 1
SIZE = 224
SECRET = "costfilter-gym/hidden-split/v1"
CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)


def keyed(*parts: object) -> int:
    raw = "|".join([SECRET, *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def load_image(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def load_mask(data: bytes) -> np.ndarray:
    m = Image.open(io.BytesIO(data)).convert("L")
    m = m.resize((SIZE, SIZE), Image.BOX)  # area average, then majority threshold
    return np.asarray(m, dtype=np.uint8) >= 128


def parse_member(name: str) -> tuple[str, list[str]] | None:
    parts = [p for p in name.split("/") if p not in ("", ".")]
    for i, part in enumerate(parts):
        if part in CATEGORIES:
            return part, parts[i + 1:]
    return None


def read_archive(path: Path):
    train: dict[str, dict[str, np.ndarray]] = {c: {} for c in CATEGORIES}
    test: dict[str, dict[tuple[str, str], np.ndarray]] = {c: {} for c in CATEGORIES}
    gt: dict[str, dict[tuple[str, str], np.ndarray]] = {c: {} for c in CATEGORIES}
    with tarfile.open(path, mode="r|xz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.lower().endswith(".png"):
                continue
            parsed = parse_member(member.name)
            if parsed is None:
                continue
            cat, rest = parsed
            fh = tf.extractfile(member)
            if fh is None:
                continue
            data = fh.read()
            if len(rest) == 3 and rest[0] == "train" and rest[1] == "good":
                train[cat][rest[2]] = load_image(data)
            elif len(rest) == 3 and rest[0] == "test":
                test[cat][(rest[1], rest[2])] = load_image(data)
            elif len(rest) == 3 and rest[0] == "ground_truth":
                stem = rest[2].replace("_mask.png", ".png")
                gt[cat][(rest[1], stem)] = load_mask(data)
    return train, test, gt


def split_category(cat: str, keys: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Return {(defect_type, file): 'dev'|'hidden'} for one category."""
    assign: dict[tuple[str, str], str] = {}
    goods = sorted([k for k in keys if k[0] == "good"], key=lambda k: keyed("good", cat, k[1]))
    half = len(goods) // 2
    for i, k in enumerate(goods):
        assign[k] = "dev" if i < half else "hidden"
    defects = [k for k in keys if k[0] != "good"]
    types = sorted({k[0] for k in defects})
    if len(types) == 1:
        ordered = sorted(defects, key=lambda k: keyed("single", cat, k[0], k[1]))
        half = len(ordered) // 2
        for i, k in enumerate(ordered):
            assign[k] = "dev" if i < half else "hidden"
        return assign
    counts = {t: sum(1 for k in defects if k[0] == t) for t in types}
    order = sorted(types, key=lambda t: (-counts[t], keyed("type", cat, t)))
    totals = {"dev": 0, "hidden": 0}
    side_of: dict[str, str] = {}
    for t in order:
        if totals["dev"] == totals["hidden"]:
            side = "dev" if keyed("tie", cat, t) % 2 == 0 else "hidden"
        else:
            side = "dev" if totals["dev"] < totals["hidden"] else "hidden"
        side_of[t] = side
        totals[side] += counts[t]
    if len(set(side_of.values())) < 2:  # guarantee both halves see defects
        smallest = order[-1]
        side_of[smallest] = "hidden" if side_of[smallest] == "dev" else "dev"
    for k in defects:
        assign[k] = side_of[k[0]]
    return assign


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_current() -> bool:
    if not MANIFEST.exists():
        return False
    try:
        manifest = json.loads(MANIFEST.read_text())
    except json.JSONDecodeError:
        return False
    if manifest.get("format_version") != FORMAT_VERSION:
        return False
    for rel, digest in manifest.get("sha256", {}).items():
        path = ROOT / rel
        if not path.exists() or path.stat().st_size != manifest.get("bytes", {}).get(rel, -1):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if is_current() and not args.force:
        print("prepared MVTec-AD arrays are current; nothing to do")
        return
    if not ARCHIVE.exists():
        raise SystemExit(f"missing {ARCHIVE}; run scripts/download.sh on a node with internet")

    train, test, gt = read_archive(ARCHIVE)

    train_images, train_cats = [], []
    for ci, cat in enumerate(CATEGORIES):
        names = sorted(train[cat])
        if not names:
            raise SystemExit(f"no training images for {cat}")
        train_images.extend(train[cat][n] for n in names)
        train_cats.extend([ci] * len(names))

    items = []  # (cat_index, defect_type, file, image, mask, half)
    for ci, cat in enumerate(CATEGORIES):
        keys = sorted(test[cat])
        if not keys:
            raise SystemExit(f"no test images for {cat}")
        assign = split_category(cat, keys)
        for key in keys:
            defect, fname = key
            if defect == "good":
                mask = np.zeros((SIZE, SIZE), dtype=bool)
            else:
                if key not in gt[cat]:
                    raise SystemExit(f"missing ground truth for {cat}/{defect}/{fname}")
                mask = gt[cat][key]
            items.append((ci, defect, fname, test[cat][key], mask, assign[key]))

    # Anonymous ids: rank of a secret-keyed hash of the source file, so the id
    # order carries no information about category, label or defect type.
    ranked = sorted(range(len(items)), key=lambda i: keyed("id", items[i][0], items[i][1], items[i][2]))
    ids = np.zeros(len(items), dtype=np.int64)
    for r, i in enumerate(ranked):
        ids[i] = 100_000 + r
    order = np.argsort(ids)

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEV_OUT.parent.mkdir(parents=True, exist_ok=True)
    HIDDEN_OUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        TRAIN_OUT,
        images=np.stack(train_images).astype(np.uint8),
        categories=np.asarray(train_cats, dtype=np.int64),
        category_names=np.asarray(CATEGORIES),
    )
    np.savez(
        INPUTS_OUT,
        images=np.stack([items[i][3] for i in order]).astype(np.uint8),
        categories=np.asarray([items[i][0] for i in order], dtype=np.int64),
        ids=ids[order],
        category_names=np.asarray(CATEGORIES),
    )
    summary = {}
    for half, path in (("dev", DEV_OUT), ("hidden", HIDDEN_OUT)):
        sel = [i for i in order if items[i][5] == half]
        np.savez_compressed(
            path,
            all_ids=np.sort(ids),
            ids=ids[sel],
            labels=np.asarray([0 if items[i][1] == "good" else 1 for i in sel], dtype=np.uint8),
            masks=np.stack([items[i][4] for i in sel]).astype(bool),
            categories=np.asarray([items[i][0] for i in sel], dtype=np.int64),
            defect_types=np.asarray([items[i][1] for i in sel]),
            category_names=np.asarray(CATEGORIES),
        )
        summary[half] = {
            "images": len(sel),
            "anomalous": int(sum(items[i][1] != "good" for i in sel)),
            "normal": int(sum(items[i][1] == "good" for i in sel)),
        }

    outputs = [TRAIN_OUT, INPUTS_OUT, DEV_OUT, HIDDEN_OUT]
    manifest = {
        "format_version": FORMAT_VERSION,
        "image_size": SIZE,
        "train_images": len(train_images),
        "test_images": len(items),
        "halves": summary,
        "sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in outputs},
        "bytes": {str(p.relative_to(ROOT)): p.stat().st_size for p in outputs},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("sha256", "bytes")}, indent=2))


if __name__ == "__main__":
    main()
