"""Build the gym's 224px VisA arrays from the official archive.

This is the ONLY code that can map anonymised VisA test inputs back to VisA
file names (and therefore to labels and masks). It lives under eval/test/ on
purpose: the raw archive (eval/test/downloads/VisA_20220922.tar) and the
hidden answers never leave eval/test/.

Uses the official one-class split (split_csv/1cls.csv: 8,659 normal training
images, 2,162 test images over 12 categories) and writes

  data/downloads/visa224/train.npz         normal training images (method-visible)
  data/downloads/visa224/test_inputs.npz   every test image, shuffled under random
                                           ids, with its category only (method-visible)
  eval/train/data/visa_dev_answers.npz     labels + masks of the visible development third
  eval/test/downloads/visa_test_answers.npz labels + masks of the hidden two thirds

Split: per category, one third of the normal test images go to the visible
development slice and two thirds to the hidden slice. Defective images are
grouped by their primary defect type (first entry of the label in the
category's image_anno.csv) and whole types are assigned to one side, aiming
at the same one-third / two-thirds ratio, so the hidden slice holds defect
types the development slice never shows (categories with a single primary
type split their defective images by the same ratio).

Images are resized to 224x224 (bicubic). Masks are binarised (any non-zero
value is defect), box-averaged to 224x224 and thresholded at one half; a
defect that would vanish at that threshold keeps its strongest cells.

Run inside the container:  python -m eval.test.prepare_visa
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "eval" / "test" / "downloads" / "VisA_20220922.tar"
TRAIN_OUT = ROOT / "data" / "downloads" / "visa224" / "train.npz"
INPUTS_OUT = ROOT / "data" / "downloads" / "visa224" / "test_inputs.npz"
DEV_OUT = ROOT / "eval" / "train" / "data" / "visa_dev_answers.npz"
HIDDEN_OUT = ROOT / "eval" / "test" / "downloads" / "visa_test_answers.npz"
MANIFEST = ROOT / "eval" / "test" / "downloads" / "visa_prepare_manifest.json"

FORMAT_VERSION = 1
SIZE = 224
DEV_FRACTION = 1.0 / 3.0
ID_OFFSET = 200_000
SECRET = "costfilter-gym/visa-hidden-split/v1"
CATEGORIES = (
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
)


def keyed(*parts: object) -> int:
    raw = "|".join([SECRET, *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def load_image(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def load_mask(data: bytes) -> np.ndarray:
    raw = np.asarray(Image.open(io.BytesIO(data)).convert("L")) != 0
    avg = np.asarray(Image.fromarray(raw.astype(np.uint8) * 255).resize((SIZE, SIZE), Image.BOX), dtype=np.uint8)
    mask = avg >= 128
    if raw.any() and not mask.any():  # tiny defect: keep its strongest cells rather than lose it
        mask = avg >= max(1, int(avg.max()) // 2)
    return mask


def norm(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def split_counts(n: int) -> int:
    """Number of items of n that go to the development side (at least one of each side when n >= 2)."""
    k = int(round(n * DEV_FRACTION))
    if n >= 2:
        k = min(max(k, 1), n - 1)
    return k


def split_category(cat: str, normals: list[str], defects: dict[str, str]) -> dict[str, str]:
    """normals: image paths; defects: {image path: primary type}. Returns {path: 'dev'|'hidden'}."""
    assign: dict[str, str] = {}
    goods = sorted(normals, key=lambda p: keyed("good", cat, p))
    k = split_counts(len(goods))
    for i, p in enumerate(goods):
        assign[p] = "dev" if i < k else "hidden"
    types = sorted(set(defects.values()))
    if len(types) == 1:
        ordered = sorted(defects, key=lambda p: keyed("single", cat, p))
        k = split_counts(len(ordered))
        for i, p in enumerate(ordered):
            assign[p] = "dev" if i < k else "hidden"
        return assign
    counts = {t: sum(1 for v in defects.values() if v == t) for t in types}
    total = float(sum(counts.values()))
    target = {"dev": DEV_FRACTION * total, "hidden": (1.0 - DEV_FRACTION) * total}
    have = {"dev": 0, "hidden": 0}
    side_of: dict[str, str] = {}
    for t in sorted(types, key=lambda t: (-counts[t], keyed("type", cat, t))):
        deficit = {s: target[s] - have[s] for s in have}
        side = max(deficit, key=lambda s: (deficit[s], s == "hidden"))
        side_of[t] = side
        have[side] += counts[t]
    if len(set(side_of.values())) < 2:  # both sides must see defects
        smallest = min(types, key=lambda t: (counts[t], keyed("type", cat, t)))
        side_of[smallest] = "dev" if side_of[smallest] == "hidden" else "hidden"
    for p, t in defects.items():
        assign[p] = side_of[t]
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
    for rel in manifest.get("sha256", {}):
        path = ROOT / rel
        if not path.exists() or path.stat().st_size != manifest.get("bytes", {}).get(rel, -1):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if is_current() and not args.force:
        print("prepared VisA arrays are current; nothing to do")
        return
    if not ARCHIVE.exists():
        raise SystemExit(f"missing {ARCHIVE}; run scripts/download.sh on a node with internet")

    with tarfile.open(ARCHIVE, mode="r:") as tf:
        members = {norm(m.name): m for m in tf.getmembers() if m.isfile()}

        def read(name: str) -> bytes:
            fh = tf.extractfile(members[norm(name)])
            if fh is None:
                raise SystemExit(f"cannot read {name} from {ARCHIVE}")
            return fh.read()

        split_rows = list(csv.DictReader(io.StringIO(read("split_csv/1cls.csv").decode("utf-8"))))
        primary: dict[str, str] = {}
        for cat in CATEGORIES:
            for row in csv.DictReader(io.StringIO(read(f"{cat}/image_anno.csv").decode("utf-8"))):
                label = (row.get("label") or "").strip()
                if label and label != "normal":
                    primary[norm(row["image"])] = label.split(",")[0].strip()

        train_rows = [r for r in split_rows if r["split"] == "train"]
        test_rows = [r for r in split_rows if r["split"] == "test"]
        for r in split_rows:
            if r["object"] not in CATEGORIES:
                raise SystemExit(f"unexpected category {r['object']}")
            if r["split"] == "train" and r["label"] != "normal":
                raise SystemExit("1cls split lists an anomalous training image")

        # Read every needed member once, in archive order (fast sequential I/O).
        wanted = {norm(r["image"]) for r in split_rows} | {norm(r["mask"]) for r in test_rows if r["label"] != "normal" and r["mask"]}
        missing = sorted(w for w in wanted if w not in members)
        if missing:
            raise SystemExit(f"{len(missing)} files listed in 1cls.csv are missing from the archive, e.g. {missing[:3]}")
        decoded: dict[str, np.ndarray] = {}
        mask_names = {norm(r["mask"]) for r in test_rows if r["label"] != "normal" and r["mask"]}
        for k, name in enumerate(sorted(wanted, key=lambda n: members[n].offset_data)):
            data = read(name)
            decoded[name] = load_mask(data) if name in mask_names else load_image(data)
            if (k + 1) % 2000 == 0:
                print(f"decoded {k + 1}/{len(wanted)}", flush=True)

    cat_index = {c: i for i, c in enumerate(CATEGORIES)}
    train_rows.sort(key=lambda r: (cat_index[r["object"]], norm(r["image"])))
    train_images = np.stack([decoded[norm(r["image"])] for r in train_rows]).astype(np.uint8)
    train_cats = np.asarray([cat_index[r["object"]] for r in train_rows], dtype=np.int64)

    items = []  # (cat_index, path, defect_type, image, mask, half)
    for cat in CATEGORIES:
        rows = [r for r in test_rows if r["object"] == cat]
        normals = [norm(r["image"]) for r in rows if r["label"] == "normal"]
        defects = {}
        for r in rows:
            if r["label"] != "normal":
                p = norm(r["image"])
                defects[p] = primary.get(p, "unlabelled")
        assign = split_category(cat, normals, defects)
        for r in sorted(rows, key=lambda r: norm(r["image"])):
            p = norm(r["image"])
            if r["label"] == "normal":
                mask = np.zeros((SIZE, SIZE), dtype=bool)
                dtype = "good"
            else:
                if not r["mask"]:
                    raise SystemExit(f"anomalous test image without mask: {p}")
                mask = decoded[norm(r["mask"])]
                dtype = defects[p]
            items.append((cat_index[cat], p, dtype, decoded[p], mask, assign[p]))

    ranked = sorted(range(len(items)), key=lambda i: keyed("id", items[i][0], items[i][1]))
    ids = np.zeros(len(items), dtype=np.int64)
    for r, i in enumerate(ranked):
        ids[i] = ID_OFFSET + r
    order = np.argsort(ids)

    for p in (TRAIN_OUT, DEV_OUT, HIDDEN_OUT):
        p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TRAIN_OUT, images=train_images, categories=train_cats, category_names=np.asarray(CATEGORIES))
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
            labels=np.asarray([0 if items[i][2] == "good" else 1 for i in sel], dtype=np.uint8),
            masks=np.stack([items[i][4] for i in sel]).astype(bool),
            categories=np.asarray([items[i][0] for i in sel], dtype=np.int64),
            defect_types=np.asarray([items[i][2] for i in sel]),
            category_names=np.asarray(CATEGORIES),
        )
        summary[half] = {
            "images": len(sel),
            "anomalous": int(sum(items[i][2] != "good" for i in sel)),
            "normal": int(sum(items[i][2] == "good" for i in sel)),
        }
    empty = sum(1 for it in items if it[2] != "good" and not it[4].any())
    outputs = [TRAIN_OUT, INPUTS_OUT, DEV_OUT, HIDDEN_OUT]
    manifest = {
        "format_version": FORMAT_VERSION,
        "image_size": SIZE,
        "train_images": int(len(train_rows)),
        "test_images": len(items),
        "halves": summary,
        "anomalous_images_with_empty_224px_mask": empty,
        "sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in outputs},
        "bytes": {str(p.relative_to(ROOT)): p.stat().st_size for p in outputs},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("sha256", "bytes")}, indent=2))


if __name__ == "__main__":
    main()
