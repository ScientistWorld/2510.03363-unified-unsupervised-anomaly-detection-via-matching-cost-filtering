#!/bin/bash
# Fetch every external artifact this gym needs. Idempotent: each fetch is
# skipped when its target already exists and verifies, so the script works
# from an empty data/downloads/ and an empty eval/test/downloads/ on any node
# with internet access.
#
# Estimated download size: ~6.0 GB (MVTec-AD 5.26 GB, DTD 0.63 GB, DINOv2 0.09 GB)
# Estimated disk usage after preparation: ~7 GB
#
# Layout:
#   eval/test/downloads/mvtec_anomaly_detection.tar.xz
#       Official MVTec-AD archive (CC BY-NC-SA 4.0). It contains the test
#       ground truth, so it lives ONLY under eval/test/ (gitignored). The
#       preparation step (scripts/prepare_data.sh) turns it into anonymised
#       test inputs under data/downloads/ and hidden answers under eval/test/.
#   data/downloads/dtd/dtd-r1.0.1.tar.gz
#       Describable Textures Dataset, used only as an anomaly-synthesis source.
#   data/downloads/dinov2/dinov2_vits14_pretrain.pth
#       Official DINOv2 ViT-S/14 weights (frozen feature extractor).
#   data/downloads/dinov2/repo/
#       facebookresearch/dinov2 source at a pinned commit (model definition).

set -euo pipefail
cd "$(dirname "$0")/.."

MVTEC_URL="https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/main/mvtec_anomaly_detection.tar.xz"
MVTEC_SHA256="cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"   # official MVTec-AD archive hash
DTD_URL="https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
DINO_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
DINO_REPO="https://github.com/facebookresearch/dinov2.git"
DINO_COMMIT="7764ea0f912e53c92e82eb78a2a1631e92725fc8"

mkdir -p data/downloads/dtd data/downloads/dinov2 eval/test/downloads

fetch() {  # fetch <url> <dest>
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then
    echo "present: $dest"
    return 0
  fi
  echo "downloading: $url -> $dest"
  local attempt
  for attempt in $(seq 1 30); do
    if curl -L --fail --http1.1 --retry 5 --retry-all-errors --retry-delay 10 \
         --speed-limit 50000 --speed-time 120 -sS -C - -o "$dest.part" "$url"; then
      mv "$dest.part" "$dest"
      return 0
    fi
    echo "  transfer interrupted (attempt $attempt); resuming in 15s" >&2
    sleep 15
  done
  echo "failed to download $url" >&2
  return 1
}

# 1) MVTec-AD (answers inside -> eval/test/downloads only)
MVTEC_TAR=eval/test/downloads/mvtec_anomaly_detection.tar.xz
fetch "$MVTEC_URL" "$MVTEC_TAR"
if [ ! -f "$MVTEC_TAR.sha256ok" ]; then
  echo "$MVTEC_SHA256  $MVTEC_TAR" | sha256sum -c -
  touch "$MVTEC_TAR.sha256ok"
fi

# 2) DTD textures (anomaly synthesis source for training only)
fetch "$DTD_URL" data/downloads/dtd/dtd-r1.0.1.tar.gz

# 3) DINOv2 ViT-S/14 weights + model code at a pinned commit
fetch "$DINO_URL" data/downloads/dinov2/dinov2_vits14_pretrain.pth
if [ ! -f data/downloads/dinov2/repo/dinov2/models/vision_transformer.py ]; then
  rm -rf data/downloads/dinov2/repo
  git clone --quiet "$DINO_REPO" data/downloads/dinov2/repo
  git -C data/downloads/dinov2/repo checkout --quiet "$DINO_COMMIT"
fi

echo "downloads complete; run scripts/prepare_data.sh (inside the container) to build the 224px arrays"
