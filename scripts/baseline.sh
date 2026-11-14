#!/bin/bash
# AnomalDF baseline: frozen DINOv2 patch matching against the whole category
# memory bank (AnomalyDINO's full-shot setting, paper Table XXIV Exp. IDs 3/11),
# on MVTec-AD and VisA -> outputs/anomaldf/{predictions,predictions_visa}.npz
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-cuda}"
python -m baseline.anomaldf --dataset mvtec --device "$DEVICE" --output outputs/anomaldf
python -m baseline.anomaldf --dataset visa --device "$DEVICE" --output outputs/anomaldf
