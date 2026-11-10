#!/bin/bash
# AnomalDF+UCF: train the cost-volume filtering network on normal images with
# synthetic anomalies, then predict every test input with the paper's fused
# inference (0.5 * UCF anomaly probability + 0.5 * AnomalDF map, same N=3
# templates; method/fusion.py).
# Training-setup defaults (epochs, lr, foreground restriction) come from
# method/ucf_setup.json (chosen on the visible dev slice by method/select_setup.py).
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-0}"
python -m method.train_ucf \
  --seed "$SEED" \
  --output outputs/anomaldf_ucf \
  --checkpoint "checkpoints/anomaldf_ucf_seed${SEED}.pt"
