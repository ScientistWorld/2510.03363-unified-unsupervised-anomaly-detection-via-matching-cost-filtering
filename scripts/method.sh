#!/bin/bash
# AnomalDF+UCF: train the cost-volume filtering network and predict on all test inputs.
# Training-setup defaults (epochs, lr, foreground restriction) come from
# method/ucf_setup.json (chosen on the visible dev slice by method/select_setup.py).
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-0}"
python -m method.train_ucf \
  --seed "$SEED" \
  --output outputs/anomaldf_ucf \
  --checkpoint "checkpoints/anomaldf_ucf_seed${SEED}.pt"
