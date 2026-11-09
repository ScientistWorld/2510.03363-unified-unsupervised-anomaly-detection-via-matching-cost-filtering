#!/bin/bash
# AnomalDF+UCF: train the cost-volume filtering network and predict on all test inputs.
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-40}"
python -m method.train_ucf \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --output outputs/anomaldf_ucf \
  --checkpoint "checkpoints/anomaldf_ucf_seed${SEED}.pt"
