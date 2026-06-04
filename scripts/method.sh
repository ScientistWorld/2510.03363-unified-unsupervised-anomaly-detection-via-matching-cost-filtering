#!/bin/bash
set -euo pipefail

cd /home/user

EPOCHS="${EPOCHS:-24}"
BATCH_SIZE="${BATCH_SIZE:-8}"
python -m method.train_ucf \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --output outputs/ucf_synthetic \
  --checkpoint checkpoints/ucf_synthetic.pt

bash scripts/evaluate.sh outputs/ucf_synthetic/predictions.npz
