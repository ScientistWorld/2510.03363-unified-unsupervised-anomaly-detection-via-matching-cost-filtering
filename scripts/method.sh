#!/bin/bash
set -euo pipefail

cd /home/user

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUTPUT="${OUTPUT:-outputs/ucf_synthetic}"
CHECKPOINT="${CHECKPOINT:-checkpoints/ucf_synthetic.pt}"
SEED="${SEED:-7}"
python -m method.train_ucf \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --output "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --seed "$SEED"

bash scripts/evaluate.sh "$OUTPUT/predictions.npz"
