#!/bin/bash
set -euo pipefail

cd /home/user

bash scripts/download.sh
EPOCHS="${EPOCHS:-40}" BATCH_SIZE="${BATCH_SIZE:-8}" bash scripts/method.sh
SEED=13 OUTPUT=outputs/ucf_synthetic_seed13 CHECKPOINT=checkpoints/ucf_synthetic_seed13.pt \
  EPOCHS="${EPOCHS:-40}" BATCH_SIZE="${BATCH_SIZE:-8}" bash scripts/method.sh
bash scripts/evaluate.sh \
  outputs/ucf_synthetic/predictions.npz \
  outputs/ucf_synthetic_seed13/predictions.npz
python validate.py --compare
