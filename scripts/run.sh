#!/bin/bash
set -euo pipefail

cd /home/user

bash scripts/download.sh
EPOCHS="${EPOCHS:-40}" BATCH_SIZE="${BATCH_SIZE:-8}" bash scripts/method.sh
SEED=13 OUTPUT=outputs/ucf_synthetic_seed13 CHECKPOINT=checkpoints/ucf_synthetic_seed13.pt \
  EPOCHS="${EPOCHS:-40}" BATCH_SIZE="${BATCH_SIZE:-8}" bash scripts/method.sh

# Keep additional-seed robustness results separate from the canonical
# scores.json contract, which must mirror scoring/reference.json exactly.
python -m eval.evaluate_predictions \
  --predictions \
  outputs/ucf_synthetic/predictions.npz \
  outputs/ucf_synthetic_seed13/predictions.npz \
  --scores scoring/robustness_scores.json

bash scripts/evaluate.sh outputs/ucf_synthetic/predictions.npz
python validate.py --compare
