#!/bin/bash
set -euo pipefail

cd /home/user

if [ ! -f outputs/ucf_synthetic/predictions.npz ]; then
  echo "Missing outputs/ucf_synthetic/predictions.npz; run scripts/method.sh or scripts/reproduce.sh first." >&2
  exit 1
fi

bash scripts/evaluate.sh outputs/ucf_synthetic/predictions.npz
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --compare
