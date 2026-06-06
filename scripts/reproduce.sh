#!/bin/bash
set -euo pipefail

cd /home/user
bash scripts/method.sh
bash scripts/evaluate_train.sh outputs/ucf_synthetic/predictions.npz
bash scripts/evaluate_test.sh outputs/ucf_synthetic/predictions.npz
python validate.py --compare
