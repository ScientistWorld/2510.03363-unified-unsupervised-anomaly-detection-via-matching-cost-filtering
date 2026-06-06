#!/bin/bash
set -euo pipefail

cd /home/user

python -m baseline.raw_cost_volume --output outputs/raw_cost_volume
bash scripts/evaluate.sh outputs/raw_cost_volume/predictions.npz
