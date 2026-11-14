#!/bin/bash
# Reproduce the reference end to end (after scripts/download.sh): data
# preparation (MVTec-AD and VisA), the full-shot AnomalDF baseline,
# AnomalDF+UCF training (seed 0, one model per dataset) with full-shot
# inference, and both evaluations.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/prepare_data.sh
bash scripts/baseline.sh
[ -f method/ucf_setup.json ] || python -m method.select_setup  # dev-slice setup selection (committed result)
SEED=0 bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
python validate.py --compare --post-design
