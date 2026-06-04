#!/bin/bash
set -euo pipefail

cd /home/user

PREDICTIONS="${1:-outputs/ucf_synthetic/predictions.npz}"
python -m eval.evaluate_predictions --predictions "$PREDICTIONS" --scores scoring/scores.json
