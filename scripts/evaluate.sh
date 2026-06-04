#!/bin/bash
set -euo pipefail

cd /home/user

if [ "$#" -eq 0 ]; then
  set -- outputs/ucf_synthetic/predictions.npz
fi

python -m eval.evaluate_predictions --predictions "$@" --scores scoring/scores.json
