#!/bin/bash
# Score every method under outputs/*/predictions.npz on the visible development
# slice (labels/masks in eval/train/data/dev_answers.npz).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f eval/train/data/dev_answers.npz ]; then
  echo "eval/train/data/dev_answers.npz missing; run scripts/prepare_data.sh" >&2
  exit 1
fi
if [ "$#" -eq 0 ]; then
  mapfile -t prediction_artifacts < <(find outputs -mindepth 2 -maxdepth 2 -type f -name predictions.npz | sort)
  if [ "${#prediction_artifacts[@]}" -eq 0 ]; then
    echo "No prediction artifacts found under outputs/*/predictions.npz" >&2
    exit 1
  fi
  set -- "${prediction_artifacts[@]}"
fi

python -m eval.train.evaluate --predictions "$@" --scores scoring/scores_train.json
