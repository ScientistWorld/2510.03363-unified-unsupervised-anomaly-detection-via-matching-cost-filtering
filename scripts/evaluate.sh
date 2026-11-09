#!/bin/bash
# Canonical reproduction scores (scoring/scores.json) on the visible
# development slice, for every method under outputs/ (or the given artifacts).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
  mapfile -t prediction_artifacts < <(find outputs -mindepth 2 -maxdepth 2 -type f -name predictions.npz | sort)
  set -- "${prediction_artifacts[@]}"
fi
python -m eval.train.evaluate --predictions "$@" --scores scoring/scores.json
