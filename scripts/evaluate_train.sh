#!/bin/bash
set -euo pipefail

cd /home/user

if [ "$#" -eq 0 ]; then
  mapfile -t prediction_artifacts < <(find outputs -mindepth 2 -maxdepth 3 -type f -name predictions.npz | sort)
  if [ "${#prediction_artifacts[@]}" -eq 0 ]; then
    echo "No prediction artifacts found under outputs/*/predictions.npz" >&2
    exit 1
  fi
  set -- "${prediction_artifacts[@]}"
fi

python -m eval.train.evaluate --predictions "$@" --scores scoring/scores_train.json
