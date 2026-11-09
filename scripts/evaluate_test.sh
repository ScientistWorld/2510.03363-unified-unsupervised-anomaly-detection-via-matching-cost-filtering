#!/bin/bash
# Hidden test: score every method under outputs/*/predictions.npz against the
# answers held in eval/test/ (rebuilt from the archive if absent).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f eval/test/downloads/test_answers.npz ]; then
  python -m eval.test.prepare_mvtec
fi
if [ "$#" -eq 0 ]; then
  mapfile -t prediction_artifacts < <(find outputs -mindepth 2 -maxdepth 2 -type f -name predictions.npz | sort)
  if [ "${#prediction_artifacts[@]}" -eq 0 ]; then
    echo "No prediction artifacts found under outputs/*/predictions.npz" >&2
    exit 1
  fi
  set -- "${prediction_artifacts[@]}"
fi

python -m eval.test.evaluate --predictions "$@" --scores scoring/scores_test.json
