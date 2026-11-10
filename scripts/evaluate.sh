#!/bin/bash
# Canonical reproduction scores (scoring/scores.json) on the visible
# development slice, for every method directory under outputs/ (or the given
# artifacts). Same evaluator and protocol as scripts/evaluate_train.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
  artifacts=()
  while IFS= read -r method_dir; do
    artifacts+=("$method_dir/predictions.npz")
  done < <(find outputs -mindepth 1 -maxdepth 1 -type d -not -name '.*' 2>/dev/null | sort)
  if [ "${#artifacts[@]}" -eq 0 ]; then
    echo "No method directories found under outputs/" >&2
    exit 1
  fi
  set -- "${artifacts[@]}"
fi
python -m eval.train.evaluate --predictions "$@" --scores scoring/scores.json
