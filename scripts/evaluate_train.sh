#!/bin/bash
# Development slice: score every method directory under outputs/ (outputs/
# <method>/predictions.npz) against the visible development answers in
# eval/train/data/dev_answers.npz. A method directory without a valid artifact
# is scored as failed (worst value on every metric), never skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f eval/train/data/dev_answers.npz ]; then
  echo "eval/train/data/dev_answers.npz missing; run scripts/prepare_data.sh" >&2
  exit 1
fi
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

python -m eval.train.evaluate --predictions "$@" --scores scoring/scores_train.json
