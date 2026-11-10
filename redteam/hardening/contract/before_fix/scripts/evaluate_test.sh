#!/bin/bash
# Hidden test: score every method directory under outputs/ (outputs/<method>/
# predictions.npz) against the answers held in eval/test/ (rebuilt from the
# archive if absent). A method directory without a valid artifact is scored
# as failed (worst value on every metric), never skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f eval/test/downloads/test_answers.npz ]; then
  python -m eval.test.prepare_mvtec
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

python -m eval.test.evaluate --predictions "$@" --scores scoring/scores_test.json
