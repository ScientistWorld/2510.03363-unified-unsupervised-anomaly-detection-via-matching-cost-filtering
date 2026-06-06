#!/bin/bash
set -euo pipefail

cd /home/user

if [ "$#" -eq 0 ]; then
  if [ ! -f outputs/ucf_synthetic/predictions.npz ]; then
    echo "Missing outputs/ucf_synthetic/predictions.npz" >&2
    exit 1
  fi
  set -- outputs/ucf_synthetic/predictions.npz
fi

python -m eval.test.evaluate \
  --predictions "$@" \
  --scores scoring/scores_test.json
