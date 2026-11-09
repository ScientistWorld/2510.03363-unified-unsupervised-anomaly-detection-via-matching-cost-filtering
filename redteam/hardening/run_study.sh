#!/bin/bash
# Hardening study: training-seed noise, better and weaker variants, the
# full-memory-bank brute force, and shortcut payloads, all scored by the
# unchanged evaluators. Resumable: finished artifacts and score files are kept.
# Requires the reference run (outputs/anomaldf, outputs/anomaldf_ucf,
# checkpoints/anomaldf_ucf_seed0.pt) to exist.
set -uo pipefail
cd "$(dirname "$0")/../.."

H=redteam/hardening
OUT=$H/outputs
SC=$H/scores
mkdir -p "$OUT" "$SC"

step() {  # step <name> <command...>
  local name=$1
  shift
  if [ -f "$OUT/$name/predictions.npz" ]; then
    echo "== skip $name (exists)"
    return 0
  fi
  echo "== $name: $*"
  local t0=$SECONDS
  "$@" || echo "!! $name failed"
  echo "== $name done in $((SECONDS - t0))s"
}

score_all() {
  for d in "$OUT"/*/; do
    local name
    name=$(basename "$d")
    [ -f "$d/predictions.npz" ] || continue
    [ -f "$SC/${name}_test.json" ] || python -m eval.test.evaluate --predictions "$d/predictions.npz" --scores "$SC/${name}_test.json" || echo "!! scoring $name (test) failed"
    [ -f "$SC/${name}_train.json" ] || python -m eval.train.evaluate --predictions "$d/predictions.npz" --scores "$SC/${name}_train.json" || echo "!! scoring $name (train) failed"
  done
  python -m redteam.hardening.summarize || echo "!! summarize failed"
}

# Fast group: inference-only variants of the trained reference, brute force, weak variant, shortcuts.
step ucf_tta4 python -m method.train_ucf --predict-only --checkpoint checkpoints/anomaldf_ucf_seed0.pt --test-draws 4 --output "$OUT/ucf_tta4"
step ucf_nearest_templates python -m method.train_ucf --predict-only --checkpoint checkpoints/anomaldf_ucf_seed0.pt --test-templates nearest --output "$OUT/ucf_nearest_templates"
step knn_fullshot python -m baseline.anomaldf --templates all --output "$OUT/knn_fullshot"
E=$(python -c "import json; print(json.load(open('method/ucf_setup.json'))['epochs'])")
if [ "$E" -ge 4 ]; then
  step ucf_quarter_epochs python -m method.train_ucf --epochs $((E / 4)) --output "$OUT/ucf_quarter_epochs" --checkpoint checkpoints/anomaldf_ucf_quarter_epochs.pt
fi
step ucf_tenth_data python -m method.train_ucf --train-fraction 0.1 --output "$OUT/ucf_tenth_data" --checkpoint checkpoints/anomaldf_ucf_tenth_data.pt
python -m redteam.hardening.shortcuts || echo "!! shortcuts failed"
step shortcut_dev_supervised python -m redteam.hardening.dev_supervised_shortcut
score_all

# Slow group: full trainings.
step ucf_seed1 python -m method.train_ucf --seed 1 --output "$OUT/ucf_seed1" --checkpoint checkpoints/anomaldf_ucf_seed1.pt
step ucf_seed2 python -m method.train_ucf --seed 2 --output "$OUT/ucf_seed2" --checkpoint checkpoints/anomaldf_ucf_seed2.pt
score_all
step ucf_n8_templates python -m method.train_ucf --templates 8 --output "$OUT/ucf_n8_templates" --checkpoint checkpoints/anomaldf_ucf_n8.pt
step ucf_no_guidance python -m method.train_ucf --no-guidance --output "$OUT/ucf_no_guidance" --checkpoint checkpoints/anomaldf_ucf_noguide.pt
score_all
