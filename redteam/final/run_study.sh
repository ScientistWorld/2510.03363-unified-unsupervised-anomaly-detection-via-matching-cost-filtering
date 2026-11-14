#!/bin/bash
# Final hardening study (both experiments): training-seed noise, better and
# weaker variants, the paper's N=3 inference and lambda sensitivity, shortcut
# payloads, all scored by the gym's evaluators (python -m eval.{train,test}.evaluate).
# Requires the reference run (outputs/anomaldf, outputs/anomaldf_ucf and the
# seed-0 checkpoints). Resumable: finished artifacts and score files are kept.
# With two visible GPUs, the MVTec-AD and VisA variants run in parallel.
set -uo pipefail
cd "$(dirname "$0")/../.."

F=redteam/final
OUT=$F/outputs
SC=$F/scores
REF=outputs/anomaldf_ucf
CK=checkpoints
mkdir -p "$OUT" "$SC" tmp
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

pred_name() { if [ "$1" = mvtec ]; then echo predictions.npz; else echo "predictions_$1.npz"; fi; }
ckpt() {  # ckpt <dataset> <tag>
  if [ "$1" = mvtec ]; then echo "$CK/anomaldf_ucf_$2.pt"; else echo "$CK/anomaldf_ucf_$1_$2.pt"; fi
}

step() {  # step <dataset> <name> <command...>   (skipped when the artifact exists)
  local ds=$1 name=$2
  shift 2
  if [ -f "$OUT/$name/$(pred_name "$ds")" ]; then
    echo "== skip $name/$ds (exists)"
    return 0
  fi
  echo "== $name/$ds: $*"
  local t0=$SECONDS
  "$@" || echo "!! $name/$ds failed"
  echo "== $name/$ds done in $((SECONDS - t0))s"
}

train() {  # train <dataset> <device> <name> <checkpoint tag> [train_ucf args...]
  local ds=$1 dev=$2 name=$3 tag=$4
  shift 4
  step "$ds" "$name" python -m method.train_ucf --dataset "$ds" --device "$dev" --output "$OUT/$name" --checkpoint "$(ckpt "$ds" "$tag")" "$@"
}

predict() {  # predict <dataset> <device> <name> <checkpoint tag> [train_ucf args...]   (inference only)
  local ds=$1 dev=$2 name=$3 tag=$4
  shift 4
  step "$ds" "$name" python -m method.train_ucf --predict-only --dataset "$ds" --device "$dev" --output "$OUT/$name" --checkpoint "$(ckpt "$ds" "$tag")" "$@"
}

fused() {  # fused <dataset> <name> <lambda> <run dirs...>   (rebuilt from stored components)
  local ds=$1 name=$2 lam=$3
  shift 3
  step "$ds" "$name" python -m method.fusion --dataset "$ds" --lam "$lam" --output "$OUT/$name" --runs "$@"
}

stream() {  # stream <dataset> <device>
  local ds=$1 dev=$2
  # inference-only variants of the seed-0 reference model
  predict "$ds" "$dev" ucf_n3_random seed0 --test-templates random
  step "$ds" anomaldf_n3 python -m baseline.anomaldf --dataset "$ds" --device "$dev" --templates 3 --output "$OUT/anomaldf_n3"
  fused "$ds" ucf_alone 1.0 "$REF"
  fused "$ds" fused_lam0 0.0 "$REF"
  fused "$ds" fused_lam0p25 0.25 "$REF"
  fused "$ds" fused_lam0p75 0.75 "$REF"
  # training seeds and the seed ensemble
  train "$ds" "$dev" ucf_seed1 seed1 --seed 1
  train "$ds" "$dev" ucf_seed2 seed2 --seed 2
  fused "$ds" ucf_ensemble3 0.5 "$REF" "$OUT/ucf_seed1" "$OUT/ucf_seed2"
  # better candidate: train with more templates (closer to the full-bank volume used at test time)
  train "$ds" "$dev" ucf_train_n16 n16 --templates 16
  # paper ablations (worse in the paper)
  train "$ds" "$dev" ucf_focal_only focal --loss focal
  train "$ds" "$dev" ucf_no_guidance noguide --no-guidance
}

score_all() {  # every directory under OUT is a method; a missing artifact is scored as failed
  for d in "$OUT"/*/; do
    local name
    name=$(basename "$d")
    [ -f "$SC/${name}_test.json" ] || python -m eval.test.evaluate --predictions "$d/predictions.npz" --scores "$SC/${name}_test.json" || echo "!! scoring $name (test) failed"
    [ -f "$SC/${name}_train.json" ] || python -m eval.train.evaluate --predictions "$d/predictions.npz" --scores "$SC/${name}_train.json" || echo "!! scoring $name (train) failed"
  done
  python -m redteam.final.summarize || echo "!! summarize failed"
}

T0=$SECONDS
if [ "$NGPU" -ge 2 ]; then
  stream mvtec cuda:0 > tmp/study_mvtec.log 2>&1 &
  p1=$!
  stream visa cuda:1 > tmp/study_visa.log 2>&1 &
  p2=$!
  wait $p1
  wait $p2
  cat tmp/study_mvtec.log tmp/study_visa.log
else
  stream mvtec cuda:0
  stream visa cuda:0
fi
echo "== study variants done at $((SECONDS - T0))s"

python -m redteam.final.shortcuts || echo "!! shortcuts failed"
[ -f "$OUT/shortcut_dev_supervised/predictions_visa.npz" ] || python -m redteam.final.dev_supervised_shortcut || echo "!! dev-supervised shortcut failed"
score_all
echo "== study scored at $((SECONDS - T0))s"
python -m redteam.final.summarize --bootstrap || echo "!! bootstrap summary failed"
echo "== bootstrap done at $((SECONDS - T0))s"
