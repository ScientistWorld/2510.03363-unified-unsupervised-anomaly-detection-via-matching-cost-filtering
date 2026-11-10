#!/bin/bash
# Hardening study: training-seed noise, better and weaker variants, lambda
# sensitivity, the full-memory-bank brute force, shortcut payloads and the
# evaluation-contract checks (contract_checks.sh), all scored by the gym's
# evaluators (python -m eval.{train,test}.evaluate).
# Resumable: finished artifacts and score files are kept.
# Requires the reference run (outputs/anomaldf, outputs/anomaldf_ucf,
# checkpoints/anomaldf_ucf_seed0.pt, method/ucf_setup.json) to exist.
set -uo pipefail
cd "$(dirname "$0")/../.."

H=redteam/hardening
OUT=$H/outputs
SC=$H/scores
REF=outputs/anomaldf_ucf
CK=checkpoints
mkdir -p "$OUT" "$SC"

step() {  # step <name> <command...>   (skipped when OUT/<name>/predictions.npz exists)
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

score_all() {  # every directory under OUT is a method; a missing artifact is scored as failed
  for d in "$OUT"/*/; do
    local name
    name=$(basename "$d")
    [ -f "$SC/${name}_test.json" ] || python -m eval.test.evaluate --predictions "$d/predictions.npz" --scores "$SC/${name}_test.json" || echo "!! scoring $name (test) failed"
    [ -f "$SC/${name}_train.json" ] || python -m eval.train.evaluate --predictions "$d/predictions.npz" --scores "$SC/${name}_train.json" || echo "!! scoring $name (train) failed"
  done
  python -m redteam.hardening.summarize || echo "!! summarize failed"
}

predict() {  # predict <checkpoint> <output> [extra train_ucf args...]  (inference only, selected setup)
  local ckpt=$1 out=$2
  shift 2
  python -m method.train_ucf --predict-only --checkpoint "$ckpt" --output "$out" "$@"
}

fused() {  # fused <output> <lambda> <run dirs...>   (rebuild from stored components)
  local out=$1 lam=$2
  shift 2
  python -m method.fusion --lam "$lam" --output "$out" --runs "$@"
}

# ---- Fast group: inference-only variants of the seed-0 reference, brute force, shortcuts.
step ucf_nearest predict "$CK/anomaldf_ucf_seed0.pt" "$OUT/ucf_nearest" --test-templates nearest
step ucf_tta4 predict "$CK/anomaldf_ucf_seed0.pt" "$OUT/ucf_tta4" --test-draws 4
step ucf_fullbank predict "$CK/anomaldf_ucf_seed0.pt" "$OUT/ucf_fullbank" --baseline-bank all
step ucf_alone fused "$OUT/ucf_alone" 1.0 "$REF"
step fused_lam0p25 fused "$OUT/fused_lam0p25" 0.25 "$REF"
step fused_lam0p75 fused "$OUT/fused_lam0p75" 0.75 "$REF"
step fused_lam0 fused "$OUT/fused_lam0" 0.0 "$REF"
step knn_fullshot python -m baseline.anomaldf --templates all --output "$OUT/knn_fullshot"
python -m redteam.hardening.shortcuts || echo "!! shortcuts failed"
bash $H/contract_checks.sh || echo "!! contract checks failed"
step shortcut_dev_supervised python -m redteam.hardening.dev_supervised_shortcut
score_all

# ---- Seeds (training noise) and the better variants built on them.
step ucf_seed1 python -m method.train_ucf --seed 1 --output "$OUT/ucf_seed1" --checkpoint "$CK/anomaldf_ucf_seed1.pt"
step ucf_seed2 python -m method.train_ucf --seed 2 --output "$OUT/ucf_seed2" --checkpoint "$CK/anomaldf_ucf_seed2.pt"
step ucf_nearest_seed1 predict "$CK/anomaldf_ucf_seed1.pt" "$OUT/ucf_nearest_seed1" --seed 1 --test-templates nearest
step ucf_nearest_seed2 predict "$CK/anomaldf_ucf_seed2.pt" "$OUT/ucf_nearest_seed2" --seed 2 --test-templates nearest
step ucf_ensemble3 fused "$OUT/ucf_ensemble3" 0.5 "$REF" "$OUT/ucf_seed1" "$OUT/ucf_seed2"
step ucf_ensemble3_nearest fused "$OUT/ucf_ensemble3_nearest" 0.5 "$OUT/ucf_nearest" "$OUT/ucf_nearest_seed1" "$OUT/ucf_nearest_seed2"
score_all

# ---- Weaker variants (paper ablations and reduced training), selected setup otherwise.
E=$(python -c "import json; print(json.load(open('method/ucf_setup.json'))['epochs'])")
step ucf_no_guidance python -m method.train_ucf --no-guidance --output "$OUT/ucf_no_guidance" --checkpoint "$CK/anomaldf_ucf_noguide.pt"
step ucf_focal_only python -m method.train_ucf --loss focal --output "$OUT/ucf_focal_only" --checkpoint "$CK/anomaldf_ucf_focal.pt"
step ucf_tenth_data python -m method.train_ucf --train-fraction 0.1 --output "$OUT/ucf_tenth_data" --checkpoint "$CK/anomaldf_ucf_tenth_data.pt"
if [ "$E" -ge 4 ]; then
  step ucf_quarter_epochs python -m method.train_ucf --epochs $((E / 4)) --output "$OUT/ucf_quarter_epochs" --checkpoint "$CK/anomaldf_ucf_quarter_epochs.pt"
fi
score_all
python -m redteam.hardening.summarize --bootstrap || echo "!! bootstrap summary failed"
