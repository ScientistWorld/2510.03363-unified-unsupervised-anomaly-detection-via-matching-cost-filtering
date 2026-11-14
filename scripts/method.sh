#!/bin/bash
# AnomalDF+UCF on MVTec-AD and VisA (one unified model per dataset, as in the
# paper): train the cost-volume filtering network on normal images with
# synthetic anomalies (N = 3 random templates per input, 40 epochs), then
# predict every test input against the whole category memory bank with the
# paper's fused inference (0.5 * UCF anomaly probability + 0.5 * AnomalDF map;
# paper Table XXIV full-shot protocol, method/fusion.py).
#   -> outputs/anomaldf_ucf/{predictions,predictions_visa}.npz
# With two visible GPUs the two datasets train in parallel (one per device).
# Extra arguments for method.train_ucf can be passed in UCF_EXTRA_ARGS.
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-0}"
OUT="${OUT:-outputs/anomaldf_ucf}"
read -r -a EXTRA <<< "${UCF_EXTRA_ARGS:-}"
mkdir -p tmp
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

run_ds() {  # run_ds <dataset> <device>
  local ds=$1 dev=$2 ckpt
  if [ "$ds" = mvtec ]; then ckpt="checkpoints/anomaldf_ucf_seed${SEED}.pt"; else ckpt="checkpoints/anomaldf_ucf_${ds}_seed${SEED}.pt"; fi
  python -m method.train_ucf --dataset "$ds" --device "$dev" --seed "$SEED" \
    --output "$OUT" --checkpoint "$ckpt" ${EXTRA[@]+"${EXTRA[@]}"}
}

if [ "$NGPU" -ge 2 ]; then
  run_ds mvtec cuda:0 > "tmp/method_mvtec_seed${SEED}.log" 2>&1 &
  p1=$!
  run_ds visa cuda:1 > "tmp/method_visa_seed${SEED}.log" 2>&1 &
  p2=$!
  rc=0
  wait $p1 || rc=1
  wait $p2 || rc=1
  cat "tmp/method_mvtec_seed${SEED}.log" "tmp/method_visa_seed${SEED}.log"
  exit $rc
else
  run_ds mvtec cuda:0
  run_ds visa cuda:0
fi
