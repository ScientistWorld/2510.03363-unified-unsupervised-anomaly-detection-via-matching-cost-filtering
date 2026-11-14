#!/bin/bash
# Job entry point: prepare both datasets, reproduce the reference end to end
# (full-shot AnomalDF baseline and AnomalDF+UCF seed 0 on MVTec-AD and VisA,
# paper Table XXIV protocol), score it on both slices, then run the final
# hardening study (seeds, better/weaker variants, the paper's N=3 inference,
# lambda sensitivity, shortcut payloads, paired bootstrap). Every stage is
# resumable: finished artifacts are kept, so a rerun only rescores; CUDA is
# required only when something has to be trained or predicted. With two
# GPUs, MVTec-AD and VisA run in parallel.
set -euo pipefail
cd /home/user
mkdir -p tmp

python - <<'PYEOF'
import os, torch, numpy, scipy, sklearn, PIL
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), flush=True)
print("gpus", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
print("cpus (affinity)", len(os.sched_getaffinity(0)), flush=True)
PYEOF

T0=$SECONDS
bash scripts/prepare_data.sh
echo "== prepared data in $((SECONDS - T0))s"
need_gpu() {  # UCF training/inference needs CUDA; scoring finished artifacts does not
  python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA unavailable: UCF training needs a GPU')"
}
have() {  # have <method>: both experiments' artifacts exist
  [ -f "outputs/$1/predictions.npz" ] && [ -f "outputs/$1/predictions_visa.npz" ]
}

have anomaldf || { need_gpu; bash scripts/baseline.sh; }
echo "== baseline done at $((SECONDS - T0))s"
[ -f method/ucf_setup.json ] || { need_gpu; python -m method.select_setup; }
# reference: seed 0 on both datasets; the epoch-10 snapshot is the study's
# "quarter of the schedule" weaker variant (written outside outputs/)
have anomaldf_ucf || { need_gpu; SEED=0 UCF_EXTRA_ARGS="--snapshot-epochs 10 --snapshot-dir redteam/final/outputs" bash scripts/method.sh; }
echo "== reference UCF done at $((SECONDS - T0))s"

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
echo "== reference scored at $((SECONDS - T0))s"

bash redteam/final/run_study.sh
echo "== study done at $((SECONDS - T0))s"

python validate.py --compare --post-design || echo "!! validate reported errors"
