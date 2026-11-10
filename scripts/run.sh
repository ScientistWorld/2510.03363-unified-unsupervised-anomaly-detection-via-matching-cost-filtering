#!/bin/bash
# Job entry point: prepare data, reproduce the reference end to end (AnomalDF
# baseline, dev-slice setup selection, AnomalDF+UCF seed 0 with the paper's
# fused inference), score it on both slices, then run the hardening study
# (seeds, better/weaker variants, lambda sensitivity, brute force, shortcuts,
# evaluation-contract checks). Every stage is resumable: finished artifacts
# are kept, so a rerun only rescores; CUDA is required only when something
# has to be trained or predicted.
set -euo pipefail
cd /home/user

python - <<'EOF'
import torch, numpy, scipy, sklearn, PIL
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), flush=True)
EOF

T0=$SECONDS
bash scripts/prepare_data.sh
echo "== prepared data in $((SECONDS - T0))s"
need_gpu() {  # UCF training/inference needs CUDA; scoring finished artifacts does not
  python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA unavailable: UCF training needs a GPU')"
}

[ -f outputs/anomaldf/predictions.npz ] || { need_gpu; bash scripts/baseline.sh; }
echo "== baseline done at $((SECONDS - T0))s"
[ -f method/ucf_setup.json ] || { need_gpu; python -m method.select_setup; }
echo "== training setup selected on the dev slice at $((SECONDS - T0))s"
[ -f outputs/anomaldf_ucf/predictions.npz ] || { need_gpu; SEED=0 bash scripts/method.sh; }
echo "== reference UCF done at $((SECONDS - T0))s"

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
echo "== reference scored at $((SECONDS - T0))s"

bash redteam/hardening/run_study.sh
echo "== study done at $((SECONDS - T0))s"

python validate.py --compare --post-design || echo "!! validate reported errors"
