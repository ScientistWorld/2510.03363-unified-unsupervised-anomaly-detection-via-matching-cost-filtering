#!/bin/bash
set -euo pipefail

cd /home/user

# The baseline map is produced together with UCF predictions because both use
# the same saved cost-volume run. Run a short one-epoch pass if no artifact
# exists, then evaluate the baseline entries in the shared output file.
if [ ! -f outputs/ucf_synthetic/predictions.npz ]; then
  EPOCHS=1 bash scripts/method.sh
else
  bash scripts/evaluate.sh outputs/ucf_synthetic/predictions.npz
fi
