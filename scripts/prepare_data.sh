#!/bin/bash
# Build the 224px arrays from the archives fetched by scripts/download.sh.
# Runs inside the container (no internet needed); idempotent.
#   data/downloads/mvtec224/{train,test_inputs}.npz   method-visible MVTec-AD inputs
#   data/downloads/visa224/{train,test_inputs}.npz    method-visible VisA inputs
#   data/downloads/dtd/dtd224.npy                     anomaly-synthesis textures
#   eval/train/data/{dev,visa_dev}_answers.npz        visible development answers
#   eval/test/downloads/{test,visa_test}_answers.npz  hidden test answers
set -euo pipefail
cd "$(dirname "$0")/.."
python -m eval.test.prepare_mvtec
python -m eval.test.prepare_visa
python -m data.prepare_dtd
