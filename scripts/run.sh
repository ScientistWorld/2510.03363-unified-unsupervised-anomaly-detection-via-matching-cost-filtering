#!/bin/bash
set -euo pipefail

cd /home/user

bash scripts/download.sh
EPOCHS="${EPOCHS:-4}" BATCH_SIZE="${BATCH_SIZE:-8}" bash scripts/method.sh
python validate.py --compare
