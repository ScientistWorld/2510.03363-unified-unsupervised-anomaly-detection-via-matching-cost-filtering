#!/bin/bash
set -euo pipefail
cd /home/user
# Limit threads for the scheduler's single allocated CPU. No GPU code runs here.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m redteam.snapshot
while read -r case_name; do
  if [ ! -f "redteam/$case_name/status.json" ]; then
    bash "redteam/$case_name/run.sh"
  fi
done < <(python -c 'import json; print("\n".join(json.load(open("redteam/cases.json"))))')
python -m redteam.summarize
