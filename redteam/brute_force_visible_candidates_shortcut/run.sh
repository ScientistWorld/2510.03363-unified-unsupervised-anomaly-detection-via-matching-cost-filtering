#!/bin/bash
set -euo pipefail
cd /home/user
python -m redteam.run_case brute_force_visible_candidates_shortcut
