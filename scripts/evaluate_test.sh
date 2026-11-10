#!/bin/bash
# Hidden test: score every method on disk against the answers held in
# eval/test/ (rebuilt from the archive if absent). The methods are every
# directory under outputs/ (outputs/<method>/predictions.npz) plus every method
# named in scoring/reference.json or scoring/scores.json. A method without a
# valid artifact - including one whose directory is missing - is scored as
# failed (worst value on every metric), never skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f eval/test/downloads/test_answers.npz ]; then
  python -m eval.test.prepare_mvtec
fi
if [ "$#" -eq 0 ]; then
  artifacts=()
  while IFS= read -r name; do
    artifacts+=("outputs/$name/predictions.npz")
  done < <(
    {
      for d in outputs/*/; do [ -d "$d" ] && basename "$d"; done
      python - <<'EOF'
import json, re
from pathlib import Path
for f in ("scoring/reference.json", "scoring/scores.json"):
    try:
        exps = json.loads(Path(f).read_text()).get("experiments", {})
    except (OSError, ValueError, AttributeError):
        continue
    for exp in exps.values() if isinstance(exps, dict) else ():
        results = exp.get("results") if isinstance(exp, dict) else None
        for name in results if isinstance(results, dict) else ():
            if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", str(name)):
                print(name)
EOF
    } | sort -u
  )
  if [ "${#artifacts[@]}" -eq 0 ]; then
    echo "No methods found under outputs/ or in scoring/" >&2
    exit 1
  fi
  set -- "${artifacts[@]}"
fi

python -m eval.test.evaluate --predictions "$@" --scores scoring/scores_test.json
