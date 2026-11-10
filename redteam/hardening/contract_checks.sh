#!/bin/bash
# Evaluation-contract checks for the fixes of this hardening pass, each run
# against the evaluator/scripts BEFORE the fix (git revision 875f845, copied to
# redteam/hardening/contract/before_fix/) and AFTER
# (the working tree). Every payload is built from method-visible material only.
#
#   absent_dir      the proposed method's directory does not exist at all (a run
#                   that crashed before creating outputs/<method>/). Before: the
#                   method silently dropped out of scores_test.json (an empty
#                   entry costs nothing in the rank). After: it is scored failed.
#   name_collision  a second directory that sanitises to the same method key
#                   ('anomaldf-ucf' next to 'anomaldf_ucf'). Before: the
#                   evaluator exited before writing any score for any method.
#                   After: every artifact is scored; the exact-name directory
#                   keeps the key.
#   oversized_header  anomaly_maps declares 2,000,000 x 224 x 224 float32
#                   (~400 GB) in its .npy header. After: rejected from the
#                   header before any data is read.
#
# Results: redteam/hardening/contract/{before,after}_<check>.json plus the
# exit codes in redteam/hardening/contract/summary.json.
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
H=redteam/hardening
C=$H/contract
W=tmp/contract_checks
rm -rf "$W"
mkdir -p "$C" "$W"
REF_PRED=$ROOT/outputs/anomaldf_ucf/predictions.npz
BASE_PRED=$ROOT/outputs/anomaldf/predictions.npz
BAD_PRED=$ROOT/$H/outputs/shortcut_constant_outputs/predictions.npz

declare -A RC

# Scratch workspace: same layout as the repo root, with eval/ (and its answers)
# linked in and the evaluator of the requested revision.
make_ws() {  # make_ws <dir> <before|after>
  local ws=$1 which=$2
  mkdir -p "$ws/scripts" "$ws/scoring" "$ws/outputs"
  cp scoring/reference.json scoring/scores.json "$ws/scoring/"
  if [ "$which" = before ]; then
    # pre-fix evaluate_test.sh and evaluator (git revision 875f845, extracted to
    # $H/contract/before_fix/); answers linked from the real eval/test/downloads
    cp "$H/contract/before_fix/scripts/evaluate_test.sh" "$ws/scripts/evaluate_test.sh"
    cp -r "$H/contract/before_fix/eval" "$ws/eval"
    ln -s "$ROOT/eval/test/downloads" "$ws/eval/test/downloads"
  else
    cp scripts/evaluate_test.sh "$ws/scripts/evaluate_test.sh"
    ln -s "$ROOT/eval" "$ws/eval"
  fi
}

# ---- absent_dir: only the baseline directory exists; anomaldf_ucf is named in scoring/ but absent.
for which in before after; do
  ws=$W/absent_$which
  make_ws "$ws" "$which"
  mkdir -p "$ws/outputs/anomaldf"
  ln -s "$BASE_PRED" "$ws/outputs/anomaldf/predictions.npz"
  (cd "$ws" && bash scripts/evaluate_test.sh) > "$C/${which}_absent_dir.log" 2>&1
  RC[${which}_absent_dir]=$?
  [ -f "$ws/scoring/scores_test.json" ] && cp "$ws/scoring/scores_test.json" "$C/${which}_absent_dir.json"
done

# ---- name_collision: exact-name dir holds a failing artifact, a look-alike holds the reference's.
for which in before after; do
  ws=$W/collide_$which
  make_ws "$ws" "$which"
  mkdir -p "$ws/outputs/anomaldf_ucf" "$ws/outputs/anomaldf-ucf"
  ln -s "$BAD_PRED" "$ws/outputs/anomaldf_ucf/predictions.npz"
  ln -s "$REF_PRED" "$ws/outputs/anomaldf-ucf/predictions.npz"
  (cd "$ws" && bash scripts/evaluate_test.sh outputs/anomaldf_ucf/predictions.npz outputs/anomaldf-ucf/predictions.npz) > "$C/${which}_name_collision.log" 2>&1
  RC[${which}_name_collision]=$?
  [ -f "$ws/scoring/scores_test.json" ] && cp "$ws/scoring/scores_test.json" "$C/${which}_name_collision.json"
done

# ---- oversized_header: valid ids/scores, maps header claiming ~400 GB.
python - "$W/oversized/predictions.npz" <<'EOF'
import io, sys, zipfile
from pathlib import Path
import numpy as np
from numpy.lib import format as npy_format
from data.mvtec import load_test_inputs
out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
_, _, ids = load_test_inputs()
def npy(arr):
    b = io.BytesIO(); np.save(b, arr); return b.getvalue()
hdr = io.BytesIO()
npy_format.write_array_header_1_0(hdr, {"descr": "<f4", "fortran_order": False, "shape": (2_000_000, 224, 224)})
with zipfile.ZipFile(out, "w") as z:
    z.writestr("ids.npy", npy(ids.astype(np.int64)))
    z.writestr("image_scores.npy", npy(np.zeros(ids.size, np.float32)))
    z.writestr("anomaly_maps.npy", hdr.getvalue() + b"\0" * 4096)
print("wrote", out)
EOF
for which in before after; do
  ws=$W/oversized_$which
  make_ws "$ws" "$which"
  mkdir -p "$ws/outputs/oversized_header"
  ln -s "$ROOT/$W/oversized/predictions.npz" "$ws/outputs/oversized_header/predictions.npz"
  (cd "$ws" && bash scripts/evaluate_test.sh outputs/oversized_header/predictions.npz) > "$C/${which}_oversized_header.log" 2>&1
  RC[${which}_oversized_header]=$?
  [ -f "$ws/scoring/scores_test.json" ] && cp "$ws/scoring/scores_test.json" "$C/${which}_oversized_header.json"
done

{
  echo "{"
  first=1
  for k in "${!RC[@]}"; do
    [ $first -eq 1 ] || echo ","
    first=0
    printf '  "%s_exit_code": %s' "$k" "${RC[$k]}"
  done
  echo
  echo "}"
} > "$C/summary.json"
cat "$C/summary.json"
for f in "$C"/*.json; do
  [ "$(basename "$f")" = summary.json ] && continue
  python - "$f" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))["experiments"]["rgb_mvtec_multiclass"]
for name, status in d["_validity"].items():
    r = d["results"][name]
    print(f"{sys.argv[1].split('/')[-1]:32s} {name:22s} {status[:60]:60s} I-AUROC={r['image_auroc']:.2f} P-AUROC={r['pixel_auroc']:.2f}")
EOF
done
rm -rf "$W"
