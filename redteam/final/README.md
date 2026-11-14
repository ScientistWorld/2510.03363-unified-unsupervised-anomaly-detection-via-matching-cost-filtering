# Final hardening study (both experiments)

`run_study.sh` is run by `scripts/run.sh` after the reference. It trains and scores the study rows with the gym's own evaluators:

- training seeds;
- better and weaker variants;
- the paper's N=3 random-template inference;
- lambda sensitivity;
- shortcut payloads (`shortcuts.py`, `dev_supervised_shortcut.py`).

`summarize.py --bootstrap` writes `summary.{json,md}`: ranks, seed spread, and paired category-stratified bootstrap intervals on the hidden slices.

- Prediction artifacts go to `outputs/` (gitignored).
- Per-row score files go to `scores/` (tracked).
- Results are discussed in `HARDENING.md` at the workspace root.

`redteam/hardening/` is the previous pass's study, run under the earlier N=3, MVTec-only protocol. Its summary is the "before" record. Its scripts target that protocol and are not run any more.
