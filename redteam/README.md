Audit run entry point: `bash redteam/run_all.sh` in the packaged compute container.
Each named folder has a `run.sh` that prepares its submission, exits that process,
and invokes exactly `bash scripts/evaluate_test.sh` with no arguments. The
unchanged evaluator sees one persisted `outputs/ucf_synthetic/predictions.npz`.
All original outputs and score files are restored after each attempt.

Payloads use scientist-visible method/data/train-evaluation sources only; none
imports the hidden evaluator. `snapshot.py` and `summarize.py` are auditor-only
bookkeeping. Reference predictions reused by shortcuts come from the newly
trained seed-7 run, not from hidden cached predictions.

`scores_test.json`, `scores_train.json`, `status.json`, `summary.json`, and
`summary.csv` are evidence. Prediction/checkpoint artifacts and verbose logs are
ignored and can be regenerated. Resource negative/NaN payloads are deliberately
impossible self-reports. They are shortcuts, not valid resource reductions.

Ranks use the absolute coefficient, the declared better direction, and one
population standard deviation per metric. The scientific cohort excludes all
shortcuts. The complete finite audit cohort is illustrative: later scientists'
attempts will change those denominators. Exact ties and Pareto improvements are
reported separately because they do not depend on those unknown denominators.
