# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-04 00:00] - none
- Read the paper and identified UCF's core computational steps: hierarchical feature extraction, patch-template cost-volume construction, guided 3D filtering, class-aware loss, and top-k anomaly scoring.
- Implemented a compact RGB reproduction environment and staged it for the first compute-node run.

### [2026-06-04 00:00] - none
- First CPU job began training and showed steadily decreasing loss, but timed out before inference/evaluation.
- Reduced procedural split size, cost-volume resolution, and default smoke epochs so the next job can complete end-to-end.

### [2026-06-04 10:23] - none
- Second CPU job completed compact UCF training and wrote prediction artifacts plus preliminary scores.
- The run failed during diagnostic AUPRO reporting because NumPy 2 removed `np.trapz`; this is fixed in the evaluator.
- Preliminary four-epoch scores did not yet support the core claim, so the next job increases training to 24 epochs.

### [2026-06-04 10:28] - core_claim
- Third CPU job completed `scripts/run.sh` end-to-end with validation passing.
- Compact UCF improved primary pixel AUROC from 71.1 to 75.4 and improved localization diagnostics over the raw matching-cost baseline.
- Constraint metric is included: UCF adds 0.16M parameters at the reduced scale.

## Stop Justification
- Not stopped. A final 40-epoch compact job is being submitted to improve the reproduced score while preserving the paper's training schedule.
