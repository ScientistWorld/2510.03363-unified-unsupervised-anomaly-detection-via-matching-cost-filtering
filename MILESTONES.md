# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-04 06:29] - core_claim
- Completed the compact RGB UCF reproduction end-to-end on the compute node.
- The reproduced UCF run improved pixel AUROC over the raw matching-cost baseline from 71.1 to 74.9 while preserving the same frozen template feature source.

### [2026-06-06 21:37] - core_claim
- Audited the inherited workspace after milestone reset and restored `core_claim`.
- Verified the implementation contains the paper's cost-volume construction, guided 3D filtering, class-adaptor loss path, baseline map, and artifact-based evaluator.

### [2026-06-06 21:52] - core_claim
- Reviewed failed `af7cf417-13b` job. Both default and `SEED=13` UCF runs trained and the extra seed improved pixel AUROC from 71.1 to 76.6, but validation failed because an extra non-reference experiment was written to canonical `scores.json`.
- Fixed `scripts/run.sh` so robustness scores are written separately and canonical `scores.json` is restored before validation.

### [2026-06-06 22:07] - core_claim_plus
- Reran the corrected packaged job `b5f79477-1df` successfully through `validate.py --compare`.
- Canonical `scores.json` remains valid for the paper-referenced experiment, and the generated robustness sidecar confirms the UCF gain under `SEED=13`: image AUROC 51.7 to 60.0 and pixel AUROC 71.1 to 76.6.

### [2026-06-06 22:32] - core_claim_plus
- Fixed packaging audit issues: `scripts/reproduce.sh` now executes the method before evaluation, and generated `jobs/`/`trajectory/` artifacts were removed from the git index.
- Expanded canonical artifact evaluation to report AP, F1, AUPRO, stricter AUPRO, and category-mean AUROC metrics in addition to the primary AUROC metrics.

### [2026-06-06 22:37] - core_claim_plus
- Final verification passed after tightening method-agnostic documentation, generic scoring metric names, and train/test evaluation support for future method variants.
- `python validate.py --compare` and Python compilation checks passed.

### [2026-06-06 22:46] - core_claim_plus
- Audit-driven fidelity patch prepared: UCF cost volumes now preserve matching positions as channels and feature layers as 3D depth, closer to the paper's filtering input.
- Updated scoring constraints to include paper-correct `+43.8M` AnomalDF UCF parameters and `+32.7G` FLOPs, added image F1, and made `scripts/run.sh` regenerate canonical outputs before validation.
- Submitting a compute job to re-train and re-score the corrected compact implementation before finalizing this state.

### [2026-06-06 23:10] - core_claim_plus
- Final verification tightened evaluator reuse and constraint scoring: train/test wrappers now enumerate prediction artifacts by default, and evaluators require `added_params_M` plus `added_flops_G` instead of assigning zero overhead to incomplete artifacts.
- The existing generated prediction artifact is stale relative to the current method metadata writer, so final sign-off is deferred until a CPU revalidation job regenerates scores from the current code.

### [2026-06-06 23:35] - core_claim_plus
- CPU job `4c0da269-693` completed with exit code 0, regenerated current-code prediction artifacts and all score files, and passed `python validate.py --compare`.
- Corrected-layout UCF improved the primary pixel AUROC from 70.7 to 74.9 and improved pixel AP, pixel F1, AUPRO, and category-mean pixel AUROC, with measured compact overhead of 0.224M parameters and 0.064G FLOPs.
- Image-level AUROC/AP did not improve on the compact procedural split, so the final milestone remains `core_claim_plus`.

## Stop Justification
- Completed final verification at milestone `core_claim_plus`. The compact gym is usable and validated, but it intentionally covers a reduced RGB procedural setting rather than the full MVTec/DINOv2-AnomalDF evaluation suite.
