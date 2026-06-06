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

## Stop Justification
- Completed at milestone `core_claim_plus`.
