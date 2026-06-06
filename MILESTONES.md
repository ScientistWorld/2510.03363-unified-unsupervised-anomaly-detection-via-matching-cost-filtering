# Reproduction Milestones

**Current: core_claim**

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

## Stop Justification
- Current workspace is being extended toward `core_claim_plus` with an additional deterministic seed run.
