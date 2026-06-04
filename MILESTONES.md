# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-04 10:38] - core_claim
- Implemented the paper's UCF computation at compact RGB scale: hierarchical template matching cost volumes, guided 3D filtering with residual channel/spatial attention, class-aware segmentation loss, and fused anomaly maps.
- Completed an end-to-end compute job and wrote `scoring/scores.json`.
- UCF improved pixel AUROC from 71.114 to 74.942 over the direct matching-cost baseline on the procedural multi-class RGB anomaly benchmark, with 0.16M additional parameters.

## Stop Justification
- Completed at milestone `core_claim`.
