# Evaluation Targets

## Primary: rgb_mvtec_multiclass

Tests whether a multi-class RGB anomaly detector improves image-level detection and pixel-level localization under a normal-only training protocol.

- **Metric:** `pixel_auroc` (higher is better)
- **Secondary benefit:** `image_auroc` (higher is better)
- **Constraint:** `ucf_params_M` (lower is better)
- **How to run:** execute `bash scripts/reproduce.sh`, or evaluate an existing artifact with `bash scripts/evaluate.sh outputs/ucf_synthetic/predictions.npz`.
