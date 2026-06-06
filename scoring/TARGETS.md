# Evaluation Targets

## Primary: rgb_mvtec_multiclass

Tests whether a multi-class RGB anomaly detector improves image-level detection and pixel-level localization under a normal-only training protocol.

- **Metric:** `pixel_auroc` (higher is better)
- **Secondary benefits:** `image_auroc`, `image_ap`, `image_f1_max`, `pixel_ap`, `pixel_f1_max`, `pixel_aupro`, `pixel_aupro_5pct`, `category_mean_image_auroc`, and `category_mean_pixel_auroc` (higher is better)
- **Constraints:** `added_params_M` and `added_flops_G` (lower is better)
- **How to run:** execute `bash scripts/reproduce.sh`, or evaluate an existing artifact with `bash scripts/evaluate.sh outputs/ucf_synthetic/predictions.npz`.

Optional robustness results may be written to generated sidecar files, but the canonical validation target is the single experiment in `scoring/scores.json`.

The paper's broader plug-in/generalization claim spans RGB, RGB-3D, and RGB-Text settings. This compact gym targets the RGB core claim only; cross-modal reproduction is intentionally out of scope for the current milestone.
