# Progress

## What Works

Paper analysis is complete for the core UCF pipeline. The workspace now contains a compact RGB reproduction with deterministic procedural anomaly data, UCF cost-volume construction, guided 3D filtering, training, artifact-writing, and method-agnostic evaluation.

`python validate.py --reference-only` passes. A local tiny forward pass was attempted, but the login environment does not have NumPy installed; dependencies are declared in `environment/setup.sh` and are installed in the job container.

The first compute job successfully entered training and the loss decreased from 1.3078 to 0.2122 over six epochs, but the CPU test job timed out before evaluation. This indicates the implementation is structurally valid but the initial compact scale was still too slow for a one-hour CPU smoke run.

The second compute job completed training and wrote prediction artifacts plus an initial `scores.json`, then failed in diagnostic AUPRO reporting because NumPy 2 no longer exposes `np.trapz`. The bug is fixed by using `np.trapezoid` with a fallback. The preliminary 4-epoch scores were image AUROC 51.7 to 52.7 but pixel AUROC 71.1 to 69.9, so UCF did not yet support the core claim.

The third compute job completed end-to-end. At the compact synthetic scale, UCF improved the primary localization metric from 71.1 pixel AUROC to 75.4 pixel AUROC and improved diagnostic AUPRO/F1, with a 0.16M-parameter overhead. This reaches `core_claim` for a reduced RGB anomaly-localization setting.

The fourth compute job used the final 40-epoch default, matching the paper's epoch count at the compact scale. It completed and validated cleanly. Pixel AUROC remained above the raw matching-cost baseline at 74.9 versus 71.1, while image AUROC improved slightly to 52.5. The 24-epoch run had a marginally better pixel AUROC, but the final packaged default is the paper-aligned 40-epoch schedule.

The 2026-06-06 continuation audit restored `core_claim` after the milestone reset. The method code is identifiable as UCF rather than a surrogate: it builds hierarchical patch-template cosine cost volumes, converts them to anomaly costs, uses min-pooled matching maps as the raw baseline, filters the 3D volume with residual channel/spatial attention and image/map guidance, trains with focal, class, soft-IoU, and SSIM losses, and writes prediction artifacts consumed by an evaluator that does not import `method/`.

## Results

Successful compact reproduction results are written to `scoring/scores.json`:

- `anomaldf` baseline: image AUROC 51.7, pixel AUROC 71.1, UCF parameters 0.00M.
- `anomaldf_ucf`: image AUROC 52.5, pixel AUROC 74.9, UCF parameters 0.16M.

## Remaining

To go beyond `core_claim`, the next meaningful step would be replacing the compact procedural feature source with cached DINOv2/AnomalDF features on MVTec-AD or a closer public industrial anomaly dataset. The current environment is complete and usable for the reduced UCF benchmark.

## Issues

The exact MVTec-AD dataset and DINOv2/AnomalDF backbone are not used in this first pass to avoid large downloads and GPU-heavy feature extraction. The reproduction instead targets the UCF algorithm itself in a compact RGB setting.

The bare login Python still lacks NumPy, so `scripts/evaluate.sh` cannot run outside the configured container. This is not a packaging failure: `environment/setup.sh` installs NumPy and scikit-learn, and completed compute jobs have run the evaluator and `validate.py --compare` successfully inside the container.

## Deviations from Paper

- The paper evaluates on MVTec-AD with DINOv2/AnomalDF features and normal-image templates. This reproduction uses a deterministic procedural multi-class RGB anomaly dataset and a frozen handcrafted feature pyramid. This is a scale/resource substitution for the feature source and data, not a replacement of the UCF computation: the implementation still builds hierarchical patch-template cosine cost volumes, applies guided 3D cost filtering, trains on synthetic anomaly masks, and evaluates baseline versus UCF maps.
- The paper trains for 40 epochs with batch size 8. The final compact default now uses 40 epochs and batch size 8, but the procedural split and feature/cost-volume scale remain reduced to fit the CPU test-node limit. The script keeps epochs configurable through `EPOCHS`.
