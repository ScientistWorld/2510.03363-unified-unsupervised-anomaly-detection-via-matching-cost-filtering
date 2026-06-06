# Progress

## What Works

Paper analysis is complete for the core UCF pipeline. The workspace now contains a compact RGB reproduction with deterministic procedural anomaly data, UCF cost-volume construction, guided 3D filtering, training, artifact-writing, and method-agnostic evaluation.

`python validate.py --reference-only` passes. A local tiny forward pass was attempted, but the login environment does not have NumPy installed; dependencies are declared in `environment/setup.sh` and are installed in the job container.

The first compute job successfully entered training and the loss decreased from 1.3078 to 0.2122 over six epochs, but the CPU test job timed out before evaluation. This indicates the implementation is structurally valid but the initial compact scale was still too slow for a one-hour CPU smoke run.

The second compute job completed training and wrote prediction artifacts plus an initial `scores.json`, then failed in diagnostic AUPRO reporting because NumPy 2 no longer exposes `np.trapz`. The bug is fixed by using `np.trapezoid` with a fallback. The preliminary 4-epoch scores were image AUROC 51.7 to 52.7 but pixel AUROC 71.1 to 69.9, so UCF did not yet support the core claim.

The third compute job completed end-to-end. At the compact synthetic scale, UCF improved the primary localization metric from 71.1 pixel AUROC to 75.4 pixel AUROC and improved diagnostic AUPRO/F1, with a 0.16M-parameter overhead. This reaches `core_claim` for a reduced RGB anomaly-localization setting.

The fourth compute job used the final 40-epoch default, matching the paper's epoch count at the compact scale. It completed and validated cleanly. Pixel AUROC remained above the raw matching-cost baseline at 74.9 versus 71.1, while image AUROC improved slightly to 52.5. The 24-epoch run had a marginally better pixel AUROC, but the final packaged default is the paper-aligned 40-epoch schedule.

The 2026-06-06 continuation audit restored `core_claim` after the milestone reset. The method code is identifiable as UCF rather than a surrogate: it builds hierarchical patch-template cosine cost volumes, converts them to anomaly costs, uses min-pooled matching maps as the raw baseline, filters the 3D volume with residual channel/spatial attention and image/map guidance, trains with focal, class, soft-IoU, and SSIM losses, and writes prediction artifacts consumed by an evaluator that does not import `method/`.

The attempted additional-seed run on 2026-06-06 trained both the default split and `SEED=13`. The extra seed also supported the claim, improving image AUROC from 51.7 to 60.0 and pixel AUROC from 71.1 to 76.6. The job failed only at the final validation step because `scoring/scores.json` contained an extra experiment not present in the paper-derived `reference.json`. `scripts/run.sh` now writes combined robustness results to `scoring/robustness_scores.json` and restores canonical `scores.json` to the reference experiment before running `validate.py --compare`.

The corrected `b5f79477-1df` job completed with exit code 0. It reran the default setting, reran `SEED=13`, wrote robustness metrics separately, restored canonical `scores.json`, and passed `python validate.py --compare`. This reaches `core_claim_plus` for the compact benchmark.

The 2026-06-06 packaging audit tightened two issues before final handoff: `scripts/reproduce.sh` now runs the method from scratch before evaluation, and tracked generated `jobs/` and `trajectory/` artifacts were removed from the git index so ignored run logs do not ship. The canonical evaluator was also extended to emit all diagnostic metrics declared in `reference.json` when run inside the packaged environment.

Final verification on 2026-06-06 kept the milestone at `core_claim_plus` and tightened gym usability: from-scratch-visible docs were made method-agnostic, the parameter constraint was renamed to `added_params_M`, train/test evaluators now score any conforming prediction artifact, and compact train/test parameter scores were corrected to 0.16M.

The 2026-06-06 audit pass found a UCF fidelity issue: the compact implementation flattened matching positions and feature layers into one 3D depth axis, while the paper uses matching/template positions as channels and feature layers as the 3D depth. The method was patched to preserve a `[batch, matching_channels, feature_layers, height, width]` cost-volume layout, and the `4c0da269-693` CPU job re-trained and re-scored this corrected implementation successfully.

Final verification found that the previous prediction artifact was produced before the corrected metadata writer and expanded metric schema. The evaluator now rejects submitted artifacts that omit `added_params_M` or `added_flops_G`, and train/test wrappers discover all conforming `outputs/*/predictions.npz` artifacts by default. The final revalidation job regenerated `outputs/ucf_synthetic`, `scoring/scores.json`, `scoring/scores_train.json`, and `scoring/scores_test.json` from the current code and passed `python validate.py --compare`.

## Results

Successful compact reproduction results are written to `scoring/scores.json`:

- `anomaldf` baseline: image AUROC 54.6, pixel AUROC 70.7, added parameters/FLOPs 0.00M/0.000G.
- `anomaldf_ucf`: image AUROC 53.0, pixel AUROC 74.9, added parameters/FLOPs 0.224M/0.064G.
- UCF also improved pixel AP from 3.66 to 4.11, pixel F1 from 7.17 to 8.42, AUPRO from 29.21 to 32.12, and category-mean pixel AUROC from 70.75 to 74.84 in the final corrected-layout revalidation job `4c0da269-693`.

Additional robustness result, kept outside canonical `scores.json` in generated `scoring/robustness_scores.json`: with `SEED=13`, UCF improved image AUROC from 51.7 to 60.0 and pixel AUROC from 71.1 to 76.6.

## Remaining

To go beyond `core_claim_plus`, the next meaningful step would be replacing the compact procedural feature source with cached DINOv2/AnomalDF features on MVTec-AD or a closer public industrial anomaly dataset. The current environment is complete and usable for the reduced UCF benchmark.

## Issues

The exact MVTec-AD dataset and DINOv2/AnomalDF backbone are not used in this first pass to avoid large downloads and GPU-heavy feature extraction. The reproduction instead targets the UCF algorithm itself in a compact RGB setting.

The bare login Python still lacks NumPy, so `scripts/evaluate.sh` cannot run outside the configured container. This is not a packaging failure: `environment/setup.sh` installs NumPy and scikit-learn, and completed compute jobs have run the evaluator and `validate.py --compare` successfully inside the container.

The final corrected-layout job improved the primary localization metrics but reduced image-level AUROC and image AP versus the raw matching-cost baseline. This keeps the milestone at `core_claim_plus` rather than moving toward broader secondary claims.

The 2026-06-06 `af7cf417-13b` job exited with code 1 because the validator rejected an extra `rgb_synthetic_seed13` experiment in `scores.json`. The experiment outputs were valid; the packaging contract is fixed by storing extra-seed metrics in `scoring/robustness_scores.json` and leaving canonical `scores.json` aligned with `reference.json`.

The login-node dependency gap prevents local artifact rescoring with NumPy, but final verification did not require another training run. `python validate.py --compare` and `python -m compileall eval method data baseline` pass on the login node for structural and syntax checks.

## Deviations from Paper

- The paper evaluates on MVTec-AD with DINOv2/AnomalDF features and normal-image templates. This reproduction uses a deterministic procedural multi-class RGB anomaly dataset and a frozen handcrafted feature pyramid. This is a scale/resource substitution for the feature source and data, not a replacement of the UCF computation: the implementation still builds hierarchical patch-template cosine cost volumes, applies guided 3D cost filtering, trains on synthetic anomaly masks, and evaluates baseline versus UCF maps.
- The paper trains for 40 epochs with batch size 8. The final compact default now uses 40 epochs and batch size 8, but the procedural split and feature/cost-volume scale remain reduced to fit the CPU test-node limit. The script keeps epochs configurable through `EPOCHS`.
