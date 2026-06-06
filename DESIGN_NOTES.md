# Design Notes

## Checks Performed

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side for train/test path copy-paste errors.
- Checked `eval/train/evaluate.py` and `eval/test/evaluate.py` for method detachment, cross-imports, and output schema mismatch.
- Checked `scoring/reference.json` for paper-agreement metrics, runtime metrics, and missing required metric fields.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.
- Ran `bash -n` on both train/test evaluation scripts.

## Changes Made

- Fixed a self-containment problem in the held-out test slice: `scripts/evaluate_test.sh` and `eval/test/evaluate.py` no longer require `scoring/scores_train.json`. The test evaluator now scores the same prediction artifact interface independently.
- Added metric triangulation to both train and test slices:
  - `pixel_ap` for sparse-pixel precision-recall behavior.
  - `pixel_aupro_5pct` for stricter low-false-positive localization.
  - `category_mean_image_auroc` and `category_mean_pixel_auroc` to reduce category-frequency shortcutting.
- Completed required metadata in `scoring/reference.json` by adding descriptions for every nonzero metric.
- Changed missing method-parameter metadata handling from a low default to a conservative paper-scale default (`43.0` million parameters). This prevents future methods from avoiding the parameter constraint by omitting metadata while still allowing explicit `params_m`, `ucf_params_M`, `method_params_M`, or `added_params_M` fields.

## Runtime Concerns For Runner Iteration

- The train/test split is deterministic and stratified by category and anomaly label, but it is still a half-split of one synthetic test artifact. If the held-out scores are noisy, prefer increasing generated test samples in the method/data stage over changing the split procedure.
- The strict `pixel_aupro_5pct` metric may be more volatile than full `pixel_aupro` on very small anomaly masks. Watch for all-zero or near-identical values when real scores are produced.
- The legacy `scripts/evaluate.sh` path was out of scope for this design pass and still writes the original unsplit `scoring/scores.json`; train/test scoring should use `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`.
- Evaluation remains artifact-based: future methods must write `predictions.npz` with labels, categories, indices, masks, baseline scores/maps, and fused scores/maps. If a future method does not use UCF, it can still populate the fused arrays with its final predictions.

## Completed Runner Check

- Job `1ffa5b29-c1f` ran `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` successfully on CPU and wrote `scoring/scores_train.json` plus `scoring/scores_test.json`.
- The score files contain the same method identifiers (`anomaldf`, `anomaldf_ucf`, `ucf_synthetic_seed13`) and all 10 nonzero reference metrics for each method on both slices.
- The train/test files are not byte-identical. For the final `anomaldf_ucf` artifact, primary `pixel_auroc` is 77.5035 on train and 72.7855 on test; the baseline is 73.2767 on train and 69.3723 on test.
- The strict low-FPR localization metrics are low but nonzero rather than saturated: `anomaldf_ucf` has `pixel_aupro_5pct` 5.1266 on train and 3.0399 on test.
- The UCF parameter constraint uses the conservative 43.0M fallback because the existing prediction artifacts do not report parameter metadata.

## Fixed Patterns

- Fixed hidden-test dependency on visible train scores.
- Fixed missing metric descriptions in the reference schema.
- Fixed a parameter-constraint shortcut caused by rewarding omitted parameter metadata.
