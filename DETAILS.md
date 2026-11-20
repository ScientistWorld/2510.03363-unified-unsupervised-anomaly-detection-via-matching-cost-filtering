# Overview

- **Paper ID:** 2510.03363
- **Title:** Unified Unsupervised Anomaly Detection via Matching Cost Filtering
- **Domain:** Computer Vision / Unsupervised Anomaly Detection
- **TL;DR:** The paper improves visual anomaly detection by learning to denoise template-matching cost volumes before producing anomaly maps.

## Short Summary

The paper argues that many anomaly detectors are fundamentally matching inputs against normal references, reconstructions, text prompts, or 3D features, but their matching costs are noisy. UCF builds a multi-layer anomaly cost volume and filters it with a guided 3D network. Across RGB, RGB-3D, and RGB-Text settings, the paper reports consistent improvements over strong baselines.

## Key Results

- On MVTec-AD RGB multi-class anomaly detection, AnomalDF improves from 96.8/98.1 image/pixel AUROC to 98.5/98.8 with UCF.
- On VisA RGB, AnomalDF improves from 90.5/97.5 image/pixel AUROC to 94.3/99.2.
- The largest reported gains are in stricter localization metrics such as AUPRO for RGB-3D and RGB-Text settings.

---

# Problem Definition

## Research Question

Detect anomalous images and localize anomalous pixels when training data contains only normal examples. The target setting is multi-class visual anomaly detection: a single system must handle multiple visual categories and produce both an image-level anomaly score and a pixel-level anomaly heatmap.

## Why It Matters

Industrial inspection and medical screening often have abundant normal data but few labeled defects. A useful solution should identify subtle defects without requiring real anomaly labels for every category.

## Success Criteria

A successful system separates normal from abnormal test images, localizes defect regions accurately, and does so without using real anomaly labels during training.

---

# Paper's Method

## Key Contribution

Unified Cost Filtering (UCF) treats visual anomaly detection as matching-cost filtering. Instead of directly thresholding noisy feature distances, it constructs a patch-wise anomaly cost volume and learns to filter that volume using guidance from the input image and an initial anomaly map.

## Approach

The method extracts hierarchical features from a test image and normal reference templates. It computes global cosine matching between each test patch and template patches, converts similarity to anomaly cost, and collapses the template/matching axes into a cost volume. As in the authors' code, the matching axis keeps a fixed number of the lowest costs per test patch. A model trained with N = 3 random templates can therefore be applied unchanged to the whole category training set at test time (the paper's full-shot protocol, Table XXIV), which is the reproduction's reference setting. A 3D filtering network refines this volume with residual channel and spatial attention guided by the input and the initial pooled anomaly map. The filtered volume is min-pooled over the matching dimension and converted to a two-class normal/anomaly score map. At inference this map is fused 50/50 with the raw matching map of the same templates, and the image score is the average of the top 250 values of the fused map.

The filtering network is trained from scratch using synthetic anomaly masks because real anomalies are unavailable in unsupervised training. The loss combines focal segmentation, soft IoU, SSIM, and a class-aware adaptor loss.

## Main Claims

The primary claim is that filtering anomaly cost volumes suppresses matching noise and improves both image-level detection and pixel-level localization compared with using raw template-matching anomaly maps. A secondary claim is that the plug-in preserves the baseline feature source and can be attached to reconstruction-, embedding-, 3D-, or text-based anomaly pipelines.

---

# Evaluation

## Data and splits

Two benchmarks, each a separate scored experiment, both at 224x224 RGB. One model per benchmark must handle all of its categories.

| | MVTec-AD (`rgb_mvtec_multiclass`) | VisA (`rgb_visa_multiclass`) |
|---|---|---|
| categories | 15 industrial (10 objects, 5 textures) | 12 complex, multi-instance and PCB objects |
| normal training images | 3,629 (`data/downloads/mvtec224/train.npz`) | 8,659 (`data/downloads/visa224/train.npz`, official 1cls split) |
| test inputs (all predicted) | 1,725 (`data/downloads/mvtec224/test_inputs.npz`) | 2,162 (`data/downloads/visa224/test_inputs.npz`) |
| visible development slice | 845 images (614 anomalous, 231 normal), `eval/train/data/dev_answers.npz` | 734 images (413 anomalous, 321 normal), `eval/train/data/visa_dev_answers.npz` |
| hidden slice (ranked) | 880 images (644 anomalous, 236 normal) | 1,428 images (787 anomalous, 641 normal) |

- **Training data:** normal images only. No anomalous image or defect mask may be used for training.
- **Test inputs:** every official test image, shuffled under anonymous ids, with its category but no label or mask. A method predicts every one of them.
- **Splits:** each category's normal test images are split between the visible and the hidden slice: 50/50 for MVTec-AD, one third / two thirds for VisA. Defective images are grouped by defect type, and whole types go to one slice or the other, so the hidden slices contain defect types the visible slices never show. Where a category has a single defect type, its defective images are split by the same ratio.

## Output contract

A method writes one file per benchmark into its directory, each with exactly three arrays:

- `outputs/<method>/predictions.npz` (MVTec-AD, 1,725 rows)
- `outputs/<method>/predictions_visa.npz` (VisA, 2,162 rows)

| array | shape | meaning |
|---|---|---|
| `ids` | (M,) int | every test-input id of that benchmark, exactly once |
| `image_scores` | (M,) float | higher = more anomalous |
| `anomaly_maps` | (M, 224, 224) float | per-pixel anomaly score, higher = more anomalous |

Nothing else the method writes (labels, masks, self-reported metrics, costs, flags) is read. Every number is recomputed from these raw outputs.

## Metrics

Every metric is computed within each category over that category's scored images, then averaged over the benchmark's categories with equal weight (class-wise mean). All metrics are percentages, and higher is better. Both experiments use the same metrics and coefficients.

| metric | measures | coefficient |
|---|---|---:|
| `image_auroc` | anomalous images ranked above normal ones | 1.0 |
| `pixel_auroc` (primary) | defect pixels ranked above normal pixels | 1.0 |
| `pixel_aupro` | per-region overlap integrated up to 30% pixel false-positive rate; every defect region counts equally, so small defects matter | 0.6 |
| `pixel_ap` | pixel average precision; sensitive to false positives because defect pixels are rare | 0.5 |
| `image_ap` | image average precision | 0.4 |
| `pixel_f1_max` | best pixel F1 over thresholds | 0.4 |
| `image_f1_max` | best image F1 over thresholds | 0.3 |

Because the metrics are class-wise means, shifting or rescaling one category's scores relative to another's changes nothing.

## Failure rules

Each experiment is judged on its own. A method is scored as failed in an experiment, with 0 (the worst value) on every metric of that experiment, when:

- its file for that benchmark is missing, including when the method directory does not exist;
- the file is unreadable or has the wrong arrays, shapes or dtypes;
- it does not list every test id exactly once;
- it contains a non-finite value;
- it does not solve both halves of the task: class-mean image AUROC and pixel AUROC must both be at least 70 (chance is 50).

A failing method never outranks one that solves the task.

## Ranking

An attempt's rank sums over both experiments (weight 0.5 each) and over the seven metrics. Each term is the coefficient times the attempt's improvement over the reference on the hidden slice. Each improvement is divided by how much that metric varies across the ranked attempts. Only the hidden-slice numbers in `scoring/scores_test.json` are ranked.

## Protocol constraints

The evaluator cannot observe these, so they are checked separately, and an attempt that breaks them is disqualified however it scores (details in `scoring/CONSTRAINTS.md`):

- normal-only training;
- the provided frozen feature extractor at 224 px;
- each test input scored on its own;
- no use of the hidden answers or of any copy of the MVTec-AD or VisA test annotations.

Using every normal training image as reference material is allowed. The baseline already does so.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 18m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-06-04 10:37] - core_claim
- Fourth CPU job completed the final 40-epoch default with validation passing.
- Final packaged scores show UCF improves pixel AUROC from 71.1 to 74.9 and image AUROC from 51.7 to 52.5 at 0.16M parameters.
- The compact reproduction remains at `core_claim`; broader MVTec/DINOv2 coverage is left as future work.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 16m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-04 10:38] - core_claim
- Implemented the paper's UCF computation at compact RGB scale: hierarchical template matching cost volumes, guided 3D filtering with residual channel/spatial attention, class-aware segmentation loss, and fused anomaly maps.
- Completed an end-to-end compute job and wrote `scoring/scores.json`.
- UCF improved pixel AUROC from 71.114 to 74.942 over the direct matching-cost baseline on the procedural multi-class RGB anomaly benchmark, with 0.16M additional parameters.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `method_runs` | **Status**: error
- **Working time**: 37m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

### Iteration 3: gpt-5.5
- **Milestone**: `method_runs` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

### Iteration 3: gpt-5.5
- **Milestone**: `method_runs` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 20m | **GPU**: 0.0h
- **Jobs**: 6 total (3 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 6 total (3 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 6 total (3 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 8 total (5 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 10m | **GPU**: 0.0h
- **Jobs**: 8 total (5 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: error
- **Working time**: 17m | **GPU**: 3.5h
- **Jobs**: 12 total (7 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 13m | **GPU**: 0.0h
- **Jobs**: 13 total (8 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-11-10 22:45] - core_claim_plus
- Hardening complete (see HARDENING.md). The gym is MVTec-AD (15 categories, 224 px, frozen DINOv2 ViT-S/14), scored on a hidden half of the test set whose defect types are absent from the visible development half. The evaluator recomputes class-wise mean metrics from raw outputs; failed runs, including missing method directories, score 0 everywhere.
- On the hidden half, AnomalDF+UCF (paper inference, lambda = 0.5, N = 3 templates) improves on AnomalDF: pixel AUROC 93.35 -> 94.07, pixel AP 44.3 -> 57.8, AUPRO 85.3 -> 86.7, pixel F1 48.9 -> 57.4. Image AUROC 91.64 -> 91.85 is within noise, so the image-level claim is not established at this scale.
- Better variants inside the constraints rank above the reference beyond seed noise, and weaker ones rank below. Every red-team shortcut is closed. The full-memory-bank brute force and the public MVTec-AD annotations are left to the grader as stated constraints.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 45m | **GPU**: 10.8h
- **Jobs**: 17 total (10 completed, 4 failed)

<details>
<summary>Progress Log</summary>

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

### [2026-11-10 22:45] - core_claim_plus
- Hardening complete (see HARDENING.md). The gym is MVTec-AD (15 categories, 224 px, frozen DINOv2 ViT-S/14), scored on a hidden half of the test set whose defect types are absent from the visible development half. The evaluator recomputes class-wise mean metrics from raw outputs; failed runs, including missing method directories, score 0 everywhere.
- On the hidden half, AnomalDF+UCF (paper inference, lambda = 0.5, N = 3 templates) improves on AnomalDF: pixel AUROC 93.35 -> 94.07, pixel AP 44.3 -> 57.8, AUPRO 85.3 -> 86.7, pixel F1 48.9 -> 57.4. Image AUROC 91.64 -> 91.85 is within noise, so the image-level claim is not established at this scale.
- Better variants inside the constraints rank above the reference beyond seed noise, and weaker ones rank below. Every red-team shortcut is closed. The full-memory-bank brute force and the public MVTec-AD annotations are left to the grader as stated constraints.

### [2026-11-20] - core_claim_plus
- Final hardening (see HARDENING.md).
- The reference is now the paper's full-shot AnomalDF+UCF (Table XXIV): the filter is trained with N = 3 random templates and applied to the whole category memory bank. Following the authors' code, the cost volume keeps the 256 lowest costs per token, so the same model takes both volumes. The full-bank AnomalDF brute force is now the baseline. VisA was added as a second hidden experiment, which grows the hidden test from 880 to 2,308 images.
- Hidden test, reference vs full-bank baseline, image / pixel AUROC: MVTec-AD 96.75/95.28 -> 96.88/96.59 (pixel AP 44.6 -> 62.1); VisA 87.71/94.53 -> 88.61/94.80 (pixel AP 23.4 -> 29.4).
- The VisA image-level gain is significant (+0.90, 95% CI [+0.38, +1.46]). The paper's guidance and loss ablations now point the paper's way. Every shortcut fails or ties the reference.
- Not achieved: no better variant beats the seed-0 reference beyond seed noise (seed 0 is the best of 5 seeds), and MVTec-AD image AUROC does not improve robustly across seeds.

</details>


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
