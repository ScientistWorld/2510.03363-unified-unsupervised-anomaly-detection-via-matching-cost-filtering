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

A successful system separates normal from abnormal test images, localizes defect regions accurately, and does so without changing the upstream normal-template feature source or using real anomaly labels during training.

---

# Paper's Method

## Key Contribution

Unified Cost Filtering (UCF) treats visual anomaly detection as matching-cost filtering. Instead of directly thresholding noisy feature distances, it constructs a patch-wise anomaly cost volume and learns to filter that volume using guidance from the input image and an initial anomaly map.

## Approach

The method extracts hierarchical features from a test image and normal reference templates. It computes global cosine matching between each test patch and template patches, converts similarity to anomaly cost, and collapses the template/matching axes into a cost volume. A 3D filtering network refines this volume with residual channel and spatial attention guided by the input and the initial pooled anomaly map. The filtered volume is min-pooled over the matching dimension, converted to a two-class normal/anomaly score map, and the image score is the average of the top anomaly pixels.

The filtering network is trained from scratch using synthetic anomaly masks because real anomalies are unavailable in unsupervised training. The loss combines focal segmentation, soft IoU, SSIM, and a class-aware adaptor loss.

## Main Claims

The primary claim is that filtering anomaly cost volumes suppresses matching noise and improves both image-level detection and pixel-level localization compared with using raw template-matching anomaly maps. A secondary claim is that the plug-in preserves the baseline feature source and can be attached to reconstruction-, embedding-, 3D-, or text-based anomaly pipelines.

---

# Evaluation

## Metrics

Image-level AUROC measures whether anomalous images receive higher image scores than normal images. Pixel-level AUROC measures whether anomalous pixels receive higher heatmap scores than normal pixels. Both metrics are reported as percentages, and higher is better.

The constraint metric is added trainable parameter count in millions. Lower is better because the benchmark should reward improved anomaly detection without simply scaling model capacity.

## Baselines and Targets

The paper's selected reference for this compact reproduction is the RGB multi-class AnomalDF comparison on MVTec-AD. The target method is the same baseline with the anomaly-map refinement plug-in added.

## Evaluation Protocol

Training uses normal images only, with synthetic anomalies allowed for learning pixel-level supervision. Evaluation is on held-out normal and anomalous images with ground-truth masks. The evaluator reads saved prediction artifacts and computes metrics from those artifacts; it does not import or run the submitted method.

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


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
