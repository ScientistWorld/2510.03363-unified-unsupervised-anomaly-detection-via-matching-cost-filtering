# Evaluation Targets

Two scored experiments, weight 0.5 each, with the same metrics, coefficients and failure rules. Each is multi-class unsupervised anomaly detection and localisation: one model per benchmark covers all of its categories and is trained on normal images only. Each is scored on a hidden slice of the official test set whose defect types are absent from the visible development slice.

## rgb_mvtec_multiclass (weight 0.5)

MVTec-AD, 15 categories. Hidden half: 880 images.

## rgb_visa_multiclass (weight 0.5)

VisA, 12 categories, official 1cls split. Hidden two thirds: 1,428 images. Added so that image-level differences between good methods can clear the noise of the hidden test: MVTec-AD alone has only 880 hidden images. VisA is the paper's second RGB benchmark.

## Scored metrics (both experiments)

- **Primary metric:** `pixel_auroc` (class-wise mean, higher is better).
- **Scored metrics and coefficients** (all class-wise means, higher is better): `image_auroc` 1.0, `pixel_auroc` 1.0, `pixel_aupro` 0.6, `pixel_ap` 0.5, `image_ap` 0.4, `pixel_f1_max` 0.4, `image_f1_max` 0.3. Image and pixel AUROC are the paper's headline metrics (Tables I and III). The pixel-level AP, F1 and AUPRO are where the paper reports its largest gains (Sec. IV-A3, Fig. 8, Table XXIV).
- **Scored constraint:** in each experiment, both halves of the task must be solved, with class-mean image AUROC ≥ 70 and pixel AUROC ≥ 70. Otherwise every metric of that experiment is 0. A broken, partial or non-finite artifact, a missing file for that benchmark, or a missing method directory also scores 0 everywhere in that experiment.
- **Protocol constraints (the paper's operating point):** normal-only training; the frozen DINOv2 ViT-S/14 feature source at 224 px; each test input scored independently; no test annotations. The evaluator cannot observe these, so `scoring/CONSTRAINTS.md` states them and attempts are checked against them separately. Matching against every normal training image is allowed. It is the baseline's own protocol (AnomalyDINO full-shot).

## Paper reference

Paper Table XXIV, full template bank, 256 px, whole test sets (image AUROC / pixel AUROC):

| benchmark | AnomalDF | AnomalDF+UCF |
|---|---|---|
| MVTec-AD | 99.0 / 97.5 | 99.3 / 98.9 |
| VisA | 94.6 / 98.3 | 95.5 / 99.4 |

`scoring/reference.json` gives the other metrics, and the N = 3 rows (MVTec-AD 96.8 / 98.1 → 98.5 / 98.8; VisA 90.5 / 97.4 → 94.3 / 99.2).

The gym's own reference is the reproduced AnomalDF+UCF row in `scoring/scores_test.json`: 224 px, compact filter, hidden slices. Image / pixel AUROC:

| benchmark | AnomalDF (full bank) | AnomalDF+UCF |
|---|---|---|
| MVTec-AD | 96.75 / 95.28 | 96.88 / 96.59 |
| VisA | 87.71 / 94.53 | 88.61 / 94.80 |

## How to run

```bash
bash scripts/download.sh          # once, on a node with internet
bash scripts/reproduce.sh         # prepare data, AnomalDF baseline, AnomalDF+UCF, both evaluations
bash scripts/evaluate_train.sh    # visible development slices -> scoring/scores_train.json
bash scripts/evaluate_test.sh     # hidden slices -> scoring/scores_test.json (every method on disk)
```

## Out of scope

The following are not reproduced:

- the paper's RGB-3D (MVTec 3D-AD, Eyecandies) and RGB-Text (zero-/few-shot) settings;
- its other RGB baselines (GLAD, HVQ-Trans, UniAD, Dinomaly);
- its other RGB datasets (MPDD, BTAD);
- its higher-resolution rows (448/672 px).

Self-reported cost numbers (parameters, FLOPs, memory, time) are not scored, because the evaluator cannot recompute them from raw outputs.
