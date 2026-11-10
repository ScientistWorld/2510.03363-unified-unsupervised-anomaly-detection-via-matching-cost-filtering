# Evaluation Targets

## Primary: rgb_mvtec_multiclass (weight 1.0)

Multi-class unsupervised anomaly detection and localisation on MVTec-AD: one model for all 15 categories, normal-only training, scored on a hidden half of the official test set whose defect types are absent from the visible development half.

- **Primary metric:** `pixel_auroc` (class-wise mean, higher is better).
- **Scored metrics and coefficients** (all class-wise means, higher is better): `image_auroc` 1.0, `pixel_auroc` 1.0, `pixel_aupro` 0.6, `pixel_ap` 0.5, `image_ap` 0.4, `pixel_f1_max` 0.4, `image_f1_max` 0.3. Image and pixel AUROC are the paper's headline metrics (Table I). The pixel-level AP, F1 and AUPRO are where the paper reports its largest gains (Sec. IV-A3, Fig. 8, Table XXIV).
- **Scored constraint:** both halves of the task must be solved, with class-mean image AUROC ≥ 70 and pixel AUROC ≥ 70. Otherwise every metric is 0. A broken, partial or non-finite artifact, or a missing method directory, also scores 0 everywhere.
- **Protocol constraints (the paper's "without sacrificing" side):** normal-only training; the frozen DINOv2 ViT-S/14 feature source at 224 px; at most N = 3 normal reference images per test input (no full memory bank); each test input scored independently. The evaluator cannot observe these, so `scoring/CONSTRAINTS.md` states them and attempts are checked against them separately.
- **Paper reference:** AnomalDF 96.8 / 98.1 and AnomalDF+UCF 98.5 / 98.8 image / pixel AUROC (Table I, 256 px, whole test set). Table XXIV gives the other five metrics in `scoring/reference.json`. The gym's own reference is the reproduced AnomalDF+UCF row in `scoring/scores_test.json` (hidden half: 91.85 / 94.07).

## How to run

```bash
bash scripts/download.sh          # once, on a node with internet
bash scripts/reproduce.sh         # prepare data, AnomalDF baseline, AnomalDF+UCF, both evaluations
bash scripts/evaluate_train.sh    # visible development half -> scoring/scores_train.json
bash scripts/evaluate_test.sh     # hidden half -> scoring/scores_test.json (every method on disk)
```

## Out of scope

The paper's RGB-3D (MVTec 3D-AD, Eyecandies) and RGB-Text (zero-/few-shot) settings, and its other RGB baselines (GLAD, HVQ-Trans, UniAD, Dinomaly) and datasets (VisA, MPDD, BTAD), are not reproduced. Self-reported cost numbers (parameters, FLOPs, memory, time) are not scored, because the evaluator cannot recompute them from raw outputs.
