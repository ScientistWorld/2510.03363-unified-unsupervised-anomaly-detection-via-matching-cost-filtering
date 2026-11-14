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

An attempt's rank sums over both experiments (weight 1.0 each) and over the seven metrics. Each term is the coefficient times the attempt's improvement over the reference on the hidden slice. Each improvement is divided by how much that metric varies across the ranked attempts. Only the hidden-slice numbers in `scoring/scores_test.json` are ranked.

## Protocol constraints

The evaluator cannot observe these, so they are checked separately, and an attempt that breaks them is disqualified however it scores (details in `scoring/CONSTRAINTS.md`):

- normal-only training;
- the provided frozen feature extractor at 224 px;
- each test input scored on its own;
- no use of the hidden answers or of any copy of the MVTec-AD or VisA test annotations.

Using every normal training image as reference material is allowed. The baseline already does so.
