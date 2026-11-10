# Evaluation

## Data and splits

The benchmark is MVTec-AD (15 industrial categories: 10 objects, 5 textures) at 224x224 RGB. One model must handle all categories.

- **Training data:** the 3,629 normal training images of all 15 categories (`data/downloads/mvtec224/train.npz`). No anomalous image or defect mask may be used for training.
- **Test inputs:** all 1,725 official test images, shuffled under anonymous ids, with their category but no label or mask (`data/downloads/mvtec224/test_inputs.npz`). A method predicts every one of them.
- **Development half (visible):** 845 of those test images (614 anomalous, 231 normal) have answers in `eval/train/data/dev_answers.npz`, scored by `scripts/evaluate_train.sh`.
- **Hidden half:** the other 880 test images (644 anomalous, 236 normal) are scored by `scripts/evaluate_test.sh` against answers that only `eval/test/` holds. Each category's normal test images are split 50/50, and whole defect types go to one half or the other, so the hidden half contains defect types that never appear in the development half. Categories with a single defect type split their defective images 50/50.

## Output contract

A method writes `outputs/<method>/predictions.npz` with exactly three arrays:

| array | shape | meaning |
|---|---|---|
| `ids` | (1725,) int | every test-input id exactly once |
| `image_scores` | (1725,) float | higher = more anomalous |
| `anomaly_maps` | (1725, 224, 224) float | per-pixel anomaly score, higher = more anomalous |

Nothing else the method writes (labels, masks, self-reported metrics, costs, flags) is read. Every number is recomputed from these raw outputs.

## Metrics

Every metric is computed within each category over that category's scored images and then averaged over the 15 categories with equal weight (class-wise mean). All are percentages, higher is better.

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

A method is scored as failed, with 0 (the worst value) on every metric, when its artifact is missing (including a method directory that does not exist), unreadable, has the wrong arrays, shapes or dtypes, does not list every test id exactly once, or contains a non-finite value. It also fails when it does not solve both halves of the task: its class-mean image AUROC and pixel AUROC must both be at least 70 (chance is 50). A failing method never outranks one that solves the task.

## Ranking

An attempt's rank sums, over the seven metrics, the coefficient times its improvement over the reference on the hidden half, with each improvement divided by how much that metric varies across the ranked attempts. Only the hidden-half numbers in `scoring/scores_test.json` are ranked.

## Protocol constraints

The evaluator cannot observe these, so they are checked separately, and an attempt that breaks them is disqualified however it scores (details in `scoring/CONSTRAINTS.md`): normal-only training; the provided frozen feature extractor at 224 px; at most three normal reference images per test input; each test input scored on its own; no use of the hidden answers or of any copy of the MVTec-AD test annotations.
