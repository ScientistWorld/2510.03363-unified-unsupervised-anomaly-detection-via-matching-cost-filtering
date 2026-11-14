# Scientist Constraints

These rules define the problem. The hidden-test evaluator can only see a method's predictions, so it cannot check them; every attempt is checked against them separately, and an attempt that breaks one is disqualified whatever it scores.

## Data

- **Normal-only training.** Anything learned (weights, statistics, thresholds, prototypes, memory banks, calibration) may be fitted only on the normal training images of the two benchmarks: `data/downloads/mvtec224/train.npz` (MVTec-AD, 3,629 images) and `data/downloads/visa224/train.npz` (VisA, 8,659 images). Synthetic anomalies made from them, and the DTD texture images provided for synthesising anomalies, are also allowed.
- **Development answers are for evaluation only.** `eval/train/data/dev_answers.npz` and `eval/train/data/visa_dev_answers.npz` (labels and masks of the visible development slices) may be used to score runs and to choose among a small number of configurations. They must never serve as a training target or be used to fit parameters. The hidden slices contain defect types absent from the visible ones, so fitting the visible defects does not transfer.
- **No hidden or external answers.** Do not read `eval/test/`. Do not download, reconstruct or match against any copy of the MVTec-AD or VisA test images, labels or masks, and do not use any other anomaly-detection dataset.
- **Each test input is scored on its own.** A test input's image score and map may depend only on that input, its category, the training data and the model. They may not use statistics, matches or rankings computed across other test inputs (no transductive or batch-level scoring over the test set).

## Resources (the paper's operating point)

- **Feature extractor:** the provided frozen DINOv2 ViT-S/14 weights (`scripts/download.sh`), applied to the 224x224 inputs. Do not fine-tune it, replace it with a larger or different pretrained network, or raise the input resolution.
- **Normal reference data:** any of the provided normal training images of the input's dataset may be used as reference material at test time, including all of them: the baseline itself matches every input against the whole training set of its category.
- **Learned components** may be trained from scratch on the data above. Their size and compute are not scored.

## Output contract

For each dataset, write one file per method directory, as described in `briefing/evaluation.md`:

- MVTec-AD: `outputs/<method>/predictions.npz`
- VisA: `outputs/<method>/predictions_visa.npz`

Each file holds `ids` (every test id of that dataset exactly once), `image_scores` (one finite float per id) and `anomaly_maps` (one finite 224x224 float map per id). A missing or invalid file fails that experiment.

To score:

- `bash scripts/evaluate_train.sh` scores the visible development slices.
- `bash scripts/evaluate_test.sh` scores the hidden slices and writes `scoring/scores_test.json`. It needs no arguments and scores every method on disk.
