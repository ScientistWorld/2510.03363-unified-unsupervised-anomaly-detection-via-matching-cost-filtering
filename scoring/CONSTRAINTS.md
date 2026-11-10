# Scientist Constraints

These rules define the problem. The hidden-test evaluator can only see a method's predictions, so it cannot check them; every attempt is checked against them separately, and an attempt that breaks one is disqualified whatever it scores.

## Data

- **Normal-only training.** Anything learned (weights, statistics, thresholds, prototypes, calibration) may be fitted only on the 3,629 normal training images in `data/downloads/mvtec224/train.npz`, on synthetic anomalies made from them, and on the DTD texture images provided for synthesising anomalies.
- **Development answers are for evaluation only.** `eval/train/data/dev_answers.npz` (labels and masks of the visible half) may be used to score runs and to choose among a small number of configurations. It must never serve as a training target or be used to fit parameters. The hidden half contains defect types absent from the visible half, so fitting the visible defects does not transfer.
- **No hidden or external answers.** Do not read `eval/test/`. Do not download, reconstruct or match against any copy of the MVTec-AD test images, labels or masks, and do not use any other anomaly-detection dataset.
- **Each test input is scored on its own.** A test input's image score and map may depend only on that input, its category, the training data and the model. They may not use statistics, matches or rankings computed across other test inputs (no transductive or batch-level scoring over the test set).

## Resources (the paper's operating point)

- **Feature extractor:** the provided frozen DINOv2 ViT-S/14 weights (`scripts/download.sh`), applied to the 224x224 inputs. Do not fine-tune it, replace it with a larger or different pretrained network, or raise the input resolution.
- **Template budget:** for each test input, the patch-level comparison against normal data may use at most **N = 3** normal training images of the same category. A method may choose which three images to use, but the choice may rely on at most one global descriptor vector per training image. Keeping a memory bank of patch features from the whole training set, or matching an input against more than three training images (for example by averaging several random draws of three), exceeds the budget.
- **Learned components** may be trained from scratch on the data above. Their size is not scored, but they must not store patch features of the training set as a lookup memory that sidesteps the template budget.

## Output contract

Write `outputs/<method>/predictions.npz` with `ids` (every test id exactly once), `image_scores` (one finite float per id) and `anomaly_maps` (one finite 224x224 float map per id), as described in `briefing/evaluation.md`. Score with `bash scripts/evaluate_train.sh` (visible half). `bash scripts/evaluate_test.sh` scores the hidden half and writes `scoring/scores_test.json`; it needs no arguments and scores every method on disk.
