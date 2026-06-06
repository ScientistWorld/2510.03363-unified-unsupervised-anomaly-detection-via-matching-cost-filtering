# Scientist Constraints

Future improvements must use the same procedural dataset split, image size, category set, normal-only training condition, and evaluation script. Real anomaly masks may be used only for evaluation, not for training.

The feature source, number of reference examples per image, image resolution, procedural data generator, and train/test split must remain fixed unless the experiment name is changed. Improvements should come from better anomaly scoring, calibration, model design, or training strategy rather than larger backbones, more references, higher image resolution, extra data, or test-time access to labels.

Submitted methods must write the same prediction artifact fields consumed by `scripts/evaluate.sh`: image labels, pixel masks, image scores, and anomaly maps for the baseline and proposed method names declared in `scoring/reference.json`.
