# Scientist Constraints

Future improvements must use the same procedural dataset split, image size, category set, normal-only training condition, and evaluation script. Real anomaly masks may be used only for evaluation, not for training.

The frozen template feature source and the number of normal templates per image must remain fixed unless the experiment name is changed. Improvements should come from better cost construction, cost filtering, loss design, or training strategy rather than larger backbones, more templates, higher image resolution, extra data, or test-time access to labels.

Submitted methods must write the same prediction artifact fields consumed by `scripts/evaluate.sh`: image labels, pixel masks, image scores, and anomaly maps for the baseline and proposed method names declared in `scoring/reference.json`.
