# Evaluation

## Metrics

Image-level AUROC measures whether anomalous images receive higher image scores than normal images. Pixel-level AUROC measures whether anomalous pixels receive higher heatmap scores than normal pixels. Both metrics are reported as percentages, and higher is better.

Additional ranking and localization diagnostics include image/pixel average precision, best pixel F1 over thresholds, per-region overlap at low false-positive rates, and category-mean AUROC. The constraint metric is added trainable parameter count in millions. Lower is better because the benchmark should reward improved anomaly detection without simply scaling model capacity.

## Baselines and Targets

The paper's selected reference for this compact reproduction is an RGB multi-class comparison on MVTec-AD. The reproduced benchmark uses a compact procedural substitute with the same output contract: an image-level anomaly score and a pixel-level anomaly heatmap for each test image.

## Evaluation Protocol

Training uses normal images only, with synthetic anomalies allowed for learning pixel-level supervision. Evaluation is on held-out normal and anomalous images with ground-truth masks. The evaluator reads saved prediction artifacts and computes metrics from those artifacts; it does not import or run the submitted method.
