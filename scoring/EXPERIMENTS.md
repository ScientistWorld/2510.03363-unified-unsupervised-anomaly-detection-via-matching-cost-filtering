# Experiment Overview

## rgb_mvtec_multiclass

This experiment asks whether a single anomaly detector, trained on normal images only, can both flag defective images and outline the defect inside them across 15 different industrial categories (MVTec-AD at 224x224). The reference point is a matching detector that compares each input with a few normal images of its category, using a fixed pretrained feature extractor. A contribution must make that comparison more reliable without enlarging the resources it uses: no more normal reference images per input, no larger feature extractor or input resolution, and no real defects in training.

Two questions are scored together. Image-level metrics (AUROC, AP, best F1) ask whether anomalous images rank above normal ones. Pixel-level metrics (AUROC, AP, best F1, and per-region overlap up to 30% false-positive rate) ask whether the heatmap marks the defect and nothing else; region overlap gives every defect region equal weight, so small defects count as much as large ones. Every metric is averaged over categories, so a method cannot gain by ordering categories against each other, and a method must clear 70 image AUROC and 70 pixel AUROC to be scored at all.

The ranked numbers come from a hidden half of the official test set whose defect types never appear in the visible development half. Tuning to the visible defects therefore does not carry over, and a method only gains by detecting kinds of defects it has never seen.
