# Experiment Overview

Both experiments ask the same question on two different benchmarks. Can a single anomaly detector, trained on normal images only, flag defective images and outline the defect inside them, across every category of the benchmark at 224x224?

The reference point is a matching detector that compares each input with the normal training images of its category, using a fixed pretrained feature extractor. A contribution must make that comparison more reliable at the same operating point: the same feature extractor and input resolution, and no real defects in training.

Two questions are scored together in each experiment:

- **Image level** (AUROC, AP, best F1): do anomalous images rank above normal ones?
- **Pixel level** (AUROC, AP, best F1, and per-region overlap up to 30% false-positive rate): does the heatmap mark the defect and nothing else? Region overlap gives every defect region equal weight, so small defects count as much as large ones.

Every metric is averaged over categories, so a method cannot gain by ordering categories against each other. To be scored at all, a method must clear 70 image AUROC and 70 pixel AUROC in that experiment.

The ranked numbers come from hidden slices of the official test sets whose defect types never appear in the visible development slices. Tuning to the visible defects therefore does not carry over, and a method only gains by detecting kinds of defects it has never seen.

## rgb_mvtec_multiclass

MVTec-AD: 15 industrial categories (objects and textures), with a hidden half of the test set (880 images). Most categories hold one centred object or a homogeneous texture. Defects range from large structural damage to small contaminations.

## rgb_visa_multiclass

VisA: 12 categories with complex structure, several instances per image (candles, capsules, macaroni) or dense circuit boards. The hidden two thirds of the test set (1,428 images) are ranked. Defects are often tiny relative to the image, so pixel AP and F1 are low for every method and leave a lot of room. The experiment also enlarges the hidden test, so that image-level differences between good methods can be told apart from sampling noise.
