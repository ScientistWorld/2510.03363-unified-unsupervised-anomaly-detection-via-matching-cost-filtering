# Problem Definition

## Research Question

Detect anomalous images and localize anomalous pixels when training data contains only normal examples. The target setting is multi-class visual anomaly detection: a single system must handle multiple visual categories and produce both an image-level anomaly score and a pixel-level anomaly heatmap.

## Why It Matters

Industrial inspection and medical screening often have abundant normal data but few labeled defects. A useful solution should identify subtle defects without requiring real anomaly labels for every category.

## Success Criteria

A successful system separates normal from abnormal test images, localizes defect regions accurately, and does so without using real anomaly labels during training.
