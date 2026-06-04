# Paper's Method

## Key Contribution

Unified Cost Filtering (UCF) treats visual anomaly detection as matching-cost filtering. Instead of directly thresholding noisy feature distances, it constructs a patch-wise anomaly cost volume and learns to filter that volume using guidance from the input image and an initial anomaly map.

## Approach

The method extracts hierarchical features from a test image and normal reference templates. It computes global cosine matching between each test patch and template patches, converts similarity to anomaly cost, and collapses the template/matching axes into a cost volume. A 3D filtering network refines this volume with residual channel and spatial attention guided by the input and the initial pooled anomaly map. The filtered volume is min-pooled over the matching dimension, converted to a two-class normal/anomaly score map, and the image score is the average of the top anomaly pixels.

The filtering network is trained from scratch using synthetic anomaly masks because real anomalies are unavailable in unsupervised training. The loss combines focal segmentation, soft IoU, SSIM, and a class-aware adaptor loss.

## Main Claims

The primary claim is that filtering anomaly cost volumes suppresses matching noise and improves both image-level detection and pixel-level localization compared with using raw template-matching anomaly maps. A secondary claim is that the plug-in preserves the baseline feature source and can be attached to reconstruction-, embedding-, 3D-, or text-based anomaly pipelines.
