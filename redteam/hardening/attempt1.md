# Attempt 1 (job 9a3c5938-cd3, cancelled after the reference and fast group)

UCF readout v1: decoder -> 3D conv L->1 into 64 learned channels -> min over
those channels -> conv -> softmax. The filtered volume no longer carried the
D*N matching dimension, so the readout lost the matching evidence.

| run | slice | image_auroc | pixel_auroc | image_ap | image_f1_max | pixel_ap | pixel_f1_max | pixel_aupro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AnomalDF (N=3) | test | 88.781 | 94.968 | 95.849 | 88.530 | 39.994 | 49.973 | 81.062 |
| AnomalDF+UCF v1 | test | 80.741 | 86.613 | 93.605 | 84.529 | 61.039 | 61.803 | 62.131 |
| kNN full memory bank | test | 91.864 | 96.008 | 97.134 | 89.258 | 41.031 | 49.700 | 81.676 |
| dev-supervised probe (shortcut) | test | 77.996 | 94.950 | 89.699 | 87.305 | 45.868 | 43.869 | 81.647 |
| AnomalDF (N=3) | train | 92.342 | 90.857 | 97.151 | 90.323 | 37.247 | 40.794 | 82.849 |
| AnomalDF+UCF v1 | train | 84.641 | 74.068 | 94.745 | 85.147 | 38.622 | 43.407 | 64.763 |
| kNN full memory bank | train | 94.551 | 93.571 | 98.113 | 92.602 | 41.910 | 45.498 | 83.886 |
| dev-supervised probe (shortcut) | train | 84.560 | 98.368 | 93.368 | 88.692 | 77.320 | 68.507 | 89.188 |

Shortcut payloads built from the v1 reference all failed the hardened
evaluator: constant outputs, constant image scores, NaN maps, half population,
duplicate ids, invented answers, forged score file (overwritten), and the
baseline replacement (only the replaced baseline row failed; the reference row
was unchanged).

Timing on one A100 80GB: data preparation 209 s, AnomalDF 50 s, UCF training
40 epochs x ~10 s, prediction ~30 s, full-memory kNN 31 s.
