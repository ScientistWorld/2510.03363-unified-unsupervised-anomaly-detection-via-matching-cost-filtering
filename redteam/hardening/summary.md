## test slice

| run | status | rank (honest) | rank (full) | image_auroc | pixel_auroc | pixel_aupro | pixel_ap | image_ap | pixel_f1_max | image_f1_max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| knn_fullshot | ok | +6.227 | +0.405 | 91.86 | 96.01 | 81.68 | 41.03 | 97.13 | 49.70 | 89.26 |
| ucf_1epoch | ok | +3.391 | +0.353 | 84.89 | 93.93 | 86.12 | 46.34 | 93.83 | 52.09 | 87.37 |
| anomaldf_baseline | ok | +4.595 | +0.261 | 88.78 | 94.97 | 81.06 | 39.99 | 95.85 | 49.97 | 88.53 |
| ucf_tta4 | ok | +0.279 | +0.043 | 80.10 | 87.77 | 61.08 | 61.33 | 93.37 | 61.70 | 84.51 |
| ucf_nearest_templates | ok | +0.297 | +0.039 | 80.98 | 86.90 | 61.25 | 60.70 | 93.58 | 61.74 | 84.51 |
| reference | ok | +0.000 | +0.000 | 79.94 | 87.40 | 60.45 | 60.58 | 93.27 | 61.38 | 84.51 |
| shortcut_baseline_replacement:anomaldf_ucf | ok |  | +0.000 | 79.94 | 87.40 | 60.45 | 60.58 | 93.27 | 61.38 | 84.51 |
| shortcut_negative_costs | ok |  | +0.000 | 79.94 | 87.40 | 60.45 | 60.58 | 93.27 | 61.38 | 84.51 |
| shortcut_dev_supervised | ok |  | -0.040 | 78.00 | 94.95 | 81.65 | 45.87 | 89.70 | 43.87 | 87.30 |
| ucf_seed2 | ok | +0.140 | -0.044 | 81.30 | 87.34 | 57.42 | 59.41 | 93.45 | 61.15 | 84.63 |
| ucf_no_guidance | ok | +0.047 | -0.072 | 80.97 | 90.18 | 73.09 | 46.94 | 92.64 | 52.33 | 85.35 |
| ucf_n8_templates | ok | -0.457 | -0.155 | 81.02 | 86.73 | 58.07 | 56.28 | 93.44 | 57.95 | 85.04 |
| ucf_seed1 | ok | -1.687 | -0.307 | 79.27 | 85.65 | 54.98 | 55.52 | 92.83 | 57.49 | 84.66 |
| shortcut_baseline_replacement:anomaldf | failed: task gate: image_auroc=50.00, pixel_auroc=50.00 (bot |  | -8.521 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_constant_image_scores | failed: task gate: image_auroc=50.00, pixel_auroc=87.40 (bot |  | -8.521 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_constant_outputs | failed: task gate: image_auroc=50.00, pixel_auroc=50.00 (bot |  | -8.521 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_duplicate_ids | failed: ids must list every test input exactly once |  | -8.521 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_forged_scores | failed: task gate: image_auroc=50.00, pixel_auroc=50.00 (bot |  | -8.521 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_half_population | failed: ids must list every test input exactly once |  | -8.521 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_invented_answers | failed: task gate: image_auroc=51.14, pixel_auroc=51.19 (bot |  | -8.521 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_nan_maps | failed: non-finite image scores or anomaly maps |  | -8.521 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

seed spread (max-min over 3 training seeds): {"image_auroc": 2.034, "pixel_auroc": 1.75, "pixel_aupro": 5.472, "pixel_ap": 5.06, "image_ap": 0.616, "pixel_f1_max": 3.884, "image_f1_max": 0.147, "rank_honest": 1.828, "rank_full": 0.307}

## train slice

| run | status | rank (honest) | rank (full) | image_auroc | pixel_auroc | pixel_aupro | pixel_ap | image_ap | pixel_f1_max | image_f1_max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shortcut_dev_supervised | ok |  | +2.482 | 84.56 | 98.37 | 89.19 | 77.32 | 93.37 | 68.51 | 88.69 |
| knn_fullshot | ok | +8.370 | +1.321 | 94.55 | 93.57 | 83.89 | 41.91 | 98.11 | 45.50 | 92.60 |
| anomaldf_baseline | ok | +5.341 | +0.960 | 92.34 | 90.86 | 82.85 | 37.25 | 97.15 | 40.79 | 90.32 |
| ucf_no_guidance | ok | +0.464 | +0.483 | 82.62 | 86.92 | 77.12 | 37.05 | 93.33 | 42.46 | 84.73 |
| ucf_1epoch | ok | +0.413 | +0.248 | 87.56 | 73.47 | 87.97 | 30.44 | 95.17 | 38.72 | 88.40 |
| ucf_tta4 | ok | +0.322 | +0.037 | 87.11 | 71.05 | 66.48 | 37.18 | 95.45 | 42.38 | 87.26 |
| ucf_nearest_templates | ok | +0.428 | +0.029 | 87.79 | 69.72 | 67.85 | 36.63 | 95.75 | 42.41 | 87.01 |
| reference | ok | +0.000 | +0.000 | 86.52 | 71.02 | 65.39 | 37.07 | 95.26 | 42.56 | 86.85 |
| shortcut_negative_costs | ok |  | +0.000 | 86.52 | 71.02 | 65.39 | 37.07 | 95.26 | 42.56 | 86.85 |
| ucf_n8_templates | ok | -1.306 | -0.135 | 84.91 | 70.86 | 63.89 | 35.70 | 94.64 | 41.59 | 85.17 |
| ucf_seed2 | ok | -2.824 | -0.372 | 83.68 | 66.13 | 63.26 | 34.01 | 94.08 | 39.75 | 85.76 |
| ucf_seed1 | ok | -4.621 | -0.618 | 82.73 | 65.42 | 59.91 | 30.34 | 93.88 | 36.92 | 84.60 |
| shortcut_constant_image_scores | failed: task gate: image_auroc=50.00, pixel_auroc=71.02 (bot |  | -8.103 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_constant_outputs | failed: task gate: image_auroc=50.00, pixel_auroc=50.00 (bot |  | -8.103 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_duplicate_ids | failed: ids must list every test input exactly once |  | -8.103 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_forged_scores | failed: task gate: image_auroc=50.00, pixel_auroc=50.00 (bot |  | -8.103 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_half_population | failed: ids must list every test input exactly once |  | -8.103 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_invented_answers | failed: task gate: image_auroc=50.14, pixel_auroc=50.67 (bot |  | -8.103 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_nan_maps | failed: non-finite image scores or anomaly maps |  | -8.103 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

seed spread (max-min over 3 training seeds): {"image_auroc": 3.797, "pixel_auroc": 5.597, "pixel_aupro": 5.481, "pixel_ap": 6.733, "image_ap": 1.378, "pixel_f1_max": 5.642, "image_f1_max": 2.249, "rank_honest": 4.621, "rank_full": 0.618}

## paired bootstrap vs reference (hidden test, 200 image resamples)

| run | d image AUROC | 95% CI | d pixel AUROC (28px) | 95% CI |
|---|---:|---|---:|---|
| anomaldf_baseline | +8.82 | [+6.35, +11.69] | +7.50 | [+5.44, +9.83] |
| knn_fullshot | +11.94 | [+9.32, +14.66] | +8.56 | [+6.35, +11.02] |
| shortcut_dev_supervised | -1.99 | [-6.12, +2.03] | +7.39 | [+5.19, +9.66] |
| ucf_1epoch | +4.92 | [+2.04, +7.80] | +6.34 | [+4.45, +8.53] |
| ucf_n8_templates | +1.13 | [-0.50, +2.53] | -0.52 | [-2.13, +0.88] |
| ucf_nearest_templates | +1.06 | [-0.04, +2.10] | -0.52 | [-1.12, -0.01] |
| ucf_no_guidance | +0.99 | [-2.28, +4.20] | +2.70 | [+0.47, +5.04] |
| ucf_seed1 | -0.65 | [-2.34, +1.39] | -1.60 | [-2.50, -0.73] |
| ucf_seed2 | +1.33 | [+0.02, +2.88] | +0.09 | [-0.73, +0.97] |
| ucf_tta4 | +0.17 | [-0.28, +0.66] | +0.39 | [+0.19, +0.66] |
