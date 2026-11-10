## test slice (class-mean metrics)

| run | group | status | rank (honest) | rank (full) | image_auroc | pixel_auroc | pixel_aupro | pixel_ap | image_ap | pixel_f1_max | image_f1_max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ucf_fullbank | over template budget | ok | +7.206 | +0.277 | 96.20 | 95.99 | 89.28 | 58.89 | 96.72 | 58.94 | 96.08 |
| ucf_ensemble3_nearest | better | ok | +4.541 | +0.221 | 94.02 | 95.49 | 88.99 | 60.30 | 95.79 | 59.26 | 94.56 |
| ucf_nearest_seed1 | better (seed) | ok | +4.404 | +0.200 | 94.22 | 95.28 | 88.75 | 59.65 | 95.87 | 58.91 | 94.65 |
| ucf_nearest | better | ok | +4.031 | +0.157 | 93.96 | 95.42 | 88.73 | 58.43 | 95.74 | 58.05 | 94.55 |
| ucf_nearest_seed2 | better (seed) | ok | +3.476 | +0.132 | 93.77 | 95.31 | 88.58 | 58.06 | 95.30 | 57.80 | 94.56 |
| ucf_focal_only | weaker | ok | +2.589 | +0.127 | 92.59 | 94.89 | 88.34 | 59.91 | 95.47 | 57.81 | 94.43 |
| ucf_tta4 | over template budget | ok | +2.455 | +0.116 | 93.37 | 94.39 | 88.06 | 58.72 | 95.31 | 58.59 | 94.33 |
| ucf_ensemble3 | better | ok | +0.754 | +0.070 | 92.06 | 94.12 | 86.90 | 59.78 | 94.98 | 58.50 | 93.32 |
| ucf_seed1 | seed | ok | +0.329 | +0.038 | 91.85 | 94.01 | 86.69 | 59.11 | 94.95 | 58.11 | 93.24 |
| reference | reference | ok | +0.000 | +0.000 | 91.85 | 94.07 | 86.69 | 57.82 | 94.67 | 57.42 | 93.21 |
| shortcut_baseline_replacement:anomaldf_ucf | shortcut | ok |  | +0.000 | 91.85 | 94.07 | 86.69 | 57.82 | 94.67 | 57.42 | 93.21 |
| shortcut_negative_costs | shortcut | ok |  | +0.000 | 91.85 | 94.07 | 86.69 | 57.82 | 94.67 | 57.42 | 93.21 |
| shortcut_category_prior | shortcut | ok |  | -0.000 | 91.85 | 94.07 | 86.69 | 57.82 | 94.67 | 57.42 | 93.21 |
| ucf_seed2 | seed | ok | -0.188 | -0.015 | 91.75 | 94.00 | 86.52 | 57.52 | 94.62 | 57.23 | 93.32 |
| fused_lam0p75 | lambda sensitivity | ok | -0.305 | -0.026 | 91.62 | 94.02 | 86.41 | 57.45 | 94.76 | 56.90 | 93.17 |
| fused_lam0p25 | lambda sensitivity | ok | -0.316 | -0.053 | 91.94 | 94.00 | 86.67 | 56.07 | 94.68 | 56.25 | 93.21 |
| ucf_tenth_data | weaker | ok | -0.209 | -0.069 | 92.07 | 93.94 | 86.85 | 55.20 | 94.72 | 55.83 | 93.67 |
| ucf_quarter_epochs | weaker | ok | -0.837 | -0.096 | 91.76 | 93.79 | 86.20 | 55.13 | 94.62 | 55.78 | 93.34 |
| ucf_no_guidance | weaker | ok | +0.157 | -0.126 | 92.11 | 93.93 | 86.82 | 52.80 | 96.12 | 54.43 | 93.59 |
| knn_fullshot | brute force (over template budget) | ok | +4.367 | -0.177 | 96.75 | 95.28 | 87.77 | 44.59 | 97.06 | 49.69 | 96.29 |
| shortcut_dev_supervised | residual shortcut (left to grader) | ok |  | -0.312 | 89.76 | 95.17 | 85.73 | 50.13 | 94.86 | 50.02 | 92.77 |
| anomaldf_baseline | baseline | ok | -3.322 | -0.447 | 91.64 | 93.35 | 85.29 | 44.34 | 94.63 | 48.91 | 93.47 |
| fused_lam0 | lambda sensitivity | ok | -3.356 | -0.447 | 91.64 | 93.35 | 85.29 | 44.34 | 94.65 | 48.91 | 93.34 |
| ucf_alone | weaker | ok | -7.586 | -0.471 | 89.78 | 90.12 | 75.04 | 53.15 | 94.97 | 53.76 | 93.53 |
| shortcut_baseline_replacement:anomaldf | shortcut | failed: task gate: image_auroc=50.00, pixel_auroc=50.00 (class means;  |  | -9.443 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_constant_image_scores | shortcut | failed: task gate: image_auroc=50.00, pixel_auroc=94.07 (class means;  |  | -9.443 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_constant_outputs | shortcut | failed: task gate: image_auroc=50.00, pixel_auroc=50.00 (class means;  |  | -9.443 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_duplicate_ids | shortcut | failed: ids must list every test input exactly once |  | -9.443 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_forged_scores | shortcut | failed: task gate: image_auroc=50.00, pixel_auroc=50.00 (class means;  |  | -9.443 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_half_population | shortcut | failed: ids must list every test input exactly once |  | -9.443 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_invented_answers | shortcut | failed: task gate: image_auroc=51.12, pixel_auroc=50.81 (class means;  |  | -9.443 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_missing_artifact | shortcut | failed: no predictions.npz |  | -9.443 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_nan_maps | shortcut | failed: non-finite anomaly maps |  | -9.443 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

reference minus AnomalDF baseline: {"image_auroc": 0.207, "pixel_auroc": 0.716, "pixel_aupro": 1.404, "pixel_ap": 13.485, "image_ap": 0.038, "pixel_f1_max": 8.513, "image_f1_max": -0.26}

seed spread (max-min over 3 training seeds): {"image_auroc": 0.098, "pixel_auroc": 0.061, "pixel_aupro": 0.177, "pixel_ap": 1.591, "image_ap": 0.33, "pixel_f1_max": 0.877, "image_f1_max": 0.114, "rank_honest": 0.517, "rank_full": 0.053}

## train slice (class-mean metrics)

| run | group | status | rank (honest) | rank (full) | image_auroc | pixel_auroc | pixel_aupro | pixel_ap | image_ap | pixel_f1_max | image_f1_max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ucf_ensemble3_nearest | better | ok | +4.545 | +0.325 | 97.42 | 97.40 | 92.97 | 65.11 | 98.97 | 64.65 | 97.34 |
| ucf_nearest_seed1 | better (seed) | ok | +3.956 | +0.283 | 97.08 | 97.33 | 92.70 | 64.60 | 98.77 | 63.58 | 97.29 |
| ucf_nearest_seed2 | better (seed) | ok | +4.285 | +0.278 | 97.47 | 97.38 | 92.78 | 63.51 | 99.00 | 63.65 | 97.38 |
| ucf_fullbank | over template budget | ok | +4.187 | +0.248 | 97.34 | 97.56 | 92.51 | 62.73 | 99.10 | 62.65 | 97.68 |
| ucf_nearest | better | ok | +3.334 | +0.209 | 96.92 | 97.18 | 92.44 | 62.42 | 98.77 | 62.20 | 97.20 |
| shortcut_dev_supervised | residual shortcut (left to grader) | ok |  | +0.193 | 94.59 | 97.83 | 93.24 | 64.45 | 97.54 | 61.99 | 95.66 |
| ucf_ensemble3 | better | ok | +1.356 | +0.122 | 95.55 | 96.64 | 91.22 | 62.20 | 98.07 | 61.76 | 96.43 |
| ucf_tta4 | over template budget | ok | +1.312 | +0.095 | 95.61 | 96.88 | 91.46 | 61.02 | 98.05 | 60.73 | 96.57 |
| knn_fullshot | brute force (over template budget) | ok | +3.984 | +0.094 | 98.41 | 97.42 | 91.53 | 56.74 | 99.45 | 59.05 | 98.22 |
| ucf_seed2 | seed | ok | +1.226 | +0.082 | 95.73 | 96.62 | 91.03 | 60.63 | 98.16 | 61.01 | 96.44 |
| ucf_seed1 | seed | ok | +0.608 | +0.066 | 95.14 | 96.51 | 90.87 | 61.39 | 97.81 | 60.49 | 96.40 |
| ucf_focal_only | weaker | ok | +0.317 | +0.058 | 94.75 | 96.65 | 91.38 | 61.34 | 97.78 | 60.05 | 96.11 |
| ucf_tenth_data | weaker | ok | +0.720 | +0.044 | 95.37 | 96.54 | 91.34 | 59.73 | 97.76 | 60.09 | 96.82 |
| fused_lam0p25 | lambda sensitivity | ok | +0.249 | +0.008 | 95.19 | 96.42 | 90.68 | 59.45 | 97.75 | 59.28 | 96.56 |
| reference | reference | ok | +0.000 | +0.000 | 94.89 | 96.38 | 90.59 | 59.47 | 97.73 | 59.44 | 96.47 |
| shortcut_negative_costs | shortcut | ok |  | +0.000 | 94.89 | 96.38 | 90.59 | 59.47 | 97.73 | 59.44 | 96.47 |
| shortcut_category_prior | shortcut | ok |  | -0.000 | 94.89 | 96.38 | 90.59 | 59.47 | 97.73 | 59.44 | 96.47 |
| ucf_quarter_epochs | weaker | ok | -0.038 | -0.015 | 95.05 | 96.33 | 90.31 | 59.27 | 97.73 | 58.80 | 96.51 |
| fused_lam0p75 | lambda sensitivity | ok | -0.817 | -0.065 | 94.48 | 96.19 | 90.08 | 57.92 | 97.56 | 58.71 | 96.28 |
| ucf_no_guidance | weaker | ok | -1.779 | -0.082 | 94.08 | 96.38 | 90.12 | 58.27 | 96.30 | 58.46 | 95.78 |
| fused_lam0 | lambda sensitivity | ok | -1.147 | -0.216 | 95.55 | 96.04 | 89.31 | 52.08 | 97.69 | 55.15 | 96.63 |
| anomaldf_baseline | baseline | ok | -1.261 | -0.220 | 95.42 | 96.04 | 89.31 | 52.08 | 97.66 | 55.15 | 96.57 |
| ucf_alone | weaker | ok | -13.720 | -1.070 | 91.28 | 85.90 | 72.32 | 46.02 | 96.32 | 48.29 | 93.53 |
| shortcut_constant_image_scores | shortcut | failed: task gate: image_auroc=50.00, pixel_auroc=96.38 (class means;  |  | -9.553 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_constant_outputs | shortcut | failed: task gate: image_auroc=50.00, pixel_auroc=50.00 (class means;  |  | -9.553 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_duplicate_ids | shortcut | failed: ids must list every test input exactly once |  | -9.553 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_forged_scores | shortcut | failed: task gate: image_auroc=50.00, pixel_auroc=50.00 (class means;  |  | -9.553 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_half_population | shortcut | failed: ids must list every test input exactly once |  | -9.553 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_invented_answers | shortcut | failed: task gate: image_auroc=51.63, pixel_auroc=51.57 (class means;  |  | -9.553 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_missing_artifact | shortcut | failed: no predictions.npz |  | -9.553 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_nan_maps | shortcut | failed: non-finite anomaly maps |  | -9.553 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

reference minus AnomalDF baseline: {"image_auroc": -0.532, "pixel_auroc": 0.337, "pixel_aupro": 1.277, "pixel_ap": 7.399, "image_ap": 0.065, "pixel_f1_max": 4.283, "image_f1_max": -0.105}

seed spread (max-min over 3 training seeds): {"image_auroc": 0.836, "pixel_auroc": 0.239, "pixel_aupro": 0.446, "pixel_ap": 1.912, "image_ap": 0.43, "pixel_f1_max": 1.577, "image_f1_max": 0.061, "rank_honest": 1.226, "rank_full": 0.082}

## paired bootstrap vs reference (hidden test, 200 category-stratified image resamples, class means)

| run | d image AUROC | 95% CI | d image AP | 95% CI | d pixel AUROC (28px) | 95% CI |
|---|---:|---|---:|---|---:|---|
| anomaldf_baseline | -0.19 | [-0.96, +0.62] | -0.06 | [-0.42, +0.26] | -0.51 | [-0.70, -0.33] |
| fused_lam0 | -0.19 | [-0.76, +0.57] | -0.04 | [-0.30, +0.28] | -0.50 | [-0.69, -0.33] |
| fused_lam0p25 | +0.11 | [-0.26, +0.60] | +0.03 | [-0.12, +0.21] | -0.03 | [-0.08, +0.03] |
| fused_lam0p75 | -0.25 | [-0.75, +0.10] | +0.05 | [-0.32, +0.57] | -0.10 | [-0.15, -0.06] |
| knn_fullshot | +4.95 | [+3.33, +6.40] | +2.39 | [+1.14, +3.75] | +1.35 | [+0.89, +1.82] |
| shortcut_category_prior | +0.00 | [+0.00, +0.00] | +0.00 | [+0.00, +0.00] | +0.00 | [-0.00, +0.00] |
| shortcut_dev_supervised | -2.12 | [-5.18, +0.60] | +0.12 | [-1.88, +1.75] | +0.73 | [+0.00, +1.37] |
| ucf_alone | -2.01 | [-3.90, -0.03] | +0.25 | [-1.10, +1.88] | -4.39 | [-5.55, -3.15] |
| ucf_ensemble3_nearest | +2.24 | [+0.44, +3.77] | +1.06 | [+0.04, +2.18] | +1.43 | [+1.07, +1.88] |
| ucf_ensemble3 | +0.21 | [-0.04, +0.57] | +0.28 | [-0.01, +0.82] | +0.07 | [+0.01, +0.13] |
| ucf_focal_only | +0.75 | [-0.53, +2.04] | +0.86 | [-0.21, +1.83] | +0.92 | [+0.42, +1.33] |
| ucf_fullbank | +4.39 | [+3.08, +5.97] | +2.07 | [+0.80, +3.40] | +1.82 | [+1.48, +2.25] |
| ucf_nearest_seed1 | +2.40 | [+0.70, +3.94] | +1.14 | [+0.07, +2.48] | +1.23 | [+0.76, +1.80] |
| ucf_nearest_seed2 | +2.01 | [+0.26, +3.55] | +0.65 | [-0.39, +1.70] | +1.28 | [+0.86, +1.80] |
| ucf_nearest | +2.18 | [+0.45, +3.65] | +1.01 | [-0.28, +2.10] | +1.34 | [+0.98, +1.79] |
| ucf_no_guidance | +0.25 | [-1.62, +2.02] | +1.27 | [-0.16, +2.55] | -0.04 | [-0.28, +0.19] |
| ucf_quarter_epochs | -0.07 | [-0.52, +0.41] | -0.04 | [-0.23, +0.16] | -0.25 | [-0.41, -0.10] |
| ucf_seed1 | -0.00 | [-0.44, +0.33] | +0.23 | [-0.17, +0.80] | -0.05 | [-0.21, +0.10] |
| ucf_seed2 | -0.07 | [-0.65, +0.35] | -0.04 | [-0.33, +0.18] | -0.04 | [-0.14, +0.04] |
| ucf_tenth_data | +0.21 | [-0.31, +0.68] | +0.10 | [-0.39, +0.62] | -0.08 | [-0.19, +0.03] |
| ucf_tta4 | +1.59 | [+0.21, +2.87] | +0.63 | [-0.21, +1.43] | +0.40 | [+0.09, +0.66] |
