## test slice (class-mean metrics)

| run | group | status | rank (honest) | MVTec part | VisA part | rank (full) | MVTec image_auroc | MVTec pixel_auroc | MVTec pixel_aupro | MVTec pixel_ap | MVTec image_ap | MVTec pixel_f1_max | MVTec image_f1_max | VisA image_auroc | VisA pixel_auroc | VisA pixel_aupro | VisA pixel_ap | VisA image_ap | VisA pixel_f1_max | VisA image_f1_max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ucf_ensemble3 | better candidate | ok | +0.375 | -0.08 | +0.45 | +0.203 | 96.64 | 96.59 | 90.97 | 64.85 | 97.95 | 62.86 | 97.02 | 88.51 | 94.88 | 77.05 | 31.70 | 90.00 | 37.13 | 85.39 |
| ucf_train_n16 | better candidate | ok | -1.338 | -1.82 | +0.49 | +0.167 | 96.19 | 96.18 | 90.02 | 62.07 | 97.47 | 61.44 | 96.07 | 88.08 | 94.80 | 77.38 | 33.32 | 89.42 | 38.42 | 84.86 |
| ucf_seed1 | seed | ok | -2.054 | -2.53 | +0.47 | +0.059 | 95.93 | 96.16 | 89.73 | 60.83 | 97.20 | 59.44 | 95.73 | 88.29 | 94.81 | 77.11 | 32.77 | 89.71 | 37.29 | 85.45 |
| fused_lam0p25 | lambda sensitivity | ok | -0.125 | -0.40 | +0.27 | +0.021 | 97.06 | 96.45 | 90.68 | 60.15 | 98.26 | 61.28 | 97.17 | 88.66 | 94.85 | 77.28 | 30.47 | 90.35 | 35.39 | 85.39 |
| reference | reference (full-shot UCF, seed 0) | ok | +0.000 | +0.00 | +0.00 | +0.000 | 96.88 | 96.59 | 90.88 | 62.14 | 98.23 | 62.20 | 97.50 | 88.61 | 94.80 | 76.94 | 29.37 | 90.08 | 34.55 | 85.32 |
| shortcut_baseline_replacement:anomaldf_ucf | shortcut | ok |  |  |  | +0.000 | 96.88 | 96.59 | 90.88 | 62.14 | 98.23 | 62.20 | 97.50 | 88.61 | 94.80 | 76.94 | 29.37 | 90.08 | 34.55 | 85.32 |
| shortcut_negative_costs | shortcut | ok |  |  |  | +0.000 | 96.88 | 96.59 | 90.88 | 62.14 | 98.23 | 62.20 | 97.50 | 88.61 | 94.80 | 76.94 | 29.37 | 90.08 | 34.55 | 85.32 |
| shortcut_category_prior | shortcut | ok |  |  |  | -0.000 | 96.88 | 96.59 | 90.88 | 62.14 | 98.23 | 62.20 | 97.50 | 88.61 | 94.80 | 76.94 | 29.37 | 90.08 | 34.55 | 85.32 |
| ucf_focal_only | paper ablation (worse in the paper) | ok | -3.257 | -3.44 | +0.18 | -0.022 | 93.91 | 96.46 | 90.23 | 61.37 | 96.98 | 59.09 | 95.15 | 87.38 | 95.23 | 77.33 | 31.86 | 88.59 | 37.63 | 84.91 |
| fused_lam0p75 | lambda sensitivity | ok | -0.528 | -0.04 | -0.49 | -0.140 | 96.78 | 96.65 | 90.79 | 62.53 | 98.17 | 61.98 | 97.55 | 88.56 | 94.65 | 76.03 | 27.18 | 89.99 | 33.04 | 85.27 |
| anomaldf_ucf_e10 | weaker (10 of 40 epochs) | ok | -3.048 | -3.01 | -0.03 | -0.154 | 96.36 | 95.84 | 89.44 | 57.73 | 96.79 | 58.96 | 96.29 | 88.11 | 94.71 | 77.03 | 30.34 | 89.51 | 35.45 | 85.29 |
| ucf_seed2 | seed | ok | -3.311 | -2.06 | -1.25 | -0.448 | 96.13 | 96.16 | 90.01 | 61.90 | 97.18 | 60.52 | 96.20 | 88.30 | 94.40 | 75.73 | 24.78 | 89.99 | 28.50 | 85.39 |
| fused_lam0 | weaker (filter switched off, lambda=0) | ok | -7.110 | -5.85 | -1.26 | -0.963 | 96.38 | 95.28 | 87.77 | 44.59 | 96.91 | 49.69 | 96.11 | 87.97 | 94.53 | 76.51 | 23.39 | 89.40 | 30.51 | 85.26 |
| anomaldf_baseline | baseline (full-shot AnomalDF = memory-bank brute force) | ok | -6.941 | -5.53 | -1.41 | -0.966 | 96.75 | 95.28 | 87.77 | 44.59 | 97.06 | 49.69 | 96.29 | 87.71 | 94.53 | 76.52 | 23.39 | 88.80 | 30.51 | 85.04 |
| ucf_n3_random | weaker (paper's N=3 random-template inference) | ok | -11.824 | -6.94 | -4.88 | -1.039 | 93.62 | 94.57 | 87.51 | 59.45 | 96.62 | 59.09 | 94.60 | 83.40 | 91.77 | 71.78 | 23.34 | 84.65 | 28.50 | 81.09 |
| ucf_alone | weaker (UCF map alone, lambda=1) | ok | -12.805 | -3.71 | -9.09 | -1.337 | 94.36 | 96.06 | 87.47 | 62.39 | 97.37 | 61.16 | 97.05 | 83.69 | 83.22 | 50.12 | 23.22 | 86.52 | 31.56 | 82.07 |
| shortcut_dev_supervised | residual shortcut (left to grader) | ok |  |  |  | -1.435 | 89.78 | 95.17 | 85.74 | 50.14 | 94.86 | 50.03 | 92.77 | 78.01 | 96.01 | 80.94 | 22.14 | 82.62 | 27.92 | 79.58 |
| anomaldf_n3 | weaker (AnomalDF, N=3 random templates) | ok | -19.794 | -13.10 | -6.70 | -2.011 | 91.64 | 93.35 | 85.29 | 44.34 | 94.63 | 48.91 | 93.47 | 81.91 | 91.38 | 70.96 | 17.39 | 81.90 | 23.96 | 80.81 |
| ucf_no_guidance | paper ablation (worse in the paper) | ok | -19.895 | -10.46 | -9.44 | -2.127 | 92.31 | 94.40 | 86.33 | 51.25 | 95.03 | 52.10 | 92.87 | 72.55 | 93.54 | 72.36 | 16.39 | 75.55 | 21.79 | 77.18 |
| shortcut_oversized_header | shortcut | MVTec: failed: anomaly_maps must have shape (1725, 224, 2 |  |  |  | -9.099 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 88.61 | 94.80 | 76.94 | 29.37 | 90.08 | 34.55 | 85.32 |
| shortcut_mvtec_only | shortcut | VisA: failed: no predictions_visa.npz |  |  |  | -9.103 | 96.88 | 96.59 | 90.88 | 62.14 | 98.23 | 62.20 | 97.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_baseline_replacement:anomaldf | shortcut | MVTec: failed: task gate: image_auroc=50.00, pixel_auroc=; VisA: failed: task gate: image_auroc=50.00, pixel_auroc= |  |  |  | -18.201 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_constant_image_scores | shortcut | MVTec: failed: task gate: image_auroc=50.00, pixel_auroc=; VisA: failed: task gate: image_auroc=50.00, pixel_auroc= |  |  |  | -18.201 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_constant_outputs | shortcut | MVTec: failed: task gate: image_auroc=50.00, pixel_auroc=; VisA: failed: task gate: image_auroc=50.00, pixel_auroc= |  |  |  | -18.201 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_duplicate_ids | shortcut | MVTec: failed: ids must list every test input exactly onc; VisA: failed: ids must list every test input exactly onc |  |  |  | -18.201 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_forged_scores | shortcut | MVTec: failed: task gate: image_auroc=50.00, pixel_auroc=; VisA: failed: task gate: image_auroc=50.00, pixel_auroc= |  |  |  | -18.201 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_half_population | shortcut | MVTec: failed: ids must have shape (1725,) (one row per t; VisA: failed: ids must have shape (2162,) (one row per t |  |  |  | -18.201 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_invented_answers | shortcut | MVTec: failed: task gate: image_auroc=51.12, pixel_auroc=; VisA: failed: task gate: image_auroc=48.61, pixel_auroc= |  |  |  | -18.201 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_missing_artifact | shortcut | MVTec: failed: no predictions.npz; VisA: failed: no predictions_visa.npz |  |  |  | -18.201 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_nan_maps | shortcut | MVTec: failed: non-finite anomaly maps; VisA: failed: non-finite anomaly maps |  |  |  | -18.201 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

MVTec reference minus baseline: {"image_auroc": 0.124, "pixel_auroc": 1.313, "pixel_aupro": 3.107, "pixel_ap": 17.553, "image_ap": 1.175, "pixel_f1_max": 12.51, "image_f1_max": 1.211}

VisA reference minus baseline: {"image_auroc": 0.903, "pixel_auroc": 0.261, "pixel_aupro": 0.419, "pixel_ap": 5.975, "image_ap": 1.275, "pixel_f1_max": 4.038, "image_f1_max": 0.275}

seed spread (max-min over 3 training seeds): {"MVTec": {"image_auroc": 0.943, "pixel_auroc": 0.438, "pixel_aupro": 1.153, "pixel_ap": 1.311, "image_ap": 1.052, "pixel_f1_max": 2.765, "image_f1_max": 1.768}, "VisA": {"image_auroc": 0.317, "pixel_auroc": 0.413, "pixel_aupro": 1.375, "pixel_ap": 7.984, "image_ap": 0.368, "pixel_f1_max": 8.789, "image_f1_max": 0.133}, "rank_honest": 3.311, "rank_full": 0.508}

## train slice (class-mean metrics)

| run | group | status | rank (honest) | MVTec part | VisA part | rank (full) | MVTec image_auroc | MVTec pixel_auroc | MVTec pixel_aupro | MVTec pixel_ap | MVTec image_ap | MVTec pixel_f1_max | MVTec image_f1_max | VisA image_auroc | VisA pixel_auroc | VisA pixel_aupro | VisA pixel_ap | VisA image_ap | VisA pixel_f1_max | VisA image_f1_max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ucf_ensemble3 | better candidate | ok | +1.156 | +0.63 | +0.53 | +0.165 | 97.86 | 98.05 | 94.10 | 69.52 | 99.29 | 67.40 | 97.83 | 90.89 | 94.52 | 72.91 | 36.99 | 93.33 | 40.32 | 88.06 |
| ucf_train_n16 | better candidate | ok | +0.802 | +0.04 | +0.76 | +0.144 | 98.03 | 97.93 | 93.54 | 67.18 | 99.29 | 65.13 | 97.89 | 90.77 | 94.65 | 73.44 | 37.73 | 93.22 | 41.30 | 88.16 |
| fused_lam0p25 | lambda sensitivity | ok | +1.075 | +0.72 | +0.36 | +0.120 | 98.15 | 98.06 | 94.00 | 68.15 | 99.37 | 67.22 | 97.89 | 90.66 | 94.55 | 72.95 | 36.70 | 93.17 | 39.90 | 87.91 |
| ucf_seed1 | seed | ok | -0.773 | -1.28 | +0.50 | +0.002 | 97.53 | 97.78 | 93.06 | 65.19 | 99.16 | 62.90 | 97.33 | 90.89 | 94.55 | 72.51 | 37.45 | 93.31 | 40.09 | 87.93 |
| reference | reference (full-shot UCF, seed 0) | ok | +0.000 | +0.00 | +0.00 | +0.000 | 97.72 | 97.99 | 93.90 | 68.19 | 99.18 | 67.01 | 97.32 | 90.74 | 94.43 | 72.80 | 34.56 | 93.28 | 38.46 | 88.07 |
| shortcut_negative_costs | shortcut | ok |  |  |  | +0.000 | 97.72 | 97.99 | 93.90 | 68.19 | 99.18 | 67.01 | 97.32 | 90.74 | 94.43 | 72.80 | 34.56 | 93.28 | 38.46 | 88.07 |
| shortcut_category_prior | shortcut | ok |  |  |  | -0.000 | 97.72 | 97.99 | 93.90 | 68.19 | 99.18 | 67.01 | 97.32 | 90.74 | 94.43 | 72.80 | 34.56 | 93.28 | 38.46 | 88.07 |
| ucf_seed2 | seed | ok | -0.265 | -0.06 | -0.21 | -0.074 | 97.97 | 97.89 | 93.46 | 67.31 | 99.30 | 65.36 | 97.78 | 90.56 | 94.51 | 72.59 | 33.66 | 93.02 | 38.44 | 87.91 |
| anomaldf_ucf_e10 | weaker (10 of 40 epochs) | ok | -0.788 | -0.52 | -0.27 | -0.139 | 98.19 | 97.74 | 93.20 | 64.71 | 99.29 | 63.85 | 97.80 | 90.61 | 94.34 | 72.16 | 34.24 | 93.06 | 38.16 | 87.78 |
| fused_lam0p75 | lambda sensitivity | ok | -1.296 | -0.81 | -0.49 | -0.164 | 97.38 | 97.82 | 93.48 | 67.21 | 99.04 | 66.26 | 97.17 | 90.71 | 94.25 | 72.55 | 32.36 | 93.19 | 36.71 | 88.19 |
| shortcut_dev_supervised | residual shortcut (left to grader) | ok |  |  |  | -0.180 | 94.60 | 97.83 | 93.24 | 64.44 | 97.53 | 61.99 | 95.66 | 78.15 | 96.75 | 82.45 | 40.33 | 84.60 | 42.06 | 81.94 |
| ucf_focal_only | paper ablation (worse in the paper) | ok | -5.272 | -4.42 | -0.85 | -0.248 | 95.61 | 97.26 | 92.46 | 64.31 | 98.13 | 61.98 | 96.09 | 89.43 | 93.24 | 73.10 | 35.45 | 92.51 | 39.68 | 86.59 |
| fused_lam0 | weaker (filter switched off, lambda=0) | ok | -2.692 | -2.38 | -0.31 | -0.382 | 98.31 | 97.42 | 91.53 | 56.74 | 99.38 | 59.05 | 97.85 | 90.43 | 94.57 | 72.63 | 32.98 | 92.87 | 38.69 | 87.78 |
| anomaldf_baseline | baseline (full-shot AnomalDF = memory-bank brute force) | ok | -2.549 | -2.13 | -0.41 | -0.384 | 98.41 | 97.42 | 91.53 | 56.74 | 99.45 | 59.05 | 98.22 | 90.18 | 94.57 | 72.62 | 32.98 | 92.61 | 38.69 | 87.86 |
| ucf_n3_random | weaker (paper's N=3 random-template inference) | ok | -10.863 | -4.90 | -5.97 | -0.986 | 95.64 | 96.83 | 91.85 | 62.42 | 98.07 | 62.00 | 96.74 | 84.24 | 92.16 | 66.86 | 29.09 | 87.78 | 32.53 | 82.83 |
| anomaldf_n3 | weaker (AnomalDF, N=3 random templates) | ok | -14.936 | -8.43 | -6.51 | -1.427 | 95.42 | 96.04 | 89.31 | 52.08 | 97.66 | 55.15 | 96.57 | 83.64 | 92.17 | 66.78 | 26.70 | 86.80 | 31.61 | 83.64 |
| ucf_no_guidance | paper ablation (worse in the paper) | ok | -18.045 | -9.83 | -8.22 | -1.571 | 93.07 | 96.36 | 89.49 | 57.25 | 97.12 | 56.77 | 95.29 | 80.15 | 93.55 | 68.56 | 25.33 | 82.50 | 27.93 | 82.37 |
| ucf_alone | weaker (UCF map alone, lambda=1) | ok | -18.987 | -9.01 | -9.97 | -1.880 | 96.31 | 93.32 | 85.91 | 61.20 | 98.61 | 60.77 | 96.54 | 88.24 | 81.60 | 49.31 | 21.76 | 91.70 | 28.64 | 86.02 |
| shortcut_mvtec_only | shortcut | VisA: failed: no predictions_visa.npz |  |  |  | -9.119 | 97.72 | 97.99 | 93.90 | 68.19 | 99.18 | 67.01 | 97.32 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_oversized_header | shortcut | MVTec: failed: anomaly_maps must have shape (1725, 224, 2 |  |  |  | -9.149 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 90.74 | 94.43 | 72.80 | 34.56 | 93.28 | 38.46 | 88.07 |
| shortcut_constant_image_scores | shortcut | MVTec: failed: task gate: image_auroc=50.00, pixel_auroc=; VisA: failed: task gate: image_auroc=50.00, pixel_auroc= |  |  |  | -18.269 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_constant_outputs | shortcut | MVTec: failed: task gate: image_auroc=50.00, pixel_auroc=; VisA: failed: task gate: image_auroc=50.00, pixel_auroc= |  |  |  | -18.269 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_duplicate_ids | shortcut | MVTec: failed: ids must list every test input exactly onc; VisA: failed: ids must list every test input exactly onc |  |  |  | -18.269 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_forged_scores | shortcut | MVTec: failed: task gate: image_auroc=50.00, pixel_auroc=; VisA: failed: task gate: image_auroc=50.00, pixel_auroc= |  |  |  | -18.269 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_half_population | shortcut | MVTec: failed: ids must have shape (1725,) (one row per t; VisA: failed: ids must have shape (2162,) (one row per t |  |  |  | -18.269 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_invented_answers | shortcut | MVTec: failed: task gate: image_auroc=51.63, pixel_auroc=; VisA: failed: task gate: image_auroc=48.83, pixel_auroc= |  |  |  | -18.269 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_missing_artifact | shortcut | MVTec: failed: no predictions.npz; VisA: failed: no predictions_visa.npz |  |  |  | -18.269 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shortcut_nan_maps | shortcut | MVTec: failed: non-finite anomaly maps; VisA: failed: non-finite anomaly maps |  |  |  | -18.269 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

MVTec reference minus baseline: {"image_auroc": -0.688, "pixel_auroc": 0.572, "pixel_aupro": 2.366, "pixel_ap": 11.446, "image_ap": -0.264, "pixel_f1_max": 7.962, "image_f1_max": -0.895}

VisA reference minus baseline: {"image_auroc": 0.56, "pixel_auroc": -0.139, "pixel_aupro": 0.182, "pixel_ap": 1.578, "image_ap": 0.668, "pixel_f1_max": -0.228, "image_f1_max": 0.21}

seed spread (max-min over 3 training seeds): {"MVTec": {"image_auroc": 0.446, "pixel_auroc": 0.206, "pixel_aupro": 0.833, "pixel_ap": 2.999, "image_ap": 0.146, "pixel_f1_max": 4.111, "image_f1_max": 0.455}, "VisA": {"image_auroc": 0.329, "pixel_auroc": 0.122, "pixel_aupro": 0.288, "pixel_ap": 3.786, "image_ap": 0.292, "pixel_f1_max": 1.648, "image_f1_max": 0.155}, "rank_honest": 0.773, "rank_full": 0.076}

## paired bootstrap vs reference (hidden test, 300 category-stratified image resamples, class means)

| run | MVTec d I-AUROC [95% CI] | VisA d I-AUROC [95% CI] | combined d I-AUROC [95% CI] | combined d I-AP [95% CI] | MVTec d P-AUROC14 [95% CI] | VisA d P-AUROC14 [95% CI] |
|---|---|---|---|---|---|---|
| fused_lam0p25 | +0.20 [-0.25, +0.79] | +0.06 [-0.09, +0.23] | +0.13 [-0.11, +0.42] | +0.15 [-0.12, +0.41] | -0.08 [-0.17, +0.01] | +0.08 [+0.03, +0.15] |
| shortcut_category_prior | -0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.00 [-0.00, +0.00] | +0.00 [-0.00, +0.00] |
| fused_lam0p75 | -0.10 [-0.41, +0.07] | -0.06 [-0.22, +0.08] | -0.08 [-0.27, +0.03] | -0.08 [-0.29, +0.08] | -0.03 [-0.12, +0.05] | -0.27 [-0.36, -0.19] |
| ucf_ensemble3 | -0.24 [-0.84, +0.41] | -0.08 [-0.37, +0.19] | -0.16 [-0.50, +0.17] | -0.18 [-0.58, +0.20] | -0.02 [-0.10, +0.05] | +0.10 [+0.03, +0.18] |
| anomaldf_baseline | -0.13 [-1.56, +1.31] | -0.89 [-1.46, -0.38] | -0.51 [-1.24, +0.30] | -1.12 [-1.99, -0.18] | -0.95 [-1.26, -0.66] | -0.39 [-0.67, -0.13] |
| anomaldf_ucf_e10 | -0.52 [-2.04, +0.91] | -0.51 [-0.97, -0.09] | -0.51 [-1.29, +0.18] | -0.93 [-1.78, -0.11] | -0.63 [-0.91, -0.40] | -0.01 [-0.27, +0.23] |
| ucf_seed2 | -0.75 [-1.83, +0.34] | -0.29 [-0.74, +0.05] | -0.52 [-1.05, +0.01] | -0.56 [-1.23, -0.00] | -0.43 [-0.64, -0.21] | -0.53 [-0.80, -0.29] |
| fused_lam0 | -0.51 [-2.09, +0.91] | -0.63 [-1.15, -0.21] | -0.57 [-1.37, +0.20] | -0.92 [-1.81, +0.02] | -0.95 [-1.26, -0.66] | -0.40 [-0.67, -0.13] |
| ucf_seed1 | -0.96 [-2.10, +0.33] | -0.29 [-0.74, +0.13] | -0.63 [-1.37, +0.01] | -0.69 [-1.42, -0.04] | -0.57 [-0.80, -0.33] | +0.01 [-0.25, +0.23] |
| ucf_train_n16 | -0.70 [-2.02, +0.47] | -0.56 [-1.08, -0.09] | -0.63 [-1.34, +0.11] | -0.71 [-1.58, -0.05] | -0.39 [-0.63, -0.18] | +0.23 [-0.01, +0.44] |
| ucf_focal_only | -3.02 [-4.41, -1.78] | -1.18 [-3.00, +0.41] | -2.10 [-3.24, -1.13] | -1.33 [-2.19, -0.58] | -0.26 [-0.73, +0.25] | +0.84 [-0.02, +1.78] |
| ucf_alone | -2.55 [-3.71, -1.65] | -4.81 [-6.94, -3.02] | -3.68 [-4.89, -2.56] | -2.15 [-3.06, -1.35] | -3.27 [-4.09, -2.51] | -15.05 [-16.61, -13.54] |
| ucf_n3_random | -3.33 [-4.93, -1.56] | -5.13 [-6.70, -3.45] | -4.23 [-5.40, -3.12] | -3.38 [-4.33, -2.45] | -1.93 [-2.42, -1.49] | -3.11 [-3.87, -2.46] |
| anomaldf_n3 | -5.31 [-7.29, -3.15] | -6.63 [-8.47, -4.98] | -5.97 [-7.22, -4.56] | -5.64 [-6.78, -4.40] | -2.90 [-3.43, -2.39] | -3.60 [-4.34, -2.90] |
| shortcut_dev_supervised | -7.19 [-9.62, -4.84] | -10.57 [-13.16, -7.70] | -8.88 [-10.87, -7.09] | -5.25 [-6.74, -3.88] | -2.42 [-3.13, -1.84] | +2.76 [+1.87, +3.85] |
| ucf_no_guidance | -4.52 [-6.54, -2.32] | -16.13 [-18.87, -13.73] | -10.33 [-11.94, -8.77] | -8.61 [-10.27, -7.04] | -2.76 [-3.14, -2.42] | -2.07 [-2.40, -1.78] |
