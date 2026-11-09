# Attempt 2 (job 2570bd84-d0f, completed)

UCF readout v2: filtered cost volume keeps the D*N matching channels (residual on
last-layer costs) -> min over matching channels -> conv -> softmax; paper setup
(40 epochs, lr 1e-3, anomalies anywhere in the image). The filter still
over-fitted to synthetic anomalies: longer training lowered every AUROC, and
object categories collapsed (dev pixel AUROC metal_nut 48.7, transistor 63.5;
hidden screw image AUROC 29.3).

| run | slice | image_auroc | pixel_auroc | image_ap | image_f1_max | pixel_ap | pixel_f1_max | pixel_aupro |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| anomaldf | test | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| anomaldf_ucf | test | 79.944 | 87.395 | 93.267 | 84.514 | 60.582 | 61.376 | 60.449 |
| ucf_1epoch | test | 84.885 | 93.927 | 93.834 | 87.365 | 46.344 | 52.090 | 86.122 |
| ucf_n8_templates | test | 81.021 | 86.733 | 93.435 | 85.044 | 56.280 | 57.946 | 58.074 |
| ucf_nearest_templates | test | 80.978 | 86.902 | 93.583 | 84.514 | 60.705 | 61.744 | 61.246 |
| ucf_no_guidance | test | 80.974 | 90.178 | 92.641 | 85.351 | 46.941 | 52.335 | 73.089 |
| ucf_seed1 | test | 79.268 | 85.645 | 92.833 | 84.661 | 55.522 | 57.493 | 54.977 |
| ucf_seed2 | test | 81.302 | 87.341 | 93.450 | 84.626 | 59.407 | 61.152 | 57.422 |
| ucf_tta4 | test | 80.096 | 87.773 | 93.367 | 84.514 | 61.327 | 61.699 | 61.083 |
| anomaldf | train | 92.342 | 90.857 | 97.151 | 90.323 | 37.247 | 40.794 | 82.849 |
| anomaldf_ucf | train | 86.524 | 71.018 | 95.255 | 86.848 | 37.071 | 42.561 | 65.392 |
| ucf_1epoch | train | 87.562 | 73.467 | 95.167 | 88.399 | 30.444 | 38.724 | 87.968 |
| ucf_n8_templates | train | 84.906 | 70.864 | 94.644 | 85.168 | 35.700 | 41.585 | 63.892 |
| ucf_nearest_templates | train | 87.787 | 69.716 | 95.753 | 87.009 | 36.626 | 42.409 | 67.847 |
| ucf_no_guidance | train | 82.622 | 86.925 | 93.332 | 84.733 | 37.053 | 42.464 | 77.124 |
| ucf_seed1 | train | 82.727 | 65.421 | 93.878 | 84.599 | 30.339 | 36.919 | 59.911 |
| ucf_seed2 | train | 83.677 | 66.129 | 94.084 | 85.761 | 34.006 | 39.747 | 63.264 |
| ucf_tta4 | train | 87.111 | 71.052 | 95.454 | 87.262 | 37.182 | 42.384 | 66.477 |

Seed spread (max-min, hidden test, seeds 0/1/2): image AUROC 3.80, pixel AUROC 5.60, AUPRO 5.48, pixel AP 6.73.
Paired bootstrap (hidden, AnomalDF minus UCF): image AUROC +8.82 [6.35, 11.69], pixel AUROC(28px) +7.50 [5.44, 9.83].
