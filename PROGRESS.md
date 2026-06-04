# Progress

## What Works

Paper analysis is complete for the core UCF pipeline. The workspace now contains a compact RGB reproduction with deterministic procedural anomaly data, UCF cost-volume construction, guided 3D filtering, training, artifact-writing, and method-agnostic evaluation.

`python validate.py --reference-only` passes. A local tiny forward pass was attempted, but the login environment does not have NumPy installed; dependencies are declared in `environment/setup.sh` and are installed in the job container.

The first compute job successfully entered training and the loss decreased from 1.3078 to 0.2122 over six epochs, but the CPU test job timed out before evaluation. This indicates the implementation is structurally valid but the initial compact scale was still too slow for a one-hour CPU smoke run.

The second compute job completed training and wrote prediction artifacts plus an initial `scores.json`, then failed in diagnostic AUPRO reporting because NumPy 2 no longer exposes `np.trapz`. The bug is fixed by using `np.trapezoid` with a fallback. The preliminary 4-epoch scores were image AUROC 51.7 to 52.7 but pixel AUROC 71.1 to 69.9, so UCF did not yet support the core claim.

## Results

No successful reproduced scores yet. A failed 4-epoch run produced preliminary artifacts, but the next submitted job runs a longer reduced-scale training pass and should exit cleanly with `scoring/scores.json`.

## Remaining

Run the longer reduced compute job and inspect whether UCF improves over the raw template-matching baseline. If the method runs but does not beat the baseline, adjust loss weighting or procedural anomaly difficulty while preserving the UCF algorithm.

## Issues

The exact MVTec-AD dataset and DINOv2/AnomalDF backbone are not used in this first pass to avoid large downloads and GPU-heavy feature extraction. The reproduction instead targets the UCF algorithm itself in a compact RGB setting.

## Deviations from Paper

- The paper evaluates on MVTec-AD with DINOv2/AnomalDF features and normal-image templates. This reproduction uses a deterministic procedural multi-class RGB anomaly dataset and a frozen handcrafted feature pyramid. This is a scale/resource substitution for the feature source and data, not a replacement of the UCF computation: the implementation still builds hierarchical patch-template cosine cost volumes, applies guided 3D cost filtering, trains on synthetic anomaly masks, and evaluates baseline versus UCF maps.
- The paper trains for 40 epochs with batch size 8. The submitted smoke jobs use fewer epochs and smaller procedural splits to fit the CPU test-node limit. The script keeps epochs configurable through `EPOCHS`.
