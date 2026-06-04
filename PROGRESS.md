# Progress

## What Works

Paper analysis is complete for the core UCF pipeline. The workspace now contains a compact RGB reproduction with deterministic procedural anomaly data, UCF cost-volume construction, guided 3D filtering, training, artifact-writing, and method-agnostic evaluation.

`python validate.py --reference-only` passes. A local tiny forward pass was attempted, but the login environment does not have NumPy installed; dependencies are declared in `environment/setup.sh` and will be installed in the job container.

## Results

No reproduced scores yet. The next submitted job runs `scripts/run.sh`, which trains the compact UCF model and writes `scoring/scores.json`.

## Remaining

Run the first compute job and inspect whether UCF improves over the raw template-matching baseline. If the method runs but does not beat the baseline, adjust training duration, loss weighting, or procedural anomaly difficulty while preserving the UCF algorithm.

## Issues

The exact MVTec-AD dataset and DINOv2/AnomalDF backbone are not used in this first pass to avoid large downloads and GPU-heavy feature extraction. The reproduction instead targets the UCF algorithm itself in a compact RGB setting.

## Deviations from Paper

- The paper evaluates on MVTec-AD with DINOv2/AnomalDF features and normal-image templates. This reproduction uses a deterministic procedural multi-class RGB anomaly dataset and a frozen handcrafted feature pyramid. This is a scale/resource substitution for the feature source and data, not a replacement of the UCF computation: the implementation still builds hierarchical patch-template cosine cost volumes, applies guided 3D cost filtering, trains on synthetic anomaly masks, and evaluates baseline versus UCF maps.
- The paper trains for 40 epochs with batch size 8. The first submitted job uses fewer epochs for a smoke-scale reproduction. The script keeps epochs configurable through `EPOCHS`.
