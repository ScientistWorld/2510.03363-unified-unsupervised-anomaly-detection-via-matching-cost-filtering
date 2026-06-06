# Experiment Overview

## rgb_mvtec_multiclass

Tests the central visual anomaly-detection claim in a compact multi-class RGB setting. The experiment asks whether the reproduced system can improve anomaly image ranking and pixel localization over a direct template-matching baseline while keeping the same frozen feature source and normal-template data.

## rgb_synthetic_seed13

Optional robustness setting for the compact benchmark. It repeats the same normal-only anomaly-detection evaluation with a different deterministic training seed and output artifact, checking whether the measured improvement is not tied to a single initialization.
