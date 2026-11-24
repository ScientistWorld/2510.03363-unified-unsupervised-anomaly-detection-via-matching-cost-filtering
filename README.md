# Unified Unsupervised Anomaly Detection via Matching Cost Filtering

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2510.03363"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2510.03363-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2510.03363](https://arxiv.org/abs/2510.03363) |
|:---|:---|
| **Domain** | Computer Vision / Unsupervised Anomaly Detection |
| **TL;DR** | The paper improves visual anomaly detection by learning to denoise template-matching cost volumes before producing anomaly maps. |

## Summary

The paper argues that many anomaly detectors are fundamentally matching inputs against normal references, reconstructions, text prompts, or 3D features, but their matching costs are noisy. UCF builds a multi-layer anomaly cost volume and filters it with a guided 3D network. Across RGB, RGB-3D, and RGB-Text settings, the paper reports consistent improvements over strong baselines.

## Problem

Detect anomalous images and localize anomalous pixels when training data contains only normal examples. The target setting is multi-class visual anomaly detection: a single system must handle multiple visual categories and produce both an image-level anomaly score and a pixel-level anomaly heatmap.

## The paper's method

Unified Cost Filtering (UCF) treats visual anomaly detection as matching-cost filtering. Instead of directly thresholding noisy feature distances, it constructs a patch-wise anomaly cost volume and learns to filter that volume using guidance from the input image and an initial anomaly map.

## Key results of the paper

- On MVTec-AD RGB multi-class anomaly detection, AnomalDF improves from 96.8/98.1 image/pixel AUROC to 98.5/98.8 with UCF.
- On VisA RGB, AnomalDF improves from 90.5/97.5 image/pixel AUROC to 94.3/99.2.
- The largest reported gains are in stricter localization metrics such as AUPRO for RGB-3D and RGB-Text settings.

## Run it

```bash
pip install sciworld
sw research 2510.03363 --mode innovate --agent claude-code
sw evaluate 2510.03363
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

