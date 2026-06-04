# Overview

- **Paper ID:** 2510.03363
- **Title:** Unified Unsupervised Anomaly Detection via Matching Cost Filtering
- **Domain:** Computer Vision / Unsupervised Anomaly Detection
- **TL;DR:** The paper improves visual anomaly detection by learning to denoise template-matching cost volumes before producing anomaly maps.

## Short Summary

The paper argues that many anomaly detectors are fundamentally matching inputs against normal references, reconstructions, text prompts, or 3D features, but their matching costs are noisy. UCF builds a multi-layer anomaly cost volume and filters it with a guided 3D network. Across RGB, RGB-3D, and RGB-Text settings, the paper reports consistent improvements over strong baselines.

## Key Results

- On MVTec-AD RGB multi-class anomaly detection, AnomalDF improves from 96.8/98.1 image/pixel AUROC to 98.5/98.8 with UCF.
- On VisA RGB, AnomalDF improves from 90.5/97.5 image/pixel AUROC to 94.3/99.2.
- The largest reported gains are in stricter localization metrics such as AUPRO for RGB-3D and RGB-Text settings.
