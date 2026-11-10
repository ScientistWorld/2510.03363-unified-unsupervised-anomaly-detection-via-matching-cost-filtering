# Hardening report

## Verdict

Better results now score higher and worse ones lower on the hidden test. Every shortcut in `REDTEAM.md` is closed: each one now fails (scores 0 on every metric) or ties the reference exactly. Two shortcuts remain that the evaluator cannot see, and both are left to the grader (see "Shortcuts left"). The reproduction shows the paper's claim clearly at pixel level. At image level the gain points the right way but is inside the test's noise. That is stated plainly below.

## What changed

The gym the red team audited was a compact procedural RGB substitute. Its evaluator trusted method-written labels, masks, ids and costs, and its generator could rebuild the test answers. Earlier in this iteration it was rebuilt:

- **Data.** It now uses MVTec-AD, the paper's own dataset: 15 categories at 224 px, one model for all, trained on the 3,629 normal images only. The 1,725 test images are given to methods under anonymous ids, with category but no label or mask.
- **Split and answers.** The test set is split into a visible development half (845 images) and a hidden half (880 images). Answers for the hidden half exist only under `eval/test/`, as do the archive and the only code that can rebuild them. Whole defect types are assigned to one half, so the hidden half holds defect types the development half never shows.
- **What the evaluator reads.** It reads only `ids`, `image_scores` and `anomaly_maps`, and recomputes every metric. Every metric is a class-wise mean, as in the paper's tables.
- **Failure rules.** A run fails, scoring 0 on every metric, if it does not list every test id exactly once, has a wrong shape or dtype, contains a non-finite value, or has class-mean image AUROC or pixel AUROC below 70.
- **Costs.** Self-reported parameter and FLOP counts were removed from scoring, because nothing can recompute them.
- **Method.** The reproduction is AnomalDF+UCF on a frozen DINOv2 ViT-S/14, restored to the paper's own pipeline:
  - inference fused with λ = 0.5;
  - image score from the top-250 values of the map;
  - the class-aware adaptor's γ rule;
  - N = 3 random same-category templates;
  - 40 epochs, batch 8, Adam with a plateau scheduler.
- **Setup choice.** Epochs, learning rate and whether synthetic anomalies are kept to the object foreground were chosen on the development half only (`method/ucf_setup.json`).

This session closed the remaining contract gaps, rewrote the stale documentation, and re-checked everything in one CPU job, `ca3feb76-e4a` (40 min, exit 0, `validate.py` passed):

1. **A missing method directory scores as failed.** `scripts/evaluate_{train,test}.sh` and `scripts/evaluate.sh` now score every directory under `outputs/` (symlinked ones included). They also score every method named in `scoring/reference.json` or `scoring/scores.json`. Before, a run that crashed before creating `outputs/<method>/` was missing from `scores_test.json`, and the rank skips a missing entry, so failing cost nothing. Now it scores 0 everywhere.
2. **No artifact can stop scoring for the others.** Before, two directories whose names map to the same method key (e.g. `anomaldf-ucf` next to `anomaldf_ucf`) made the evaluator exit before writing any score for any method. Now both are scored. The directory named exactly after the key keeps it, and the other becomes `<key>__dup1`.
3. **Array shapes are checked from the `.npy` headers before any data is read.** A header claiming about 400 GB is rejected immediately.
4. **Docs rewritten.** `briefing/evaluation.md`, `scoring/{TARGETS,CONSTRAINTS,DIRECTION,EXPERIMENTS}.md` still described the procedural gym, including its parameter/FLOP constraint metrics. They now describe the MVTec-AD gym and its failure rules. `CONSTRAINTS.md` states the paper's operating point precisely:
   - normal-only training;
   - the frozen ViT-S/14 at 224 px;
   - at most N = 3 normal training images per test input, chosen with at most one global descriptor per training image;
   - each test input scored on its own.
5. **Rescoring.** All 60 saved study rows (30 artifacts × 2 halves) were rescored with the changed evaluator. Every metric is identical to the previous scoring. The only change is the failure message of one shortcut, `shortcut_half_population`, whose ids are now rejected by the header check before the id check. `scoring/scores_{train,test}.json` and `scores.json` were regenerated from the reference runs (AnomalDF, AnomalDF+UCF) and are identical to before.

The reference method, the baseline and the data were not changed in this session. All reference, variant and shortcut artifacts were produced from the final code in job `1e7c6766-a3f`, and the code those runs depend on has not changed since.

## Do better results score higher?

The rank is Σ |coef| · (attempt − reference) / std. Coefficients come from `scoring/reference.json`. The std is taken over the 20 honest hidden-test runs listed below. "Full" also counts the 13 shortcut rows; failed rows drag its std up.

**Before (red-team audit, procedural gym, honest cohort):**

| variant | expected | rank |
|---|---|---:|
| normal-only patch calibration | better | +12.69 |
| 1 epoch instead of 40 | worse | −0.89 |
| filtering removed | worse | **+0.61** |
| 160 epochs | — | −1.32 |
| fresh reference, seeds 7 / 13 / 23 | equal | −0.33 / +0.48 / −0.18 (spread 0.82) |

There, removing the paper's filter scored above the reference. The reference's image AUROC was below the baseline's (55.3 vs 58.0) and near chance.

**After (hidden half, 880 images):**

| run | expected | rank (honest) | rank (full) | I-AUROC | P-AUROC | AUPRO | P-AP |
|---|---|---:|---:|---:|---:|---:|---:|
| ucf_ensemble3_nearest (3 seeds, 3 nearest templates) | better | **+4.54** | +0.22 | 94.02 | 95.49 | 88.99 | 60.30 |
| ucf_nearest_seed1 | better | **+4.40** | +0.20 | 94.22 | 95.28 | 88.75 | 59.65 |
| ucf_nearest (3 nearest templates instead of 3 random) | better | **+4.03** | +0.16 | 93.96 | 95.42 | 88.73 | 58.43 |
| ucf_nearest_seed2 | better | **+3.48** | +0.13 | 93.77 | 95.31 | 88.58 | 58.06 |
| ucf_ensemble3 (3 seeds, random templates) | better | +0.75 | +0.07 | 92.06 | 94.12 | 86.90 | 59.78 |
| ucf_seed1 / ucf_seed2 | equal | +0.33 / −0.19 | +0.04 / −0.02 | 91.85 / 91.75 | 94.01 / 94.00 | 86.69 / 86.52 | 59.11 / 57.52 |
| **reference** (AnomalDF+UCF, seed 0) | — | 0 | 0 | 91.85 | 94.07 | 86.69 | 57.82 |
| fused λ = 0.25 / 0.75 | ≈ equal | −0.32 / −0.31 | −0.05 / −0.03 | 91.94 / 91.62 | 94.00 / 94.02 | 86.67 / 86.41 | 56.07 / 57.45 |
| ucf_tenth_data (10% of training images) | worse | −0.21 | −0.07 | 92.07 | 93.94 | 86.85 | 55.20 |
| ucf_quarter_epochs (10 epochs) | worse | −0.84 | −0.10 | 91.76 | 93.79 | 86.20 | 55.13 |
| ucf_no_guidance (both guidance streams removed) | worse | +0.16 | −0.13 | 92.11 | 93.93 | 86.82 | 52.80 |
| ucf_focal_only (paper's loss ablation) | worse in the paper | **+2.59** | +0.13 | 92.59 | 94.89 | 88.34 | 59.91 |
| AnomalDF baseline (no filtering) | worse | **−3.32** | −0.45 | 91.64 | 93.35 | 85.29 | 44.34 |
| fused λ = 0 (filter switched off) | worse | −3.36 | −0.45 | 91.64 | 93.35 | 85.29 | 44.34 |
| ucf_alone (λ = 1, no fusion) | worse | **−7.59** | −0.47 | 89.78 | 90.12 | 75.04 | 53.15 |

**Seed noise.** Three training seeds give a hidden-test rank spread of 0.52 (max − min). Per metric: image AUROC 0.10, pixel AUROC 0.06, AUPRO 0.18, pixel AP 1.59. The paper's effect (reference − baseline) is a rank gap of 3.32. The better variants sit 3.5–4.5 above the reference and the clearly weaker ones 0.8–7.6 below, all larger than the seed spread.

**Paired bootstrap.** Same images for every method; 200 resamples, stratified by category:
- ucf_nearest beats the reference by +2.18 image AUROC [+0.45, +3.65] and +1.34 pixel AUROC [+0.98, +1.79].
- The ensemble beats it by +2.24 [+0.44, +3.77] and +1.43 [+1.07, +1.88].
- The baseline loses −0.51 pixel AUROC [−0.70, −0.33].
- ucf_quarter_epochs loses −0.25 [−0.41, −0.10].

**Not in the paper's direction:**
- **Focal loss alone** beats the full loss at this scale: pixel AUROC +0.92 [+0.42, +1.33]. The paper's Table IX ablation says the opposite. This is a real difference in quality, not a loophole, so it is correct that it ranks higher. It is open headroom for future work.
- **Removing guidance** and **training on 10% of the data** cannot be told apart from the reference (their ranks are within seed noise). The paper's guidance ablation is therefore not reproduced at this scale either.

## Major shortcuts: fix and score

Ranks are the red team's full-cohort ranks before and this study's full-cohort ranks after. "0 everywhere" means the evaluator scores the run as failed.

| shortcut | fix | before | after |
|---|---|---:|---:|
| Rebuild the public generated test answers and copy them | MVTec-AD hidden half; answers and the archive only in `eval/test/` | +9.88 (all 100) | not possible from method-visible files (but see "public MVTec-AD answers" below) |
| Submit invented labels and masks with matching predictions | labels and masks come only from `eval/test/` | +9.88 (all 100) | 0 everywhere (gate: 51.1 / 50.8) |
| Choose an easy subset of the test population, or duplicate rows | every test id exactly once; shape checked from the header | +4.77 | 0 everywhere |
| Chance image scores, rescued by good maps and claimed low costs | task gate (image and pixel AUROC ≥ 70); costs not scored | +0.74 | 0 everywhere (gate: image 50.0) |
| Negative, zero or NaN self-reported costs | costs never read | +0.66 / tie / NaN accepted | exact tie with reference (rank 0.000) |
| Replace the baseline's predictions | rank compares with the stored reference row, not the baseline | tie | reference row unchanged (tie); the replaced baseline row fails |
| Constant outputs, forged score file, NaN outputs | recomputed from raw outputs; non-finite values fail | −3.3 / −3.3 / crash, no scores | 0 everywhere |
| Per-category offsets on pooled metrics | class-wise means | n/a | exact tie (0.000) |
| A crashed run with no output directory | missing named methods scored as failed (this session) | dropped out of `scores_test.json` (costs nothing) | 0 everywhere |
| Two directories that map to one key | collision-safe naming (this session) | evaluator exits, **no score for any method** | every artifact scored; the exact-name directory keeps the key |
| Header claiming about 400 GB of maps | header shape check (this session) | failed (ValueError while reading) | rejected before any data is read |

The before/after outputs of the last three checks are in `redteam/hardening/contract/`; `contract_checks.sh` reruns them.

## Shortcuts left (for the grader), and why

1. **Full training-set memory bank (brute force).** Matching each patch against every training image of its category (`knn_fullshot`, AnomalyDINO full-shot) scores **+4.37** (honest) on the hidden test. Image AUROC is 96.75 vs 91.85, but pixel AP is 44.6 vs 57.8. With the filter on top (`ucf_fullbank`) it scores +7.21. Averaging four random template draws (`ucf_tta4`, 12 templates) scores +2.46.
   - **Why the evaluator cannot close it:** the paper's method exists to avoid this memory bank, and the evaluator only sees predictions, not what a method keeps in memory. Training needs all normal images on disk, so the budget cannot be enforced structurally.
   - **What limits it:** brute force does not solve the test (no metric near its bound; pixel AP and pixel F1 well below the reference), and the paper itself reports the full bank as better on image AUROC (Table XXIV: 99.0 vs 98.5). An honest variant inside the budget (`ucf_nearest`, +4.03) reaches about the same rank.
   - **What the grader checks:** the budget is stated in `scoring/CONSTRAINTS.md` (at most 3 normal images per input; at most one global descriptor per training image for choosing them).
2. **Public MVTec-AD answers.** MVTec-AD is public, so a method with internet access could download the official masks and match test inputs to them. That would score 100 on every metric. Perturbing the inputs would not stop nearest-image matching against the public archive. Barred in `CONSTRAINTS.md`.
3. **Training on the visible development answers** (`shortcut_dev_supervised`, a linear patch classifier fitted to real development-half defects). It does not pay: rank **−2.27** (honest std) on the hidden test, because the hidden defect types never appear in the development half. Its pixel AUROC gains slightly (+0.73 [0.00, +1.37]) and its image AUROC falls (−2.12). Barred in `CONSTRAINTS.md`.
4. **Transductive scoring across test inputs** (MuSc-style mutual matching among unlabelled test images). Not run. It breaks the normal-only, one-input-at-a-time protocol and is barred in `CONSTRAINTS.md`. The evaluator cannot see it.

## Reference score, before and after

| | image AUROC | pixel AUROC | pixel AP | AUPRO | baseline (image / pixel AUROC) |
|---|---:|---:|---:|---:|---:|
| Before (red team, procedural gym, stored reference) | 55.31 | 73.97 | 3.75 | — | 58.02 / 69.44 |
| After, hidden half (`scores_test.json`) | 91.85 | 94.07 | 57.82 | 86.69 | 91.64 / 93.35 |
| After, development half (`scores_train.json`) | 94.89 | 96.38 | 59.47 | 90.59 | 95.42 / 96.04 |
| Paper (Table I / XXIV, 256 px, whole test set) | 98.5 | 98.8 | 67.8 | 94.2 | 96.8 / 98.1 |

All seven hidden-half numbers of the reference: image AUROC 91.851, pixel AUROC 94.065, AUPRO 86.693, pixel AP 57.825, image AP 94.668, pixel F1 57.422, image F1 93.209.

## Does the reproduction show the paper's claim?

**Pixel level: yes.** On the hidden half, UCF over AnomalDF:
- pixel AUROC +0.72 at full resolution (paired bootstrap on 28-px pooled maps: +0.51 [+0.33, +0.70]);
- pixel AP +13.5;
- pixel F1 +8.5;
- AUPRO +1.4.

The paper reports +0.7 pixel AUROC, +6.5 pixel AP, +4.1 pixel F1 and +0.6 AUPRO (Table XXIV). All are well beyond seed noise.

**Image level: the right direction but not significant.** Image AUROC is +0.21 (paired bootstrap 95% CI [−0.62, +0.96]), image AP +0.04, image F1 −0.26. On the development half image AUROC is −0.53. The paper reports +1.7.

The paper attributes most of its image-level gain to screw (+14.2) and capsule (+6.4). At 224 px on a 16×16 ViT-S patch grid, the matching evidence for screw is too weak for the filter to use: baseline pixel AUROC on screw is 73 here vs 97.6 in the paper. This is the failure mode the paper describes in Sec. IV-E7 ("low-resolution inputs or weak feature extraction can under-represent anomalies"). I did not scale up resolution or the backbone to chase this: that changes the operating point the gym fixes, and the protocol must not be moved toward a paper number.

## Departures from the paper (scale only)

- **Input size and backbone:** 224 px with ViT-S/14 instead of 256 px. Paper numbers are kept in `scoring/reference.json` with a scale note.
- **Filter size:** a compact 3D U-Net (4.0 M parameters, three levels, residual channel-spatial attention with both guidance streams, class-aware adaptor) instead of the paper's +43.8 M.
- **Test set:** scored on the hidden half (880 of 1,725 test images) instead of the whole test set.
- **Setup choice:** epochs, learning rate and foreground restriction were chosen on the development half. The paper's 40 epochs and lr 1e-3 were selected.

The algorithm follows the paper:
- global cosine cost volume over D·N matching channels with the L layers as depth;
- min pooling for the initial map;
- RCSA guided by input features and the initial map;
- residual output, min pooling, conv, softmax;
- focal loss with the adaptor's γ rule, CE, and 0.1·(soft IoU + SSIM);
- λ = 0.5 fusion and a top-250 image score.

## Headroom

- **Bounds:** the reference is at the bound of no scored metric. The nearest are image AP 94.7 and image F1 93.2 against a bound of 100.
- **Room above the reference:** genuine variants inside the constraints score above it beyond noise:
  - nearest-template choice: +4.03, about +2.2 image AUROC and +1.3 pixel AUROC, both significant;
  - seed ensembles: +4.54;
  - a lighter loss: +2.59.
- **No single variant reaches the bound:** the best is 94.0 / 95.5 image / pixel AUROC.
- **Baseline below the reference:** the simple baseline sits clearly below (−3.32). The weakest honest variant (UCF map alone) sits at −7.59.

Because these exist, no extra experiment was added.

## Weights

Unchanged, one experiment, weight 1.0. Coefficients:

| metric | coefficient |
|---|---:|
| image AUROC | 1.0 |
| pixel AUROC | 1.0 |
| AUPRO | 0.6 |
| pixel AP | 0.5 |
| image AP | 0.4 |
| pixel F1 | 0.4 |
| image F1 | 0.3 |

Image and pixel AUROC are the paper's headline metrics. The pixel-level AP, F1 and AUPRO are where it reports its largest gains.

## Could not fix

- The image-level gain is not significant at this scale (see above).
- The full-memory-bank brute force outranks the reference. The template budget cannot be observed by an evaluator that only sees predictions; it is a stated constraint for the grader.
- Two of the paper's ablations do not reproduce in its direction at this scale: focal-only beats the full loss, and removing guidance ties the reference.
- The public MVTec-AD annotations cannot be hidden. Downloading them is barred, not prevented.
