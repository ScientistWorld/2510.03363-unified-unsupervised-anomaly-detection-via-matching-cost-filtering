# Hardening report (final pass)

## Verdict

Both items in `REVIEW.md` are addressed.

- **The full-train memory bank no longer pays.** The reference is now the paper's own best AnomalDF+UCF setting, its full-shot protocol (Table XXIV). The memory-bank brute force is now the baseline, at rank −3.79 on the hidden test.
- **The hidden test is 2.6× larger.** I added VisA, the paper's second RGB benchmark, as a second scored experiment. The hidden test grows from 880 to 2,308 images. The paper's image-level effect is now significant on VisA:
  - image AUROC +0.90, paired 95% CI [+0.38, +1.46];
  - image AP +1.27;
  - in combined image AP, the baseline is −1.12 [−1.99, −0.18] below the reference.
- **Weaker variants separate at image level**, beyond the paired intervals:
  - the paper's N = 3 inference;
  - the paper's guidance and loss ablations.
- **Shortcuts.** Every shortcut in this study fails or ties the reference exactly.

What is **not** achieved:
- **No better variant clears the noise.** No variant I tried beats the reference on the hidden test beyond noise, at image level or overall.
  - The reference run (training seed 0) is the best of 5 training seeds on both slices.
  - 3- and 5-seed ensembles beat the average seed by +1.2 to +1.3 rank, but seed 0 by only +0.1 to +0.2.
  - The test can resolve image-level differences of about 0.5 class-mean AUROC (paired CI half-width 0.3–0.7). The variants I had simply do not improve image detection by that much.
- **MVTec-AD image AUROC is not robust across seeds.** Seed 0 gains +0.12 over the full-bank baseline. The 5-seed mean is −0.55, and only 1 of 5 seeds is above the baseline. The pixel-level gains and all VisA image-level gains hold for every seed.

## What changed in this pass

### 1. Reference = the paper's full-shot setting (review item 2)

**Why.** The old gym barred the full memory bank by a stated N = 3 template budget. The evaluator cannot observe that budget. The brute force scored +4.37, above every honest variant shown, and the paper itself reports the full bank as better (Table XXIV).

**What changed.**
- **The reference** is AnomalDF+UCF under AnomalyDINO's full-shot protocol, as in Table XXIV Exp. IDs 4 and 12. The filter is trained with N = 3 random templates, exactly as before. It is then applied without retraining to the cost volume against every training image of the input's category.
- **The cost-volume shape.** The paper says the volume shape "is adapted to match the input shape required by our models". The authors' released code (CostFilter-AD, `pre_min_dim`) does this by keeping a fixed number of lowest matching costs per input token. `method/ucf_model.py` now does the same: K = D = 256, in template-token order. One model therefore takes the 3-template volume in training and the full-bank volume at test time.
- **The baseline** is full-shot AnomalDF (Table XXIV IDs 3 and 11): exhaustive matching against the whole bank. The brute force that used to be a loophole is now the comparison point.
- **Inference is the paper's:** λ = 0.5 fusion and top-250 image score. Training is unchanged from the paper: 40 epochs, batch 8, Adam, focal + CE + 0.1 · (soft IoU + SSIM), the γ rule.
- **Constraints.** The template budget was removed from `scoring/CONSTRAINTS.md`, since it was unenforceable and is now moot.
- **Retraining.** The reference was retrained from the final code, because the cost-volume layout changed.

### 2. VisA as a second hidden experiment (review item 1)

**Data.** `rgb_visa_multiclass` uses VisA (12 categories, official 1cls split: 8,659 normal training images, 2,162 test images) at 224 px. Each dataset gets its own unified model, as in the paper.
- **Archive.** It sits in `eval/test/downloads/`. `scripts/download.sh` fetches it from the public S3 bucket and pins its SHA-256.
- **The only code that rebuilds the answers** is `eval/test/prepare_visa.py`.
- **Split.** One third of the test set is a visible development slice (734 images). Two thirds are hidden (1,428 images: 787 anomalous, 641 normal). Whole primary defect types go to one side, as for MVTec-AD.

**Evaluators.** Both `eval/{train,test}/evaluate.py` now score each method directory in both experiments: `predictions.npz` for MVTec-AD and `predictions_visa.npz` for VisA.
- A missing or invalid file, or a failed task gate (image or pixel AUROC < 70), scores 0 on every metric of that experiment.
- Everything else is unchanged: class-wise means, header shape checks, id coverage, finite values, and collision-safe naming.

**Weights.** MVTec-AD 0.5, VisA 0.5. `validate.py` requires the weights to sum to 1. The paper treats the two benchmarks alike (Tables I, III, XXIV). Coefficients are unchanged.

### 3. Smaller fixes

- Training/inference scripts take `--dataset` and `--device`; with two GPUs the two datasets run in parallel.
- Per-dataset metadata files avoid a write race.
- Checkpoint resume restores the synthesis generator state on CPU. A resumed run crashed before this fix; found when a worker node died mid-job.
- `--widths` option for the filter.

### 4. Study

`redteam/final/` holds the study scripts.
- `summary.md` / `summary.json` are the final results.
- `summary_job1.*` is the first job, before seeds 3–4 and the wide filter.
- The previous pass's study stays in `redteam/hardening/` as the "before" record.

All rows were produced by the final code in jobs `f88f34e8-c85` and `9108bab3-421`, with the resumed job `af72abdf-466` in between. The reference and baseline were rerun end to end. `scoring/scores_{train,test}.json` and `scores.json` hold only the reference and baseline rows. `python validate.py --compare --post-design` passes.

## Do better results score higher?

**How the rank is computed.** The rank is Σ_e w_e Σ_m |coef_m| · (attempt − reference) / std_e,m. Weights are 0.5 per experiment, and std is taken over the 21 honest hidden-test runs below.

**Before** (previous pass, MVTec-only hidden half, its own honest cohort and units):

| variant | expected | rank |
|---|---|---:|
| 3 nearest templates (N = 3 protocol) | better | +4.03 |
| 3-seed ensemble, nearest templates | better | +4.54 |
| full memory bank (barred by a rule only) | brute force | **+4.37** |
| UCF on top of the full bank | over budget | +7.21 |
| focal loss only (paper: worse) | worse | **+2.59** |
| guidance removed (paper: worse) | worse | **+0.16** |
| AnomalDF baseline | worse | −3.32 |
| seed spread (3 seeds) | | 0.52 |

**After** (final gym, hidden test of both experiments, honest cohort):

| run | expected | rank | MVTec part | VisA part | MVTec I-AUROC / P-AUROC / P-AP | VisA I-AUROC / P-AUROC / P-AP |
|---|---|---:|---:|---:|---|---|
| 3-seed ensemble (seeds 0–2) | better | +0.20 | −0.05 | +0.25 | 96.64 / 96.59 / 64.8 | 88.51 / 94.88 / 31.7 |
| 5-seed ensemble (seeds 0–4) | better | +0.10 | −0.24 | +0.34 | 96.32 / 96.60 / 65.1 | 88.63 / 94.91 / 32.6 |
| **reference** (full-shot UCF, seed 0) | — | 0 | 0 | 0 | 96.88 / 96.59 / 62.1 | 88.61 / 94.80 / 29.4 |
| 3-seed ensemble, λ = 0.25 | better? | −0.03 | −0.30 | +0.27 | 97.06 / 96.39 / 61.7 | 88.56 / 94.89 / 31.7 |
| λ = 0.25 / 0.75 | ≈ equal | −0.06 / −0.29 | | | 97.06 / 96.45 / 60.1 · 96.78 / 96.65 / 62.5 | 88.66 / 94.85 / 30.5 · 88.56 / 94.65 / 27.2 |
| 5-seed ensemble, λ = 0.25 | better? | −0.21 | −0.51 | +0.30 | 96.74 / 96.39 / 61.9 | 88.64 / 94.89 / 32.2 |
| wider filter (128/192/256, ~4× params) | better? | −0.49 | −0.07 | −0.42 | 96.49 / 96.71 / 64.8 | 88.08 / 94.55 / 27.1 |
| trained with 16 templates | better? | −0.75 | −1.01 | +0.26 | 96.19 / 96.18 / 62.1 | 88.08 / 94.80 / 33.3 |
| seeds 1 / 2 / 3 / 4 | equal | −1.14 / −1.83 / −1.21 / −1.35 | | | 95.93–96.13 / 96.12–96.31 / 60.8–62.0 | 88.16–88.73 / 94.40–94.81 / 24.8–32.8 |
| 10 of 40 epochs | worse | −1.68 | −1.65 | −0.02 | 96.36 / 95.84 / 57.7 | 88.11 / 94.71 / 30.3 |
| focal loss only (paper Table IX) | worse | **−1.81** | −1.91 | +0.09 | 93.91 / 96.46 / 61.4 | 87.38 / 95.23 / 31.9 |
| filter switched off (λ = 0) | worse | −3.88 | | | 96.38 / 95.28 / 44.6 | 87.97 / 94.53 / 23.4 |
| **AnomalDF full bank (baseline = brute force)** | worse | **−3.79** | −3.01 | −0.78 | 96.75 / 95.28 / 44.6 | 87.71 / 94.53 / 23.4 |
| UCF with the paper's N = 3 random templates | worse | −6.57 | −3.82 | −2.75 | 93.62 / 94.57 / 59.5 | 83.40 / 91.77 / 23.3 |
| UCF map alone (λ = 1) | worse | −7.22 | | | 94.36 / 96.06 / 62.4 | 83.69 / 83.22 / 23.2 |
| AnomalDF, N = 3 random templates | worse | −10.95 | | | 91.64 / 93.35 / 44.3 | 81.91 / 91.38 / 17.4 |
| guidance removed (paper ablation) | worse | **−11.04** | −5.75 | −5.29 | 92.31 / 94.40 / 51.3 | 72.55 / 93.54 / 16.4 |

**Training-seed noise.**
- Five seeds give a hidden-test rank spread of 1.83 (max − min); seed 0 is the top.
- The paper's effect (reference − baseline) is 3.79, 2.1× that spread.
- Against the 5-seed mean, the baseline is −2.68.
- Per-metric seed spread on the hidden test:
  - MVTec-AD: image AUROC 0.94, pixel AUROC 0.47, AUPRO 1.28, pixel AP 1.3, pixel F1 2.8.
  - VisA: image AUROC 0.57, pixel AUROC 0.41, AUPRO 1.4, **pixel AP 8.0, pixel F1 8.8**. VisA pixel AP and F1 swing 24.8–32.8 across seeds.
- On the dev slice the seed spread is 0.67.

**Ordering.**
- The weaker variants order as expected.
- The paper's two ablations now point the paper's way: focal only −1.81, guidance removed −11.04. Before this pass they ranked at or above the reference.
- Every seed beats the baseline in rank.
- Every "better" candidate beats the average seed: ensembles +1.2 to +1.3, the wider filter +0.6.
- No candidate beats the seed-0 reference by more than the seed noise. On the dev slice, the ensembles beat it by +0.6 to +0.9 against a seed spread of 0.67.

**Paired bootstrap on the hidden test** (300 category-stratified resamples, same images for every method, Δ vs reference):

| run | MVTec Δ I-AUROC | VisA Δ I-AUROC | combined Δ I-AUROC | combined Δ I-AP | VisA Δ P-AUROC (14×14) |
|---|---|---|---|---|---|
| AnomalDF baseline | −0.13 [−1.56, +1.31] | **−0.89 [−1.46, −0.38]** | −0.51 [−1.24, +0.30] | **−1.12 [−1.99, −0.18]** | **−0.39 [−0.67, −0.13]** |
| 10 of 40 epochs | −0.52 [−2.04, +0.91] | **−0.51 [−0.97, −0.09]** | −0.51 [−1.29, +0.18] | **−0.93 [−1.78, −0.11]** | −0.01 [−0.27, +0.23] |
| focal only | **−3.02 [−4.41, −1.78]** | −1.18 [−3.00, +0.41] | **−2.10 [−3.24, −1.13]** | **−1.33 [−2.19, −0.58]** | +0.84 [−0.02, +1.78] |
| paper's N = 3 inference | **−3.33** | **−5.13** | **−4.23 [−5.40, −3.12]** | **−3.38** | **−3.11** |
| guidance removed | **−4.52** | **−16.13** | **−10.33 [−11.94, −8.77]** | **−8.61** | **−2.07** |
| 5-seed ensemble | −0.54 [−1.41, +0.30] | +0.04 [−0.31, +0.41] | −0.25 [−0.71, +0.15] | −0.29 [−0.88, +0.15] | **+0.17 [+0.08, +0.25]** |
| 3-seed ensemble | −0.24 [−0.84, +0.41] | −0.08 [−0.37, +0.19] | −0.16 [−0.50, +0.17] | −0.18 [−0.58, +0.20] | **+0.10 [+0.03, +0.18]** |
| λ = 0.25 | +0.20 [−0.25, +0.79] | +0.06 [−0.09, +0.23] | +0.13 [−0.11, +0.42] | +0.15 [−0.12, +0.41] | **+0.08 [+0.03, +0.15]** |

In the old gym the paper's image-level effect was +0.21 [−0.62, +0.96]. Now it is significant on VisA, and on combined image AP.

## Major shortcuts: fix and score

Ranks are full-cohort ranks in each pass's own cohort. "0 everywhere" means the evaluator scores the run as failed in that experiment.

| shortcut | fix | before | after |
|---|---|---:|---:|
| **Full training-set memory bank (brute force)** | it is now the baseline; the reference is the paper's full-shot UCF | +4.37 honest (above every honest variant) | **−3.79** honest; every seed of the reference beats it |
| UCF over the full bank / averaged template draws | moot: the reference already uses the full bank | +7.21 / +2.46 | — (they *are* the reference protocol) |
| Submit one benchmark only (new contract) | a missing file fails that experiment | n/a | VisA 0 everywhere, rank −4.75 |
| Oversized `.npy` header (400 GB claim) | header shape check, per experiment | rejected | MVTec 0 everywhere, VisA scored; rank −4.76 |
| Invented answers, constant / NaN outputs, forged score file, half population, duplicate ids, missing artifact | unchanged rules, now on both experiments | 0 everywhere | 0 everywhere in both experiments (rank −9.51) |
| Self-reported negative costs / perfect metrics | never read | exact tie | exact tie (0.000) |
| Per-category offsets | class-wise means | exact tie | exact tie (0.000) |
| Baseline predictions replaced by zeros | rank compares with the stored reference row | tie | reference row unchanged; the replaced baseline fails |

## Shortcuts left (for the grader), and why

1. **Public test annotations.** MVTec-AD and VisA are public. A method with internet access could download the masks and match test inputs to them; nothing in the inputs can stop nearest-image matching against the public archives. Barred in `scoring/CONSTRAINTS.md`.
2. **Training on the visible development answers** (`shortcut_dev_supervised`, a linear patch probe fitted to real dev-slice defects). It does not pay. Against the reference it scores:
   - image AUROC −7.2 (MVTec-AD) and −10.6 (VisA);
   - full-cohort rank −0.74, below the baseline.

   Pixel AUROC on VisA does gain (+1.2), which the rank outweighs. The hidden defect types never appear in the dev slices. Barred.
3. **Transductive scoring** (statistics across test inputs). Not run. Barred, and invisible to the evaluator.
4. **A larger or fine-tuned backbone, or higher input resolution.** These would very likely help: the paper's own 448/672 px rows are better. They are fixed by `CONSTRAINTS.md` as the paper's operating point and cannot be observed from predictions.

The template budget is no longer in this list: it was removed, not left.

## Reference score, before and after

| | image AUROC | pixel AUROC | AUPRO | pixel AP | image AP | pixel F1 | image F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Before: MVTec hidden half, N = 3 reference | 91.85 | 94.07 | 86.69 | 57.82 | 94.67 | 57.42 | 93.21 |
| After: MVTec-AD hidden half (`scores_test.json`) | 96.88 | 96.59 | 90.88 | 62.14 | 98.23 | 62.20 | 97.50 |
| After: VisA hidden two thirds (`scores_test.json`) | 88.61 | 94.80 | 76.94 | 29.37 | 90.08 | 34.55 | 85.32 |
| After: MVTec-AD dev half (`scores_train.json`) | 97.73 | 97.99 | 93.90 | 68.19 | 99.18 | 67.01 | 97.32 |
| After: VisA dev third (`scores_train.json`) | 90.74 | 94.43 | 72.80 | 34.56 | 93.28 | 38.47 | 88.07 |
| Paper, MVTec-AD full bank 256 px (Table XXIV ID 4) | 99.3 | 98.9 | 96.6 | 68.7 | 99.8 | 65.5 | 98.6 |
| Paper, VisA full bank 256 px (Table XXIV ID 12) | 95.5 | 99.4 | 87.0 | 45.9 | 96.3 | 46.6 | 91.5 |

The baseline (full-bank AnomalDF) on the hidden test:
- MVTec-AD: 96.75 / 95.28 / 87.77 / 44.59 (image AUROC / pixel AUROC / AUPRO / pixel AP);
- VisA: 87.71 / 94.53 / 76.52 / 23.39 (same order).

## Does the reproduction show the paper's claim?

Reference minus full-bank baseline on the hidden test, with the paper's Table XXIV full-bank gains alongside:

| | I-AUROC | P-AUROC | AUPRO | P-AP | I-AP | P-F1 | I-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MVTec-AD, seed 0 | +0.12 | +1.31 | +3.11 | +17.6 | +1.17 | +12.5 | +1.21 |
| MVTec-AD, 5-seed mean | **−0.55** | +0.99 | +2.30 | +17.0 | +0.25 | +11.2 | +0.02 |
| paper MVTec-AD (ID 3→4) | +0.3 | +1.4 | +4.9 | – | +0.5 | +6.8 | +0.2 |
| VisA, seed 0 | +0.90 | +0.26 | +0.42 | +6.0 | +1.27 | +4.0 | +0.28 |
| VisA, 5-seed mean | +0.71 | +0.14 | +0.20 | +5.9 | +1.15 | +3.0 | +0.32 |
| paper VisA (ID 11→12) | +0.9 | +1.1 | +0.3 | – | +0.6 | +2.3 | +0.6 |

- **Pixel level: yes, on both datasets and for every seed.** The pixel AUROC, AUPRO, AP and F1 gains on MVTec-AD hold for 5 of 5 seeds. On VisA, pixel AP holds for 5 of 5 seeds and pixel AUROC and AUPRO for 4 of 5.
- **VisA image level: yes, for all 5 seeds.** The bootstrap interval excludes 0.
- **MVTec-AD image level: no, not robustly.**
  - The UCF probability hurts MVTec screw (image AUROC 83.4 → 71.6 for seed 0).
  - Only seed 0 learns to flag missing or misplaced transistors (96.2 vs 81.7–82.7 for seeds 1 and 2).
  - The paper's own full-bank gain here is small (+0.3).
  - I did not change the method to chase it.

## Headroom

- **Bounds.** The reference is at the bound of no scored metric. Its highest are MVTec-AD image AP 98.2 and image F1 97.5. VisA is far from every bound: image AUROC 88.6, pixel AP 29.4, pixel F1 34.5, AUPRO 76.9.
- **Room above the reference.** Variants that beat the average seed by more than the seed noise exist on both slices:
  - 5-seed ensemble +1.21 over the seed mean on the hidden test;
  - VisA pixel AUROC +0.17 [+0.08, +0.25] over seed 0 itself.

  Against the single seed-0 row they gain only +0.1 to +0.2. A future method has to beat a top-of-distribution draw of the paper's method.
- **Brute force and simple baselines.** The brute force is the baseline, 3.79 below. The simple N = 3 AnomalDF is 10.95 below. No simple baseline outranks the reference.
- **Added experiment.** VisA (described above), where the paper's method falls furthest short: tiny defects, multi-instance categories.

## Weights

- MVTec-AD 0.5 and VisA 0.5. Before: one experiment at 1.0.
- Equal, because the paper reports both benchmarks side by side as its RGB results.
- Coefficients unchanged: image AUROC 1.0, pixel AUROC 1.0, AUPRO 0.6, pixel AP 0.5, image AP 0.4, pixel F1 0.4, image F1 0.3.

## Could not fix

- **No better variant beats the reference beyond noise on the hidden test.** I tried 3- and 5-seed ensembles, λ = 0.25 fusion, a 4× wider filter and 16-template training. The reference run happens to be the best of five training seeds. The variants beat the seed average but not that draw.
- **MVTec-AD image AUROC does not robustly improve over the full-bank baseline.** It does for seed 0 only (5-seed mean −0.55). The paper reports +0.3 here.
- **VisA pixel AP and F1 are seed-noisy** (8-point spread). They carry coefficients 0.5 and 0.4 as before.
- **Unenforceable constraints.** The public annotations, transductive scoring and backbone/resolution limits can only be stated, not enforced, by an evaluator that sees predictions.
