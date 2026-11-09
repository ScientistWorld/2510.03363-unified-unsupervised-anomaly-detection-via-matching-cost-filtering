"""Summarize actual test runs using the stated population-normalized rank.

Future-attempt standard deviations are unknowable during this audit. We report
both a clean scientific cohort and the complete finite audit cohort explicitly,
and distinguish population-independent dominance from illustrative totals.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import statistics

ROOT = Path('/home/user/redteam')
EXP = 'rgb_mvtec_multiclass'


def main():
    configs = json.loads((ROOT / 'cases.json').read_text())
    spec = json.loads((ROOT / 'reference/reference.json').read_text())['experiments'][EXP]
    original = json.loads((ROOT / 'reference/scores_test.json').read_text())['experiments'][EXP]['results']
    metrics = spec['metrics']
    ref = original['anomaldf_ucf']
    values = {'original_reference': ref}
    statuses = {}
    for name in configs:
        folder = ROOT / name
        if (folder / 'status.json').exists():
            statuses[name] = json.loads((folder / 'status.json').read_text())
        if (folder / 'scores_test.json').exists() and statuses.get(name, {}).get('returncode') == 0:
            row = json.loads((folder / 'scores_test.json').read_text())['experiments'][EXP]['results']['anomaldf_ucf']
            if all(math.isfinite(row.get(k, float('nan'))) for k in metrics):
                values[name] = row
    clean = {k: v for k, v in values.items() if 'shortcut' not in k}
    cohorts = {'scientific_cohort': clean, 'all_finite_audit_attempts': values}
    scales, ranks = {}, {}
    for cohort_name, cohort in cohorts.items():
        scale = {k: statistics.pstdev([r[k] for r in cohort.values()]) for k in metrics}
        scales[cohort_name] = scale
        ranks[cohort_name] = {}
        # A ranked attempt must be included in the population that supplies
        # its denominator. Do not report shortcuts against a clean-only scale.
        for name, row in cohort.items():
            ranks[cohort_name][name] = spec['weight'] * sum(
                abs(d['coefficient']) * (1 if d['higher_is_better'] else -1)
                * (row[k] - ref[k]) / scale[k]
                for k, d in metrics.items() if scale[k] > 1e-12)
    seeds = [values[n] for n in ('reference_seed7', 'reference_seed13', 'reference_seed23') if n in values]
    spread = {k: {'min': min(r[k] for r in seeds), 'max': max(r[k] for r in seeds),
                  'range': max(r[k] for r in seeds) - min(r[k] for r in seeds),
                  'std': statistics.pstdev(r[k] for r in seeds)} for k in metrics} if seeds else {}
    rank_spread = {}
    for cohort_name, cohort_ranks in ranks.items():
        s = [cohort_ranks[n] for n in ('reference_seed7', 'reference_seed13', 'reference_seed23') if n in cohort_ranks]
        rank_spread[cohort_name] = {'min': min(s), 'max': max(s), 'range': max(s) - min(s)} if s else {}
    dominance = {}
    for name, row in values.items():
        signed = [(1 if d['higher_is_better'] else -1) * (row[k] - ref[k]) for k, d in metrics.items()]
        dominance[name] = dict(nonworse_all=all(x >= -1e-8 for x in signed),
                               better_any=any(x > 1e-8 for x in signed),
                               exact_tie=all(abs(x) <= 1e-8 for x in signed))
    report = dict(reference=ref, baseline=original['anomaldf'], metrics=metrics,
                  values=values, ranks=ranks, cohort_members={k: list(v) for k, v in cohorts.items()},
                  population_std=scales, seed_spread=spread, seed_rank_spread=rank_spread,
                  dominance=dominance, statuses=statuses,
                  rank_note='Signed improvement uses higher_is_better and absolute coefficient; population std is cohort-specific, not metric units. Future-population ranks cannot be known exactly.')
    (ROOT / 'summary.json').write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = ['case,image_auroc,pixel_auroc,image_ap,pixel_ap,pixel_f1_max,pixel_aupro,added_params_M,added_flops_G,scientific_cohort_rank,all_audit_rank']
    for name, row in values.items():
        fields = [name] + [f'{row[k]:.8g}' for k in ('image_auroc','pixel_auroc','image_ap','pixel_ap','pixel_f1_max','pixel_aupro','added_params_M','added_flops_G')]
        fields += [f'{ranks[c][name]:.8g}' if name in ranks[c] else '' for c in cohorts]
        lines.append(','.join(fields))
    (ROOT / 'summary.csv').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
