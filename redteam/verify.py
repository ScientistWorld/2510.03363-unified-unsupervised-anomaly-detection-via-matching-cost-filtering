"""Lightweight checks of saved audit evidence; no model/evaluator execution."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
START = '11cca7d'


def main():
    configs = json.loads((AUDIT / 'cases.json').read_text())
    summary = json.loads((AUDIT / 'summary.json').read_text())
    for source in AUDIT.rglob('*.py'):
        ast.parse(source.read_text(), filename=str(source))
    for name in configs:
        folder = AUDIT / name
        assert (folder / 'run.sh').is_file(), name
        subprocess.run(['bash', '-n', str(folder / 'run.sh')], check=True)
        status = json.loads((folder / 'status.json').read_text())
        assert status['test_command'] == ['bash', 'scripts/evaluate_test.sh'], name
        assert status['returncode'] == (1 if name == 'nan_outputs_shortcut' else 0), name
        if status['returncode'] == 0:
            assert (folder / 'scores_test.json').is_file(), name
        else:
            assert 'Input contains NaN' in (folder / 'test_error.txt').read_text()

    expected_hashes = json.loads((AUDIT / 'reference/evaluator_hashes.json').read_text())
    hashes = {}
    for path, expected in expected_hashes.items():
        content = (ROOT / path).read_bytes()
        hashes[path] = hashlib.sha256(content).hexdigest()
        assert hashes[path] == expected, path
        assert content == subprocess.check_output(['git', 'show', f'{START}:{path}'], cwd=ROOT), path
    for score in ('scores_test.json', 'scores_train.json'):
        original = subprocess.check_output(['git', 'show', f'{START}:scoring/{score}'], cwd=ROOT)
        assert (ROOT / 'scoring' / score).read_bytes() == original, score
        assert (AUDIT / 'reference' / score).read_bytes() == original, score

    unchanged = ['baseline', 'briefing', 'data', 'environment', 'eval', 'method', 'scoring', 'scripts']
    delta = subprocess.check_output(['git', 'diff', '--name-only', START, '--', *unchanged], cwd=ROOT).decode()
    assert not delta.strip(), delta
    metrics = summary['metrics']
    control = summary['values']['reference_seed7']
    replacement = summary['values']['baseline_replacement_shortcut']
    assert all(control[k] == replacement[k] for k in metrics)
    assert all(name in summary['cohort_members']['scientific_cohort']
               for name in summary['ranks']['scientific_cohort'])
    assert len(summary['values']) == 20
    assert len(configs) == 21
    for name in ('ground_truth_shortcut', 'brute_force_visible_candidates_shortcut'):
        assert summary['dominance'][name]['nonworse_all'], name
        assert summary['dominance'][name]['better_any'], name
    assert summary['ranks']['all_finite_audit_attempts']['failed_detection_resource_shortcut'] > 0
    assert 'nan_resources_shortcut' not in summary['values']

    report = dict(starting_revision=START, case_count=len(configs),
                  successful_hidden_invocations=20, expected_rejections=['nan_outputs_shortcut'],
                  all_test_commands=['bash scripts/evaluate_test.sh'],
                  evaluator_sha256=hashes, unchanged_directories=unchanged,
                  original_scoring_files_restored=True,
                  baseline_replacement_shortcut_ties_fresh_seed7=True,
                  finite_population_size=len(summary['values']), checks='passed')
    (AUDIT / 'integrity.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Audit evidence checks passed: 21 cases, 20 successful hidden invocations, 1 expected rejection.')


if __name__ == '__main__':
    main()
