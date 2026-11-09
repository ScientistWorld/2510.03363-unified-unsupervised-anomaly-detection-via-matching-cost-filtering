"""Run one real submission and the unmodified no-argument hidden test.

The submitter process exits before evaluation starts. The evaluator uses only
persisted output files in its own Python process, with no environment changes.
The harness temporarily isolates outputs so discovery sees exactly one method.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    configs = json.loads((AUDIT / 'cases.json').read_text())
    cfg = configs[args.name]
    folder = AUDIT / args.name
    folder.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        from redteam.methods import prepare
        prepare(args.name, cfg)
        return
    started = time.time()
    print(f'BEGIN {args.name}', flush=True)
    subprocess.run([sys.executable, '-m', 'redteam.run_case', args.name, '--prepare-only'], cwd=ROOT, check=True)
    backup = AUDIT / '_state' / 'active_outputs'
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise RuntimeError('Previous evaluation did not restore outputs; recover before rerunning')
    output = ROOT / 'outputs'
    had_outputs = output.exists()
    if had_outputs:
        output.rename(backup)
    score_paths = [ROOT / 'scoring/scores_train.json', ROOT / 'scoring/scores_test.json']
    score_backups = {p: p.read_bytes() if p.exists() else None for p in score_paths}
    try:
        (output / 'ucf_synthetic').mkdir(parents=True)
        shutil.copy2(folder / 'artifact/predictions.npz', output / 'ucf_synthetic/predictions.npz')
        # Always persist an ordinary visible evaluation, when it succeeds.
        with (folder / 'evaluate_train.log').open('w') as log:
            train = subprocess.run(['bash', 'scripts/evaluate_train.sh'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if train.returncode == 0:
            shutil.copy2(score_paths[0], folder / 'scores_train.json')
        if cfg['kind'] == 'score_file_shortcut':
            # Shortcut: directly write fabricated scores; test must overwrite them.
            from redteam.methods import reference_artifact
            forged = {'experiments': {'rgb_mvtec_multiclass': {'results': {'anomaldf_ucf': {
                'image_auroc': 100.0, 'pixel_auroc': 100.0,
                'added_params_M': -1e6, 'added_flops_G': -1e6}}}}}
            score_paths[1].write_text(json.dumps(forged))
            (folder / 'forged_scores.json').write_text(json.dumps(forged, indent=2))
        # Separate fresh process. Exactly the final grader command, no arguments.
        with (folder / 'evaluate_test.log').open('w') as log:
            result = subprocess.run(['bash', 'scripts/evaluate_test.sh'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            shutil.copy2(score_paths[1], folder / 'scores_test.json')
        else:
            (folder / 'test_error.txt').write_text((folder / 'evaluate_test.log').read_text()[-4000:])
        status = dict(case=args.name, config=cfg, returncode=result.returncode,
                      train_returncode=train.returncode, elapsed_seconds=time.time() - started,
                      test_command=['bash', 'scripts/evaluate_test.sh'],
                      evaluator_environment='unchanged; separate process after submitter exits')
        (folder / 'status.json').write_text(json.dumps(status, indent=2))
    finally:
        shutil.rmtree(output)
        if had_outputs:
            backup.rename(output)
        for p, content in score_backups.items():
            if content is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(content)
    print(f'END {args.name}: test exit={result.returncode}', flush=True)


if __name__ == '__main__':
    main()
