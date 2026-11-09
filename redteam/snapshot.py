"""Audit-only frozen scoring inputs; never imported by shortcut payloads."""
import json
import hashlib
from pathlib import Path

root = Path('/home/user')
out = root / 'redteam/reference'
out.mkdir(parents=True, exist_ok=True)
for name in ('reference.json', 'scores_test.json', 'scores_train.json'):
    target = out / name
    if not target.exists():
        target.write_bytes((root / 'scoring' / name).read_bytes())
hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [root / 'scripts/evaluate_test.sh', *sorted((root / 'eval').rglob('*.py'))]}
(out / 'evaluator_hashes.json').write_text(json.dumps(hashes, indent=2, sort_keys=True))
