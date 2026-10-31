"""Prepare, select, refit or use frozen ZY01–ZY07 deployment models."""
import argparse
import json
from pathlib import Path
from herb_spectra.deployment import prepare, select, fit_full, predict, template

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('stage', choices=['prepare', 'select', 'fit', 'template', 'predict'])
p.add_argument('--run', type=Path); p.add_argument('--gate', type=Path); p.add_argument('--release', type=Path)
p.add_argument('--bundle', type=Path); p.add_argument('--input', type=Path); p.add_argument('--output', type=Path)
p.add_argument('--mode', choices=['future', 'training-diagnostic'], default='future')
p.add_argument('--confirm-compatible-measurement', action='store_true')
a = p.parse_args()
required = {'prepare': ['run'], 'select': ['run', 'gate'], 'fit': ['run', 'release'],
            'template': ['bundle', 'output'], 'predict': ['bundle', 'input', 'output']}[a.stage]
if any(getattr(a, k) is None for k in required): p.error('Required: ' + ', '.join('--' + k for k in required))
result = {'prepare': lambda: prepare(a.run), 'select': lambda: select(a.run, a.gate),
          'fit': lambda: fit_full(a.run, a.release), 'template': lambda: template(a.bundle, a.output),
          'predict': lambda: predict(a.bundle, a.input, a.output, mode=a.mode,
              confirm_compatible_measurement=a.confirm_compatible_measurement)}[a.stage]()
print(json.dumps(result, ensure_ascii=False), flush=True)
