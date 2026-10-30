"""Run train-only selection on fixed, externally sourced spectral splits.

Spec is a list, or {"datasets": [...]}; data_path is absolute or relative to
the spec file. Each entry requires id, task, data_path and source_url. NPZ keys:
X_train, y_train, X_test, y_test; optional train_ids and test_ids. A single target
is evaluated per entry. Provenance and upstream split audits are not inferred.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from herb_spectra.benchmark import benchmark_dataset, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose an empty/new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    content = json.loads(args.spec.read_text())
    specs = content['datasets'] if isinstance(content, dict) else content
    if not isinstance(specs, list) or not specs:
        parser.error('Spec must contain a nonempty dataset list')
    ids = [v['id'] for v in specs]
    if len(set(ids)) != len(ids):
        parser.error('Dataset IDs must be unique, including fold/target when relevant')
    for value in ids:
        if Path(value).name != value or value in {'.', '..'}:
            parser.error('Dataset IDs must be plain directory names')
    manifest = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': args.seed,
                'spec_path': str(args.spec.resolve()), 'spec_sha256': hashlib.sha256(args.spec.read_bytes()).hexdigest(),
                'versions': {v: importlib.metadata.version(v) for v in ['numpy', 'scipy', 'scikit-learn']},
                'models': args.models, 'datasets': []}
    results = []
    for raw in specs:
        spec = dict(raw)
        path = Path(spec['data_path'])
        spec['data_path'] = str((path if path.is_absolute() else args.spec.resolve().parent/path).resolve())
        try:
            if not spec.get('source_url'):
                raise ValueError('Each dataset requires a source_url')
            manifest['datasets'].append({'id': spec['id'], 'data_path': spec['data_path'],
                                         'sha256': hashlib.sha256(Path(spec['data_path']).read_bytes()).hexdigest()})
            results.extend(benchmark_dataset(spec, args.output/spec['id'], args.models, args.seed))
        except Exception as exc:
            results.append({'id': spec['id'], 'task': spec.get('task'), 'model': '_dataset_validation',
                            'status': 'fail', 'error': f'{type(exc).__name__}: {exc}'})
        write_json(args.output/'results.json', results)
        write_json(args.output/'manifest.json', manifest)
    manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['all_selected_passed'] = all(v['status'] == 'pass' for v in results)
    write_json(args.output/'manifest.json', manifest)
    print(json.dumps({'all_selected_passed': manifest['all_selected_passed'],
                      'cases': len(results), 'results': str(args.output/'results.json')}, indent=2))
    return 0 if manifest['all_selected_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
