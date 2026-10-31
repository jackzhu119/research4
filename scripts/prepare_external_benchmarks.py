"""Fetch pinned public herb spectra and prepare reproducible held-out folds.

Only source links/hashes and benchmark metadata are committed. Raw source arrays
and derived NPZ files remain under ignored outputs/. Row hashes identify spectra,
not independent physical plants, batches, or replicate groups.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_ids(X):
    return np.asarray([hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
                       for row in X])


def absorbance(X):
    X = np.asarray(X, dtype=np.float64)
    if not np.isfinite(X).all() or np.any(X <= 0):
        raise ValueError('Percent transmission must be finite and strictly positive')
    return -np.log10(X / 100.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=REPO/'outputs/external-data/ultrair')
    parser.add_argument('--output', type=Path, default=REPO/'outputs/external-benchmark-inputs')
    parser.add_argument('--download', action='store_true', help='Fetch missing files from fixed commit URLs')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose an empty/new output directory')
    source_manifest = json.loads((REPO/'data/external/ultrair_sources.json').read_text())
    metadata = json.loads((REPO/'data/external/ultrair_metadata.json').read_text())
    def check_source(row):
        path = args.source_root/row['repository_path']
        if not path.exists():
            if not args.download:
                raise FileNotFoundError(f'{path}: use --download or supply the existing source root')
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(row['source_url'], timeout=45) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != row['sha256']:
                raise ValueError(f'Download hash mismatch: {row["repository_path"]}')
            path.write_bytes(content)
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f'Source hash mismatch: {path}')
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(check_source, source_manifest['files']))
    args.output.mkdir(parents=True, exist_ok=True)
    specs = []; audits = []
    for dataset in metadata['datasets']:
        classification = dataset['task'] == 'geographic origin classification'
        task = 'classification' if classification else 'regression'
        target_columns = [None] if classification else range(len(dataset['label_names']))
        coverage = {}; labels_by_id = {}; first_set = None
        for fold in range(1, 6):
            partitions = {}
            for split in ['train', 'valid', 'test']:
                folder = args.source_root/dataset['folder']/f'fold-{fold}'/split
                raw = np.load(folder/dataset['x_filename'], allow_pickle=False)
                y = np.load(folder/dataset['y_filename'], allow_pickle=False)
                ids = row_ids(raw)
                if len(set(ids)) != len(ids):raise ValueError('Duplicate source rows')
                if len(raw) != len(y) or not np.isfinite(y).all():raise ValueError('Invalid targets')
                for key, value in zip(ids, y):
                    previous = labels_by_id.setdefault(str(key), np.asarray(value))
                    if not np.array_equal(previous, value):raise ValueError('Cross-fold label conflict')
                partitions[split] = (absorbance(raw), y, ids)
            X_train = np.concatenate([partitions[s][0] for s in ['train', 'valid']])
            Y_train = np.concatenate([partitions[s][1] for s in ['train', 'valid']])
            train_ids = np.concatenate([partitions[s][2] for s in ['train', 'valid']])
            X_test, Y_test, test_ids = partitions['test']
            if set(train_ids) & set(test_ids):raise ValueError('Training/test spectrum overlap')
            all_ids = set(train_ids) | set(test_ids)
            if first_set is None:first_set = all_ids
            if all_ids != first_set:raise ValueError('Different source population across folds')
            for key in test_ids:coverage[str(key)] = coverage.get(str(key), 0) + 1
            for column in target_columns:
                suffix = '' if column is None else f'_target{column}'
                item_id = f'ultrair_{dataset["id"]}{suffix}_fold{fold}'
                filename = args.output/(item_id+'.npz')
                y_train = Y_train if column is None else Y_train[:, column]
                y_test = Y_test if column is None else Y_test[:, column]
                np.savez_compressed(filename, X_train=X_train, y_train=y_train,
                                    X_test=X_test, y_test=y_test,
                                    train_ids=train_ids, test_ids=test_ids)
                record = {
                    'id': item_id, 'dataset': 'ultrair_'+dataset['id']+suffix,
                    'herb': dataset['id'].replace('_lc', ''), 'fold': fold, 'task': task,
                    'data_path': str(filename.resolve()), 'data_sha256': digest(filename),
                    'source_url': source_manifest['repository']+'/tree/'+source_manifest['commit'],
                    'source_commit': source_manifest['commit'],
                    'source_dataset': dataset['id'], 'target_column': column,
                    'target_name': 'geographic origin' if column is None else dataset['label_names'][column],
                    'target_unit': 'class' if column is None else 'LC-MS relative abundance (a.u.)',
                    'preprocess_modes': ['raw', 'snv'],
                    'input_transform': '-log10(percent_transmission/100), all 1868 ordered points; no resizing',
                    'split_policy': 'Official outer test; official train+valid merged for inner 3-fold selection',
                    'identity_scope': 'SHA256 of spectrum row; physical sample and batch IDs unavailable',
                    'n_train': len(y_train), 'n_test': len(y_test), 'n_features': X_train.shape[1],
                }
                if column is None:record['class_names'] = dataset['label_names']
                specs.append(record)
                audits.append({'id': item_id, 'train_ids': train_ids.tolist(), 'test_ids': test_ids.tolist(),
                               'exact_train_test_overlap': 0, 'finite': True,
                               'train_target_range': [float(np.min(y_train)), float(np.max(y_train))],
                               'test_target_range': [float(np.min(y_test)), float(np.max(y_test))]})
        if set(coverage) != first_set or set(coverage.values()) != {1}:
            raise ValueError('Official test folds do not cover each spectrum exactly once')
    def write(name, value):
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    write('spec.json', {'datasets': specs})
    write('spec_classification.json', {'datasets': [s for s in specs if s['task']=='classification']})
    write('spec_regression.json', {'datasets': [s for s in specs if s['task']=='regression']})
    write('spec_octave.json', {'datasets': [s for s in specs if s['task']=='regression' and s['target_column']==0]})
    write('spec_smoke.json', {'datasets': [s for s in specs if s['fold']==1 and s['target_column'] in {None,0}]})
    write('split_audit.json', audits)
    write('preparation.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                             'source_commit': source_manifest['commit'], 'n_fold_tasks': len(specs),
                             'classification_fold_tasks': 10, 'regression_fold_tasks': 50,
                             'all_source_files_hash_verified': len(source_manifest['files']),
                             'all_points_retained': True, 'no_test_statistics_fitted': True})
    print(f'Prepared {len(specs)} fold/tasks; source files verified: {len(source_manifest["files"])}')


if __name__ == '__main__':main()
