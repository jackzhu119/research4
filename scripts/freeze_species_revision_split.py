"""Extend the historical 60/20/20 pools without moving any original ZY sample.

No spectral values or model scores are read. Source hashes, annotation hashes,
allocation policy and all IDs are retained so later methods share one split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 20261101
ROLES = ('train', 'calibration', 'prediction')
HISTORICAL = Path(__file__).resolve().parents[1] / 'data/research/species_revision_20261108/historical_split_plan.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(n):
    if n < 3:
        raise ValueError('Each origin group needs at least one sample per pool')
    train, calibration = max(1, 60 * n // 100), max(1, 20 * n // 100)
    if train + calibration >= n:
        train = n - calibration - 1
    return train, calibration, n - train - calibration


def make_split(rows, seed=SEED, inherited=None):
    if inherited is None:
        inherited = {row['record_id']: ('train' if row['role'] == 'training' else row['role'])
                     for row in json.loads(HISTORICAL.read_text())['records']}
    groups = defaultdict(list)
    ids = [row['physical_sample_id'] for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError('Confirmed unique physical samples are required; repeats need grouped allocation')
    if len({row['record_id'] for row in rows}) != len(rows):
        raise ValueError('record_id must be unique')
    for row in rows:
        groups[row['origin_group_id']].append(row)
    output = []
    for group, members in sorted(groups.items()):
        members = sorted(members, key=lambda x: x['record_id'])
        group_seed = int.from_bytes(hashlib.sha256(f'{seed}:{group}'.encode()).digest()[:8], 'big')
        order = np.random.default_rng(group_seed).permutation(len(members))
        n_train, n_calibration, _ = counts(len(members))
        for position, index in enumerate(order):
            row = members[index]
            role = ROLES[0] if position < n_train else ROLES[1] if position < n_train + n_calibration else ROLES[2]
            if group.startswith('ZY'):
                if row['record_id'] not in inherited:
                    raise ValueError('Historical ZY records require an explicit saved role')
                role = inherited[row['record_id']]
                if role not in ROLES:
                    raise ValueError('Unknown historical role')
            output.append({key: row[key] for key in (
                'record_id', 'physical_sample_id', 'physical_sample_id_basis',
                'origin_group_id', 'species', 'origin', 'source_workbook',
                'excel_column', 'scan_header', 'batch_id', 'dataset_version')}
                | {'role': role, 'tiny_group': 'true' if len(members) < 10 else 'false',
                   'split_seed': str(seed), 'historically_used_in_model_development':
                   'true' if group.startswith('ZY') else 'unknown'})
    return sorted(output, key=lambda x: x['record_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / 'data/research/species_revision_20261108'
    output.mkdir(parents=True, exist_ok=True)
    metadata = root / 'data/metadata_template.csv'
    registry_path = root / 'data/active_workbooks.json'
    registry = json.loads(registry_path.read_text())
    with metadata.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    workbook_counts = Counter(row['source_workbook'] for row in rows)
    if workbook_counts != Counter({item['path']: item['record_count'] for item in registry['workbooks']}):
        raise ValueError('Metadata and active registry disagree')
    split = make_split(rows)
    manifest = output / 'split_manifest.csv'
    with manifest.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(split)
    summaries = []
    for group in sorted({row['origin_group_id'] for row in split}):
        selected = [row for row in split if row['origin_group_id'] == group]
        summaries.append({'origin_group_id': group, 'species': selected[0]['species'],
                          'origin': selected[0]['origin'], 'total': len(selected),
                          **{role: sum(row['role'] == role for row in selected) for role in ROLES},
                          'tiny_group': len(selected) < 10})
    summary = {'dataset_version': registry['version'], 'total': len(split),
               'roles': dict(Counter(row['role'] for row in split)), 'by_group': summaries,
               'by_species': {kind: {role: sum(row['species'] == kind and row['role'] == role for row in split)
                                    for role in ROLES} for kind in sorted(registry['kind_counts'])},
               'physical_samples_unique': len({row['physical_sample_id'] for row in split}),
               'manifest_sha256': sha(manifest)}
    protocol = {
        'protocol_id': 'species-revision-frozen-split-20261108-v1',
        'date': '2026-11-08', 'dataset_version': registry['version'],
        'authorization': 'User requested advance training/calibration/prediction split and reuse of prior experiments.',
        'source_protocol': 'docs/CALIBRATION_OPTIMIZATION_PROTOCOL.md (historical research branch)',
        'historical_split_source': {
            'copied_path': 'data/research/species_revision_20261108/historical_split_plan.json',
            'repository_path': 'data/processed/zaoren_threeway_20261031_v1/split_plan.json',
            'source_commit': '3fc996e99cce8ff5799b5a2af9b565f982dfc899',
            'github_blob_sha': 'cd2e6ba12e647ba9535e01b3e74eb87bc51dc763',
            'sha256': sha(HISTORICAL),
            'records_inherited_unchanged': 855,
            'original_roles': {'training': 513, 'calibration': 171, 'prediction': 171}},
        'split_seed': SEED, 'model_seed': 42,
        'allocation': 'Original ZY855 retain every historical role. Added groups use max(1,floor(.60*N)) training, max(1,floor(.20*N)) calibration, remainder prediction; reserve >=1 prediction. N=3 uses 1/1/1; N=5 uses 3/1/1.',
        'assignment': 'ZY records inherited from saved historical split, not rerandomized. For added817 records: sort record_id; independent NumPy PCG64 generator per group seeded by first 8 bytes of SHA256(seed:group); no spectral data or scores used.',
        'manifest_sha256': sha(manifest), 'metadata_sha256': sha(metadata),
        'active_registry_sha256': sha(registry_path),
        'workbook_hashes': {item['path']: item['sha256'] for item in registry['workbooks']},
        'pool_rules': {
            'train': 'Fit preprocessing, variable selection and models; choose all hyperparameters and final method by training-only CV.',
            'calibration': 'Fixed support pool for prespecified adaptation only; do not select the final model/method with these labels. Calibration-only baselines may train exclusively here.',
            'prediction': 'One locked final comparison; labels used for scoring only. Do not select hyperparameters, features, seeds, frequency intervals, calibration strength or methods using these results.'},
        'tasks_share_master_split': ['four_kind_classification', 'zaoren_vs_suanzaoren', 'within_kind_origin_group_classification'],
        'inherited_methods': ['original/SNV preprocessing', 'SVM', 'PLS-DA', 'MC feature selection', 'source-only / support-only / full-refit / frozen-representation / anchor-score calibration controls'],
        'training_cv': 'Reuse historical 5/2/2 nested-CV principles and seeds when class counts permit. Select feasible training-only folds from counts before fitting; tiny classes with one training specimen cannot support stratified tuning and require prespecified parameters or a blocked result.',
        'tiny_group_policy': 'Keep all groups; N<10 explicitly flagged. No oversampling creates independent specimens. A one-sample prediction denominator is diagnostic only.',
        'batch_scope': 'Internal specimen-disjoint calibration. Original ZY groups each have one user-declared source batch; no new batch counts or IDs were provided. No real cross-batch calibration claim.',
        'history_scope': 'The 855 original ZY specimens were used in historical CV and model development. This split isolates the new run, but is not a previously unseen external or blind confirmation dataset.',
        'botanical_scope': 'Chinese user-confirmed kind categories, not independent Latin botanical authentication.',
        'origin_scope': 'Preserve workbook source groups; 山西 and 山西灵石 are not asserted to be mutually exclusive administrative territories.',
        'cross_kind_transfer_scope': 'Source and target origin classes differ; reuse source representation with a newly fitted target head. Cross-kind score-column calibration is invalid without matching label semantics. Actual positive transfer requires matched target query/support and target-only baselines.',
        'external_confirmation_needed': 'Independently collected new batches covering known target groups, with acquisition metadata, for a confirmatory cross-batch result.'}
    for name, value in [('split_summary.json', summary), ('protocol.json', protocol)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
