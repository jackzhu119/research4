"""Prepare an auditable, record-disjoint pharmaceutical NIR transfer benchmark.

The published Shootout 2002 data measure corresponding tablet records on two
instruments. This mirror does not retain original physical sample identifiers;
pairing is reconstructed from exact agreement of all three reference fields and
row order. Recovered record IDs must not be described as independently verified
physical sample or batch IDs. SampleType is a supplied split, not a class label.

Default protocol: Instrument 1 Train+Validate (195 records) is the source;
Instrument 2 Test (460 different records) is the target pool. Later calibration
must split this target pool into support/query by target_ids. Paired source
records' Instrument 2 spectra and paired target records' Instrument 1 spectra
are deliberately omitted from the output.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np


REPO = Path(__file__).resolve().parents[1]
COMMIT = '0403cd4014af9b9d1dcac20c42ca3543ca56a9d5'
BASE = 'https://raw.githubusercontent.com/caseykneale/ChemometricsTools.jl/' + COMMIT + '/data/'
SOURCES = {
    'Pharma_Instrument1.csv': 'a760cf1a72a2268b59ee43a651430d0e30cdadfc9a9923dde5e72bef946aaaea',
    'Pharma_Instrument2.csv': '36e46b5ff67e46e53c885df83da54c962c56206f3df7c6647a6a41b207dc86f2',
    'Readme.md': '54a6619f108107c16f98e2c1357b0c95c12be3e06a50006236e7bbb86974a723',
}
REFERENCES = [
    {'url': 'https://eigenvector.com/data/tablets/index.html',
     'role': 'Original dataset provider; URL currently not retrievable through the web tool.'},
    {'url': 'https://www.spectroscopyeurope.com/td-column/back-basics-final-calibration',
     'role': 'Davies and Fearn, 2007; assay is mg/tablet; NIR 600–1898 nm; the 40 validation tablets represent two production batches.'},
    {'url': 'https://raw.githubusercontent.com/cran/ChemometricsWithRData/master/man/shootout.Rd',
     'role': 'Package dataset documentation: corresponding tablets on two instruments, nominal active ingredient 200 mg/tablet, signal log(1/T). This separate documentation says 654 tablets; it is not used to change the actual mirror rows.'},
    {'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC9823907/',
     'doi': '10.3390/molecules28010406',
     'role': 'Primary research methods confirm 655 tablets, 1310 paired spectra, 600–1898 nm at 2 nm; active pharmaceutical ingredient prediction.'},
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def checked_file(directory, name, download):
    path = directory/name
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f'{path}: supply the existing mirror or use --download')
        directory.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BASE+name, timeout=45) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SOURCES[name]:
            raise ValueError(f'Download hash mismatch: {name}')
        path.write_bytes(content)
    if sha256(path) != SOURCES[name]:
        raise ValueError(f'Pinned source hash mismatch: {path}')
    return path


def load_csv(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames
    if fields[:4] != ['SampleType', 'weight', 'hardness', 'assay']:
        raise ValueError('Unexpected reference/split fields')
    axis = np.asarray([float(value) for value in fields[4:]])
    if not np.array_equal(axis, np.arange(600., 1900., 2.)):
        raise ValueError('Mirror spectral axis differs from the audited 650-point grid')
    X = np.asarray([[float(row[column]) for column in fields[4:]] for row in rows])
    references = np.asarray([[float(row[column]) for column in fields[1:4]] for row in rows])
    split = np.asarray([row['SampleType'] for row in rows])
    if not np.isfinite(X).all() or not np.isfinite(references).all():
        raise ValueError('Nonfinite spectral or reference values')
    if dict(Counter(split)) != {'Train': 155, 'Test': 460, 'Validate': 40}:
        raise ValueError('Supplied split differs from the audited source')
    keys = [tuple(row[column] for column in fields[1:4]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Reference fingerprints no longer uniquely identify mirror records')
    ids = np.asarray(['shootout-record-' + hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).hexdigest()
                      for key in keys])
    return X, references, split, ids, axis, keys


def hashes(X):
    return [hashlib.sha256(row.astype('<f8').tobytes()).hexdigest() for row in X]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=REPO/'outputs/external-discovery/public')
    parser.add_argument('--output', type=Path, default=REPO/'outputs/research-stage2/pharma')
    parser.add_argument('--source-instrument', type=int, choices=[1, 2], default=1)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a new/empty output directory')
    paths = {name: checked_file(args.data_dir, name, args.download) for name in SOURCES}
    instruments = {i: load_csv(paths[f'Pharma_Instrument{i}.csv']) for i in [1, 2]}
    first, second = instruments[1], instruments[2]
    if not np.array_equal(first[1], second[1]) or not np.array_equal(first[2], second[2]) or first[5] != second[5]:
        raise ValueError('Reference fields or split/order do not match across instruments')
    if not np.array_equal(first[3], second[3]) or not np.array_equal(first[4], second[4]):
        raise ValueError('Recovered record pairing or feature grids differ')
    target_instrument = 3-args.source_instrument
    Xs, reference, split, ids, axis, _ = instruments[args.source_instrument]
    Xt = instruments[target_instrument][0]
    source_mask = np.isin(split, ['Train', 'Validate'])
    target_mask = split == 'Test'
    source_ids, target_ids = ids[source_mask], ids[target_mask]
    if set(source_ids) & set(target_ids):
        raise ValueError('A recovered tablet record appears in source and target')
    source_hashes, target_hashes = hashes(Xs[source_mask]), hashes(Xt[target_mask])
    if set(source_hashes) & set(target_hashes):
        raise ValueError('An exact spectrum appears in both source and target')
    if len(set(source_hashes)) != len(source_hashes) or len(set(target_hashes)) != len(target_hashes):
        raise ValueError('Exact duplicate spectra within a benchmark partition')
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output/'transfer_input.npz'
    np.savez_compressed(output,
                        X_source=Xs[source_mask], y_source=reference[source_mask, 2], source_ids=source_ids,
                        X_target=Xt[target_mask], y_target=reference[target_mask, 2], target_ids=target_ids,
                        source_original_row_indices=np.flatnonzero(source_mask),
                        target_original_row_indices=np.flatnonzero(target_mask),
                        source_split_labels=split[source_mask], target_split_labels=split[target_mask],
                        feature_axis=axis, feature_axis_unit=np.asarray('nm'), target_unit=np.asarray('mg/tablet'),
                        source_instrument=np.asarray(f'Instrument {args.source_instrument}'),
                        target_instrument=np.asarray(f'Instrument {target_instrument}'))
    registry = {
        'dataset_id': 'idrc_shootout_2002_tablets',
        'description': 'Pharmaceutical tablet NIR calibration-transfer engineering benchmark; not Chinese medicinal-herb data',
        'mirror_repository': 'https://github.com/caseykneale/ChemometricsTools.jl', 'mirror_commit': COMMIT,
        'files': [{'filename': name, 'url': BASE+name, 'bytes': path.stat().st_size, 'sha256': sha256(path)}
                  for name, path in paths.items()],
        'metadata_sources': REFERENCES,
        'target': {'name': 'assay', 'description': 'Proprietary active ingredient amount per tablet',
                   'unit': 'mg/tablet', 'unit_evidence': REFERENCES[1]['url'],
                   'limitation': 'Molecule identity is not supplied; chemical reference values are taken from the public mirror, not independently remeasured.'},
        'license': {'mirror_statement': 'Package Readme says author obtained permission to redistribute via LinkedIn.',
                    'direct_permission_independently_inspected': False,
                    'data_specific_license_text_found': False,
                    'delivery_policy': 'Local research evaluation; commit downloader/provenance/hash only, not mirrored raw spectral arrays.'},
    }
    audit = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha256(__file__), 'output_sha256': sha256(output),
        'all_pinned_source_files_verified': True,
        'observed_shape_each_instrument': [len(split), len(axis)],
        'axis': {'first': float(axis[0]), 'last': float(axis[-1]), 'step': 2., 'unit': 'nm'},
        'signal': 'Source log(1/T) spectra; no double-log, resizing, scaling, selection, or fitting applied by this adapter.',
        'source': {'instrument': args.source_instrument, 'supplied_splits': ['Train', 'Validate'],
                   'rows': int(source_mask.sum()), 'target_range': [float(reference[source_mask, 2].min()), float(reference[source_mask, 2].max())],
                   'ids': source_ids.tolist()},
        'target_pool': {'instrument': target_instrument, 'supplied_splits': ['Test'],
                        'rows': int(target_mask.sum()), 'target_range': [float(reference[target_mask, 2].min()), float(reference[target_mask, 2].max())],
                        'ids': target_ids.tolist(), 'support_query_partitioned': False},
        'all_values_finite': True, 'record_ids_unique_all_655': True,
        'instrument_reference_columns_match_exactly': True,
        'source_target_id_overlap': 0, 'source_target_exact_spectrum_overlap': 0,
        'source_partition_paired_target_instrument_spectra_omitted': True,
        'target_partition_paired_source_instrument_spectra_omitted': True,
        'identity': {'original_physical_sample_id_column_available': False,
                     'recovered_record_id': 'SHA256 of exact (weight,hardness,assay) source string tuple; all 655 tuples unique and identical across instrument files.',
                     'evidence': 'Published dataset is paired by tablet; CSV record pairing is inferred from same order and complete reference-field agreement.',
                     'physical_sample_independence_verified': False,
                     'batch_identity_verified': False},
        'document_conflicts': [
            {'source': 'Mirror data/Readme.md', 'claim': '653 tablets; 600–1398 nm',
             'observed': '655 rows per instrument; 650 columns named 600–1898 at step2.',
             'resolution': 'Use actual hash-verified arrays; 655/600–1898 is additionally supported by the 2023 primary research methods. Preserve the incorrect mirror description in provenance.'},
            {'source': REFERENCES[2]['url'], 'claim': '654 tablets in separate R package documentation',
             'resolution': 'Different documentation/version; no deletion or inferred correction applied to the 655-row CSV mirror.'}],
        'protocol': 'Source-only fitting uses 195 supplied Train+Validate records from one instrument. Target calibration/query pool is the 460 supplied Test records from the other instrument. Later support/query must use disjoint target_ids and reserve query labels for final scoring.',
        'limitations': ['Recovered record-disjointness is enforced; independent physical plants/tablets or batches are not proven by IDs absent from the mirror.',
                        'Source includes the supplied Validate subset; this is a declared new transfer protocol, not the original Shootout competition protocol.',
                        'SampleType is a partition field and must never be treated as a class label.',
                        'The 40 Validate rows were described in the 2007 article as production batches, but no explicit batch mapping is present here; no inferred batch labels are emitted.',
                        'No support/query labels were used for fitting by this preparation script; it does not run or claim improvement of any model.'],
    }
    write_json(args.output/'source_registry.json', registry)
    write_json(args.output/'audit.json', audit)
    print(json.dumps({'output': str(output), 'source_rows': int(source_mask.sum()),
                      'target_pool_rows': int(target_mask.sum()), 'features': len(axis),
                      'target_unit': 'mg/tablet', 'recovered_record_overlap': 0,
                      'physical_sample_independence_verified': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':main()
