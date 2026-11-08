"""Prepare hash-checked paired spectra for the kind-label revision.

Read only the active registry. Workbook XML is streamed because the uploaded
XLSX dimension metadata incorrectly reports A1:A1. No model fitting occurs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import posixpath
import re
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from openpyxl import load_workbook
from openpyxl.utils.cell import column_index_from_string

MAIN = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sheet_members(z):
    relations = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    targets = {r.get('Id'): r.get('Target') for r in relations}
    book = ET.fromstring(z.read('xl/workbook.xml'))
    result = {}
    for sheet in book.find(MAIN + 'sheets'):
        target = targets[sheet.get(REL + 'id')]
        result[sheet.get('name')] = (target.lstrip('/') if target.startswith('/')
                                     else posixpath.normpath('xl/' + target))
    return result


def shared_strings(z):
    if 'xl/sharedStrings.xml' not in z.namelist():
        return []
    root = ET.fromstring(z.read('xl/sharedStrings.xml'))
    return [''.join(t.text or '' for t in s.iter(MAIN + 't')) for s in root]


def cell_value(cell, strings):
    if cell.get('t') == 'inlineStr':
        return ''.join(t.text or '' for t in cell.iter(MAIN + 't'))
    v = cell.find(MAIN + 'v')
    if v is None or v.text is None:
        return None
    if cell.get('t') == 's':
        return strings[int(v.text)]
    return v.text


def read_sheet(z, member, strings, columns):
    """Return headers, coordinate and samples in explicit metadata column order."""
    wanted = [1] + columns
    wanted_set = set(wanted)
    headers = None
    values = []
    original_row_numbers = []
    with z.open(member) as stream:
        for _, row in ET.iterparse(stream, events=('end',)):
            if row.tag != MAIN + 'row':
                continue
            cells = {}
            for cell in row:
                coordinate = cell.get('r', '')
                match = re.match(r'([A-Z]+)', coordinate)
                if match is None:
                    raise ValueError(f'Unaddressed worksheet cell: {coordinate}')
                column = column_index_from_string(match.group(1))
                if column in wanted_set:
                    cells[column] = cell_value(cell, strings)
            if int(row.get('r')) == 1:
                headers = [cells.get(col) for col in wanted]
            elif any(cells.get(col) is not None for col in wanted):
                if any(cells.get(col) is None for col in wanted):
                    raise ValueError(f'Missing numeric value in {member} row {row.get("r")}')
                try:
                    values.append([float(cells[col]) for col in wanted])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'Nonnumeric cell in {member} row {row.get("r")}') from exc
                original_row_numbers.append(int(row.get('r')))
            row.clear()
    if headers is None or len(values) < 2:
        raise ValueError(f'Insufficient rows: {member}')
    matrix = np.asarray(values, dtype=np.float64)
    axis = matrix[:, 0]
    spectra = matrix[:, 1:].T
    if not np.isfinite(matrix).all():
        raise ValueError(f'Nonfinite coordinates or spectra in {member}')
    if not np.all(np.diff(axis) > 0):
        raise ValueError(f'Coordinates not strictly increasing in {member}')
    return headers, axis, spectra, original_row_numbers


def duplicate_groups(matrix, identifiers):
    buckets = defaultdict(list)
    for identifier, row in zip(identifiers, matrix):
        buckets[hashlib.sha256(np.asarray(row, dtype='<f8').tobytes()).hexdigest()].append(str(identifier))
    return [ids for ids in buckets.values() if len(ids) > 1]


def independent_spot_checks(root, entries, metadata, arrays):
    """Cross-check a distinct Excel reader on prespecified records/cells."""
    lookup = {str(value): i for i, value in enumerate(arrays['record_ids'])}
    checks = []
    for group in ('ZY01', 'ZY04', 'HQ03', 'ZR02', 'TM04'):
        entry = next(e for e in entries if e['origin_group_id'] == group)
        sample = sorted((r for r in metadata if r['origin_group_id'] == group),
                        key=lambda r: column_index_from_string(r['excel_column']))[-1]
        column = column_index_from_string(sample['excel_column'])
        workbook = load_workbook(root / entry['path'], read_only=True, data_only=True)
        try:
            for sheet, key in (('吸收系数', 'X_alpha'), ('折射率', 'X_n')):
                workbook[sheet].reset_dimensions()
                rows = list(workbook[sheet].iter_rows(min_row=2, max_row=5,
                            min_col=column, max_col=column, values_only=True))
                actual = np.asarray([row[0] for row in rows], dtype=float)
                expected = arrays[key][lookup[sample['record_id']], :4]
                if not np.array_equal(actual, expected):
                    raise ValueError(f'Independent source-cell check failed: {group}/{sheet}')
                checks.append({'record_id': sample['record_id'], 'sheet': sheet,
                               'excel_column': sample['excel_column'],
                               'rows': [2, 3, 4, 5], 'exact_match': True})
        finally:
            workbook.close()
    return {'method': 'openpyxl data_only/read_only, reset_dimensions; distinct from preparation XML decoder',
            'cells_checked': 4 * len(checks), 'checks': checks}


def prepare(root, destination):
    registry_path = root / 'data/active_workbooks.json'
    metadata_path = root / 'data/metadata_template.csv'
    registry = json.loads(registry_path.read_text())
    with metadata_path.open(encoding='utf-8-sig', newline='') as f:
        metadata = list(csv.DictReader(f))
    by_path = defaultdict(list)
    for item in metadata:
        by_path[item['source_workbook']].append(item)
    entries = registry['workbooks']
    if set(by_path) != {entry['path'] for entry in entries}:
        raise ValueError('Metadata and active registry paths disagree')
    records = []
    audits = []
    batches = []
    primary_axes = []
    transfer_axis = np.linspace(.21, 1.49, 128)
    for entry in entries:
        path = root / entry['path']
        digest = sha(path)
        if digest != entry['sha256']:
            raise ValueError(f'Hash mismatch: {path}')
        rows = sorted(by_path[entry['path']], key=lambda r: column_index_from_string(r['excel_column']))
        if len(rows) != entry['record_count']:
            raise ValueError(f'Record count mismatch: {path}')
        if any(r['species'] != entry['species'] or r['origin_group_id'] != entry['origin_group_id'] for r in rows):
            raise ValueError(f'Metadata labels disagree with registry: {path}')
        columns = [column_index_from_string(r['excel_column']) for r in rows]
        if len(set(columns)) != len(columns):
            raise ValueError(f'Duplicate metadata column: {path}')
        with ZipFile(path) as z:
            members = sheet_members(z)
            strings = shared_strings(z)
            ah, aa, ax, ar = read_sheet(z, members['吸收系数'], strings, columns)
            nh, na, nx, nr = read_sheet(z, members['折射率'], strings, columns)
        if ah != nh or ah[1:] != [r['scan_header'] for r in rows]:
            raise ValueError(f'Absorption/refractive-index headers or metadata disagree: {path}')
        if not np.array_equal(aa, na) or ar != nr:
            raise ValueError(f'Absorption/refractive-index coordinates differ: {path}')
        mask = (aa >= .1) & (aa <= 1.5)
        if mask.sum() < 2:
            raise ValueError(f'Insufficient main-interval data: {path}')
        if aa[0] > transfer_axis[0] or aa[-1] < transfer_axis[-1]:
            raise ValueError(f'Transfer interval would require extrapolation: {path}')
        primary_axes.append(aa[mask])
        batches.append((ax[:, mask], nx[:, mask], aa[mask],
                        np.vstack([np.interp(transfer_axis, aa, x) for x in ax]),
                        np.vstack([np.interp(transfer_axis, na, x) for x in nx])))
        records.extend(rows)
        audits.append({'path': entry['path'], 'sha256': digest, 'origin_group_id': entry['origin_group_id'],
                       'species': entry['species'], 'records': len(rows), 'numeric_rows': len(aa),
                       'axis_min': float(aa[0]), 'axis_max': float(aa[-1]),
                       'primary_points': int(mask.sum()), 'primary_min': float(aa[mask][0]),
                       'primary_max': float(aa[mask][-1]), 'header_coordinate_pairing': 'exact',
                       'all_numeric_values_finite': True, 'axis_strictly_increasing': True})
        print(f'checked {entry["origin_group_id"]}: {len(rows)} samples, {mask.sum()} main points', flush=True)
    same_axis = all(np.array_equal(primary_axes[0], axis) for axis in primary_axes[1:])
    if same_axis:
        axis = primary_axes[0]
        alpha = np.concatenate([b[0] for b in batches])
        refractive = np.concatenate([b[1] for b in batches])
        main_axis_policy = 'Original shared coordinate points in 0.1 <= f <= 1.5; no interpolation'
    else:
        axis = transfer_axis.copy()
        alpha = np.concatenate([b[3] for b in batches])
        refractive = np.concatenate([b[4] for b in batches])
        main_axis_policy = ('Axes differ: historical common 128-point 0.21-1.49 grid; linear interpolation '
                            'without extrapolation. This narrows the original main interval and must be disclosed.')
    identifiers = np.asarray([r['record_id'] for r in records])
    if len(set(identifiers)) != len(metadata) or len(records) != registry['total_active_records']:
        raise ValueError('Duplicated/missing record IDs')
    duplicate_alpha = duplicate_groups(alpha, identifiers)
    duplicate_n = duplicate_groups(refractive, identifiers)
    duplicate_joint = duplicate_groups(np.concatenate([alpha, refractive], axis=1), identifiers)
    arrays = {'record_ids': identifiers, 'X_alpha': alpha, 'X_n': refractive, 'axis': axis,
              'X_alpha_transfer': np.concatenate([b[3] for b in batches]),
              'X_n_transfer': np.concatenate([b[4] for b in batches]), 'axis_transfer': transfer_axis,
              'origin_group_ids': np.asarray([r['origin_group_id'] for r in records]),
              'species': np.asarray([r['species'] for r in records]),
              'origins': np.asarray([r['origin'] for r in records]),
              'dataset_version': np.asarray(registry['version'])}
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / 'features.npz'
    np.savez_compressed(output, **arrays)
    audit = {'dataset_version': registry['version'], 'registry_sha256': sha(registry_path),
             'metadata_sha256': sha(metadata_path), 'features_sha256': sha(output),
             'records': len(records), 'workbooks': len(entries), 'feature_count': int(len(axis)),
             'same_primary_axes': same_axis, 'main_axis_policy': main_axis_policy,
             'transfer_axis_policy': 'Historical 128-point linspace(0.21,1.49); per-spectrum linear interpolation without extrapolation',
             'transfer_feature_count': 128, 'frequency_unit': 'Workbook coordinate labeled Thz; instrument metrology not independently verified',
             'duplicate_alpha_groups': duplicate_alpha, 'duplicate_n_groups': duplicate_n,
             'duplicate_joint_groups': duplicate_joint,
             'duplicate_policy': 'Report only; no automatic exclusions or reassignment. Investigate if any group spans roles.',
             'model_input_policy': 'Numeric spectra only; IDs, paths, origin labels and specimen metadata are never predictors',
             'preprocessing_policy': 'No fitted transforms, selectors, or model parameters estimated here; SNV/scaling remain in training pipelines',
             'workbook_checks': audits}
    if same_axis:
        audit['independent_spot_checks'] = independent_spot_checks(root, entries, metadata, arrays)
    else:
        audit['independent_spot_checks'] = {'status': 'not applicable to interpolated main features'}
    (destination / 'feature_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ('records', 'workbooks', 'feature_count', 'same_primary_axes', 'duplicate_joint_groups')}, ensure_ascii=False), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('data/research/species_revision_20261108'))
    args = parser.parse_args()
    root = args.root.resolve()
    destination = args.output if args.output.is_absolute() else root / args.output
    prepare(root, destination)


if __name__ == '__main__':
    main()
