"""Independently validate the confirmed-name migration without changing inputs.

Only the JSON validation receipt is written. Spectral numeric values are not
recomputed: whole-file hashes protect renamed files, ZIP CRC checks cover all
members, and streamed first rows verify cross-sheet record correspondence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import posixpath
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from openpyxl import load_workbook


MAIN = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
SHEETS = ('时域', '频域', '折射率', '吸收系数')
VERSION = '2026-11-08-confirmed-origin-kind-v1'
BASIS = 'user_confirmed_origin_and_kind_20261108'
KINDS = {'枣仁': 364, '酸枣仁': 500, '黄芪': 403, '天麻': 405}
EXPECTED = {
    'ZY01': ('陕西延安枣仁.xlsx', '陕西延安', '枣仁', 350),
    'ZY02': ('云南枣仁.xlsx', '云南', '枣仁', 5),
    'ZY03': ('内蒙古酸枣仁.xlsx', '内蒙古', '酸枣仁', 100),
    'ZY04': ('山西灵石酸枣仁.xlsx', '山西灵石', '酸枣仁', 100),
    'ZY05': ('安徽亳州酸枣仁.xlsx', '安徽亳州', '酸枣仁', 100),
    'ZY06': ('河北石家庄酸枣仁.xlsx', '河北石家庄', '酸枣仁', 100),
    'ZY07': ('山西酸枣仁.xlsx', '山西', '酸枣仁', 100),
    'HQ01': ('黄芪曲靖.xlsx', '曲靖', '黄芪', 100),
    'HQ02': ('黄芪内蒙古固阳.xlsx', '内蒙古固阳', '黄芪', 100),
    'HQ03': ('黄芪甘肃.xlsx', '甘肃', '黄芪', 100),
    'HQ04': ('黄芪山西大同.xlsx', '山西大同', '黄芪', 103),
    'ZR01': ('陕西榆林枣仁.xlsx', '陕西榆林', '枣仁', 3),
    'ZR02': ('辽宁枣仁.xlsx', '辽宁', '枣仁', 3),
    'ZR03': ('内蒙古枣仁.xlsx', '内蒙古', '枣仁', 3),
    'TM01': ('天麻贵州雷山.xlsx', '贵州雷山', '天麻', 100),
    'TM02': ('天麻吉林长白山.xlsx', '吉林长白山', '天麻', 102),
    'TM03': ('天麻安徽毫州.xlsx', '安徽毫州', '天麻', 103),
    'TM04': ('天麻云南.xlsx', '云南', '天麻', 100),
}


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def csv_bytes(value: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(value.decode('utf-8-sig'))))


def sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def normalized(value):
    return '' if value is None else str(value)


def sheet_first_rows(z: ZipFile) -> dict:
    """Read actual header and first data row; ignore inaccurate dimensions."""
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        root = ET.fromstring(z.read('xl/sharedStrings.xml'))
        shared = [''.join(node.text or '' for node in item.iter(MAIN + 't')) for item in root]
    relations = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    targets = {item.get('Id'): item.get('Target') for item in relations}
    book = ET.fromstring(z.read('xl/workbook.xml'))
    output = {}
    for sheet in book.find(MAIN + 'sheets'):
        target = targets[sheet.get(REL + 'id')]
        member = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
        rows = []
        with z.open(member) as handle:
            for _, row in ET.iterparse(handle, events=('end',)):
                if row.tag != MAIN + 'row':
                    continue
                cells = []
                for cell in row:
                    value = cell.find(MAIN + 'v')
                    if cell.get('t') == 'inlineStr':
                        text = ''.join(node.text or '' for node in cell.iter(MAIN + 't'))
                    elif cell.get('t') == 's':
                        text = shared[int(value.text)]
                    else:
                        text = value.text if value is not None else ''
                    cells.append({'cell': cell.get('r'), 'value': text})
                rows.append({'row_number': row.get('r'), 'cells': cells})
                row.clear()
                if len(rows) == 2:
                    break
        output[sheet.get('name')] = {'member': member, 'rows': rows}
    return output


class Validator:
    def __init__(self, root: Path):
        self.root = root
        self.checks = []
        self.files = []
        self.hashes = {}
        self.baseline_commit = None

    def check(self, name: str, passed: bool, details=None):
        item = {'check': name, 'passed': bool(passed)}
        if details is not None:
            item['details'] = details
        self.checks.append(item)

    def digest(self, relative: str) -> str:
        if relative not in self.hashes:
            self.hashes[relative] = sha(self.root / relative)
        return self.hashes[relative]

    def head(self, relative: str) -> bytes:
        if not self.baseline_commit:
            raise ValueError('A validated immutable migration_base_commit is required')
        return subprocess.check_output(['git', '-C', str(self.root), 'show', self.baseline_commit + ':' + relative])

    def run(self):
        root = self.root
        registry = json.loads((root / 'data/active_workbooks.json').read_text())
        baseline = registry.get('migration_base_commit', '')
        self.check('migration_baseline_is_full_commit_sha', bool(re.fullmatch(r'[0-9a-f]{40}', baseline)))
        if not re.fullmatch(r'[0-9a-f]{40}', baseline):
            raise ValueError('registry.migration_base_commit must be a full 40-character commit SHA')
        resolved = subprocess.check_output(['git', '-C', str(root), 'rev-parse', baseline + '^{commit}'], text=True).strip()
        self.check('migration_baseline_commit_available', resolved == baseline)
        self.baseline_commit = baseline
        entries = registry['workbooks']
        independence_path = root / 'data/annotations/specimen_independence_user_declaration_20261108.json'
        independence = json.loads(independence_path.read_text()) if independence_path.is_file() else None
        if independence is not None:
            expected_groups = {g for g in EXPECTED if not g.startswith('ZY')}
            self.check('new817_independence_declaration',
                       independence.get('verbatim_answer') == '每列都是独立实物样品'
                       and independence.get('scope', {}).get('record_count') == 817
                       and set(independence.get('scope', {}).get('groups', [])) == expected_groups
                       and independence.get('applied_annotation', {}).get('physical_sample_id_basis')
                       == 'user_declared_independent_20261108')
        by_group = {item['origin_group_id']: item for item in entries}
        by_path = {item['path']: item for item in entries}
        kind_counts = Counter()
        for item in entries:
            kind_counts[item['species']] += item['record_count']
        self.check('registry_version', registry['version'] == VERSION)
        self.check('registry_exact_18_groups', len(entries) == 18 and set(by_group) == set(EXPECTED))
        self.check('registry_1672_records', sum(item['record_count'] for item in entries) == 1672
                   and registry['total_active_records'] == 1672 and registry['total_active_workbooks'] == 18)
        self.check('registry_four_kind_counts', dict(kind_counts) == KINDS == registry['kind_counts'], dict(kind_counts))
        manifest = csv_rows(root / 'data/manifest.csv')
        manifest_by_path = {item['path']: item for item in manifest}
        self.check('active_manifest_exact_paths', len(manifest) == 18 and set(manifest_by_path) == set(by_path))
        mapping = csv_rows(root / 'data/confirmed_names_20261108.csv')
        mapped = {item['group_id']: item for item in mapping}
        self.check('rename_map_exact_18_groups', len(mapping) == 18 and set(mapped) == set(EXPECTED))
        audit = json.loads((root / 'data/validation/confirmed_names_20261108/upload_audit.json').read_text())
        audited = {item['name']: item for item in audit}
        self.check('upload_audit_exact_18_files', len(audit) == 18 and set(audited) == {x[0] for x in EXPECTED.values()})
        snap = root / 'data/archive/pre_confirmation_20261108'
        for filename in ('active_workbooks.json', 'manifest.csv', 'metadata_template.csv'):
            self.check('snapshot_matches_migration_baseline_' + filename,
                       (snap / filename).read_bytes() == self.head('data/' + filename))
        old_registry = json.loads((snap / 'active_workbooks.json').read_text())
        old_by_path = {item['path']: item for item in old_registry['workbooks']}
        old_by_group = {item.get('origin_group_id'): item for item in old_registry['workbooks']
                        if item.get('origin_group_id')}
        for group, (filename, origin, kind, count) in EXPECTED.items():
            item = by_group[group]
            rel = item['path']
            path = root / rel
            self.check(group + '_expected_confirmed_labels',
                       (path.name, item['origin'], item['species'], item['record_count'])
                       == (filename, origin, kind, count))
            self.check(group + '_file_exists', path.is_file(), rel)
            if not path.is_file():
                continue
            digest = self.digest(rel)
            self.check(group + '_registry_and_manifest_hash_size',
                       digest == item['sha256'] == manifest_by_path[rel]['sha256']
                       and path.stat().st_size == int(manifest_by_path[rel]['size_bytes']))
            self.check(group + '_rename_map_matches_registry',
                       all(normalized(mapped[group][key]) == normalized(value) for key, value in {
                           'new_path': rel, 'uploaded_filename': filename, 'origin': origin, 'species': kind,
                           'record_count': count, 'output_sha256': digest, 'label_basis': BASIS,
                           'source_sha256': item['source_sha256']}.items()))
            self.check(group + '_source_upload_hash_matches', item['source_sha256'] == audited[filename]['sha256'])
            source = mapped[group]['source_original_path']
            self.check(group + '_source_original_path_and_hash', bool(source) and (root / source).is_file()
                       and self.digest(source) == mapped[group]['source_sha256'])
            self.check(group + '_user_confirmation_scope',
                       item['annotation_basis'] == BASIS and item['label_verification_level'] == 'user_confirmation'
                       and item['origin_verified'] is True and item['species_verified'] is True)
            if group.startswith('ZY'):
                previous = old_by_group[group]
                self.check(group + '_numbered_bytes_unchanged', digest == previous['sha256']
                           and item['previous_active_path'] == previous['path'])
            elif group != 'HQ03':
                self.check(group + '_raw_upload_bytes_unchanged', digest == audited[filename]['sha256'])
            else:
                self.check('HQ03_complete_reexport_matches_upload', digest == audited[filename]['sha256'])
                archived = item['superseded_original_path']
                previous = old_by_path[item['previous_active_path']]
                self.check('HQ03_old_incomplete_version_retained', (root / archived).is_file()
                           and self.digest(archived) == previous['sha256'])
            with ZipFile(path) as z:
                bad = z.testzip()
                self.check(group + '_all_zip_members_crc', bad is None, bad)
                structure = sheet_first_rows(z)
                self.check(group + '_four_expected_sheets', set(structure) == set(SHEETS))
                indices_by_sheet = {}
                record_counts = {}
                for sheet in SHEETS:
                    rows = structure[sheet]['rows']
                    self.check(group + '_' + sheet + '_nonempty_header_and_data', len(rows) == 2
                               and rows[0]['row_number'] == '1' and rows[1]['cells'])
                    header = rows[0]['cells']
                    skip = 2 if sheet in ('时域', '频域') else 1
                    samples = header[skip:]
                    record_counts[sheet] = len(samples)
                    pattern = (re.compile(re.escape(group) + r'-(\d{3})(.*)$') if group.startswith('ZY')
                               else re.compile(r'(?:样品|样本)\s*(\d+)(.*)$'))
                    indices = []
                    for sample in samples:
                        found = pattern.fullmatch(sample['value'])
                        if not found:
                            raise AssertionError(f'{group}/{sheet}: unexpected sample header {sample}')
                        indices.append(int(found[1]))
                        if sheet != '时域':
                            self.check(group + '_' + sheet + '_' + str(indices[-1]) + '_no_unexpected_suffix', found[2] == '')
                    self.check(group + '_' + sheet + '_complete_unique_sample_indices',
                               len(indices) == count and set(indices) == set(range(1, count + 1)))
                    indices_by_sheet[sheet] = indices
                self.check(group + '_four_sheet_indices_correspond',
                           len({tuple(x) for x in indices_by_sheet.values()}) == 1)
                if group == 'HQ03':
                    with z.open(structure['折射率']['member']) as handle:
                        total_rows = 0
                        for _, row in ET.iterparse(handle, events=('end',)):
                            if row.tag == MAIN + 'row':
                                total_rows += 1
                                row.clear()
                    self.check('HQ03_refractive_index_2578_data_rows', total_rows == 2579, total_rows)
            self.files.append({'group': group, 'path': rel, 'sha256': digest,
                               'size_bytes': path.stat().st_size, 'sheet_record_counts': record_counts})
        original_manifest = csv_rows(root / 'data/archive/zaoren_superseded_20261031/manifest.csv')
        original_before = csv_bytes(self.head('data/archive/zaoren_superseded_20261031/manifest.csv'))
        original_by_group = {item['origin_group_id']: item for item in original_before}
        for item in original_manifest:
            group = item['origin_group_id']
            previous = original_by_group[group]
            self.check(group + '_historical_original_unmodified',
                       all(item[key] == previous[key] for key in previous)
                       and self.digest(item['archived_path']) == previous['sha256']
                       and (root / item['archived_path']).stat().st_size == int(previous['size_bytes']))
            self.check(group + '_historical_current_pointer', item['current_replacement_path'] == by_group[group]['path'])
        metadata = csv_rows(root / 'data/metadata_template.csv')
        old_metadata = csv_rows(snap / 'metadata_template.csv')
        self.check('metadata_1672_unique_scan_and_record_ids', len(metadata) == 1672
                   and len({r['scan_id'] for r in metadata if r['scan_id']}) == 1672
                   and len({r['record_id'] for r in metadata if r['record_id']}) == 1672)
        new_by_scan = {r['scan_id']: r for r in metadata}
        stable_fields = ('record_id', 'physical_sample_id', 'physical_sample_id_basis')
        unchanged = 0
        for old in old_metadata:
            current = new_by_scan.get(old['scan_id'])
            self.check('preserve_original_scan_' + old['scan_id'], current is not None)
            if old['origin_group_id'].startswith('ZY'):
                self.check('preserve_ZY_identifiers_' + old['record_id'],
                           current is not None and all(current[k] == old[k] for k in stable_fields))
                unchanged += 1
        self.check('preserved_all_855_ZY_records', unchanged == 855)
        counts = Counter(row['source_workbook'] for row in metadata)
        self.check('metadata_uses_only_active_paths', set(counts) == set(by_path))
        self.check('metadata_per_workbook_counts', all(counts[p] == item['record_count'] for p, item in by_path.items()))
        header_cache = {}
        for row in metadata:
            item = by_path[row['source_workbook']]
            group = item['origin_group_id']
            self.check('metadata_labels_' + row['record_id'],
                       all(row[key] == item[key] for key in ('origin', 'species', 'origin_group_id', 'uploaded_filename'))
                       and row['origin_basis'] == BASIS == row['species_basis']
                       and row['label_verification_level'] == 'user_confirmation'
                       and row['origin_verified'] == row['species_verified'] == 'true'
                       and row['dataset_version'] == VERSION)
            if group.startswith('ZY'):
                self.check('metadata_ZY_independence_' + row['record_id'],
                           row['physical_sample_id'] == row['record_id']
                           and row['physical_sample_id_basis'] == 'user_declared_independent_20261031'
                           and not row['batch_id'])
            else:
                if independence is not None:
                    self.check('metadata_new817_declared_independence_' + row['record_id'],
                               group in independence['scope']['groups']
                               and row['physical_sample_id'] == row['record_id']
                               and row['physical_sample_id_basis'] == item.get('independence_basis')
                               == 'user_declared_independent_20261108'
                               and not row['batch_id'] and 'source_batches_per_origin_user_declared' not in item)
                else:
                    self.check('metadata_no_undeclared_independence_or_batch_' + row['record_id'],
                               not row['physical_sample_id'] and not row['physical_sample_id_basis'] and not row['batch_id']
                               and 'independence_basis' not in item and 'source_batches_per_origin_user_declared' not in item)
            # Match the metadata column and header to the actual absorption worksheet.
            if item['path'] not in header_cache:
                with ZipFile(root / item['path']) as z:
                    first = sheet_first_rows(z)['吸收系数']['rows'][0]['cells']
                header_cache[item['path']] = {re.match(r'[A-Z]+', c['cell'])[0]: c['value'] for c in first}
            self.check('metadata_actual_absorption_column_' + row['record_id'],
                       row['sheet'] == '吸收系数' and header_cache[item['path']].get(row['excel_column']) == row['scan_header'])
        summary = root / 'data/confirmed_workbooks_20261108.xlsx'
        book = load_workbook(summary, read_only=False, data_only=False)
        formula_cells = [f'{s.title}!{c.coordinate}' for s in book for row in s for c in row if c.data_type == 'f']
        self.check('summary_has_no_formulas', not formula_cells, formula_cells)
        sheet = book['当前18份工作簿']
        rows = list(sheet.values)
        keys = rows[0]
        summary_rows = [dict(zip(keys, row)) for row in rows[1:]]
        self.check('summary_exact_18_rows', len(summary_rows) == 18)
        for row in summary_rows:
            group = row['group_id']
            self.check('summary_csv_consistency_' + group,
                       all(normalized(value) == normalized(mapped[group][key]) for key, value in row.items()))
        for row in sheet.iter_rows(min_row=2):
            link = row[6].hyperlink
            self.check('summary_hyperlink_' + row[0].value, link is not None
                       and (summary.parent / link.target).resolve() == (root / row[6].value).resolve())
        book.close()
        docs = ('README.md', 'data/CURRENT_DATASET.md', 'AGENTS.md', 'data/processed/zaoren_20261031/README.md')
        all_links = []
        for doc in docs:
            path = root / doc
            text = path.read_text()
            if doc != 'data/processed/zaoren_20261031/README.md':
                self.check('current_document_counts_' + doc, '1672' in text and '18' in text)
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
                target = target.strip().split(' "', 1)[0].strip('<>')
                parsed = urlsplit(target)
                if parsed.scheme or not parsed.path:
                    continue
                resolved = (path.parent / unquote(parsed.path)).resolve()
                all_links.append({'document': doc, 'target': target, 'exists': resolved.exists()})
                self.check('local_document_link_' + doc + '_' + target, resolved.exists())
        current = (root / 'data/CURRENT_DATASET.md').read_text()
        current_rows = [line for line in current.splitlines() if re.match(r'\| (?:ZY|HQ|ZR|TM)\d{2} \|', line)]
        self.check('current_dataset_table_exact_18_rows', len(current_rows) == 18)
        for line in current_rows:
            cells = [x.strip() for x in line.split('|')[1:-1]]
            group = cells[0]
            filename, origin, kind, count = EXPECTED[group]
            self.check('current_dataset_document_mapping_' + group,
                       filename in cells[1] and cells[2] == kind and cells[3].startswith(origin) and int(cells[4]) == count)
        readme = (root / 'README.md').read_text()
        for kind, count in KINDS.items():
            books = sum(item['species'] == kind for item in entries)
            self.check('readme_kind_count_' + kind, f'| {kind} | {books} | {count} |' in readme)
        self.check('git_diff_whitespace', subprocess.run(['git', '-C', str(root), 'diff', '--check'],
                   capture_output=True, text=True).returncode == 0)
        return {'files': self.files, 'document_links_checked': len(all_links),
                'metadata_rows': len(metadata), 'preserved_original_scan_ids': len(old_metadata),
                'preserved_ZY_record_ids': unchanged, 'kind_counts': dict(kind_counts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    validator = Validator(args.repository.resolve())
    details = {}
    try:
        details = validator.run()
    except Exception as exc:
        validator.check('unexpected_validation_exception', False, f'{type(exc).__name__}: {exc}')
    failures = [item for item in validator.checks if not item['passed']]
    # Keep the durable receipt reviewable: aggregate repetitive successful
    # per-record/header checks, while retaining each individual failure above.
    repeated = ('preserve_original_scan_', 'preserve_ZY_identifiers_', 'metadata_labels_',
                'metadata_ZY_independence_', 'metadata_no_undeclared_independence_or_batch_',
                'metadata_actual_absorption_column_')
    summary = {}
    for item in validator.checks:
        name = item['check']
        for prefix in repeated:
            if name.startswith(prefix):
                name = prefix.rstrip('_')
                break
        name = re.sub(r'_\d+_no_unexpected_suffix$', '_all_headers_no_unexpected_suffix', name)
        group = summary.setdefault(name, {'check': name, 'checks_run': 0, 'checks_passed': 0})
        group['checks_run'] += 1
        group['checks_passed'] += int(item['passed'])
    result = {'date': '2026-11-08', 'dataset_version': VERSION,
              'validator': 'scripts/validate_confirmed_dataset.py',
              'baseline_commit': validator.baseline_commit,
              'status': 'passed' if not failures else 'failed',
              'checks_run': len(validator.checks), 'checks_passed': len(validator.checks) - len(failures),
              'failures': failures, 'details': details,
              'scope': 'File/annotation migration integrity only; no new spectral modeling or laboratory verification.',
              'check_summary': list(summary.values())}
    destination = validator.root / 'data/validation/confirmed_names_20261108/validation_result.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'checks_run', 'checks_passed', 'failures')}, ensure_ascii=False))
    print('Validation receipt:', destination)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
