"""Export verified supplementary tables to an ordinary, editable XLSX data file.

This is a mechanical format export of existing numerical tables. It fits no model,
changes no source file and never treats repeated predictions as new specimens.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import numbers
import re
import zipfile

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'data/active_workbooks.json').exists())
BASE = ROOT / 'paper/revision_20261115'
CURRENT = ROOT / 'paper/current'
EVIDENCE = BASE / 'Supplementary_Data_S1'
MANIFEST = json.loads((CURRENT / 'manifest.json').read_text())
ALLOW = {r['path']: r for r in MANIFEST['files']}
FROZEN = {r['path']: r for r in json.loads((BASE / 'FILE_MANIFEST.json').read_text())['files']}
TARGET = BASE / 'Supplementary_Tables.xlsx'
RECEIPT = BASE / 'SUBMISSION_TABLES_VERIFICATION.json'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_source(path: Path) -> tuple[str, str]:
    digest = sha(path)
    if path.is_relative_to(CURRENT):
        relative = str(path.relative_to(CURRENT))
        assert relative in ALLOW, f'Not in current source allowlist: {relative}'
        assert digest == ALLOW[relative]['sha256'], f'Changed current table: {relative}'
        return digest, 'Current corrected-results allowlist'
    relative = str(path.relative_to(BASE))
    assert relative in FROZEN, f'Unregistered derived or molecular table: {relative}'
    assert digest == FROZEN[relative]['sha256'], f'Changed frozen table: {relative}'
    return digest, 'Frozen derived or accepted molecular source'


sources: list[tuple[Path, str, str]] = []


def add(path: Path, name: str, scope: str) -> None:
    assert len(name) <= 31 and name not in {s[1] for s in sources}
    sources.append((path, name, scope))


for path in sorted((CURRENT / 'tables').rglob('*')):
    if path.name.endswith(('.csv', '.csv.gz')):
        suffix = str(path.relative_to(CURRENT / 'tables')).removesuffix('.gz').removesuffix('.csv')
        name = re.sub(r'[^A-Za-z0-9_]', '_', suffix)[:27]
        add(path, f'T{len(sources)+1:02}_{name}', 'Complete corrected result or protocol table, including negative comparisons')

for name in ['dataset_by_species', 'dataset_by_workbook', 'nominal_thickness_by_workbook',
             'task_role_counts', 'record_role_assignment']:
    add(CURRENT / 'methods' / f'{name}.csv', name[:31], 'Recorded collection and specimen allocation; thickness is nominal header annotation')
for name in ['metadata', 'split_manifest']:
    add(CURRENT / 'inputs' / f'{name}.csv', 'Input_' + name, 'Frozen specimen identity, corrected labels and role allocation')
for name, tab in [
    ('feature_primary_replayed', 'Primary_representation'),
    ('feature_budget_all_fixedK_metrics', 'Fixed_feature_budgets'),
    ('feature_budget_paired_bootstrap', 'Paired_budget_intervals'),
    ('feature_modality_composition', 'Retained_modalities'),
    ('feature_subset_overlap_by_refit', 'Feature_subset_overlap'),
    ('adaptation_controls', 'Support_controls'),
    ('B1_learning_curves_current', 'Support_learning_curves'),
    ('B5_acquisition_curves_current', 'Label_acquisition_curves'),
    ('B5_acquisition_metrics_current', 'Label_acquisition_metrics'),
    ('spectra_means_current', 'Source_mean_spectra')]:
    suffix = '.csv.gz' if name == 'B5_acquisition_metrics_current' else '.csv'
    add(EVIDENCE / 'derived_checks' / (name + suffix), tab,
        'Previously verified calculation on recorded specimens and saved predictions; no new training')
for folder, tab in [('acid_d2f61d35', 'Molecular_direct_grid5'), ('acid_f626f77f', 'Molecular_RIJ_grid5')]:
    add(EVIDENCE / 'molecular_sources' / folder / 'ir_modes.csv', tab,
        'Accepted isolated protocatechuic-acid frequencies and independent infrared intensities')

authorship = json.loads((BASE / 'AUTHORSHIP.json').read_text())
first = authorship['authors'][0]
last = authorship['authors'][-1]
header_font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
header_fill = PatternFill('solid', fgColor='404040')
body_font = Font(name='Calibri', size=11, color='000000')


def value_for_excel(value):
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    return str(value)


def cell(sheet, value, header=False):
    result = WriteOnlyCell(sheet, value=value)
    if isinstance(value, str):
        # CSV strings are data, including values that could begin with '='.
        result.data_type = 's'
    result.font = header_font if header else body_font
    if header:
        result.fill = header_fill
        result.alignment = Alignment(vertical='center', wrap_text=True)
    return result


frames = []
records = []
for path, tab, scope in sources:
    digest, verification = verified_source(path)
    frame = pd.read_csv(path, low_memory=False)
    assert frame.shape[0] + 1 <= 1_048_576, tab
    assert frame.shape[1] <= 16_384, tab
    for column in frame.select_dtypes(include=['object', 'string']).columns:
        assert frame[column].dropna().astype(str).str.len().max() <= 32767, (tab, column)
    frames.append(frame)
    records.append({'sheet': tab, 'source': str(path.relative_to(ROOT)), 'sha256': digest,
                    'source_bytes': path.stat().st_size, 'data_rows': len(frame),
                    'columns': len(frame.columns), 'scope': scope,
                    'source_verification': verification})
# Keep common-format supplementary uploads below the journal's preferred size.
# All 51 source tables are retained; the B1--B5 groups are separate attachments.
groups = [('Supplementary_Tables.xlsx', [
    i for i, (path, _, _) in enumerate(sources)
    if not re.search(r'/tables/B[1-5]/', str(path))
])]
groups += [(f'Supplementary_B{b}.xlsx', [
    i for i, (path, _, _) in enumerate(sources) if f'/tables/B{b}/' in str(path)
]) for b in range(1, 6)]
RECEIPT.write_text(json.dumps({'status': 'REBUILDING_SPLIT_FILES'}, indent=2)+'\n')
workbook_records = []
numeric_cells = 0
string_cells = 0

for filename, positions in groups:
    target = BASE / filename
    workbook = Workbook(write_only=True)
    workbook.properties.title = 'Supplementary tables for terahertz medicinal-material discrimination'
    workbook.properties.creator = '; '.join(a['name'] for a in authorship['authors'])
    workbook.properties.description = 'Verified, editable export of existing corrected results, specimen allocations and accepted molecular tables.'
    index = workbook.create_sheet('Source_Index')
    index.sheet_view.showGridLines = False
    index.freeze_panes = 'A9'
    for letter, width in [('A', 30), ('B', 76), ('C', 12), ('D', 10), ('E', 68), ('F', 76)]:
        index.column_dimensions[letter].width = width
    index.append(['Supplementary tables for terahertz medicinal-material discrimination'])
    index.append(['Manuscript', 'Terahertz fingerprints and machine learning for complementary discrimination of medicinal materials'])
    index.append(['First author', first['name'], 'ORCID', first.get('orcid', '')])
    index.append(['Corresponding author', last['name'], 'ORCID', last.get('orcid', '')])
    index.append(['Data scope', '18 workbooks; 1672 specimen records. Four spectral views and repeated predictions do not multiply the physical-specimen count.'])
    index.append(['Interpretation', 'Results preserve all included positive and negative comparisons. Source groups and internal validation do not constitute independent new-batch validation.'])
    index.append(['Representation', 'Values are exported from verified CSV tables. Empty values remain empty; nonfinite numeric values, if any, are literal strings. Original CSV bytes remain the authoritative source.'])
    index.append([cell(index, x, True) for x in ['Sheet', 'Repository source', 'Data rows', 'Columns', 'Source SHA-256', 'Scope']])
    index.row_dimensions[8].height = 28
    for index_row, position in enumerate(positions, 9):
        record = records[position]
        record['workbook'] = filename
        cells = [cell(index, record[k]) for k in ['sheet', 'source', 'data_rows', 'columns', 'sha256', 'scope']]
        for item in cells:
            item.alignment = Alignment(vertical='top', wrap_text=True)
        index.row_dimensions[index_row].height = 30
        index.append(cells)
    index.auto_filter.ref = f'A8:F{8+len(positions)}'

    for position in positions:
        path, tab, scope = sources[position]
        frame = frames[position]
        sheet = workbook.create_sheet(tab)
        sheet.freeze_panes = 'A2'
        sheet.sheet_view.showGridLines = False
        sheet.row_dimensions[1].height = 30
        sheet.auto_filter.ref = f'A1:{get_column_letter(len(frame.columns))}{len(frame)+1}'
        for i, column in enumerate(frame.columns, 1):
            lengths = frame[column].head(150).dropna().astype(str).str.len()
            sample_width = lengths.max() if len(lengths) else 0
            sheet.column_dimensions[get_column_letter(i)].width = min(38, max(12, len(str(column))+2, int(sample_width or 0)+2))
        sheet.append([cell(sheet, str(column), True) for column in frame.columns])
        for row in frame.itertuples(index=False, name=None):
            values = [value_for_excel(v) for v in row]
            # Omit physical empty XML cells; empty values retain their positions.
            sheet.append([None if v is None else cell(sheet, v) for v in values])
    workbook.save(target)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None

    reopened = load_workbook(target, read_only=True, data_only=False)
    assert reopened.sheetnames == ['Source_Index'] + [sources[i][1] for i in positions]
    rows_in_file = 0
    for position in positions:
        frame, record = frames[position], records[position]
        sheet = reopened[record['sheet']]
        iterator = sheet.iter_rows(values_only=True)
        assert tuple(next(iterator)) == tuple(str(x) for x in frame.columns)
        count = 0
        for actual, original in zip(iterator, frame.itertuples(index=False, name=None), strict=True):
            actual = tuple(actual) + (None,) * (len(frame.columns) - len(actual))
            assert len(actual) == len(frame.columns), (record['sheet'], count)
            for saved, source in zip(actual, original):
                expected = value_for_excel(source)
                if isinstance(expected, numbers.Real) and not isinstance(expected, bool):
                    assert isinstance(saved, numbers.Real) and math.isclose(saved, expected, rel_tol=1e-14, abs_tol=1e-15), (record['sheet'], count, saved, expected)
                    numeric_cells += 1
                else:
                    assert saved == expected, (record['sheet'], count, saved, expected)
                    if isinstance(expected, str):
                        string_cells += 1
            count += 1
        assert count == record['data_rows']
        record['readback_data_rows'] = count
        record['all_cells_readback_verified'] = True
        rows_in_file += count
    reopened.close()
    info = {'file': filename, 'bytes': target.stat().st_size, 'sha256': sha(target),
            'sheets': len(positions)+1, 'data_sheets': len(positions),
            'data_rows': rows_in_file, 'zip_crc': 'PASS',
            'all_rows_and_cells_readback_verified': True,
            'file_below_10_MiB': target.stat().st_size < 10*1024*1024}
    workbook_records.append(info)
    print(json.dumps(info), flush=True)

for (path, _, _), record in zip(sources, records):
    assert sha(path) == record['sha256'], path

receipt = {'schema': 'research4-submission-tables-verification-v2', 'status': 'PASS',
           'file': TARGET.name, 'bytes': TARGET.stat().st_size, 'sha256': sha(TARGET),
           'workbooks': workbook_records,
           'total_sheets': sum(w['sheets'] for w in workbook_records),
           'data_sheets': len(sources), 'total_data_rows': sum(r['data_rows'] for r in records),
           'numeric_cells_readback_checked': numeric_cells,
           'string_cells_readback_checked': string_cells,
           'zip_crc': 'PASS', 'all_source_hashes_verified': True,
           'all_rows_and_cells_readback_verified': True,
           'numeric_readback_tolerance': {'relative': 1e-14, 'absolute': 1e-15},
           'all_source_bytes_unchanged': True, 'new_model_training': False,
           'new_quantum_calculations': False,
           'source_manifest_sha256': sha(CURRENT/'manifest.json'),
           'all_files_below_10_MiB': all(w['file_below_10_MiB'] for w in workbook_records),
           'files': records}
RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k: v for k, v in receipt.items() if k not in ['files', 'workbooks']}, ensure_ascii=False))
