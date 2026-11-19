"""Publish origin labels and a material overview from frozen workbooks."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from manuscript_content import TITLE
from publication_metadata import BASE, ROOT, ORIGINS_EN, PROVENANCE

PREVIOUS = ROOT / 'paper/revision_20261116'
SOURCE = PREVIOUS / 'Supplementary_Tables.xlsx'
TARGET = BASE / 'Supplementary_Tables.xlsx'
METHODS = ROOT / 'paper/current/methods'
LABELS = PROVENANCE['origin_display_updates']
ORIGIN_COLUMNS = {
    'dataset_by_workbook': 3,
    'nominal_thickness_by_workbook': 3,
    'record_role_assignment': 6,
    'Input_metadata': 7,
    'Input_split_manifest': 6,
    'LOWO_OOF_predictions': 11,
}
MATERIALS = {'酸枣仁': 'Suanzaoren', '枣仁': 'Zaoren', '黄芪': 'Huangqi', '天麻': 'Tianma'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(name):
    with (METHODS / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def add_material_overview(book):
    sheet = book.create_sheet('Material_and_Origin', 1)
    sheet.append(['Material', '材料', 'Group', 'Origin', '产地', 'Total', 'TRAIN', 'CAL', 'PRED', '1.5 mm', '1 mm'])
    rows = read_rows('dataset_by_workbook.csv')
    thickness = {row['origin_group_id']: row for row in read_rows('nominal_thickness_by_workbook.csv')}
    for material, english in MATERIALS.items():
        for row in rows:
            if row['species'] != material:
                continue
            group = row['origin_group_id']
            origin_zh = LABELS.get(group, {}).get('publication_zh', row['origin'])
            sizes = [int(row[k]) for k in ('records', 'train', 'calibration', 'prediction')]
            recorded = [int(thickness[group][k]) for k in ('thickness_1_5mm', 'thickness_1mm')]
            assert sizes[0] == sum(sizes[1:]) == sum(recorded)
            sheet.append([english, material, group, ORIGINS_EN[group], origin_zh, *sizes, *recorded])
    assert sheet.max_row == 19
    sheet.freeze_panes = 'F2'
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='234A61')
        cell.alignment = Alignment(horizontal='center')
    for index, width in enumerate([15, 12, 10, 28, 16, 10, 10, 10, 10, 12, 10], 1):
        sheet.column_dimensions[get_column_letter(index)].width = width


book = load_workbook(SOURCE)
assert len(book.sheetnames) == 43 and book['Source_Index']['A2'].value == 'Manuscript'
changed = {}
index = book['Source_Index']
if index['B2'].value != TITLE:
    changed[('Source_Index', 'B2')] = (index['B2'].value, TITLE)
    index['B2'] = TITLE
alias_counts = Counter()
for sheet_name, column in ORIGIN_COLUMNS.items():
    sheet = book[sheet_name]
    assert sheet.cell(1, column).value == 'origin', (sheet_name, sheet.cell(1, column).value)
    aliases = {alias['recorded_zh']: alias['publication_zh'] for alias in LABELS.values()}
    for row in range(2, sheet.max_row + 1):
        cell = sheet.cell(row, column)
        if cell.value in aliases:
            original = cell.value
            changed[(sheet_name, cell.coordinate)] = (original, aliases[original])
            cell.value = aliases[original]
            alias_counts[original] += 1
assert alias_counts['山西'] == 502 and alias_counts['安徽毫州'] > 0, alias_counts
add_material_overview(book)
book.save(TARGET)

before = load_workbook(SOURCE, read_only=True, data_only=False)
after = load_workbook(TARGET, read_only=True, data_only=False)
assert [name for name in after.sheetnames if name != 'Material_and_Origin'] == before.sheetnames
observed = {}
for name in before.sheetnames:
    assert before[name].max_row == after[name].max_row
    assert before[name].max_column == after[name].max_column
    for old_row, new_row in zip(before[name].iter_rows(), after[name].iter_rows()):
        for old, new in zip(old_row, new_row):
            if old.value != new.value:
                observed[(name, old.coordinate)] = (old.value, new.value)
assert observed == changed
assert [sum(after['Material_and_Origin'].cell(row, col).value for row in range(2, 20))
        for col in range(6, 12)] == [1672, 999, 334, 339, 1670, 2]

receipt = json.loads((PREVIOUS/'SUPPLEMENTARY_TABLES_VERIFICATION.json').read_text())
receipt.update(
    status='PASS',
    workbook_sha256=sha(TARGET),
    previous_revision_workbook_sha256=sha(SOURCE),
    title_only_update=False,
    prior_numeric_cells_and_ids_unchanged=True,
    original_cells_updated=len(changed),
    origin_label_updates=dict(alias_counts),
    added_sheet='Material_and_Origin',
    changed_cells=[{'sheet': name, 'cell': cell, 'from': old, 'to': new}
                   for (name, cell), (old, new) in changed.items()],
    sheets=after.sheetnames,
    scientific_models_trained=0,
)
(BASE/'SUPPLEMENTARY_TABLES_VERIFICATION.json').write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
print(f'PASS: {len(after.sheetnames)} sheets, {dict(alias_counts)} origin display updates, source values and IDs preserved.')
