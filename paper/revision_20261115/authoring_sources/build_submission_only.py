"""Build the verified journal-upload-only archive after finalize_submission.py.

The archive contains journal files, one production version of each figure and
brief upload instructions. Author-review documents, metadata, code and evidence
archives stay outside this ZIP. No model is fitted and no journal is contacted.

ZIP metadata and the verification receipt are deterministic. This permits a
second finalizer/builder pass to include the receipt in the canonical delivery
manifest without introducing a checksum cycle.
"""
from pathlib import Path
import hashlib
import json
import re
import zipfile

from docx import Document
from docx.oxml.ns import qn


ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / 'data/active_workbooks.json').exists()
)
BASE = ROOT / 'paper/revision_20261115'
OUTPUT = ROOT / 'outputs/revision_20261115'
ARCHIVE_NAME = 'research4_Upload_20261115.zip'
PACKAGE_ROOT = 'Journal_Submission'
RECEIPT_NAME = 'SUBMISSION_ONLY_VERIFICATION.json'
ZIP_DATE = (2026, 11, 15, 0, 0, 0)

DOCUMENTS = {
    '01_Manuscript_Optimized.docx': '01_Manuscript.docx',
    '05_Title_Page.docx': '02_Title_Page.docx',
    '04_Cover_Letter.docx': '03_Cover_Letter.docx',
    '02_Supplementary_Information.docx': '04_Supplementary_Information.docx',
    '08_Declarations.docx': 'Optional/Declarations.docx',
}
MAIN_FIGURES = [
    'Fig_model_selection',
    'fig1_spectra_current',
    'fig2_feature_budget',
    'fig3_selected_modalities',
    'fig4_adaptation_controls',
]
README = """APPLIED SPECTROSCOPY JOURNAL SUBMISSION FILES

Extract this archive and upload its individual files in the journal system.
Do not upload this ZIP as the main manuscript.

01_Manuscript.docx: Manuscript or Main document.
02_Title_Page.docx: Title page.
03_Cover_Letter.docx: Cover letter.
04_Supplementary_Information.docx: Supplemental material.
Supplementary_Tables.xlsx and Supplementary_B1.xlsx through
Supplementary_B5.xlsx, plus Supplementary_LOWO.xlsx: Supplemental data;
each workbook is below 10 MiB.
Figures/Figure_1.eps through Figures/Figure_5.eps: Five main figures.
Supplementary_Figure/Figure_S1.eps: Supplementary figure S1.
Optional/Declarations.docx: Use only if the system asks for a separate
declaration file. The relevant declarations are already in the manuscript.

The EPS files are the production figure versions. Do not register additional
copies of the same figure under different file formats. Use the journal's
actual file-type options and check its generated PDF before final submission.

The package includes named ORCIDs for first author Ziyuan Zhu and corresponding
author Yin Shen. The submitting author must authorize their own ORCID in their
journal account. No journal submission has been performed by this package.
"""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def document_text(document):
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    cells = [
        cell.text
        for table in document.tables
        for row in table.rows
        for cell in row.cells
    ]
    return '\n'.join(paragraphs + cells)


def source_mapping():
    mapping = dict(DOCUMENTS)
    mapping['Supplementary_Tables.xlsx'] = 'Supplementary_Tables.xlsx'
    mapping['Supplementary_LOWO.xlsx'] = 'Supplementary_LOWO.xlsx'
    for number in range(1, 6):
        name = f'Supplementary_B{number}.xlsx'
        mapping[name] = name
    for number, name in enumerate(MAIN_FIGURES, 1):
        mapping[f'figures/{name}.eps'] = f'Figures/Figure_{number}.eps'
    mapping['figures/Fig_S_quantum_accepted_PCA.eps'] = (
        'Supplementary_Figure/Figure_S1.eps'
    )
    require(len(mapping) == 18, 'Expected 18 submission files plus README.')
    return mapping


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = read_json(BASE / 'FILE_MANIFEST.json')
    approved = {row['path']: row for row in manifest['files']}
    delivery = read_json(BASE / 'DELIVERY_VERIFICATION.json')
    visual = read_json(BASE / 'VISUAL_REVIEW.json')
    workbooks = read_json(BASE / 'SUBMISSION_TABLES_VERIFICATION.json')
    require(delivery['status'] == 'PASS', 'Run the final delivery verifier first.')
    require(visual['status'] == 'PASS', 'Final visual review is not approved.')
    require(workbooks['status'] == 'PASS', 'Supplementary workbook review failed.')
    reviewed = {row['path']: row['sha256'] for row in visual['files']}
    verified_books = {row['file']: row for row in workbooks['workbooks']}

    payloads = {}
    entries = []
    word_checks = []
    forbidden = re.compile(
        r'\b(?:calycosin|spinosin|drafting)\b|毛蕊异黄酮|斯皮诺素', re.I
    )
    for source_name, target_name in source_mapping().items():
        path = BASE / source_name
        require(path.is_file(), f'Missing submission source: {source_name}')
        data = path.read_bytes()
        digest = sha256(data)
        require(source_name in approved, f'Unmanifested source: {source_name}')
        row = approved[source_name]
        require(
            row['sha256'] == digest and row['bytes'] == len(data),
            f'Stale source hash for {source_name}; rerun finalize_submission.py.',
        )

        if path.suffix == '.docx':
            require(reviewed.get(source_name) == digest,
                    f'Stale document visual review: {source_name}')
            with zipfile.ZipFile(path) as source_zip:
                require(source_zip.testzip() is None,
                        f'Invalid Word archive: {source_name}')
            document = Document(path)
            text = document_text(document)
            require(not forbidden.search(text),
                    f'Excluded molecule or drafting wording in {source_name}')
            require('0009-0005-1153-4926' in text,
                    f'Missing first-author ORCID: {source_name}')
            require('0000-0002-3221-7875' in text,
                    f'Missing corresponding-author ORCID: {source_name}')
            word_checks.append({
                'source': source_name,
                'excluded_molecule_and_drafting_scan': 'PASS',
                'both_author_orcids_present': True,
                'tables': len(document.tables),
                'inline_figures': len(document.inline_shapes),
                'native_equations': len(document.element.findall('.//' + qn('m:oMath'))),
            })
            if source_name == '01_Manuscript_Optimized.docx':
                require(len(document.tables) == 4, 'Main manuscript must have four tables.')
                require(len(document.inline_shapes) == 5, 'Main manuscript must have five figures.')
                require(len(document.element.findall('.//' + qn('m:oMath'))) == 8,
                        'Main manuscript must have eight native equations.')
                require('assisted manuscript revision' in text,
                        'Main manuscript must use the confirmed revision disclosure.')
        elif path.suffix == '.xlsx':
            require(path.name in verified_books, f'Unverified workbook: {path.name}')
            require(verified_books[path.name]['sha256'] == digest,
                    f'Stale workbook verification: {path.name}')
            require(len(data) < 10 * 1024 * 1024,
                    f'Workbook exceeds the intended 10 MiB limit: {path.name}')
            with zipfile.ZipFile(path) as source_zip:
                require(source_zip.testzip() is None,
                        f'Invalid workbook archive: {source_name}')
        elif path.suffix == '.eps':
            require(data.startswith(b'%!PS-Adobe-3.0 EPSF-3.0'),
                    f'Invalid EPS header: {source_name}')
            require(b'%%BoundingBox:' in data, f'Missing EPS bounds: {source_name}')
        else:
            raise RuntimeError(f'Unexpected submission file type: {source_name}')

        payloads[target_name] = data
        entries.append({'path': target_name, 'source': source_name,
                        'bytes': len(data), 'sha256': digest})

    readme_data = README.encode('utf-8')
    payloads['README.txt'] = readme_data
    entries.append({'path': 'README.txt', 'source': 'embedded upload instructions',
                    'bytes': len(readme_data), 'sha256': sha256(readme_data)})
    require(len(payloads) == 19, 'Expected exactly 19 archive entries.')
    target = OUTPUT / ARCHIVE_NAME
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(f'{PACKAGE_ROOT}/{name}', date_time=ZIP_DATE)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED,
                            compresslevel=9)
    with zipfile.ZipFile(target) as archive:
        require(archive.testzip() is None, 'Submission ZIP CRC validation failed.')
        require(len(archive.namelist()) == 19, 'Unexpected ZIP entry count.')
        require(all(not name.endswith('/') for name in archive.namelist()),
                'Unexpected directory entries in ZIP.')
        for row in entries:
            data = archive.read(f"{PACKAGE_ROOT}/{row['path']}")
            require(len(data) == row['bytes'] and sha256(data) == row['sha256'],
                    f"Archive member verification failed: {row['path']}")

    receipt = {
        'schema': 'research4-submission-only-verification-v1',
        'status': 'PASS', 'file': ARCHIVE_NAME,
        'bytes': target.stat().st_size, 'sha256': sha256(target.read_bytes()),
        'archive_entries': 19, 'package_root': PACKAGE_ROOT,
        'main_figures': 5, 'supplementary_figures': 1,
        'supplementary_workbooks': 7, 'english_journal_documents': 5,
        'declarations_are_optional': True,
        'main_native_equations': 8, 'main_editable_tables': 4,
        'all_source_hashes_match_canonical_manifest': True,
        'all_entry_hashes_and_crc_verified': True,
        'no_review_documents_code_logs_or_metadata_in_zip': True,
        'deterministic_zip_metadata': True,
        'journal_submission_performed': False,
        'document_checks': word_checks,
        'entries': sorted(entries, key=lambda row: row['path']),
    }
    serialized = json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'
    (BASE / RECEIPT_NAME).write_text(serialized, encoding='utf-8')
    (OUTPUT / RECEIPT_NAME).write_text(serialized, encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'archive': str(target),
                      'bytes': receipt['bytes'], 'sha256': receipt['sha256'],
                      'entries': receipt['archive_entries']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
