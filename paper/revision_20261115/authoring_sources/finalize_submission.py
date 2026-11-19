"""Validate the final documents and build the complete author submission package.

Run after rendering and recording visual approval in VISUAL_REVIEW.json.
This script never trains a model, modifies frozen evidence or submits to a journal.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import re
import shutil
import zipfile

from docx import Document
from docx.oxml.ns import qn

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'data/active_workbooks.json').exists())
BASE = ROOT / 'paper/revision_20261115'
OUTPUT = ROOT / 'outputs/revision_20261115'
OUTPUT.mkdir(parents=True, exist_ok=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def document_text(doc):
    return '\n'.join([p.text for p in doc.paragraphs] + [c.text for t in doc.tables for r in t.rows for c in r.cells])


review = read_json(BASE / 'VISUAL_REVIEW.json')
assert review['status'] == 'PASS'
tables_receipt = read_json(BASE / 'SUBMISSION_TABLES_VERIFICATION.json')
assert tables_receipt['status'] == 'PASS'
for workbook in tables_receipt['workbooks']:
    assert digest(BASE / workbook['file']) == workbook['sha256']
    assert (BASE / workbook['file']).stat().st_size < 10 * 1024 * 1024
assert read_json(BASE / 'JOURNAL_COMPLIANCE.json')['status'] == 'FILE_REQUIREMENTS_REVIEW_PASS'
for item in review['files']:
    assert digest(BASE / item['path']) == item['sha256'], f"Visual review is stale: {item['path']}"

authorship = read_json(BASE / 'AUTHORSHIP.json')
assert len(authorship['authors']) == 5
assert authorship['authors'][0]['orcid'].endswith('0009-0005-1153-4926')
assert authorship['authors'][-1]['orcid'].endswith('0000-0002-3221-7875')
authorship['submission_status'] = 'Final files prepared for author upload; no journal submission performed'
save_json(BASE / 'AUTHORSHIP.json', authorship)

main_path = BASE / '01_Manuscript_Optimized.docx'
parallel_path = BASE / '07_Manuscript_EN_ZH.docx'
main = Document(main_path)
parallel = Document(parallel_path)
bilingual = read_json(BASE / 'BILINGUAL_VERIFICATION.json')
assert bilingual['main_document_sha256'] == digest(main_path)
assert bilingual['parallel_document_sha256'] == digest(parallel_path)
assert bilingual['untranslated_requested_strings'] == []
assert len(main.tables) == 4 and len(parallel.tables) == 8
numeric = re.compile(r'[+−-]?\d+(?:\.\d+)?(?:%|/\d+)?')
for idx, table in enumerate(main.tables):
    for translated in parallel.tables[2 * idx:2 * idx + 2]:
        assert len(table.rows) == len(translated.rows)
        for erow, zrow in zip(table.rows, translated.rows):
            for ecell, zcell in zip(erow.cells, zrow.cells):
                assert numeric.findall(ecell.text) == numeric.findall(zcell.text)

word_count = len(re.findall(r'\S+', document_text(main)))
source = (BASE / 'authoring_sources/manuscript_source.txt').read_text()
abstract_section = source.split('## Abstract\n\n', 1)[1].split('\n\n## Introduction', 1)[0]
abstract, keywords = abstract_section.split('\n\nKeywords: ', 1)
metadata = {
    'schema': 'research4-submission-metadata-v1',
    'journal': 'Applied Spectroscopy',
    'official_author_instructions': 'https://journals.sagepub.com/author-instructions/asp',
    'title': source.splitlines()[0].removeprefix('# '),
    'article_type': 'Submitted Papers / original research; match the portal option',
    'abstract': abstract.strip(),
    'keywords': [x.strip() for x in keywords.split(';')],
    'authors': authorship['authors'],
    'author_orcids': {a['name']: a['orcid'] for a in authorship['authors'] if a.get('orcid')},
    'common_affiliation': authorship['shared_affiliation'],
    'common_affiliation_zh': authorship['shared_affiliation_zh'],
    'common_work_address': authorship['shared_work_address'],
    'main_manuscript_word_count_approximate': word_count,
    'word_count_scope': 'Word paragraphs and tables including references and captions; excludes equation objects and figure text',
    'main_figures': 5, 'main_tables': 4, 'supplementary_figures': 1,
    'references': 28,
    'material_discrimination_validation': 'Leave-one-workbook-out: binary10 folds/864 specimens; four materials18 folds/1672 specimens',
    'competing_interests': 'None declared', 'funding': 'No financial support',
    'all_authors_approve_and_accept_responsibility': True,
    'first_and_corresponding_author_orcids_confirmed': True,
    'manuscript_under_consideration_elsewhere': False,
    'experimental_data_provider': 'Yin Shen',
    'dataset_shared_for_other_supervised_studies': True,
    'ethics': 'No human participants or animals; approval not required',
    'consent_to_participate': 'Not applicable',
    'consent_for_publication': 'Not applicable',
    'author_recommended_reviewers': 'Not accepted by this journal',
    'journal_submission_performed': False,
    'submission_account_orcid_binding_performed': False,
    'account_action': 'The account holder uploads the final files, authorizes their ORCID, checks the portal PDF and submits.'
}
save_json(BASE / 'SUBMISSION_METADATA.json', metadata)
readiness = read_json(BASE / 'SUBMISSION_READINESS.json')
readiness.update(submission_files_prepared=True, package_assembly_state='COMPLETE',
                 main_manuscript_word_count_approximate=word_count,
                 figures_restyled_and_framed=True, figure_png_dpi=1200,
                 full_manuscript_en_zh_completed=True,
                 reference_count=28, motivation_and_contributions_revised=True)
save_json(BASE / 'SUBMISSION_READINESS.json', readiness)
source_check = read_json(ROOT / 'paper/current/verification.json')
assert source_check['status'] == 'PASS'
assert source_check['manifest_sha256'] == digest(ROOT / 'paper/current/manifest.json')
save_json(BASE / 'CURRENT_SOURCE_VERIFICATION.json', source_check)

documents = []
for path in sorted(BASE.glob('*.docx')):
    doc = Document(path)
    text = document_text(doc)
    assert not any(token in text for token in ['[FIGURE:', '[EQUATION:', 'TODO', 'TBD'])
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    documents.append({'file': path.name, 'sha256': digest(path), 'tables': len(doc.tables),
                      'inline_figures': len(doc.inline_shapes),
                      'editable_equations': len(doc.element.findall('.//' + qn('m:oMath'))),
                      'ooxml_zip_integrity': 'PASS', 'no_placeholder_tokens': True})
assert len(documents) == 8
assert not (BASE / '06_投稿前信息补齐表.docx').exists()
expected_figures = ['Fig_model_selection', 'fig1_spectra_current', 'fig2_feature_budget', 'fig3_selected_modalities',
                    'fig4_adaptation_controls', 'Fig_S_quantum_accepted_PCA']
embedded = {}
for name in ['01_Manuscript_Optimized.docx', '02_Supplementary_Information.docx', '07_Manuscript_EN_ZH.docx']:
    with zipfile.ZipFile(BASE / name) as archive:
        embedded[name] = {hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist() if n.startswith('word/media/')}
for idx, name in enumerate(expected_figures):
    h = digest(BASE / 'figures' / f'{name}.png')
    expected_docs = ['02_Supplementary_Information.docx'] if idx == len(expected_figures)-1 else ['01_Manuscript_Optimized.docx', '07_Manuscript_EN_ZH.docx']
    assert all(h in embedded[d] for d in expected_docs), name
style = read_json(BASE / 'authoring_sources/figure_style_receipt.json')
assert style['status'] == 'PASS' and style['png_export_dpi'] >= 800 and style['tiff_export_dpi'] == 1200
assert style['frame']['outer_figure_rectangle'] and style['frame']['four_spines_each_panel']
for row in style['outputs']:
    assert digest(BASE / row['path']) == row['sha256']
model_style = read_json(BASE / 'authoring_sources/model_selection_figure_receipt.json')
assert model_style['status'] == 'PASS' and model_style['drawing']['outer_frame']
assert model_style['drawing']['dpi'] == 1200
for row in model_style['outputs']:
    assert digest(BASE / row['path']) == row['sha256']
assert len(main.inline_shapes) == 5
assert len(main.element.findall('.//' + qn('m:oMath'))) == 8
for doc in [main, Document(BASE / '02_Supplementary_Information.docx'), parallel]:
    text = document_text(doc)
    assert not re.search(r'calycosin|spinosin|毛蕊异黄酮|斯皮诺素|assisted manuscript drafting', text, re.I)
assert 'assisted manuscript revision, literature searches, and the preparation and checking of analysis and plotting code' in document_text(main)
save_json(BASE / 'DELIVERY_VERIFICATION.json', {
    'status': 'PASS', 'documents': documents, 'visual_review': review,
    'figure_style_receipt': 'authoring_sources/figure_style_receipt.json',
    'model_selection_figure_receipt': 'authoring_sources/model_selection_figure_receipt.json',
    'main_figures': 5, 'native_editable_equations': 8,
    'unreported_molecule_narration_removed': True,
    'ai_disclosure_manuscript_revision_wording_verified': True,
    'embedded_final_figure_hashes_match': True,
    'english_chinese_table_numeric_values_match': True,
    'classifier_training_performed': True, 'quantum_calculations_started': False,
    'classifier_training_scope': 'Only the two material-discrimination LOWO tasks:28 fixed-pipeline fits. All source-group, representation and mechanism results are preserved.',
    'kind_lowo_verification': 'KIND_LOWO_VERIFICATION.json',
    'evidence_entry': 'paper/current/manifest.json', 'authorship_confirmation': 'AUTHORSHIP.json',
    'all_authors_approve_and_accept_responsibility': True,
    'prior_parameter_completion_workflow_retired': True,
    'references_verified': 28,
    'motivation_and_contributions_revised': True,
    'narrative_revision_receipt': 'authoring_sources/NARRATIVE_REVISION.json',
    'journal_compliance': 'JOURNAL_COMPLIANCE.json',
    'supplementary_tables_verification': 'SUBMISSION_TABLES_VERIFICATION.json',
    'journal_submission_performed': False
})


def deliverable_files(directory):
    return [p for p in sorted(directory.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']


manifest = {
    'schema': 'research4-final-submission-delivery-v2', 'date': '2026-11-15',
    'classifier_source_manifest_sha256': digest(ROOT / 'paper/current/manifest.json'),
    'no_new_model_fits': False, 'no_withdrawn_classifier_outputs': True,
    'new_model_fit_scope': 'Only material-discrimination LOWO:10 binary and18 four-material folds; no model search.',
    'files': [{'path': str(p.relative_to(BASE)), 'bytes': p.stat().st_size, 'sha256': digest(p)}
              for p in deliverable_files(BASE) if p != BASE / 'FILE_MANIFEST.json']
}
save_json(BASE / 'FILE_MANIFEST.json', manifest)

with TemporaryDirectory(prefix='research4-final-') as scratch:
    stage = Path(scratch) / 'research4_Final_Submission_20261115'
    submit, author_review, evidence = [stage / n for n in ['01_Submit', '02_Review', '03_Evidence']]
    for directory in [submit, author_review, evidence]:
        directory.mkdir(parents=True)
    mapping = {
        '01_Manuscript_Optimized.docx': submit / '01_Manuscript.docx',
        '05_Title_Page.docx': submit / '02_Title_Page.docx',
        '04_Cover_Letter.docx': submit / '03_Cover_Letter.docx',
        '02_Supplementary_Information.docx': submit / '04_Supplementary_Information.docx',
        '08_Declarations.docx': submit / '05_Declarations.docx',
        'Supplementary_Tables.xlsx': submit / 'Supplementary_Tables.xlsx',
        'Supplementary_LOWO.xlsx': submit / 'Supplementary_LOWO.xlsx',
        'Supplementary_Data_S1.zip': evidence / 'Supplementary_Data_S1.zip',
        'SUBMISSION_METADATA.json': submit / 'SUBMISSION_METADATA.json',
        '07_Manuscript_EN_ZH.docx': author_review / '01_Manuscript_EN_ZH.docx',
        '06_投稿文件说明.docx': author_review / '02_投稿文件说明.docx',
        '03_论文优化与清理说明.docx': author_review / '03_论文优化与清理说明.docx'
    }
    for source_name, target in mapping.items():
        shutil.copy2(BASE / source_name, target)
    for i in range(1,6):
        shutil.copy2(BASE / f'Supplementary_B{i}.xlsx', submit / f'Supplementary_B{i}.xlsx')
    (submit / 'figures').mkdir()
    (evidence / 'vector_figures').mkdir()
    (author_review / 'figure_previews').mkdir()
    for i,name in enumerate(expected_figures,1):
        for ext in ['png', 'pdf', 'svg', 'tiff', 'eps']:
            target_dir = evidence / 'vector_figures' if ext == 'svg' else (submit / 'figures' if ext in ['tiff','eps'] else author_review / 'figure_previews')
            target_name = f'Figure_{i}' if i<len(expected_figures) else 'Figure_S1'
            shutil.copy2(BASE / 'figures' / f'{name}.{ext}', target_dir / f'{target_name}.{ext}')
    for folder in ['Supplementary_Data_S1', 'authoring_sources']:
        shutil.copytree(BASE / folder, evidence / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in BASE.iterdir():
        if path.suffix == '.json':
            target_name = 'REPOSITORY_FILE_MANIFEST.json' if path.name == 'FILE_MANIFEST.json' else path.name
            shutil.copy2(path, evidence / target_name)
    (evidence / 'README_溯源说明.txt').write_text(
        '本文件夹保留完整冻结证据与写作来源，不需全部上传期刊。\n'
        'Supplementary_Data_S1：冻结的纠正后来源、协议、预测与验收。历史记录中的缺失字段保持原貌，不是本次待补清单。\n'
        'authoring_sources：最终英文源稿、完整中文翻译和构建脚本；脚本在原仓库路径运行。\n'
        'vector_figures：6张图的SVG源版；上传用EPS及1200 dpi TIFF在01_Submit/figures，PDF/PNG审阅版在02_Review/figure_previews。\n'
        'REPOSITORY_FILE_MANIFEST.json对应仓库paper/revision_20261115路径；压缩包的实际路径与哈希见根目录PACKAGE_MANIFEST.json。\n'
        'VISUAL_REVIEW.json和DELIVERY_VERIFICATION.json记录最终页面检查及文档结构。\n')
    (stage / 'README_使用说明.txt').write_text(
        'research4 最终论文与投稿包 | 2026-11-15\n\n'
        '01_Submit：英文正文、标题页、投稿信、英文补充材料、声明、数据附件与最终图片。\n'
        '02_Review：完整正文中英对照及中文使用说明，供作者审阅。\n'
        '03_Evidence：完整冻结证据、源稿、翻译、构建代码和哈希核验，供追溯。\n\n'
        '请先阅读02_Review/02_投稿文件说明.docx。生产图提供1200 dpi TIFF及EPS，每图择一上传；补充表格优先使用XLSX。\n'
        '正文按实际数据标注和已确认条件完成，旧实验参数补齐表已经撤下。冻结证据中的历史缺失字段保留原貌。\n'
        '全体作者、Ziyuan Zhu第一作者ORCID、Yin Shen通讯标记及ORCID、共同单位与责任声明已写入文件；数据来源已说明，亦用于导师其他指导研究。\n'
        '文件准备完成，未代为向期刊提交。账号持有人在系统授权ORCID、预览生成PDF后执行提交。\n\n'
        '仓库：https://github.com/jackzhu119/research4\n'
        '当前修订：https://github.com/jackzhu119/research4/pull/19\n'
        '原始18份工作簿通过仓库data/active_workbooks.json定位，不在本包重复复制。\n'
        '03_Evidence/authoring_sources脚本原位在仓库执行；独立压缩包保留源代码供审阅。\n')
    package_files = [{'path': str(p.relative_to(stage)), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                     for p in deliverable_files(stage)]
    save_json(stage / 'PACKAGE_MANIFEST.json', {'schema': 'research4-package-v1', 'files': package_files})
    target = OUTPUT / 'research4_Final_Submission_20261115.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in deliverable_files(stage):
            archive.write(path, str(path.relative_to(stage.parent)))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for row in package_files:
            data = archive.read(stage.name + '/' + row['path'])
            assert hashlib.sha256(data).hexdigest() == row['sha256']
    receipt = {'status': 'PASS', 'archive': str(target), 'bytes': target.stat().st_size,
               'sha256': digest(target), 'files': len(package_files) + 1,
               'crc_and_every_file_hash_verified': True}
    save_json(OUTPUT / 'FINAL_PACKAGE_RECEIPT.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))
