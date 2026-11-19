"""Package verified journal files and separate author-reading material."""
from pathlib import Path
import json,hashlib,zipfile,shutil,subprocess
from docx import Document
from publication_metadata import AI_DISCLOSURE, AFFILIATIONS, AUTHORS, CORRESPONDING, PENDING_CONFLICTS, PROVENANCE_PATH

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'paper/revision_20261116_r2';OLD=ROOT/'paper/revision_20261115'
OUT=ROOT/'outputs/revision_20261116_r2';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
visual=read(BASE/'VISUAL_REVIEW.json');assert visual['status']=='PASS'
for row in visual['files']:assert sha(BASE/row['path'])==row['sha256'],row['path']
artifact=read(BASE/'ARTIFACT_VERIFICATION.json');assert artifact['status']=='PASS'
for row in artifact['files']:assert sha(BASE/row['path'])==row['sha256']
tables=read(BASE/'SUPPLEMENTARY_TABLES_VERIFICATION.json');assert tables['status']=='PASS'
assert sha(BASE/'Supplementary_Tables.xlsx')==tables['workbook_sha256']
for row in tables['source_files'].values():assert sha(ROOT/row['path'])==row['sha256']
evidence=read(BASE/'EVIDENCE_MANIFEST.json');assert sha(ROOT/'paper/current/manifest.json')==evidence['current_manifest_sha256']
for row in evidence['analysis_files']:assert sha(BASE/row['path'])==row['sha256']
audit=read(BASE/'analysis/EVIDENCE_AUDIT.json');assert audit['status']=='PASS' and audit['scientific_models_trained_in_this_revision']==0
current=read(ROOT/'paper/current/verification.json');assert current['status']=='PASS' and current['checked_files']==182
for row in evidence['current_files']:assert sha(ROOT/'paper/current'/row['path'])==row['sha256']
for row in evidence['unchanged_B_workbooks']:
 assert sha(OLD/row['file'])==row['sha256'];assert row['bytes']<10*1024**2
for row in read(BASE/'figures/FIGURE_VERIFICATION.json')['outputs']:
 assert sha(BASE/'figures'/row['file'])==row['sha256']
assert read(BASE/'FIGURE_QUALITY_REVIEW.json')['status']=='PASS'
protected=['paper/revision_20261115/Supplementary_Data_S1','data/active_workbooks.json','data/raw']
assert not subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only','HEAD','--',*protected],text=True).strip()
compliance=read(BASE/'JOURNAL_COMPLIANCE.json');compliance['status']='AUTHOR_DECLARATION_PENDING' if PENDING_CONFLICTS else 'FILE_REQUIREMENTS_AND_VISUAL_REVIEW_PASS'
compliance['checked_date']='2026-11-19'
compliance.pop('author_AI_declaration_baseline_unchanged',None)
compliance['author_metadata_updated']=True
compliance['AI_disclosure_location']='Acknowledgements subsection at the end of the manuscript'
compliance.pop('AI_disclosure_retained',None)
compliance['author_declarations_complete']=not PENDING_CONFLICTS
compliance['Sage_AI_policy']='https://www.sagepub.com/journals/publication-ethics-policies/artificial-intelligence-policy'
compliance['conflict_disclosure_policy']='https://www.sagepub.com/journals/publication-ethics-policies/declaration-of-conflicting-interests'
compliance['main_manuscript_words_approximate']=artifact['main_word_count_approximate'];compliance['all_document_pages_visually_reviewed']=sum(row['pages'] for row in visual['files'])
write(BASE/'JOURNAL_COMPLIANCE.json',compliance)
meta=read(BASE/'DOCUMENT_BUILD.json')
title_text='\n'.join(p.text for p in Document(BASE/'02_Title_Page.docx').paragraphs)
for author in AUTHORS:
 assert author['name'] in title_text
 if author.get('email'):assert author['email'] in title_text
 if author.get('orcid'):assert author['orcid'].split('/')[-1] in title_text
fields={'journal':'Applied Spectroscopy','article_type':'Submitted Papers / original research; match the portal label','title':meta['title'],
 'abstract':meta['abstract'],'unstructured_abstract_words':meta['abstract_words'],'keywords':meta['keywords'],'authors':AUTHORS,
 'affiliations':AFFILIATIONS,
 'corresponding_author':CORRESPONDING['name'],'corresponding_email':CORRESPONDING['email'],
 'main_figures':6,'main_tables':4,'supplementary_figures':1,'references':28,
 'word_count_approximate':artifact['main_word_count_approximate'],'word_count_scope':artifact['word_count_scope'],
 'AI_disclosure':AI_DISCLOSURE,
 'author_declarations_complete':not PENDING_CONFLICTS,
 'submission_performed':False,'ORCID_account_authorization_performed':False}
write(BASE/'SUBMISSION_FIELDS.json',fields)
guide='''Applied Spectroscopy submission files

01_Submit contains the English manuscript, title page, cover letter and
supplementary information. Upload the six main EPS figures and the separate
supplementary EPS using the corresponding figure file categories. Main tables
and equations are editable within the manuscript. Each figure has a full frame.

Supplementary_Tables.xlsx and workbooks B1–B5 provide the six Excel data
supplements. The Material_and_Origin sheet lists the 18 geographic groups by
material, with specimen splits and recorded pellet thicknesses. The main workbook includes all
LOWO folds and predictions, E1 predictions, stability and checkpoint diagnostics.
The optional declarations file repeats the manuscript statements; use it only
if the portal requests a separate declarations upload.

02_Read contains the full English-Chinese parallel manuscript and revision
record for the authors. These two documents are for reading, not English-paper
upload. 03_Evidence preserves verification receipts and new analysis tables.
The immutable repository sources are identified in EVIDENCE_MANIFEST.json.

The primary 93.44% result concerns internal five-source Suanzaoren discrimination.
The same-ID full-spectrum SVM scored 82.67%. Workbook-held-out material results
and additional-label controls use distinct protocols and are not added to that
score. Rankings can be supplied to other estimators, but a cross-model accuracy
gain has not been measured here.

The seven-author byline follows the title page. The manuscript describes pharmacy
procurement and spectral measurements by Yiyi Yang at Southwest University.
ZY07 identifies the Taiyuan, Shanxi Suanzaoren group.

Official guidance: https://journals.sagepub.com/author-instructions/ASP
The journal-specific requirements were checked using the archived official text;
the live page returned HTTP 403 during this revision. The portal's generated PDF
and account-linked ORCID authorization must be checked by the submitting author.
No journal submission was performed.

中文说明
01_Submit为英文投稿文件，02_Read为中英文对照与修改记录，03_Evidence为核查依据。
作者页列出七位作者及其单位。补充表的 Material_and_Origin 工作表按药材归列产地、样品划分和厚度。
上传时将文件放入对应类型，使用系统预览核对公式、图表与作者后，再正式提交。
'''
if PENDING_CONFLICTS:
 guide += '\nAuthor review copy. The corresponding author still needs the competing-interest declaration from ' + ', '.join(PENDING_CONFLICTS) + '.\n'
(OUT/'README.txt').write_text(guide)
entries={
 'README.txt':OUT/'README.txt',
 '01_Submit/Manuscript.docx':BASE/'01_Manuscript.docx',
 '01_Submit/Title_Page.docx':BASE/'02_Title_Page.docx',
 '01_Submit/Cover_Letter.docx':BASE/'03_Cover_Letter.docx',
 '01_Submit/Supplementary_Information.docx':BASE/'04_Supplementary_Information.docx',
 '01_Submit/Optional/Declarations.docx':BASE/'05_Declarations.docx',
 '01_Submit/Supplementary_Tables.xlsx':BASE/'Supplementary_Tables.xlsx',
 '02_Read/Bilingual.docx':BASE/'06_Manuscript_EN_ZH.docx',
 '02_Read/Revision_Notes.docx':BASE/'07_Revision_Notes_EN_ZH.docx',
 '02_Read/Submission_Fields.json':BASE/'SUBMISSION_FIELDS.json',
}
for key in ['1','2','3','4','5','6']:
 entries[f'01_Submit/Figures/Figure_{key}.eps']=BASE/f'figures/Figure_{key}.eps'
entries['01_Submit/Supplementary_Figure/Figure_S1.eps']=BASE/'figures/Figure_S1.eps'
for i in range(1,6):entries[f'01_Submit/Supplementary_B{i}.xlsx']=OLD/f'Supplementary_B{i}.xlsx'
for name in ['EVIDENCE_MANIFEST.json','ARTIFACT_VERIFICATION.json','SUPPLEMENTARY_TABLES_VERIFICATION.json','JOURNAL_COMPLIANCE.json','VISUAL_REVIEW.json','FIGURE_QUALITY_REVIEW.json']:
 entries['03_Evidence/'+name]=BASE/name
entries['03_Evidence/Publication_Provenance.json']=PROVENANCE_PATH
for p in sorted((BASE/'analysis').iterdir()):
 if p.is_file():entries['03_Evidence/analysis/'+p.name]=p
manifest={'package':'Submission.zip','files':[{'path':name,'sha256':sha(p),'bytes':p.stat().st_size,'repository_source':str(p.relative_to(ROOT))} for name,p in sorted(entries.items())]}
manifest_bytes=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
target=OUT/'Submission.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name,p in sorted(entries.items()):z.write(p,name)
 z.writestr('FILE_MANIFEST.json',manifest_bytes)
with zipfile.ZipFile(target) as z:
 assert z.testzip() is None
 assert len(z.namelist())==len(entries)+1
 for row in manifest['files']:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
receipt={'status':'PASS','package':'Submission.zip','sha256':sha(target),'bytes':target.stat().st_size,
 'entries':len(entries)+1,'CRC_and_extracted_hashes_verified':True,'visual_pages_reviewed':compliance['all_document_pages_visually_reviewed'],
 'author_metadata_synced':True,'author_declarations_complete':not PENDING_CONFLICTS,'AI_disclosure_synced':True,'raw_workbooks_unchanged':18,'old_Supplementary_Data_S1_unchanged':True,'six_standard_XLSX_supplements_below_10MiB':True,'journal_submission_performed':False}
write(BASE/'PACKAGE_VERIFICATION.json',receipt)
package_alias='Review.zip' if PENDING_CONFLICTS else 'Paper.zip'
for short,src in [(package_alias,target),('Paper.docx',BASE/'01_Manuscript.docx'),('Parallel.docx',BASE/'06_Manuscript_EN_ZH.docx'),('Changes.docx',BASE/'07_Revision_Notes_EN_ZH.docx')]:
 shutil.copyfile(src,Path('/workspace')/short)
print(json.dumps(receipt,indent=2))
