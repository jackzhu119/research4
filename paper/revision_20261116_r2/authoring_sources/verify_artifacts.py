"""Check publication metadata, retained disclosures, and the delivered documents."""
from pathlib import Path
import json,hashlib,re,zipfile
from docx import Document
from docx.oxml.ns import qn
from manuscript_content import TITLE,BLOCKS
from publication_metadata import (
    AFFILIATIONS, AUTHORS, AUTHOR_NAMES, AUTHOR_APPROVAL, AI_DISCLOSURE,
    AI_DISCLOSURE_ZH, CONFLICTS, PROVENANCE,
)

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'paper/revision_20261116_r2';SRC=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
lock=json.loads((SRC/'AUTHOR_METADATA_LOCK.json').read_text())
docs={p.name:Document(p) for p in BASE.glob('*.docx')}
assert len(docs)==7
main=docs['01_Manuscript.docx'];parallel=docs['06_Manuscript_EN_ZH.docx'];supp=docs['04_Supplementary_Information.docx']
byline=', '.join(a['name']+str(a['affiliation_id'])+('*' if a['corresponding'] else '') for a in AUTHORS)
assert [a['order'] for a in AUTHORS]==list(range(1,8)) and all(a['approved'] for a in AUTHORS)
assert all(a['competing_interest']=='none' for a in AUTHORS)
assert AUTHOR_NAMES == ['Ziyuan Zhu', 'Jiayi Ren', 'Yiyi Yang', 'Wenchen Jiang', 'Shuyu Zhou', 'Mengting Zhu', 'Yin Shen']
for name in ['01_Manuscript.docx','02_Title_Page.docx','04_Supplementary_Information.docx','06_Manuscript_EN_ZH.docx']:
 bylines=[p for p in docs[name].paragraphs if p.text==byline]
 assert len(bylines)==1,name
 assert [r.text for r in bylines[0].runs if r.font.superscript] == ['1','1','1','2','1','1','1*']
 text=[p.text for p in docs[name].paragraphs]
 for aff in AFFILIATIONS:
  assert str(aff['id'])+' '+aff['name'] in text,(name,aff['id'])
 for item in lock['header'][2:]:
  assert item['text'] in text,(name,item['text'])
for name in ['01_Manuscript.docx','06_Manuscript_EN_ZH.docx']:
 text=[p.text for p in docs[name].paragraphs]
 for item in (lock['end_matter'][1], lock['end_matter'][7], lock['end_matter'][15]):
  assert item['text'] in text,(name,item['text'])
 assert text.count('AI disclosure')==text.count(AI_DISCLOSURE)==1
 disclosure_heading=text.index('AI disclosure')
 assert text.index('Acknowledgements') < disclosure_heading < text.index('Author contributions')
 assert text[disclosure_heading+1]==AI_DISCLOSURE
 assert text.count(AI_DISCLOSURE_ZH)==int(name=='06_Manuscript_EN_ZH.docx')
 assert lock['ai_disclosure'] not in text
 assert any('Yiyi Yang performed the spectroscopic measurements and publicly shared the spectral data.' in p and 'Wenchen Jiang revised the machine-learning code and compiled the experimental results.' in p for p in text)
 assert PROVENANCE['measurement']['operator_en'] in '\n'.join(text)
title_paragraphs=[p.text for p in docs['02_Title_Page.docx'].paragraphs]
assert [p for p in title_paragraphs if '@' in p and p.split(':',1)[0] in AUTHOR_NAMES]==[f"{a['name']}: {a['email']}" for a in AUTHORS]
assert all(any(a['work_address'] in p for p in title_paragraphs) for a in AFFILIATIONS)
assert 'Authors: '+', '.join(AUTHOR_NAMES) in [p.text for p in docs['05_Declarations.docx'].paragraphs]
cover=Document(SRC/'templates/03_Cover_Letter.docx');newcover=docs['03_Cover_Letter.docx']
template_cover=[(i,p) for i,p in enumerate(cover.paragraphs) if not p.text.startswith('The use of OpenAI ChatGPT/Codex')]
assert len(template_cover)==len(newcover.paragraphs)
for (i,a),b in zip(template_cover,newcover.paragraphs):
 if i not in {2,3,4,5,6,8}:assert a.text==b.text,('cover',i)
assert len(main.tables)==4 and len(parallel.tables)==8
assert len(main.inline_shapes)==6 and len(supp.inline_shapes)==1
assert len(main.element.findall('.//'+qn('m:oMath')))==8
assert len(supp.element.findall('.//'+qn('m:oMath')))==3
numeric=re.compile(r'[+−-]?\d+(?:\.\d+)?(?:%|/\d+)?')
for i,t in enumerate(main.tables):
 for u in parallel.tables[2*i:2*i+2]:
  assert len(t.rows)==len(u.rows)
  for a,b in zip(t.rows,u.rows):
   for x,y in zip(a.cells,b.cells):assert numeric.findall(x.text)==numeric.findall(y.text),(i,x.text,y.text)
texts={name:'\n'.join([p.text for p in d.paragraphs]+[c.text for t in d.tables for r in t.rows for c in r.cells]) for name,d in docs.items()}
ai_markers=re.compile(r'\b(?:OpenAI|ChatGPT|Codex|AI[- ](?:generated|assisted|disclosure))\b|AI\s*使用披露|AI\s*生成',re.I)
for name,text in texts.items():
 assert TITLE in text
 for allowed in ('AI disclosure', 'AI 使用披露', AI_DISCLOSURE, AI_DISCLOSURE_ZH):
  if name in ['01_Manuscript.docx','06_Manuscript_EN_ZH.docx']:
   text=text.replace(allowed,'',1)
 assert not ai_markers.search(text),name
 assert not re.search(r'\[ref\d|\bTODO\b|\bTBD\b|INSERT HERE|fill in here|\bplaceholder\b|待补|待填|codex-annotation|购买.*线上|购自网络|purchased online|Shanxi group|国家农业信息化工程|National Engineering Research Center',text,re.I),name
 if name in ['01_Manuscript.docx','04_Supplementary_Information.docx','06_Manuscript_EN_ZH.docx']:
  assert not re.search(r'calycosin|spinosin|protocatechuic|Hessian|毛蕊异黄酮|斯皮诺素|原儿茶酸',text,re.I),name
 with zipfile.ZipFile(BASE/name) as z:
  assert z.testzip() is None
  assert not any(p.startswith('word/comments') for p in z.namelist())
 for tag in ['w:ins','w:del','w:commentRangeStart','w:moveFrom','w:moveTo','w:pBdr']:
  assert not docs[name].element.findall('.//'+qn(tag)),(name,tag)
 assert not docs[name].settings.element.findall('.//'+qn('w:trackRevisions')),name
 assert docs[name].core_properties.last_modified_by=='' and docs[name].core_properties.comments=='',name
for name in ['01_Manuscript.docx','02_Title_Page.docx','03_Cover_Letter.docx','05_Declarations.docx','06_Manuscript_EN_ZH.docx']:
 assert CONFLICTS in texts[name] and AUTHOR_APPROVAL in texts[name],name
for name in ['01_Manuscript.docx','06_Manuscript_EN_ZH.docx']:
 for value in ['traditional Chinese medicine pharmacies','Southwest University','Taiyuan (Shanxi)','1,670 specimens','two Zaoren specimens from Yunnan','1.5 mm','1 mm']:
  assert value in texts[name],(name,value)
source_rows=supp.tables[0].rows
assert len(source_rows)==19
assert [r.cells[0].text for r in source_rows[1:]]==['Suanzaoren']*5+['Zaoren']*5+['Huangqi']*4+['Tianma']*4
assert [r.cells[2].text for r in source_rows if r.cells[1].text=='ZY07']==['Taiyuan, Shanxi']
for statement in ['93.44%','82.67%','10.78 percentage points','91.78%','98.84%','65.61%','0.8110','0.8360']:
 assert statement in texts['01_Manuscript.docx'],statement
refs=json.loads((SRC/'resolved_references.json').read_text());assert len(refs)==28
reference_paragraphs=[p.text for p in main.paragraphs if re.match(r'^\d+\. ',p.text)]
assert len(reference_paragraphs)==28 and [int(p.split('.')[0]) for p in reference_paragraphs]==list(range(1,29))
assert reference_paragraphs==[str(i)+'. '+refs[str(i)] for i in range(1,29)]
assert 'https://github.com/jackzhu119/research4' in texts['01_Manuscript.docx']
for model in ['XQM-0.4L','DHG101-0B','FA2104N']:
 assert model in texts['01_Manuscript.docx'],model
word_count=len(re.findall(r'\S+',texts['01_Manuscript.docx']))
result={'status':'PASS','baseline_archive_sha256':lock['baseline_archive_sha256'],
 'author_order':AUTHOR_NAMES, 'author_metadata_and_affiliations_verified':True,
 'AI_disclosure_confined_to_acknowledgements':True, 'author_conflict_declarations_complete':True,
 'acquisition_origin_and_specimen_counts_verified':True,
 'main_tables':4,'main_figures':6,'supplementary_figures':1,'native_main_equations':8,'native_supplement_equations':3,
 'bilingual_table_numbers_match':True,'references':28,'abstract_words':len(next(b['en'] for b in BLOCKS if b['kind']=='abstract').split()),
 'main_word_count_approximate':word_count,'word_count_scope':'Paragraphs and tables including references and captions; excludes equation and figure text',
 'editorial_comments_or_tracked_changes':False,'placeholder_or_internal_citation_tokens':False,
 'files':[{'path':name,'sha256':sha(BASE/name),'bytes':(BASE/name).stat().st_size} for name in sorted(docs)]}
(BASE/'ARTIFACT_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))
