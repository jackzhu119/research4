"""Check delivered documents against the uploaded author baseline and source data."""
from pathlib import Path
import json,hashlib,re,zipfile
from docx import Document
from docx.oxml.ns import qn
from manuscript_content import TITLE,BLOCKS

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'paper/revision_20261116';SRC=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
lock=json.loads((SRC/'AUTHOR_METADATA_LOCK.json').read_text())
docs={p.name:Document(p) for p in BASE.glob('*.docx')}
assert len(docs)==7
main=docs['01_Manuscript.docx'];parallel=docs['06_Manuscript_EN_ZH.docx'];supp=docs['04_Supplementary_Information.docx']
for name in ['01_Manuscript.docx','06_Manuscript_EN_ZH.docx']:
 text=[p.text for p in docs[name].paragraphs]
 for item in lock['header']+lock['end_matter']:
  assert item['text'] in text,(name,item['text'])
 assert text.count(lock['ai_disclosure'])==1
for name in ['02_Title_Page.docx','05_Declarations.docx']:
 template=SRC/'templates'/('Declarations.docx' if name.startswith('05') else name)
 original=Document(template);changed=docs[name];assert len(original.paragraphs)==len(changed.paragraphs)
 for i,(a,b) in enumerate(zip(original.paragraphs,changed.paragraphs)):
  if i==(1 if name.startswith('05') else 0):continue
  if a.text=='Five main figures and one supplementary figure':assert b.text=='Six main figures and one supplementary figure'
  else:assert a.text==b.text,(name,i)
cover=Document(SRC/'templates/03_Cover_Letter.docx');newcover=docs['03_Cover_Letter.docx']
for i,(a,b) in enumerate(zip(cover.paragraphs,newcover.paragraphs)):
 if i not in range(2,7):assert a.text==b.text,('cover',i)
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
for name,text in texts.items():
 assert TITLE in text
 assert not re.search(r'\[ref\d|\bTODO\b|\bTBD\b|INSERT HERE|待补|待填|codex-annotation',text),name
 if name in ['01_Manuscript.docx','04_Supplementary_Information.docx','06_Manuscript_EN_ZH.docx']:
  assert not re.search(r'calycosin|spinosin|protocatechuic|Hessian|毛蕊异黄酮|斯皮诺素|原儿茶酸',text,re.I),name
 with zipfile.ZipFile(BASE/name) as z:
  assert z.testzip() is None
  assert 'word/comments.xml' not in z.namelist()
 for tag in ['w:ins','w:del','w:commentRangeStart','w:pBdr']:
  assert not docs[name].element.findall('.//'+qn(tag)),(name,tag)
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
 'verbatim_authorship_acknowledgements_contributions_AI_and_declarations':True,
 'title_page_author_fields_unchanged':True,'cover_AI_approval_and_signatures_unchanged':True,
 'main_tables':4,'main_figures':6,'supplementary_figures':1,'native_main_equations':8,'native_supplement_equations':3,
 'bilingual_table_numbers_match':True,'references':28,'abstract_words':len(next(b['en'] for b in BLOCKS if b['kind']=='abstract').split()),
 'main_word_count_approximate':word_count,'word_count_scope':'Paragraphs and tables including references and captions; excludes equation and figure text',
 'editorial_comments_or_tracked_changes':False,'placeholder_or_internal_citation_tokens':False,
 'files':[{'path':name,'sha256':sha(BASE/name),'bytes':(BASE/name).stat().st_size} for name in sorted(docs)]}
(BASE/'ARTIFACT_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))
