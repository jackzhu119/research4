"""Update only the title cell of the 43-sheet previously verified workbook."""
from pathlib import Path
import json,hashlib,shutil
from openpyxl import load_workbook
from manuscript_content import TITLE
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'paper/revision_20261116_r2';OLD=ROOT/'paper/revision_20261116'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=OLD/'Supplementary_Tables.xlsx';target=BASE/'Supplementary_Tables.xlsx'
b=load_workbook(source);assert len(b.sheetnames)==43 and b['Source_Index']['A2'].value=='Manuscript'
b['Source_Index']['B2']=TITLE;b.save(target)
a=load_workbook(source,read_only=True,data_only=False);c=load_workbook(target,read_only=True,data_only=False)
assert a.sheetnames==c.sheetnames
changed=[]
for name in a.sheetnames:
 assert a[name].max_row==c[name].max_row and a[name].max_column==c[name].max_column
 for r1,r2 in zip(a[name].iter_rows(),c[name].iter_rows()):
  for x,y in zip(r1,r2):
   if x.value!=y.value:changed.append((name,x.coordinate,x.value,y.value))
assert changed==[('Source_Index','B2',a['Source_Index']['B2'].value,TITLE)],changed
receipt=json.loads((OLD/'SUPPLEMENTARY_TABLES_VERIFICATION.json').read_text())
receipt.update(status='PASS',workbook_sha256=sha(target),previous_revision_workbook_sha256=sha(source),title_only_update=True,all_other_cells_unchanged=True,changed_cells=[{'sheet':'Source_Index','cell':'B2'}],sheets=c.sheetnames,scientific_models_trained=0)
(BASE/'SUPPLEMENTARY_TABLES_VERIFICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print('PASS:43 sheets; only Source_Index!B2 changed; all other cells identical.')
