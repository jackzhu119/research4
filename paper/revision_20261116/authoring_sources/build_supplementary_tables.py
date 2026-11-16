"""Export existing corrected data and new verified summaries; no model training."""
from pathlib import Path
import hashlib, json, re
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from manuscript_content import TITLE

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'paper/revision_20261116'; OLD=ROOT/'paper/revision_20261115'
CURRENT=ROOT/'paper/current';AN=BASE/'analysis'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
original=OLD/'Supplementary_Tables.xlsx';book=load_workbook(original)
removed=[n for n in book.sheetnames if n.startswith('Molecular_') or n=='Source_Index']
for n in removed:del book[n]
mapping={
 'Primary_representation':AN/'primary_models.csv',
 'Primary_fold_metrics':AN/'primary_fold_metrics.csv',
 'Primary_paired_effects':AN/'paired_primary_effects.csv',
 'Within_pipeline_stability':AN/'v4_stability.csv',
 'Selection_probability':AN/'v4_primary_selection_probability.csv',
 'Cross_pipeline_K62':AN/'cross_pipeline_K62.csv',
 'Fixed_training_MC_stability':AN/'fixed_train_stability.csv',
 'Crossed_mask_contexts':AN/'crossed_masks.csv',
 'LOWO_pooled_metrics':AN/'lowo_metrics.csv',
 'E1_outer_parameters':CURRENT/'results/focused_followup_20261116/e1/outer_parameters.csv',
 'MLP_checkpoint_diagnostics':CURRENT/'results/focused_followup_20261116/e6/checkpoint_diagnostics.csv',
 'E1_OOF_predictions':CURRENT/'results/focused_followup_20261116/e1/oof_predictions.csv.gz',
}
def append_frame(name,frame):
 if name in book.sheetnames:del book[name]
 ws=book.create_sheet(name);ws.append(list(frame.columns))
 for row in frame.itertuples(index=False,name=None):
  vals=[]
  for val in row:
   if pd.isna(val):val=None
   elif hasattr(val,'item'):val=val.item()
   if isinstance(val,str) and val.startswith(('=','+','-','@')):val="'"+val
   vals.append(val)
  ws.append(vals)
 return ws
for name,path in mapping.items():append_frame(name,pd.read_csv(path))
lowo_folds=[];lowo_predictions=[]
for task in ['binary_kinds','four_kinds']:
 for filename,frames in [('fold_metrics.csv',lowo_folds),('oof_predictions.csv',lowo_predictions)]:
  frame=pd.read_csv(CURRENT/'results/kind_lowo_20261115'/task/filename);frame.insert(0,'evaluation_task',task);frames.append(frame)
append_frame('LOWO_all_folds',pd.concat(lowo_folds,ignore_index=True))
append_frame('LOWO_OOF_predictions',pd.concat(lowo_predictions,ignore_index=True))
index=book.create_sheet('Source_Index',0)
index.append(['Supplementary tables for medicinal-material terahertz discrimination'])
index.append(['Manuscript',TITLE]);index.append(['First author','Ziyuan Zhu','ORCID','https://orcid.org/0009-0005-1153-4926'])
index.append(['Corresponding author','Yin Shen','ORCID','https://orcid.org/0000-0002-3221-7875'])
index.append(['Collection','18 workbooks; 1,672 physical records; original TRAIN/CAL/PRED roles preserved'])
index.append(['Primary comparison','300 Suanzaoren TRAIN specimens; three fivefold partitions; 900 related predictions'])
index.append(['Interval policy','Primary_paired_effects uses 10,000 source-stratified physical-ID draws, seed 20261116; conditional and unadjusted.'])
index.append(['Historical intervals','Paired_budget_intervals preserves its archived method; do not substitute its interval definition for Primary_paired_effects.'])
index.append(['Selection scope','Within-pipeline stability, cross-pipeline overlap and fixed-training MC variation are distinct estimands.'])
index.append(['Expanded-search qualification','First-gamma cache defect affects the separate expanded worker. Its performance/cost claims are excluded from the primary comparison.'])
index.append(['Reuse scope','Ranking interface is classifier-independent; measured feature-selection gain here is with SVM.'])
index.append(['Original corrected data','Existing nonmolecular sheets are retained; B1–B5 source workbooks remain byte-identical.'])
index.append([]);index.append(['Sheet','Source','SHA256','Scope'])
for name,path in mapping.items():index.append([name,str(path.relative_to(ROOT)),sha(path),'Verified saved predictions or masks; see EVIDENCE_MANIFEST.json'])
for name in ['LOWO_all_folds','LOWO_OOF_predictions']:index.append([name,'paper/current/results/kind_lowo_20261115','See EVIDENCE_MANIFEST.json','Fixed full-spectrum SVM; whole-workbook groups'])
for name in book.sheetnames:
 if name not in {'Source_Index',*mapping,'LOWO_all_folds','LOWO_OOF_predictions'}:
  index.append([name,'paper/revision_20261115/Supplementary_Tables.xlsx',sha(original),'Unchanged corrected source sheet; original protocol definitions apply'])
for ws in book:
 ws.freeze_panes='A2';ws.sheet_view.showGridLines=False
 if ws.max_row>1:ws.auto_filter.ref=ws.dimensions
 for cell in ws[1]:cell.font=Font(name='Calibri',size=11,bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='34495E');cell.alignment=Alignment(wrap_text=True,vertical='center')
 ws.row_dimensions[1].height=30
 for col in range(1,ws.max_column+1):
  vals=[len(str(ws.cell(row,col).value or '')) for row in range(1,min(ws.max_row,150)+1)]
  ws.column_dimensions[get_column_letter(col)].width=min(65,max(12,max(vals,default=12)+2))
 for row in ws.iter_rows(min_row=2):
  for cell in row:cell.font=Font(name='Calibri',size=10);cell.alignment=Alignment(vertical='top',wrap_text=isinstance(cell.value,str) and len(cell.value)>60)
target=BASE/'Supplementary_Tables.xlsx';book.save(target)
reloaded=load_workbook(target,read_only=True,data_only=True)
assert reloaded['E1_OOF_predictions'].max_row==901
assert reloaded['LOWO_OOF_predictions'].max_row==2537
assert reloaded['LOWO_all_folds'].max_row==29
assert target.stat().st_size<10*1024**2
# Check retained cells, not ZIP/XML serialization or cosmetic workbook metadata.
oldbook=load_workbook(original,read_only=True,data_only=True)
preserved=[]
def canonical_rows(ws):
 rows=[]
 for row in ws.values:
  row=list(row)
  while row and row[-1] is None:row.pop()
  rows.append(tuple(row))
 while rows and not rows[-1]:rows.pop()
 return rows
for name in oldbook.sheetnames:
 if name in removed or name=='Primary_representation':continue
 assert canonical_rows(oldbook[name])==canonical_rows(reloaded[name]),name
 preserved.append(name)
receipt={'status':'PASS','workbook_sha256':sha(target),'sheets':reloaded.sheetnames,
 'preserved_original_sheet_values':preserved,'removed_molecular_sheets':[n for n in removed if n.startswith('Molecular_')],
 'source_workbook_sha256':sha(original),'source_files':{n:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for n,p in mapping.items()},
 'E1_predictions':900,'LOWO_predictions':2536,'LOWO_folds':28,'scientific_models_trained':0}
(BASE/'SUPPLEMENTARY_TABLES_VERIFICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','sheets':len(reloaded.sheetnames),'bytes':target.stat().st_size,'preserved_sheets':len(preserved)},indent=2))
