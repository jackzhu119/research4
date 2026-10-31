"""Seal every final audit file except this index's own bytes; no model fitting."""
import ast,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

OUT=Path(__file__).resolve().parent
TARGET=OUT/'files_sha256.json'

def main():
 files={};fit_calls=[]
 for path in sorted(OUT.rglob('*')):
  if not path.is_file() or path==TARGET:continue
  files[str(path.relative_to(OUT))]=hashlib.sha256(path.read_bytes()).hexdigest()
  if path.suffix=='.py':
   tree=ast.parse(path.read_text(),filename=str(path))
   for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ['fit','fit_transform','partial_fit']:
     fit_calls.append({'file':str(path.relative_to(OUT)),'line':node.lineno,'method':node.func.attr})
 if fit_calls:raise AssertionError('Unexpected training call in independent audit code: '+str(fit_calls))
 result={'sealed_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Every current file under independent-audit/ except this index itself; no exclusions by extension or artifact type','files_sha256':files,'file_count':len(files),'training_method_calls_in_audit_source':fit_calls,'new_audit_model_fits':0,'preserved_audit_reader_corrections':['review_attempt01_header_whitespace/','source_review_attempt01_unittest_format/'],'scientific_certification':False}
 with TARGET.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
 print(json.dumps({'index_sha256':hashlib.sha256(TARGET.read_bytes()).hexdigest(),'file_count':len(files),'new_audit_model_fits':0}))

if __name__=='__main__':main()
