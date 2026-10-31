"""Archive evidence and render fixed scientific figures; never refit models."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import re
import shutil

ROOT=Path('/workspace/research4')
SOURCE=ROOT/'outputs/approved-zaoren-20261031'
DEST=ROOT/'reports/approved-zaoren/2026-10-31'
if DEST.exists():raise FileExistsError('Preserve existing archival evidence')
DEST.mkdir(parents=True)
SHA=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ignore(folder,names):
    return [name for name in names if name in {'.cache','joblib','__pycache__'} or Path(name).suffix.lower() in {'.npz','.joblib','.pkl','.pyc'}]

def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

for name in ['experiments','independent-audit','synthetic-transfer-results']:
    shutil.copytree(SOURCE/name,DEST/name,ignore=ignore)
for name in ['quality_report.json','record_manifest.csv']:
    target=DEST/'data-preparation'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(SOURCE/'data-preparation'/name,target)
(DEST/'datasets').mkdir();shutil.copy2(SOURCE/'datasets/generation.json',DEST/'datasets/generation.json')
(DEST/'synthetic-transfer-inputs').mkdir();shutil.copy2(SOURCE/'synthetic-transfer-inputs/input_scope.json',DEST/'synthetic-transfer-inputs/input_scope.json')
helpers=DEST/'execution_helpers';helpers.mkdir()
for name in ['run_frozen_matrix.py','recalculate_results.py','archive_results.py']:
    shutil.copy2(SOURCE/name,helpers/name)
shutil.copy2(SOURCE/'timing_precheck.json',DEST/'timing_precheck.json')
shutil.copy2(SOURCE/'all-tests.log',DEST/'all-tests.log')
test_text=(DEST/'all-tests.log').read_text();match=re.search(r'Ran (\d+) tests in ([0-9.]+)s',test_text)
assert match and int(match.group(1))==100 and test_text.rstrip().endswith('OK')
write(DEST/'tests_summary.json',{'scope':'unit and integration engineering checks; not 100 real-data model experiments',
      'tests':100,'reported_status':'OK','reported_seconds':float(match.group(2)),
      'log':'all-tests.log','log_sha256':SHA(DEST/'all-tests.log')})
print('Evidence copied without caches, pickles, joblib or NPZ inputs.')
