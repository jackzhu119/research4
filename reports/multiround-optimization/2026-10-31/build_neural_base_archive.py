"""Compact and validate the executed dense-source search without model fitting."""
from pathlib import Path
import base64
import csv
import gzip
import hashlib
import io
import json
import shutil

root=Path(__file__).resolve().parents[2]
run=root/'outputs/multiround-20261031/neural-base'
archive=run/'compact_archive'
models=run/'deliverable_models'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
status=json.loads((run/'status.json').read_text())
if status['status']!='completed':raise ValueError('Archive only after final completion')
archive.mkdir(exist_ok=True);models.mkdir(exist_ok=True)
files=sorted(p for base in [run/'development',run/'final'] for p in base.rglob('*') if p.is_file())
original={str(p.relative_to(run)):digest(p) for p in files}
for p in run.glob('*.json'):original[str(p.relative_to(run))]=digest(p)
write(archive/'original_files_sha256.json',original)
training=list(p for p in files if p.name=='training.json')
with gzip.open(archive/'training_checkpoints.jsonl.gz','wt',encoding='utf-8',newline='\n') as f:
    for p in training:
        r={'path':str(p.relative_to(run)),'sha256':digest(p),'text':p.read_bytes().decode('utf-8')}
        f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
predictions=[p for p in files if p.name=='predictions.csv.gz']
rows=0;final_rows=0
with gzip.open(archive/'prediction_files.jsonl.gz','wt',encoding='utf-8',newline='\n') as exact,gzip.open(archive/'all_predictions.csv.gz','wt',encoding='utf-8',newline='') as combined:
    writer=csv.DictWriter(combined,fieldnames=['scope','candidate_id','episode','seed','record_id','y_true','y_predicted','is_correct'],lineterminator='\n');writer.writeheader()
    for p in predictions:
        b=p.read_bytes();exact.write(json.dumps({'path':str(p.relative_to(run)),'sha256':digest(p),'base64':base64.b64encode(b).decode('ascii')},separators=(',',':'))+'\n')
        scope=p.relative_to(run).parts[0]
        with gzip.open(p,'rt',encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                row['scope']=scope;row.setdefault('episode','');writer.writerow(row);rows+=1
                if scope=='final':final_rows+=1
for p in run.glob('*.json'):shutil.copy2(p,archive/p.name)
for p in [root/'outputs/multiround-20261031/neural-base-development.log',root/'outputs/multiround-20261031/neural-base-final.log',root/'outputs/multiround-20261031/neural-base-tests.log',root/'outputs/multiround-20261031/neural-base-development-audit.log',root/'outputs/multiround-20261031/neural-base-final-audit.log',Path(__file__),root/'outputs/multiround-20261031/audit_neural_base.py',root/'outputs/multiround-20261031/audit_neural_base_final.py']:
    if p.is_file():shutil.copy2(p,archive/p.name)
candidate=json.loads((run/'selection_lock.json').read_text())['choice']['selected']['id']
model_files=[]
for base in [run/'development',run/'final']:
    for p in sorted(base.rglob('model.pt')):
        if p.parent.name!=candidate:continue
        for q in [p,p.parent/'preprocessor.joblib',p.parent/'training.json']:
            dest=models/q.relative_to(run);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,dest)
            if q.suffix in {'.pt','.joblib'}:model_files.append({'relative_path':str(dest.relative_to(models)),'sha256':digest(dest),'bytes':dest.stat().st_size,'scope':q.relative_to(run).parts[0]})
write(models/'manifest.json',{'selected_candidate':candidate,'source_contexts':21,'model_weights':21,'preprocessors':21,'artifacts':model_files,'all_source_only':True,'adapted_models':False})
restored=0
with gzip.open(archive/'training_checkpoints.jsonl.gz','rt',encoding='utf-8') as f:
    for line in f:
        r=json.loads(line);b=r['text'].encode('utf-8');assert hashlib.sha256(b).hexdigest()==r['sha256'];assert b==(run/r['path']).read_bytes();restored+=1
with gzip.open(archive/'prediction_files.jsonl.gz','rt',encoding='utf-8') as f:
    for line in f:
        r=json.loads(line);b=base64.b64decode(r['base64']);assert hashlib.sha256(b).hexdigest()==r['sha256'];assert b==(run/r['path']).read_bytes();restored+=1
assert rows==44496+513 and final_rows==513
assert len(model_files)==42
for r in model_files:assert digest(models/r['relative_path'])==r['sha256']
write(archive/'archive_validation.json',{'status':'pass','training_metadata_files':len(training),'prediction_files':len(predictions),'byte_exact_originals_restorable':restored,'combined_prediction_rows':rows,'final_prediction_rows':final_rows,'model_weights_copied':21,'preprocessors_copied':21,'additional_model_fits':0})
(archive/'README.md').write_text('''# Supervised dense source-model search evidence

This is a retrospective same-batch source-representation search on real uploaded
spectra with anonymous origin groups. It uses no calibration labels. The locked
b006 model is SNV, source-only feature scaling, no PCA, hidden64-32, ReLU,200epochs,
unweighted source cross-entropy. It is independently implemented in PyTorch and
is not an exact MATLAB newff reproduction.

24candidates x6training-only tasks x3seeds =432development source fits,44496query
predictions. A source-only4/2 selection/gate locks the configuration before the
three final513source models predict171fixed specimens each. All four research
rounds were development-locked before the common final gate was released.

The final mean accuracy89.0838% is close to SVM88.8889%, but mean macroF1 .7729
is lower than SVM .8616. Neither this source improvement nor replacing a neural
backbone is a few-shot calibration gain. All seeds, imbalanced-class failures,
and all candidate development results remain inspectable.

`training_checkpoints.jsonl.gz` stores each original UTF-8training.json as text
plus original path and SHA; encoding its text field to UTF-8 recovers exact bytes.
`prediction_files.jsonl.gz` preserves each original compressed CSV file as base64
plus its path and SHA, allowing even original gzip bytes to be recovered.
`all_predictions.csv.gz` provides the same45009rows as an ordinary unified CSV.
`original_files_sha256.json` covers original source weights,preprocessors,training
metadata,predictions,metrics and root manifests. All435metadata and435compressed
prediction files were byte-compared after restoration. The435development/final
cache contexts are retained locally; routine reproduction regenerates the search.

Only18locked development sources plus3final sources are exported separately
as21PyTorchstate_dicts and21source-fittedpreprocessors for downstream adaptation.
`development_arithmetic_audit.json` checks44496prediction rows and source-only
selection using independent sklearn formulas; `final_arithmetic_and_resume_audit.json`
checks513final rows,inference replay,pointwise predictions and no-fit resume.
Their scripts assume the original execution output layout,not this compact folder.

All855specimens have previous research exposure. Each origin currently has one
batch; these splits and repeated tasks are not independently acquired domains.
The anonymous labels have not independently verified real geographic names.
''',encoding='utf-8')
for base in [archive,models]:
    write(base/'files_sha256.json',{str(p.relative_to(base)):digest(p) for p in sorted(base.rglob('*')) if p.is_file() and p.name!='files_sha256.json'})
print(json.dumps({'status':'pass','archive':str(archive),'model_directory':str(models),'training_checkpoints':len(training),'prediction_files':len(predictions),'prediction_rows':rows,'byte_exact_restorable_files':restored}))
