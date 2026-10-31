#!/usr/bin/env python3
"""Six fresh INNER-only neural trajectories; no outer evaluation or deployment selection."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,importlib.metadata,json,os,shutil,sys,tarfile,time
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from herb_spectra.upgrade_neural import (UpgradeNet,PROFILES,CLASSES,MODALITIES,row_snv,read_pack,pack_modalities,state_hash,metric_record)
from herb_spectra.leakage_guard import FitScope,LeakageError,audit_pointwise_inference

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/upgrade-20261031-v1/neural'
BASE=ROOT/'reports/upgrade-20261031-v1/neural'
RUN_NAME='zaoren-neural-learning-pilot-20261031-v1'
SPLIT='outer00.inner00'
CHECKPOINTS=(12,24,48,96)
FAMILIES=('cnn_alpha','fusion_all_four')
SOURCE_FILES=('scripts/run_source_learning_pilot.py','tests/test_source_learning_pilot.py','src/herb_spectra/upgrade_neural.py','src/herb_spectra/leakage_guard.py','src/herb_spectra/classification.py','src/herb_spectra/selection.py','src/herb_spectra/multiplant_database.py')


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if p.suffix=='.gz' else open
    with opener(p,'wt',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def encode(labels):
    try:return np.asarray([CLASSES.index(str(v)) for v in labels],dtype=np.int64)
    except ValueError as exc:raise LeakageError('Unknown anonymous class') from exc


def read_inner_truth(path,ids):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    lookup={r['record_id']:r['label'] for r in rows}
    if len(lookup)!=len(rows) or set(lookup)!=set(ids):raise LeakageError('Inner truth must match exact validation IDs')
    y=np.asarray([lookup[str(i)] for i in ids]);encode(y)
    if tuple(np.unique(y))!=CLASSES:raise LeakageError('All seven validation classes required')
    return y


def validate_pair(source,validation):
    expected={'record_ids','X_alpha','X_n','X_time','X_frequency'}
    if set(source)!=expected|{'source_labels'} or set(validation)!=expected:raise LeakageError('Strict source/validation feature schema')
    a=source['record_ids'];b=validation['record_ids']
    if len(a)!=342 or len(b)!=342 or set(a)&set(b):raise LeakageError('Fixed342/342 disjoint source and validation required')
    if len(set(a))!=342 or len(set(b))!=342 or len(source['source_labels'])!=342:raise LeakageError('Unique paired IDs required')
    if tuple(np.unique(source['source_labels']))!=CLASSES:raise LeakageError('Seven source classes required')
    encode(source['source_labels'])
    for name,n in [('alpha',612),('n',612),('time',512),('frequency',256)]:
        for p in [source,validation]:
            v=p['X_'+name]
            if v.shape!=(342,n) or not np.issubdtype(v.dtype,np.number) or not np.isfinite(v).all():raise LeakageError('Invalid paired numeric spectrum shape')


def check_read_access(path,allowed):
    if isinstance(path,int):return
    try:p=Path(path).resolve()
    except (TypeError,ValueError):return
    protected=(ROOT/'data',OLD,BASE)
    if any(p.is_relative_to(tree) for tree in protected) and p not in allowed:raise PermissionError('Pilot cannot read other data, old neural inputs, truth or results')


def install_read_guard(out):
    def hook(event,args):
        if event=='open' and args:check_read_access(args[0],set())
    sys.addaudithook(hook)
    # An intentional failed read proves the declared Python API boundary. The
    # hook raises before bytes/labels from the outer truth can be opened.
    forbidden=OLD/'scoring_truth/outer00.final.csv'
    try:
        with forbidden.open('rb') as f:f.read(1)
    except PermissionError:
        write_json(out/'outer_read_rejection.json',{'at_utc':now(),'intentional_forbidden_path':str(forbidden),'read_rejected_before_bytes':True,'OS_sandbox':False})
    else:raise LeakageError('Outer read probe was not rejected')


def profile():return {**PROFILES[1],'epochs':96}


def code_manifest():return {n:sha(ROOT/n) for n in SOURCE_FILES}


def check_inner_input_hashes(original,context,frozen):
    for key,field in [('source','source_pack_sha256'),('validation','inference_pack_sha256')]:
        if sha(original[key])!=context[field]:raise LeakageError('Prepared inner pack changed')
    if sha(original['truth'])!=frozen['scoring_truth/outer00.inner00.csv']:raise LeakageError('Frozen inner truth bytes changed')


def prepare(out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Prepare requires a new empty directory')
    archive=BASE/'inner_outer_00_evidence.tar.gz';frozen=json.loads((BASE/'files_sha256.json').read_text())
    if sha(archive)!=frozen[archive.name]:raise LeakageError('Frozen source archive changed')
    member='selection/outer00.inner00/cnn_alpha/regularized24/seed42/context.json'
    proof=json.loads((BASE/'archive_validation.json').read_text())
    matching=[r for r in proof['records'] if r.get('archive')==str(archive.relative_to(ROOT)) and r.get('member')==member]
    if len(matching)!=1:raise LeakageError('Immutable inner-context binding missing')
    training_member='selection/outer00.inner00/cnn_alpha/regularized24/seed42/training.json'
    training_proof=[r for r in proof['records'] if r.get('archive')==str(archive.relative_to(ROOT)) and r.get('member')==training_member]
    if len(training_proof)!=1:raise LeakageError('Immutable source-ID metadata missing')
    with tarfile.open(archive,'r:gz') as t:
        raw=t.extractfile(member).read();training_raw=t.extractfile(training_member).read()
    if hashlib.sha256(training_raw).hexdigest()!=training_proof[0]['sha256']:raise LeakageError('Frozen training metadata changed')
    original_training=json.loads(training_raw)
    if hashlib.sha256(raw).hexdigest()!=matching[0]['sha256']:raise LeakageError('Inner context bytes changed')
    context=json.loads(raw)
    original={'source':OLD/'inputs'/SPLIT/'source.npz','validation':OLD/'inputs'/SPLIT/'inference.npz','truth':OLD/'scoring_truth'/f'{SPLIT}.csv'}
    check_inner_input_hashes(original,context,frozen)
    source=read_pack(original['source'],True);val=read_pack(original['validation']);validate_pair(source,val);truth=read_inner_truth(original['truth'],val['record_ids'])
    if source['record_ids'].tolist()!=original_training['source_ids']:raise LeakageError('Source IDs differ from immutable source training record')
    out.mkdir(parents=True,exist_ok=True);(out/'inputs').mkdir()
    copies={}
    for k,p in original.items():
        target=out/'inputs'/({'source':'source.npz','validation':'validation.npz','truth':'validation_truth.csv'}[k]);shutil.copyfile(p,target)
        copies[k]={'path':str(target.relative_to(out)),'sha256':sha(target),'original_path':str(p.relative_to(ROOT)),'bytes':target.stat().st_size}
    inputs={'protocol_id':RUN_NAME,'split_id':SPLIT,'copied_inputs':copies,'source_ids':source['record_ids'].tolist(),'validation_ids':val['record_ids'].tolist(),'source_class_counts':dict(zip(*np.unique(source['source_labels'],return_counts=True))),'validation_class_counts':dict(zip(*np.unique(truth,return_counts=True))),'old_source_archive':{'path':str(archive.relative_to(ROOT)),'sha256':sha(archive),'context_member':member,'context_sha256':matching[0]['sha256'],'frozen_validation_truth_sha256':frozen['scoring_truth/outer00.inner00.csv'],'source_identity_member':training_member,'source_identity_member_sha256':training_proof[0]['sha256']},'prepared_at_utc':now(),'outer_features_or_truth_loaded':False,'physical_independence_basis':'user-declared; exposed development data; no new-batch confirmation'}
    # Convert NumPy integer counts before the strict JSON writer.
    for k in ['source_class_counts','validation_class_counts']:inputs[k]={str(c):int(v) for c,v in inputs[k].items()}
    write_json(out/'input_manifest.json',inputs)
    protocol={'protocol_id':RUN_NAME,'split_id':SPLIT,'purpose':'INNER-only epoch diagnostic; no outer evaluation or deployment configuration selection','families':list(FAMILIES),'seeds':[42,43,44],'class_order':list(CLASSES),'profile':profile(),'maximum_epochs':96,'checkpoints':list(CHECKPOINTS),'batch_size':64,'optimizer':{'name':'Adam','learning_rate':.0005,'weight_decay':.001},'fresh_initialization_each_family_seed':True,'no_early_stopping_or_scheduler':True,'input_manifest_sha256':sha(out/'input_manifest.json'),'source_sha256':code_manifest(),'selection_from_validation':'No winner chosen; fixed checkpoint descriptions and full curves only','outer_evaluation':'NOT_RUN_FORBIDDEN','deployment_model_affected':False,'training_loss_definition':'Source train-mode minibatch CE multiplied by batch size, summed and divided by N; distinct from fixed-model full-source eval CE','evaluation':'eval/no_grad with fixed source class weights; parameters/BN/optimizer/grads/RNG unchanged','metric_replay_tolerance':{'count_metrics':1e-12,'cross_entropy':1e-5,'logits':1e-5},'new_real_specimens':0,'study_status':'EXPLORATORY historical source-only diagnostic','OS_truth_sandbox':False,'versions':{k:importlib.metadata.version(k) for k in ['numpy','torch','scikit-learn','joblib']},'created_at_utc':now()}
    write_json(out/'protocol.json',protocol)
    return {'protocol_sha256':sha(out/'protocol.json'),'input_manifest_sha256':sha(out/'input_manifest.json'),'source_sha256':protocol['source_sha256'],'families':2,'fresh_trajectories':6,'maximum_epochs':96,'outer_allowed':False}


def verify_release(out,release):
    p=json.loads((out/'protocol.json').read_text());inputs=json.loads((out/'input_manifest.json').read_text())
    expected={'protocol_sha256':sha(out/'protocol.json'),'input_manifest_sha256':sha(out/'input_manifest.json'),'source_sha256':code_manifest()}
    if not release.get('allow_source_training') or any(release.get(k)!=v for k,v in expected.items()):raise LeakageError('Root source release missing or changed')
    if p['source_sha256']!=expected['source_sha256'] or p['input_manifest_sha256']!=expected['input_manifest_sha256'] or p['families']!=list(FAMILIES) or p['seeds']!=[42,43,44] or p['checkpoints']!=list(CHECKPOINTS) or p['maximum_epochs']!=96 or p['profile']!=profile() or p['split_id']!=SPLIT or p['protocol_id']!=RUN_NAME or p['batch_size']!=64 or p['optimizer']!={'name':'Adam','learning_rate':.0005,'weight_decay':.001}:raise LeakageError('Frozen candidate or implementation changed')
    for r in inputs['copied_inputs'].values():
        if sha(out/r['path'])!=r['sha256']:raise LeakageError('Copied inner input changed')
    return p,inputs


def gradients_hash(model):
    return joblib.hash({k:None if p.grad is None else p.grad.detach().cpu().numpy() for k,p in model.named_parameters()})


def evaluate(model,optimizer,x,labels,weights):
    before=state_hash(model);opt=joblib.hash(optimizer.state_dict());grads=gradients_hash(model);rng=torch.get_rng_state().clone();old=model.training
    model.eval()
    with torch.no_grad():
        scores=model(x);ce=float(nn.CrossEntropyLoss(weight=weights)(scores,torch.as_tensor(encode(labels))));scores=scores.cpu().numpy()
    model.train(old)
    if before!=state_hash(model) or opt!=joblib.hash(optimizer.state_dict()) or grads!=gradients_hash(model) or not torch.equal(rng,torch.get_rng_state()):raise LeakageError('Evaluation changed learned or stochastic state')
    m=metric_record(labels,np.asarray(CLASSES)[scores.argmax(1)]);m['cross_entropy_source_class_weights']=ce
    return m,scores


def train_epoch(model,optimizer,criterion,tensors,target,generator):
    model.train();order=torch.randperm(len(target),generator=generator);total=0.
    for left in range(0,len(target),64):
        chosen=order[left:left+64];optimizer.zero_grad(set_to_none=True);loss=criterion(model({k:v[chosen] for k,v in tensors.items()}),target[chosen]);loss.backward();optimizer.step();total+=float(loss.detach())*len(chosen)
    return total/len(target)


def fit_transforms(source,family,scope):
    features={};scalers={}
    for name in MODALITIES[family]:
        raw=source['X_'+name];x=row_snv(raw) if name=='alpha' else raw
        scaler=scope.fit(StandardScaler(),x,sample_ids=source['record_ids'],component='scaler_'+name);scalers[name]=scaler;features[name]=scaler.transform(x).astype(np.float32)
    return features,scalers


def train(out,release_path):
    out=Path(out);release=json.loads(Path(release_path).read_text());protocol,inputs=verify_release(out,release)
    if (out/'training_started.json').exists():raise FileExistsError('No resume or overwrite: use a fresh protocol/run')
    write_json(out/'source_release_snapshot.json',release);write_json(out/'training_started.json',{'at_utc':now(),'protocol_sha256':sha(out/'protocol.json'),'outer_allowed':False})
    install_read_guard(out)
    source=read_pack(out/inputs['copied_inputs']['source']['path'],True);val=read_pack(out/inputs['copied_inputs']['validation']['path']);validate_pair(source,val);yval=read_inner_truth(out/inputs['copied_inputs']['truth']['path'],val['record_ids'])
    if source['record_ids'].tolist()!=inputs['source_ids'] or val['record_ids'].tolist()!=inputs['validation_ids']:raise LeakageError('Frozen roles changed')
    all_metrics=[];trajectory_records=[]
    for family in FAMILIES:
        for seed in (42,43,44):
            trajectory=out/'trajectories'/family/f'seed{seed}';trajectory.mkdir(parents=True)
            scope=FitScope(source['record_ids'],SPLIT,RUN_NAME,out/'fit_scope_events.jsonl',code_version=sha(out/'protocol.json'))
            x,scalers=fit_transforms(source,family,scope);v={name:scaler.transform(row_snv(val['X_'+name]) if name=='alpha' else val['X_'+name]).astype(np.float32) for name,scaler in scalers.items()}
            torch.manual_seed(seed);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
            model=UpgradeNet(family,profile(),{k:a.shape[1] for k,a in x.items()});initial=state_hash(model)
            optimizer=torch.optim.Adam(model.parameters(),lr=.0005,weight_decay=.001);generator=torch.Generator().manual_seed(seed)
            encoded=encode(source['source_labels']);counts=np.bincount(encoded,minlength=7);weights=torch.as_tensor(len(encoded)/(7*counts),dtype=torch.float32);criterion=nn.CrossEntropyLoss(weight=weights)
            xt={k:torch.as_tensor(a,dtype=torch.float32) for k,a in x.items()};vt={k:torch.as_tensor(a,dtype=torch.float32) for k,a in v.items()};target=torch.as_tensor(encoded,dtype=torch.long)
            joblib.dump(scalers,trajectory/'source_scalers.joblib');write_json(trajectory/'metadata.json',{'family':family,'seed':seed,'profile':profile(),'source_ids':inputs['source_ids'],'class_order':list(CLASSES),'source_class_weights':weights.tolist(),'input_dimensions':{k:a.shape[1] for k,a in x.items()},'initial_state_sha256':initial,'optimizer_fresh':True,'epochs':96,'same_trajectory_checkpoints':list(CHECKPOINTS),'outer_samples_seen':False})
            records=[];predictions=[];checkpoints=[];started=time.perf_counter()
            with scope.operation('torch_training_fixed96',source['record_ids'],X=x,y=encoded):
                for epoch in range(1,97):
                    opt_loss=train_epoch(model,optimizer,criterion,xt,target,generator)
                    train_metrics,train_scores=evaluate(model,optimizer,xt,source['source_labels'],weights);val_metrics,val_scores=evaluate(model,optimizer,vt,yval,weights)
                    record={'family':family,'seed':seed,'epoch':epoch,'source_optimization_loss':opt_loss,**{'source_'+k:train_metrics[k] for k in ['accuracy','macro_f1','balanced_accuracy','cross_entropy_source_class_weights']},**{'validation_'+k:val_metrics[k] for k in ['accuracy','macro_f1','balanced_accuracy','cross_entropy_source_class_weights']},'source_records':342,'validation_records':342,'outer_records':0}
                    records.append(record);all_metrics.append(record)
                    for role,ids,labels,scores in [('source_eval',source['record_ids'],source['source_labels'],train_scores),('inner_validation',val['record_ids'],yval,val_scores)]:
                        for rid,label,score in zip(ids,labels,scores,strict=True):predictions.append({'epoch':epoch,'role':role,'record_id':str(rid),'y_true':str(label),'y_predicted':CLASSES[int(np.argmax(score))],**{'score_'+c:float(z) for c,z in zip(CLASSES,score,strict=True)}})
                    if epoch in CHECKPOINTS:
                        before=state_hash(model);opt_before=joblib.hash(optimizer.state_dict());gen_before=generator.get_state().clone();rng_before=torch.get_rng_state().clone();old=model.training
                        audit=audit_pointwise_inference(model,vt,batch_sizes=(1,3,64),atol=1e-5,rtol=1e-5);model.train(old)
                        if before!=state_hash(model) or opt_before!=joblib.hash(optimizer.state_dict()) or not torch.equal(gen_before,generator.get_state()) or not torch.equal(rng_before,torch.get_rng_state()):raise LeakageError('Checkpoint pointwise audit changed trajectory')
                        path=trajectory/f'epoch{epoch:03}.pt';torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'generator_state':generator.get_state(),'torch_rng_state':torch.get_rng_state(),'state_sha256':state_hash(model),'source_ids':inputs['source_ids'],'protocol_sha256':sha(out/'protocol.json')},path)
                        checkpoints.append({'epoch':epoch,'path':str(path.relative_to(out)),'sha256':sha(path),'state_sha256':state_hash(model),'validation_pointwise_audit':audit,'source_metrics':train_metrics,'inner_validation_metrics':val_metrics})
                    write_json(out/'training_progress.json',{'family':family,'seed':seed,'epoch':epoch,'completed_trajectories':len(trajectory_records),'outer_evaluated':False})
            write_csv(trajectory/'epoch_metrics.csv',records);write_csv(trajectory/'epoch_predictions.csv.gz',predictions);write_json(trajectory/'checkpoints.json',checkpoints)
            trajectory_records.append({'family':family,'seed':seed,'initial_state_sha256':initial,'final_state_sha256':state_hash(model),'epochs':96,'source_count':342,'validation_count':342,'checkpoints':checkpoints,'elapsed_seconds':time.perf_counter()-started})
    write_csv(out/'epoch_metrics.csv',all_metrics);write_json(out/'trajectory_records.json',trajectory_records);write_json(out/'training_completed.json',{'at_utc':now(),'fresh_real_models':6,'checkpoint_roles':24,'epoch_metric_rows':576,'per_epoch_score_rows':393984,'outer_predictions':0,'outer_metrics':0,'deployment_selection_affected':False,'study_status':'EXPLORATORY INNER diagnostic; overlapping source/validation roles across epochs and seeds do not create independent samples'})
    return {'fresh_models':6,'epochs_each':96,'checkpoints':24,'outer_predictions':0}


def count_metrics(labels,predicted):
    matrix=np.zeros((7,7),dtype=int)
    for y,p in zip(encode(labels),encode(predicted),strict=True):matrix[y,p]+=1
    tp=np.diag(matrix).astype(float);actual=matrix.sum(1);found=matrix.sum(0)
    recalls=np.divide(tp,actual,out=np.zeros(7),where=actual!=0);f1=np.divide(2*tp,actual+found,out=np.zeros(7),where=(actual+found)!=0)
    return {'accuracy':float(tp.sum()/matrix.sum()),'macro_f1':float(f1.mean()),'balanced_accuracy':float(recalls.mean())}


def finish(out):
    out=Path(out)
    if (out/'files_sha256.json').exists():raise FileExistsError('Completed pilot may not be overwritten')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    completed=json.loads((out/'training_completed.json').read_text());protocol,inputs=verify_release(out,json.loads((out/'source_release_snapshot.json').read_text()))
    if completed.get('fresh_real_models')!=6 or completed.get('checkpoint_roles')!=24 or completed.get('outer_predictions')!=0:raise LeakageError('Incomplete fixed-budget source pilot')
    source=read_pack(out/inputs['copied_inputs']['source']['path'],True);val=read_pack(out/inputs['copied_inputs']['validation']['path']);validate_pair(source,val);yval=read_inner_truth(out/inputs['copied_inputs']['truth']['path'],val['record_ids'])
    with (out/'epoch_metrics.csv').open(encoding='utf-8-sig',newline='') as f:metrics=list(csv.DictReader(f))
    if len(metrics)!=576 or len({(r['family'],r['seed'],r['epoch']) for r in metrics})!=576:raise LeakageError('Wrong trajectory/epoch population')
    lookup={(r['family'],int(r['seed']),int(r['epoch'])):r for r in metrics};checks=0;maximum=0.;ce_maximum=0.;logit_maximum=0.;raw_rows=0;checkpoints_replayed=0
    for family in FAMILIES:
        for seed in (42,43,44):
            directory=out/'trajectories'/family/f'seed{seed}';metadata=json.loads((directory/'metadata.json').read_text());weights=np.asarray(metadata['source_class_weights']);groups={}
            with gzip.open(directory/'epoch_predictions.csv.gz','rt',encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f):groups.setdefault((int(row['epoch']),row['role']),[]).append(row);raw_rows+=1
            if len(groups)!=192:raise LeakageError('Missing source/validation epoch predictions')
            for epoch in range(1,97):
                for role,truth_ids,truth_labels,prefix in [('source_eval',source['record_ids'],source['source_labels'],'source'),('inner_validation',val['record_ids'],yval,'validation')]:
                    rows=groups[(epoch,role)];ids=[r['record_id'] for r in rows]
                    if ids!=truth_ids.tolist() or [r['y_true'] for r in rows]!=truth_labels.tolist():raise LeakageError('Repeated/changed prediction IDs or truth')
                    scores=np.asarray([[float(r['score_'+c]) for c in CLASSES] for r in rows]);pred=np.asarray(CLASSES)[scores.argmax(1)]
                    if pred.tolist()!=[r['y_predicted'] for r in rows] or not np.isfinite(scores).all():raise LeakageError('Prediction scores/classes differ')
                    values=count_metrics(truth_labels,pred);saved=lookup[(family,seed,epoch)]
                    for key,value in values.items():
                        diff=abs(value-float(saved[prefix+'_'+key]));maximum=max(maximum,diff);checks+=1
                        if diff>1e-12:raise LeakageError('Independent count metrics differ')
                    encoded=encode(truth_labels);shift=scores.max(1);logsum=shift+np.log(np.exp(scores-shift[:,None]).sum(1));loss=logsum-scores[np.arange(len(encoded)),encoded];ce=float(np.sum(loss*weights[encoded])/np.sum(weights[encoded]));diff=abs(ce-float(saved[prefix+'_cross_entropy_source_class_weights']));ce_maximum=max(ce_maximum,diff);checks+=1
                    if diff>1e-5:raise LeakageError('Independent eval CE differs')
            scalers=joblib.load(directory/'source_scalers.joblib');transformed={};validation={}
            for name,scaler in scalers.items():
                raw=row_snv(source['X_'+name]) if name=='alpha' else source['X_'+name]
                if not np.allclose(scaler.mean_,raw.mean(0),atol=1e-12,rtol=1e-12) or not np.allclose(scaler.var_,raw.var(0),atol=1e-12,rtol=1e-12):raise LeakageError('Stored scaler not current source statistics')
                transformed[name]=torch.as_tensor(scaler.transform(raw).astype(np.float32));validation[name]=torch.as_tensor(scaler.transform(row_snv(val['X_'+name]) if name=='alpha' else val['X_'+name]).astype(np.float32))
            stored=json.loads((directory/'checkpoints.json').read_text())
            if [r['epoch'] for r in stored]!=list(CHECKPOINTS):raise LeakageError('Checkpoint budget changed')
            for item in stored:
                path=out/item['path']
                if sha(path)!=item['sha256']:raise LeakageError('Checkpoint bytes changed')
                checkpoint=torch.load(path,map_location='cpu',weights_only=False);model=UpgradeNet(family,profile(),metadata['input_dimensions']);model.load_state_dict(checkpoint['model_state_dict']);model.eval()
                if state_hash(model)!=item['state_sha256'] or checkpoint['source_ids']!=inputs['source_ids']:raise LeakageError('Stored checkpoint/source provenance differs')
                before=state_hash(model)
                for role,x in [('source_eval',transformed),('inner_validation',validation)]:
                    with torch.no_grad():scores=model(x).cpu().numpy()
                    expected=np.asarray([[float(r['score_'+c]) for c in CLASSES] for r in groups[(item['epoch'],role)]],dtype=np.float32);diff=float(np.max(np.abs(scores-expected)));logit_maximum=max(logit_maximum,diff);checks+=1
                    if not np.allclose(scores,expected,atol=1e-5,rtol=1e-5):raise LeakageError('Checkpoint replay differs')
                if state_hash(model)!=before:raise LeakageError('Checkpoint replay changed model state')
                checkpoints_replayed+=1
    if raw_rows!=393984 or checkpoints_replayed!=24:raise LeakageError('Unexpected diagnostic denominator')
    # Descriptive curves across3 training seeds: not independent biological repeats.
    summary=[]
    for family in FAMILIES:
        for epoch in range(1,97):
            rows=[lookup[(family,seed,epoch)] for seed in (42,43,44)];record={'family':family,'epoch':epoch,'seed_count':3}
            for key in ['source_optimization_loss','source_accuracy','source_macro_f1','source_balanced_accuracy','source_cross_entropy_source_class_weights','validation_accuracy','validation_macro_f1','validation_balanced_accuracy','validation_cross_entropy_source_class_weights']:
                vals=np.asarray([float(r[key]) for r in rows]);record[key+'_mean']=float(vals.mean());record[key+'_sample_sd']=float(vals.std(ddof=1))
            summary.append(record)
    write_csv(out/'epoch_seed_summary.csv',summary);write_csv(out/'checkpoint_seed_summary.csv',[r for r in summary if r['epoch'] in CHECKPOINTS])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for family in FAMILIES:
        rows=[r for r in summary if r['family']==family];epochs=[r['epoch'] for r in rows];fig,axs=plt.subplots(1,2,figsize=(12,4.5))
        for key,label,color in [('source_cross_entropy_source_class_weights','Source eval CE','#2563a6'),('validation_cross_entropy_source_class_weights','Inner validation CE','#c86a1b')]:
            avg=np.asarray([r[key+'_mean'] for r in rows]);sd=np.asarray([r[key+'_sample_sd'] for r in rows]);axs[0].plot(epochs,avg,label=label,color=color);axs[0].fill_between(epochs,avg-sd,avg+sd,color=color,alpha=.15)
        for key,label,color in [('validation_accuracy','Validation accuracy','#2563a6'),('validation_macro_f1','Validation Macro-F1','#c86a1b'),('validation_balanced_accuracy','Validation balanced accuracy','#327c45')]:
            avg=np.asarray([r[key+'_mean'] for r in rows]);sd=np.asarray([r[key+'_sample_sd'] for r in rows]);axs[1].plot(epochs,avg,label=label,color=color);axs[1].fill_between(epochs,avg-sd,avg+sd,color=color,alpha=.12)
        for ax in axs:
            for epoch in CHECKPOINTS:ax.axvline(epoch,color='grey',alpha=.15,linewidth=.8)
            ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
        axs[0].set_ylabel('Eval CE (fixed source class weights)');axs[1].set_ylabel('Inner validation score');axs[1].set_ylim(0,1.05)
        fig.suptitle(f'{family}: fixed outer00.inner00, three fresh training seeds');fig.text(.5,.005,'342 source /342 inner validation; shaded sample SD across seeds, not biological CI; no outer evaluation.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(out/f'{family}_learning_curve.png',dpi=180);fig.savefig(out/f'{family}_learning_curve.pdf');plt.close(fig)
    # Audit actual learned scopes: six trajectories, fifteen scalers, six NN fits.
    events=[json.loads(line) for line in (out/'fit_scope_events.jsonl').read_text().splitlines()];starts=[r for r in events if r['event']=='fit_start'];done=[r for r in events if r['event']=='fit_completed']
    if len(starts)!=21 or len(done)!=21 or any(set(r['fit_sample_ids'])!=set(inputs['source_ids']) for r in starts+done):raise LeakageError('Actual source fit scope differs')
    write_json(out/'independent_validation.json',{'status':'SOURCE_ONLY_PILOT_REPLAY_PASS_EXPLORATORY','count_or_CE_checks':checks,'maximum_count_metric_difference':maximum,'maximum_CE_difference':ce_maximum,'maximum_checkpoint_logit_difference':logit_maximum,'checkpoints_replayed':24,'per_epoch_prediction_rows':raw_rows,'unique_source_records':342,'unique_validation_records':342,'fresh_trajectories':6,'source_scalers':15,'NN_fixed96_operations':6,'actual_completed_learning_operations':21,'new_fits_in_this_review':0,'outer_predictions':0,'new_independent_specimens':0,'deployment_selection_affected':False})
    (out/'README.md').write_text('# 酸枣仁源内学习曲线pilot\n\n仅原outer00.inner00：342source／342内层validation；七类ZY01—ZY07。两固定家族、全部seed42/43/44，每个fresh模型96轮，固定12/24/48/96保存24快照。没有外层预测、没有部署选参反馈。\n\n`protocol.json`及root source release在训练前锁定；`epoch_metrics.csv`是576条轨迹×epoch记录，`epoch_seed_summary.csv`为描述均值／SD，`checkpoint_seed_summary.csv`不选择胜者。每轨迹保存全epoch的source-eval及inner-validation预测，合计393984行；重复epoch／seed不增加真实N。源预处理／类别权重／BN学习仅342source。验证全程eval/no_grad，逐轮参数／BN／optimizer／grad／RNG检查不变，24快照全342validation逐点／批量／换序检查通过。\n\n`independent_validation.json`复算全部指标、CE与24快照，当前属于历史开发数据上的源内诊断，不是总体OOF或新批次准确率。旧96.02%与本结果分开。未知批次、历史曝光、参考波形及单位限制仍在；Python read hook不是OS沙箱。\n\n源优化loss包含训练态dropout与逐batch变化权重；源eval CE和验证eval CE在固定checkpoint、相同源类别权重下计算。所有种子和负结果保留，不能用验证曲线挑最大值称未见准确率；进一步参数选择须另锁协议。\n')
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()};write_json(out/'files_sha256.json',files)
    return {'actual_source_models':6,'checkpoints_replayed':24,'outer_predictions':0,'source_fit_operations':21,'files_indexed':len(files)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','train','finish']);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--release',type=Path);args=parser.parse_args()
    if args.stage=='train' and not args.release:parser.error('train requires a root source release')
    result=prepare(args.output.resolve()) if args.stage=='prepare' else train(args.output.resolve(),args.release) if args.stage=='train' else finish(args.output.resolve())
    print(json.dumps(result,ensure_ascii=False))
