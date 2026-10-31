#!/usr/bin/env python3
"""Export immutable INNER source optimization losses; no training or inference.

Training loss is the recorded batch-size-weighted mean of train-mode batch CE,
not validation loss or fixed-checkpoint full-source CE. Preserve old archives.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import statistics
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/upgrade-20261031-v1/neural'
FAMILIES=('cnn_alpha','cnn_n','dual_alpha_mcuve','fusion_all_four','fusion_alpha_n','fusion_time_frequency','resnet_alpha','se_alpha')
PROFILES={'small12':12,'regularized24':24}
DISPLAY={'cnn_alpha':'Alpha CNN','cnn_n':'Refractive-index CNN','dual_alpha_mcuve':'Alpha + MC-point MLP','fusion_all_four':'Four representations','fusion_alpha_n':'Alpha + refractive index','fusion_time_frequency':'Time + frequency','resnet_alpha':'Alpha ResNet1D','se_alpha':'Alpha SE-ResNet1D'}


def sha(raw):return hashlib.sha256(raw).hexdigest()
def json_write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_csv(path,rows):
    if not rows:raise ValueError('Cannot export an empty table')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def verified_json(raw,expected):
    if sha(raw)!=expected:raise ValueError('Archived metadata SHA mismatch')
    return json.loads(raw)


def validate_record(d):
    family=d['family'];profile=d['profile']['id'];seed=d['seed'];split=d['split_id'];loss=d['loss_curve'];ids=d['source_ids']
    valid_splits={f'outer{o:02}.inner{i:02}' for o in range(5) for i in range(2)}
    if family not in FAMILIES or profile not in PROFILES or seed not in (42,43,44) or split not in valid_splits:raise ValueError('Unexpected source curve identity')
    if d['epochs']!=PROFILES[profile] or len(loss)!=PROFILES[profile]:raise ValueError('Epoch count does not match frozen profile')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in loss):raise ValueError('Nonfinite or invalid CE')
    if len(ids)!=342 or len(set(ids))!=342 or any(not isinstance(v,str) for v in ids):raise ValueError('Source records must be342 unique IDs')
    return split,family,profile,seed


def validate_population(records):
    keys=[validate_record(d) for d in records]
    expected={(f'outer{o:02}.inner{i:02}',f,p,s) for o in range(5) for i in range(2) for f in FAMILIES for p in PROFILES for s in (42,43,44)}
    if len(keys)!=480 or len(set(keys))!=480 or set(keys)!=expected:raise ValueError('Incomplete or duplicate curve population')


def integrity_probes():
    fixture={'family':'cnn_alpha','profile':{'id':'small12'},'seed':42,'split_id':'outer00.inner00','epochs':12,'source_ids':[f'fixture{i}' for i in range(342)],'loss_curve':[2.-i*.01 for i in range(12)]}
    validate_record(fixture);checks=['valid synthetic metadata accepted']
    cases=[]
    d={**fixture,'loss_curve':[float('nan')]+fixture['loss_curve'][1:]};cases.append(('NaN CE rejected',lambda:validate_record(d)))
    short={**fixture,'loss_curve':fixture['loss_curve'][:-1]};cases.append(('Truncated epoch history rejected',lambda:validate_record(short)))
    dup={**fixture,'source_ids':['same']*342};cases.append(('Repeated source IDs rejected',lambda:validate_record(dup)))
    cases.append(('Changed archived bytes rejected',lambda:verified_json(b'{}','0'*64)))
    cases.append(('Duplicate/incomplete population rejected',lambda:validate_population([fixture]*480)))
    for label,operation in cases:
        try:operation()
        except ValueError:checks.append(label)
        else:raise AssertionError('Integrity probe failed: '+label)
    return {'status':'PASS_FOR_EXPLICIT_INTEGRITY_PROBES','checks':checks,'synthetic_fixtures_are_not_training_results':True}


def collect():
    index=json.loads((BASE/'files_sha256.json').read_text());manifest=json.loads((BASE/'archive_validation.json').read_text())
    proofs={(r.get('archive'),r.get('member')):r for r in manifest['records'] if r.get('archive') and r.get('member')}
    records=[];sources={};member_proofs=[]
    for outer in range(5):
        p=BASE/f'inner_outer_{outer:02}_evidence.tar.gz';digest=sha(p.read_bytes())
        if digest!=index[p.name]:raise ValueError('Archive SHA differs from frozen index')
        sources[str(p.relative_to(ROOT))]=digest
        with tarfile.open(p,'r:gz') as t:
            selected=[m for m in t.getmembers() if m.name.endswith('/training.json')]
            if len(selected)!=96:raise ValueError('Expected96 source metadata members per archive')
            for member in selected:
                raw=t.extractfile(member).read();proof=proofs[(str(p.relative_to(ROOT)),member.name)];d=verified_json(raw,proof['sha256'])
                key=validate_record(d)
                if key[0]!=member.name.split('/')[1] or not key[0].startswith(f'outer{outer:02}.'):raise ValueError('Archive/member/source context mismatch')
                records.append(d);member_proofs.append({'archive':str(p.relative_to(ROOT)),'member':member.name,'sha256':proof['sha256'],'source_id_count':len(d['source_ids']),'source_ids_sha256':sha(json.dumps(d['source_ids'],separators=(',',':')).encode())})
    validate_population(records)
    return records,sources,member_proofs


def export(out):
    started=datetime.now(timezone.utc).isoformat();probes=integrity_probes();records,archives,members=collect()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Refuse to overwrite a nonempty diagnostic directory')
    out.mkdir(parents=True,exist_ok=True)
    curves=[];long=[];groups=defaultdict(list)
    for d in sorted(records,key=lambda z:(z['split_id'],z['family'],z['profile']['id'],z['seed'])):
        split,family,profile,seed=validate_record(d);loss=d['loss_curve'];prior=statistics.mean(loss[-6:-3]);last=statistics.mean(loss[-3:]);drop=prior-last
        key={'split_id':split,'family':family,'profile':profile,'seed':seed}
        relative=Path('curves')/split/family/profile/f'seed{seed}.csv'
        single=[{'epoch':i+1,'source_train_batch_mean_CE':v} for i,v in enumerate(loss)];write_csv(out/relative,single)
        for row in single:long.append({**key,**row})
        curves.append({**key,'epochs':len(loss),'source_records':342,'first_epoch_loss':loss[0],'last_epoch_loss':loss[-1],'previous3_mean':prior,'last3_mean':last,'tail_drop':drop,'tail_relative_drop':drop/prior,'curve_csv':str(relative)})
        groups[(family,profile)].append(d)
    summaries=[];epochs=[]
    for (family,profile),items in sorted(groups.items()):
        cs=[r for r in curves if r['family']==family and r['profile']==profile]
        summaries.append({'family':family,'profile':profile,'curves':len(items),'first_loss_mean':statistics.mean(r['first_epoch_loss'] for r in cs),'first_loss_sample_sd':statistics.stdev(r['first_epoch_loss'] for r in cs),'final_loss_mean':statistics.mean(r['last_epoch_loss'] for r in cs),'final_loss_sample_sd':statistics.stdev(r['last_epoch_loss'] for r in cs),'tail_relative_drop_median':statistics.median(r['tail_relative_drop'] for r in cs),'tail_decrease_curves':sum(r['tail_drop']>0 for r in cs)})
        for i in range(PROFILES[profile]):
            vals=[d['loss_curve'][i] for d in items];epochs.append({'family':family,'profile':profile,'epoch':i+1,'curves':len(vals),'loss_mean':statistics.mean(vals),'loss_sample_sd':statistics.stdev(vals)})
    write_csv(out/'curve_summary.csv',curves);write_csv(out/'epoch_loss_long.csv',long);write_csv(out/'group_summary.csv',summaries);write_csv(out/'group_epoch_summary.csv',epochs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for profile in PROFILES:
        fig,axs=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
        for ax,family in zip(axs.flat,FAMILIES):
            rows=[r for r in epochs if r['profile']==profile and r['family']==family];x=[r['epoch'] for r in rows];mean=[r['loss_mean'] for r in rows];sd=[r['loss_sample_sd'] for r in rows]
            ax.plot(x,mean,color='#2563a6',linewidth=1.8,label='Mean (30 source trajectories)');ax.fill_between(x,[max(0,a-b) for a,b in zip(mean,sd)],[a+b for a,b in zip(mean,sd)],color='#2563a6',alpha=.18,label='+/- sample SD')
            ax.set_title(DISPLAY[family]);ax.grid(alpha=.2);ax.set_xlabel('Epoch');ax.set_ylabel('Source batch optimization CE');ax.set_xlim(1,PROFILES[profile]);ax.set_ylim(bottom=0)
        handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.02))
        fig.suptitle(f'{profile}: recorded INNER source optimization losses (not validation loss)',fontsize=15)
        fig.text(.5,.003,'Overlapping folds and training seeds; SD is descriptive, not a biological confidence interval.',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(out/f'{profile}_source_loss.png',dpi=180);fig.savefig(out/f'{profile}_source_loss.pdf');plt.close(fig)
    validation={'status':'VERIFIED_EXISTING_SOURCE_LOSS_EXPORT_NOT_NEW_TRAINING','source_curves':480,'source_records_per_curve':342,'long_rows':len(long),'family_profile_groups':16,'group_epoch_rows':len(epochs),'all480_tail_means_decreased':sum(r['tail_drop']>0 for r in curves)==480,'archive_member_SHA_checks':len(members),'archive_SHA_checks':len(archives),'per_epoch_validation_metrics_available':False,'new_model_fits':0,'model_inference_calls':0,'outer_metrics_read':False,'scientific_status':'POSTHOC_SOURCE_DIAGNOSTIC_EXPLORATORY','integrity_probes':probes}
    json_write(out/'validation.json',validation);json_write(out/'input_archive_sha256.json',archives);json_write(out/'member_provenance.json',members)
    readme='''# 酸枣仁既有源训练损失导出

仅复用冻结神经实验的480条inner source训练损失，不训练、不预测、不读取outer成绩。每条342个source ID、12／24轮；5tar和480成员SHA均实际核对。数据是训练态各batch优化CE的人数加权均值，含dropout与变化中的权重，不是验证loss或固定checkpoint的全源eval loss。

- `curves/`：480份逐曲线CSV；`epoch_loss_long.csv`合并8640条epoch记录。
- `curve_summary.csv`：480条首尾／末段下降描述；`group_summary.csv`：16组描述均值及样本SD。
- `group_epoch_summary.csv`：288条组×epoch均值／SD；两个profile各一张PNG及PDF。
- `member_provenance.json`：原tar／成员／SHA与源ID数；`validation.json`记录完整性负例与边界。

30条／组来自重叠五外折、两内折及三seed，不是30组生物独立重复；阴影为描述性SD，不是置信区间。不同profile改变类别权重及多个参数，曲线不用于跨profile排最优。没有每epoch验证曲线与中途权重，不能从末段loss下降断言延长训练一定提高泛化；后续实验仍NOT_RUN。

详见`docs/DEPLOYMENT_NEURAL_DIAGNOSTIC_20261031.md`。复现需要新空输出目录：

```bash
.runtime/bin/python scripts/export_deployment_source_curves.py --output outputs/source-curves-reproduction-new
```

有限SHA和schema检查不证明绝对零泄漏，不改变历史曝光／批次身份。没有新增真实样品或校准／分子计算。
'''
    (out/'README.md').write_text(readme)
    json_write(out/'command.json',{'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'argv':sys.argv,'script':str(Path(__file__).relative_to(ROOT)),'script_sha256':sha(Path(__file__).read_bytes()),'versions':{k:importlib.metadata.version(k) for k in ['matplotlib','numpy']},'exit_status':'completed','new_model_fits':0})
    # Re-read exported per-curve numeric cells and compare directly to metadata.
    check_count=0
    for d in records:
        split,family,profile,seed=validate_record(d);p=out/'curves'/split/family/profile/f'seed{seed}.csv'
        with p.open(encoding='utf-8-sig',newline='') as f:rs=list(csv.DictReader(f))
        if [float(r['source_train_batch_mean_CE']) for r in rs]!=d['loss_curve']:raise ValueError('CSV readback differs from immutable loss values')
        check_count+=len(rs)
    json_write(out/'csv_readback.json',{'numeric_loss_cells_checked':check_count,'maximum_difference':0.,'new_model_fits':0})
    files={str(p.relative_to(out)):sha(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()}
    json_write(out/'files_sha256.json',files)
    print(json.dumps({'files_indexed':len(files),'curves':480,'numeric_loss_cells_checked':check_count,'new_fits':0,'output':str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();export(args.output.resolve())
