"""Bounded, registered development pilot: conservative external THz pretraining.

Historical inputs/results remain immutable. Prediction and truth scoring are
separate child processes; only current-source OOF predictions select epochs.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import gc
import gzip
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "cross-plant-conservative-pilot-20261104-v1"
OLD = ROOT / "reports/cross-plant-deep-transfer-20261031-v1/neural"
CLASSES = [f"ZY{i:02d}" for i in range(1, 8)]
SEEDS = (42, 43, 44)
ARMS = ("A", "Aextra_L2SP", "B_HQ_TM_L2SP")
CPU_CAP, WALL_CAP, RSS_CAP = 600., 1800., int(1.5 * 1024**3)
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def objhash(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, v, *, immutable=True):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if immutable and p.exists():
        raise FileExistsError(str(p))
    tmp = p.with_name(p.name+".writing")
    with tmp.open("w") as f:
        json.dump(v, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def dependencies():
    for name in THREADS:
        if os.environ.get(name) != "1":
            raise ValueError("Single-thread environment is required before importing numerical libraries")
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler
    from herb_spectra import cross_plant_transfer as old
    from herb_spectra import leakage_guard as guard
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return np, torch, StandardScaler, old, guard


def protocol():
    base = ["src/herb_spectra/cross_plant_transfer.py", "src/herb_spectra/upgrade_neural.py",
            "src/herb_spectra/leakage_guard.py", "src/herb_spectra/classification.py"]
    sources = {p: sha(ROOT/p) for p in base}
    sources[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    return {"run_id": RUN_ID, "status": "REGISTERED_BEFORE_TRAINING",
        "study_status": "DEVELOPMENT_NOT_NEW_CONFIRMATION",
        "physical_status": "EXPLORATORY_SPECTRAL_SHAPE_UNITS_INSTRUMENT_BATCH_ANCESTRY_NOT_FULLY_VERIFIED",
        "target_N": 855, "classes": CLASSES, "seeds": list(SEEDS), "arms": list(ARMS),
        "external_observations": {"huangqi": 403, "tianma": 405},
        "external_label_spaces_separate": True, "ginseng_excluded": "upstream biological ancestry uncertain",
        "molecular_or_literature_records_are_training_samples": False,
        "axis": {"min_THz": .21, "max_THz": 1.49, "points": 128,
                 "interpolation": "existing fixed pointwise linear, no extrapolation"},
        "preprocessing": "pointwise SNV; independently source-fitted target/external StandardScaler",
        "architecture": {"width": 12, "blocks": 2, "dropout": .35, "residual": True, "attention": False},
        "optimizer": {"learning_rate": .0005, "weight_decay": .001, "batch": 64},
        "pretraining": {"steps": 120, "batch": 32, "A": [], "Aextra_L2SP": ["target"],
                        "B_HQ_TM_L2SP": ["huangqi", "tianma"]},
        "L2SP": {"lambda": .001, "arms": list(ARMS[1:]), "parameters": "encoder named_parameters only",
                 "anchor": "actual pretrained weights; BN buffers reset before target fitting",
                 "penalty": "lambda * sum of squared parameter deviations"},
        "epoch_selection": {"candidates": list(range(1, 49)), "selection_arm": "A",
            "source_rule": "within each original outer: pool original two inner queries per seed, average all three seeds; MacroF1 then BA then Accuracy then earliest epoch",
            "all_source_epoch_locks_before_any_outer_prediction": True,
            "same_target_epochs_for_all_arms_and_seeds": True},
        "extra_compute_control": "B-Aextra compares external vs target-source supervised pretraining under same steps and L2SP; heads differ so not exact FLOP matching",
        "historical_96_02_and_87_21_are_not_direct_same_protocol_baselines": True,
        "resource_limits": {"total_actual_child_cpu_seconds": CPU_CAP, "wall_seconds": WALL_CAP,
            "threads": 1, "best_effort_RSS_bytes": RSS_CAP, "virtual_AS_GiB": 10,
            "concurrent_root_DFT": True, "wall_speedup_comparison_allowed": False},
        "prepared_root": str(OLD.relative_to(ROOT)),
        "preparation_sha256": sha(OLD/"preparation.json"),
        "split_path": "reports/staged-nested-optimization/2026-10-31/core/splits.json",
        "split_sha256": sha(ROOT/"reports/staged-nested-optimization/2026-10-31/core/splits.json"),
        "source_sha256": sources}


def verify_inputs(out, np, old):
    registered = read(out/"protocol.json")
    if registered != protocol():
        raise ValueError("Registered source/protocol/input metadata changed")
    prep = read(OLD/"preparation.json")
    if sha(ROOT/old.SPLITS)!=old.SPLITS_SHA:
        raise ValueError('Original fixed split bytes changed')
    splits={f['fold']:f for f in read(ROOT/old.SPLITS)}
    for c in prep["contexts"]:
        fold=splits[c['outer_fold']]
        reference=fold['inner_folds'][c['inner_fold']] if c['stage']=='inner' else fold
        expected_source=reference['train_sample_ids']
        expected_query=reference['validation_sample_ids' if c['stage']=='inner' else 'test_sample_ids']
        if c['source_ids']!=expected_source or c['query_ids']!=expected_query:
            raise ValueError('Prepared roles differ from original fixed memberships')
        for role in ("source", "query"):
            if sha(OLD/c[role+"_pack"]) != c[role+"_sha256"]:
                raise ValueError("Prepared input changed")
            pack = old.read_pack(OLD/c[role+"_pack"], role == "source")
            if pack["record_ids"].tolist() != c[role+"_ids"]:
                raise ValueError("Fixed split order changed")
    auxiliary = {}
    for name in ("huangqi", "tianma"):
        rec = prep["auxiliary"][name]
        if sha(OLD/rec["path"]) != rec["sha256"]:
            raise ValueError("External prepared input changed")
        auxiliary[name] = old.read_pack(OLD/rec["path"], True)
    # Directly inspect canonical axis/column exports; no target label is loaded.
    audit = {}
    for name, (path, expected) in old.SOURCES.items():
        if name == "ginseng":
            continue
        if sha(ROOT/path) != expected:
            raise ValueError("Canonical exported data changed: "+name)
        with np.load(ROOT/path, allow_pickle=False) as a:
            axis = a["axis_alpha" if name == "zaoren" else "axis"]
            x = a["X_alpha" if name == "zaoren" else "X"]
            if not np.isfinite(x).all() or not np.all(np.diff(axis) > 0):
                raise ValueError("Invalid alpha/frequency export")
            audit[name] = {"source_sha256": expected, "alpha_shape": list(x.shape),
                "native_frequency_range_THz": [float(axis.min()), float(axis.max())],
                "native_axis_sha256": old.array_fingerprint(axis), "column_quantity": "existing alpha export",
                "absolute_alpha_unit_instrument_and_batches_verified": False,
                "rows_are_new_downloaded_samples": False}
    return prep, auxiliary, audit


class BudgetStop(RuntimeError):
    pass


def select_epoch(records):
    if len(records) != 48 or [r["epoch"] for r in records] != list(range(1, 49)):
        raise ValueError("All predeclared epochs required")
    return min(records, key=lambda r: (-r["macro_f1"], -r["balanced_accuracy"], -r["accuracy"], r["epoch"]))


def assert_membership(source_ids, query_ids):
    if len(set(source_ids)) != len(source_ids) or len(set(query_ids)) != len(query_ids) or set(source_ids) & set(query_ids):
        raise ValueError("Duplicate or overlapping source/query identities")


def encoder_l2_sp(encoder, anchor):
    parameters=dict(encoder.named_parameters())
    if set(parameters)!=set(anchor) or any(parameters[k].shape!=anchor[k].shape for k in parameters):
        raise ValueError('L2-SP anchor does not match current encoder parameters')
    return sum((value-anchor[name].detach()).square().sum() for name,value in parameters.items())


def verified_truth(context, old):
    path=OLD/context['truth_path']
    if sha(path)!=context['truth_sha256']:
        raise ValueError('Frozen scoring labels changed')
    return old._truth(path)


def fit(source, query, auxiliary, arm, epochs, seed, directory, split_id, np, torch, StandardScaler, old, guard,
        *, return_epoch_predictions=False):
    old.validate_pack(source, True)
    old.validate_pack(query, False)
    assert_membership(source["record_ids"].tolist(), query["record_ids"].tolist())
    if seed not in SEEDS or arm not in ARMS or not 1 <= epochs <= 48:
        raise ValueError("Unregistered training configuration")
    directory.mkdir(parents=True, exist_ok=False)
    ids = source["record_ids"]
    scope = guard.FitScope(ids, split_id+"."+arm, RUN_ID, directory/"fit_scope.jsonl", code_version=sha(__file__))
    scaler = scope.fit(StandardScaler(), old.row_snv(source["X"]), sample_ids=ids, component="target_source_scaler")
    x = scaler.transform(old.row_snv(source["X"])).astype(np.float32)
    y, weights, _ = old._weights(source["source_labels"], CLASSES)
    model = old.TransferNet({"width":12,"blocks":2,"dropout":.35,"residual":True,"attention":False}, seed)
    initial_encoder, initial_head = old.state_hash(model.encoder), old.state_hash(model.head)
    pools = [] if arm == "A" else ["target"] if arm == "Aextra_L2SP" else ["huangqi", "tianma"]
    prepared, heads, pool_scopes = {}, torch.nn.ModuleDict(), {}
    for i, name in enumerate(pools):
        pool = source if name == "target" else auxiliary[name]
        if len(pool['record_ids'])<32:raise ValueError('Fixed pretraining batch requires at least 32 source records')
        assert_membership(pool["record_ids"].tolist(), [] if name == "target" else ids.tolist()+query["record_ids"].tolist())
        classes = CLASSES if name == "target" else np.unique(pool["source_labels"]).tolist()
        own = guard.FitScope(pool["record_ids"], split_id+".pretrain."+name, RUN_ID,
            directory/"fit_scope.jsonl", code_version=sha(__file__))
        if name == "target":
            z = x
        else:
            transform = own.fit(StandardScaler(), old.row_snv(pool["X"]), sample_ids=pool["record_ids"], component="external_source_scaler")
            z = transform.transform(old.row_snv(pool["X"])).astype(np.float32)
            np.savez_compressed(directory/(name+"-scaler.npz"), mean=transform.mean_, scale=transform.scale_)
        py, pw, _ = old._weights(pool["source_labels"], classes)
        prepared[name] = (pool, z, py, torch.as_tensor(pw, dtype=torch.float32), torch.Generator().manual_seed(seed+30000+i))
        pool_scopes[name] = own
        with torch.random.fork_rng():
            torch.manual_seed(seed+20000+i)
            heads[name] = torch.nn.Linear(16, len(classes))
    target_schedule, pre_schedule, losses, epoch_scores = [], [], [], []
    start = time.process_time()
    with gzip.open(directory/"batch_trace.jsonl.gz", "xt") as trace:
        def event(stage, name, step, chosen, pool, values, target, loss):
            actual = pool["record_ids"][chosen]
            allowed = pool_scopes[name].allowed_ids if stage == "pretrain" else scope.allowed_ids
            if len(set(actual.tolist())) != len(actual) or set(actual.tolist())-set(allowed):
                raise ValueError("Actual optimization inputs escaped allowed current source")
            trace.write(json.dumps({"stage":stage,"head":name,"step":step,"fit_sample_ids":actual.tolist(),
                "X_fingerprint":guard.array_fingerprint(values),"y_fingerprint":guard.array_fingerprint(target),
                "loss":float(loss)}, separators=(",", ":"))+"\n")
        if pools:
            optimizer = torch.optim.Adam([*model.encoder.parameters(), *heads.parameters()], lr=.0005, weight_decay=.001)
            torch.manual_seed(seed+40000)
            for step in range(120):
                name = pools[step % len(pools)]
                pool, z, py, pw, generator = prepared[name]
                chosen = torch.randperm(len(z), generator=generator)[:32].numpy()
                model.encoder.train()
                with pool_scopes[name].operation("actual_pretraining_batch", pool["record_ids"][chosen], X=z[chosen], y=py[chosen]):
                    optimizer.zero_grad(set_to_none=True)
                    logits = heads[name](model.encoder(torch.as_tensor(z[chosen])))
                    loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(py[chosen]), weight=pw)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite pretraining loss")
                    loss.backward(); optimizer.step()
                event("pretrain",name,step,chosen,pool,z[chosen],py[chosen],loss.detach())
                pre_schedule.append({"pool":name,"indices":chosen.tolist()})
        anchor = {name: value.detach().clone() for name,value in model.encoder.named_parameters()}
        pretrained = old.state_hash(model.encoder)
        bn_before = {n: v.detach().cpu().numpy().tolist() for n,v in model.encoder.named_buffers()}
        for layer in model.encoder.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                layer.reset_running_stats()
        bn_after = {n: v.detach().cpu().numpy().tolist() for n,v in model.encoder.named_buffers()}
        if old.state_hash(model.head) != initial_head:
            raise ValueError("External pretraining changed the fresh paired target head")
        reset_encoder = old.state_hash(model.encoder)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0005, weight_decay=.001)
        torch.manual_seed(seed+50000)
        generator = torch.Generator().manual_seed(seed+60000)
        q = scaler.transform(old.row_snv(query["X"])).astype(np.float32)
        step = 0
        for epoch in range(1, epochs+1):
            order = torch.randperm(len(x), generator=generator).numpy(); target_schedule.append(order.tolist())
            model.train(); total = 0.
            for left in range(0, len(x), 64):
                chosen = order[left:left+64]
                with scope.operation("actual_target_batch", ids[chosen], X=x[chosen], y=y[chosen]):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model({"alpha":torch.as_tensor(x[chosen])})
                    loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(y[chosen]), weight=torch.as_tensor(weights,dtype=torch.float32))
                    penalty = encoder_l2_sp(model.encoder,anchor)
                    if arm != "A":
                        loss = loss + .001 * penalty
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite target loss")
                    loss.backward(); optimizer.step()
                event("target","ZY",step,chosen,source,x[chosen],y[chosen],loss.detach()); step += 1
                total += float(loss.detach())*len(chosen)
            losses.append(total/len(x))
            model.eval()
            if return_epoch_predictions:
                with torch.no_grad():
                    epoch_scores.append(model({"alpha":torch.as_tensor(q)}).cpu().numpy())
            # Mutable last-epoch recovery file belongs only to this new run.
            temporary = directory/"last-state.pt.writing"
            torch.save(model.state_dict(), temporary)
            os.replace(temporary, directory/"last-state.pt")
            save(directory/"progress.json", {"epoch":epoch,"epochs":epochs,"target_steps":step,"arm":arm,"seed":seed}, immutable=False)
        model.eval()
        with torch.no_grad():
            scores = model({"alpha":torch.as_tensor(q)}).cpu().numpy()
        inference = guard.audit_pointwise_inference(model,{"alpha":q if not return_epoch_predictions else q[:7]},batch_sizes=(1,3),atol=1e-5,rtol=1e-5)
    np.savez_compressed(directory/"target-scaler.npz",mean=scaler.mean_,scale=scaler.scale_)
    np.savez_compressed(directory/"predictions.npz",record_ids=query["record_ids"],scores=scores,
        epoch_scores=np.asarray(epoch_scores, dtype=np.float32))
    metadata = {"run_id":RUN_ID,"status":"FITTED_PREDICTION_LOCKED","arm":arm,"seed":seed,"epochs":epochs,
        "split_id":split_id,"source_ids":ids.tolist(),"query_ids":query["record_ids"].tolist(),
        "source_fingerprint":guard.source_fingerprint(source["X"],source["source_labels"]),
        "initial_encoder_sha256":initial_encoder,"initial_ZY_head_sha256":initial_head,
        "pretrained_encoder_sha256":pretrained,"reset_encoder_sha256":reset_encoder,
        "anchor_sha256":old.state_hash(torch.nn.ParameterDict({k.replace('.', '__'):torch.nn.Parameter(v,requires_grad=False) for k,v in anchor.items()})),
        "BN_before_reset":bn_before,"BN_after_reset":bn_after,
        "target_batch_schedule_sha256":objhash(target_schedule),"pretrain_schedule_sha256":objhash(pre_schedule),
        "pretraining_pool_steps":{p:sum(v['pool']==p for v in pre_schedule) for p in pools},
        "pretraining_unique_gradient_records":{p:len({int(i) for step in pre_schedule if step['pool']==p for i in step['indices']}) for p in pools},
        "target_steps":step,"target_epoch_loss":losses,"actual_CPU_seconds_inside_fit":time.process_time()-start,
        "encoder_L2SP_lambda":0. if arm=="A" else .001,"inference_audit":inference,
        "fitted_state_sha256":old.state_hash(model),"prediction_sha256":sha(directory/"predictions.npz"),
        "model_sha256":sha(directory/"last-state.pt"),"query_labels_received":False,
        "OS_truth_sandbox":False,"numerical_features_contain_ID_paths_or_chemical_descriptors":False}
    save(directory/"training.json",metadata)
    del model,optimizer,heads,prepared,anchor
    gc.collect()
    return metadata


def worker(out):
    start = time.process_time()
    np,torch,StandardScaler,old,guard = dependencies()
    save(out/'environment.json',{'python':sys.version,'executable':sys.executable,
        'numpy':np.__version__,'torch':torch.__version__,
        'other_distributions':{n:importlib.metadata.version(n) for n in ('scipy','scikit-learn','joblib')},
        'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
        'device':'cpu','thread_environment':{n:os.environ[n] for n in THREADS},'concurrent_root_DFT':True})
    def stop(sig,_):
        raise BudgetStop("Actual pilot resource signal "+str(sig))
    for sig in (signal.SIGTERM,signal.SIGXCPU,signal.SIGINT):
        signal.signal(sig,stop)
    prep,aux,audit = verify_inputs(out,np,old)
    save(out/"canonical-data-audit.json",audit)
    state={"status":"PARTIAL","started_at_utc":utc(),"study_status":"DEVELOPMENT_NOT_NEW_CONFIRMATION","completed_source_fits":0,"completed_outer_fits":0}
    forbidden=[OLD/c['truth_path'] for c in prep['contexts']]
    try:
        for c in prep['contexts']:
            if c['stage']!='inner':continue
            source=old.read_pack(OLD/c['source_pack'],True);query=old.read_pack(OLD/c['query_pack'],False)
            for seed in SEEDS:
                d=out/'source_models'/c['split_id']/f'seed{seed}'
                with old.forbid_named_truth(forbidden):
                    fit(source,query,aux,'A',48,seed,d,c['split_id'],np,torch,StandardScaler,old,guard,return_epoch_predictions=True)
                state['completed_source_fits']+=1
                save(out/'progress.json',{**state,'event':'SOURCE_FIT_CLOSED','split_id':c['split_id'],'seed':seed},immutable=False)
        # Only source truth is joined after source prediction files are immutable.
        choices=[]
        for outer in range(5):
            contexts=[c for c in prep['contexts'] if c['outer_fold']==outer and c['stage']=='inner']
            summaries=[]
            for epoch in range(1,49):
                per_seed=[]
                for seed in SEEDS:
                    truth={};predicted={}
                    for c in contexts:
                        actual=verified_truth(c,old)
                        with np.load(out/'source_models'/c['split_id']/f'seed{seed}'/'predictions.npz',allow_pickle=False) as a:
                            ids=a['record_ids'];scores=a['epoch_scores'][epoch-1]
                        truth.update(actual);predicted.update(dict(zip(ids, np.asarray(CLASSES)[scores.argmax(1)])))
                    if len(truth)!=684 or set(truth)!=set(predicted):raise ValueError('Incomplete source OOF denominator')
                    per_seed.append(old.metric_record([truth[k] for k in truth],[predicted[k] for k in truth]))
                summaries.append({'epoch':epoch,**{m:float(np.mean([r[m] for r in per_seed])) for m in ('macro_f1','balanced_accuracy','accuracy')}})
            chosen=select_epoch(summaries)
            choices.append({'outer_fold':outer,'epochs':chosen['epoch'],'source_selected_metrics':chosen,'source_curve':summaries})
        save(out/'source_epoch_lock.json',{'status':'ALL_SOURCE_CHOICES_LOCKED_BEFORE_OUTER','protocol_sha256':sha(out/'protocol.json'),'choices':choices,'outer_truth_used':False})
        for c in prep['contexts']:
            if c['stage']!='outer':continue
            source=old.read_pack(OLD/c['source_pack'],True);query=old.read_pack(OLD/c['query_pack'],False)
            epoch=choices[c['outer_fold']]['epochs']
            for seed in SEEDS:
                matched=[]
                for arm in ARMS:
                    d=out/'outer_models'/c['split_id']/f'seed{seed}'/arm
                    with old.forbid_named_truth(forbidden):
                        meta=fit(source,query,aux,arm,epoch,seed,d,c['split_id'],np,torch,StandardScaler,old,guard)
                    matched.append(meta);state['completed_outer_fits']+=1
                    save(out/'progress.json',{**state,'event':'OUTER_FIT_CLOSED','split_id':c['split_id'],'seed':seed,'arm':arm},immutable=False)
                for key in ('initial_encoder_sha256','initial_ZY_head_sha256','target_batch_schedule_sha256'):
                    if len({m[key] for m in matched})!=1:raise ValueError('Paired target state/order differs: '+key)
        files={str(p.relative_to(out)):sha(p) for p in (out/'outer_models').rglob('predictions.npz')}
        if len(files)!=45:raise ValueError('All arms, five outer folds and all three seeds must finish')
        save(out/'outer_prediction_lock.json',{'status':'ALL_OUTER_PREDICTIONS_LOCKED_BEFORE_SCORING','files':files,
            'source_epoch_lock_sha256':sha(out/'source_epoch_lock.json'),'source_fits':30,'outer_fits':45})
        state['status']='PREDICTIONS_COMPLETE_NOT_YET_SCORED'
    except BaseException as exc:
        state.update(status='PARTIAL',reason=str(exc),exception_type=type(exc).__name__)
        (out/'worker_traceback.txt').write_text(traceback.format_exc())
    state.update(ended_at_utc=utc(),actual_process_CPU_seconds=time.process_time()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save(out/'worker_result.json',state)
    return 0 if state['status']=='PREDICTIONS_COMPLETE_NOT_YET_SCORED' else 2


def score(out):
    np,torch,StandardScaler,old,guard=dependencies()
    lock=read(out/'outer_prediction_lock.json')
    if read(out/'protocol.json')!=protocol() or read(out/'registration.json')['protocol_sha256']!=sha(out/'protocol.json'):
        raise ValueError('Registered scoring provenance changed')
    if lock['source_epoch_lock_sha256']!=sha(out/'source_epoch_lock.json'):
        raise ValueError('Source-only epoch lock changed')
    for path,expected in lock['files'].items():
        if sha(out/path)!=expected:raise ValueError('Frozen outer predictions changed')
    prep=read(OLD/'preparation.json');metrics={};rows=[];comparisons=[]
    for seed in SEEDS:
        by_arm={}
        for arm in ARMS:
            truth={};prediction={}
            for c in prep['contexts']:
                if c['stage']!='outer':continue
                truth.update(verified_truth(c,old))
                path=out/'outer_models'/c['split_id']/f'seed{seed}'/arm/'predictions.npz'
                with np.load(path,allow_pickle=False) as a:ids=a['record_ids'];scores=a['scores']
                found=np.asarray(CLASSES)[scores.argmax(1)]
                for rid,pred,s in zip(ids,found,scores):
                    if str(rid) in prediction:raise ValueError('Duplicate outer prediction ID')
                    prediction[str(rid)]=str(pred)
                    rows.append({'seed':seed,'arm':arm,'outer_fold':c['outer_fold'],'record_id':str(rid),'true_label':truth[str(rid)],'prediction':str(pred),**{'score_'+c:float(v) for c,v in zip(CLASSES,s)}})
            if len(prediction)!=855 or set(prediction)!=set(truth):raise ValueError('Incomplete outer denominator')
            metrics[f'{seed}.{arm}']=old.metric_record([truth[k] for k in truth],[prediction[k] for k in truth])
            by_arm[arm]=(truth,prediction)
        for ref in ('A','Aextra_L2SP'):
            truth,base=by_arm[ref];candidate=by_arm['B_HQ_TM_L2SP'][1]
            corrected=sum(base[k]!=truth[k] and candidate[k]==truth[k] for k in truth)
            harmed=sum(base[k]==truth[k] and candidate[k]!=truth[k] for k in truth)
            comparisons.append({'seed':seed,'candidate':'B_HQ_TM_L2SP','reference':ref,'corrected':corrected,'harmed':harmed,'net_correct':corrected-harmed,'accuracy_delta_pp':100*(corrected-harmed)/855})
    old.csvwrite(out/'all_outer_predictions.csv.gz',rows)
    mean={arm:{m:float(np.mean([metrics[f'{s}.{arm}'][m] for s in SEEDS])) for m in ('accuracy','macro_f1','balanced_accuracy')} for arm in ARMS}
    save(out/'metrics.json',{'status':'DONE_DEVELOPMENT_ONLY','study_status':'DEVELOPMENT_NOT_NEW_CONFIRMATION','per_seed':metrics,'mean':mean,'paired_changes':comparisons,'sample_denominator_per_seed':855,'total_unique_target_samples':855,'historical_96_02_direct_comparison_allowed':False})
    print(json.dumps({'status':'DONE_DEVELOPMENT_ONLY','mean':mean}),flush=True)
    return 0


def monitor_owned_child(child, batch_start, used, *, waiter=None, reader=None, clock=None, sleeper=None, sender=None):
    """Bound one owned, single-process CPU payload; unknown wait4 never costs zero."""
    waiter=waiter or os.wait4;clock=clock or time.monotonic;sleeper=sleeper or time.sleep
    reader=reader or (lambda:Path(f'/proc/{child.pid}/stat').read_text())
    pidfd=None
    if sender is None:
        try:pidfd=os.pidfd_open(child.pid)
        except ProcessLookupError:pass
        def sender(sig):
            if pidfd is not None:
                try:signal.pidfd_send_signal(pidfd,sig)
                except ProcessLookupError:pass
    peak=0;stop=None;sent=None;missing_since=None;failure=None;last_cpu=0.;kill_sent=False
    try:
        while True:
            try:
                pid,status,usage=waiter(child.pid,os.WNOHANG)
            except BaseException as exc:
                failure=repr(exc);stop='UNKNOWN_WAIT4_ACCOUNTING';pid=0
                if sent is None:sent=clock();sender(signal.SIGTERM)
            if pid:
                child.returncode=os.waitstatus_to_exitcode(status)
                return {'usage':usage,'exit_code':child.returncode,'stop_reason':stop,'peak_RSS_bytes_sampled':peak,'monitor_error':failure}
            try:
                text=reader();fields=text[text.rfind(')')+1:].split()
                if len(fields)<22:raise ValueError('Incomplete process CPU/RSS observation')
                cpu=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
                last_cpu=max(last_cpu,cpu)
                rss=max(0,int(fields[21]))*os.sysconf('SC_PAGE_SIZE');peak=max(peak,rss)
                missing_since=None
                if used+last_cpu>=CPU_CAP-10:stop=stop or 'ACTUAL_CPU_GUARD'
                if rss>RSS_CAP:stop=stop or 'RSS_GUARD'
            except FileNotFoundError:
                missing_since=clock() if missing_since is None else missing_since
                if clock()-missing_since>=.5:stop=stop or 'BOUNDED_PROCESS_OBSERVATION_EXPIRED'
            except BaseException as exc:
                failure=repr(exc);stop=stop or 'MONITOR_FAILED_CLOSED'
            now=clock()
            if now-batch_start>=WALL_CAP-5:stop=stop or 'WALL_GUARD'
            if stop and sent is None:
                sender(signal.SIGTERM);sent=now
            if sent is not None and now-sent>=1 and not kill_sent:
                sender(signal.SIGKILL);kill_sent=True
            if sent is not None and now-sent>=3:
                return {'usage':None,'exit_code':None,'stop_reason':'UNRESOLVED_WAIT4_AFTER_BOUNDED_KILL',
                    'peak_RSS_bytes_sampled':peak,'monitor_error':failure,'actual_CPU_seconds_lower_bound':used+last_cpu}
            sleeper(.1)
    finally:
        if pidfd is not None:os.close(pidfd)


def supervise(out):
    if not out.is_dir():raise ValueError('Register before launching the supervisor')
    with (out/'.pilot.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return supervise_locked(out)


def supervise_locked(out):
    if not out.is_dir() or read(out/'protocol.json')!=protocol() or not (out/'registration.json').exists():
        raise ValueError('Register the frozen protocol before training')
    registration=read(out/'registration.json')
    if registration['protocol_sha256']!=sha(out/'protocol.json') or (out/'resource_ledger.json').exists():
        raise ValueError('Protocol changed or this attempt already started; preserve prior artifacts')
    start=time.monotonic();used=0.;phases=[]
    env={**os.environ,**{key:'1' for key in THREADS},'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT/'.runtime/quantum-pilot-site')+os.pathsep+str(ROOT/'.runtime/quantum-upgrade-site')}
    for phase in ('worker','score'):
        remaining=CPU_CAP-used
        if remaining<20:break
        parent_pid=os.getpid()
        def limits():
            resource.setrlimit(resource.RLIMIT_CPU,(max(1,int(remaining)-10),int(remaining)))
            resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2)
            if ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)!=0:os._exit(126)
            if os.getppid()!=parent_pid:os._exit(125)
        command=[sys.executable,str(Path(__file__).resolve()),phase,'--output',str(out.resolve())]
        row={'phase':phase,'command':command,'started_at_utc':utc(),'status':'RUNNING','prior_actual_CPU_seconds':used}
        with (out/(phase+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
            row['pid']=child.pid
            save(out/'resource_ledger.json',{'status':'RUNNING','actual_closed_CPU_seconds':used,
                'active_CPU_seconds_final':None,'CPU_cap_seconds':CPU_CAP,'wall_cap_seconds':WALL_CAP,
                'RSS_best_effort_cap_bytes':RSS_CAP,'threads':1,'phases':[*phases,row]},immutable=False)
            observed=monitor_owned_child(child,start,used)
        usage=observed.pop('usage')
        if usage is None:
            row.update(observed,status='UNKNOWN_CPU_PARTIAL',ended_at_utc=utc(),actual_CPU_seconds=None,
                log_sha256=sha(out/(phase+'.log')))
            phases.append(row)
            save(out/'resource_ledger.json',{'status':'UNKNOWN_CPU_PARTIAL','actual_CPU_seconds':None,
                'known_closed_CPU_seconds':used,'CPU_cap_seconds':CPU_CAP,'phases':phases},immutable=False)
            save(out/'completion.json',{'status':'PARTIAL','actual_CPU_seconds':None,'known_closed_CPU_seconds':used,
                'reason':'Unknown final wait4 cost blocks further scoring/restarts; manual review required',
                'quantum_calculations':0,'all_predeclared_seeds_and_folds_complete':False})
            return 2
        actual=float(usage.ru_utime+usage.ru_stime);used+=actual
        row.update(observed,status='EXITED' if observed['exit_code']==0 and observed['stop_reason'] is None else 'PARTIAL',
            ended_at_utc=utc(),actual_CPU_seconds=actual,wait4_peak_RSS_KiB=usage.ru_maxrss,log_sha256=sha(out/(phase+'.log')))
        phases.append(row)
        save(out/'resource_ledger.json',{'status':row['status'],'actual_CPU_seconds':used,'wall_seconds':time.monotonic()-start,
            'CPU_cap_seconds':CPU_CAP,'wall_cap_seconds':WALL_CAP,'RSS_best_effort_cap_bytes':RSS_CAP,'threads':1,'phases':phases},immutable=False)
        if row['status']!='EXITED':break
    done=len(phases)==2 and all(p['status']=='EXITED' for p in phases)
    save(out/'completion.json',{'status':'DONE_DEVELOPMENT_ONLY' if done else 'PARTIAL','actual_CPU_seconds':used,
        'wall_seconds':time.monotonic()-start,'protocol_sha256':sha(out/'protocol.json'),'quantum_calculations':0,
        'all_predeclared_seeds_and_folds_complete':done,'new_independent_confirmation':False})
    return 0 if done else 2


def register(out):
    if out.exists():raise FileExistsError('New registrations cannot overwrite existing evidence')
    out.mkdir(parents=True)
    save(out/'protocol.json',protocol())
    save(out/'registration.json',{'status':'REGISTERED_NOT_RUN','registered_at_utc':utc(),
        'protocol_sha256':sha(out/'protocol.json'),'training_fits':0,'quantum_calculations':0,
        'source_sha256':sha(__file__)})
    print(json.dumps(read(out/'registration.json')))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('register','supervise','worker','score'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raise SystemExit({'register':register,'supervise':supervise,'worker':worker,'score':score}[a.stage](a.output.resolve()))


if __name__=='__main__':main()
