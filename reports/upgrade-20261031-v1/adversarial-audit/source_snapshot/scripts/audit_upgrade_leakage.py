"""Run bounded, real C/D/H/I leakage diagnostics on one fixed legacy outer fold.

The training worker receives source labels and label-free query arrays only.
Python audit hooks add a logical file-read guard; they are not an OS sandbox.
This is an exploratory diagnostic, not a new independent evaluation.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from herb_spectra.classification import classification_metrics
from herb_spectra import selection
from herb_spectra.spectral_models import SpectralPreprocessor


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz"
SPLITS = ROOT / "reports/staged-nested-optimization/2026-10-31/core/splits.json"
DEFAULT_OUT = ROOT / "outputs/upgrade-20261031-v1/adversarial-audit"
DEFAULT_REPORT = ROOT / "reports/upgrade-20261031-v1/adversarial-audit"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def array_sha(value):
    value = np.asarray(value)
    return hashlib.sha256(canonical({"shape":value.shape,"dtype":value.dtype.str}).encode()+np.ascontiguousarray(value).tobytes()).hexdigest()


def model_state(bundle):
    scaler, selector, model = (bundle[k] for k in ("scaler", "selector", "model"))
    state = {"C":float(model.C),"gamma":float(model._gamma),"classes":model.classes_.tolist()}
    for key, obj, names in (
        ("scaler", scaler, ("mean_", "var_", "scale_", "n_samples_seen_")),
        ("selector", selector, ("selected_indices_", "coefficients_", "signed_stability_", "importances_")),
        ("svm", model, ("support_", "support_vectors_", "dual_coef_", "intercept_", "n_iter_")),
    ):
        state[key] = {name:array_sha(getattr(obj,name)) for name in names}
    return state


def assert_unchanged(left, right, description):
    if canonical(left) != canonical(right):
        raise AssertionError("Leakage/state invariant failed: "+description)


def block_group_permutation(labels, groups, blocks, seed):
    """Permute one label per physical group within prespecified membership blocks."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    if len(labels) != len(groups):
        raise ValueError("One group is required for each source row")
    output = labels.copy(); rng = np.random.default_rng(seed); seen = set()
    for block in blocks:
        block = np.asarray(block, dtype=int)
        if seen & set(block.tolist()):
            raise ValueError("Permutation blocks overlap")
        seen.update(block.tolist())
        unique = np.unique(groups[block]); values = []
        for group in unique:
            all_rows = np.flatnonzero(groups == group)
            if not set(all_rows.tolist()) <= set(block.tolist()):
                raise ValueError("A physical group crosses permutation blocks")
            found = np.unique(labels[all_rows])
            if len(found) != 1:
                raise ValueError("Physical group has inconsistent labels")
            values.append(found[0])
        permuted = rng.permutation(np.asarray(values))
        for group, label in zip(unique,permuted,strict=True):
            output[groups == group] = label
    if seen != set(range(len(labels))):
        raise ValueError("Permutation blocks must cover every source row")
    return output


@contextmanager
def traced_mc_pls(scope, Z, y, identifiers, runs, seed):
    """Bind each actual inner PLS.fit to the verified MC subsample identities."""
    x_fit, target, encoded, classes, _ = selection._prepare(Z,y,"classification",identifiers)
    ordered_ids = np.unique(identifiers)
    rng = np.random.default_rng(seed)
    subsamples = [selection._subsample(encoded,len(x_fit),.8,rng) for _ in range(runs)]
    original_coefficients, original_pls = selection._coefficients, selection.PLSRegression
    count = [0]; current = [None]

    class TracedPLS(original_pls):
        def fit(self, X, y, **kwargs):
            ids = current[0]
            if ids is None:
                raise AssertionError("Unbound nested PLS fit")
            with scope.operation(component="MC-UVE.internal_PLS.fit",sample_ids=ids,X=X,y=y):
                return super().fit(X,y,**kwargs)

    def coefficients(X, Y, components):
        sample = subsamples[count[0]]
        if not np.array_equal(X,x_fit[sample]) or not np.array_equal(Y,target[sample]):
            raise AssertionError("Observed MC rows differ from the declared subsample")
        current[0] = ordered_ids[sample]
        count[0] += 1
        return original_coefficients(X,Y,components)

    with patch.object(selection,"_coefficients",coefficients),patch.object(selection,"PLSRegression",TracedPLS):
        yield
    if count[0] != runs:
        raise AssertionError("Not every declared MC run executed")


def fit_representation(X, y, ids, axis, scope, protocol):
    prep = SpectralPreprocessor("snv",axis)
    per_row = prep.transform(X)
    scaler = scope.fit(StandardScaler(),per_row,sample_ids=ids,component="StandardScaler.fit")
    Z = scaler.transform(per_row)
    selector = selection.MCUVESelector(n_runs=protocol["mc_runs"],n_components=1,
        n_features_to_select=protocol["selected_features"],sample_fraction=.8,task="classification",random_state=42)
    with traced_mc_pls(scope,Z,y,ids,protocol["mc_runs"],42):
        scope.fit(selector,Z,y,sample_ids=ids,component="MC-UVE.fit",groups=ids)
    return prep,scaler,selector,Z[:,selector.selected_indices_]


def transform(bundle, X):
    Z = bundle["scaler"].transform(bundle["preprocessor"].transform(X))
    return Z[:,bundle["selector"].selected_indices_]


def train_nested(train, protocol, log):
    """No query values or query labels enter this training API."""
    from herb_spectra.leakage_guard import FitScope
    X,y,ids,axis = (train[k] for k in ("X","y","ids","axis"))
    classes = np.unique(y); records = []
    for fold in range(2):
        source,valid = (train[f"inner_{fold}_{kind}"] for kind in ("train","validation"))
        if set(ids[source]) & set(ids[valid]):
            raise AssertionError("Source/validation physical IDs overlap")
        if set(y[source]) != set(classes) or set(y[valid]) != set(classes):
            raise ValueError("A fixed inner fold lacks one of the seven classes")
        scope = FitScope(allowed_ids=ids[source],split_id=f"outer0.inner{fold}",
                         dataset_version=protocol["dataset_version"],log_path=log,
                         code_version=protocol["script_sha256"])
        prep,scale,selector,Z = fit_representation(X[source],y[source],ids[source],axis,scope,protocol)
        valid_Z = scale.transform(prep.transform(X[valid]))[:,selector.selected_indices_]
        for C in protocol["C_candidates"]:
            model = scope.fit(SVC(C=C,kernel="rbf",gamma="scale",class_weight="balanced",
                decision_function_shape="ovr",break_ties=False,random_state=42),Z,y[source],
                sample_ids=ids[source],component=f"SVC.fit.C{C:g}")
            pred = model.predict(valid_Z)
            records.append({"inner_fold":fold,"C":C,"metrics":classification_metrics(y[valid],pred,classes),
                            "selected_indices":selector.selected_indices_.tolist(),
                            "validation_ids":ids[valid].tolist(),"validation_prediction":pred.tolist()})
    means = {str(C):{m:float(np.mean([r["metrics"][m] for r in records if r["C"]==C]))
                          for m in ("macro_f1","balanced_accuracy","accuracy")} for C in protocol["C_candidates"]}
    chosen = min(protocol["C_candidates"],key=lambda C:(-means[str(C)]["macro_f1"],
        -means[str(C)]["balanced_accuracy"],-means[str(C)]["accuracy"],C))
    scope=FitScope(allowed_ids=ids,split_id="outer0.full_source",dataset_version=protocol["dataset_version"],log_path=log,
                   code_version=protocol["script_sha256"])
    prep,scale,selector,Z=fit_representation(X,y,ids,axis,scope,protocol)
    model=scope.fit(SVC(C=chosen,kernel="rbf",gamma="scale",class_weight="balanced",
        decision_function_shape="ovr",break_ties=False,random_state=42),Z,y,sample_ids=ids,component="SVC.final_fit")
    return {"preprocessor":prep,"scaler":scale,"selector":selector,"model":model}, {
        "selected_C":chosen,"inner_candidate_means":means,"inner_records":records,
        "source_ids":ids.tolist(),"class_order":classes.tolist(),"outer_labels_used":False}


def forbid_reads(paths):
    blocked = [Path(p).resolve() for p in paths]; attempts=[]
    def hook(event,args):
        if event != "open" or not args or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=Path(os.fsdecode(args[0])).resolve()
        if any(path==p or p in path.parents for p in blocked):
            attempts.append(str(path))
            raise PermissionError("Forbidden scorer/historical input in training worker")
    sys.addaudithook(hook)
    return attempts


def worker(train_path, query_path, protocol_path, destination, deny_paths):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    attempts=forbid_reads(deny_paths)
    probe=Path(deny_paths[0])/"labels_original.json"
    try:
        probe.read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError("Intentional truth-file access was not blocked")
    protocol=json.loads(Path(protocol_path).read_text())
    with np.load(train_path,allow_pickle=False) as source:
        train={k:source[k] for k in source.files}
    with np.load(query_path,allow_pickle=False) as incoming:
        if set(incoming.files)!={"X","ids","axis"}:
            raise AssertionError("Inference input must contain no label or metadata features")
        query={k:incoming[k] for k in incoming.files}
    if set(train["ids"]) & set(query["ids"]):
        raise AssertionError("Physical group enters source and query")
    bundle,choice=train_nested(train,protocol,destination/"fit_scope_manifest.jsonl")
    joblib.dump(bundle,destination/"model.joblib",compress=3)
    bundle=joblib.load(destination/"model.joblib")
    state=model_state(bundle);Z=transform(bundle,query["X"])
    predicted=bundle["model"].predict(Z);scores=bundle["model"].decision_function(Z)
    # Real batch/order/singleton check, retaining original SVC voting semantics.
    singles=np.concatenate([bundle["model"].decision_function(transform(bundle,row[None,:])) for row in query["X"]])
    reversed_scores=bundle["model"].decision_function(transform(bundle,query["X"][::-1]))[::-1]
    pointwise=max(float(np.max(np.abs(scores-singles))),float(np.max(np.abs(scores-reversed_scores))))
    if pointwise>1e-10:
        raise AssertionError("Pointwise inference exceeded the predeclared1e-10 tolerance")
    assert_unchanged(state,model_state(bundle),"inference changed learned state")
    # Save label-free inference before the parent process opens the scorer truth.
    np.savez_compressed(destination/"predictions.npz",record_ids=query["ids"],predicted=predicted,
                        scores=scores,class_order=bundle["model"].classes_)
    write_json(destination/"selection_lock.json",choice)
    write_json(destination/"training_state.json",state)
    write_json(destination/"worker_isolation.json",{"scope":"logical Python audit hook, not OS sandbox",
        "forbidden_read_attempts":attempts,"intentional_truth_probe_blocked":str(probe.resolve()) in attempts,
        "training_api_accepts_query_labels":False,"query_file_keys":list(query),
        "pointwise_score_max_difference":pointwise,"pointwise_tolerance":1e-10,
        "post_inference_state_unchanged":True,"source_samples":len(train["ids"]),"query_samples":len(query["ids"]),
        "completed_at_utc":datetime.now(timezone.utc).isoformat()})


def load_predictions(path):
    with np.load(path,allow_pickle=False) as values:
        d={k:values[k] for k in values.files}
    if len(d["record_ids"])!=len(set(d["record_ids"].tolist())):
        raise ValueError("Duplicate query prediction would inflate the denominator")
    return d


def score_saved_predictions(path, truth):
    d=load_predictions(path)
    if set(d["record_ids"])!=set(truth):
        raise ValueError("Prediction/truth identities differ")
    labels=np.asarray([truth[str(i)] for i in d["record_ids"]])
    # Independent count arithmetic; do not call the training metric helper.
    classes=d["class_order"].tolist();lookup={str(c):i for i,c in enumerate(classes)}
    matrix=np.zeros((len(classes),len(classes)),dtype=int)
    for actual,predicted in zip(labels,d["predicted"],strict=True):
        matrix[lookup[str(actual)],lookup[str(predicted)]]+=1
    tp=matrix.diagonal();support=matrix.sum(axis=1);found=matrix.sum(axis=0)
    recall=np.divide(tp,support,out=np.zeros(len(classes)),where=support>0)
    precision=np.divide(tp,found,out=np.zeros(len(classes)),where=found>0)
    f1=np.divide(2*tp,support+found,out=np.zeros(len(classes)),where=(support+found)>0)
    result={"accuracy":float(tp.sum()/matrix.sum()),"macro_f1":float(f1.mean()),
            "balanced_accuracy":float(recall[support>0].mean()),"class_order":classes,
            "confusion_matrix":matrix.tolist(),"correct":int(tp.sum()),"samples":len(labels),
            "prediction_sha256":sha(path),"independent_metric_method":"direct raw confusion counts",
            "per_class":{str(c):{"precision":float(p),"recall":float(r),"f1":float(f),"support":int(n)}
                         for c,p,r,f,n in zip(classes,precision,recall,f1,support,strict=True)}}
    return result


def prepare(output, report):
    output,report=Path(output),Path(report)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new diagnostic directory")
    output.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    with np.load(INPUT,allow_pickle=False) as a:
        X,y,ids,axis,groups=(a[k] for k in ("X_alpha","y","record_ids","axis","groups"))
    if not np.array_equal(ids,groups) or len(set(ids.tolist()))!=855:
        raise ValueError("Diagnostic needs the declared unique physical IDs")
    fold=json.loads(SPLITS.read_text())[0];lookup={str(v):i for i,v in enumerate(ids)}
    source=np.asarray([lookup[i] for i in fold["train_sample_ids"]]);query=np.asarray([lookup[i] for i in fold["test_sample_ids"]])
    local={str(ids[g]):i for i,g in enumerate(source)}
    arrays={"X":X[source],"y":y[source],"ids":ids[source],"axis":axis}
    for i,inner in enumerate(fold["inner_folds"]):
        arrays[f"inner_{i}_train"]=np.asarray([local[v] for v in inner["train_sample_ids"]])
        arrays[f"inner_{i}_validation"]=np.asarray([local[v] for v in inner["validation_sample_ids"]])
    inputs=output/"trainer_inputs";inputs.mkdir();scorer=output/"scorer_only";scorer.mkdir()
    np.savez_compressed(inputs/"source.npz",**arrays)
    np.savez_compressed(inputs/"query.npz",X=X[query],ids=ids[query],axis=axis)
    changed=X[query].copy();changed[0,:20]+=np.linspace(-50.,50.,20)
    np.savez_compressed(inputs/"query_changed.npz",X=changed,ids=ids[query],axis=axis)
    truth={str(i):str(t) for i,t in zip(ids[query],y[query],strict=True)}
    shuffled=np.random.default_rng(20261121).permutation(y[query])
    write_json(scorer/"labels_original.json",truth)
    write_json(scorer/"labels_shuffled.json",{str(i):str(t) for i,t in zip(ids[query],shuffled,strict=True)})
    protocol={"protocol_id":"upgrade-20261031-v1-onefold-adversarial","dataset_version":"2026-10-31-zaoren-numbered-v1",
        "scope":"EXPLORATORY one old outer fold0; no full855 OOF or independent confirmation",
        "mc_runs":10,"PLS_components":1,"selected_features":62,"C_candidates":[1.,10.],
        "preprocessing":"per-row SNV followed by source-only StandardScaler",
        "gamma":"scale","class_weight":"balanced","model_seed":42,"outer_fold":0,"inner_folds":2,
        "selection_metric_order":["macro_f1","balanced_accuracy","accuracy","lower_C"],
        "permutation_seeds":list(range(20261111,20261121)),
        "permutation_rule":"group labels shuffled within the two fixed inner membership blocks, retaining block class counts; conditional null",
        "pointwise_tolerance":1e-10,"input_sha256":sha(INPUT),"legacy_splits_sha256":sha(SPLITS),
        "script_sha256":sha(Path(__file__)),"requires_root_release_after_minimum_guards":True,
        "limitations":["Historical855 already exposed; user-declared physical independence; one lot per origin",
                       "10 permutations have coarse empirical resolution and are not10independent studies",
                       "Python audit hook is a logical read restriction, notOS sandbox/security certification",
                       "Checks cover this bounded workflow, notall possible models or absolute zero leakage"]}
    write_json(report/"protocol.json",protocol)
    write_json(report/"input_manifest.json",{"files":{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
        "source_ids":ids[source].tolist(),"query_ids":ids[query].tolist(),"source_class_counts":dict(Counter(y[source].tolist())),
        "query_class_counts":dict(Counter(y[query].tolist())),"source_query_disjoint":not bool(set(ids[source])&set(ids[query])),
        "query_truth_is_not_an_input_to_worker":True,"created_at_utc":datetime.now(timezone.utc).isoformat()})
    return {"status":"prepared_no_fits","protocol_sha256":sha(report/"protocol.json"),"input_manifest_sha256":sha(report/"input_manifest.json")}


def run(output, report, release_path):
    output,report=Path(output),Path(report);protocol=json.loads((report/"protocol.json").read_text())
    release=json.loads(Path(release_path).read_text())
    expected={"protocol_sha256":sha(report/"protocol.json"),"input_manifest_sha256":sha(report/"input_manifest.json"),"script_sha256":sha(Path(__file__))}
    if release.get("allow_minimal_adversarial_fit") is not True or any(release.get(k)!=v for k,v in expected.items()):
        raise ValueError("Root release does not bind this frozen diagnostic")
    if protocol["script_sha256"]!=sha(Path(__file__)):
        raise ValueError("Diagnostic code changed after preparation")
    with np.load(output/"trainer_inputs/source.npz",allow_pickle=False) as src:
        source={k:src[k] for k in src.files}
    blocked=[str(output/"scorer_only"),str(INPUT),str(ROOT/"reports/staged-nested-optimization/2026-10-31/core/outer_predictions.csv.gz")]
    variants=[("baseline",output/"trainer_inputs/source.npz",output/"trainer_inputs/query.npz"),
              ("shuffled_test_truth",output/"trainer_inputs/source.npz",output/"trainer_inputs/query.npz"),
              ("changed_test_input",output/"trainer_inputs/source.npz",output/"trainer_inputs/query_changed.npz")]
    blocks=[source["inner_0_train"],source["inner_0_validation"]]
    for seed in protocol["permutation_seeds"]:
        changed=dict(source);changed["y"]=block_group_permutation(source["y"],source["ids"],blocks,seed)
        path=output/"trainer_inputs"/f"source_permuted_{seed}.npz";np.savez_compressed(path,**changed)
        variants.append((f"source_permutation_{seed}",path,output/"trainer_inputs/query.npz"))
    events=[]
    for name,train_path,query_path in variants:
        destination=output/"workers"/name
        command=[sys.executable,str(Path(__file__).resolve()),"worker","--train",str(train_path),"--query",str(query_path),
            "--protocol",str(report/"protocol.json"),"--destination",str(destination)]
        for p in blocked: command += ["--deny",p]
        started=datetime.now(timezone.utc).isoformat();before=time.monotonic()
        completed=subprocess.run(command,cwd=ROOT,env={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"},capture_output=True,text=True)
        events.append({"name":name,"argv":command,"started_at_utc":started,"finished_at_utc":datetime.now(timezone.utc).isoformat(),"seconds":time.monotonic()-before,"returncode":completed.returncode})
        (destination/"worker_stdout.log").write_text(completed.stdout+completed.stderr)
        if completed.returncode:
            write_json(report/"execution_failure.json",events[-1]);raise RuntimeError("Diagnostic worker failed: "+name)
    write_json(report/"command_invocations.json",events)
    baseline=output/"workers/baseline";other=output/"workers/shuffled_test_truth";changed=output/"workers/changed_test_input"
    left=json.loads((baseline/"training_state.json").read_text())
    choice=json.loads((baseline/"selection_lock.json").read_text())
    for directory in (other,changed):
        assert_unchanged(left,json.loads((directory/"training_state.json").read_text()),"C/D trained artifacts")
        assert_unchanged(choice,json.loads((directory/"selection_lock.json").read_text()),"C/D source-only selection")
    a,b=(load_predictions(p/"predictions.npz") for p in (baseline,other))
    if not all(np.array_equal(a[k],b[k]) for k in a):raise AssertionError("Test-label permutation changed inference")
    changed_prediction=load_predictions(changed/"predictions.npz")
    if not np.array_equal(a["scores"][1:],changed_prediction["scores"][1:]):
        raise AssertionError("Changing one test input affected other samples' scores")
    original=json.loads((output/"scorer_only/labels_original.json").read_text())
    shuffled=json.loads((output/"scorer_only/labels_shuffled.json").read_text())
    original_score=score_saved_predictions(baseline/"predictions.npz",original)
    shuffled_score=score_saved_predictions(other/"predictions.npz",shuffled)
    permutation=[]
    for seed in protocol["permutation_seeds"]:
        directory=output/"workers"/f"source_permutation_{seed}"
        permutation.append({"seed":seed,**score_saved_predictions(directory/"predictions.npz",original)})
    majority=str(max(Counter(source["y"].tolist()),key=Counter(source["y"].tolist()).get))
    majority_pred=np.repeat(majority,len(original));truth=np.asarray(list(original.values()))
    majority_metrics=classification_metrics(truth,majority_pred,np.unique(source["y"]))
    scores=np.asarray([r["accuracy"] for r in permutation])
    audit={"status":"TECHNICAL_CHECKS_PASS_EXPLORATORY_DATA","C_test_label_invariance":"PASS","D_test_input_fit_state_invariance":"PASS",
        "E_pointwise_inference":"PASS","H_source_group_permutation":"EXECUTED_CONDITIONAL_NULL_10","I_independent_metrics":"PASS",
        "whole_pipeline_scope":"one fixed historical outer fold0, two fixed inner folds, MC10/LV1/K62, two C choices",
        "source_model_paths_executed":len(variants),"original_score":original_score,"shuffled_test_truth_score":shuffled_score,
        "D_changed_sample_prediction_may_change":True,"D_other_query_rows_scores_unchanged":True,
        "majority_class":majority,"majority_baseline":majority_metrics,"permutation_results":permutation,
        "permutation_accuracy_mean":float(scores.mean()),"permutation_accuracy_sample_sd":float(scores.std(ddof=1)),
        "permutation_accuracy_min":float(scores.min()),"permutation_accuracy_max":float(scores.max()),
        "coarse_conditional_accuracy_tail_fraction_plus1":float((1+np.sum(scores>=original_score["accuracy"]))/11),
        "empirical_fraction_is_not_independent_batch_significance":True,"physical_independence_status":"USER_DECLARED_EXPLORATORY",
        "new_classifier_fit_during_final_audit":0,"absolute_zero_leakage_proven":False,
        "not_covered":["new batches","all5outerfold adversarial fits","allpossible models","unknown parent/preparation relations","OS-level filesystem isolation"]}
    write_json(report/"selection_lock.json",choice);write_json(report/"leakage_audit.json",audit)
    fit_entries=[]
    for name,_,_ in variants:
        p=output/"workers"/name/"fit_scope_manifest.jsonl"
        fit_entries.append({"variant":name,"source_log":str(p),"sha256":sha(p),"entries":sum(1 for _ in p.open())})
        for filename in ("predictions.npz","training_state.json","selection_lock.json","worker_isolation.json"):
            target=report/"worker_evidence"/name/filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((output/"workers"/name/filename).read_bytes())
        target=report/"worker_evidence"/name/"fit_scope_manifest.jsonl.gz"
        target.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
    write_json(report/"fit_scope_manifest.json",{"logs":fit_entries,"minimum_guards_release":release,"all_fits_have_explicit_sample_ids":True})
    write_json(report/"frozen_inputs_manifest.json",json.loads((report/"input_manifest.json").read_text()))
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="stage",required=True)
    for name in ("prepare","run"):
        p=sub.add_parser(name);p.add_argument("--output",type=Path,default=DEFAULT_OUT);p.add_argument("--report",type=Path,default=DEFAULT_REPORT)
        if name=="run":p.add_argument("--release",type=Path,required=True)
    p=sub.add_parser("worker");p.add_argument("--train",type=Path,required=True);p.add_argument("--query",type=Path,required=True)
    p.add_argument("--protocol",type=Path,required=True);p.add_argument("--destination",type=Path,required=True);p.add_argument("--deny",action="append",default=[])
    args=parser.parse_args()
    if args.stage=="worker":worker(args.train,args.query,args.protocol,args.destination,args.deny);return
    result=prepare(args.output,args.report) if args.stage=="prepare" else run(args.output,args.report,args.release)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
