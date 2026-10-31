"""Recreate all final score decisions from frozen source weights, with no fits."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from herb_spectra.calibration_exploration import feature_payload
from herb_spectra.within_batch_holdout import prepare_holdout

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/"pyproject.toml").exists())
RUN=HERE/"run"
p=json.loads((RUN/"protocol_snapshot.json").read_text())
manifest=json.loads((RUN/"run_manifest.json").read_text())
lock=json.loads((RUN/"selection_lock.json").read_text())
metadata=json.loads((RUN/"final_metadata.json").read_text())
with np.load(ROOT/p["input"],allow_pickle=False) as f:
    arrays={k:f[k] for k in f.files}
roles,truth,_=prepare_holdout(arrays,json.loads((ROOT/p["split_plan"]).read_text()))
saved=defaultdict(list)
with gzip.open(RUN/"final_predictions.csv.gz","rt",encoding="utf-8") as f:
    for row in csv.DictReader(f):saved[(row["selector"],int(row["repeat"]),row["method"])].append(row)
cases,rows,checks=0,0,0
source_info=[]
for selector in p["selectors"]:
    entry=next(e for e in manifest["frozen_sources"]["models"] if e["selector"]==selector)
    source_path=ROOT/entry["path"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()==entry["sha256"]
    pipe=joblib.load(source_path);before=joblib.hash(pipe)
    source=feature_payload(pipe,roles["training"]["X_alpha"])[1]
    support=feature_payload(pipe,roles["calibration"]["X_alpha"])[1]
    query=feature_payload(pipe,roles["prediction"]["X_alpha"])[1]
    baseline=np.argmax(query,axis=1)
    classes=pipe.classes_
    assert np.array_equal(classes[baseline],pipe.predict(roles["prediction"]["X_alpha"]))
    np.savez_compressed(HERE/f"{selector}_final_source_scores.npz",source_scores=source,support_pool_scores=support,
                        query_scores=query,source_ids=roles["training"]["record_ids"],support_pool_ids=roles["calibration"]["record_ids"],
                        query_ids=roles["prediction"]["record_ids"],classes=classes)
    meta=next(v for v in metadata if v["selector"]==selector)
    candidate=lock["choices"][selector]["diagnostic"]
    assert lock["choices"][selector]["selected"] is None
    sg=np.sort(source,axis=1);sg=sg[:,-1]-sg[:,-2]
    cap=candidate["cap_fraction"]*np.median(sg)
    gate=np.quantile(sg,candidate["source_gate_quantile"])
    qq=np.sort(query,axis=1);gap=qq[:,-1]-qq[:,-2]
    for draw in meta["draws"]:
        indices=np.array([list(roles["calibration"]["record_ids"]).index(v) for v in draw["support_ids"]])
        ss=support[indices];ys=roles["calibration"]["y"][indices]
        assert len(set(draw["support_ids"]))==7
        assert set(ys.tolist())==set(classes.tolist())
        assert not set(draw["support_ids"])&set(roles["prediction"]["record_ids"].tolist())
        design=np.column_stack([ss,np.ones(len(ss))])
        onehot=np.eye(7)[[list(classes).index(v) for v in ys]]
        weights=np.linalg.solve(design.T@design+np.eye(8),design.T@(onehot-ss))
        raw=np.column_stack([query,np.ones(len(query))])@weights
        legacy=np.argmax(query+raw,axis=1)
        residual=np.clip(raw,-cap,cap)*candidate["blend"]
        residual[gap>gate]=0.
        if np.count_nonzero(classes[np.argmax(ss,axis=1)]!=ys)<candidate["minimum_support_errors"]:
            residual[:]=0.
        diagnostic=np.argmax(query+residual,axis=1)
        predictions={"no_calibration":baseline,"traditional_lambda1":legacy,"conservative_selected":baseline,
                     "same_structure_zero_support":baseline,"same_structure_no_labels":baseline,
                     "prespecified_development_diagnostic":diagnostic}
        for method,pred in predictions.items():
            records=saved[(selector,draw["repeat"],method)]
            assert len(records)==171
            assert [r["record_id"] for r in records]==roles["prediction"]["record_ids"].tolist()
            assert [r["y_predicted"] for r in records]==classes[pred].tolist()
            assert [r["y_true"] for r in records]==[truth[str(v)] for v in roles["prediction"]["record_ids"]]
            cases+=1;rows+=len(records);checks+=4+len(records)
    assert before==joblib.hash(pipe)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()==entry["sha256"]
    source_info.append({"selector":selector,"source_sha256":entry["sha256"],"fingerprint_unchanged":before})
result={"status":"pass","method_cases":cases,"predictions_replayed":rows,"checks":checks,"new_source_fits":0,
        "independent_calibration_formula":"8x8 ridge solved directly; labels only from support; source median/quantile limits; no production adapter call",
        "query_labels_used_for_prediction":False,"sources":source_info}
(HERE/"final_replay_audit.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
