"""Append-only anchored legacy-format audit; never replace source numbers.

The single candidate source must have 98 exact fixed-window matches to the
same registered old group, matching explicit column IDs, an identical axis,
and every full spectrum within a fixed absolute 1e-12 tolerance. This is a
known-file provenance check, not a nearest-neighbour label assignment.
"""
from __future__ import annotations
import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import sys

for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="1"
import numpy as np
from intake_supplemental_workbooks_20261104 import fingerprint, jwrite, csvwrite, sha, ROOT

BASE=ROOT/"data/supplemental/teacher_20261104_v1"
OUT=ROOT/"data/supplemental/teacher_20261104_v2"
REPORT=ROOT/"reports/supplemental-legacy-formatting-20261104-v1"
TOLERANCE=1e-12


def verify_format_equivalence(current, old, *, tolerance=TOLERANCE):
    if current.shape!=old.shape:raise ValueError("Different spectrum matrix shape")
    if not np.array_equal(np.isfinite(current),np.isfinite(old)):raise ValueError("Finite patterns differ")
    if not np.isfinite(current).all():raise ValueError("Nonfinite spectra are not formatting equivalents")
    differences=np.max(np.abs(current-old),axis=1)
    if np.any(differences>tolerance):raise ValueError("Changes exceed fixed formatting tolerance")
    return differences


def main():
    resource.setrlimit(resource.RLIMIT_AS,(int(1.5*1024**3),int(1.5*1024**3)))
    if OUT.exists() or REPORT.exists():raise FileExistsError("Fresh append-only correction directories required")
    sources_doc=json.loads((BASE/"sources.json").read_text());sources=sources_doc["sources"]
    metadata=list(csv.DictReader((BASE/"sample_metadata.csv").open(encoding="utf-8-sig")))
    target=next(s for s in sources if s["source_id"]=="SUP006")
    if target["sha256"]!="071fb64540c5a4791ddaf164f981c3b187c270dc05f55edbccdf7e56b619d38b":
        raise ValueError("Unexpected known legacy-format candidate")
    local=[r for r in metadata if r["source_id"]=="SUP006"]
    known_groups={g for r in local for g in json.loads(r["legacy_group_ids"])}
    if len(local)!=100 or known_groups!={"HQ03"} or sum(bool(json.loads(r["legacy_match_ids"])) for r in local)!=98:
        raise ValueError("Unique same-source exact-match anchors are absent")
    oldpath=ROOT/"data/multiplant_database/huangqi/alpha.npz"
    with np.load(oldpath,allow_pickle=False) as z:
        mask=np.asarray([str(v).startswith("HQ03-") for v in z["record_ids"]]);old=z["X"][mask];oldids=z["record_ids"][mask].astype(str);axis=z["axis"]
    with np.load(ROOT/target["modalities"]["alpha"]["artifact"],allow_pickle=False) as z:
        current=z["X"];newaxis=z["axis"]
    if not np.array_equal(axis,newaxis) or old.shape!=(100,2578):raise ValueError("Axis or legacy group dimensions differ")
    if [int(r["original_base_id"]) for r in local]!=list(range(1,101)) or oldids.tolist()!=[f"HQ03-S{i:04d}" for i in range(1,101)]:
        raise ValueError("Explicit header/legacy record bindings differ")
    diffs=verify_format_equivalence(current,old);audit=[];window=(axis>=.1)&(axis<=1.5)
    for i,(r,new,previous,oldid) in enumerate(zip(local,current,old,oldids,strict=True)):
        changed=np.flatnonzero(new!=previous)
        r["legacy_exact_match_ids"]=r["legacy_match_ids"]
        if not json.loads(r["legacy_match_ids"]):
            r["legacy_match_ids"]=json.dumps([str(oldid)])
            r["legacy_match_basis"]="ANCHORED_SOURCE_HEADER_FORMAT_EQUIVALENCE_ATOL_1E-12"
        else:r["legacy_match_basis"]="EXACT_FULL_OR_EXACT_FIXED_WINDOW"
        r["legacy_group_ids"]=json.dumps(["HQ03"]);r["task_group_id"]="HQ03";r["is_new_unmatched"]="false"
        r["entity_id"]="MEAS-"+fingerprint(axis,previous,window)[:24]
        r["legacy_format_variant"]=str(bool(len(changed))).lower()
        audit.append({"sample_id":r["sample_id"],"legacy_record_id":str(oldid),"source_header":r["original_header"],
          "changed_full_axis_cells":len(changed),"changed_fixed_window_cells":int(np.sum(window[changed])),"max_absolute_difference":float(diffs[i]),
          "changed_indices_zero_based":json.dumps(changed.tolist()),"original_numbers_changed":False,
          "old_fingerprint":fingerprint(axis,previous),"current_fingerprint":fingerprint(axis,new),"match_basis":r["legacy_match_basis"]})
    # All other groups retain their original identity and exact source values.
    local_ids={r["sample_id"] for r in local}
    for r in metadata:
        if r["sample_id"] not in local_ids:
            r["legacy_exact_match_ids"]=r["legacy_match_ids"]
            r["legacy_match_basis"]="EXACT_FULL_OR_EXACT_FIXED_WINDOW" if json.loads(r["legacy_match_ids"]) else "NO_LEGACY_MATCH"
            r["legacy_format_variant"]="false"
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True)
    fields=list(metadata[0]);csvwrite(OUT/"sample_metadata.csv",metadata,fields)
    row_lookup={r["sample_id"]:r for r in metadata}
    # Only source metadata arrays are altered; X/axis bytes must remain exact.
    target["task_group_id"]="HQ03";target["legacy_source_group_match"]="ANCHORED_SOURCE_HEADER_FLOAT_FORMAT_EQUIVALENCE"
    for mode,rec in target["modalities"].items():
        if rec.get("status")!="READ_COMPLETE":continue
        with np.load(ROOT/rec["artifact"],allow_pickle=False) as z:payload={k:z[k] for k in z.files}
        original_X=payload["X"].copy();original_axis=payload["axis"].copy();rr=[row_lookup[str(v)] for v in payload["record_ids"]]
        payload.update(labels=np.asarray([r["task_group_id"] for r in rr]),origin_group_ids=np.asarray([r["task_group_id"] for r in rr]),
          entity_ids=np.asarray([r["entity_id"] for r in rr]),physical_sample_ids=np.asarray([r["entity_id"] for r in rr]),
          legacy_match_ids=np.asarray([r["legacy_match_ids"] for r in rr]))
        dest=OUT/"sources/SUP006"/(mode+".npz");dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**payload)
        if not np.array_equal(original_X,payload["X"],equal_nan=True) or not np.array_equal(original_axis,payload["axis"]):raise RuntimeError("Numeric values changed")
        rec.update(artifact=str(dest.relative_to(ROOT)),artifact_sha256=sha(dest),metadata_overlay="HQ03_PROVENANCE_FORMAT_VARIANT_VALUES_UNCHANGED")
    registry=json.loads((BASE/"merged_registry.json").read_text())
    for rec in registry["datasets"]:
        if rec["species"]!="huangqi":continue
        with np.load(ROOT/rec["artifact"],allow_pickle=False) as z:payload={k:z[k] for k in z.files}
        rr=[row_lookup[str(v)] for v in payload["record_ids"]]
        payload.update(labels=np.asarray([r["task_group_id"] for r in rr]),origin_group_ids=np.asarray([r["task_group_id"] for r in rr]),
          entity_ids=np.asarray([r["entity_id"] for r in rr]),physical_sample_ids=np.asarray([r["entity_id"] for r in rr]),
          legacy_match_ids=np.asarray([r["legacy_match_ids"] for r in rr]))
        dest=OUT/"merged/huangqi"/Path(rec["artifact"]).name;dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**payload)
        rec.update(artifact=str(dest.relative_to(ROOT)),sha256=sha(dest),groups=dict(Counter(payload["labels"].tolist())))
    sources_doc.update(status="COMPLETE_EXPLORATORY_WITH_BLOCKED_UPLOADS",metadata_path=str((OUT/"sample_metadata.csv").relative_to(ROOT)),
        additive_parent="data/supplemental/teacher_20261104_v1",legacy_format_audit="reports/supplemental-legacy-formatting-20261104-v1/formatting_audit.csv")
    jwrite(OUT/"sources.json",sources_doc)
    registry.update(status="COMPLETE",sample_metadata_sha256=sha(OUT/"sample_metadata.csv"));jwrite(OUT/"merged_registry.json",registry)
    csvwrite(REPORT/"formatting_audit.csv",audit,list(audit[0]))
    result={"status":"PASS_ANCHORED_LEGACY_FORMAT_EQUIVALENCE","source_id":"SUP006","source_workbook_sha256":target["sha256"],
      "old_group":"HQ03","old_source_alpha_sha256":sha(oldpath),"strict_absolute_tolerance":TOLERANCE,"relative_tolerance":0,
      "sample_headers_exactly_paired":True,"exact_window_anchors":98,"records_with_any_full_axis_format_difference":int(np.count_nonzero(diffs)),
      "maximum_absolute_difference":float(diffs.max()),"new_independent_huangqi_samples":0,"new_unmatched_total":9,
      "total_records":len(metadata),"legacy_bound_records":sum(r["is_new_unmatched"]=="false" for r in metadata),
      "numerical_values_changed":False,"historical_active_files_modified":False,"initial_v1_preserved":True,"model_fits":0,
      "metadata":"data/supplemental/teacher_20261104_v2/sample_metadata.csv","registry":"data/supplemental/teacher_20261104_v2/merged_registry.json",
      "code_sha256":sha(__file__)}
    jwrite(REPORT/"result.json",result)
    files=[{"path":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size} for base in (OUT,REPORT) for p in sorted(base.rglob('*')) if p.is_file()]
    jwrite(REPORT/"artifact_index.json",{"status":"SEALED_ADDITIVE_METADATA_CORRECTION","files":files})
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
