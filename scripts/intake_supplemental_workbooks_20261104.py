"""Anonymous, additive, read-only spectral intake; no model fitting or Git.

The upload receipt is an input list, never a filesystem discovery mechanism.
Byte aliases share one raw archive and one declared batch. Exact spectral
entities remain one leakage group even when filenames or headers differ.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import time
import warnings
import zipfile

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
from openpyxl import Workbook
from herb_spectra.multiplant_database import MODALITIES, read_raw_sheet, workbook_health, numeric_profile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/supplemental/teacher_20261104_v1"
REPORT = ROOT / "reports/supplemental-intake-20261104-v1"
RAW = ROOT / "data/incoming/teacher_20261104_v1"
FIELDS = ["sample_id","entity_id","species","task_group_id","source_id","batch_id",
          "source_workbook_hash","original_base_id","original_header","excel_column",
          "legacy_match_ids","legacy_group_ids","is_new_unmatched","ambiguity_flags",
          "canonical_for_model","alpha_finite","alpha_fingerprint","measurement_fingerprint_0p1_1p5",
          "physical_sample_independence_basis","batch_basis"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def jwrite(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")


def csvwrite(path, rows, fields):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(rows)


def species(name):
    if "黄芪" in name: return "huangqi"
    if "天麻" in name: return "tianma"
    return "zaoren"


def base_id(header):
    text = str(header).strip()
    m = re.match(r"^(ZY\d{2}-\d{3,})(?:_|$)", text)
    if m: return m.group(1)
    m = re.match(r"^(?:样品|样本)\s*(\d+)(?:_|$)", text)
    if m: return str(int(m.group(1)))
    raise ValueError("Unrecognized explicit sample identity: " + text)


def fingerprint(axis, row, mask=None):
    axis = np.asarray(axis, dtype="<f8"); row = np.asarray(row, dtype="<f8")
    if axis.ndim != 1 or row.shape != axis.shape: raise ValueError("Axis/value shape mismatch")
    if mask is not None: axis, row = axis[mask], row[mask]
    # Canonicalize signed zero and NaN payloads solely for identity grouping;
    # archived and numeric output values are never changed.
    a, x = axis.copy(), row.copy(); a[a == 0] = 0; x[x == 0] = 0
    a[np.isnan(a)] = np.nan; x[np.isnan(x)] = np.nan
    h = hashlib.sha256(); h.update(b"axis-value-float64-v1\0"); h.update(len(a).to_bytes(8,"little")); h.update(a.tobytes()); h.update(x.tobytes())
    return h.hexdigest()


def legacy_index():
    full, fixed = defaultdict(list), defaultdict(list)
    files = []
    for plant in ("zaoren","huangqi","tianma"):
        p = ROOT / "data/multiplant_database" / plant / "alpha.npz"
        with np.load(p, allow_pickle=False) as z:
            axis, X = z["axis"], z["X"]
            ids = z["record_ids"].astype(str); groups = z["filename_groups"].astype(str)
            mask = (axis >= .1) & (axis <= 1.5)
            for rid, group, row in zip(ids, groups, X, strict=True):
                rec = {"species":plant,"record_id":str(rid),"group_id":str(group)}
                full[fingerprint(axis,row)].append(rec)
                fixed[fingerprint(axis,row,mask)].append(rec)
        files.append({"path":str(p.relative_to(ROOT)),"sha256":sha(p),"records":len(X),"axis_points":len(axis)})
    return full, fixed, files


def prepare_sources(receipt):
    uploads = []; by_hash = {}; unique = []
    for i, item in enumerate(receipt["files"],1):
        row = {"upload_id":f"UPLOAD{i:03d}", **item}
        if item["download_status"] != "DOWNLOADED":
            row.update(status="BLOCKED_TRANSFER_LIMIT",source_id=None,sha256=None); uploads.append(row); continue
        p = Path(item["path"]); digest = sha(p)
        if digest not in by_hash:
            sid = f"SUP{len(unique)+1:03d}"; archive = RAW / "raw" / f"RAW{len(unique)+1:03d}.xlsx"
            archive.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,archive)
            if sha(archive) != digest: raise RuntimeError("Raw copy hash mismatch")
            src = {"source_id":sid,"species":species(item["original_name"]),"raw_path":str(archive.relative_to(ROOT)),
                   "sha256":digest,"bytes":p.stat().st_size,"aliases":[],"batch_id":sid+"-B01","modalities":{}}
            by_hash[digest] = src; unique.append(src)
        src = by_hash[digest]; src["aliases"].append(item["original_name"])
        if species(item["original_name"]) != src["species"]: src["cross_species_alias_conflict"] = True
        row.update(status="BYTE_ALIAS" if len(src["aliases"])>1 else "ARCHIVED_UNIQUE_RAW",source_id=src["source_id"],
                   raw_path=src["raw_path"],sha256=digest)
        row.pop("path",None); uploads.append(row)
    province_tokens = ("河北","辽宁","陕西","山西","安徽","河南","云南","甘肃","内蒙古","吉林","贵州")
    for src in unique:
        tokens = sorted({token for token in province_tokens if any(token in name for name in src["aliases"])})
        src["alias_named_origin_tokens"] = tokens
        src["alias_name_conflicts"] = (["CONFLICTING_PROVINCE_NAMES_SAME_EXACT_WORKBOOK"] if len(tokens)>1 else [])
        src["named_origin_truth_status"] = "NOT_ASSIGNED_ANONYMOUS_TASK_ONLY"
    jwrite(RAW/"upload_receipts.json",{"created_at_utc":datetime.now(timezone.utc).isoformat(),"uploads":uploads,
               "received_files":len(uploads),"unique_downloaded_raw_files":len(unique),"blocked_files":sum(u["source_id"] is None for u in uploads),
               "bytes_preserved":True,"filenames_are_unverified_aliases_not_features":True,"user_declaration":receipt["user_declared_grain"]})
    return uploads, unique


def sheet_record(path, mode):
    sheet = read_raw_sheet(path, MODALITIES[mode]); profile = numeric_profile(sheet)
    bases = [base_id(v) for v in sheet.headers]
    if len(bases) != len(set(bases)): raise ValueError("Repeated explicit base IDs; cannot pair modalities")
    return sheet, profile, bases


def save_modality(src, mode, sheet, bases, rows):
    expected = [r["original_base_id"] for r in rows]; lookup = {b:i for i,b in enumerate(bases)}
    if set(expected) != set(bases): raise ValueError("Explicit base-ID set differs from alpha; no row deletion allowed")
    order = np.array([lookup[b] for b in expected]); p = DATA/"sources"/src["source_id"]/(mode+".npz"); p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,X=sheet.X[order],axis=sheet.axis,record_ids=np.asarray([r["sample_id"] for r in rows]),
        entity_ids=np.asarray([r["entity_id"] for r in rows]),labels=np.asarray([r["task_group_id"] for r in rows]),
        origin_group_ids=np.asarray([r["task_group_id"] for r in rows]),batch_ids=np.asarray([r["batch_id"] for r in rows]),
        legacy_match_ids=np.asarray([r["legacy_match_ids"] for r in rows]),original_headers=np.asarray(sheet.headers)[order],
        excel_columns=np.asarray(sheet.columns)[order],reference_headers=np.asarray(sheet.references),reference_values=sheet.reference_values,
        physical_sample_ids=np.asarray([r["entity_id"] for r in rows]))
    return {"status":"READ_COMPLETE","artifact":str(p.relative_to(ROOT)),"artifact_sha256":sha(p),"records":len(rows),
            "axis_points":len(sheet.axis),"profile":numeric_profile(sheet),"alpha_base_id_alignment":"PASS",
            "nonfinite_cells":int((~np.isfinite(sheet.X)).sum())}


def excel_export(src, rows):
    """Compact numeric export without copying blank/style-only cells or names."""
    book = Workbook(write_only=True); count = 0
    for mode in MODALITIES:
        meta = src["modalities"].get(mode,{})
        if meta.get("status") != "READ_COMPLETE": continue
        with np.load(ROOT/meta["artifact"], allow_pickle=False) as z:
            X, axis, ids, ref = z["X"],z["axis"],z["record_ids"],z["reference_values"]
            ws = book.create_sheet(MODALITIES[mode]); ws.append([meta["profile"]["axis_header"],*ids.tolist(),*[f"REF{i+1:03d}" for i in range(len(ref))]])
            for k,a in enumerate(axis):
                values = [float(a),*X[:,k].tolist(),*ref[:,k].tolist()]
                # Excel cannot represent NaN/Inf; blanks are explicitly recorded,
                # while NPZ/raw retain all original numeric/nonfinite values.
                ws.append([v if isinstance(v,str) or np.isfinite(v) else None for v in values])
            count += 1
    if not count: return {"status":"NOT_EXPORTED_NO_READABLE_MODALITIES"}
    p = DATA/"anonymous_workbooks"/(src["source_id"]+"_anonymous.xlsx");p.parent.mkdir(parents=True,exist_ok=True);book.save(p)
    return {"status":"EXPORTED_NUMERIC_ONLY","path":str(p.relative_to(ROOT)),"sha256":sha(p),"sheets":count,
            "style_or_empty_cells_copied":False,"nonfinite_excel_values":"BLANK_WITH_ORIGINAL_PRESERVED_IN_NPZ_AND_RAW"}


def merge(sources, rows_by_source):
    registry = []
    for plant in ("zaoren","huangqi","tianma"):
        for mode in MODALITIES:
            parts=[]; axes=[]; metadata=[]
            for src in sources:
                if src["species"] != plant or src["modalities"].get(mode,{}).get("status") != "READ_COMPLETE": continue
                with np.load(ROOT/src["modalities"][mode]["artifact"],allow_pickle=False) as z:
                    use = np.asarray([r["canonical_for_model"] == "true" and r["alpha_finite"] == "true"
                                      and not json.loads(r["ambiguity_flags"]) for r in rows_by_source[src["source_id"]]])
                    if use.any():
                        parts.append({k:z[k][use] for k in ("X","record_ids","entity_ids","labels","origin_group_ids","batch_ids","legacy_match_ids","physical_sample_ids")})
                        axes.append(z["axis"].copy());metadata.extend([r for r,u in zip(rows_by_source[src["source_id"]],use,strict=True) if u])
            if not parts: continue
            # Do not interpolate heterogeneous axes during intake. Every exact
            # axis family remains a separate, explicitly indexed task interface.
            families=defaultdict(list)
            for i,a in enumerate(axes): families[hashlib.sha256(np.asarray(a,dtype="<f8").tobytes()).hexdigest()].append(i)
            for family,ix in families.items():
                name = mode if len(families)==1 else mode+"_axis_"+family[:12]
                p=DATA/"merged"/plant/(name+".npz");p.parent.mkdir(parents=True,exist_ok=True)
                payload={k:np.concatenate([parts[i][k] for i in ix]) for k in parts[0]};payload["axis"]=axes[ix[0]]
                if len(payload["entity_ids"]) != len(set(payload["entity_ids"].tolist())): raise RuntimeError("Duplicate entity in merged model matrix")
                np.savez_compressed(p,**payload)
                registry.append({"species":plant,"modality":mode,"axis_family":family,"artifact":str(p.relative_to(ROOT)),"sha256":sha(p),
                                 "records":len(payload["X"]),"axis_points":len(payload["axis"]),"groups":dict(Counter(payload["labels"].tolist())),
                                 "nonfinite_cells":int((~np.isfinite(payload["X"])).sum()),"status":"EXPLORATORY_USER_DECLARED_SOURCE_BATCHES"})
    return registry


def main():
    p=argparse.ArgumentParser();p.add_argument("--receipt",type=Path,required=True);p.add_argument("--skip-excel",action="store_true");a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(int(1.5*1024**3),int(1.5*1024**3)))
    start=time.time()
    if DATA.exists() or RAW.exists() or REPORT.exists(): raise FileExistsError("Fresh additive output directories required")
    DATA.mkdir(parents=True);REPORT.mkdir(parents=True)
    receipt=json.loads(a.receipt.read_text());uploads,sources=prepare_sources(receipt)
    full,fixed,legacy_files=legacy_index(); rows=[]; bysource={}; seen_entities={}; next_group={"zaoren":8,"huangqi":5,"tianma":5}
    alpha_payloads={}; source_health={}
    for src in sources:
        sid=src["source_id"];path=ROOT/src["raw_path"];src["source_names_are_not_model_inputs"]=True
        health=workbook_health(path);source_health[sid]=health;src["health"]=health
        try:
            if health["sheet_members"].get(MODALITIES["alpha"]) in health["bad_members"]: raise ValueError("Absorption ZIP member failed CRC")
            sheet,profile,bases=sheet_record(path,"alpha");mask=(sheet.axis>=.1)&(sheet.axis<=1.5)
            hits=[]
            for value in sheet.X:
                all_hits=full.get(fingerprint(sheet.axis,value),[])
                if not all_hits and mask.sum()>3: all_hits=fixed.get(fingerprint(sheet.axis,value,mask),[])
                hits.append(all_hits)
            groups={x["group_id"] for h in hits for x in h if x["species"]==src["species"]}
            all_matched=all(any(x["species"]==src["species"] for x in h) for h in hits)
            if len(groups)==1 and all_matched: group=next(iter(groups));src["legacy_source_group_match"]="ALL_ALPHA_RECORDS_EXACT"
            else:
                prefix={"zaoren":"ZY","huangqi":"HQ","tianma":"TM"}[src["species"]];group=f"{prefix}{next_group[src['species']]:02d}";next_group[src["species"]]+=1
                src["legacy_source_group_match"]="PARTIAL_OR_NONE"
            src["task_group_id"]=group;src["alpha_shape"]=list(sheet.X.shape);local=[]
            for i,(header,col,base,value,hit) in enumerate(zip(sheet.headers,sheet.columns,bases,sheet.X,hits,strict=True),1):
                fp=fingerprint(sheet.axis,value); window=fingerprint(sheet.axis,value,mask); flags=[]
                if src.get("cross_species_alias_conflict"): flags.append("CROSS_SPECIES_BYTE_ALIAS")
                if any(h["species"]!=src["species"] for h in hit): flags.append("CROSS_SPECIES_EXACT_LEGACY_MATCH")
                legacy_ids=sorted({h["record_id"] for h in hit});legacy_groups=sorted({h["group_id"] for h in hit})
                if len(legacy_groups)>1: flags.append("EXACT_MATCH_MULTIPLE_LEGACY_GROUPS")
                # Exact fixed-window identity is the conservative leakage entity;
                # numerical values outside the model window remain preserved.
                entity="MEAS-"+window[:24]
                rid=f"{sid}-S{i:05d}";canonical=entity not in seen_entities
                if canonical: seen_entities[entity]=(rid,src["species"],group)
                else:
                    previous=seen_entities[entity]
                    if previous[1:]!=(src["species"],group): flags.append("CROSS_GROUP_EXACT_INTAKE_MATCH")
                row={"sample_id":rid,"entity_id":entity,"species":src["species"],"task_group_id":group,"source_id":sid,"batch_id":src["batch_id"],
                  "source_workbook_hash":src["sha256"],"original_base_id":base,"original_header":header,"excel_column":col,
                  "legacy_match_ids":json.dumps(legacy_ids,ensure_ascii=False),"legacy_group_ids":json.dumps(legacy_groups),"is_new_unmatched":str(not bool(hit)).lower(),
                  "ambiguity_flags":json.dumps(flags),"canonical_for_model":str(canonical).lower(),"alpha_finite":str(bool(np.isfinite(value).all())).lower(),
                  "alpha_fingerprint":fp,"measurement_fingerprint_0p1_1p5":window,"physical_sample_independence_basis":"USER_DECLARATION_20261104_NOT_LAB_AUDIT",
                  "batch_basis":"ONE_UNIQUE_WORKBOOK_DECLARED_SOURCE_BATCH_BYTE_ALIASES_NOT_NEW_BATCH"}
                rows.append(row);local.append(row)
            bysource[sid]=local;src["modalities"]["alpha"]=save_modality(src,"alpha",sheet,bases,local)
            print(json.dumps({"source":sid,"species":src["species"],"group":group,"alpha_shape":sheet.X.shape,"legacy_records":sum(bool(h) for h in hits),"status":"ALPHA_READY"}),flush=True)
        except Exception as exc:
            src["modalities"]["alpha"]={"status":"BLOCKED_READ_ERROR","error":type(exc).__name__+": "+str(exc)};src["task_group_id"]="";bysource[sid]=[]
            print(json.dumps({"source":sid,"status":"ALPHA_BLOCKED","error":str(exc)}),flush=True)
        assert sha(path)==src["sha256"]
    # Propagate entity-label conflicts to every copy, not only the later row.
    entity_labels=defaultdict(set)
    for row in rows: entity_labels[row["entity_id"]].add((row["species"],row["task_group_id"]))
    for row in rows:
        if len(entity_labels[row["entity_id"]])>1:
            row["ambiguity_flags"]=json.dumps(sorted(set(json.loads(row["ambiguity_flags"])+["CROSS_GROUP_EXACT_INTAKE_MATCH"])))
    csvwrite(DATA/"sample_metadata.csv",rows,FIELDS)
    alpha_merged=merge(sources,bysource)
    jwrite(DATA/"alpha_ready.json",{"status":"ALPHA_READY_FOR_SPLIT_PROTOCOL","metadata":"data/supplemental/teacher_20261104_v1/sample_metadata.csv",
       "metadata_sha256":sha(DATA/"sample_metadata.csv"),"merged":alpha_merged,"records":len(rows),"canonical_entities":len(seen_entities),
       "legacy_records":sum(r["is_new_unmatched"]=="false" for r in rows),"new_unmatched_records":sum(r["is_new_unmatched"]=="true" for r in rows),
       "source_groups":[{k:src.get(k) for k in ("source_id","species","task_group_id","alpha_shape","aliases")} for src in sources],
       "new_confirmatory_claim":"NOT_MADE_PENDING_SPLIT_AND_EXPOSURE_AUDIT","legacy_sources":legacy_files})
    # Alpha interfaces are immutable and ready before the larger time-domain pass.
    for src in sources:
        sid=src["source_id"];path=ROOT/src["raw_path"];health=source_health[sid]
        if not bysource[sid]: continue
        for mode in ("n","time","frequency"):
            member=health["sheet_members"].get(MODALITIES[mode])
            if member is None or member in health["bad_members"]:
                src["modalities"][mode]={"status":"BLOCKED_CORRUPT_OR_MISSING_SHEET","error":health["bad_members"].get(member),"zip_member":member};continue
            try:
                sheet,profile,bases=sheet_record(path,mode);src["modalities"][mode]=save_modality(src,mode,sheet,bases,bysource[sid])
            except Exception as exc: src["modalities"][mode]={"status":"BLOCKED_READ_OR_ALIGNMENT_ERROR","error":type(exc).__name__+": "+str(exc)}
            print(json.dumps({"source":sid,"modality":mode,"status":src["modalities"][mode]["status"]}),flush=True)
        if not a.skip_excel: src["anonymous_workbook"]=excel_export(src,bysource[sid])
        assert sha(path)==src["sha256"]
    # Merge additional modalities separately without touching immutable alpha.
    # Existing alpha files are not rewritten; only new mode paths are created.
    alpha_saved={src["source_id"]:src["modalities"].pop("alpha",None) for src in sources}
    extra_merged=merge(sources,bysource)
    for src in sources:
        if alpha_saved[src["source_id"]] is not None: src["modalities"]["alpha"]=alpha_saved[src["source_id"]]
    jwrite(DATA/"sources.json",{"status":"COMPLETE_EXPLORATORY_WITH_BLOCKED_UPLOADS","sources":sources,"metadata_path":str((DATA/"sample_metadata.csv").relative_to(ROOT)),
          "species_tasks_separate":True,"active_registry_modified":False,"legacy_sources":legacy_files,"blocked_uploads":[u for u in uploads if u["source_id"] is None]})
    jwrite(DATA/"merged_registry.json",{"status":"COMPLETE","datasets":alpha_merged+extra_merged,"sample_metadata_sha256":sha(DATA/"sample_metadata.csv")})
    result={"status":"COMPLETE_WITH_EXPLICIT_BLOCKERS","started_from_receipt_sha256":sha(a.receipt),"source_code_sha256":sha(__file__),"reader_source_sha256":sha(ROOT/"src/herb_spectra/multiplant_database.py"),
       "uploads":len(uploads),"downloaded_uploads":sum(u["source_id"] is not None for u in uploads),"unique_raw_workbooks":len(sources),"byte_alias_uploads":sum(u["status"]=="BYTE_ALIAS" for u in uploads),
       "blocked_uploads":sum(u["source_id"] is None for u in uploads),"all_column_records":len(rows),"unique_exact_measurement_entities":len(seen_entities),
       "legacy_matched_records":sum(r["is_new_unmatched"]=="false" for r in rows),"new_unmatched_records":sum(r["is_new_unmatched"]=="true" for r in rows),
       "ambiguous_records":sum(bool(json.loads(r["ambiguity_flags"])) for r in rows),"elapsed_wall_seconds":time.time()-start,"process_CPU_seconds":time.process_time(),
       "peak_RSS_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"source_bytes_rewritten":False,"model_fits":0,"quantum_evaluations":0,"no_confirmation_claim":True,
       "raw_receipts":"data/incoming/teacher_20261104_v1/upload_receipts.json","sources":"data/supplemental/teacher_20261104_v1/sources.json","registry":"data/supplemental/teacher_20261104_v1/merged_registry.json"}
    jwrite(REPORT/"result.json",result)
    files=[]
    for root in (RAW,DATA,REPORT):
        for path in sorted(root.rglob("*")):
            if path.is_file(): files.append({"path":str(path.relative_to(ROOT)),"sha256":sha(path),"bytes":path.stat().st_size})
    jwrite(REPORT/"artifact_index.json",{"status":"SEALED_INTAKE_NOT_MODEL_EVALUATION","files":files,"count":len(files)})
    print(json.dumps(result),flush=True)


if __name__=="__main__": main()
