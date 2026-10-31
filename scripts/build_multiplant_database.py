"""Build a new read-only multimodal database; never replace original workbooks."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from herb_spectra.multiplant_database import (
    MODALITIES, RawSheet, align_sheets, base_header, digest, numeric_profile,
    read_raw_sheet, thickness_from_header, workbook_health, numeric_row_hash, validate_zaoren_row_bindings,
)


ROOT = Path(__file__).resolve().parents[1]
LINEAGE_FIELDS = ["dataset_id","record_id","physical_sample_id","parent_sample_id","preparation_id",
                  "batch_id","measurement_id","modality","label","source_file","source_hash",
                  "independence_evidence","representation_pair_id","original_header","excel_column",
                  "thickness_mm","filename_group","exposure_status","quality_status"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def read_source(item):
    path = ROOT/item["source_file"]
    before = digest(path)
    assert before == item["expected_sha256"]
    health = workbook_health(path)
    records = []
    destination = ROOT/"outputs/multiplant-primary-20261031"/item["source_id"]
    destination.mkdir(parents=True, exist_ok=True)
    for mode, name in MODALITIES.items():
        member = health["sheet_members"].get(name)
        entry = {"modality":mode,"sheet":name,"zip_member":member}
        if member is None or member in health["bad_members"]:
            entry.update(status="BLOCKED_CORRUPT_OR_MISSING_SHEET", error=health["bad_members"].get(member))
            records.append(entry); continue
        try:
            sheet = read_raw_sheet(path,name)
            profile = numeric_profile(sheet)
            bases = [base_header(h,item["dataset"]) for h in sheet.headers]
            target = destination/(mode+".npz")
            np.savez_compressed(target,X=sheet.X,axis=sheet.axis,headers=sheet.headers,
                                bases=bases,column_numbers=sheet.columns,reference_values=sheet.reference_values)
            entry.update(status="READ_COMPLETE", artifact=str(target.relative_to(ROOT)),
                         artifact_sha256=digest(target),profile=profile)
        except Exception as exc:
            entry.update(status="BLOCKED_READ_ERROR", error=type(exc).__name__+": "+str(exc))
        records.append(entry)
        print(json.dumps({"source":item["source_id"],"modality":mode,"status":entry["status"]}),flush=True)
    assert digest(path) == before
    return {**item,"sha256":before,"health":health,"sheets":records,"input_bytes_unchanged":True,
            "conversion":"actual_read_all_cells_in_this_run"}


def source_payload(entry):
    with np.load(ROOT/entry["artifact"], allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def initialize_sources(secondary):
    registry=json.loads((ROOT/"data/active_workbooks.json").read_text())
    primary=[{"dataset":"zaoren","source_id":s["origin_group_id"],"source_file":s["path"],
              "expected_sha256":s["sha256"],"filename_group":s["origin_group_id"]}
             for s in registry["workbooks"] if s["dataset"]=="zaoren"]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(read_source,primary))
    if secondary:
        manifest=json.loads(secondary.read_text())
        assert manifest["source_workbooks_rewritten"]==0
        for src in manifest["source_workbooks"]:
            assert digest(ROOT/src["source_file"]) == src["sha256"] == src["expected_sha256"]
            entries=[]
            for old in src["sheets"]:
                entry={"modality":old["modality"],"sheet":old["sheet"],"status":old["status"],
                       "zip_member":old.get("zip_member"),"error":old.get("error")}
                if old["status"]=="READ_COMPLETE":
                    assert digest(ROOT/old["artifact"])==old["artifact_sha256"]
                    entry.update(artifact=old["artifact"],artifact_sha256=old["artifact_sha256"],
                                 profile={"record_count":old["records"],"axis_points":old["axis_points"],
                                  "axis_header":old["coordinate_header"],"axis_unit":old["coordinate_unit"],
                                  "axis_min":old["axis_min"],"axis_max":old["axis_max"],
                                  "axis_finite":old["axis_finite"],"axis_strictly_increasing":old["axis_strictly_increasing"],
                                  "nonfinite_cells":old["nonfinite_cells"],"negative_cells":old["negative_cells"],
                                  "headers":old["sample_headers"],"excel_columns":old["sample_excel_columns"],
                                  "reference_headers":old["reference_column_headers"],"physical_unit_verified":False,
                                  "axis_sha256":old["axis_sha256"]})
                entries.append(entry)
            results.append({"dataset":src["dataset"],"source_id":src["source_id"],"source_file":src["source_file"],
                            "sha256":src["sha256"],"sheets":entries,"input_bytes_unchanged":src["source_bytes_unchanged"],
                            "filename_group":src["source_id"],"health":{"container_crc_pass":src["container_crc_pass"],
                             "bad_members":{m["member"]:m for m in src["zip_members"] if m["crc"]!="pass"}},
                            "conversion":"same_task_fresh_read_raw_conversion_cache_verified_by_input_and_artifact_SHA",
                            "reader_manifest_sha256":digest(secondary)})
    else:
        receipt=json.loads((ROOT/"data/incoming/teacher_20261031_v1/receipts.json").read_text())
        extra=[]
        for i,s in enumerate([s for s in registry["workbooks"] if s["dataset"]=="huangqi"]):
            extra.append({"dataset":"huangqi","source_id":f"HQ{i+1:02d}","source_file":s["path"],"expected_sha256":s["sha256"],"filename_group":f"HQ{i+1:02d}"})
        for i,s in enumerate([s for s in receipt["uploads"] if s["dataset"]=="gastrodia"]):
            extra.append({"dataset":"tianma","source_id":f"TM{i+1:02d}","source_file":s["archive_path"],"expected_sha256":s["sha256"],"filename_group":f"TM{i+1:02d}"})
        with ProcessPoolExecutor(max_workers=2) as pool: results.extend(pool.map(read_source,extra))
    return results


def build(output, secondary=None):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError("New empty database directory required")
    output.mkdir(parents=True,exist_ok=True)
    sources=initialize_sources(secondary)
    write_json(output/"data_quality/source_workbooks.json",sources)
    collected=defaultdict(lambda:defaultdict(list));lineage=[];inventory=[];reference_arrays={}
    for src in sources:
        valid={e["modality"]:e for e in src["sheets"] if e["status"]=="READ_COMPLETE"}
        payloads={m:source_payload(e) for m,e in valid.items()}
        if not payloads: continue
        bases={m:[base_header(h,src["dataset"]) for h in p["headers"]] for m,p in payloads.items()}
        main=bases.get("alpha",next(iter(bases.values())))
        if len(main)!=len(set(main)):raise ValueError("Repeated sample base in primary modality")
        for mode,p in payloads.items():
            if len(bases[mode])!=len(set(bases[mode])) or set(bases[mode])!=set(main):
                raise ValueError(f"Ambiguous or missing modality correspondence: {src['source_id']}/{mode}")
            positions={v:i for i,v in enumerate(bases[mode])};order=[positions[b] for b in main]
            ids=main if src["dataset"]=="zaoren" else [f"{src['source_id']}-S{int(b):04d}" for b in main]
            X=p["X"][order];headers=p["headers"][order];columns=p["column_numbers"][order]
            labels=np.asarray([src["source_id"] if src["dataset"]=="zaoren" else ""]*len(ids))
            collected[src["dataset"]][mode].append((p["axis"],X,np.asarray(ids),labels,np.asarray([src["filename_group"]]*len(ids))))
            if "reference_values" in p and p["reference_values"].size:reference_arrays[src["source_id"]+"_"+mode]=p["reference_values"]
            for rid,header,col in zip(ids,headers,columns,strict=True):
                lineage.append({"dataset_id":src["dataset"],"record_id":str(rid),"physical_sample_id":str(rid) if src["dataset"]=="zaoren" else "",
                  "parent_sample_id":"","preparation_id":"","batch_id":"","measurement_id":"","modality":mode,
                  "label":src["source_id"] if src["dataset"]=="zaoren" else "","source_file":src["source_file"],"source_hash":src["sha256"],
                  "independence_evidence":"user_declared_independent_20261031" if src["dataset"]=="zaoren" else "unknown",
                  "representation_pair_id":str(rid),"original_header":str(header),"excel_column":str(col),
                  "thickness_mm":thickness_from_header(header) if thickness_from_header(header) is not None else "",
                  "filename_group":src["filename_group"],"exposure_status":"historical_development" if src["dataset"] in {"zaoren","huangqi"} else "new_development",
                  "quality_status":"EXPLORATORY_USER_DECLARED_SINGLE_BATCH" if src["dataset"]=="zaoren" else "BLOCKED_INDEPENDENCE_UNKNOWN"})
    combined={}
    for dataset,modes in collected.items():
        combined[dataset]={}
        for mode,parts in modes.items():
            if any(not np.array_equal(parts[0][0],p[0]) for p in parts):
                raise ValueError(f"Axes differ; no automatic interpolation: {dataset}/{mode}")
            axis=parts[0][0];X=np.concatenate([p[1] for p in parts]);ids=np.concatenate([p[2] for p in parts]);labels=np.concatenate([p[3] for p in parts]);groups=np.concatenate([p[4] for p in parts])
            assert len(ids)==len(set(ids.tolist()))
            target=output/dataset/(mode+".npz");target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,X=X,axis=axis,record_ids=ids,labels=labels,filename_groups=groups)
            combined[dataset][mode]={'X':X,'axis':axis,'ids':ids,'labels':labels,'target':target}
            inventory.append({'dataset_id':dataset,'modality':mode,'records':len(ids),'axis_points':len(axis),'physical_sample_count_declared':len(ids) if dataset=='zaoren' else '',
                              'independence_basis':'user_declaration_only' if dataset=='zaoren' else 'unknown','scientific_labels_supplied':dataset=='zaoren',
                              'file':str(target.relative_to(ROOT)),'sha256':digest(target),'nonfinite_cells':int((~np.isfinite(X)).sum()),'negative_cells':int((X<0).sum()),
                              'status':'EXPLORATORY' if dataset=='zaoren' else 'BLOCKED_INDEPENDENCE_UNKNOWN'})
    z=combined['zaoren'];reference=z['alpha']['ids'];aligned={'record_ids':reference,'y':z['alpha']['labels'],'physical_sample_ids':reference}
    alignment={}
    for mode,p in z.items():
        assert set(p['ids'])==set(reference),mode
        order={v:i for i,v in enumerate(p['ids'])};X=p['X'][[order[v] for v in reference]];axis=p['axis']
        if mode in {'alpha','n'}:
            mask=(axis>=.1)&(axis<=1.5);X=X[:,mask];axis=axis[mask]
        assert np.isfinite(X).all(),mode
        aligned['X_'+mode]=X;aligned['axis_'+mode]=axis
        alignment[mode]={'records':len(X),'points':len(axis),'ids_match_alpha_exactly':True,'headers_paired_by_base_ID_not_column_order':True}
    aligned_target=output/'zaoren/aligned_modalities.npz';np.savez_compressed(aligned_target,**aligned)
    previous=ROOT/'outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz'
    previous_match=None
    if previous.exists():
        with np.load(previous,allow_pickle=False) as old:
            assert np.array_equal(reference,old['record_ids']);assert np.array_equal(aligned['X_alpha'],old['X_alpha']);assert np.array_equal(aligned['X_n'],old['X_n'])
        previous_match=True
    write_json(output/'zaoren/alignment.json',{'status':'EXPLORATORY','modalities':alignment,'paired_records':len(reference),'old_alpha_n_values_and_ids_exactly_reproduced':previous_match,'historical_cache_comparison_status':'PASS' if previous_match else 'NOT_RUN_CACHE_NOT_PRESENT','aligned_sha256':digest(aligned_target),'full_frequency_range_is_not_optimized_from_labels':True})
    write_csv(output/'dataset_inventory.csv',inventory)
    write_csv(output/'metadata/sample_lineage.csv',lineage,LINEAGE_FIELDS)
    target=ROOT/'data/metadata/sample_lineage.csv';target.parent.mkdir(exist_ok=True);shutil.copyfile(output/'metadata/sample_lineage.csv',target)
    if reference_arrays:np.savez_compressed(output/'metadata/reference_waveforms.npz',**reference_arrays)
    write_json(output/'metadata/readiness.json',{'zaoren':'EXPLORATORY_USER_DECLARED_SINGLE_BATCH','huangqi':'BLOCKED_INDEPENDENCE_LABEL_UNIT_BATCH_UNKNOWN','tianma':'BLOCKED_INDEPENDENCE_LABEL_UNIT_BATCH_UNKNOWN','wheat':'BLOCKED_AVERAGE_CURVES_AND_UNKNOWN_LABELS','confirmation_dataset':'MISSING','new_model_fits':0})
    return sources,combined,lineage,inventory


def finalize(output,sources,combined,lineage,inventory,secondary=None):
    """Add averaged wheat, fixed historical folds and nonselective QC only."""
    output=Path(output).resolve()
    receipt=json.loads((ROOT/'data/incoming/teacher_20261031_v1/receipts.json').read_text())
    wheat=[]
    for number,item in enumerate(sorted((r for r in receipt['uploads'] if r['dataset']=='wheat'),key=lambda r:(r['file_name']!='小麦.xlsx',r['file_name'])),1):
        path=ROOT/item['archive_path'];assert digest(path)==item['sha256']
        raw=read_raw_sheet(path,'Sheet1');sid=f'WH{number:02d}'
        ids=np.asarray([f'{sid}-M{i+1:03d}' for i in range(len(raw.X))])
        target=output/'wheat'/(sid+'_averaged_unknown_quantity.npz');target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,X=raw.X,axis=raw.axis,record_ids=ids,labels=np.asarray(['']*len(ids)),
                            original_headers=raw.headers,reference_values=raw.reference_values)
        profile=numeric_profile(raw);profile.update(source_file=item['archive_path'],source_hash=item['sha256'],dataset_id='wheat',source_id=sid,status='BLOCKED_AVERAGED_CURVES_UNKNOWN_LABELS')
        wheat.append(profile)
        inventory.append({'dataset_id':'wheat','modality':sid+'_averaged_unknown_quantity','records':len(ids),'axis_points':len(raw.axis),'physical_sample_count_declared':'',
                          'independence_basis':'unknown_averages','scientific_labels_supplied':False,'file':str(target.relative_to(ROOT)),'sha256':digest(target),
                          'nonfinite_cells':int((~np.isfinite(raw.X)).sum()),'negative_cells':int((raw.X<0).sum()),'status':'BLOCKED_AVERAGES_LABELS_UNKNOWN'})
        for rid,head,col in zip(ids,raw.headers,raw.columns,strict=True):
            lineage.append(dict(zip(LINEAGE_FIELDS,['wheat',str(rid),'','','','','','averaged_unknown_quantity','',item['archive_path'],item['sha256'],'unknown',
                                                   str(rid),head,str(col),'',sid,'new_development','BLOCKED_AVERAGES_LABELS_UNKNOWN'])))
    write_json(output/'data_quality/wheat_workbooks.json',wheat)
    # Bind original numeric rows and original columns to all four final matrices.
    bindings=[]
    for src in [s for s in sources if s['dataset']=='zaoren']:
        for entry in [s for s in src['sheets'] if s['status']=='READ_COMPLETE']:
            mode=entry['modality'];original_values=source_payload(entry);destination=combined['zaoren'][mode]
            destination_index={str(v):i for i,v in enumerate(destination['ids'])}
            mask=(original_values['axis']>=.1)&(original_values['axis']<=1.5) if mode in {'alpha','n'} else np.ones(len(original_values['axis']),dtype=bool)
            for source_index,(header,col,row) in enumerate(zip(original_values['headers'],original_values['column_numbers'],original_values['X'],strict=True)):
                rid=base_header(header,'zaoren');index=destination_index[rid]
                assert np.array_equal(row,destination['X'][index],equal_nan=True)
                bindings.append({'dataset_id':'zaoren','modality':mode,'record_id':rid,'source_file':src['source_file'],'source_hash':src['sha256'],
                                 'source_sheet':entry['sheet'],'original_header':str(header),'excel_column':str(col),'source_spectrum_index':source_index,
                                 'destination_row_index':index,'source_numeric_sha256':numeric_row_hash(row),'full_matrix_row_sha256':numeric_row_hash(destination['X'][index]),
                                 'model_matrix_row_sha256':numeric_row_hash(row[mask]),'source_axis_sha256':numeric_row_hash(original_values['axis']),
                                 'model_axis_sha256':numeric_row_hash(original_values['axis'][mask])})
    binding_path=output/'zaoren/modality_alignment.csv';write_csv(binding_path,bindings)
    with np.load(output/'zaoren/aligned_modalities.npz',allow_pickle=False) as archive:
        binding_result=validate_zaoren_row_bindings({k:archive[k] for k in archive.files},binding_path)
    alignment_path=output/'zaoren/alignment.json';alignment=json.loads(alignment_path.read_text())
    alignment.update(original_sheet_value_binding_file=str(binding_path.relative_to(ROOT)),original_sheet_value_binding_sha256=digest(binding_path),
                     original_sheet_value_binding_rows=len(bindings),value_binding_validation=binding_result,
                     original_headers_and_columns_preserved=True,full_source_row_hash_matches_full_database_row=True)
    write_json(alignment_path,alignment)
    # The old corrupt Gansu upload is an alias, never an additional cohort.
    aliases=[]
    for item in receipt['uploads']:
        rec={k:item.get(k) for k in ['file_name','archive_path','sha256','status','identical_to_active','same_named_active_file','same_named_active_sha256']}
        assert digest(ROOT/item['archive_path'])==item['sha256']
        if item['dataset']=='huangqi' and not item['identical_to_active']:
            rec['health']=workbook_health(ROOT/item['archive_path']);rec['sheets']=[]
            for mode,name in MODALITIES.items():
                member=rec['health']['sheet_members'].get(name)
                if member in rec['health']['bad_members']:
                    rec['sheets'].append({'modality':mode,'status':'BLOCKED_CORRUPT_SHEET'});continue
                try:
                    raw=read_raw_sheet(ROOT/item['archive_path'],name)
                    # Re-export equality is tested against the active raw conversion, not label scores.
                    active=next(s for s in sources if s['source_file']==item['same_named_active_file'])
                    candidate=next((e for e in active['sheets'] if e['modality']==mode and e['status']=='READ_COMPLETE'),None)
                    cached=source_payload(candidate) if candidate else None
                    same=bool(cached is not None and np.array_equal(raw.axis,cached['axis']) and np.array_equal(raw.X,cached['X'],equal_nan=True))
                    rec['sheets'].append({'modality':mode,'status':'READ_COMPLETE_ALIAS_ONLY','record_count':len(raw.X),'axis_points':len(raw.axis),'numeric_equal_active':same})
                except Exception as exc: rec['sheets'].append({'modality':mode,'status':'BLOCKED_READ_ERROR','error':str(exc)})
        aliases.append(rec)
    write_json(output/'data_quality/incoming_aliases.json',aliases)
    write_csv(output/'dataset_inventory.csv',inventory)
    write_csv(output/'metadata/sample_lineage.csv',lineage,LINEAGE_FIELDS)
    shutil.copyfile(output/'metadata/sample_lineage.csv',ROOT/'data/metadata/sample_lineage.csv')
    # Exact and near-duplicate diagnostics never delete or choose frequencies.
    buckets=defaultdict(list);rows=[];ids=[];datasets=[];axis=None
    for dataset,modes in combined.items():
        p=modes['alpha']
        if axis is None:axis=p['axis']
        assert np.array_equal(axis,p['axis'])
        for rid,row in zip(p['ids'],p['X'],strict=True):
            value=np.asarray(row,dtype='<f8').copy();value[value==0]=0
            buckets[hashlib.sha256(value.tobytes()).hexdigest()].append({'dataset_id':dataset,'record_id':str(rid)})
            rows.append(row);ids.append(str(rid));datasets.append(dataset)
    duplicate=[v for v in buckets.values() if len(v)>1]
    matrix=np.asarray(rows);centered=matrix-matrix.mean(axis=1,keepdims=True);norm=np.linalg.norm(centered,axis=1)
    normalized=np.divide(centered,norm[:,None],out=np.zeros_like(centered),where=norm[:,None]>np.finfo(float).eps)
    similarity=np.clip(normalized@normalized.T,-1,1);a,b=np.triu_indices(len(matrix),1);scores=similarity[a,b]
    selected=np.flatnonzero(scores>=.99999);rank=selected[np.argsort(scores[selected])[::-1][:30]]
    write_json(output/'data_quality/duplicates.json',{'scope':'complete alpha grid; QC only, no automatic removal or feature choice','record_count':len(matrix),
        'exact_groups':duplicate,'cross_plant_exact_groups':[g for g in duplicate if len({r['dataset_id'] for r in g})>1],
        'near_threshold':.99999,'near_pair_count':len(selected),'cross_plant_near_pair_count':sum(datasets[a[k]]!=datasets[b[k]] for k in selected),
        'top_near_pairs':[{'record_ids':[ids[a[k]],ids[b[k]]],'datasets':[datasets[a[k]],datasets[b[k]]],'correlation':float(scores[k])} for k in rank],
        'similarity_is_not_independence_evidence':True,'records_removed':0})
    original=ROOT/'reports/approved-zaoren/2026-10-31/experiments/alpha_primary/splits.json'
    old=json.loads(original.read_text());mapping={str(i):str(y) for i,y in zip(combined['zaoren']['alpha']['ids'],combined['zaoren']['alpha']['labels'],strict=True)}
    folds=[]
    for fold in old:
        train,test=fold['train_sample_ids'],fold['test_sample_ids'];assert not set(train)&set(test)
        assert set(train)|set(test)==set(mapping)
        inners=[]
        for j,inner in enumerate(fold['inner_folds']):
            itrain=inner['train_sample_ids'];valid=inner.get('validation_sample_ids',inner.get('test_sample_ids'))
            assert valid is not None and not (set(itrain)&set(valid) or set(itrain)&set(test) or set(valid)&set(test))
            assert set(itrain)|set(valid)==set(train)
            counts={role:dict(Counter(mapping[r] for r in members)) for role,members in [('train',itrain),('validation',valid)]}
            assert all(len(c)==7 and min(c.values())>=2 for c in counts.values())
            inners.append({'inner_fold':j,'train_ids':itrain,'validation_ids':valid,'class_counts':counts})
        folds.append({'outer_fold':fold['fold'],'train_ids':train,'test_ids':test,'inner_folds':inners,
                      'class_counts':{role:dict(Counter(mapping[r] for r in members)) for role,members in [('train',train),('test',test)]}})
    splitdir=ROOT/'data/splits/upgrade-20261031-v1';splitdir.mkdir(parents=True,exist_ok=True)
    write_json(splitdir/'split_manifest.json',{'protocol_id':'upgrade-20261031-v1','dataset_id':'zaoren','dataset_version':'2026-10-31-zaoren-numbered-v1',
        'status':'EXPLORATORY_USER_DECLARED_INDEPENDENCE_SINGLE_BATCH','source_split_file':str(original.relative_to(ROOT)),'source_split_sha256':digest(original),
        'split_seed':20261031,'outer_folds':folds,'record_ids':list(mapping),'same_as_historical_split':True,'confirmation_data_available':False,
        'independence_evidence':'user_declared_independent_20261031; no raw sampling log or common-ancestor verification','cross_batch_identifiable':False})
    write_json(output/'metadata/build_manifest.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'builder_sha256':digest(Path(__file__)),
        'reader_sha256':digest(ROOT/'src/herb_spectra/multiplant_database.py'),'secondary_raw_cache_manifest_sha256':digest(secondary) if secondary else None,
        'input_source_files':len(sources)+len(wheat)+1,'selected_main_source_workbooks':len(sources)+len(wheat),'old_corrupt_alias_not_added_as_samples':True,
        'lineage_rows':len(lineage),'unique_zaoren_record_ids':855,'huangqi_alpha_records':403,'tianma_alpha_records':405,
        'wheat_average_columns':sum(r['record_count'] for r in wheat),'records_removed':0,'labels_or_original_values_changed':False,
        'new_model_fits':0,'no_confirmation_dataset':True,'git_starting_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    if secondary:
        destination=output/'data_quality/secondary_read_evidence';destination.mkdir(parents=True,exist_ok=True)
        for name in ['manifest.json','read_secondary.py','independent_reader_validation.json']:
            p=secondary.parent/name
            if p.exists():shutil.copyfile(p,destination/name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'data/multiplant_database');parser.add_argument('--secondary-cache',type=Path)
    args=parser.parse_args();sources,combined,lineage,inventory=build(args.output,args.secondary_cache)
    finalize(args.output,sources,combined,lineage,inventory,args.secondary_cache)
    print(json.dumps({'source_workbooks':len(sources),'datasets':list(combined),'lineage_rows':len(lineage),'modalities':len(inventory),'new_model_fits':0}),flush=True)
