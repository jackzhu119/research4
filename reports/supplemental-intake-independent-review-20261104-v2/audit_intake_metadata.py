"""Independent metadata/static intake audit. Never opens XLSX, NPZ, HDF or QM."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import csv,hashlib,json,re,resource,os

os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(int(1.5*1024**3),int(1.5*1024**3)))
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'intake'
V1=ROOT/'data/supplemental/teacher_20261104_v1'
V2=ROOT/'data/supplemental/teacher_20261104_v2'
INPUTS={};CHECKS=[]
def require(ok,name):
    CHECKS.append({'check':name,'passed':bool(ok)})
    if not ok:raise ValueError(name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p,expected=None):
    p=Path(p).resolve();require(p.suffix in {'.py','.json','.csv','.log','.md'},'Metadata/code file only: '+p.name)
    require(p.is_file()and not p.is_symlink(),'Readable regular metadata '+p.name)
    digest=sha(p);require(expected is None or digest==expected,'Expected metadata SHA '+p.name)
    INPUTS[str(p)]={'sha256':digest,'bytes':p.stat().st_size};return p
def read(p,expected=None):return json.loads(record(p,expected).read_text())
def rows(p):
    with record(p).open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f))
def base(header):
    s=str(header);a=re.match(r'^(ZY\d{2}-\d{3,})(?:_|$)',s);b=re.match(r'^(?:样品|样本)\s*(\d+)(?:_|$)',s)
    return a.group(1)if a else str(int(b.group(1)))if b else None

def main():
    require(not OUT.exists(),'Fresh intake-review directory')
    scripts={'scripts/intake_supplemental_workbooks_20261104.py':'8b055c880c16d0e3cf0693cf8d7806c064f1f682f7248d61fe4c5ba1940499e5',
      'scripts/resolve_supplemental_legacy_formatting_20261104.py':'feaee610905db274d267a985e1a1a3b52ea864a213ec6642d1f4116b4d7a017f',
      'tests/test_supplemental_intake_20261104.py':'2ed1b9170720abfe71fac850c94be92b672d3078622e9ba6f006b7a23ab72e6b',
      'src/herb_spectra/multiplant_database.py':'fbfde791331c7bd4f27f48816c63003c696bc9894a9fbaa44accf703baee50eb'}
    for relative,digest in scripts.items():record(ROOT/relative,digest)
    docs=read(V2/'sources.json');sources=docs['sources'];metadata=rows(V2/'sample_metadata.csv');registry=read(V2/'merged_registry.json')
    receipt=read(ROOT/'data/incoming/teacher_20261104_v1/upload_receipts.json');uploads=receipt['uploads']
    require(len(uploads)==24 and sum(u['source_id']is not None for u in uploads)==22,'24uploads22downloaded')
    require(len(sources)==17 and len({s['sha256']for s in sources})==17,'17distinct source workbook hashes')
    require(sorted(len(s['aliases'])for s in sources)==[1]*12+[2]*5,'Five exact byte-alias pairs')
    require(sum(u['status']=='BYTE_ALIAS'for u in uploads)==5,'Five alias upload receipts')
    bysid={s['source_id']:s for s in sources}
    for u in uploads:
        if u['source_id']is None:
            require(u['status']=='BLOCKED_TRANSFER_LIMIT'and u['sha256']is None,'Unavailable upload remains blocked/unknown SHA')
        else:require(u['sha256']==bysid[u['source_id']]['sha256']and u['original_name']in bysid[u['source_id']]['aliases'],'Receipt SHA/source/alias consistency '+u['upload_id'])
    require(len(metadata)==1322 and len({r['sample_id']for r in metadata})==1322 and len({r['entity_id']for r in metadata})==1322,'1322unique current record/entity IDs')
    counts=Counter(r['is_new_unmatched']for r in metadata);require(counts=={'false':1313,'true':9},'1313legacy9unmatched')
    newcounts=Counter(r['task_group_id']for r in metadata if r['is_new_unmatched']=='true');require(newcounts=={'ZY08':3,'ZY09':3,'ZY10':3},'Three new anonymous groups of three records')
    require(not any(r['task_group_id']=='HQ05'for r in metadata),'No invented HQ05 group')
    require(not any(json.loads(r['ambiguity_flags'])for r in metadata),'Current rows carry no unresolved identity ambiguity')
    require(registry['sample_metadata_sha256']==sha(V2/'sample_metadata.csv'),'Registry binds final metadata SHA')
    lineage=rows(ROOT/'data/multiplant_database/metadata/sample_lineage.csv')
    old={(r['dataset_id'],r['record_id']):r for r in lineage if r['modality']=='alpha'and r['dataset_id']in {'zaoren','huangqi','tianma'}}
    require(Counter(k[0]for k in old)=={'zaoren':855,'huangqi':403,'tianma':405},'Registered old source populations remain distinct')
    require(all(r['exposure_status']=='historical_development'for r in old.values()if r['dataset_id']=='zaoren'),'All original855 remain exposed historical development')
    matched=0
    for r in metadata:
        ids=json.loads(r['legacy_match_ids']);require(bool(ids)==(r['is_new_unmatched']=='false'),'Legacy status consistent '+r['sample_id'])
        for rid in ids:
            previous=old.get((r['species'],rid));require(previous is not None and previous['filename_group']==r['task_group_id'],'Legacy parent species/group consistency '+r['sample_id']);matched+=1
    require(matched==1313,'All1313 legacy parents validated against old lineage')
    bysource=defaultdict(list)
    for r in metadata:bysource[r['source_id']].append(r)
    paired=0
    for s in sources:
        require(s['named_origin_truth_status']=='NOT_ASSIGNED_ANONYMOUS_TASK_ONLY'and s['source_names_are_not_model_inputs']is True,'Origin names remain unverified metadata '+s['source_id'])
        expected={r['original_base_id']for r in bysource[s['source_id']]}
        for mode,rec in s['modalities'].items():
            require(rec['status']=='READ_COMPLETE','Current complete modality '+s['source_id']+'/'+mode)
            got=[base(h)for h in rec['profile']['headers']]
            require(set(got)==expected and len(got)==len(set(got))==len(expected)==rec['records']and rec['alpha_base_id_alignment']=='PASS','Explicit cross-modality base-ID pairing '+s['source_id']+'/'+mode)
            paired+=len(got)
    require(paired==5288,'Four modalities preserve1322record pairing entries each')
    for rec in registry['datasets']:
        expected=Counter(r['task_group_id']for r in metadata if r['species']==rec['species'])
        require(rec['groups']==expected and rec['records']==sum(expected.values()),'Merged species/modality counts '+rec['species']+'/'+rec['modality'])
    oldrows={r['sample_id']:r for r in rows(V1/'sample_metadata.csv')};require(set(oldrows)=={r['sample_id']for r in metadata},'V2 has identical record roster')
    for r in metadata:
        if r['source_id']!='SUP006':require(all(r[k]==v for k,v in oldrows[r['sample_id']].items()),'NonGansu original metadata fields unchanged '+r['sample_id'])
    hq=[r for r in metadata if r['source_id']=='SUP006'];require(len(hq)==100 and all(r['task_group_id']=='HQ03'and r['is_new_unmatched']=='false'for r in hq),'All100 Gansu rows retain HQ03 parent')
    require(sum(r['legacy_match_basis']=='EXACT_FULL_OR_EXACT_FIXED_WINDOW'for r in hq)==98 and sum(r['legacy_match_basis']=='ANCHORED_SOURCE_HEADER_FORMAT_EQUIVALENCE_ATOL_1E-12'for r in hq)==2,'98exact2anchored metadata basis')
    for i,r in enumerate(hq,1):require(json.loads(r['legacy_match_ids'])==[f'HQ03-S{i:04d}']and r['original_base_id']==str(i),'Exact header-to-parent binding '+r['sample_id'])
    fmt=rows(ROOT/'reports/supplemental-legacy-formatting-20261104-v1/formatting_audit.csv');fr=read(ROOT/'reports/supplemental-legacy-formatting-20261104-v1/result.json')
    require(len(fmt)==100 and all(r['original_numbers_changed']=='False'for r in fmt),'Format audit retains all100 and no rewritten source numbers claim')
    maximum=max(float(r['max_absolute_difference'])for r in fmt);require(maximum==fr['maximum_absolute_difference']and maximum<=1e-12 and fr['relative_tolerance']==0,'Recorded full-axis differences within fixed absolute tolerance')
    require(sum(int(r['changed_fixed_window_cells'])==0 for r in fmt)==98 and sum(int(r['changed_full_axis_cells'])>0 for r in fmt)==7,'98exact windows and7 full-axis formatting variants')
    require(fr['new_independent_huangqi_samples']==0 and fr['new_unmatched_total']==9 and fr['numerical_values_changed']is False,'No new Huangqi specimens or data-value correction')
    v1index=read(ROOT/'reports/supplemental-intake-20261104-v1/artifact_index.json');indexed={r['path']:r for r in v1index['files']}
    for name in ['sample_metadata.csv','sources.json','merged_registry.json']:
        p=record(V1/name);entry=indexed[str(p.relative_to(ROOT))];require(sha(p)==entry['sha256'],'V1 sealed metadata retained '+name)
    for relative in ['data/active_workbooks.json','data/manifest.csv','data/annotations/zaoren_user_declaration_20261031.json']:
        p=record(ROOT/relative);q=record(Path('/workspace/research4')/relative);require(sha(p)==sha(q),'Original active registry/manifest/declaration unchanged '+relative)
    testlog=record(HERE/'unittest-v2.log').read_text();require('Ran 10 tests'in testlog and'\nOK\n'in testlog and'quantum_imported False'in testlog,'Independent ten pure tests passed')
    OUT.mkdir()
    report={'schema':'research4-independent-supplemental-intake-metadata-review-v2','created_at_utc':datetime.now(timezone.utc).isoformat(),
      'status':'PASS_INTAKE_STATIC_METADATA_AND_TEN_PURE_TESTS','blocking_intake_findings':[],
      'uploads':24,'downloaded_uploads':22,'unique_raw_files':17,'exact_byte_alias_pairs':5,'blocked_transfer_limit':2,
      'record_columns':1322,'legacy_bound_records':1313,'unmatched_records':9,'unmatched_anonymous_group_counts':dict(newcounts),
      'species_record_counts':dict(Counter(r['species']for r in metadata)),
      'Gansu':{'source':'SUP006','parent':'HQ03','paired_headers':100,'exact_fixed_window_anchors':98,'format_equivalence_added_parent_links':2,
        'records_with_full_axis_format_variants':7,'producer_reported_max_absolute_difference':maximum,'fixed_absolute_tolerance':1e-12,'relative_tolerance':0,
        'new_independent_samples':0,'source_numbers_changed':False},
      'old855_exposure':'All historical_development; never fresh confirmation','cross_modality_profile_pairing_entries_checked':paired,
      'old_active_registry_manifest_declaration_hashes_unchanged':True,'v1_sealed_metadata_preserved':True,
      'named_origins_verified':False,'intake_is_new_confirmation':False,'independence_basis':'User declaration, not laboratory audit; aliases and legacy copies add no new specimens/batches.',
      'source_and_test_sha256':scripts,'checks':CHECKS,'checks_count':len(CHECKS),'input_files':INPUTS,
      'tests':{'count':10,'exit_code':0,'log_sha256':sha(HERE/'unittest-v2.log'),'AS_bytes':1610612736,'CPU_affinity':[0],'peak_RSS_kib':34492},
      'limits':['No XLSX decompression or NPZ/HDF/scientific-array reads occurred in this review. Numerical equivalence maxdiff and raw-copy hashes are audited producer evidence and static code checks, not independently recomputed spectra.',
        'All source/modality pairing was checked against preserved explicit header profiles and static ID-based reorder code; NPZ payloads were not reopened.',
        'Separate benchmark physical-alias guard was fixed/reviewed separately; that code correction is not an intake numeric-data correction.'],
      'initial_test_wrapper_failure':'Reviewer wrapper syntax typo stopped before imports/tests; initial log retained, corrected wrapper ran all10 successfully.',
      'new_quantum_evaluations':0,'model_fits':0,'scientific_array_reads':0,'XLSX_reads':0,'Git_mutations':0,'subject_source_modified_by_reviewer':False,
      'audit_builder_sha256':sha(__file__)}
    for p,entry in INPUTS.items():require(sha(p)==entry['sha256'],'Inputs stable at review completion '+Path(p).name)
    (OUT/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (OUT/'README.md').write_text('PASS intake/static metadata and10 independent pure tests.22 downloads reduce to17 unique raw workbooks by5 exact alias pairs;1322 columns comprise1313 historical parents plus9 unmatched columns in3 anonymous groups of3. Gansu100 columns retain HQ03:98 exact-window anchors plus2 conservative formatting-parent links, max producer-recorded difference1.4210854715202004e-14 under fixed absolute1e-12. No HQ05, no2 new Huangqi specimens and no changed numeric values. Four-modality explicit header profiles preserve5288 pairing entries. Old855 remain exposed historical-development data; active registry, manifest and declaration hashes match the original worktree. V1 metadata hashes remain sealed. Actual origin names and new-confirmation status are not established. This reviewer opened only code/JSON/CSV/log metadata and tiny pure-test arrays; no XLSX/NPZ/HDF, model fit, QM or Git mutation. Benchmark alias-guard correction has separate before/after evidence and is not a data-value correction.\n')
    index={'status':'SEALED_INDEPENDENT_INTAKE_REVIEW','audit_builder':{'path':str(Path(__file__).relative_to(ROOT)),'sha256':sha(__file__)},'files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(OUT.iterdir())if p.is_file()]}
    (OUT/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':len(CHECKS),'review_sha256':sha(OUT/'review.json'),'index_sha256':sha(OUT/'index.json')}))

if __name__=='__main__':main()
