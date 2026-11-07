"""Read-only source/destination byte and genuine closed-V5 provenance check."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
ARCHIVE=ROOT/'reports/pyscf-spinosin-resumable-20261104-v5/clean-149-checkpoint-archive'
TARGET=ROOT/'outputs/pyscf-spinosin-staged-20261105-v6b/spinosin-C001-healthy149'
SEAL=ARCHIVE/'seals/v5-spinosin-clean-prefix-149-001.json'
RECEIPT=ROOT/'reports/pyscf-spinosin-staged-20261105-v6b/resource-protocol/healthy149-seed-copy-20261105.json'
RESULT=RECEIPT.with_name('healthy149-seed-copy-independent-check-20261105.json')
SEAL_SHA='be974fbb6718ed2e52200b2cce8896196e756ebb87a825891d889cde2b5d499f'
RECEIPT_SHA='0f3ffe706bc2285689b38799ab9ce2299916227343589aedd5cdc058e3c18e6b'
SNAPSHOT_SHA='3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    assert not RESULT.exists()and not(OUT/'index.json').exists()
    assert sha(SEAL)==SEAL_SHA and sha(RECEIPT)==RECEIPT_SHA
    seal=json.loads(SEAL.read_text());receipt=json.loads(RECEIPT.read_text())
    assert receipt['target_cache']==str(TARGET)and receipt['completed_units']==149 and receipt['unit149_included']is False
    assert receipt['forensic_release_used']is False and receipt['new_Q_payload_launched']is False
    refs={**receipt['input_references']}
    for entry in list(refs.values())+receipt['archive_dependencies']:assert sha(entry['path'])==entry['sha256']
    header=json.loads((ARCHIVE/'header.json').read_text());assert header['binding']['stable_snapshot']['snapshot_manifest_sha256']==SNAPSHOT_SHA
    files=[];records=[]
    for relative,digest in seal['files'].items():
        src=ARCHIVE/relative;dest=TARGET/relative
        assert src.is_file()and dest.is_file()and not src.is_symlink()and not dest.is_symlink()
        assert sha(src)==sha(dest)==digest and src.stat().st_size==dest.stat().st_size
        assert (src.stat().st_dev,src.stat().st_ino)!=(dest.stat().st_dev,dest.stat().st_ino)
        files.append({'relative_path':relative,'bytes':dest.stat().st_size,'sha256':digest})
        if relative.endswith('/complete.json'):records.append((dest,json.loads(dest.read_text())))
    assert len(files)==373 and sum(f['bytes']for f in files)==1018125
    actual={str(p.relative_to(TARGET))for p in TARGET.rglob('*')if p.is_file()}
    assert actual==set(seal['files'])|{'cache.lock'}and(TARGET/'cache.lock').stat().st_size==0
    assert not(TARGET/'seals').exists()and not any(p.name.startswith('000149-')for p in(TARGET/'units').iterdir())
    assert sorted(r['index']for _,r in records)==list(range(149))
    ledgerpath=Path(refs['closed_V5_ledger']['path']);ledger=json.loads(ledgerpath.read_text());rows={r['phase']:r for r in ledger['phases']};producers={};counts=Counter()
    for p,r in records:
        ev=r['execution_evidence'];supervision=ev['resource_supervision'];proof=supervision['proof'];phase=proof['phase'];row=rows[phase];counts[phase]+=1
        assert ev['schema']=='research4-rhf-unit-producer-v1'and ev['phase']=='hessian-segment'
        assert ev['snapshot_manifest_sha256']==SNAPSHOT_SHA and ev['source_worker']['sha256']=='b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190'
        assert r['header_sha256']==sha(TARGET/'header.json')and r['binding_sha256']==header['binding_sha256']
        assert row['status']=='EXITED'and row['exit_code']==0 and all(row[k]is False for k in ['wall_terminated','memory_terminated','cpu_terminated','monitor_failed','hard_killed'])
        assert proof['schema']=='research4-pyscf-actual-cpu-supervisor-proof-v5'and proof['child_pid']==row['pid']
        assert proof['supervisor_script_sha256']=='33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee'
        if phase not in producers:
            assert sha(row['supervisor_proof_path'])==row['supervisor_proof_sha256']==supervision['proof_sha256']
            assert json.loads(Path(row['supervisor_proof_path']).read_text())==proof and sha(row['log'])==row['log_sha256']
            resultpath=Path(ev['output_directory'])/'result.json';res=json.loads(resultpath.read_text())
            assert res['status']=='CHECKPOINT_SEGMENT_COMPLETE'and res['stable_MF_state_unchanged']is True
            assert res['native_direct_driver_control']['original_driver_restored']is True and res['native_direct_driver_control']['restoration_failures']==0
            assert res['resource_supervision']==supervision and res['source_worker']==ev['source_worker']and res['snapshot_manifest_sha256']==SNAPSHOT_SHA
            producers[phase]={'phase':phase,'proof_sha256':supervision['proof_sha256'],'result_sha256':sha(resultpath),'actual_CPU_seconds':row['actual_cpu_seconds']}
    assert len(producers)==16 and receipt['unique_healthy_V5_producers']==16
    forbidden=[n for n in sys.modules if n.split('.')[0]in {'numpy','scipy','pyscf','h5py'}];assert not forbidden
    assert sha(RECEIPT)==RECEIPT_SHA and sha(SEAL)==SEAL_SHA
    report={'schema':'research4-independent-healthy149-seed-copy-check-v6b-v1','status':'PASS_BYTE_IDENTICAL_HEALTHY149_SEED_NO_QUANTUM',
      'observed_at_utc':datetime.now(timezone.utc).isoformat(),'source_seal':{'path':str(SEAL),'sha256':SEAL_SHA},
      'copy_receipt':{'path':str(RECEIPT),'sha256':RECEIPT_SHA},'source_archive':str(ARCHIVE),'target_cache':str(TARGET),
      'copied_files_byte_SHA_checked':373,'copied_file_bytes':1018125,'NPY_files_byte_SHA_only':223,'NPY_values_parsed':False,
      'source_and_destination_all_payloads_byte_identical':True,'all_destination_inodes_distinct_from_archive':True,
      'extra_target_files_only_empty_runtime_cache_lock':True,'source_seal_not_copied_into_runtime_cache':True,
      'unit_count':149,'unit_indices':'0..148','unit149_present':False,'all149_genuine_healthy_V5':True,'forensic_release_used':False,
      'source_snapshot_sha256':SNAPSHOT_SHA,'header_sha256':sha(TARGET/'header.json'),'binding_sha256':header['binding_sha256'],
      'producer_count':16,'producer_unit_counts':dict(counts),'producers':list(producers.values()),'copied_files':files,
      'global_lock_scope':'Copy producer records lock acquisition/release; this independent byte checker does not reacquire or independently replay historical lock ownership.',
      'old_failed_cache_read_or_modified':False,'old_source_files_modified':False,'ledger_or_authorization_edited':False,
      'actual_QM_execution_or_real_allocation_by_checker':False,'numeric_library_imports':forbidden,'active_NN_array_reads':0,'Git_operations':0,
      'interpretation':'RHF-only restart seed verified. Full RKS partial/H1/response/Hessian/frequencies remain NOT_RUN by this checker; a new valid V6B parent must qualify subsequent execution.',
      'checker_source':{'path':str(Path(__file__)),'sha256':sha(__file__)}}
    with RESULT.open('x')as f:f.write(json.dumps(report,indent=2)+'\n')
    with(OUT/'index.json').open('x')as f:f.write(json.dumps({'status':'SEALED_INDEPENDENT_HEALTHY149_COPY_CHECK','files':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}for p in [Path(__file__),RESULT]]},indent=2)+'\n')
    print(json.dumps({'status':report['status'],'receipt_path':str(RESULT),'receipt_sha256':sha(RESULT),'index_sha256':sha(OUT/'index.json')}))

if __name__=='__main__':main()
