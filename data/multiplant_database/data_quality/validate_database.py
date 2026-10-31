"""Validate the generated database without fitting any model."""
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from herb_spectra.multiplant_database import digest,validate_zaoren_row_bindings
from herb_spectra.leakage_guard import validate_lineage_partitions


ROOT=Path(__file__).resolve().parents[3]
DB=ROOT/'data/multiplant_database'


def main():
    checks=0
    inventory=list(csv.DictReader((DB/'dataset_inventory.csv').open(encoding='utf-8-sig')))
    source_groups=list(csv.DictReader((DB/'metadata/source_groups.csv').open(encoding='utf-8-sig')))
    for row in source_groups:
        assert digest(ROOT/row['source_file'])==row['source_hash'];checks+=1
    for row in inventory:
        p=ROOT/row['file'];assert digest(p)==row['sha256'];checks+=1
        with np.load(p,allow_pickle=False) as z:
            assert len(z['X'])==int(row['records'])==len(z['record_ids']);checks+=1
            assert z['X'].shape[1]==int(row['axis_points'])==len(z['axis']);checks+=1
            assert len(set(z['record_ids'].tolist()))==len(z['record_ids']);checks+=1
            assert int((~np.isfinite(z['X'])).sum())==int(row['nonfinite_cells']);checks+=1
            if row['dataset_id']!='zaoren':
                assert not any(str(v) for v in z['labels']);checks+=1
    lineage=list(csv.DictReader((DB/'metadata/sample_lineage.csv').open(encoding='utf-8-sig')))
    assert len(lineage)==6399;checks+=1
    assert len({(r['dataset_id'],r['record_id'],r['modality']) for r in lineage})==len(lineage);checks+=1
    for row in lineage:
        if row['dataset_id']!='zaoren':
            assert not row['physical_sample_id'] and not row['label'];checks+=1
        assert not row['batch_id'];checks+=1
    assert (DB/'metadata/sample_lineage.csv').read_bytes()==(ROOT/'data/metadata/sample_lineage.csv').read_bytes();checks+=1
    with np.load(DB/'zaoren/aligned_modalities.npz',allow_pickle=False) as z:
        bindings=validate_zaoren_row_bindings({k:z[k] for k in z.files},DB/'zaoren/modality_alignment.csv');checks+=3420
    split=json.loads((ROOT/'data/splits/upgrade-20261031-v1/split_manifest.json').read_text())
    original=ROOT/split['source_split_file'];assert digest(original)==split['source_split_sha256'];checks+=1
    old=json.loads(original.read_text());zaoren=[r for r in lineage if r['dataset_id']=='zaoren']
    for folded,previous in zip(split['outer_folds'],old,strict=True):
        assert folded['train_ids']==previous['train_sample_ids'] and folded['test_ids']==previous['test_sample_ids'];checks+=1
        r=validate_lineage_partitions(zaoren,{'train':folded['train_ids'],'test':folded['test_ids']});assert r['status']=='EXPLORATORY';checks+=1
        for inner in folded['inner_folds']:
            r=validate_lineage_partitions(zaoren,{'train':inner['train_ids'],'validation':inner['validation_ids'],'forbidden_outer_test':folded['test_ids']})
            assert r['status']=='EXPLORATORY';checks+=1
    unknown=[]
    for dataset in ['huangqi','tianma','wheat']:
        rows=[r for r in lineage if r['dataset_id']==dataset];ids=list(dict.fromkeys(r['record_id'] for r in rows))
        r=validate_lineage_partitions(rows,{'catalog_only':ids});assert r['status']=='BLOCKED';checks+=1
        unknown.append({'dataset_id':dataset,'status':r['status'],'unknown_ids':len(r['unknown_independence_record_ids'])})
    build=json.loads((DB/'metadata/build_manifest.json').read_text())
    assert digest(ROOT/'scripts/build_multiplant_database.py')==build['builder_sha256'];checks+=1
    assert digest(ROOT/'src/herb_spectra/multiplant_database.py')==build['reader_sha256'];checks+=1
    log=(DB/'data_quality/database_unit_tests.log').read_text();assert 'Ran 7 tests' in log and log.rstrip().endswith('OK');checks+=1
    result={'status':'EXPLORATORY_ZAOREN_OTHER_TASKS_BLOCKED','technical_artifact_checks_passed':True,
            'checked_at_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'inventory_rows':len(inventory),
            'selected_primary_sources':len(source_groups),'lineage_rows':len(lineage),'row_binding_rows':bindings['modality_rows'],
            'zaoren_saved_outer_contexts':5,'zaoren_saved_inner_contexts':10,'unknown_task_readiness':unknown,
            'database_tests_passed':7,'new_model_fits':0,'new_independent_confirmation':False,
            'actual_command':'.runtime/bin/python data/multiplant_database/data_quality/validate_database.py',
            'database_test_command':'PYTHONPATH=tests .runtime/bin/python -m unittest test_multiplant_database -v',
            'original_workbooks_rewritten':0,'records_or_labels_deleted':0,
            'not_run_checks':[
                {'check':'original_sampling_logs_and_common_biological_parent_verification','status':'NOT_RUN','reason':'original logs not supplied'},
                {'check':'true_named_origin_species_and_batch_authentication','status':'NOT_RUN','reason':'verification records not supplied'},
                {'check':'independent_instrument_reference_and_absorption_length_unit_verification','status':'NOT_RUN','reason':'instrument/reference/unit documentation not supplied'},
                {'check':'real_cross_batch_or_pristine_confirmation_evaluation','status':'NOT_RUN','reason':'no qualifying confirmation dataset'},
            ],
            'uncovered_risks':['Sampling logs and biological ancestors unverified','Each zaoren group currently one batch','Other plants lack physical IDs and units','Known corrupt/empty modalities remain unavailable','Strict near-correlation threshold cannot exclude all repeats'],
            'script_sha256':digest(Path(__file__))}
    (DB/'data_quality/database_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
