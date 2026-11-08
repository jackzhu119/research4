"""Guard the prepared input lineage and source-code SHA before final binding.

This adds no fits and never parses prediction truth. It preserves the already
written preparation source, role packs and historical membership snapshot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from prepare_supplemental_splits_20261104 import (ROOT, checked, reference, sha, jwrite,
    require, load_npz, assert_partition_identity, ROLES)


def source_records(pack):
    labels = pack['y'] if 'y' in pack else pack['labels']
    return [{'record_id':str(i),'entity_id':str(e),'legacy_match_ids':str(a),'label':str(y)}
            for i,e,a,y in zip(pack['record_ids'],pack['entity_ids'],pack['legacy_match_ids'],labels)]


def check_source_partitions(records, task):
    record_ids={r['record_id'] for r in records}
    counts={'inner_folds':0,'source_calibration_episodes':0}
    validated=[]
    for fold in task.get('inner_folds',[]):
        train,valid=fold['train_ids'],fold['valid_ids']
        require(len(train)==len(set(train)) and len(valid)==len(set(valid)), 'Repeated inner-fold record ID')
        require(not set(train)&set(valid) and set(train+valid)==record_ids,'Inner fold must partition current source exactly')
        assignment={i:role for role,values in (('source',train),('query',valid)) for i in values}
        assert_partition_identity(records,assignment)
        counts['inner_folds']+=1;validated+=valid
    if task.get('inner_folds'):
        require(len(validated)==len(record_ids) and set(validated)==record_ids,'Each source record must have one OOF validation role')
    for episode in task.get('source_calibration_episodes',[]):
        groups=[episode[k] for k in ('base_ids','support_ids','query_ids')]
        require(all(len(v)==len(set(v)) for v in groups),'Repeated pseudorole ID')
        assignment={i:role for role,values in zip(ROLES,groups) for i in values}
        require(sum(map(len,groups))==len(record_ids),'Pseudo roles do not cover source once')
        assert_partition_identity(records,assignment)
        for shot,ids in episode['shots'].items():
            require(len(ids)==len(set(ids)) and set(ids)<=set(episode['support_ids']),'Support shot is not a distinct subset of its reserved pool')
        counts['source_calibration_episodes']+=1
    return counts


def guarded_bind(repo, output):
    repo,output=Path(repo).resolve(),Path(output).resolve()
    prepared=json.loads((output/'task_drafts.json').read_text())
    original_split=checked(repo,prepared['split_script'])
    recipe_path=checked(repo,prepared['recipe']);recipe=json.loads(recipe_path.read_text())
    for key in ('intake_manifest','intake_metadata','intake_sources','legacy_split'):
        checked(repo,prepared[key])
    checks={}
    for task in prepared['tasks']:
        for role in ROLES:
            checked(repo,task[role+'_pack'])
        # Hash integrity only: prediction labels are never parsed by this binder.
        checked(repo,task['truth'])
        source=load_npz(repo/task['source_pack']['path'])
        checks[task['id']]=check_source_partitions(source_records(source),task)
        if task.get('inherit_selection_from'):
            require(not task['inner_folds'] and not task['source_calibration_episodes'],
                    'Expanded singleton classes cannot enter ordinary CV/meta selection')
    guard={'status':'PASS_PREPARATION_SHA_AND_SOURCE_FOLD_LINEAGE','checked_at_utc':datetime.now(timezone.utc).isoformat(),
           'preparation_script':reference(original_split,repo),'source_checks':checks,
           'prediction_truth_parsed':False,'new_model_fits':0,'independent_confirmation_established':False}
    recipe.update(tasks=prepared['tasks'],intake_manifest=prepared['intake_manifest'],intake_sources=prepared['intake_sources'],
                  intake_metadata=prepared['intake_metadata'],recipe=prepared['recipe'],preparation=reference(output/'task_drafts.json',repo),
                  binding_guard=guard,status='BOUND_INPUTS_READY_NO_MODEL_FITS')
    code=sorted((repo/'src/herb_spectra').glob('*.py'))+[repo/'scripts/run_supplemental_benchmark_20261104.py',original_split,Path(__file__).resolve()]
    require(all(p.is_file() for p in code),'Complete benchmark source is required before binding')
    recipe['code_files']={str(p.relative_to(repo)):sha(p) for p in code}
    path=output/'bound_protocol.json';jwrite(path,recipe)
    receipt={'status':'DONE_INPUT_BINDING_ONLY','bound_protocol':reference(path,repo),'guard':guard}
    jwrite(output/'guarded_binding_receipt.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('data/splits/supplemental_spectra_20261104_v1'))
    args=parser.parse_args()
    print(json.dumps(guarded_bind(ROOT,ROOT/args.output)))


if __name__=='__main__':main()
