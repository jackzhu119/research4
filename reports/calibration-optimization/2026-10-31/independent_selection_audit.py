"""Independent aggregation/selection audit; does not import project runners."""
from array import array
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path

import numpy as np


REPO = Path('/workspace/research4')
BASE = REPO / 'outputs/zaoren-calibration-optimization-20261031'
RUN = BASE / 'run'
METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1', 'matched_accuracy',
           'matched_balanced_accuracy', 'matched_macro_f1')
failures = []
checks = 0
max_difference = 0.


def check(condition, description):
    global checks
    checks += 1
    if not condition:
        failures.append(description)


def near(left, right, description, tolerance=1e-12):
    global max_difference
    difference = abs(float(left)-float(right))
    max_difference = max(max_difference, difference)
    check(difference <= tolerance, f'{description}: difference={difference}')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


protocol_path = REPO / 'data/research/calibration_optimization_protocol_20261031.json'
protocol = read_json(protocol_path)
plan_path = REPO / protocol['fixed_split']
plan = read_json(plan_path)
manifest = read_json(RUN/'run_manifest.json')
candidates = read_json(RUN/'candidate_grid.json')
candidate_lookup = {c['id']: c for c in candidates}
identity = next(c for c in candidates if c['family'] == 'identity')
locked = read_json(RUN/'selection_lock.json')
aggregate_original = read_json(RUN/'development_aggregate.json')
status = read_json(RUN/'status.json')
scope_lookup = {s['id']: s for s in protocol['scopes']}
original_labels = {r['record_id']: r['origin_group_id'] for r in plan['records']}
outer_roles = {role: {r['record_id'] for r in plan['records'] if r['role'] == role}
               for role in ('training', 'calibration', 'prediction')}

check(status['status'] == 'completed', 'run completed')
check(sha(protocol_path) == manifest['protocol_sha256'], 'protocol hash matches')
check(sha(plan_path) == manifest['plan_sha256'], 'plan hash matches')
check(sha(REPO/protocol['source_input_npz']) == manifest['input_sha256'], 'NPZ input hash matches')
check(sha(RUN/'run_manifest.json') == locked['run_manifest_sha256'], 'selection manifest hash matches')
grid_sha = hashlib.sha256(json.dumps(candidates,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
check(grid_sha == manifest['grid_sha256'] == locked['candidate_grid_sha256']
      == protocol['candidates']['grid_sha256'], 'candidate grid hashes match')
for path, expected in manifest['source_code_sha256'].items():
    check(sha(REPO/path) == expected, 'source hash '+path)
for path, expected in manifest['frozen_source_artifact_sha256'].items():
    check(sha(REPO/path) == expected, 'frozen artifact hash '+path)
check(sha(REPO/protocol['source_model']['existing_seven_class_models'])
      == manifest['frozen_source_manifest_sha256'], 'frozen manifest hash')
check(sha(plan_path) == manifest['frozen_source_training_plan_sha256'], 'original source training cohort hash')

accumulator = {}
file_count = row_count = model_fits = candidate_calls = freeze_assertions = 0
partition_snapshots = {}
freeze_fingerprints = []
rare_order = list(permutations(range(3)))
all_classes = sorted(set(original_labels.values()))

for scope in protocol['scopes']:
    sid = scope['id']
    excluded = set(scope['exclude_groups'])
    classes = [c for c in all_classes if c not in excluded]
    expected_training = {r for r in outer_roles['training'] if original_labels[r] not in excluded}
    expected_counts = protocol['development']['seven_class_counts' if not excluded else 'six_class_counts']
    training_order = sorted(expected_training)
    for selector in protocol['selectors']:
        folder = RUN/'development'/sid/selector
        csv_paths = sorted(folder.glob('episode_*.csv.gz'))
        check(len(csv_paths) == protocol['development']['episodes'], f'{sid}/{selector}: 30 episode files')
        for path in csv_paths:
            episode = int(path.name.split('_')[1].split('.')[0])
            stage = 'selection' if episode < protocol['development']['selection_episodes'] else 'gate'
            meta = read_json(path.with_name(path.name.replace('.csv.gz','.json')))
            check(sha(path) == meta['metrics_file_sha256'], f'{sid}/{selector}/{episode}: CSV hash')
            check(meta['failures'] == [], f'{sid}/{selector}/{episode}: no candidate failures')
            check(meta['source_classifier_fits'] == 1, 'one source classifier fit per episode')
            check(meta['source_params'] == {'C':10.,'preprocessing':'snv','fit_scope':'episode_source_only'},
                  'fixed source parameters and fit scope')
            fingerprint = meta.get('source_fingerprint_unchanged','')
            check(len(fingerprint)==32 and all(c in '0123456789abcdef' for c in fingerprint),
                  f'{sid}/{selector}/{episode}: stored freeze assertion')
            freeze_assertions += bool(fingerprint)
            freeze_fingerprints.append(fingerprint)
            part = {k:set(v) for k,v in meta['partition'].items()}
            check(set(part)=={'source','support_pool','query'}, 'episode role names')
            check({k:len(v) for k,v in part.items()}==expected_counts, 'episode role sizes')
            check(set.union(*part.values())==expected_training, 'episode covers only full outer training scope')
            check(not(part['source']&part['support_pool'] or part['source']&part['query'] or part['support_pool']&part['query']),
                  'episode roles disjoint')
            check(not(set.union(*part.values()) & (outer_roles['calibration']|outer_roles['prediction'])),
                  'development never uses outer calibration/prediction specimens')
            for name, identifiers in part.items():
                check({original_labels[r] for r in identifiers}==set(classes), 'every episode role covers all scope classes')
                check(len(meta['partition'][name])==len(identifiers), 'no repeated IDs in episode role')
            key = sid,episode
            if key in partition_snapshots:
                check(meta['partition']==partition_snapshots[key], 'source families share exact development partitions')
            else:
                partition_snapshots[key] = meta['partition']
            # Independently regenerate the recorded PCG64 class-wise partition.
            rng=np.random.default_rng(protocol['development']['split_seed']+episode)
            regenerated={name:[] for name in ('source','support_pool','query')}
            for label in classes:
                group=np.asarray(sorted(r for r in expected_training if original_labels[r]==label))
                order=group[list(rare_order[episode%6])] if len(group)==3 else rng.permutation(group)
                ns=max(1,int(.6*len(group))); npool=max(1,int(.2*len(group)))
                regenerated['source'].extend(order[:ns].tolist())
                regenerated['support_pool'].extend(order[ns:ns+npool].tolist())
                regenerated['query'].extend(order[ns+npool:].tolist())
            check(all(sorted(regenerated[name])==sorted(meta['partition'][name]) for name in regenerated),
                  'episode seeded assignments and rare-class permutation exactly reproduced')
            draws=meta['draws']; expected_draw_count=protocol['development']['support_draws']*len(scope['shots'])
            check(len(draws)==expected_draw_count, 'development support draw count')
            draw_lookup={}
            for d in draws:
                dkey=(d['repeat'],d['shots_per_class']); ids=d['support_ids']; selected=set(ids)
                check(dkey not in draw_lookup, 'development unique repeat/shot')
                draw_lookup[dkey]=selected
                check(len(ids)==len(selected)==d['shots_per_class']*len(classes), 'balanced distinct support specimens')
                check(selected<=part['support_pool'] and not(selected&part['query']), 'supports only from support pool')
                check(Counter(original_labels[r] for r in ids)==Counter({c:d['shots_per_class'] for c in classes}),
                      'per-class K support labels')
            for repeat in range(protocol['development']['support_draws']):
                prev=set()
                for k in sorted(scope['shots']):
                    check(prev<=draw_lookup[(repeat,k)], 'nested development supports')
                    prev=draw_lookup[(repeat,k)]
            expected_rows=len(candidates)*expected_draw_count
            counter=Counter(); seen=set(); local_count=0
            with gzip.open(path,'rt',encoding='utf-8',newline='') as handle:
                for row in csv.DictReader(handle):
                    k=int(row['shots_per_class']); repeat=int(row['repeat']); cid=row['candidate_id']
                    check(row['scope']==sid and row['selector']==selector and int(row['episode'])==episode
                          and row['stage']==stage, 'CSV episode/scope/stage labels')
                    check(cid in candidate_lookup and row['family']==candidate_lookup[cid]['family'], 'CSV candidate identity')
                    casekey=k,repeat,cid
                    check(casekey not in seen, 'no repeated development case')
                    seen.add(casekey)
                    check((repeat,k) in draw_lookup, 'CSV maps to an actual support draw')
                    check(row['status']=='pass' and row['error']=='', 'development candidate result passed')
                    key=(sid,selector,k,stage,cid)
                    if key not in accumulator:
                        accumulator[key]={'values':[array('d') for _ in METRICS],'count':0,'failures':0}
                    entry=accumulator[key]
                    if row['status']=='pass':
                        for i,name in enumerate(METRICS):
                            value=float(row[name]); entry['values'][i].append(value)
                            check(math.isfinite(value) and 0<=value<=1, 'development metric within bounds')
                        entry['count']+=1
                    else:
                        entry['failures']+=1
                    local_count+=1; counter[k]+=1
            check(local_count==expected_rows==meta['candidate_fit_calls'], 'exact episode candidate evaluation count')
            check(all(counter[k]==len(candidates)*protocol['development']['support_draws'] for k in scope['shots']),
                  'all shots have all candidates/draws')
            row_count+=local_count; file_count+=1
            model_fits+=meta['source_classifier_fits']; candidate_calls+=meta['candidate_fit_calls']

aggregates={key:{**{name:math.fsum(entry['values'][i])/entry['count'] for i,name in enumerate(METRICS)},
                 'evaluations':entry['count'],'failures':entry['failures']} for key,entry in accumulator.items()}
check(len(aggregate_original)==len(aggregates), 'aggregate entry count')
for row in aggregate_original:
    key=(row['scope'],row['selector'],row['shots_per_class'],row['stage'],row['candidate_id'])
    actual=aggregates[key]
    for name in METRICS:
        near(actual[name],row[name], 'independently reaggregated '+name)
    check(actual['evaluations']==row['evaluations'] and actual['failures']==row['failures'], 'aggregate counts')

selection=protocol['selection']; order=selection['family_rank']; independent_choices=[]


def preference(candidate):
    return (order[candidate['family']],candidate.get('blend',candidate.get('strength',0.)),
            -candidate.get('regularization',0.),-candidate.get('pseudocount',0.),
            candidate.get('gate_margin') is None,candidate['id'])


for scope in protocol['scopes']:
    sid=scope['id']
    for selector in protocol['selectors']:
        for k in scope['shots']:
            stages={stage:{c['id']:aggregates[(sid,selector,k,stage,c['id'])] for c in candidates}
                    for stage in ('selection','gate')}
            def eligible(cid,stage):
                value=stages[stage][cid]; reference=stages[stage][identity['id']]
                return (not value['failures']
                    and value['accuracy']-reference['accuracy']>selection['minimum_accuracy_gain']
                    and value['macro_f1']-reference['macro_f1']>=-selection['max_macro_f1_loss']
                    and value['balanced_accuracy']-reference['balanced_accuracy']>=-selection['max_balanced_accuracy_loss']
                    and value['accuracy']-value['matched_accuracy']>selection['minimum_accuracy_gain']
                    and value['macro_f1']-value['matched_macro_f1']>=-selection['max_macro_f1_loss']
                    and value['balanced_accuracy']-value['matched_balanced_accuracy']>=-selection['max_balanced_accuracy_loss'])
            def best(ids):
                high=max(stages['selection'][cid]['accuracy'] for cid in ids)
                near_ids=[cid for cid in ids if stages['selection'][cid]['accuracy']>=high-selection['tie_tolerance']]
                return min((candidate_lookup[cid] for cid in near_ids),key=preference)
            permissible=[c['id'] for c in candidates if eligible(c['id'],'selection')]
            proposed=best(permissible) if permissible else identity
            deployed=proposed if proposed['family']!='identity' and eligible(proposed['id'],'gate') else identity
            actual=locked['choices'][sid][selector][str(k)]
            check(proposed==actual['proposed'], 'independent proposed candidate '+str((sid,selector,k)))
            check(deployed==actual['selected'], 'independent deployed candidate '+str((sid,selector,k)))
            check((deployed['family']=='identity')==actual['fallback_to_identity'], 'identity fallback flag')
            for family in sorted({c['family'] for c in candidates}):
                family_ids=[c['id'] for c in candidates if c['family']==family and not stages['selection'][c['id']]['failures']]
                expected=best(family_ids) if family_ids else identity
                saved=actual['family_winners'][family]
                check(saved['candidate']==expected, 'independent family selected candidate '+str((sid,selector,k,family)))
                check(saved['passes_gate']==(eligible(expected['id'],'gate') if family_ids else False),
                      'independent family gate '+str((sid,selector,k,family)))
            independent_choices.append({'scope':sid,'selector':selector,'shots_per_class':k,
                'eligible_selection_candidates':len(permissible),'proposed':proposed['id'],'selected':deployed['id'],
                'fallback_to_identity':deployed['family']=='identity',
                'selection_accuracy_gain':stages['selection'][proposed['id']]['accuracy']-stages['selection'][identity['id']]['accuracy'],
                'gate_accuracy_gain':stages['gate'][proposed['id']]['accuracy']-stages['gate'][identity['id']]['accuracy']})

check(file_count==120 and row_count==696000 and candidate_calls==696000 and model_fits==120,
      '120 development fits and 696000 candidates verified')
check(model_fits==locked['development_source_classifier_fits']==status['new_development_source_classifier_fits'],
      'recorded development classifier fit totals')
check(candidate_calls==locked['development_candidate_fit_calls']==status['development_candidate_fit_calls'],
      'recorded candidate totals')
lock_sha=sha(RUN/'selection_lock.json')
check(read_json(RUN/'selection_lock_sha256.json')['sha256']==lock_sha==status['selection_lock_sha256'],
      'selection lock hash stable across all final metadata')
events=[]
for line in (BASE/'run.log').read_text().splitlines():
    try:
        event=json.loads(line)
        if isinstance(event,dict): events.append(event)
    except json.JSONDecodeError:
        pass
lock_events=[(i,e) for i,e in enumerate(events) if e.get('stage')=='all_choices_locked']
final_events=[(i,e) for i,e in enumerate(events) if e.get('stage')=='final_model_completed']
check(len(lock_events)==1 and len(final_events)==4, 'one lock and four final model completion events')
if lock_events:
    index,event=lock_events[0]
    check(event['sha256']==lock_sha, 'logged lock SHA equals current locked file')
    check(all(i>index for i,_ in final_events), 'all final model events occur after selection lock')
    prior_events=[e for e in events[:index] if e.get('stage')=='development_model_completed']
    check(len(prior_events)==4, 'four development families completed before lock')
for meta in read_json(RUN/'final_metadata.json'):
    scope=scope_lookup[meta['scope']]; excluded=set(scope['exclude_groups'])
    expected_source={r for r in outer_roles['training'] if original_labels[r] not in excluded}
    expected_query={r for r in outer_roles['prediction'] if original_labels[r] not in excluded}
    check(set(meta['source_sample_ids'])==expected_source, 'final source IDs equal correct fixed training scope')
    check(set(meta['query_ids'])==expected_query, 'final query IDs equal correct fixed prediction scope')
    check(meta['source_fingerprint_before']==meta['source_fingerprint_after'], 'final source frozen fingerprint')
    check(meta['query_labels_used_to_fit'] is False and meta['query_statistics_fitted'] is False,
          'final fit-scope assertions')

result={'status':'pass' if not failures else 'fail',
    'auditor':'independent selection and provenance audit; no project aggregation/selection function imported',
    'development_files':file_count,'development_rows':row_count,'independent_aggregate_entries':len(aggregates),
    'classifier_fits_recorded':model_fits,'development_freeze_assertions_verified':freeze_assertions,
    'distinct_recorded_source_fingerprints':len(set(freeze_fingerprints)),
    'checks_evaluated':checks,'failed_checks':failures,'maximum_aggregate_difference':max_difference,
    'independently_recomputed_choices':independent_choices,'selection_lock_sha256':lock_sha,
    'lock_log_precedes_all_final_completions':bool(lock_events and all(i>lock_events[0][0] for i,_ in final_events)),
    'limits':['Development metrics independently reaggregated from saved rows; no per-sample development predictions were stored for re-scoring here.',
              '120 development freeze checks verify recorded assertions and matching source-code hash; models were not refitted or externally attested.',
              'Chronology is supported by log order, immutable lock hashes and static code review, not an independent execution attestation.',
              'Physical origin truth cannot be verified from repeated digital label consistency.']}
(BASE/'independent_selection_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('independently_recomputed_choices','limits','failed_checks')},ensure_ascii=False))
print(json.dumps({'failed_check_counts':dict(Counter(failures))},ensure_ascii=False))
print(json.dumps(independent_choices,ensure_ascii=False))
raise SystemExit(0 if not failures else 1)
