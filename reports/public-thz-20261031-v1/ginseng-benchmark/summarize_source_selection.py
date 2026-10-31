"""Read-only source-fit diagnostic. Never reads outer truth or changes selection."""
from pathlib import Path
import collections
import itertools
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from herb_spectra.public_thz_benchmark import verify_selected, read, write, sha

out = Path(__file__).resolve().parent
_, _, lock = verify_selected(out)
if (out / 'outer_release_snapshot.json').exists():
    raise RuntimeError('This diagnostic must precede outer prediction release')
records = read(out / 'inner_records.json')
warnings = collections.Counter()
kinds = collections.Counter()
summed_seconds = 0.0
metadata = {}
for record in records:
    model = read((out / record['estimator_path']).with_suffix('.json'))
    metadata[(record['split_id'], record['candidate_id'], record['seed'])] = model
    kinds[model['estimator']['model_kind']] += 1
    for warning in model['estimator']['warnings']:
        warnings[(warning['category'], warning['message'])] += 1
    summed_seconds += model['elapsed_seconds']

feature_stability = []
for fold in lock['outer_folds']:
    for role in ['R1', 'R2', 'R3']:
        choice = fold['choices'][role]
        cases = [metadata[(r['split_id'], r['candidate_id'], r['seed'])]
                 for r in records
                 if r['outer_group'] == fold['outer_group']
                 and r['candidate_id'] == choice['config']['id']]
        selected = [set(c['retained_indices']) for c in cases]
        jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(selected, 2)]
        frequency = collections.Counter(i for s in selected for i in s)
        feature_stability.append({
            'outer_export_group': fold['outer_group'], 'role': role,
            'config': choice['config'], 'source_validation_cases': len(cases),
            'selected_mean_inner_metrics': choice['mean_inner_metrics'],
            'retained_counts': sorted(set(len(s) for s in selected)),
            'mean_pairwise_jaccard': float(np.mean(jac)),
            'frequency_across_source_subsets_and_seeds': dict(sorted(frequency.items())),
            'scope': 'Selected source-development configurations; not outer performance',
        })
events = collections.Counter()
with (out / 'fit_scope_events.jsonl').open() as stream:
    for line in stream:
        event = json.loads(line)
        if event['event'] == 'fit_start':
            events[event['component']] += 1
summary = {
    'selection_lock_sha256': sha(out / 'selection_lock.json'),
    'source_classifiers': len(records), 'source_model_kinds': dict(kinds),
    'failed_candidate_cases': lock['source_failures'],
    'source_fit_start_components': dict(events),
    'summed_trial_elapsed_seconds_including_cache_and_serialization': summed_seconds,
    'warnings': [{'category': k[0], 'message': k[1], 'count': v}
                 for k, v in warnings.items()],
    'source_only_feature_stability': feature_stability,
    'outer_evaluated': False, 'outer_truth_read': False,
    'scientific_status': 'EXPLORATORY_OBSERVATION_LEVEL_UNKNOWN_UPSTREAM',
    'not_new_biological_independence_evidence': True,
}
write(out / 'source_selection_diagnostics.json', summary)
print(json.dumps({k: v for k, v in summary.items() if k != 'source_only_feature_stability'}, indent=2))
