"""Summarize all actual saved fit costs after the sealed run, without refitting."""
from pathlib import Path
import collections
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from herb_spectra.public_thz_benchmark import read, sha, verify_predictions_before_truth, write

out = Path(__file__).resolve().parent
verify_predictions_before_truth(out)
result = {'scoring_manifest_sha256': sha(out / 'scoring_manifest.json'),
          'prediction_lock_sha256': sha(out / 'prediction_lock.json'),
          'no_new_model_fits_in_this_summary': True}
for stage, filename in [('source', 'inner_records.json'), ('outer', 'outer_model_records.json')]:
    records = read(out / filename)
    kinds = collections.Counter()
    warns = collections.Counter()
    for record in records:
        model = read((out / record['estimator_path']).with_suffix('.json'))
        kinds[model['estimator']['model_kind']] += 1
        for warning in model['estimator']['warnings']:
            warns[(warning['category'], warning['message'])] += 1
    result[stage] = {'actual_classifiers': len(records), 'actual_model_kind_counts': dict(kinds),
                     'summed_trial_elapsed_seconds_including_cache_and_serialization': sum(r['elapsed_seconds'] for r in records),
                     'warnings': [{'category': k[0], 'message': k[1], 'count': v} for k, v in warns.items()]}
events = collections.Counter()
with (out / 'fit_scope_events.jsonl').open() as stream:
    for line in stream:
        event = json.loads(line)
        if event['event'] == 'fit_start':
            events[event['component']] += 1
result['complete_run_fit_start_components'] = dict(events)
result['outer_alias_role_fits_without_extra_estimators'] = len(read(out / 'outer_role_aliases.json'))
result['recorded_commands'] = [json.loads(line) for line in (out / 'driver_events.jsonl').read_text().splitlines()]
write(out / 'execution_costs.json', result)
print(json.dumps({k: v for k, v in result.items() if k != 'recorded_commands'}, indent=2))
