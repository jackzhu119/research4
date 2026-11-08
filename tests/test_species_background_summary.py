"""Tests of OOF pooling and checkpoint/scope handling, using synthetic fixtures."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import summarize_species_background as summary


class SummaryTests(unittest.TestCase):
    def test_count_pooling_does_not_average_unequal_fold_accuracies(self):
        metrics = summary.confusion_metrics(Counter({('a', 'a'): 9, ('b', 'a'): 1}), ('a', 'b'))
        self.assertEqual(metrics['correct'], 9)
        self.assertEqual(metrics['denominator'], 10)
        self.assertAlmostEqual(metrics['accuracy'], .9)
        self.assertNotAlmostEqual(metrics['accuracy'], (.0 + 1.) / 2)
        self.assertAlmostEqual(metrics['macro_f1'], 9 / 19)

    def test_partial_folds_and_repeated_seeds_never_inflate_specimens(self):
        ids = frozenset(str(i) for i in range(10))
        replicas = {}
        for seed in summary.CLASSIC_SEEDS:
            rep = summary.Replicate(('a', 'b'), ids)
            for fold in range(5):
                rep.add(fold, [{'record_id': str(i), 'truth': 'a' if i < 5 else 'b',
                                'prediction': 'a' if i < 5 else 'b'}
                               for i in (fold, fold + 5)])
            replicas[(seed, -1)] = rep
        result = summary.summarize_replicates(replicas, summary.CLASSIC_SEEDS)
        self.assertTrue(result['eligible_for_development_selection'])
        self.assertEqual(result['unique_specimens'], 10)
        self.assertEqual(result['repeated_prediction_denominator'], 30)
        replicas[(42, -1)].folds.remove(4)
        self.assertFalse(summary.summarize_replicates(replicas, summary.CLASSIC_SEEDS)['eligible_for_development_selection'])

    def test_duplicate_specimen_cannot_be_complete(self):
        rep = summary.Replicate(('a', 'b'), frozenset(('one', 'two')))
        rep.add(0, [{'record_id': 'one', 'truth': 'a', 'prediction': 'a'}])
        rep.add(1, [{'record_id': 'one', 'truth': 'a', 'prediction': 'a'}])
        self.assertTrue(rep.result((42, -1))['invalid_reasons'])

    def _classic_job(self, root, seed, fold):
        directory = root / 'runs' / f'classical_{seed}_{fold}'
        directory.mkdir(parents=True)
        ids = [str(i) for i in range(10)]
        valid = [str(fold), str(fold + 5)]
        manifest = {'worker_protocol': 'training-only-classical-nested-development-v1',
                    'task': 'acid_origins', 'representation': 'alpha', 'outer_fold': fold,
                    'outer_split_seed': seed, 'class_order': ['a', 'b'],
                    'master_training_record_ids': ids, 'source_record_ids': [i for i in ids if i not in valid],
                    'outer_validation_record_ids': valid,
                    'master_calibration_or_prediction_used_for_fit_or_selection': False}
        summary.atomic_json(directory / 'job_manifest.json', manifest)
        identity = summary.digest(directory / 'job_manifest.json')
        summary.atomic_json(directory / 'summary.json', {'job_manifest_sha256': identity,
                           'master_calibration_or_prediction_used': False})
        rows = []
        for method in summary.CLASSIC_METHODS:
            for identifier in valid:
                truth = 'a' if int(identifier) < 5 else 'b'
                prediction = 'b' if method == 'R0' and identifier == '0' else truth
                rows.append({'method': method, 'record_id': identifier, 'y_true': truth,
                             'prediction_vote': prediction, 'prediction_score_argmax': prediction})
        with gzip.open(directory / 'validation_predictions.csv.gz', 'wt', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary.atomic_json(directory / 'complete.json', {'status': 'complete', 'job_manifest_sha256': identity,
                           'artifacts_sha256': {name: summary.digest(directory / name) for name in
                                               ('job_manifest.json', 'summary.json', 'validation_predictions.csv.gz')}})
        return directory

    def test_real_receipt_schema_full_cv_and_missing_planned_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary.atomic_json(root / 'input_lock.json', {'training_allowlist': [str(i) for i in range(10)]})
            jobs = []
            for seed in summary.CLASSIC_SEEDS:
                for fold in range(5):
                    directory = self._classic_job(root, seed, fold)
                    jobs.append({'id': directory.name, 'kind': 'classical',
                                 'outputs': [str(directory / 'complete.json')]})
            jobs.append({'id': 'not_started', 'kind': 'neural',
                         'outputs': [str(root / 'runs/not_started/complete.json')]})
            summary.atomic_json(root / 'jobs.json', {'jobs': jobs})
            result = summary.summarize(root)
            self.assertEqual(result['job_counts']['complete'], 15)
            self.assertEqual(result['job_counts']['not_run'], 1)
            self.assertEqual(len(result['development_candidates']), 1)
            r0 = next(row for row in result['results'] if row['method'] == 'R0' and row['prediction_semantics'] == 'estimator_vote')
            self.assertAlmostEqual(r0['complete_oof_metric_mean']['accuracy'], .9)
            self.assertEqual(r0['unique_specimens'], 10)
            original = (root / 'results/results_summary.json').read_bytes()
            summary.summarize(root)
            self.assertEqual((root / 'results/results_summary.json').read_bytes(), original)

    def test_changed_prediction_bytes_are_rejected_without_partial_contribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary.atomic_json(root / 'input_lock.json', {'training_allowlist': [str(i) for i in range(10)]})
            job = self._classic_job(root, summary.CLASSIC_SEEDS[0], 0)
            with (job / 'validation_predictions.csv.gz').open('ab') as handle:
                handle.write(b'changed')
            result = summary.summarize(root)
            self.assertEqual(result['job_counts']['invalid'], 1)
            self.assertEqual(result['results'], [])
            self.assertEqual(result['development_candidates'], [])


if __name__ == '__main__':
    unittest.main()
