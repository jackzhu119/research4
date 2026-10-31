"""Deployment boundary/regression checks; synthetic fixtures are not plant data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from herb_spectra.deployment import (fit_model, fresh, historical_matches,
                                    source_candidates, spectrum_hash,
                                    verify_source_gate, write, predict, load_bundle)
from herb_spectra.leakage_guard import LeakageError
from herb_spectra.staged_nested_search import fit_source_candidate


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(193)
        self.x = rng.normal(size=(28, 14))
        self.y = np.repeat(['ZY01', 'ZY02', 'ZY03', 'ZY04', 'ZY05', 'ZY06', 'ZY07'], 4)
        self.ids = np.asarray([f'fixture-{i:03}' for i in range(28)])
        self.axis = np.linspace(.1, 1.5, 14)
        self.config = {'id': 'synthetic-unit-only', 'preprocessing': 'snv',
            'mc': {'runs': 2, 'components': 1, 'ratio': .5},
            'svm': {'C': 1., 'gamma': 'scale_x0.1', 'class_weight': 'power_0.5'}}

    def test_instrumented_model_matches_original_mathematics(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / 'scope.jsonl'
            actual = fit_model(self.config, self.x, self.y, self.ids, self.axis, 42,
                split_id='synthetic.source', log_path=log, expected_source_ids=self.ids)
            legacy = fit_source_candidate(self.config, self.x, self.y, self.axis, seed=42)
            np.testing.assert_array_equal(actual.bundle.selector.selected_indices_, legacy.bundle.selector.selected_indices_)
            np.testing.assert_array_equal(actual.predict(self.x), legacy.predict(self.x))
            np.testing.assert_allclose(actual.predict_scores(self.x), legacy.predict_scores(self.x), atol=0, rtol=0)
            events = [json.loads(line) for line in log.read_text().splitlines()]
            completed = [e for e in events if e['event'] == 'fit_completed']
            self.assertEqual(sum(e['component'] == 'MC-UVE.PLSRegression.fit' for e in completed), 14)
            self.assertTrue(all(set(e['fit_sample_ids']) <= set(self.ids) for e in completed))
            self.assertFalse(actual.estimator.break_ties)

    def test_forbidden_source_row_rejected_before_any_fit(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(LeakageError):
                fit_model(self.config, self.x, self.y, self.ids, self.axis, 42,
                    split_id='synthetic.source', log_path=Path(td) / 'scope.jsonl',
                    expected_source_ids=self.ids[:-1])
            self.assertFalse((Path(td) / 'scope.jsonl').exists())

    def test_reexported_spectrum_is_not_new_after_rename_or_roundtrip(self):
        history = {str(r): spectrum_hash(x) for r, x in zip(self.ids, self.x)}
        rows = np.asarray([self.x[0], self.x[1] + 1e-14, self.x[2] + .01])
        a, b, c = historical_matches(rows, ['renamed-a', 'renamed-b', 'new-c'], history, self.x)
        self.assertEqual(a, [False, False, False])
        self.assertEqual(b, [True, False, False])
        self.assertEqual(c, [True, True, False])
        # Permuting or batching independent queries cannot change QC flags.
        self.assertEqual(historical_matches(rows[1:2], ['renamed-b'], history, self.x)[2], [True])

    def test_source_gate_binds_all_source_and_data_versions(self):
        manifest = {'protocol_sha256': 'protocol', 'preparation_sha256': 'data', 'source_sha256': {'code': 'abc'}}
        good = dict(manifest, allow_source_training=True)
        with patch('herb_spectra.deployment.verify_run', return_value=({}, {}, manifest)):
            verify_source_gate('unused', good)
            for key in manifest:
                bad = dict(good, **{key: 'changed'})
                with self.assertRaises(LeakageError): verify_source_gate('unused', bad)
            with self.assertRaises(LeakageError): verify_source_gate('unused', manifest)

    def test_source_candidates_do_not_need_outer_predictions_or_masks(self):
        candidates = source_candidates()
        self.assertEqual([c['id'] for c in candidates], [f'r2_source_fold{i:02}' for i in range(5)])
        self.assertTrue(all(set(c) == {'id', 'preprocessing', 'mc', 'svm'} for c in candidates))
        self.assertEqual([c['mc']['runs'] for c in candidates], [10, 20, 50, 50, 50])

    def test_prediction_is_label_free_pointwise_and_id_invariant(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); bundle = root / 'bundle'; bundle.mkdir()
            fitted = fit_model(self.config, self.x, self.y, self.ids, self.axis, 42,
                split_id='synthetic.source', log_path=root / 'scope.jsonl', expected_source_ids=self.ids)
            history = {str(r): spectrum_hash(x) for r, x in zip(self.ids, self.x)}
            manifest = {'axis': self.axis.tolist(), 'historical_spectrum_hashes': history}
            write(bundle / 'bundle_manifest.json', manifest)
            np.savez_compressed(bundle / 'history_reference.npz', X_alpha=self.x, record_ids=self.ids, axis=self.axis)
            # Genuinely changed numeric fixture; not an independent plant sample.
            query = self.x[:3] + np.linspace(.01, .1, self.x.shape[1])
            for tag, ids in [('a', ['new-a', 'new-b', 'new-c']), ('b', ['rename1', 'rename2', 'rename3'])]:
                np.savez_compressed(root / f'{tag}.npz', X_alpha=query, axis=self.axis, record_ids=np.array(ids))
            with patch('herb_spectra.deployment.load_bundle', return_value=(manifest, {str(s): fitted for s in (42, 43, 44)})), \
                 patch('sklearn.svm.SVC.fit', side_effect=AssertionError('Inference must not fit')):
                a = predict(bundle, root / 'a.npz', root / 'out-a', confirm_compatible_measurement=True)
                b = predict(bundle, root / 'b.npz', root / 'out-b', confirm_compatible_measurement=True)
            import csv
            with (root / 'out-a/predictions.csv').open(encoding='utf-8-sig') as f: rows_a = list(csv.DictReader(f))
            with (root / 'out-b/predictions.csv').open(encoding='utf-8-sig') as f: rows_b = list(csv.DictReader(f))
            for row_a, row_b in zip(rows_a, rows_b):
                self.assertEqual({k: v for k, v in row_a.items() if k != 'record_id'},
                                 {k: v for k, v in row_b.items() if k != 'record_id'})
            self.assertEqual(a['new_model_fits'], 0)
            self.assertFalse(a['scores_are_probabilities'])
            self.assertFalse(a['measurement_compatibility_verified'])
            self.assertTrue(a['user_attested_measurement_compatibility'])

    def test_future_rejects_renamed_roundtrip_before_any_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); bundle = root / 'bundle'; bundle.mkdir()
            history = {str(r): spectrum_hash(x) for r, x in zip(self.ids, self.x)}
            manifest = {'axis': self.axis.tolist(), 'historical_spectrum_hashes': history}
            write(bundle / 'bundle_manifest.json', manifest)
            np.savez_compressed(bundle / 'history_reference.npz', X_alpha=self.x, record_ids=self.ids, axis=self.axis)
            np.savez_compressed(root / 'query.npz', X_alpha=self.x[:1] + 1e-14, axis=self.axis,
                                record_ids=np.array(['renamed-old-row']))
            from unittest.mock import Mock
            model = Mock()
            with patch('herb_spectra.deployment.load_bundle', return_value=(manifest, {'42': model})):
                with self.assertRaises(LeakageError):
                    predict(bundle, root / 'query.npz', root / 'out', confirm_compatible_measurement=True)
            model.predict.assert_not_called()
            self.assertFalse((root / 'out').exists())

    def test_bundle_source_drift_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td)
            write(bundle / 'bundle_manifest.json', {'source_sha256': {'wrong': 'old'}, 'environment': {}})
            with patch('herb_spectra.deployment.code_versions', return_value={'actual': 'current'}), \
                 patch('herb_spectra.deployment.joblib.load') as loader:
                with self.assertRaises(LeakageError): load_bundle(bundle)
                loader.assert_not_called()

    def test_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'new'; fresh(p); write(p / 'value.json', {'value': 1})
            with self.assertRaises(FileExistsError): fresh(p)
            with self.assertRaises(FileExistsError): write(p / 'value.json', {'value': 2})
            self.assertEqual(json.loads((p / 'value.json').read_text()), {'value': 1})


if __name__ == '__main__': unittest.main()
