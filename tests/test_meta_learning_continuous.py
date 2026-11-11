"""Scientific isolation, scheduled evidence and durable-resume tests."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import sys
import fcntl
from unittest import mock

sys.path[:0] = [str(Path(__file__).resolve().parent),
               str(Path(__file__).resolve().parents[1] / 'scripts'),
               str(Path(__file__).resolve().parents[1] / 'src')]

import numpy as np
import pandas as pd

import run_meta_learning_continuous as worker


def fixture():
    frame = pd.DataFrame([{'record_id': f'{group}-{index:03d}', 'physical_sample_id': f'{group}-{index:03d}',
                           'origin_group_id': group, 'species': '酸枣仁', 'role': 'train'}
                          for group in worker.GROUPS for index in range(60)])
    rng = np.random.default_rng(91)
    matrix = rng.normal(size=(300, 40))
    for label in range(5):
        matrix[label * 60:(label + 1) * 60, label * 6:(label + 1) * 6] += 1.5
    config = {'support_draws': 2, 'embedding_dim': 8, 'learning_rate': 0.001, 'weight_decay': 0.0001}
    job = worker.build_design(frame)[0]
    return frame, matrix, config, job


class WorkerTests(unittest.TestCase):
    def test_fixed_e03_partition_and_no_nontrain_role(self):
        frame, matrix, config, job = fixture()
        jobs = worker.build_design(frame)
        self.assertEqual(len(jobs), 5)
        self.assertEqual(set(sum([item['query_ids'] for item in jobs], [])), set(frame.record_id))
        for item in jobs:
            self.assertEqual(tuple(len(item[name + '_ids']) for name in ('source', 'support', 'query')), (190, 50, 60))
            self.assertTrue(set(item['source_ids']).isdisjoint(item['query_ids']))
        mutated = frame.copy()
        mutated.loc[mutated.record_id.eq(job['query_ids'][0]), 'role'] = 'prediction'
        with self.assertRaisesRegex(ValueError, 'master TRAIN'):
            worker.validate_job(job, mutated)

    def test_support_nested_draws_and_complete_pool_deduplication(self):
        frame, matrix, config, job = fixture()
        cell = worker.Cell(job, 'alpha', matrix, frame, config)
        draws = worker.support_draws(job, cell.labels, 20)
        self.assertEqual(len(draws), 61)
        self.assertEqual(sum(item['k'] == 10 for item in draws), 1)
        first = {item['k']: set(item['ids']) for item in draws if item['draw'] == 0}
        self.assertTrue(first[1] < first[3] < first[5] < first[10])
        self.assertEqual(first[10], set(job['support_ids']))

    def test_modality_snv_remains_separate(self):
        rng = np.random.default_rng(41)
        arrays = {'X_alpha': rng.normal(100, 2, (8, 20)), 'X_n': rng.normal(2, 0.02, (8, 20))}
        fused = worker.representation(arrays, 'fusion')
        np.testing.assert_array_equal(fused[:, :20], worker.snv(arrays['X_n']))
        np.testing.assert_array_equal(fused[:, 20:], worker.snv(arrays['X_alpha']))

    def test_query_values_and_truth_do_not_fit_representation_or_model(self):
        frame, matrix, config, job = fixture()
        first = worker.Cell(job, 'alpha', matrix, frame, config)
        other_matrix = matrix.copy()
        query_indices = [first.index[value] for value in job['query_ids']]
        other_matrix[query_indices] += 1000
        other_frame = frame.copy()
        for index in query_indices:
            other_frame.loc[index, 'origin_group_id'] = worker.GROUPS[(worker.GROUPS.index(other_frame.loc[index, 'origin_group_id']) + 1) % 5]
        second = worker.Cell(job, 'alpha', other_matrix, other_frame, config)
        self.assertEqual(first.fingerprint, second.fingerprint)
        np.testing.assert_array_equal(first.Z_source, second.Z_source)
        a, b = first.make_model(), second.make_model()
        a.train_chunk(first.Z_source, first.y_source, 3, 5, 5)
        b.train_chunk(second.Z_source, second.y_source, 3, 5, 5)
        self.assertEqual(a.state_dict(), b.state_dict())

    def test_checkpoint_rotation_resume_and_corruption_recovery(self):
        frame, matrix, config, job = fixture()
        cell = worker.Cell(job, 'alpha', matrix, frame, config)
        a, b = cell.models(None)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'checkpoint.json'
            for steps in (2, 3):
                for mode, model in zip(('episodic', 'supervised'), (a, b)):
                    model.train_chunk(cell.Z_source, cell.y_source, steps, 5, 5, mode)
                worker.save_checkpoint(path, cell, a, b, 0.5, [])
            self.assertEqual(worker.load_checkpoint(path, cell)['training_steps'], 5)
            restored_a, restored_b = cell.models(worker.load_checkpoint(path, cell))
            for mode, original, restored in zip(('episodic', 'supervised'), (a, b), (restored_a, restored_b)):
                original.train_chunk(cell.Z_source, cell.y_source, 2, 5, 5, mode)
                restored.train_chunk(cell.Z_source, cell.y_source, 2, 5, 5, mode)
                self.assertEqual(original.state_dict(), restored.state_dict())
            path.write_text('{broken')
            recovered = worker.load_checkpoint(path, cell, recover=True)
            self.assertEqual(recovered['training_steps'], 2)
            self.assertEqual(len(list(path.parent.glob('checkpoint.corrupt.*.json'))), 1)
            self.assertTrue((path.parent / 'checkpoint_recovery.json').exists())

    def test_geometric_scoring_is_not_training_stop(self):
        self.assertEqual([i for i in range(1, 20) if worker.scheduled_evaluation(i * 250, 250)], [1, 2, 4, 8, 16])
        self.assertFalse(worker.scheduled_evaluation(0, 250))
        self.assertFalse(worker.scheduled_evaluation(751, 250))

    def test_sealed_scores_recompute_and_tamper_detection(self):
        frame, matrix, config, job = fixture()
        cell = worker.Cell(job, 'alpha', matrix, frame, config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            values = cell.evaluate(models=cell.models(None))
            worker.seal_evaluation(path, cell, 0, *values)
            receipt = worker.verify_evaluation(path, cell)
            self.assertEqual(len(receipt['endpoints']), 2 + 2 * len(cell.draws))
            with (path / 'metrics.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'seal'):
                worker.verify_evaluation(path, cell)

    def test_once_then_resume_advances_without_reset(self):
        frame, matrix, config, job = fixture()
        config['support_draws'] = 1
        cell = worker.Cell(job, 'alpha', matrix, frame, config)
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary), once=True, chunk_steps=2)
            worker.STOP_REQUESTED = False
            with contextlib.redirect_stdout(io.StringIO()):
                worker.execute(args, {'input_hash': 'synthetic-test-only'}, [cell])
                worker.execute(args, {'input_hash': 'synthetic-test-only'}, [cell])
            path = args.output / 'cells' / cell.cell_id
            self.assertEqual(worker.load_checkpoint(path / 'checkpoint.json', cell)['training_steps'], 4)
            self.assertEqual(len(list((path / 'snapshots').glob('*/COMPLETE.json'))), 2)
            baseline = worker.verify_evaluation(path / 'baseline', cell)
            self.assertEqual(len(baseline['endpoints']), 4 + 5 * 4)

    def test_rollback_replay_preserves_sealed_snapshot_hashes(self):
        frame, matrix, config, job = fixture()
        config['support_draws'] = 1
        cell = worker.Cell(job, 'alpha', matrix, frame, config)
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary), once=True, chunk_steps=2)
            worker.STOP_REQUESTED = False
            with contextlib.redirect_stdout(io.StringIO()):
                worker.execute(args, {'input_hash': 'test'}, [cell])
                worker.execute(args, {'input_hash': 'test'}, [cell])
            directory = args.output / 'cells' / cell.cell_id
            snapshot = directory / 'snapshots' / 'step_000000000004'
            hashes = {path.name: worker.file_hash(path) for path in snapshot.iterdir()}
            (directory / 'checkpoint.json').write_text('{broken')
            with contextlib.redirect_stdout(io.StringIO()):
                worker.execute(args, {'input_hash': 'test'}, [cell])
            self.assertEqual(hashes, {path.name: worker.file_hash(path) for path in snapshot.iterdir()})
            values = cell.evaluate(models=cell.models(worker.load_checkpoint(directory / 'checkpoint.json', cell)))
            altered = values[2].copy()
            altered[0, 0, 0] += 0.01
            with self.assertRaisesRegex(ValueError, 'immutable'):
                worker.seal_evaluation(snapshot, cell, 4, values[0], values[1], altered)
            self.assertEqual(hashes, {path.name: worker.file_hash(path) for path in snapshot.iterdir()})

    def test_verify_is_read_only_while_training_lock_is_held(self):
        frame, matrix, config, job = fixture()
        cell = worker.Cell(job, 'alpha', matrix, frame, config)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            baseline = output / 'cells' / cell.cell_id / 'baseline'
            worker.seal_evaluation(baseline, cell, 0, *cell.evaluate(models=cell.models(None)))
            held = (output / 'worker.lock').open('a+')
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = {str(path.relative_to(output)): (worker.file_hash(path), path.stat().st_mtime_ns)
                      for path in output.rglob('*') if path.is_file()}
            with mock.patch.object(worker, 'prepare', return_value=({'input_hash': 'test'}, [cell])) as prepare:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(worker.main(['--verify', '--output', str(output)]), 0)
                self.assertTrue(prepare.call_args.kwargs['readonly'])
            after = {str(path.relative_to(output)): (worker.file_hash(path), path.stat().st_mtime_ns)
                     for path in output.rglob('*') if path.is_file()}
            self.assertEqual(before, after)
            held.close()

    def test_code_hash_keys_are_portable_between_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / name for name in ('first', 'second')]
            hashes = []
            for root in roots:
                path = root / 'src' / 'module.py'
                path.parent.mkdir(parents=True)
                path.write_text('same scientific code\n')
                hashes.append(worker.science_code_hashes(root, [path]))
            self.assertEqual(hashes[0], hashes[1])
            self.assertEqual(list(hashes[0]), ['src/module.py'])

    def test_prepare_preserves_existing_protocol_and_whitelist(self):
        frame, matrix, config, job = fixture()
        root = Path(worker.__file__).resolve().parents[1]
        if not (root / 'src/herb_spectra').exists():
            root = Path('/workspace/research4')
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(root=root, output=Path(temporary), feature_cache=None,
                representations=['alpha'], outer_seeds=[20261031], chunk_steps=2, support_draws=1)
            fake_inputs = (frame, {'X_alpha': matrix, 'X_n': matrix + 3}, np.arange(matrix.shape[1]), {'synthetic_test': True})
            with mock.patch.object(worker, 'load_inputs', return_value=fake_inputs):
                worker.prepare(args)
                before = {path.name: (worker.file_hash(path), path.stat().st_mtime_ns) for path in args.output.iterdir()}
                worker.prepare(args)
                worker.prepare(args, readonly=True)
                after = {path.name: (worker.file_hash(path), path.stat().st_mtime_ns) for path in args.output.iterdir()}
            self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
