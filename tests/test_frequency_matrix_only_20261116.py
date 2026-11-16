"""Safety and completion regressions for the user-restricted matrix handover.

These use persisted synthetic points and never start an electronic calculation.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_frequency_matrix_only_20261116 as runner


class MatrixOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ctx = {'root': self.root, 'output': self.root / runner.RELATIVE_OUTPUT,
                    'case': self.root / runner.CASE, 'symbols': ['H'] * 33,
                    'coordinates': np.arange(99, dtype=float).reshape(33, 3),
                    'density': np.array([[7.]]), 'protocol': {'fixed_seed': 'original'},
                    'optimization': {'final_point': {}}, 'stability': {}}

    def point(self, job):
        return {'qualified': True, 'native_converged': True,
                'coordinates_bohr': runner.displaced(self.ctx, job).tolist(),
                'gradient': np.zeros((33, 3)).tolist(),
                'electronic_audit': {'qualified': True, 'native_converged': True,
                    'fresh_orbital_gradient_norm': 0.,
                    'fresh_energy_difference_hartree': 0., 'electron_count_error': 0.}}

    def core(self, point):
        return types.SimpleNamespace(
            _fingerprint=mock.Mock(return_value=('original-fingerprint', {})),
            _read_completed=mock.Mock(return_value=point),
            calculate_point=mock.Mock(side_effect=AssertionError('SCF forbidden in tests')),
            SCF_SETTINGS={'fresh_orbital_gradient_acceptance': 1e-8,
                'fresh_energy_difference_acceptance_hartree': 1e-10,
                'electron_count_acceptance': 1e-7})

    def test_scope_is_exactly_two_steps_of_existing_99_coordinates(self):
        jobs = runner.jobs(self.ctx)
        self.assertEqual(len(jobs), 396)
        self.assertEqual(len({job['path'] for job in jobs}), 396)
        self.assertEqual({job['step_bohr'] for job in jobs}, {.002, .001})
        self.assertTrue(all(job['path'].is_relative_to(self.ctx['case']) for job in jobs))
        for job in jobs:
            difference = runner.displaced(self.ctx, job) - self.ctx['coordinates']
            self.assertEqual(np.count_nonzero(difference), 1)
            self.assertAlmostEqual(difference.flat[job['coordinate']], job['sign'] * job['step_bohr'])
        np.testing.assert_array_equal(self.ctx['density'], [[7.]])

    def test_cached_point_requires_same_seed_fingerprint_and_fresh_quality(self):
        job = runner.jobs(self.ctx)[0]
        point = self.point(job)
        self.ctx['core'] = core = self.core(point)
        self.assertIs(runner.qualified_point(self.ctx, job), point)
        args = core._fingerprint.call_args.args
        self.assertIs(args[3], self.ctx['density'])
        self.assertIsNone(args[2])
        core._read_completed.assert_called_once_with(job['path'], 'original-fingerprint')
        point['electronic_audit']['fresh_orbital_gradient_norm'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'electronic audit'):
            runner.qualified_point(self.ctx, job)
        point['electronic_audit']['fresh_orbital_gradient_norm'] = 1e-7
        with self.assertRaisesRegex(ValueError, 'electronic audit'):
            runner.qualified_point(self.ctx, job)
        point['native_converged'] = False
        self.assertIsNone(runner.qualified_point(self.ctx, job))
        core.calculate_point.assert_not_called()

    def test_mismatched_cached_fingerprint_is_never_silently_recomputed(self):
        job = runner.jobs(self.ctx)[0]
        self.ctx['core'] = core = self.core(None)
        core._read_completed.side_effect = ValueError('Scientific input fingerprint changed')
        with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
            runner.qualified_point(self.ctx, job)
        core.calculate_point.assert_not_called()

    def test_point_claim_excludes_another_process_and_releases_at_close(self):
        path = self.ctx['output'] / '.locks/h1-c098-minus.lock'
        claim = runner.lock_file(path, nonblocking=True)
        code = ('import fcntl,sys; f=open(sys.argv[1], "a+"); '
                'fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)')
        other = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True)
        self.assertNotEqual(other.returncode, 0)
        self.assertIn(b'BlockingIOError', other.stderr)
        claim.close()
        other = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True)
        self.assertEqual(other.returncode, 0, other.stderr)

    def test_missing_one_point_cannot_assemble_or_seal(self):
        self.ctx['publication'] = types.SimpleNamespace(assemble=mock.Mock())
        self.ctx['modes'] = types.SimpleNamespace(analyze_modes=mock.Mock())
        with mock.patch.object(runner, 'qualified_point', side_effect=[None]):
            self.assertIsNone(runner.finalize(self.ctx))
        self.ctx['publication'].assemble.assert_not_called()
        self.ctx['modes'].analyze_modes.assert_not_called()
        self.assertFalse((self.ctx['output'] / 'COMPLETE.json').exists())

    def complete_synthetic_matrix(self):
        for job in runner.jobs(self.ctx):
            runner.write(job['path'] / 'result.json', self.point(job))
        self.ctx['core'] = types.SimpleNamespace(atomic_array=lambda path, value: np.save(path, value))
        self.ctx['publication'] = types.SimpleNamespace(
            assemble=mock.Mock(return_value=(np.eye(99), np.full((3, 99), 123.))),
            center_quality=mock.Mock(return_value={}))

        def analyze(symbols, xyz, matrices, dipole_derivatives, output, **kwargs):
            self.assertIsNone(dipole_derivatives)
            self.assertEqual(len(kwargs['point_quality']), 396)
            self.assertEqual(kwargs['steps_bohr'], runner.STEPS)
            output.mkdir(parents=True)
            np.savez(output / 'analysis_arrays.npz', raw_hessian=matrices[0])
            (output / 'all_modes.csv').write_text('signed_frequency_cm1\n-12.3\n')
            result = {'minimum_verified': False, 'qualification': {'minimum_verified': False},
                      'steps': [{'imaginary_count': 1, 'signed_frequencies_cm1': [-12.3]}],
                      'step_comparison': {'projected_relative_step_difference': .5}}
            runner.write(output / 'result.json', result)
            return result

        def replay(output):
            receipt = {'status': 'VERIFIED'}
            runner.write(output / 'independent_mode_verification.json', receipt)
            return receipt

        self.ctx['modes'] = types.SimpleNamespace(analyze_modes=mock.Mock(side_effect=analyze),
                                                verify_analysis=mock.Mock(side_effect=replay))
        with mock.patch.object(runner, 'qualified_point', side_effect=lambda ctx, job: self.point(job)):
            return runner.finalize(self.ctx)

    def test_whole_matrix_completion_preserves_imaginaries_and_cancels_ir(self):
        summary = self.complete_synthetic_matrix()
        self.assertEqual(summary['status'], 'COMPLETE')
        self.assertEqual(summary['qualified_points'], 396)
        self.assertFalse(summary['frequency_minimum_verified'])
        self.assertEqual(summary['signed_frequency_steps'][0]['signed_frequencies_cm1'], [-12.3])
        self.assertEqual(summary['IR_status'], 'CANCELLED_BY_USER_NOT_COMPUTED')
        self.assertEqual(summary['independent_ir_scf_points_computed'], 0)
        self.assertEqual(summary['subsequent_queue_status'], 'CANCELLED_BY_USER')
        self.assertEqual(runner.verify_complete(self.ctx['output']), summary)
        self.ctx['modes'].analyze_modes.assert_called_once()

    def test_completion_requires_full_artifact_seal_and_untampered_arrays(self):
        self.complete_synthetic_matrix()
        seal_path = self.ctx['output'] / 'COMPLETE.json'
        original = runner.read(seal_path)
        empty = {**original, 'artifacts_sha256': {}}
        runner.write(seal_path, empty)
        with self.assertRaisesRegex(ValueError, 'lacks matrix or replay'):
            runner.verify_complete(self.ctx['output'])
        runner.write(seal_path, original)
        (self.ctx['output'] / 'hessian_h1.npy').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'scientific artifact changed'):
            runner.verify_complete(self.ctx['output'])


if __name__ == '__main__':
    unittest.main()
