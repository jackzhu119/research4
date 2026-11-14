"""Fast contract tests for the isolated Spinosin queue; no SCF is executed."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pyscf_repair_core_20261111 as core
import run_pyscf_repair_20261111 as study

# Importing the dedicated executable intentionally changes its process's runtime
# allocation and runner destination. Do not leak those changes to other suites.
_original_settings = copy.deepcopy(core.SCF_SETTINGS)
_original_output, _original_progress = study.OUTPUT, study.progress
direct = importlib.import_module('background_pyscf_spinosin_direct_20261113')
core.SCF_SETTINGS.clear()
core.SCF_SETTINGS.update(_original_settings)
study.OUTPUT, study.progress = _original_output, _original_progress


class IsolatedQueueTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.output = self.root / 'reports/pyscf_spinosin_direct_20261113'
        self.legacy = self.root / 'legacy-calycosin'
        for name, value in {
            'ROOT': self.root, 'OUTPUT': self.output,
            'RUNTIME': self.output / '.background',
            'STATE': self.output / '.background/state.json',
            'PROTOCOL': self.output / 'protocol.json',
            'DOC': self.root / 'docs/PYSCF_SPINOSIN_DIRECT_RESULTS_20261113.md',
            'SCRIPT': self.root / 'scripts/background_pyscf_spinosin_direct_20261113.py',
            'LEGACY': self.legacy,
        }.items():
            self.enterContext(mock.patch.object(direct, name, value))
        self.enterContext(mock.patch.object(study, 'OUTPUT', self.output))
        self.enterContext(mock.patch.object(study, 'progress', direct.progress))
        self.enterContext(mock.patch.dict(core.SCF_SETTINGS, direct.RESOURCE_OVERRIDES))
        self.symbols = ['C'] * 75
        self.coordinates = np.arange(225, dtype=float).reshape(75, 3) / 100
        self.density = np.eye(2)
        self.inputs = self.enterContext(mock.patch.object(
            study, 'molecule_inputs', return_value=(self.symbols, self.coordinates, self.density)))
        self.frozen = {'scientific_files': {'original': 'sealed'}, 'scientific_input_sha256': 'digest'}
        self.snapshot = self.enterContext(mock.patch.object(direct, 'snapshot', return_value=self.frozen))
        self.branch = self.enterContext(mock.patch.object(direct, 'require_branch'))
        self.enterContext(mock.patch.object(direct.bg, 'OUTPUT', self.legacy))
        direct.write(self.legacy / 'frozen_protocol.json', {
            'settings': {'threads': 2, 'max_memory_mb': 5000}, 'molecules': list(study.MOLECULES)})
        direct.write(self.legacy / 'progress.json', {'molecule': 'calycosin', 'status': 'STOPPED'})

    def test_protocol_has_only_spinosin_with_explicit_resources_and_frozen_science(self):
        protocol = direct.protocol()
        self.inputs.assert_called_once_with('spinosin')
        self.assertEqual(protocol['molecule'], 'spinosin')
        self.assertEqual(protocol['execution_resource_overrides'], {'threads': 4, 'max_memory_mb': 10000})
        self.assertEqual(protocol['settings']['threads'], 4)
        self.assertEqual(protocol['settings']['max_memory_mb'], 10000)
        self.assertEqual(protocol['source_protocol']['settings'], {'threads': 2, 'max_memory_mb': 5000})
        self.assertEqual(protocol['optimization_settings'], core.OPTIMIZATION_SETTINGS)
        self.assertEqual(protocol['finite_difference_steps_bohr'], [.001, .0005])
        self.assertEqual(protocol['planned_primary_displaced_points'], 900)
        self.assertEqual(protocol['planned_vibrational_modes_per_step'], 219)
        self.assertIsNone(protocol['retry_count_limit'])
        self.assertIsNone(protocol['quality_runtime_cpu_cutoffs'])
        self.assertTrue(protocol['shared_inputs_read_only'])

    def test_protocol_rejects_wrong_atom_count(self):
        self.inputs.return_value = (['C'] * 74, np.zeros((74, 3)), None)
        with self.assertRaises(study.IntegrityError):
            direct.protocol()

    def test_prepare_is_idempotent_and_refuses_metadata_change(self):
        first = direct.prepare()
        initial_bytes = direct.PROTOCOL.read_bytes()
        self.assertEqual(direct.prepare(), first)
        self.assertEqual(direct.PROTOCOL.read_bytes(), initial_bytes)
        altered = copy.deepcopy(first)
        altered['settings']['max_memory_mb'] = 5000
        direct.write(direct.PROTOCOL, altered)
        with self.assertRaises(study.IntegrityError):
            direct.prepare()
        self.assertEqual(direct.read(direct.PROTOCOL), altered)

    def test_progress_and_status_are_isolated_from_legacy_output(self):
        before = (self.legacy / 'progress.json').read_bytes()
        study.progress(status='RUNNING', molecule='spinosin')
        with mock.patch.object(direct.bg, 'matching_process', return_value=False):
            status = direct.status()
        self.assertEqual(status['progress']['molecule'], 'spinosin')
        self.assertEqual(status['legacy_reported_molecule'], 'calycosin')
        self.assertFalse(status['legacy_spinosin_activity_reported'])
        self.assertEqual((self.legacy / 'progress.json').read_bytes(), before)

    def test_worker_calls_only_spinosin_runner_and_its_own_summary(self):
        with mock.patch.object(direct, 'recover_optimization_steps') as recover, \
             mock.patch.object(study, 'run_molecule') as run_molecule, \
             mock.patch.object(direct, 'make_summary', return_value={'status': 'DONE'}) as summary, \
             mock.patch.object(study, 'run', side_effect=AssertionError('shared queue')), \
             mock.patch.object(study, 'make_report', side_effect=AssertionError('shared report')), \
             mock.patch.object(study, 'verify', side_effect=AssertionError('shared verification')):
            self.assertEqual(direct.worker(), 0)
        recover.assert_called_once_with()
        run_molecule.assert_called_once_with('spinosin')
        summary.assert_called_once_with()
        self.assertEqual(direct.read(self.output / 'progress.json')['molecule'], 'spinosin')
        self.assertFalse((self.legacy / 'results_summary.json').exists())

    def test_worker_lock_prevents_second_computation(self):
        with mock.patch.object(direct.fcntl, 'flock', side_effect=BlockingIOError), \
             mock.patch.object(study, 'run_molecule') as run, \
             mock.patch.object(direct, 'recover_optimization_steps') as recover:
            self.assertEqual(direct.worker(), 1)
        run.assert_not_called()
        recover.assert_not_called()

    def test_supervisor_lock_prevents_duplicate_children(self):
        with mock.patch.object(direct.fcntl, 'flock', side_effect=BlockingIOError), \
             mock.patch.object(direct.subprocess, 'Popen') as spawn:
            self.assertEqual(direct.supervise('second'), 0)
        spawn.assert_not_called()

    def test_launch_is_idempotent_for_live_or_completed_supervisor(self):
        for state, alive in [({'supervisor_pid': 123, **self.frozen}, True),
                             ({'status': 'DONE', **self.frozen}, False)]:
            with self.subTest(state=state):
                direct.write(direct.STATE, state)
                with mock.patch.object(direct.bg, 'matching_process', return_value=alive), \
                     mock.patch.object(direct.subprocess, 'Popen') as spawn:
                    self.assertEqual(direct.launch()['state'], state)
                spawn.assert_not_called()

    def test_supervisor_gives_both_phases_four_threads_and_blocks_integrity_failure(self):
        child = mock.Mock(pid=123)
        child.poll.return_value = 0
        child.wait.return_value = 0
        with mock.patch.object(direct.signal, 'signal'), \
             mock.patch.object(direct.bg, 'matching_process', return_value=False), \
             mock.patch.object(direct.subprocess, 'Popen', return_value=child) as spawn, \
             mock.patch.object(direct, 'publish') as publish:
            self.assertEqual(direct.supervise('test'), 0)
        self.assertEqual([call.args[0][-1] for call in spawn.call_args_list], ['--worker', '--verify'])
        for call in spawn.call_args_list:
            for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
                self.assertEqual(call.kwargs['env'][key], '4')
        publish.assert_called_once()
        child.wait.return_value = direct.INTEGRITY_ERROR
        with mock.patch.object(direct.signal, 'signal'), \
             mock.patch.object(direct.bg, 'matching_process', return_value=False), \
             mock.patch.object(direct.subprocess, 'Popen', return_value=child) as spawn, \
             mock.patch.object(direct, 'publish') as publish:
            self.assertEqual(direct.supervise('blocked'), 42)
        self.assertEqual(spawn.call_count, 1)
        publish.assert_not_called()
        self.assertEqual(direct.read(direct.STATE)['status'], 'BLOCKED_INTEGRITY')

    def test_saved_supervisor_manifest_change_prevents_launch(self):
        direct.write(direct.STATE, {**self.frozen, 'scientific_input_sha256': 'changed'})
        with mock.patch.object(direct.subprocess, 'Popen') as spawn:
            with self.assertRaises(study.IntegrityError):
                direct.launch()
        spawn.assert_not_called()

    def test_eligible_files_exclude_calycosin_and_all_runtime_caches(self):
        allowed = [direct.SCRIPT, direct.DOC, self.output / 'result.json',
                   self.output / '.gitignore', self.output / 'jobs/spinosin/gradient.npy',
                   self.root / 'tests/test_pyscf_spinosin_direct_20261113.py']
        excluded = [self.legacy / 'calycosin/result.json', self.output / '.background/state.json',
                    self.output / '.local/density.npy', self.output / 'cache/result.json',
                    self.output / 'checkpoint/point.json', self.output / 'checkpoints/point.json',
                    self.output / '__pycache__/x.pyc', self.output / 'point.log',
                    self.output / 'x.lock', self.output / 'x.pid', self.output / 'x.tmp',
                    self.output / 'x.partial', self.output / 'x.chk', self.output / 'x.h5']
        for path in allowed + excluded:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('evidence')
        (self.output / 'linked.json').symlink_to(self.legacy / 'calycosin/result.json')
        self.assertEqual(set(direct.eligible_files()), {str(path.relative_to(self.root)) for path in allowed})

    def test_publication_refuses_unrelated_staged_files(self):
        git_result = subprocess.CompletedProcess([], 0, 'reports/pyscf_repair_20261111/calycosin.json\0', '')
        with mock.patch.object(direct, 'eligible_files', return_value=['own.json']), \
             mock.patch.object(direct.bg, 'git', return_value=git_result) as git:
            with self.assertRaises(study.IntegrityError):
                direct.publish(self.frozen, mock.Mock())
        git.assert_called_once_with('diff', '--cached', '--name-only', '-z')

    def test_partial_verify_does_not_invoke_shared_report_or_verification(self):
        with mock.patch.object(study, 'completed_result', return_value=None), \
             mock.patch.object(study, 'verify', side_effect=AssertionError('shared verification')), \
             mock.patch.object(study, 'make_report', side_effect=AssertionError('shared report')):
            self.assertEqual(direct.verify()['status'], 'PARTIAL')

    def test_summary_writes_only_dedicated_report(self):
        with mock.patch.object(study, 'completed_result', return_value=None), \
             mock.patch.object(study, 'make_report', side_effect=AssertionError('shared report')):
            self.assertEqual(direct.make_summary()['status'], 'PARTIAL')
        self.assertTrue(direct.DOC.is_file())
        self.assertTrue((self.output / 'results_summary.json').is_file())
        self.assertFalse((self.legacy / 'results_summary.json').exists())

    def optimization(self):
        opt = self.output / 'jobs/spinosin/grid5-optimize'
        saved = {'status': 'RUNNING', 'evaluations': 0, 'coordinates_bohr': self.coordinates.tolist()}
        direct.write(opt / 'progress.json', saved)
        return opt, opt / 'steps/step-00000', saved

    def completed_point(self, step):
        arrays = {'coordinates_bohr': self.coordinates, 'gradient': np.zeros((75, 3)),
                  'dipole_au': np.zeros(3), 'density': self.density}
        refs = {key: core.atomic_array(step / ('.local/density.npy' if key == 'density' else key + '.npy'), value)
                for key, value in arrays.items()}
        point = {'status': 'DONE', 'qualified': True, 'symbols': self.symbols,
                 'input_fingerprint': 'correct-fingerprint', 'array_refs': refs,
                 'density_path': refs['density']['path'], 'energy_hartree': -1.,
                 'nuclear_gradient_rms': 0., 'nuclear_gradient_max': 0.}
        point.update({key: array.tolist() for key, array in arrays.items() if key != 'density'})
        direct.write(step / 'result.json', point)
        return point

    def test_interrupted_step_is_archived_intact_without_advancing_progress(self):
        opt, step, saved = self.optimization()
        direct.write(step / 'attempt.json', {'status': 'RUNNING', 'raw': 'preserve'})
        before = (opt / 'progress.json').read_bytes()
        recovery = direct.recover_optimization_steps()
        self.assertEqual(recovery[0]['action'], 'ARCHIVE_INTERRUPTED_STEP')
        archive = Path(recovery[0]['archive_path'])
        self.assertEqual(direct.read(archive / 'attempt.json')['raw'], 'preserve')
        self.assertFalse(step.exists())
        self.assertEqual((opt / 'progress.json').read_bytes(), before)
        self.assertEqual(direct.recover_optimization_steps(), [])

    def test_done_point_is_hash_verified_and_proof_persisted_before_checkpoint(self):
        opt, step, saved = self.optimization()
        point = self.completed_point(step)
        real_write, real_validate = direct.write, direct.validate_arrays
        events = []
        def validate(*args, **kwargs):
            result = real_validate(*args, **kwargs)
            events.append('validated')
            return result
        def write(path, value):
            if Path(path) == opt / 'progress.json':
                proof_paths = list((self.output / 'recovery').glob('*.json'))
                self.assertTrue(proof_paths, 'Recovery proof must be durable before progress advances')
                proof = direct.read(proof_paths[0])
                self.assertEqual(proof['point_sha256'], direct.sha(step / 'result.json'))
                events.append('checkpoint')
            return real_write(path, value)
        with mock.patch.object(direct, 'validate_arrays', side_effect=validate), \
             mock.patch.object(core, '_fingerprint', return_value=('correct-fingerprint', {})), \
             mock.patch.object(direct, 'write', side_effect=write):
            recovery = direct.recover_optimization_steps()
        self.assertEqual(events, ['validated', 'checkpoint'])
        self.assertEqual(recovery[0]['action'], 'ROLL_FORWARD_REAL_COMPLETED_POINT')
        self.assertEqual(direct.read(opt / 'progress.json')['evaluations'], 1)
        self.assertEqual(direct.read(opt / 'progress.json')['density_path'], point['density_path'])
        self.assertTrue(step.is_dir())
        self.assertEqual(direct.recover_optimization_steps(), [])

    def test_tampered_done_density_blocks_recovery_without_checkpoint_override(self):
        opt, step, saved = self.optimization()
        point = self.completed_point(step)
        np.save(point['density_path'], self.density * 2, allow_pickle=False)
        with self.assertRaisesRegex(study.IntegrityError, 'hash mismatch'):
            direct.recover_optimization_steps()
        self.assertEqual(direct.read(opt / 'progress.json'), saved)
        self.assertTrue(step.exists())
        self.assertFalse((self.output / 'recovery').exists())

    def test_wrong_input_fingerprint_blocks_recovery_without_archiving_evidence(self):
        opt, step, saved = self.optimization()
        self.completed_point(step)
        with mock.patch.object(core, '_fingerprint', return_value=('different', {})):
            with self.assertRaisesRegex(study.IntegrityError, 'fingerprint mismatch'):
                direct.recover_optimization_steps()
        self.assertEqual(direct.read(opt / 'progress.json'), saved)
        self.assertTrue(step.exists())

    def test_done_point_json_array_disagreement_is_integrity_error(self):
        opt, step, saved = self.optimization()
        point = self.completed_point(step)
        point['gradient'][0][0] = 1.
        direct.write(step / 'result.json', point)
        with self.assertRaisesRegex(study.IntegrityError, 'JSON/array mismatch'):
            direct.recover_optimization_steps()
        self.assertEqual(direct.read(opt / 'progress.json'), saved)

    def test_done_point_requires_all_electronic_array_seals(self):
        opt, step, saved = self.optimization()
        point = self.completed_point(step)
        del point['array_refs']['density']
        direct.write(step / 'result.json', point)
        with self.assertRaisesRegex(study.IntegrityError, 'required array seals'):
            direct.recover_optimization_steps()
        self.assertEqual(direct.read(opt / 'progress.json'), saved)


class OriginalScientificManifestTests(unittest.TestCase):
    def test_all_31_original_files_remain_guarded_with_wrapper_and_supervisor(self):
        original = direct.bg.scientific_snapshot()
        current = direct.snapshot()
        self.assertEqual(len(original['scientific_files']), 31)
        self.assertTrue(set(original['scientific_files']).issubset(current['scientific_files']))
        for name, digest in original['scientific_files'].items():
            self.assertEqual(current['scientific_files'][name], digest)
        self.assertIn('scripts/background_pyscf_spinosin_direct_20261113.py', current['scientific_files'])
        self.assertIn('scripts/background_pyscf_repair_20261111.py', current['scientific_files'])
        self.assertEqual(current['scientific_input_sha256'], hashlib.sha256(
            json.dumps(current['scientific_files'], sort_keys=True).encode()).hexdigest())

    def test_change_to_any_original_guarded_file_is_refused(self):
        baseline = direct.snapshot()
        original = direct.bg.scientific_snapshot()
        for name in original['scientific_files']:
            with self.subTest(path=name):
                altered = copy.deepcopy(baseline)
                altered['scientific_files'][name] = '0' * 64
                with mock.patch.object(direct, 'snapshot', return_value=altered):
                    with self.assertRaises(study.IntegrityError):
                        direct.check_snapshot(baseline)

    def test_branch_mismatch_is_integrity_error(self):
        result = subprocess.CompletedProcess([], 0, 'research/unrelated\n', '')
        with mock.patch.object(direct.bg, 'git', return_value=result):
            with self.assertRaises(study.IntegrityError):
                direct.require_branch()


if __name__ == '__main__':
    unittest.main()
