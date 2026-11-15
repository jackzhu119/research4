import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import spinosin_gap_science_20261112 as science


class FakeCore:
    def __init__(self):
        self.SCF_SETTINGS = {'profile': 'df-grid5', 'grid_level': 5, 'threads': 2,
                             'max_memory_mb': 3000, 'xc': 'PBE', 'basis': 'def2-SVP',
                             'charge': 0, 'spin': 0, 'density_fitting': True}
        self.reference = np.asarray(science.water_control_case()['initial_geometry_bohr'])
        self.calls = []
        self.optimization_calls = []
        self.interrupt_optimizer = False

    def configure(self, profile, threads=2):
        self.SCF_SETTINGS.update(profile=profile, grid_level=5 if profile.endswith('5') else 7,
                                 threads=threads, density_fitting=profile.startswith('df-'))

    def settings(self):
        return copy.deepcopy(self.SCF_SETTINGS)

    def calculate_point(self, symbols, coords, out, dm0=None):
        coords, out = np.asarray(coords), Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.calls.append(str(out))
        density = science._array(out / '.local' / 'density.npy', np.eye(3))
        (out / '.local' / 'rolling.chk').write_bytes(b'reconstructible checkpoint')
        gradient = 0.1 * (coords - self.reference)
        dipole = np.sum(coords - self.reference, axis=0) * 0.01
        refs = {'coordinates_bohr': science._array(out / 'coordinates_bohr.npy', coords),
                'gradient': science._array(out / 'gradient.npy', gradient),
                'dipole_au': science._array(out / 'dipole_au.npy', dipole), 'density': density}
        value = {'status': 'DONE', 'qualified': True, 'native_converged': True,
                 'symbols': list(symbols), 'coordinates_bohr': coords.tolist(),
                 'settings': self.settings(), 'input_fingerprint': science._hash({'out': str(out), 'coords': coords.tolist()}),
                 'energy_hartree': -76 + float(np.sum((coords - self.reference) ** 2) * 0.05),
                 'gradient': gradient.tolist(), 'dipole_au': dipole.tolist(),
                 'gradient_path': refs['gradient']['path'], 'density_path': density['path'],
                 'nuclear_gradient_rms': float(np.sqrt(np.mean(gradient ** 2))),
                 'nuclear_gradient_max': float(np.max(np.abs(gradient))),
                 'electronic_audit': {'qualified': True, 'native_converged': True,
                     'fresh_orbital_gradient_norm': 1e-12, 'fresh_energy_difference_hartree': 1e-14,
                     'electron_count_error': 0}, 'array_refs': refs}
        science._atomic_json(out / 'result.json', value)
        return value

    def optimize_geometry(self, symbols, coords, out, dm0=None):
        out = Path(out)
        self.optimization_calls.append(str(out))
        self.reference = np.asarray(coords).copy()
        if self.interrupt_optimizer:
            self.interrupt_optimizer = False
            orphan = self.calculate_point(symbols, coords, out / 'steps' / 'step-00000', dm0)
            science._atomic_json(out / 'progress.json', {'evaluations': 0, 'coordinates_bohr': np.asarray(coords).tolist(),
                                                         'density_path': orphan['density_path']})
            raise InterruptedError('point committed before optimizer progress advanced')
        point = self.calculate_point(symbols, coords, out / 'final', dm0)
        result = {'optimizer_converged': True, 'minimum_stationarity_passed': True,
                  'coordinates_bohr': np.asarray(coords).tolist(), 'density_path': point['density_path'],
                  'final_point': point}
        science._atomic_json(out / 'result.json', result)
        return result

    def check_electronic_stability(self, symbols, coords, out, dm0=None):
        return {'tested_electronic_stability_passed': True, 'supported_for_primary_hessian': True,
                'geometry_reoptimization_required': False, 'internal_stable': True, 'external_stable': True}


class FakeModes:
    def __init__(self, negative_first=False):
        self.negative_first = negative_first
        self.calls = 0

    def analyze_modes(self, symbols, coords, hessians, derivatives, out, **kwargs):
        self.calls += 1
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        n = np.asarray(coords).size
        arrays = {'coordinates_bohr': coords,
                  'step_1_cartesian_unit_display_modes': np.eye(n)[:, :n-6]}
        for index, (hessian, derivative) in enumerate(zip(hessians, derivatives)):
            arrays[f'step_{index}_raw_hessian'] = hessian
            arrays[f'step_{index}_dipole_derivative_au'] = derivative
        np.savez(out / 'analysis_arrays.npz', **arrays)
        negative = self.negative_first and self.calls == 1
        step = {'imaginary_count': int(negative),
                'lowest_signed_eigenvalue_hartree_per_amu_bohr2': -1.0 if negative else 0.1}
        result = {'steps': [step, step],
                  'step_comparison': {'projected_spectral_step_difference_hartree_per_amu_bohr2': 1e-6},
                  'arrays_sha256': science._file_hash(out / 'analysis_arrays.npz')}
        science._atomic_json(out / 'result.json', result)
        return result

    def verify_analysis(self, output, write_receipt=False):
        output = Path(output)
        result = json.loads((output / 'result.json').read_text())
        if result['arrays_sha256'] != science._file_hash(output / 'analysis_arrays.npz'):
            raise science.ScientificIntegrityError('Changed fake mode arrays')
        return {'verified': True}


class FakeQuality:
    def __init__(self, modes, always_fail=False):
        self.modes, self.always_fail = modes, always_fail

    def audit_publication(self, modes_dir, fd, point, **kwargs):
        negative = self.modes.negative_first and self.modes.calls == 1
        refs = {str(path.resolve()): science._file_hash(path) for path in
                (Path(modes_dir) / 'result.json', Path(modes_dir) / 'analysis_arrays.npz')}
        return {'minimum_verified': not negative and not self.always_fail,
                'paper_ready_molecular_harmonic_data': not negative and not self.always_fail,
                'authenticated_artifact_sha256': refs,
                'gates': {'center_stationary': True, 'all_points_electronically_qualified': True},
                'failed_gates': ['negative_mode'] if negative else []}


def fake_ir(center, output, progress):
    return {'status': 'QUALIFIED', 'computed_complete': True, 'minimum_verified': True,
            'ir_numerically_qualified': True, 'paper_ready': True,
            'case_id': center['case_id'], 'source_fingerprint': center['source_fingerprint']}


class ScienceTests(unittest.TestCase):
    def setUp(self):
        self.core = FakeCore()
        self.modes = FakeModes()
        self.quality = FakeQuality(self.modes)
        self.vendor = {'core': self.core, 'modes': self.modes, 'quality': self.quality,
                       'helpers': SimpleNamespace(identity_check=lambda *args: {'passed': True})}
        self.case = science.water_control_case()

    def run_case(self, output, progress=None, ir=fake_ir):
        with patch.object(science, '_vendor', return_value=self.vendor), patch.object(science, '_ir', ir):
            return science.run_conformer(self.case, output, progress=progress)

    def test_full_dynamic_fd_assembly_and_pruned_point_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_case(directory)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertTrue(result['minimum_verified'] and result['ir_verified'])
            center = result['center_artifact']
            evidence = science._read_record(center['hessian_evidence_path'])
            self.assertEqual(evidence['completed_points'], 36)
            for path in center['hessian_paths']:
                np.testing.assert_allclose(np.load(path), np.eye(9) * 0.1, atol=1e-12)
            fd = Path(center['hessian_evidence_path']).parent / 'finite-differences'
            self.assertEqual(len(list(fd.glob('h*/c*/POINT_SEAL.json'))), 36)
            self.assertFalse(list(fd.glob('h*/c*/**/.local/*')))
            calls = len(self.core.calls)
            replay = self.run_case(directory)
            self.assertEqual(replay['status'], 'COMPLETE')
            self.assertEqual(len(self.core.calls), calls)

    def test_point_commit_interruption_resumes_without_repeating_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            def interrupt(event):
                if event.get('stage') == 'finite_difference_point_committed' and event['completed_points'] == 4:
                    raise InterruptedError('manual stop after committed point')
            with self.assertRaises(InterruptedError):
                self.run_case(directory, interrupt)
            fd_calls = [path for path in self.core.calls if '/finite-differences/' in path]
            self.assertEqual(len(fd_calls), 4)
            self.run_case(directory)
            fd_calls = [path for path in self.core.calls if '/finite-differences/' in path]
            self.assertEqual(len(fd_calls), 36)
            self.assertEqual(len(set(fd_calls)), 36)

    def test_orphan_optimizer_point_restarts_in_new_chunk(self):
        self.core.interrupt_optimizer = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InterruptedError):
                self.run_case(directory)
            result = self.run_case(directory)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertEqual(len(self.core.optimization_calls), 2)
            self.assertNotEqual(*self.core.optimization_calls)
            first = Path(self.core.optimization_calls[0])
            recovery = science._read_record(first / 'WORKFLOW_CHUNK.json')
            self.assertEqual(len(recovery['orphan_point_directories_preserved']), 1)
            self.assertTrue((first / 'steps' / 'step-00000' / 'gradient.npy').is_file())

    def test_chunk_commit_before_state_commit_preserves_pending_new_seed(self):
        original_save = science._save_state
        fired = False
        def interrupt_before_state(output, state, progress, stop_requested, **event):
            nonlocal fired
            if event.get('stage') == 'geometry_chunk_committed' and not fired:
                fired = True
                raise InterruptedError('chunk result committed before global state commit')
            return original_save(output, state, progress, stop_requested, **event)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(science, '_save_state', interrupt_before_state):
                with self.assertRaises(InterruptedError):
                    self.run_case(directory)
            pending = science._read_record(Path(self.core.optimization_calls[0]) / 'WORKFLOW_CHUNK.json')
            seed = Path(pending['density_ref']['path'])
            self.assertTrue(seed.exists())
            result = self.run_case(directory)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertEqual(len(self.core.optimization_calls), 1)
            self.assertTrue(seed.exists())

    def test_supported_negative_geometry_gets_new_isolated_generation(self):
        self.modes.negative_first = True
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_case(directory)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertEqual(result['generation'], 1)
            roots = list((Path(directory) / 'generations').glob('g*'))
            self.assertEqual(len(roots), 2)
            centers = [science._read_record(root / 'CENTER.json') for root in roots]
            self.assertNotEqual(centers[0]['source_fingerprint'], centers[1]['source_fingerprint'])
            self.assertEqual(sum(len(list(root.glob('finite-differences/h*/c*/POINT_SEAL.json'))) for root in roots), 72)

    def test_numerical_failure_waits_without_recomputing_same_derivatives(self):
        self.quality.always_fail = True
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_case(directory)
            self.assertEqual(result['status'], 'REFINEMENT_REQUIRED')
            self.assertFalse(result['minimum_verified'])
            self.assertTrue(result['waiting_for_new_protocol'])
            self.assertFalse((Path(directory) / 'COMPLETE.json').exists())
            calls = len(self.core.calls)
            self.run_case(directory)
            self.assertEqual(len(self.core.calls), calls)

    def test_ir_refinement_continues_same_center_and_requires_all_flags(self):
        ir_calls = []
        def refine(center, output, progress):
            ir_calls.append(center['source_fingerprint'])
            if len(ir_calls) == 1:
                return {'status': 'IR_REFINEMENT_REQUIRED', 'computed_complete': True,
                        'minimum_verified': True, 'ir_numerically_qualified': False,
                        'paper_ready': False, 'source_fingerprint': center['source_fingerprint']}
            return fake_ir(center, output, progress)
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_case(directory, ir=refine)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertEqual(len(ir_calls), 2)
            self.assertEqual(ir_calls[0], ir_calls[1])
            self.assertEqual(result['generation'], 0)
            self.assertEqual(len([path for path in self.core.calls if '/finite-differences/' in path]), 36)

    def test_changed_derivatives_or_other_case_complete_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_case(directory)
            path = Path(result['center_artifact']['hessian_paths'][0])
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaises(science.ScientificIntegrityError):
                self.run_case(directory)
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_case(directory)
            result['case_id'] = 'OTHER_LANE'
            science._atomic_json(Path(directory) / 'result.json', result)
            receipt = science._read_record(Path(directory) / 'COMPLETE.json')
            receipt['result_sha256'] = science._file_hash(Path(directory) / 'result.json')
            science._record(Path(directory) / 'COMPLETE.json', receipt)
            with self.assertRaisesRegex(science.ScientificIntegrityError, 'another case'):
                self.run_case(directory)

    def test_swallowed_optimizer_resource_guard_escalates_model(self):
        original = self.core.optimize_geometry
        profiles = []
        def guarded_optimizer(symbols, coords, out, dm0=None):
            profile = self.core.settings()['profile']
            profiles.append(profile)
            if len(profiles) > 4:
                raise InterruptedError('test protection: resource failure repeated without escalation')
            if profile.startswith('df-'):
                science._atomic_json(Path(out) / 'steps' / 'step-00000' / 'electronic-attempts.json',
                                     [{'error': 'DENSE_DF_DISABLED_J_ONLY_STREAMING_REQUIRED: with_k'}])
                raise RuntimeError('No finite electronic state; original guard was recorded by vendor')
            return original(symbols, coords, out, dm0)
        with tempfile.TemporaryDirectory() as directory, patch.object(self.core, 'optimize_geometry', guarded_optimizer):
            result = self.run_case(directory)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertEqual(profiles, ['df-grid5', 'df-grid7', 'direct-grid7'])
            self.assertEqual(result['generation'], 2)

    def test_swallowed_point_resource_guard_is_not_an_infinite_retry(self):
        attempts = []
        def unavailable(symbols, coords, out, dm0=None):
            attempts.append(str(out))
            science._atomic_json(Path(out) / 'electronic-attempts.json',
                                 [{'error': 'DENSE_DF_DISABLED_J_ONLY_STREAMING_REQUIRED: build'}])
            raise RuntimeError('No finite electronic state')
        with tempfile.TemporaryDirectory() as directory, patch.object(self.core, 'calculate_point', unavailable):
            with self.assertRaises(science.UnsupportedStreamingPath):
                science._point(self.case['symbols'], np.asarray(self.case['initial_geometry_bohr']),
                               Path(directory), self.core, 'frozen-center', None, False)
            self.assertEqual(len(attempts), 1)


class StreamingGuardTests(unittest.TestCase):
    def test_factory_guard_blocks_dense_paths_but_allows_native_j_only(self):
        class Fitting:
            _cderi = None
            def get_jk(self, dm, hermi=1, with_j=True, with_k=True):
                return 'native-j-only'
            def build(self):
                raise AssertionError('dense build was reached')
            def loop(self):
                raise AssertionError('dense loop was reached')
        class MF:
            with_df = None
            mol = SimpleNamespace(incore_anyway=False)
            @property
            def _cderi(self):
                raise AssertionError('mf._cderi property must never be read')
        def factory():
            mf = MF()
            mf.with_df = Fitting()
            return mf
        core = SimpleNamespace(make_scf=factory)
        science.install_jonly_streaming_guard(core)
        first = core.make_scf
        science.install_jonly_streaming_guard(core)
        self.assertIs(first, core.make_scf)
        mf = core.make_scf()
        self.assertEqual(mf.with_df.get_jk(None, with_k=False), 'native-j-only')
        for callback in (mf.with_df.build, mf.with_df.loop, lambda: mf.with_df.get_jk(None),
                         lambda: mf.with_df.get_jk(None, with_k=True)):
            with self.assertRaises(science.UnsupportedStreamingPath):
                callback()
        self.assertIsNone(mf.with_df._cderi)

    def test_only_authorized_spinosin_cases_and_explicit_water_control(self):
        registry = science.ROOT / 'data/research/spinosin_ir_gap_fill_20261112/inputs/cases.json'
        cases = json.loads(registry.read_text())['cases']
        for case in cases:
            _, symbols, coords = science._validate_case(case)
            self.assertEqual(len(symbols), 75)
            self.assertEqual(coords.size * 4, 900)
        wrong = copy.deepcopy(cases[0])
        wrong['case_id'] = 'C001'
        with self.assertRaises(ValueError):
            science._validate_case(wrong)

    def test_short_optimizer_segments_restore_native_solver_and_are_not_global_limits(self):
        calls = []
        def kernel(*args, **kwargs):
            calls.append(kwargs['maxsteps'])
            return kwargs['maxsteps']
        solver = SimpleNamespace(kernel=kernel)
        core = SimpleNamespace(__name__='pyscf_publication_core_20261111')
        with patch.object(science.importlib, 'import_module', return_value=solver):
            with science._bounded_optimizer_segment(core):
                self.assertEqual(solver.kernel(maxsteps=100), 12)
            self.assertIs(solver.kernel, kernel)
        self.assertEqual(calls, [12])
        self.assertIsNone(science.DESIGN['global_geometry_round_limit'])
        self.assertIsNone(science.DESIGN['global_retry_limit'])


if __name__ == '__main__':
    unittest.main()
