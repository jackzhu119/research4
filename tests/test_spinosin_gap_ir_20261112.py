"""Algebraic IR controls, raw-point replay and negative-matrix veto; no SCF."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import spinosin_gap_ir_20261112 as gap


class IRGapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.before = copy.deepcopy(gap.electronic.SCF_SETTINGS)
        gap.electronic.configure('df-grid5', threads=2)
        gap.electronic.SCF_SETTINGS['max_memory_mb'] = 3000
        self.settings = gap.electronic.settings()
        self.symbols = ['O', 'H', 'H']
        self.coords = np.asarray([[0., 0., 0.], [1.43, 0., 1.11], [-1.43, 0., 1.11]])
        masses = gap.modes.atomic_masses(self.symbols)
        _, basis, _ = gap.modes.external_basis(self.coords, masses)
        root_mass = np.sqrt(np.repeat(masses, 3))
        self.hessian = root_mass[:, None] * (basis @ np.diag([0.03, 0.07, 0.11]) @ basis.T) * root_mass[None, :]
        coupling = np.asarray([[0.01, 0.02, 0.03], [-0.02, 0.01, 0.015], [0.03, 0.04, -0.01]])
        self.derivative = coupling @ basis.T * root_mass[None, :]
        self.seed = np.asarray([[2.]])
        center_dir = self.root / 'source-center'
        self.center = self.point(self.symbols, self.coords, center_dir, self.seed, require_density=True)
        self.calls = []
        self.artifact = {'case_id': 'water-control', 'symbols': self.symbols,
            'coordinates_bohr': self.coords.tolist(), 'settings': self.settings,
            'center_point': self.center, 'source_fingerprint': 'new-water-center-001',
            'steps_bohr': [0.002, 0.001]}
        self.prepare_matrices(self.hessian)

    def tearDown(self):
        gap.electronic.SCF_SETTINGS.clear()
        gap.electronic.SCF_SETTINGS.update(self.before)
        self.temp.cleanup()

    def point(self, symbols, coordinates, output, density, *, require_density=False):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        delta = np.asarray(coordinates) - self.coords
        dipole = self.derivative @ delta.ravel()
        refs = {key: gap._array(output / name, value) for key, name, value in (
            ('coordinates_bohr', 'coordinates_bohr.npy', coordinates),
            ('dipole_au', 'dipole_au.npy', dipole),
            ('density', '.local/density.npy', self.seed))}
        (output / 'rebuildable.chk').write_bytes(b'rebuildable only')
        point = {'status': 'DONE', 'qualified': True, 'symbols': list(symbols),
            'coordinates_bohr': np.asarray(coordinates).tolist(), 'settings': gap.electronic.settings(),
            'input_fingerprint': gap._hash({'coordinates': np.asarray(coordinates).tolist(),
                'settings': gap.electronic.settings(), 'density': gap._array_hash(density)}),
            'energy_hartree': float(0.5 * delta.ravel() @ self.hessian @ delta.ravel()),
            'dipole_au': dipole.tolist(), 'density_path': refs['density']['path'],
            'array_refs': refs, 'electronic_audit': {'qualified': True,
                'native_converged': True, 'fresh_orbital_gradient_norm': 1e-12,
                'fresh_energy_difference_hartree': 0., 'electron_count_error': 0.}}
        gap._write(output / 'result.json', point)
        if hasattr(self, 'calls'):
            self.calls.append(str(output))
        return point

    def prepare_matrices(self, hessian):
        paths, derivative_paths = [], []
        for i in range(2):
            paths.append(gap._array(self.root / f'hessian_{i}.npy', hessian)['path'])
            derivative_paths.append(gap._array(self.root / f'derivative_{i}.npy', self.derivative)['path'])
        analysis_dir = self.root / ('source-modes-' + gap._array_hash(hessian))
        center_quality = {'qualified': True, 'native_converged': True,
            'fresh_orbital_gradient_norm': 1e-12, 'energy_recheck_difference_hartree': 0.,
            'electron_count_error': 0., 'nuclear_gradient_max': 0., 'nuclear_gradient_rms': 0.,
            'orbital_stability_passed': True}
        gap.modes.analyze_modes(self.symbols, self.coords, [hessian, hessian],
            [self.derivative, self.derivative], analysis_dir, center_quality=center_quality,
            point_quality=[center_quality] * 36, steps_bohr=self.artifact['steps_bohr'])
        gates = {key: True for key in ('source_artifacts_authenticated', 'distinct_steps',
            'center_electronic_quality', 'center_stationary', 'orbital_stability_passed',
            'complete_unique_displacement_ledger', 'all_points_electronically_qualified',
            'consistent_electronic_model_settings', 'raw_derivatives_reassembled',
            'all_vibrational_eigenvalues_positive', 'raw_antisymmetry_passed',
            'all_step_comparisons_passed', 'force_sum_rules_passed')}
        authenticated = {str((analysis_dir / 'result.json').resolve()): gap._sha(analysis_dir / 'result.json')}
        for ref in gap._read(analysis_dir / 'result.json')['artifacts'].values():
            file = analysis_dir / ref['path']
            authenticated[str(file.resolve())] = gap._sha(file)
        self.artifact.update(hessian_paths=paths, dipole_derivative_paths=derivative_paths,
            analysis_path=str(analysis_dir), minimum_evidence={'minimum_verified': True, 'gates': gates,
                'original_displacement_point_count': 36,
                'center_point_sha256': hashlib.sha256(json.dumps(self.center, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                'authenticated_artifact_sha256': authenticated})

    def test_linear_dipole_modal_math_mass_units_resume_and_density_cleanup(self):
        output = self.root / 'ir'
        result = gap.run_ir_check(self.artifact, output, point_evaluator=self.point)
        self.assertTrue(result['paper_ready'])
        self.assertEqual(result['mode_count'], 3)
        self.assertEqual(result['additional_cartesian_points'], 18)
        self.assertEqual(result['direct_normal_mode_points'], 18)
        self.assertEqual(len(self.calls), 37)  # 1 center + 18 Cartesian + 18 direct modal.
        vectors = np.load(result['array_refs']['mass_normalized_modes']['path'])
        expected = gap.quality.prefactor_si() * np.sum((self.derivative @ vectors) ** 2, axis=0)
        np.testing.assert_allclose(np.load(result['array_refs']['cartesian_IR_km_mol']['path']), expected, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(np.load(result['array_refs']['direct_IR_km_mol']['path']), expected, rtol=1e-11, atol=1e-12)
        self.assertTrue(all(result['independent_actual_hessian_check'][key] for key in (
            'both_actual_hessians_all_modes_strictly_positive', 'both_actual_raw_antisymmetry_passed')))
        self.assertEqual(len(list(output.rglob('density.npy'))), 1)  # Kept center only.
        self.assertEqual(len(list(output.rglob('*.chk'))), 1)
        count = len(self.calls)
        repeated = gap.run_ir_check(self.artifact, output, point_evaluator=self.point)
        self.assertEqual(repeated, result)
        self.assertEqual(len(self.calls), count)
        gap.verify_ir_result(Path(result['csv']['path']).parent)

    def test_committed_point_interruption_resumes_without_repeat_scf(self):
        output = self.root / 'ir'
        def interrupt(event):
            if event.get('stage') == 'IR_POINT_COMMITTED' and event.get('scientific_phase') == 'IR_DIRECT_NORMAL_MODE_DIPOLE':
                raise InterruptedError('explicit safe stop after one committed normal point')
        with self.assertRaises(InterruptedError):
            gap.run_ir_check(self.artifact, output, point_evaluator=self.point, progress=interrupt)
        first_calls = list(self.calls)
        self.assertEqual(len(first_calls), 20)
        result = gap.run_ir_check(self.artifact, output, point_evaluator=self.point)
        self.assertTrue(result['paper_ready'])
        self.assertEqual(len(self.calls), 37)
        self.assertEqual(len(set(self.calls)), 37)
        self.assertEqual(gap.electronic.settings(), self.settings)

    def test_negative_hessian_cannot_inherit_true_minimum_flags(self):
        self.prepare_matrices(-self.hessian)
        with self.assertRaisesRegex(ValueError, 'Hessians contradict'):
            gap.run_ir_check(self.artifact, self.root / 'ir', point_evaluator=self.point)
        self.assertEqual(self.calls, [])

    def test_required_mode_provenance_and_original_density_checksum(self):
        missing = dict(self.artifact)
        missing.pop('analysis_path')
        with self.assertRaisesRegex(ValueError, 'provenance is required'):
            gap.run_ir_check(missing, self.root / 'ir', point_evaluator=self.point)
        with Path(self.center['density_path']).open('ab') as stream:
            stream.write(b'changed seed')
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            gap.run_ir_check(self.artifact, self.root / 'ir', point_evaluator=self.point)
        self.assertEqual(self.calls, [])

    def test_waiting_states_do_not_grant_early_qualification(self):
        absent = dict(self.artifact, hessian_paths=None, dipole_derivative_paths=None)
        result = gap.run_ir_check(absent, self.root / 'ir', point_evaluator=self.point)
        self.assertEqual(result['status'], 'DERIVATIVES_ONLY_AWAITING_FULL_HESSIAN')
        self.assertFalse(result['paper_ready'])
        not_minimum = dict(self.artifact, minimum_evidence={'minimum_verified': False})
        result = gap.run_ir_check(not_minimum, self.root / 'ir', point_evaluator=self.point)
        self.assertEqual(result['status'], 'AWAITING_QUALIFIED_MINIMUM')
        self.assertFalse(result['computed_complete'])
        self.assertEqual(self.calls, [])

    def test_case_and_mode_identity_science_arrays_remain_immutable(self):
        result = gap.run_ir_check(self.artifact, self.root / 'ir', point_evaluator=self.point)
        point_ledger = gap._read(result['point_ledger']['path'])
        point_dir = Path(point_ledger[-1]['directory'])
        with (point_dir / 'dipole_au.npy').open('ab') as stream:
            stream.write(b'corrupt scientific array')
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            gap.run_ir_check(self.artifact, self.root / 'ir', point_evaluator=self.point)
        with self.assertRaisesRegex(ValueError, 'Only the two new'):
            gap.run_ir_check(dict(self.artifact, case_id='C001'), self.root / 'old-case', point_evaluator=self.point)

    def test_sampling_plan_advances_without_a_global_round_limit(self):
        first, later = gap.sampling_plan(0, [0.002, 0.001]), gap.sampling_plan(8, [0.002, 0.001])
        self.assertEqual(first['precision_level'], 0)
        self.assertEqual(later['precision_level'], 2)
        self.assertLess(max(later['cartesian_steps_bohr']), max(first['cartesian_steps_bohr']))
        self.assertEqual(len(set(later['cartesian_steps_bohr'])), 3)

    def test_modal_vector_sign_error_fails_even_when_squared_intensity_agrees(self):
        def sign_error(symbols, coordinates, output, density, *, require_density=False):
            point = self.point(symbols, coordinates, output, density, require_density=require_density)
            if 'normal_modes' in Path(output).parts:
                dipole = -np.asarray(point['dipole_au'])
                point['dipole_au'] = dipole.tolist()
                point['array_refs']['dipole_au'] = gap._array(Path(output) / 'dipole_au.npy', dipole)
                gap._write(Path(output) / 'result.json', point)
            return point
        output = self.root / 'ir'
        first = gap.run_ir_check(self.artifact, output, point_evaluator=sign_error)
        self.assertFalse(first['paper_ready'])
        self.assertEqual(first['status'], 'IR_REFINEMENT_REQUIRED')
        self.assertTrue(first['common_frame_check']['qualified'])
        self.assertIn('direct_directional_and_cartesian_vectors_disagree', first['directional_check']['failures'])
        np.testing.assert_allclose(np.load(first['array_refs']['cartesian_IR_km_mol']['path']),
                                   np.load(first['array_refs']['direct_IR_km_mol']['path']), atol=1e-11)
        before = gap._sha(Path(first['csv']['path']).parent / 'result.json')
        second = gap.run_ir_check(self.artifact, output, point_evaluator=sign_error)
        self.assertEqual(second['sampling_plan']['sampling_index'], 1)
        self.assertEqual(second['sampling_plan']['precision_level'], 1)
        self.assertEqual(gap._sha(Path(first['csv']['path']).parent / 'result.json'), before)
        self.assertFalse(second['paper_ready'])

    def test_shared_streaming_guard_never_reads_mf_cderi_or_builds_full_tensor(self):
        class FakeDF:
            _cderi = None
            def get_jk(self, density, with_j=True, with_k=True):
                return ('native-streamed-J', with_k)
            def build(self):
                self._cderi = 'forbidden dense tensor'
            def loop(self):
                return self._cderi
        class FakeMF:
            def __init__(self):
                self.with_df = FakeDF()
                self.mol = SimpleNamespace(incore_anyway=False)
            @property
            def _cderi(self):
                raise AssertionError('Reading this MF property would prebuild a giant tensor')
        core = SimpleNamespace(make_scf=lambda: FakeMF())
        evidence = gap.ensure_streaming_jonly_guard(core)
        factory = core.make_scf
        gap.ensure_streaming_jonly_guard(core)
        self.assertIs(core.make_scf, factory)
        mf = core.make_scf()
        self.assertEqual(mf.with_df.get_jk(self.seed, with_k=False), ('native-streamed-J', False))
        self.assertIsNone(mf.with_df._cderi)
        self.assertFalse(evidence['mf_cderi_property_inspected'])
        for function in (mf.with_df.build, mf.with_df.loop, lambda: mf.with_df.get_jk(self.seed)):
            with self.assertRaisesRegex(RuntimeError, 'DENSE_DF_DISABLED'):
                function()


if __name__ == '__main__':
    unittest.main()
