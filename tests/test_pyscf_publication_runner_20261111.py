"""Publication-runner integration regressions without expensive SCF work.

The checkpoint test runs the real core optimizer persistence path around a tiny
fake chemistry engine. Other tests isolate state-machine and identity behavior.
"""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import run_pyscf_publication_20261111 as runner


class BudgetPause(BaseException):
    """Model an external budget pause after durable geometry completion."""


class RunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.symbols = ['O', 'H', 'H']
        self.coords = np.array([[0., 0., 0.], [1.43, 0., 1.11], [-1.43, 0., 1.11]])

    def tearDown(self):
        runner.core.configure()

    def point(self, folder, coordinates, *, density_value=1., energy=-2.):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / 'coordinates_bohr.npy', coordinates)
        np.save(folder / 'gradient.npy', np.zeros_like(coordinates))
        np.save(folder / 'density.npy', [[density_value]])
        return {
            'qualified': True, 'native_converged': True, 'symbols': self.symbols,
            'coordinates_bohr': np.asarray(coordinates).tolist(),
            'gradient': np.zeros_like(coordinates).tolist(), 'dipole_au': [0., 0., 0.],
            'gradient_path': str(folder / 'gradient.npy'), 'density_path': str(folder / 'density.npy'),
            'energy_hartree': energy, 'nuclear_gradient_max': 0., 'nuclear_gradient_rms': 0.,
            'electronic_audit': {'fresh_orbital_gradient_norm': 0.,
                                 'fresh_energy_difference_hartree': 0., 'electron_count_error': 0.},
        }

    def optimization(self, folder, coordinates, *, density_value=1., energy=-2.):
        point = self.point(Path(folder) / 'final-center', coordinates,
                           density_value=density_value, energy=energy)
        np.save(Path(folder) / 'coordinates_bohr.npy', coordinates)
        return {'coordinates_bohr': point['coordinates_bohr'], 'density_path': point['density_path'],
                'final_point': point, 'minimum_stationarity_passed': True, 'optimizer_converged': True,
                'array_refs': {'coordinates_bohr': {
                    'path': str(Path(folder) / 'coordinates_bohr.npy')}}}

    def common_patches(self, stack):
        stack.enter_context(mock.patch.object(runner, 'progress'))
        stack.enter_context(mock.patch.object(runner, 'inputs', return_value=(self.symbols, self.coords, None)))
        stack.enter_context(mock.patch.object(runner, 'identity_check', return_value={'passed': True}))
        stack.enter_context(mock.patch.object(runner.modes, 'analyze_modes', return_value={'steps': []}))
        stack.enter_context(mock.patch.object(runner.quality, 'audit_publication',
                                            return_value={'paper_ready_molecular_harmonic_data': True}))

    def test_repeated_internal_repair_uses_final_geometry_and_density(self):
        out = self.root / 'case'
        coords1, coords2 = self.coords + .01, self.coords + .02
        opt0 = self.optimization(out / 'optimize', self.coords, density_value=1.)
        opt1 = self.optimization(out / 'reoptimized-stable-state-00', coords1, density_value=3.)
        opt2 = self.optimization(out / 'reoptimized-stable-state-01', coords2, density_value=5.)
        repaired = []
        for index, value in enumerate((2., 4.)):
            path = self.root / f'orbital-repair-{index}.npy'
            np.save(path, [[value]])
            repaired.append({'tested_electronic_stability_passed': True,
                'supported_for_primary_hessian': False, 'internal_repair_performed': True,
                'geometry_reoptimization_required': True, 'density_path': str(path)})
        closed = {'tested_electronic_stability_passed': True,
                  'supported_for_primary_hessian': True, 'internal_repair_performed': False}
        with ExitStack() as stack:
            self.common_patches(stack)
            optimize = stack.enter_context(mock.patch.object(runner, 'optimize', side_effect=[opt0, opt1, opt2]))
            stability = stack.enter_context(mock.patch.object(runner.core, 'check_electronic_stability',
                                                              side_effect=[*repaired, closed]))
            displacements = stack.enter_context(mock.patch.object(runner, 'displacement_set',
                                                                 return_value=([], [], [])))
            result = runner.stationary_case('water', 'df-grid5', self.coords, None, out)
        self.assertTrue(result['paper_ready'])
        self.assertEqual(optimize.call_count, 3)
        self.assertEqual(stability.call_count, 3)
        np.testing.assert_array_equal(optimize.call_args_list[1].args[2], [[2.]])
        np.testing.assert_array_equal(optimize.call_args_list[2].args[2], [[4.]])
        np.testing.assert_array_equal(displacements.call_args.args[2], coords2)
        np.testing.assert_array_equal(displacements.call_args.args[3], [[5.]])
        self.assertEqual(result['density_path'], opt2['density_path'])
        self.assertEqual(Path(result['center_point_path']).parent,
                         Path(opt2['final_point']['gradient_path']).parent)

    def test_budget_pause_resumes_completed_geometry_without_reoptimizing(self):
        out = self.root / 'case'
        coords = self.coords

        class Molecule:
            def atom_coords(self):
                return coords.copy()

        addons = types.SimpleNamespace(as_pyscf_method=lambda mol, scan: (mol, scan))

        def kernel(method, **kwargs):
            mol, scan = method
            scan(mol)
            return True, mol

        engine = types.SimpleNamespace(kernel=mock.Mock(side_effect=kernel))
        fake_geomopt = types.SimpleNamespace(addons=addons, geometric_solver=engine)

        def calculate(symbols, xyz, folder, **kwargs):
            return self.point(folder, xyz)

        with ExitStack() as stack:
            self.common_patches(stack)
            stack.enter_context(mock.patch.dict(sys.modules, {'pyscf.geomopt': fake_geomopt}))
            stack.enter_context(mock.patch.object(runner.core, '_fingerprint', return_value=('fixed-input', {})))
            stack.enter_context(mock.patch.object(runner.core, 'new_molecule', return_value=Molecule()))
            calculate_point = stack.enter_context(mock.patch.object(runner.core, 'calculate_point', side_effect=calculate))
            stack.enter_context(mock.patch.object(runner.core, 'check_electronic_stability', return_value={
                'tested_electronic_stability_passed': True, 'supported_for_primary_hessian': True}))
            displacements = stack.enter_context(mock.patch.object(runner, 'displacement_set',
                                                                 side_effect=[BudgetPause(), ([], [], [])]))
            with self.assertRaises(BudgetPause):
                runner.stationary_case('water', 'df-grid5', coords, None, out)
            self.assertTrue((out / 'optimize/result.json').exists())
            self.assertFalse((out / 'COMPLETE.json').exists())
            prior_points = calculate_point.call_count
            result = runner.stationary_case('water', 'df-grid5', coords, None, out)
            self.assertEqual(calculate_point.call_count, prior_points)
        self.assertTrue(result['paper_ready'])
        self.assertEqual(engine.kernel.call_count, 1)
        self.assertEqual(displacements.call_count, 2)

    def test_both_negative_mode_signs_relax_despite_uphill_finite_trials(self):
        case_dir = self.root / 'case'
        (case_dir / 'modes').mkdir(parents=True)
        direction = np.zeros((self.coords.size, 1))
        direction[0, 0] = 1.
        np.savez(case_dir / 'modes/analysis_arrays.npz', step_1_cartesian_unit_display_modes=direction)
        center = self.point(case_dir / 'center', self.coords)
        case = {'coordinates_bohr': self.coords.tolist(), 'density_path': center['density_path'],
                'energy_hartree': -2., 'optimization_path': str(case_dir / 'optimize/result.json'),
                'analysis': {'steps': [{'imaginary_count': 1},
                                      {'imaginary_count': 1, 'lowest_signed_eigenvalue_hartree_per_amu_bohr2': -1.}],
                             'step_comparison': {'projected_spectral_step_difference_hartree_per_amu_bohr2': .01}}}

        def calculate(symbols, xyz, folder, **kwargs):
            # Every finite trial is uphill, so relaxation must not require an
            # arbitrary finite-amplitude energy-decrease precondition.
            return self.point(folder, xyz, energy=-2. + abs(float(xyz[0, 0])))

        def optimize(symbols, xyz, density, folder):
            sign = 1 if 'sign+1' in str(folder) else -1
            return self.optimization(folder, xyz, energy=-2.1 if sign == 1 else -2.2)

        with ExitStack() as stack:
            self.common_patches(stack)
            stack.enter_context(mock.patch.object(runner.core, 'calculate_point', side_effect=calculate))
            relax = stack.enter_context(mock.patch.object(runner, 'optimize', side_effect=optimize))
            result = runner.negative_branches('water', 'df-grid5', case, case_dir / 'recovery')
        self.assertEqual(relax.call_count, 2)
        self.assertEqual({row['sign'] for row in result['relaxed_branches']}, {-1, 1})
        self.assertTrue(all(row['energy_change_hartree'] > 0 for row in result['trials']))
        self.assertEqual(result['selected']['sign'], -1)

    def test_tetrahedral_reflection_is_rejected_with_bond_lengths_preserved(self):
        symbols = ['C', 'H', 'F', 'O', 'N']
        vertices = np.array([[1., 1., 1.], [1., -1., -1.],
                             [-1., 1., -1.], [-1., -1., 1.]]) / np.sqrt(3.)
        reference = np.vstack([np.zeros(3), vertices * np.array([2.05, 2.55, 2.6, 2.7])[:, None]])
        mirrored = reference.copy()
        mirrored[:, 0] *= -1
        with mock.patch.dict(sys.modules, {'pyscf.data': types.SimpleNamespace(elements=object())}):
            original = runner.identity_check(symbols, reference, reference)
            changed = runner.identity_check(symbols, reference, mirrored)
        self.assertTrue(original['passed'])
        self.assertFalse(changed['passed'])
        self.assertAlmostEqual(changed['bond_length_ratio_min'], 1.)
        self.assertAlmostEqual(changed['bond_length_ratio_max'], 1.)
        self.assertFalse(changed['tetrahedral_parity'][0]['preserved'])


if __name__ == '__main__':
    unittest.main()
