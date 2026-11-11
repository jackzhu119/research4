"""Independent physical invariants and evidence integrity for the repair study.

The numerical examples are bond-spring force constants with known spectra,
not an implementation-shaped reconstruction of the production diagonalizer.
No production molecular calculations are launched by this suite.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy import constants

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pyscf_repair_core_20261111 as core
import pyscf_repair_modes_20261111 as modes


def spring_hessian(coordinates, springs):
    """Exact Hessian of central harmonic bonds at their equilibrium lengths."""
    coordinates = np.asarray(coordinates, dtype=float)
    hessian = np.zeros((coordinates.size, coordinates.size))
    for i, j, force_constant in springs:
        direction = coordinates[i] - coordinates[j]
        direction /= np.linalg.norm(direction)
        bond_block = force_constant * np.outer(direction, direction)
        a, b = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
        hessian[a, a] += bond_block
        hessian[b, b] += bond_block
        hessian[a, b] -= bond_block
        hessian[b, a] -= bond_block
    return hessian


class ElectronicAuditTests(unittest.TestCase):
    """The audit must independently reject finite but incorrect SCF states."""

    @staticmethod
    def state(*, converged=True, energy_difference=0., residual=0., electrons=2.):
        mf = mock.Mock()
        mf.converged = converged
        mf.mol.nelectron = 2
        mf.e_tot = -1.
        mf.mo_coeff = np.eye(2)
        mf.mo_occ = np.array([2., 0.])
        mf.make_rdm1.return_value = np.diag([electrons, 0.])
        mf.get_ovlp.return_value = np.eye(2)
        mf.get_hcore.return_value = np.eye(2)
        mf.get_veff.return_value = np.zeros((2, 2))
        mf.energy_tot.return_value = -1. + energy_difference
        mf.get_fock.return_value = np.eye(2)
        mf.get_grad.return_value = np.array([residual])
        return mf

    def test_exact_fresh_electronic_state_qualifies(self):
        _, audit = core._audit(self.state())
        self.assertTrue(audit['qualified'])
        self.assertEqual(audit['electron_count'], 2.)
        self.assertEqual(audit['quality_warnings'], [])

    def test_native_convergence_does_not_replace_fresh_energy_check(self):
        _, audit = core._audit(self.state(energy_difference=1e-8))
        self.assertFalse(audit['qualified'])
        self.assertIn('FRESH_ENERGY_RECHECK_TARGET_NOT_MET', audit['quality_warnings'])

    def test_native_convergence_does_not_replace_fock_residual_check(self):
        _, audit = core._audit(self.state(residual=1e-5))
        self.assertFalse(audit['qualified'])
        self.assertIn('FRESH_ORBITAL_GRADIENT_TARGET_NOT_MET', audit['quality_warnings'])

    def test_electron_count_check_is_independent(self):
        _, audit = core._audit(self.state(electrons=1.999))
        self.assertFalse(audit['qualified'])
        self.assertIn('ELECTRON_COUNT_TARGET_NOT_MET', audit['quality_warnings'])

    def test_unconverged_finite_state_does_not_qualify(self):
        _, audit = core._audit(self.state(converged=False))
        self.assertFalse(audit['qualified'])
        self.assertIn('NATIVE_SCF_NOT_CONVERGED', audit['quality_warnings'])

    def test_nonfinite_audit_rejected(self):
        with self.assertRaises(ValueError):
            core._audit(self.state(residual=float('nan')))


class EvidenceIntegrityTests(unittest.TestCase):
    def test_corrupt_saved_array_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = core.atomic_array(root / 'gradient.npy', np.zeros((2, 3)))
            core.atomic_json(root / 'result.json', {
                'status': 'DONE', 'input_fingerprint': 'same-input',
                'array_refs': {'gradient': ref},
            })
            self.assertIsNotNone(core._read_completed(root, 'same-input'))
            np.save(root / 'gradient.npy', np.ones((2, 3)))
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                core._read_completed(root, 'same-input')

    def test_different_input_cannot_reuse_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core.atomic_json(root / 'result.json', {
                'status': 'DONE', 'input_fingerprint': 'old-input', 'array_refs': {},
            })
            with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                core._read_completed(root, 'new-input')

    def test_unqualified_state_is_not_a_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core.atomic_json(root / 'result.json', {
                'status': 'UNQUALIFIED', 'input_fingerprint': 'same-input',
                'qualified': False, 'array_refs': {},
            })
            self.assertIsNone(core._read_completed(root, 'same-input'))

    @unittest.skipUnless(importlib.util.find_spec('pyscf'), 'Use .venv-pyscf for versioned PySCF fingerprints')
    def test_fingerprint_binds_geometry_grid_density_and_operation(self):
        symbols, coordinates, density = ['H', 'H'], np.array([[0., 0., 0.], [0., 0., 1.4]]), np.eye(2)
        base = core._fingerprint(symbols, coordinates, 5, density)[0]
        displaced = coordinates.copy(); displaced[1, 2] += 1e-6
        variants = [
            core._fingerprint(symbols, displaced, 5, density)[0],
            core._fingerprint(symbols, coordinates, 3, density)[0],
            core._fingerprint(symbols, coordinates, 5, density * .999)[0],
            core._fingerprint(symbols, coordinates, 5, None)[0],
            core._fingerprint(symbols, coordinates, 5, density, 'analytic_hessian')[0],
        ]
        self.assertEqual(len({base, *variants}), 6)

    def test_archived_negative_curvature_is_preserved_with_same_geometry(self):
        path = ROOT / 'data/research/pyscf_repair_20261111/history_controls'
        q = np.load(path / 'acid_old_negative_mode01_unit_cartesian.npy', allow_pickle=False).reshape(-1)
        hessian = np.load(path / 'acid_old_hessian_used.npy', allow_pickle=False)
        hessian = hessian.transpose(0, 2, 1, 3).reshape((len(q), len(q)))
        coordinates = np.load(path / 'acid_old_geometry_bohr.npy', allow_pickle=False)
        self.assertEqual(coordinates.shape, (17, 3))
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1., places=13)
        self.assertAlmostEqual(float(q @ hessian @ q), -0.053314663688628236, places=12)

    def test_archive_files_match_recorded_sha256(self):
        path = ROOT / 'data/research/pyscf_repair_20261111/history_controls'
        provenance = json.loads((path / 'provenance.json').read_text())
        self.assertTrue(provenance)
        entries = provenance if isinstance(provenance, list) else provenance.get('artifacts', provenance.get('files', []))
        self.assertTrue(entries, 'Archive provenance must list independently hashable artifacts')
        for item in entries:
            candidate = item.get('destination') or item.get('local_path') or item.get('destination_path')
            self.assertIsNotNone(candidate)
            candidate = Path(candidate)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            self.assertEqual(core.sha256(candidate), item['sha256'])


class PhysicalModeTests(unittest.TestCase):
    @staticmethod
    def hcl(force_constant=.4):
        symbols = ['H', 'Cl']
        coordinates = np.array([[0., 0., -1.], [0., 0., 1.]])
        hessian = spring_hessian(coordinates, [(0, 1, force_constant)])
        dipole = np.zeros((3, 6))
        dipole[2, [2, 5]] = [-1., 1.]
        return symbols, coordinates, hessian, dipole

    @staticmethod
    def water_springs():
        symbols = ['O', 'H', 'H']
        coordinates = np.array([[0., 0., 0.], [1., 0., 0.], [-.25, 1., 0.]])
        hessian = spring_hessian(coordinates, [(0, 1, .5), (0, 2, .5), (1, 2, .2)])
        return symbols, coordinates, hessian

    def test_diatomic_reduced_mass_frequency_from_SI_units(self):
        symbols, coordinates, hessian, dipole = self.hcl()
        result = modes.normal_modes(symbols, coordinates, hessian, dipole)
        mass_h, mass_cl = modes.atomic_masses(symbols)
        reduced_mass = mass_h * mass_cl / (mass_h + mass_cl)
        hartree = constants.physical_constants['Hartree energy'][0]
        bohr = constants.physical_constants['Bohr radius'][0]
        atomic_mass = constants.physical_constants['atomic mass constant'][0]
        expected_cm1 = np.sqrt(.4 * hartree / (reduced_mass * atomic_mass * bohr ** 2)) / (2 * np.pi * constants.c * 100)
        self.assertEqual(result['diagnostics']['external_rank'], 5)
        self.assertEqual(result['diagnostics']['vibrational_mode_count'], 1)
        self.assertAlmostEqual(result['signed_frequencies_cm1'][0], expected_cm1, places=9)
        self.assertAlmostEqual(result['signed_frequencies_THz'][0], expected_cm1 * constants.c / 1e10, places=11)

    def test_known_dipole_stretch_intensity_uses_mass_normalized_modes(self):
        symbols, coordinates, hessian, dipole = self.hcl()
        result = modes.normal_modes(symbols, coordinates, hessian, dipole)
        masses = modes.atomic_masses(symbols)
        atomic_mass = constants.physical_constants['atomic mass constant'][0]
        expected_factor = constants.N_A * constants.e ** 2 / (12 * constants.epsilon_0 * constants.c ** 2 * atomic_mass * 1000)
        expected_intensity = expected_factor * np.sum(1 / masses)
        self.assertAlmostEqual(result['IR_intensities_km_mol'][0], expected_intensity, places=9)
        # Display normalization changes the derivative; it cannot be used for IR.
        wrong_display_intensity = expected_factor * np.sum((dipole @ result['cartesian_unit_display_modes']) ** 2)
        self.assertGreater(abs(wrong_display_intensity - expected_intensity), 1.)

    def test_nonlinear_water_has_three_internal_modes_and_mass_metric(self):
        symbols, coordinates, hessian = self.water_springs()
        result = modes.normal_modes(symbols, coordinates, hessian)
        self.assertEqual(result['diagnostics']['external_rank'], 6)
        self.assertEqual(result['diagnostics']['vibrational_mode_count'], 3)
        self.assertEqual(result['diagnostics']['positive_count'], 3)
        cartesian = result['cartesian_mass_normalized_modes']
        mass_metric = np.repeat(modes.atomic_masses(symbols), 3)
        np.testing.assert_allclose(cartesian.T @ (mass_metric[:, None] * cartesian), np.eye(3), atol=1e-13)
        self.assertLess(result['diagnostics']['mass_weighted_external_residual_relative'], 1e-13)

    def test_negative_force_constant_keeps_signed_imaginary_frequency(self):
        symbols, coordinates, negative, dipole = self.hcl(-.4)
        minus = modes.normal_modes(symbols, coordinates, negative, dipole)
        plus = modes.normal_modes(symbols, coordinates, -negative, dipole)
        self.assertEqual(minus['diagnostics']['imaginary_count'], 1)
        self.assertLess(minus['eigenvalues'][0], 0)
        self.assertLess(minus['signed_frequencies_cm1'][0], 0)
        self.assertAlmostEqual(minus['signed_frequencies_cm1'][0], -plus['signed_frequencies_cm1'][0], places=9)

    def test_translation_rotation_and_atom_permutation_preserve_spectrum_and_IR(self):
        symbols, coordinates, hessian, dipole = self.hcl()
        reference = modes.normal_modes(symbols, coordinates, hessian, dipole)
        rotation = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        transform = np.kron(np.eye(2), rotation)
        rotated = modes.normal_modes(symbols, coordinates @ rotation.T + [20., -13., 42.],
                                     transform @ hessian @ transform.T, rotation @ dipole @ transform.T)
        order = np.array([3, 4, 5, 0, 1, 2])
        permuted = modes.normal_modes(symbols[::-1], coordinates[::-1], hessian[np.ix_(order, order)], dipole[:, order])
        for transformed in [rotated, permuted]:
            np.testing.assert_allclose(transformed['signed_frequencies_cm1'], reference['signed_frequencies_cm1'], atol=1e-9)
            np.testing.assert_allclose(transformed['IR_intensities_km_mol'], reference['IR_intensities_km_mol'], atol=1e-9)

    def test_raw_antisymmetry_reported_before_symmetric_analysis(self):
        symbols, coordinates, hessian, dipole = self.hcl()
        raw = hessian.copy(); raw[0, 1] += 2e-4; raw[1, 0] -= 2e-4
        result = modes.normal_modes(symbols, coordinates, raw, dipole)
        reference = modes.normal_modes(symbols, coordinates, hessian, dipole)
        self.assertAlmostEqual(result['diagnostics']['raw_max_absolute_antisymmetry_hartree_per_bohr2'], 4e-4)
        self.assertGreater(result['diagnostics']['raw_relative_antisymmetry'], 0)
        np.testing.assert_allclose(result['signed_frequencies_cm1'], reference['signed_frequencies_cm1'], atol=1e-10)
        np.testing.assert_array_equal(raw, result['raw_hessian'])

    def test_missing_or_nonfinite_hessian_is_rejected(self):
        symbols, coordinates, hessian, _ = self.hcl()
        with self.assertRaises(ValueError):
            modes.normal_modes(symbols, coordinates, hessian[:-1])
        hessian[0, 0] = np.nan
        with self.assertRaises(ValueError):
            modes.normal_modes(symbols, coordinates, hessian)


class MinimumQualificationTests(unittest.TestCase):
    @staticmethod
    def center():
        return {
            'native_converged': True, 'fresh_orbital_gradient_norm': 1e-9,
            'energy_recheck_difference_hartree': 0.,
            'nuclear_gradient_max': 1e-7, 'nuclear_gradient_rms': 1e-8,
            'orbital_stability_passed': True,
        }

    def analyze(self, *, center=None, point_quality=None, fine_scale=1., negative=False, dipoles=True):
        symbols, coordinates, hessian, dipole = PhysicalModeTests.hcl(-.4 if negative else .4)
        output = self.enterContext(tempfile.TemporaryDirectory())
        return modes.analyze_modes(
            symbols, coordinates, [hessian, hessian * fine_scale],
            [dipole, dipole] if dipoles else None, output,
            center_quality=self.center() if center is None else center,
            point_quality=[self.center() for _ in range(24)] if point_quality is None else point_quality,
        ), Path(output)

    def test_stationary_stable_positive_dual_step_data_can_qualify(self):
        result, output = self.analyze()
        self.assertTrue(result['minimum_verified'])
        self.assertTrue(result['qualification']['paper_ready_molecular_harmonic_data'])
        replay = modes.verify_analysis(output)
        self.assertEqual(replay['status'], 'VERIFIED')
        self.assertEqual(replay['mode_rows_checked'], 2)

    def test_positive_modes_cannot_replace_orbital_stability(self):
        center = self.center(); center.pop('orbital_stability_passed')
        result, _ = self.analyze(center=center)
        self.assertTrue(result['qualification']['nuclear_minimum_numerically_supported'])
        self.assertFalse(result['minimum_verified'])
        self.assertFalse(result['qualification']['paper_ready_molecular_harmonic_data'])

    def test_positive_modes_cannot_replace_tight_nuclear_stationarity(self):
        center = self.center(); center['nuclear_gradient_max'] = 3.627949137569031e-5
        center['nuclear_gradient_rms'] = 1.14218e-5
        result, _ = self.analyze(center=center)
        self.assertFalse(result['qualification']['stationary_geometry_verified'])
        self.assertFalse(result['minimum_verified'])

    def test_one_unconverged_displacement_prevents_qualification(self):
        bad = self.center(); bad['native_converged'] = False
        result, _ = self.analyze(point_quality=[self.center() for _ in range(23)] + [bad])
        self.assertFalse(result['qualification']['point_electronic_quality']['all_points_qualified'])
        self.assertFalse(result['minimum_verified'])

    def test_truncated_point_ledger_cannot_claim_complete_hessian_evidence(self):
        result, _ = self.analyze(point_quality=[self.center()])
        self.assertFalse(result['minimum_verified'])

    def test_explicit_upstream_failure_or_invalid_qualification_vetoes_analysis(self):
        for declared in [False, None, 1, 'true']:
            with self.subTest(qualified=declared):
                point = self.center(); point['qualified'] = declared
                self.assertFalse(modes.electronic_quality(point)['qualified'])

    def test_declared_electron_count_failure_or_missing_value_vetoes_analysis(self):
        for count_error in [1e-4, -1e-4, None, float('nan')]:
            with self.subTest(electron_count_error=count_error):
                point = self.center(); point['electron_count_error'] = count_error
                self.assertFalse(modes.electronic_quality(point)['qualified'])

    def test_true_upstream_flag_cannot_override_independent_bad_residual(self):
        point = self.center(); point.update(qualified=True, fresh_orbital_gradient_norm=1e-4)
        self.assertFalse(modes.electronic_quality(point)['qualified'])

    def test_dual_step_disagreement_prevents_qualification_without_losing_modes(self):
        result, _ = self.analyze(fine_scale=1.02)
        self.assertFalse(result['qualification']['projected_step_agreement_passed'])
        self.assertFalse(result['minimum_verified'])
        self.assertEqual(len(result['steps']), 2)
        self.assertEqual([len(step['signed_frequencies_cm1']) for step in result['steps']], [1, 1])

    def test_imaginary_modes_stay_in_saved_csv_and_do_not_qualify(self):
        result, output = self.analyze(negative=True)
        self.assertFalse(result['minimum_verified'])
        self.assertEqual([step['imaginary_count'] for step in result['steps']], [1, 1])
        import csv
        with (output / 'all_modes.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(float(row['signed_frequency_cm1']) < 0 for row in rows))
        self.assertTrue(all(row['imaginary'] == 'True' for row in rows))

    def test_no_dipole_derivatives_does_not_claim_paper_ready_IR_data(self):
        result, _ = self.analyze(dipoles=False)
        self.assertTrue(result['minimum_verified'])
        self.assertEqual(result['IR_status'], 'NOT_COMPUTED')
        self.assertFalse(result['qualification']['paper_ready_molecular_harmonic_data'])

    def test_changed_analysis_inputs_cannot_overwrite_saved_science(self):
        _, output = self.analyze()
        symbols, coordinates, hessian, dipole = PhysicalModeTests.hcl()
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            modes.analyze_modes(symbols, coordinates, [hessian, hessian * 1.001], [dipole, dipole], output,
                                center_quality=self.center(), point_quality=[self.center() for _ in range(24)])


@unittest.skipUnless((ROOT / 'scripts/run_pyscf_repair_20261111.py').exists(), 'Runner not yet written')
class FiniteDifferenceAssemblyTests(unittest.TestCase):
    def setUp(self):
        import run_pyscf_repair_20261111 as runner
        self.runner = runner

    def test_columns_are_coordinate_derivatives_not_transposed(self):
        dimension, step = 6, .001
        expected_hessian = np.arange(36, dtype=float).reshape(6, 6) / 10
        expected_dipole = np.arange(18, dtype=float).reshape(3, 6) / 7
        base_gradient = np.arange(6, dtype=float)
        base_dipole = np.array([1., -2., 3.])
        plus = base_gradient[None, :] + step * expected_hessian.T
        minus = base_gradient[None, :] - step * expected_hessian.T
        plus_mu = base_dipole[None, :] + step * expected_dipole.T
        minus_mu = base_dipole[None, :] - step * expected_dipole.T
        actual_hessian, actual_dipole = self.runner.assemble_fd(plus.reshape(dimension, 2, 3), minus.reshape(dimension, 2, 3), plus_mu, minus_mu, step)
        np.testing.assert_allclose(actual_hessian, expected_hessian, atol=1e-12)
        np.testing.assert_allclose(actual_dipole, expected_dipole, atol=1e-12)

    def test_missing_displacements_cannot_be_zero_filled(self):
        with self.assertRaises(ValueError):
            self.runner.assemble_fd(np.zeros((5, 6)), np.zeros((6, 6)), np.zeros((6, 3)), np.zeros((6, 3)), .001)

    def test_nonfinite_and_nonpositive_steps_rejected(self):
        gradients = np.zeros((6, 6)); dipoles = np.zeros((6, 3))
        bad = gradients.copy(); bad[2, 3] = np.nan
        with self.assertRaises(ValueError):
            self.runner.assemble_fd(bad, gradients, dipoles, dipoles, .001)
        for step in [0., -.001, float('nan')]:
            with self.assertRaises(ValueError):
                self.runner.assemble_fd(gradients, gradients, dipoles, dipoles, step)


class RunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        import run_pyscf_repair_20261111 as runner
        self.runner = runner
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary)
        self.enterContext(mock.patch.object(runner, 'ROOT', self.root))
        self.enterContext(mock.patch.object(runner, 'OUTPUT', self.root / 'reports'))

    def test_pilot_success_is_based_on_selected_water_not_full_queue_status(self):
        report = {'status': 'PARTIAL', 'completed_molecules': 1, 'molecules': {'water': {}}}
        with mock.patch.object(self.runner, 'prepare'), mock.patch.object(self.runner, 'run_molecule') as run_molecule, mock.patch.object(self.runner, 'make_report', return_value=report):
            self.assertEqual(self.runner.run(['water']), 0)
            run_molecule.assert_called_once_with('water')

    def test_optional_historical_diagnostic_failure_does_not_block_primary(self):
        report = {'status': 'PARTIAL', 'completed_molecules': 1, 'molecules': {'acid': {}}}
        with mock.patch.object(self.runner, 'prepare'), mock.patch.object(self.runner, 'old_acid_control', side_effect=RuntimeError('comparison failed')), mock.patch.object(self.runner, 'run_molecule') as primary, mock.patch.object(self.runner, 'make_report', return_value=report):
            self.assertEqual(self.runner.run(['acid']), 0)
            primary.assert_called_once_with('acid')
        error = self.runner.read(self.runner.OUTPUT / 'history_controls/acid-old-negative-direction/diagnostic_error.json')
        self.assertFalse(error['primary_numerical_calculation_blocked'])

    def test_corrupt_history_seal_remains_an_integrity_failure(self):
        out = self.runner.OUTPUT / 'history_controls/control'
        self.runner.write(out / 'result.json', {'status': 'DONE', 'value': 1})
        self.runner.write(out / 'COMPLETE.json', {'result_sha256': self.runner.sha(out / 'result.json')})
        self.runner.write(out / 'result.json', {'status': 'DONE', 'value': 2})
        with self.assertRaises(self.runner.IntegrityError):
            self.runner.completed_result(out)
        with self.assertRaises(self.runner.IntegrityError):
            self.runner.optional_diagnostic_error(out, self.runner.IntegrityError('corrupt history'))

    def mocked_molecule(self, imaginary_counts, follow_return=None):
        symbols, coordinates, hessian, dipole = PhysicalModeTests.hcl()
        point = {'qualified': True, 'native_converged': True, 'energy_hartree': -1.,
                 'coordinates_bohr': coordinates.tolist(), 'nuclear_gradient_max': 1e-7,
                 'nuclear_gradient_rms': 1e-8, 'electronic_audit': {
                    'fresh_orbital_gradient_norm': 1e-9, 'fresh_energy_difference_hartree': 0.,
                    'electron_count_error': 0.,}}
        opt = {'minimum_stationarity_passed': True, 'coordinates_bohr': coordinates.tolist(), 'final_point': point}
        analysis = {'steps': [{'imaginary_count': count,
                              'lowest_signed_eigenvalue_hartree_per_amu_bohr2': -.1 if count else .1}
                             for count in imaginary_counts],
                    'step_comparison': {'projected_spectral_step_difference_hartree_per_amu_bohr2': .001}}
        patches = [
            mock.patch.object(self.runner, 'molecule_inputs', return_value=(symbols, coordinates, None)),
            mock.patch.object(core, 'optimize_geometry', return_value=opt),
            mock.patch.object(core, 'check_electronic_stability', return_value={'tested_electronic_stability_passed': True}),
            mock.patch.object(self.runner, 'density', return_value=np.eye(2)),
            mock.patch.object(self.runner, 'complete_displacements', return_value=([hessian, hessian], [dipole, dipole], [point])),
            mock.patch.object(modes, 'analyze_modes', return_value=analysis),
            mock.patch.object(core, 'native_analytic_hessian', side_effect=RuntimeError('optional analytical comparison failed')),
        ]
        for patch in patches:
            self.enterContext(patch)
        follower = self.enterContext(mock.patch.object(self.runner, 'follow_negative_mode', return_value=follow_return))
        return self.runner.run_molecule('water'), follower

    def test_coarse_only_negative_mode_does_not_advance_geometry_round(self):
        value, follower = self.mocked_molecule([1, 0])
        follower.assert_not_called()
        self.assertFalse(value['negative_mode_followup']['geometry_round_advanced'])
        self.assertEqual(value['negative_mode_followup']['status'], 'NUMERICALLY_AMBIGUOUS_NEGATIVE_MODE_RETAINED')
        self.assertTrue((self.runner.OUTPUT / 'jobs/water/COMPLETE.json').exists())

    def test_supported_negative_without_actual_downhill_candidate_does_not_advance_round(self):
        value, follower = self.mocked_molecule([1, 1], follow_return=None)
        follower.assert_called_once()
        self.assertFalse(value['negative_mode_followup']['geometry_round_advanced'])
        state = self.runner.read(self.runner.OUTPUT / 'jobs/water/repair_state.json')
        self.assertEqual(state['round'], 0)
        self.assertEqual(state['status'], 'DONE')

    def test_signed_trial_selection_rejects_non_downhill_energies(self):
        symbols, coordinates, hessian, _ = PhysicalModeTests.hcl(-.4)
        trial = {'qualified': True, 'energy_hartree': -1., 'coordinates_bohr': coordinates.tolist(), 'density_path': 'unused'}
        out = self.root / 'negative-trials'
        with mock.patch.object(core, 'calculate_point', return_value=trial) as points:
            selected = self.runner.follow_negative_mode('test', symbols, coordinates, None, hessian, out, center_energy=-1.)
        self.assertIsNone(selected)
        self.assertEqual(points.call_count, 2)
        record = self.runner.read(out / 'selection.json')
        self.assertFalse(record['selected_for_new_round'])


if __name__ == '__main__':
    unittest.main()
