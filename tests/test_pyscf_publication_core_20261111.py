"""Focused, no-chemistry regressions for profile isolation and orbital repair."""
from contextlib import ExitStack
import tempfile
from pathlib import Path
import types
import unittest
import sys
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pyscf_publication_core_20261111 as core


class FakeMF:
    def __init__(self, log, stable=True, external=True, energy=-1.0, density=None):
        self.log = log
        self.stable, self.external, self.energy = stable, external, energy
        self.mo_coeff = np.eye(2)
        self.mo_occ = np.array([2.0, 0.0])
        self.dm = np.diag(self.mo_occ) if density is None else density

    def stability(self, *, internal, external, return_status):
        if internal:
            angle = 0.2
            rotated = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
            return rotated, None, self.stable, None
        self.log.write('rhf_real2complex: lowest eigs of H = [0.25 0.5]\n')
        return None, None, None, self.external

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            return self.dm
        return (mo_coeff * mo_occ) @ mo_coeff.T


def audit(mf):
    return mf.dm, {'qualified': True, 'fresh_energy_hartree': mf.energy}


class PublicationCoreTests(unittest.TestCase):
    def tearDown(self):
        core.configure()

    def test_effective_profile_and_grid_change_fingerprints(self):
        coords = np.zeros((1, 3))
        with mock.patch.dict('sys.modules', {'pyscf': types.SimpleNamespace(__version__='test')}):
            core.configure('direct-grid5', threads=1)
            direct, _ = core._fingerprint(['He'], coords, None, None)
            core.configure('df-grid5', threads=1)
            fitted, _ = core._fingerprint(['He'], coords, None, None)
            self.assertNotEqual(direct, fitted)
            core.configure('df-grid7', threads=2)
            fine, payload = core._fingerprint(['He'], coords, None, None)
            self.assertNotEqual(fine, fitted)
            self.assertEqual(payload['settings']['grid_level'], 7)
            self.assertEqual(payload['settings']['density_fitting_auxbasis'], 'def2-universal-jkfit')
            overridden, payload = core._fingerprint(['He'], coords, 5, None)
            self.assertNotEqual(fine, overridden)
            self.assertEqual(payload['settings']['grid_level'], 5)
            core.configure('direct-grid7', threads=2)
            exact_fine, payload = core._fingerprint(['He'], coords, None, None)
            self.assertNotEqual(exact_fine, fine)
            self.assertNotEqual(exact_fine, direct)
            self.assertEqual(payload['settings']['grid_level'], 7)
            self.assertFalse(payload['settings']['density_fitting'])
            self.assertIsNone(payload['settings']['density_fitting_auxbasis'])
            with self.assertRaises(ValueError):
                core.configure('df-grid5', threads=3)

    def _stability_case(self, *, initial_stable=False, external=True, repair_stable=True):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        out = Path(temporary.name)
        log = (out / 'pyscf.log').open('w')
        initial = FakeMF(log, stable=initial_stable, external=external)
        repaired = FakeMF(log, stable=repair_stable, external=external,
                          energy=-1.1, density=np.array([[1.9, 0.1], [0.1, 0.1]]))
        seeds = []

        def run_electronic(symbols, coords, directory, grid_level, dm, stream):
            seeds.append(np.asarray(dm).copy())
            return repaired, repaired.dm, audit(repaired)[1], []

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(core, '_fingerprint', return_value=('input', {})))
            stack.enter_context(mock.patch.object(core, '_diagnostic_mf', return_value=(initial, log)))
            stack.enter_context(mock.patch.object(core, '_audit', side_effect=audit))
            stack.enter_context(mock.patch.object(core, '_run_electronic', side_effect=run_electronic))
            result = core.check_electronic_stability(['He'], np.zeros((1, 3)), out)
        return result, seeds

    def test_unstable_orbitals_are_repaired_then_checked(self):
        result, seeds = self._stability_case()
        self.assertEqual(len(seeds), 1)
        self.assertFalse(np.allclose(seeds[0], np.diag([2.0, 0.0])))
        self.assertTrue(result['tested_electronic_stability_passed'])
        self.assertTrue(result['geometry_reoptimization_required'])
        self.assertFalse(result['supported_for_primary_hessian'])
        self.assertEqual(len(result['internal_repair_attempts']), 2)
        np.testing.assert_array_equal(np.load(result['density_path']), [[1.9, 0.1], [0.1, 0.1]])

    def test_external_instability_never_certifies(self):
        result, seeds = self._stability_case(initial_stable=True, external=False)
        self.assertEqual(seeds, [])
        self.assertFalse(result['tested_electronic_stability_passed'])
        self.assertFalse(result['supported_for_primary_hessian'])
        self.assertEqual(result['electronic_state_disposition'],
                         'UNSUPPORTED_OR_UNRESOLVED_ELECTRONIC_STABILITY')

    def test_internal_recovery_is_bounded_without_false_pass(self):
        result, seeds = self._stability_case(repair_stable=False)
        self.assertEqual(len(seeds), 3)
        self.assertFalse(result['tested_electronic_stability_passed'])
        self.assertEqual(len(result['internal_repair_attempts']), 4)


if __name__ == '__main__':
    unittest.main()
