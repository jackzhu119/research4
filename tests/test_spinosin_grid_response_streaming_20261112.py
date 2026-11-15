"""Native-equivalent full response and bounded 75-atom allocation checks."""
from pathlib import Path
import hashlib
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import numpy as np
from pyscf import dft, gto, lib
from pyscf.dft import gen_grid
from pyscf.grad import rks as native_rks

import spinosin_grid_response_streaming_20261112 as adapter


class GridResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib.num_threads(1)
        cls.mol = gto.M(atom=[('O', (0., 0., 0.)), ('H', (1.43, 0., 1.11)),
                             ('H', (-1.43, 0., 1.11))],
                       unit='Bohr', basis='def2-SVP', verbose=0, max_memory=1000)

    def tearDown(self):
        adapter.uninstall()

    def test_weights_responses_and_original_atom_owners_match_native(self):
        grids = dft.gen_grid.Grids(self.mol)
        grids.level = 0
        reference = list(adapter._NATIVE_PARTITION(grids))
        blocked = list(adapter.iter_grid_response_blocks(grids, block_size=73))
        self.assertGreater(len(blocked), self.mol.natm)
        owners = [item[0] for item in blocked]
        self.assertEqual(owners, sorted(owners))
        self.assertEqual(set(owners), set(range(self.mol.natm)))
        for owner, expected in enumerate(reference):
            parts = [item for item in blocked if item[0] == owner]
            coords = np.concatenate([item[1] for item in parts], axis=0)
            weights = np.concatenate([item[2] for item in parts])
            responses = np.concatenate([item[3] for item in parts], axis=2)
            np.testing.assert_array_equal(coords, expected[0])
            np.testing.assert_allclose(weights, expected[1], atol=1e-14, rtol=1e-13)
            np.testing.assert_allclose(responses, expected[2], atol=5e-14, rtol=1e-12)
            self.assertTrue(all(len(item[1]) <= adapter.effective_block_size(73) for item in parts))
            self.assertLess(len(parts[-1][1]), adapter.effective_block_size(73))

    def test_native_water_entire_df_gradient_matches_blocked_with_both_responses(self):
        mf = dft.RKS(self.mol).density_fit(auxbasis='def2-universal-jkfit')
        mf.xc = 'PBE'
        mf.grids.level = 5
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.max_memory = 1000
        mf.kernel()
        self.assertTrue(mf.converged)
        driver = mf.nuc_grad_method()
        driver.grid_response = True
        driver.auxbasis_response = True
        native = driver.kernel()
        before = adapter.provenance()
        events = []
        adapter.install(block_size=257, progress=events.append)
        streamed_driver = mf.nuc_grad_method()
        streamed_driver.grid_response = True
        streamed_driver.auxbasis_response = True
        streamed = streamed_driver.kernel()
        np.testing.assert_allclose(streamed, native, atol=2e-10, rtol=1e-8)
        self.assertGreater(float(np.max(np.abs(native))), 1e-3)
        self.assertEqual(adapter.provenance(), before)
        audit = adapter.get_last_audit()
        self.assertTrue(audit['completed'])
        self.assertFalse(audit['active'])
        self.assertEqual(audit['atom_count'], 3)
        self.assertEqual(audit['nao'], self.mol.nao)
        expected_hash = hashlib.sha256(np.ascontiguousarray(self.mol.atom_coords(), dtype='<f8').tobytes()).hexdigest()
        self.assertEqual(audit['geometry_hash'], expected_hash)
        self.assertEqual(audit['owner_atoms_seen'], [0, 1, 2])
        self.assertGreater(audit['blocks_done'], 3)
        self.assertLessEqual(audit['largest_AO_derivative_shape'][1], 257)
        self.assertLessEqual(audit['largest_partition_derivative_shape'][-1], 257)
        self.assertTrue(events)
        self.assertTrue(all('checkpoint_committed' not in event for event in events))
        adapter.uninstall()
        self.assertIs(native_rks.get_vxc_full_response, adapter._NATIVE_FUNCTION)

    def test_real_75_atom_dimensions_have_only_bounded_partition_and_AO_blocks(self):
        root = Path(__file__).resolve().parents[1]
        case = next(case for case in json.loads((root / 'data/research/spinosin_ir_gap_fill_20261112/inputs/cases.json').read_text())['cases']
                    if case['case_id'] == 'C003')
        mol = gto.M(atom=list(zip(case['symbols'], case['initial_geometry_bohr'])),
                    unit='Bohr', basis='def2-SVP', verbose=0)
        self.assertEqual((mol.natm, mol.nao), (75, 762))
        grids = dft.gen_grid.Grids(mol)
        grids.level = 5
        generator = adapter.iter_grid_response_blocks(grids, block_size=16)
        owner, coords, weights, responses = next(generator)
        self.assertEqual(owner, 0)
        self.assertEqual(responses.shape, (75, 3, 16))
        self.assertEqual(coords.shape, (16, 3))
        ao = dft.numint.eval_ao(mol, coords, deriv=2)
        self.assertEqual(ao.shape, (10, 16, 762))
        generator.close()
        bound = adapter.memory_bound(75, 762, 2048)
        self.assertEqual(bound['effective_block_size'], 2016)
        self.assertLess(bound['partition_derivative_bytes_bound'], 300 * 2**20)
        self.assertLess(bound['AO_derivative_bytes_bound'], 130 * 2**20)
        self.assertEqual(bound['partition_derivative_shape_bound'], [75, 75, 3, 2016])
        self.assertEqual(bound['AO_derivative_shape_bound'], [10, 2016, 762])

    def test_unsupported_paths_fail_without_native_full_atom_fallback(self):
        grids = dft.gen_grid.Grids(self.mol)
        grids.level = 0
        ni = dft.numint.NumInt()
        dm = np.eye(self.mol.nao)
        with self.assertRaises(NotImplementedError):
            adapter.get_vxc_full_response(ni, self.mol, grids, 'B3LYP', dm)
        grids.becke_scheme = gen_grid.stratmann
        with self.assertRaises(NotImplementedError):
            next(adapter.iter_grid_response_blocks(grids))
        with self.assertRaises(ValueError):
            adapter.effective_block_size(0)


if __name__ == '__main__':
    unittest.main()
