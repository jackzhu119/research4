import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import spinosin_memory_repair_20261112 as bridge


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'repair'
        self.recovery = self.root / 'recovery_inputs'
        registry = bridge.ROOT / 'data/research/spinosin_ir_gap_fill_20261112/inputs/cases.json'
        self.case = json.loads(registry.read_text())['cases'][0]
        self.receipt_dir = self.recovery / self.case['case_id']
        self.receipt_dir.mkdir(parents=True)
        self.snapshot = self.receipt_dir / 'snapshot.chk'
        self.snapshot.write_bytes(b'fixture snapshot; HDF parsing is mocked in workflow tests')
        self.density = self.receipt_dir / 'warm_density.npy'
        np.save(self.density, np.eye(2))
        self.receipt = {'case_id': self.case['case_id'], 'case_declared_input_hash': self.case['input_hash'],
                        'scientifically_qualified': False, 'gradient_available': False,
                        'stationary_point_verified': False, 'snapshot_path': str(self.snapshot),
                        'snapshot_sha256': bridge.original._file_hash(self.snapshot),
                        'warm_density_path': str(self.density),
                        'warm_density_sha256': bridge.original._file_hash(self.density),
                        'old_source_fingerprint': 'a' * 64, 'source_sha256_at_snapshot': 'b' * 64,
                        'snapshot_at_utc': '2026-11-12T13:19:38Z'}
        self.receipt_path = self.receipt_dir / 'recovery_input.json'
        self.receipt_path.write_text(json.dumps(self.receipt))
        adapter_file = Path(self.temporary.name) / 'adapter.py'
        adapter_file.write_text('# independently hashed adapter fixture\n')
        self.installs = []
        def install(block_size=2048, progress=None):
            self.installs.append(block_size)
            if progress:
                progress({'phase': 'grid_response_block', 'checkpoint_committed': True})
        self.adapter = SimpleNamespace(__file__=str(adapter_file),
                                       provenance=lambda: {'native_source_hash': 'fixed', 'formula_unchanged': True},
                                       install=install)
        self.validation = {'atom_count': 75, 'ao_count': 762, 'overlap_weighted_electrons': 320.0,
                           'seed_is_scientifically_qualified': False, 'nuclear_gradient_imported': False}
        self.addCleanup(patch.stopall)
        patch.object(bridge, 'REPAIR_ROOT', self.root).start()
        patch.object(bridge, '_adapter', return_value=self.adapter).start()
        patch.object(bridge, '_validate_warm_density', return_value=self.validation).start()

    def prepare(self, **kwargs):
        return bridge.prepare_fingerprint(self.case, recovery_root=self.recovery, **kwargs)

    def execute_bridge(self, result, progress=None, **kwargs):
        return bridge.run_conformer(self.case, result, recovery_root=self.recovery, progress=progress, **kwargs)

    def test_new_fingerprint_binds_adapter_block_and_unqualified_density(self):
        before = bridge.original.prepare_fingerprint(self.case)
        repaired = self.prepare()
        self.assertNotEqual(before['source_fingerprint'], repaired['source_fingerprint'])
        execution = repaired['payload']['case']['memory_repair_execution']
        self.assertEqual(execution['warm_density']['sha256'], self.receipt['warm_density_sha256'])
        self.assertFalse(execution['completed_gradient_reused'])
        self.assertFalse(execution['stationary_point_or_minimum_imported'])
        changed = self.prepare(block_size=512)
        self.assertNotEqual(changed['source_fingerprint'], repaired['source_fingerprint'])

    def test_run_initializes_only_warm_seed_then_calls_original_directly(self):
        output = self.root / 'runs' / self.case['case_id']
        seen = {}
        events = []
        def original_run(case, out, **kwargs):
            seen.update(case=case, state=bridge.original._read_record(Path(out) / 'workflow_state.json'))
            return {'status': 'TEST_ENTERED_ORIGINAL_WORKFLOW'}
        original_identity = importlib.import_module('spinosin_gap_science_20261112')
        with patch.object(bridge.original, 'run_conformer', original_run):
            result = self.execute_bridge(output, events.append)
        self.assertEqual(result['status'], 'TEST_ENTERED_ORIGINAL_WORKFLOW')
        state = seen['state']
        self.assertEqual(state['phase'], 'OPTIMIZE')
        self.assertEqual(state['chunk_index'], 0)
        self.assertEqual(state['density_ref']['path'], str(self.density))
        self.assertFalse(state['history'][0]['gradient_qualification_imported'])
        np.testing.assert_array_equal(state['coordinates_bohr'], self.case['initial_geometry_bohr'])
        self.assertFalse((output / 'generations').exists())
        self.assertFalse((output / 'COMPLETE.json').exists())
        self.assertTrue(any(event.get('stage') == 'repaired_warm_start_state_committed' for event in events))
        block = next(event for event in events if event.get('phase') == 'grid_response_block')
        self.assertFalse(block['checkpoint_committed'])
        self.assertIs(importlib.import_module('spinosin_gap_science_20261112'), original_identity)

    def test_restart_never_resets_progress_to_initial_warm_seed(self):
        output = self.root / 'runs' / self.case['case_id']
        calls = []
        def original_run(case, out, **kwargs):
            state = bridge.original._read_record(Path(out) / 'workflow_state.json')
            calls.append(copy.deepcopy(state))
            state.update(chunk_index=7, phase='CERTIFY_CENTER')
            bridge.original._record(Path(out) / 'workflow_state.json', state)
            return {'status': 'CONTROL_RETURN'}
        with patch.object(bridge.original, 'run_conformer', original_run):
            self.execute_bridge(output)
            self.execute_bridge(output)
        self.assertEqual(calls[0]['chunk_index'], 0)
        self.assertEqual(calls[1]['chunk_index'], 7)
        self.assertEqual(calls[1]['phase'], 'CERTIFY_CENTER')

    def test_protocol_commit_before_state_commit_can_resume_with_warm_seed(self):
        output = self.root / 'runs' / self.case['case_id']
        original_record = bridge.original._record
        def interrupted_record(path, payload):
            if Path(path).name == 'workflow_state.json':
                raise InterruptedError('simulate stop between protocol and initial state')
            return original_record(path, payload)
        with patch.object(bridge.original, '_record', interrupted_record):
            with self.assertRaises(InterruptedError):
                self.execute_bridge(output)
        self.assertTrue((output / 'SCIENCE_PROTOCOL.json').exists())
        self.assertFalse((output / 'workflow_state.json').exists())
        with patch.object(bridge.original, 'run_conformer', return_value={'status': 'RESUMED'}):
            self.execute_bridge(output)
        self.assertEqual(bridge.original._read_record(output / 'workflow_state.json')['density_ref']['sha256'], self.receipt['warm_density_sha256'])

    def test_modified_seed_or_qualified_old_result_is_rejected(self):
        np.save(self.density, np.eye(2) * 2)
        with self.assertRaises(bridge.ScientificIntegrityError):
            self.prepare()
        np.save(self.density, np.eye(2))
        receipt = copy.deepcopy(self.receipt)
        receipt['gradient_available'] = True
        self.receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(bridge.ScientificIntegrityError, 'unqualified'):
            self.prepare()

    def test_old_namespace_and_unsealed_old_artifacts_cannot_be_imported(self):
        with self.assertRaisesRegex(bridge.ScientificIntegrityError, 'new repair namespace'):
            self.execute_bridge(Path(self.temporary.name) / 'old_run')
        output = self.root / 'runs' / self.case['case_id']
        (output / 'generations').mkdir(parents=True)
        with self.assertRaisesRegex(bridge.ScientificIntegrityError, 'old scientific artifacts'):
            self.execute_bridge(output)

    def test_execution_dependencies_are_real_file_hashes_and_prepare_is_stable(self):
        dependencies = bridge.execution_dependencies(self.case, self.recovery)
        self.assertEqual(dependencies[str(self.density)], self.receipt['warm_density_sha256'])
        self.assertIn(str(self.receipt_path), dependencies)
        first = self.prepare()
        self.adapter.install()
        self.assertEqual(first, self.prepare())


class ActualWarmValidationTests(unittest.TestCase):
    def test_finite_orbitals_geometry_and_electron_count_are_rechecked(self):
        registry = bridge.ROOT / 'data/research/spinosin_ir_gap_fill_20261112/inputs/cases.json'
        case = json.loads(registry.read_text())['cases'][0]
        occupation = np.r_[np.full(160, 2.0), np.zeros(602)]
        coefficient = np.eye(762)
        mol = SimpleNamespace(natm=75, charge=0, spin=0, nelectron=320, basis='def2-SVP',
                              atom_symbol=lambda index: case['symbols'][index],
                              atom_coords=lambda: np.asarray(case['initial_geometry_bohr']),
                              nao_nr=lambda: 762, intor_symmetric=lambda name: np.eye(762))
        orbital = {'mo_coeff': coefficient, 'mo_occ': occupation, 'mo_energy': np.linspace(-1, 1, 762)}
        thread = [2]
        def num_threads(value=None):
            if value is not None:
                thread[0] = value
            return thread[0]
        pyscf = SimpleNamespace(lib=SimpleNamespace(num_threads=num_threads))
        scf = SimpleNamespace(chkfile=SimpleNamespace(load_scf=lambda name: (mol, orbital)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'density.npy'
            np.save(path, np.diag(occupation))
            refs = {'snapshot': {'path': 'fixture.chk'}, 'density': {'path': str(path)}}
            with patch.dict(sys.modules, {'pyscf': pyscf, 'pyscf.scf': scf}):
                evidence = bridge._validate_warm_density(case, {}, refs)
                self.assertEqual(evidence['overlap_weighted_electrons'], 320)
                self.assertFalse(evidence['nuclear_gradient_imported'])
                self.assertEqual(thread[0], 2)
                changed = copy.deepcopy(case)
                changed['initial_geometry_bohr'][0][0] += 0.01
                with self.assertRaisesRegex(bridge.ScientificIntegrityError, 'geometry'):
                    bridge._validate_warm_density(changed, {}, refs)


if __name__ == '__main__':
    unittest.main()
