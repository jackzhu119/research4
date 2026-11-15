"""Recovery operations tests; no molecular SCF or gradient is evaluated."""
from copy import deepcopy
import contextlib
import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_spinosin_ir_memory_repair_20261112 as worker
import manage_spinosin_ir_memory_repair_20261112 as manager

base = worker.base


def sealed(path, payload):
    base.atomic_json(path, {'payload': payload, 'payload_sha256': base.canonical_hash(payload)})


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.leader = self.root / 'reports/spinosin_ir_memory_repair_20261112/runs/C003'
        self.leader.mkdir(parents=True)
        symbols = ['C'] * 28 + ['H'] * 32 + ['O'] * 15
        self.protocol = {'case_id': 'C003', 'case_symbols': {'C003': symbols, 'C008': symbols},
            'science_source_fingerprints': {'C003': 'new-C003', 'C008': 'new-C008'},
            'scientific_file_sha256': {}, 'memory_repair': {'adapter_sha256': 'adapter', 'block_size': 2048}}
        self.protocol['input_hash'] = base.canonical_hash(self.protocol)
        base.atomic_json(self.leader / 'ops/frozen_protocol.json', self.protocol)
        sealed(self.leader / 'SCIENCE_PROTOCOL.json', {'source_fingerprint': 'new-C003'})
        self.point = self.leader / 'generations/g000000_df-grid5/optimization/chunk_00000000/steps/step-00000/result.json'
        self.point.parent.mkdir(parents=True)
        coordinates = np.arange(225, dtype=float).reshape(75, 3) / 100
        values = {'coordinates_bohr': coordinates, 'gradient': np.full((75, 3), .01), 'dipole_au': np.array([1., 2., 3.])}
        refs = {}
        for name, value in values.items():
            path = self.point.parent / (name + '.npy')
            np.save(path, value)
            refs[name] = {'path': str(path), 'sha256': base.sha(path)}
        self.raw = {'status': 'DONE', 'qualified': True, 'native_converged': True,
            'finished_at_utc': '2026-11-12T00:00:00Z', 'symbols': symbols,
            'gradient_response': {'grid_response': True, 'auxbasis_response': True},
            'array_refs': refs, **{name: value.tolist() for name, value in values.items()}}
        base.atomic_json(self.point, self.raw)
        self.audit = {'completed': True, 'atom_count': 75, 'nao': 762,
            'geometry_hash': hashlib.sha256(np.ascontiguousarray(coordinates, dtype='<f8').tobytes()).hexdigest(),
            'requested_block_size': 2048, 'effective_block_size': 2016,
            'owner_atoms_seen': list(range(75)), 'blocks_done': 100,
            'finished_at_utc': '2026-11-12T00:00:00Z'}

    def tearDown(self):
        self.temporary.cleanup()

    def gate(self):
        certificate = worker.make_gradient_certificate(self.leader, self.point, self.protocol, self.audit)
        sealed(self.leader / 'ops/first_qualified_gradient_gate.json', certificate)
        return certificate

    def test_real_gradient_gate_replays_read_only_without_minimum_claim(self):
        certificate = self.gate()
        path = self.leader / 'ops/first_qualified_gradient_gate.json'
        before = base.sha(path)
        self.assertEqual(worker.verify_gradient_certificate(self.leader, self.protocol), certificate)
        self.assertEqual(base.sha(path), before)
        self.assertIn('not a stationary point', certificate['qualification_scope'])

    def test_scf_only_unfinished_or_unqualified_points_cannot_release_follower(self):
        for mutation in ({'status': 'RUNNING'}, {'qualified': False}, {'native_converged': False},
                         {'finished_at_utc': None}, {'gradient_response': {'grid_response': False}}):
            raw = {**self.raw, **mutation}
            base.atomic_json(self.point, raw)
            with self.assertRaises((ValueError, base.IntegrityError)):
                worker.make_gradient_certificate(self.leader, self.point, self.protocol, self.audit)
        base.atomic_json(self.point, self.raw)
        bad = deepcopy(self.raw)
        del bad['array_refs']['gradient']
        base.atomic_json(self.point, bad)
        with self.assertRaises(KeyError):
            worker.make_gradient_certificate(self.leader, self.point, self.protocol, self.audit)

    def test_wrong_audit_geometry_owner_count_or_block_is_rejected(self):
        for mutation in ({'completed': False}, {'atom_count': 3}, {'nao': 24},
                         {'geometry_hash': 'old'}, {'requested_block_size': 4096},
                         {'owner_atoms_seen': [0]}, {'blocks_done': 0}):
            with self.assertRaises(ValueError):
                worker.make_gradient_certificate(self.leader, self.point, self.protocol, {**self.audit, **mutation})

    def test_sealed_raw_arrays_and_protocol_tampering_are_rejected(self):
        self.gate()
        gradient_path = Path(self.raw['array_refs']['gradient']['path'])
        np.save(gradient_path, np.zeros((75, 3)))
        with self.assertRaises(base.IntegrityError):
            worker.verify_gradient_certificate(self.leader, self.protocol)
        np.save(gradient_path, np.full((75, 3), .01))
        self.raw['qualified'] = False
        base.atomic_json(self.point, self.raw)
        with self.assertRaises(base.IntegrityError):
            worker.verify_gradient_certificate(self.leader, self.protocol)

    def test_array_shape_finite_atom_order_and_namespace_are_validated(self):
        for value in (np.zeros((3, 3)), np.full((75, 3), np.nan)):
            path = Path(self.raw['array_refs']['gradient']['path'])
            np.save(path, value)
            bad = deepcopy(self.raw)
            bad['gradient'] = np.zeros(value.shape).tolist()
            bad['array_refs']['gradient']['sha256'] = base.sha(path)
            base.atomic_json(self.point, bad)
            with self.assertRaises(base.IntegrityError):
                worker.make_gradient_certificate(self.leader, self.point, self.protocol, self.audit)
        base.atomic_json(self.point, self.raw)
        swapped = deepcopy(self.raw)
        swapped['symbols'][0], swapped['symbols'][30] = swapped['symbols'][30], swapped['symbols'][0]
        base.atomic_json(self.point, swapped)
        with self.assertRaises((ValueError, base.IntegrityError)):
            worker.make_gradient_certificate(self.leader, self.point, self.protocol, self.audit)
        with self.assertRaises(base.IntegrityError):
            worker.make_gradient_certificate(self.leader, self.root / 'outside.json', self.protocol, self.audit)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        scripts = self.root / 'scripts'
        scripts.mkdir()
        for name in ('run_spinosin_ir_memory_repair_20261112.py', 'manage_spinosin_ir_memory_repair_20261112.py',
                     'manage_spinosin_ir_gap_20261112.py', 'background_supervisor.py', 'bridge.py', 'adapter.py'):
            (scripts / name).write_text('# fixture ' + name)
        self.warm = self.root / 'warm.npy'
        self.warm.write_bytes(b'fixture unqualified warm guess')
        self.cases = {name: {'case_id': name, 'input_hash': name, 'symbols': ['C'] * 28 + ['H'] * 32 + ['O'] * 15}
                      for name in ('C003', 'C008')}
        self.calls = []
        bridge = SimpleNamespace(__file__=str(scripts / 'bridge.py'), DESIGN={'fixture': True},
                                 ScientificIntegrityError=ValueError)
        bridge.execution_dependencies = lambda case, recovery_root: {str(self.warm): base.sha(self.warm)}
        bridge.prepare_fingerprint = lambda case, threads, memory, **kwargs: {'source_fingerprint': base.canonical_hash(
            {'case': case, 'warm': base.sha(self.warm), 'block': kwargs['block_size']})}
        def run(case, output, threads, memory, progress, stop_requested, **kwargs):
            self.calls.append(case['case_id'])
            return {'fixture': True}
        bridge.run_conformer = run
        self.bridge = bridge
        self.adapter = SimpleNamespace(__file__=str(scripts / 'adapter.py'), effective_block_size=lambda size: size // 56 * 56)
        self.args = SimpleNamespace(root=self.root, case_id='C008', threads=2, max_memory_mb=3000,
            case_manifest=self.root / 'data/cases.json', output=self.root / 'reports/spinosin_ir_memory_repair_20261112/runs/C008')
        self.context = contextlib.ExitStack()
        self.context.enter_context(patch.object(worker, '__file__', str(scripts / 'run_spinosin_ir_memory_repair_20261112.py')))
        self.context.enter_context(patch.object(worker, '_ORIGINAL_SNAPSHOT', side_effect=lambda args: ({'registry_id': 'fixture'}, self.cases, {})))
        self.context.enter_context(patch.object(worker, 'modules', return_value=(bridge, self.adapter)))
        self.context.enter_context(patch.object(base, 'runtime_snapshot', return_value={'fixture': True}))

    def tearDown(self):
        self.context.close()
        self.temporary.cleanup()

    def prepare(self, block=2048, readonly=False):
        return worker.prepare(self.args, readonly, block, self.root)

    def test_frozen_warm_adapter_block_mapping_and_readonly_integrity(self):
        protocol, _, _ = self.prepare()
        path = self.args.output / 'ops/frozen_protocol.json'
        before = base.sha(path)
        self.assertEqual(self.prepare(readonly=True)[0], protocol)
        self.assertEqual(base.sha(path), before)
        self.assertIsNone(protocol['parallel_gate']['timeout'])
        self.assertIsNone(protocol['compute_policy']['cpu_stop_limit'])
        with self.assertRaises(base.IntegrityError):
            self.prepare(block=4096)
        self.warm.write_bytes(b'changed guess')
        with self.assertRaises(base.IntegrityError):
            self.prepare()

    def test_follower_wait_has_no_scientific_call_and_manual_stop_is_safe(self):
        protocol, case, proxy = self.prepare()
        events = []
        with self.assertRaises(InterruptedError):
            proxy.run_conformer(case, self.args.output, progress=events.append, stop_requested=lambda: True)
        self.assertEqual(self.calls, [])
        self.assertEqual(events[0]['phase'], 'waiting_for_peer_gradient')
        self.assertFalse(events[0]['active_computation'])

    def test_follower_calls_new_bridge_only_after_valid_gate(self):
        protocol, case, proxy = self.prepare()
        leader = self.args.output.parent / 'C003'
        def progress(event):
            if event.get('phase') == 'waiting_for_peer_gradient':
                base.atomic_json(leader / 'ops/first_qualified_gradient_gate.json', {'fixture': True})
        with patch.object(worker, 'verify_gradient_certificate', return_value={'fixture_verified': True}) as verify:
            result = proxy.run_conformer(case, self.args.output, progress=progress)
        self.assertEqual(result, {'fixture': True})
        self.assertEqual(self.calls, ['C008'])
        verify.assert_called_once_with(leader, protocol)
        self.assertTrue((self.args.output / 'ops/peer_gate_verified.json').is_file())

    def test_new_main_restores_all_old_module_hooks(self):
        before = base.prepare, base.Publisher, base.phase_from
        with patch.object(base, 'main', return_value=0):
            self.assertEqual(worker.main(['--root', str(self.root), '--case', 'C003', '--prepare']), 0)
        self.assertEqual((base.prepare, base.Publisher, base.phase_from), before)


class ManagerTests(unittest.TestCase):
    def test_manager_routes_new_detached_worker_and_records_prior_costs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / manager.RUN_ROOT
            scripts = root / 'scripts'
            scripts.mkdir()
            (scripts / manager.WORKER).write_text('# fixture new worker')
            before = manager.original.WORKER
            def original_main(arguments):
                self.assertEqual(manager.original.WORKER, manager.WORKER)
                self.assertNotIn('--once', arguments)
                self.assertEqual(arguments[-2:], ['--output', str(output)])
                base.atomic_json(output / 'launch_manifest.json', {'launches': [], 'threads_per_lane': 2,
                    'max_memory_mb_per_lane': 3000, 'cpu_wall_quality_retry_round_stop_limit': None})
                return 0
            with patch.object(manager.original, 'main', side_effect=original_main), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(manager.main(['start', '--root', str(root)]), 0)
            receipt = base.read_json(output / 'launch_manifest.json')
            self.assertEqual(receipt['historical_initial_gradient_oom_kills'], 2)
            self.assertIsNone(receipt['parallel_start_gate']['timeout'])
            self.assertEqual(receipt['grid_block_size'], 2048)
            self.assertEqual(set(receipt['prior_run_costs']), {'C003', 'C008'})
            self.assertEqual(manager.original.WORKER, before)
            self.assertEqual(len(list((output / 'launch_receipts').glob('*recovery-*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
