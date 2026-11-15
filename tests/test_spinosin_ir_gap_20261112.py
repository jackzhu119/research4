"""Operations contracts; fixtures never perform formal molecular calculations."""
import contextlib
import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import run_spinosin_ir_gap_20261112 as worker
import manage_spinosin_ir_gap_20261112 as manager


class FakeScience:
    DESIGN = {'fixture_only': True, 'primary_displacements': 900}
    class ScientificIntegrityError(ValueError):
        pass

    def __init__(self):
        self.point_fits = {}
        self.calls = 0
        self.waiting = False
        self.fail_integrity = False

    def prepare_fingerprint(self, case, threads=2, max_memory_mb=3000):
        return {'source_fingerprint': worker.canonical_hash({'case': case['input_hash'],
            'threads': threads, 'memory': max_memory_mb})}

    def run_conformer(self, case, output, threads=2, max_memory_mb=3000, progress=None, stop_requested=None):
        self.calls += 1
        if self.fail_integrity:
            raise self.ScientificIntegrityError('Sealed scientific point changed')
        fingerprint = self.prepare_fingerprint(case, threads, max_memory_mb)['source_fingerprint']
        if self.waiting:
            return {'status': 'REFINEMENT_REQUIRED', 'case_id': case['case_id'], 'source_fingerprint': fingerprint,
                'computed': True, 'minimum_verified': False, 'ir_verified': False,
                'paper_ready_molecular_harmonic_data': False,
                'waiting_for_new_protocol': True, 'continuation_eligible': False}
        for number, phase in enumerate(('geometry_opt', 'full_hessian', 'ir')):
            path = output / 'fixture_points' / f'{number}.json'
            if path.exists():
                old = worker.read_json(path)
                assert old['payload_sha256'] == worker.canonical_hash(old['payload'])
                assert old['payload']['source_fingerprint'] == fingerprint
            else:
                self.point_fits[number] = self.point_fits.get(number, 0) + 1
                payload = {'case_id': case['case_id'], 'source_fingerprint': fingerprint, 'number': number}
                worker.atomic_json(path, {'payload': payload, 'payload_sha256': worker.canonical_hash(payload)})
            if progress:
                progress({'phase': phase, 'checkpoint_committed': True, 'point': number})
            if stop_requested and stop_requested():
                raise InterruptedError('Manual stop at a fixture committed unit')
        result = {'case_id': case['case_id'], 'source_fingerprint': fingerprint, 'status': 'COMPLETE',
            'computed': True, 'minimum_verified': True, 'ir_verified': True,
            'paper_ready_molecular_harmonic_data': True, 'continuation_eligible': False}
        worker.atomic_json(output / 'result.json', result)
        (output / 'native.log').write_text('Retained fixture raw log; not a scientific measurement\n')
        return result


class Publisher:
    def __init__(self, interrupt=False, stop_after_waits=None):
        self.fields, self.events = {}, []
        self.interrupt, self.stop_after_waits, self.waits = interrupt, stop_after_waits, 0

    def update(self, **fields):
        self.fields.update(fields)
        if fields.get('phase') == 'waiting_refinement':
            self.waits += 1
            if self.stop_after_waits and self.waits >= self.stop_after_waits:
                worker.STOP_REQUESTED = True

    def progress(self, event):
        self.events.append(event)
        if self.interrupt:
            self.interrupt = False
            raise InterruptedError('Simulated termination after checkpoint')


def fixture_root(root):
    inputs = root / 'data/research/spinosin_ir_gap_fill_20261112/inputs'
    inputs.mkdir(parents=True)
    symbols = ['C'] * 28 + ['H'] * 32 + ['O'] * 15
    cases = []
    for index, name in enumerate(worker.CASES):
        coords = [[float(i) / 10 + index, .2, .3] for i in range(75)]
        xyz, sdf, array = inputs / (name + '.xyz'), inputs / (name + '.sdf'), inputs / (name + '.npy')
        xyz.write_text(name + ' fixture source bytes\n')
        sdf.write_text(name + ' fixture source bytes\n')
        np.save(array, np.asarray(coords))
        payload = {'case_id': name, 'formula': 'C28H32O15', 'charge': 0, 'spin': 0,
            'symbols': symbols, 'initial_geometry_bohr': coords, 'bohr_in_angstrom': .529177210903,
            'source_xyz_sha256': worker.sha(xyz), 'source_sdf_sha256': worker.sha(sdf),
            'stereo_inchikey': 'fixture-only'}
        cases.append({**payload, 'input_hash': worker.canonical_hash(payload), 'input_fingerprint_payload': payload,
            'source_xyz_path': str(xyz.relative_to(root)), 'source_sdf_path': str(sdf.relative_to(root)),
            'initial_geometry_bohr_path': str(array.relative_to(root)),
            'initial_geometry_bohr_npy_sha256': worker.sha(array), 'initial_density_available': False,
            'historical_C001_density_or_hessian_reuse_allowed': False})
    manifest = inputs / 'cases.json'
    worker.atomic_json(manifest, {'registry_id': 'fixture-only', 'cases': cases})
    scripts = root / 'scripts'
    scripts.mkdir()
    for name in ('run_spinosin_ir_gap_20261112.py', 'spinosin_gap_science_20261112.py', 'spinosin_gap_ir_20261112.py'):
        (scripts / name).write_text('# fixture source bytes\n')
    vendor = scripts / 'quantum_gap_vendor_20261112'
    vendor.mkdir()
    for name in ['__init__.py'] + [f'core{i}.py' for i in range(6)]:
        (vendor / name).write_text('# fixture vendor source\n')
    worker.atomic_json(inputs.parent / 'vendor_provenance.json', {'fixture_only': True})
    return manifest


def arguments(root, manifest):
    return SimpleNamespace(root=root, case_id='C003', case_manifest=manifest,
        output=root / 'reports/test/runs/C003', threads=2, max_memory_mb=3000,
        cpu_affinity=None, heartbeat_seconds=.02, poll_seconds=.001, once=True)


class SpinosinOperationsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = fixture_root(self.root)
        self.args = arguments(self.root, self.manifest)
        self.science = FakeScience()
        worker.STOP_REQUESTED = False
        self.runtime = {'python_version': 'fixture', 'libraries': {'pyscf': 'fixture'}}
        with patch.object(worker, 'runtime_snapshot', return_value=self.runtime), \
             patch.object(worker, 'load_science', return_value=self.science):
            self.protocol, self.case, _ = worker.prepare(self.args)

    def tearDown(self):
        worker.STOP_REQUESTED = False
        self.temporary.cleanup()

    def test_two_authenticated_inputs_and_code_are_frozen(self):
        self.assertEqual(set(self.protocol['case_input_hashes']), {'C003', 'C008'})
        self.assertIsNone(self.protocol['compute_policy']['cpu_stop_limit'])
        registry, cases = worker.validate_cases(self.root, self.manifest)
        registry['cases'][0]['initial_geometry_bohr'][0][0] += 1
        worker.atomic_json(self.manifest, registry)
        with self.assertRaises(worker.IntegrityError):
            worker.validate_cases(self.root, self.manifest)
        with self.assertRaises(worker.IntegrityError):
            worker.check_frozen_files(self.args, self.protocol)

    def test_interrupted_point_restarts_without_recalculation(self):
        worker.run_lane(self.args, self.protocol, self.case, self.science, Publisher(interrupt=True))
        self.assertEqual(self.science.point_fits, {0: 1})
        self.assertFalse((self.args.output / 'ops/workflow_complete.json').exists())
        worker.run_lane(self.args, self.protocol, self.case, self.science, Publisher())
        self.assertEqual(self.science.point_fits, {0: 1, 1: 1, 2: 1})
        receipt = worker.verify_completion(self.args, self.protocol)
        self.assertTrue(receipt['minimum_verified'])
        self.assertEqual(worker.verify_committed_record_hashes(self.args.output), 3)

    def test_complete_replay_is_immutable_and_does_not_call_science(self):
        worker.run_lane(self.args, self.protocol, self.case, self.science, Publisher())
        receipt = self.args.output / 'ops/workflow_complete.json'
        before = (worker.sha(receipt), receipt.stat().st_mtime_ns)
        with patch.object(self.science, 'run_conformer', side_effect=AssertionError('must not recalculate')):
            publisher = Publisher()
            worker.run_lane(self.args, self.protocol, self.case, self.science, publisher)
        self.assertEqual(publisher.fields['phase'], 'qualified_complete_idle')
        self.assertEqual(before, (worker.sha(receipt), receipt.stat().st_mtime_ns))
        (self.args.output / 'native.log').write_text('corrupt retained raw log')
        with self.assertRaisesRegex(worker.IntegrityError, 'artifact'):
            worker.verify_completion(self.args, self.protocol)

    def test_no_cross_conformer_or_false_string_qualification(self):
        result = {'case_id': 'C008', 'source_fingerprint': self.protocol['science_source_fingerprints']['C008'],
            'status': 'COMPLETE', 'computed': True, 'minimum_verified': True, 'ir_verified': True,
            'paper_ready_molecular_harmonic_data': True}
        with self.assertRaisesRegex(worker.IntegrityError, 'unqualified'):
            worker.seal_completion(self.args, self.protocol, result)
        result.update(case_id='C003', source_fingerprint=self.protocol['science_source_fingerprints']['C003'],
                      minimum_verified='false')
        with self.assertRaises(worker.IntegrityError):
            worker.seal_completion(self.args, self.protocol, result)

    def test_unqualified_waiting_remains_live_without_repeating_cached_work(self):
        self.science.waiting = True
        self.args.once = False
        publisher = Publisher(stop_after_waits=2)
        worker.run_lane(self.args, self.protocol, self.case, self.science, publisher)
        self.assertEqual(self.science.calls, 1)
        self.assertGreaterEqual(publisher.waits, 2)
        self.assertFalse((self.args.output / 'ops/workflow_complete.json').exists())
        self.assertFalse(publisher.fields['minimum_verified'])

    def test_scientific_integrity_is_not_recoverable_solver_failure(self):
        self.science.fail_integrity = True
        with self.assertRaises(worker.IntegrityError):
            worker.run_lane(self.args, self.protocol, self.case, self.science, Publisher())
        self.assertFalse((self.args.output / 'ops/latest_recoverable_failure.json').exists())

    def test_readonly_verify_with_training_lock_held(self):
        worker.run_lane(self.args, self.protocol, self.case, self.science, Publisher())
        held = (self.args.output / 'ops/worker.lock').open('a+')
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = {str(path.relative_to(self.root)): (worker.sha(path), path.stat().st_mtime_ns)
                  for path in self.root.rglob('*') if path.is_file()}
        with patch.object(worker, 'runtime_snapshot', return_value=self.runtime), \
             patch.object(worker, 'load_science', return_value=self.science), \
             patch.object(worker, 'configure_resources'):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(worker.main(['--root', str(self.root), '--case', 'C003',
                    '--output', str(self.args.output), '--verify']), 0)
        after = {str(path.relative_to(self.root)): (worker.sha(path), path.stat().st_mtime_ns)
                 for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(before, after)
        held.close()

    def test_independent_heartbeat_does_not_invent_numerical_progress(self):
        publisher = worker.Publisher(self.args, self.protocol, .02)
        try:
            publisher.progress({'phase': 'geometry_opt', 'checkpoint_committed': False, 'iteration': 0})
            first = worker.read_json(self.args.output / 'execution_state.json')
            time.sleep(.06)
            second = worker.read_json(self.args.output / 'execution_state.json')
            self.assertNotEqual(first['heartbeat_at_utc'], second['heartbeat_at_utc'])
            self.assertEqual(first['last_progress_at_utc'], second['last_progress_at_utc'])
            self.assertEqual(second['phase'], 'geometry_opt')
            self.assertGreater(second['measured_cpu_seconds'], 0)
        finally:
            publisher.close()

    def test_manager_detaches_two_lanes_and_preserves_manual_stop(self):
        backend = self.root / '.venv-quantum-gap-20261112/bin/python'
        backend.parent.mkdir(parents=True)
        backend.write_text('fixture backend executable')
        output = self.root / 'launch-test'
        def prepare(command, **kwargs):
            case = command[command.index('--case') + 1]
            out = Path(command[command.index('--output') + 1])
            worker.atomic_json(out / 'ops/frozen_protocol.json', {'input_hash': 'fixture-' + case})
            return subprocess.CompletedProcess(command, 0, '{}', '')
        with patch.object(manager, 'matches', return_value=False), \
             patch.object(manager, 'affinities', return_value={'C003': [0, 1], 'C008': [2, 3]}), \
             patch.object(manager, 'process_identity', return_value={'pid': 123456, 'start_ticks': 1}), \
             patch.object(manager.subprocess, 'run', side_effect=prepare) as preflight, \
             patch.object(manager.subprocess, 'Popen', return_value=SimpleNamespace(pid=123456)) as launch:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(manager.main(['start', '--root', str(self.root), '--output', str(output)]), 0)
        self.assertEqual((preflight.call_count, launch.call_count), (2, 2))
        for call in launch.call_args_list:
            self.assertTrue(call.kwargs['start_new_session'])
            self.assertIn('-u', call.args[0])
            self.assertNotIn('--once', call.args[0])
            self.assertEqual(call.kwargs['env']['OMP_NUM_THREADS'], '2')
            self.assertIn('--cpu-affinity', call.args[0])
        self.assertEqual(len(list((output / 'launch_receipts').glob('*.json'))), 1)
        (output / 'supervisors/C003/STOP').write_text('manual fixture stop')
        with self.assertRaisesRegex(RuntimeError, 'STOP'):
            manager.main(['start', '--root', str(self.root), '--output', str(output), '--cases', 'C003'])

    def test_real_authenticated_registry_validates_without_quantum_calculation(self):
        manifest = ROOT / 'data/research/spinosin_ir_gap_fill_20261112/inputs/cases.json'
        registry, cases = worker.validate_cases(ROOT, manifest)
        self.assertEqual(set(cases), {'C003', 'C008'})
        self.assertEqual(len(cases['C003']['symbols']), 75)
        self.assertNotEqual(cases['C003']['input_hash'], cases['C008']['input_hash'])


if __name__ == '__main__':
    unittest.main()
