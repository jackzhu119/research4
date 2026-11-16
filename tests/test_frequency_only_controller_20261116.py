"""Real detached controller routing, inherited accounting and scope completion."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/background_frequency_only_20261116.py'
spec = importlib.util.spec_from_file_location('matrix_controller', SOURCE)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
spec = importlib.util.spec_from_file_location('matrix_extension_fixture', ROOT / 'scripts/extend_all_round_validation_20261112.py')
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


class FrequencyControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'work'
        self.root.mkdir()
        for relative in (controller.SELF, extension.SELF):
            (self.root / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, self.root / relative)
        runner = self.root / controller.RUNNER
        runner.write_text('import argparse, fcntl, hashlib, json, os, pathlib\n'
            'p=argparse.ArgumentParser();p.add_argument("--root");p.add_argument("--worker-id");p.add_argument("--once",action="store_true");a=p.parse_args()\n'
            'o=pathlib.Path(a.root)/"reports/all_round_validation_20261112/frequency_only";o.mkdir(parents=True,exist_ok=True)\n'
            'with (o/"fixture.lock").open("a+") as lock:\n'
            ' fcntl.flock(lock,fcntl.LOCK_EX)\n'
            ' (o/("worker"+a.worker_id+".json")).write_text(json.dumps({"pid":os.getpid(),"sid":os.getsid(0),"threads":os.environ.get("OMP_NUM_THREADS")}))\n'
            ' if not (o/"COMPLETE.json").exists():\n'
            '  m=o/"frequency_matrix.npy";m.write_bytes(b"fixture matrix")\n'
            '  r=o/"summary.json";r.write_text(json.dumps({"status":"COMPLETE","qualified_points":396,"planned_points":396,"frequency_minimum_verified":False}))\n'
            '  (o/"COMPLETE.json").write_text(json.dumps({"summary_sha256":hashlib.sha256(r.read_bytes()).hexdigest(),"artifacts_sha256":{"frequency_matrix.npy":hashlib.sha256(m.read_bytes()).hexdigest()}}))\n')
        old_frozen = self.root / extension.RUNTIME / 'frozen'
        old_script = old_frozen / extension.OLD_SELF
        old_script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / extension.RUNTIME / 'frozen' / extension.OLD_SELF, old_script)
        for path in (self.root / extension.OLD_AUTHORIZATION, old_frozen / extension.OLD_AUTHORIZATION):
            extension.write(path, {'authorized_cpu_seconds': 990000})
        marker = str(controller.PARENT / 'ml/adoption.json')
        extension.write(self.root / marker, {'status': 'COMPLETE'})
        cfg = {'tasks': [{'id': 'ml_verification', 'command': ['must-never-run'],
                         'completion': {'path': marker, 'field': 'status', 'equals': 'COMPLETE'}},
                        {'id': 'quantum_publication', 'command': ['must-never-run-old-queue'],
                         'completion': {'path': 'old.json', 'field': 'status', 'equals': 'COMPLETE'}}],
               'publish_paths': [str(controller.PARENT)]}
        cfgpath = Path('data/research/all_round_validation_20261112/tasks.json')
        for path in (self.root / cfgpath, old_frozen / cfgpath):
            extension.write(path, cfg)
        files = {str(p.relative_to(old_frozen)): extension.sha(p) for p in old_frozen.rglob('*') if p.is_file()}
        extension.write(self.root / controller.PARENT / 'source_manifest.json', {'files': files})
        extension.write(self.root / extension.RUNTIME / 'state.json', {'status': 'RUNNING',
                        'branch': 'research/frequency-test', 'inherited_completed_tasks': [],
                        'tasks': {'ml_verification': {'status': 'COMPLETE'}}})
        extension.prepare(self.root, 1900000, 'fixture-addition')
        extension.write(self.root / controller.SOURCE_SCOPE, {
            'user_instruction': 'Only the current frequency matrix; cancel IR and queued molecules',
            'cancelled_tasks': ['quantum_ir', 'spinosin', 'all_other_queued_jobs']})
        extension.write(self.root / extension.RUNTIME / 'STOP.json', {'reason': 'scope_superseded'})
        self.git('init', '-b', 'research/frequency-test')
        self.git('config', 'user.name', 'Fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        remote = Path(self.temporary.name) / 'remote.git'
        subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
        self.git('remote', 'add', 'origin', str(remote))
        self.git('add', '.')
        self.git('commit', '-m', 'fixture')
        state = extension.read(self.root / extension.RUNTIME / 'state.json')
        state['initial_head'] = self.git('rev-parse', 'HEAD')
        extension.write(self.root / extension.RUNTIME / 'state.json', state)
        self.base = controller.load_base(self.root)

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, stderr=subprocess.DEVNULL, text=True).strip()

    def test_freezes_only_new_scope_and_preserves_authorization_and_stop(self):
        original_cpu = extension.read(self.root / controller.PARENT / 'cpu_accounting.json')
        original_manifest = extension.sha(self.root / controller.PARENT / 'source_manifest.json')
        manifest = controller.prepare(self.root)
        self.assertEqual(set(manifest['files']), {str(controller.SELF), str(controller.RUNNER), str(controller.POLICY)})
        self.assertEqual(original_cpu, extension.read(self.root / controller.PARENT / 'cpu_accounting.json'))
        self.assertEqual(original_manifest, extension.sha(self.root / controller.PARENT / 'source_manifest.json'))
        self.assertTrue((self.root / extension.RUNTIME / 'STOP.json').exists())
        self.assertEqual([x['threads'] for x in controller.policy(self.root)['workers']], [2, 2])
        controller.check_integrity(self.root, self.base)
        (controller.frozen(self.root) / controller.RUNNER).write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'Frozen frequency-only'):
            controller.check_integrity(self.root, self.base)

    def test_completion_requires_all_points_and_artifact_seals(self):
        controller.prepare(self.root)
        out = self.root / controller.OUTPUT
        extension.write(out / 'summary.json', {'status': 'COMPLETE', 'qualified_points': 395, 'planned_points': 396})
        self.assertFalse(controller.completion(self.root))
        extension.write(out / 'summary.json', {'status': 'COMPLETE', 'qualified_points': 396, 'planned_points': 396})
        self.assertFalse(controller.completion(self.root))
        (out / 'matrix.npy').write_bytes(b'matrix')
        extension.write(out / 'COMPLETE.json', {'summary_sha256': extension.sha(out / 'summary.json'),
                        'artifacts_sha256': {'matrix.npy': extension.sha(out / 'matrix.npy')}})
        self.assertTrue(controller.completion(self.root))
        (out / 'matrix.npy').write_bytes(b'corrupt')
        self.assertFalse(controller.completion(self.root))

    def test_guard_halts_publisher_as_well_as_workers(self):
        recorded = []
        class Base:
            def signal_session(self, identity): recorded.append(identity)
        controller.halt_workers(Base(), {'tasks': {'a': {'identity': 'worker'}}, 'publisher_identity': 'publisher'})
        self.assertEqual(recorded, ['worker', 'publisher'])

    def test_actual_detached_frequency_only_completion_and_publication(self):
        controller.prepare(self.root)
        head = self.git('rev-parse', 'HEAD')
        index = self.git('diff', '--cached', '--binary')
        try:
            launched = controller.start(self.root, self.base)
            self.assertTrue(launched['supervisor_alive'])
            self.assertTrue(launched['guard_alive'])
            self.assertEqual(launched['state']['supervisor_identity']['pid'], launched['state']['supervisor_identity']['sid'])
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                state = extension.read(self.root / controller.RUNTIME / 'state.json', {})
                if state.get('status') == 'COMPLETE': break
                time.sleep(.05)
            self.assertEqual(state.get('status'), 'COMPLETE', state)
            self.assertEqual(set(state['tasks']), {'frequency_matrix_0', 'frequency_matrix_1'})
            self.assertEqual(state['queued_tasks'], [])
            self.assertEqual(state['authorized_cpu_seconds'], 2890000)
            self.assertEqual(self.git('rev-parse', 'HEAD'), head)
            self.assertEqual(self.git('diff', '--cached', '--binary'), index)
            self.assertTrue((self.root / extension.RUNTIME / 'STOP.json').exists())
            self.assertTrue(controller.status(self.root, self.base, verify=True)['verified'])
            for path in (self.root / controller.OUTPUT).glob('worker*.json'):
                value = extension.read(path)
                self.assertEqual(value['pid'], value['sid'])
                self.assertEqual(value['threads'], '2')
        finally:
            state = extension.read(self.root / controller.RUNTIME / 'state.json', {})
            extension.write(self.root / controller.RUNTIME / 'STOP.json', {'reason': 'Fixture cleanup'})
            controller.halt_workers(self.base, state)
            self.base.signal_session(state.get('supervisor_identity'))
            self.base.signal_session(state.get('guard_identity'))


if __name__ == '__main__':
    unittest.main()
