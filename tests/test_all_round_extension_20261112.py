"""Additional grants retain original sources, usage, task history and control routing."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / 'scripts/extend_all_round_validation_20261112.py'
LEGACY = REPOSITORY / 'reports/all_round_validation_20261112/.background/frozen/scripts/background_all_round_validation_20261112.py'
spec = importlib.util.spec_from_file_location('extension', SOURCE)
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / extension.SELF
        source.parent.mkdir(parents=True)
        shutil.copyfile(SOURCE, source)
        frozen = self.root / extension.RUNTIME / 'frozen'
        legacy = frozen / extension.OLD_SELF
        legacy.parent.mkdir(parents=True)
        shutil.copyfile(LEGACY, legacy)
        authorization = {'authorized_cpu_seconds': 990000}
        for path in (self.root / extension.OLD_AUTHORIZATION, frozen / extension.OLD_AUTHORIZATION):
            extension.write(path, authorization)
        self.marker = str(extension.OUTPUT / 'ml/adoption.json')
        extension.write(self.root / self.marker, {'status': 'COMPLETE'})
        config = {'tasks': [
            {'id': 'ml_verification', 'command': ['must-never-execute-withdrawn-paper-code'],
             'completion': {'path': self.marker, 'field': 'status', 'equals': 'COMPLETE'}},
            {'id': 'quantum_publication', 'command': ['retained-scientific-worker'],
             'completion': {'path': 'publication_result.json', 'field': 'status', 'equals': 'COMPLETE'}},
            {'id': 'quantum_ir', 'command': ['retained-ir-worker'],
             'completion': {'path': 'ir_result.json', 'field': 'status', 'equals': 'COMPLETE'}}],
            'publish_paths': [str(extension.OUTPUT)]}
        config_path = Path('data/research/all_round_validation_20261112/tasks.json')
        extension.write(frozen / config_path, config)
        extension.write(self.root / config_path, config)
        files = {str(path.relative_to(frozen)): extension.sha(path)
                 for path in frozen.rglob('*') if path.is_file()}
        extension.write(self.root / extension.OUTPUT / 'source_manifest.json', {'files': files})
        self.original_files = files
        self.original_manifest_sha = extension.sha(self.root / extension.OUTPUT / 'source_manifest.json')
        base = extension.original(self.root)
        now = datetime.now(timezone.utc) - timedelta(seconds=1)
        old_sample = {'counter_cpu_seconds': 252378.76458, 'boot_id': 'old-destroyed-environment',
                      'cgroup_device': 24, 'cgroup_inode': 161, 'cpu_capacity': 4.0,
                      'sampled_at_utc': now.isoformat(), 'sampled_monotonic': 12345.0}
        self.old_cpu = base.account_sample({}, old_sample)
        extension.write(self.root / extension.OUTPUT / 'cpu_accounting.json', self.old_cpu)
        extension.write(self.root / extension.RUNTIME / 'state.json', {
            'status': 'RUNNING', 'authorized_cpu_seconds': 990000,
            'tasks': {'ml_verification': {'status': 'COMPLETE', 'attempts': 1}}})

    def prepare(self, amount=100000, identity=extension.DEFAULT_GRANT_ID):
        return extension.prepare(self.root, amount, identity)

    def test_grant_is_idempotent_and_original_files_and_cpu_are_unchanged(self):
        first = self.prepare()
        second = self.prepare()
        self.assertEqual(first['authorization']['cumulative_authorized_cpu_seconds'], 1090000)
        self.assertTrue(second['already_recorded'])
        self.assertEqual(len(second['authorization']['grants']), 1)
        frozen = self.root / extension.RUNTIME / 'frozen'
        self.assertEqual(self.original_files, {str(path.relative_to(frozen)): extension.sha(path)
                         for path in frozen.rglob('*') if path.is_file() and '__pycache__' not in path.parts})
        self.assertEqual(self.original_manifest_sha, extension.sha(self.root / extension.OUTPUT / 'source_manifest.json'))
        self.assertEqual(extension.read(self.root / extension.OUTPUT / 'cpu_accounting.json'), self.old_cpu)
        self.assertEqual(extension.read(self.root / extension.OLD_AUTHORIZATION)['authorized_cpu_seconds'], 990000)
        with self.assertRaisesRegex(ValueError, 'different amount'):
            self.prepare(200000)

    def test_all_roles_use_wrapper_and_future_grant_refreshes_live_runtime(self):
        self.prepare()
        base = extension.load_runtime(self.root)
        self.assertEqual(base.AUTHORIZED_CPU_SECONDS, 1090000)
        for role in ('--supervise', '--guard', '--publish'):
            command = base.own_command(self.root, role)
            self.assertEqual(command[1], str(self.root / extension.WRAPPER_COPY))
            self.assertEqual(command[-1], role)
        command = base.own_command(self.root, '--status')
        process = subprocess.run(command, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(process.stdout)['cumulative_authorized_cpu_seconds'], 1090000)
        self.prepare(50000, 'future-explicit-additional-grant')
        observed = base.sample_cpu(self.root)
        self.assertEqual(base.AUTHORIZED_CPU_SECONDS, 1140000)
        self.assertEqual(observed['authorized_cpu_seconds'], 1140000)
        self.assertGreaterEqual(observed['observed_cpu_seconds'], 252378.76458)
        self.assertGreater(observed['uncertain_upper_cpu_seconds'], 0)
        self.assertEqual(observed['remaining_cpu_seconds'], 1140000-observed['conservative_cpu_seconds'])
        base.write(self.root / extension.RUNTIME / 'state.json', {'status': 'RUNNING', 'authorized_cpu_seconds': 1090000})
        self.assertEqual(extension.read(self.root / extension.RUNTIME / 'state.json')['authorized_cpu_seconds'], 1140000)

    def test_completed_ml_is_inherited_and_excluded_from_execution(self):
        self.prepare()
        base = extension.load_runtime(self.root)
        self.assertEqual([task['id'] for task in base.config(self.root)['tasks']], ['quantum_publication', 'quantum_ir'])
        inherited = extension.read(self.root / extension.INHERITED)['tasks']
        self.assertEqual(inherited[0]['id'], 'ml_verification')
        self.assertEqual(inherited[0]['status'], 'COMPLETE')
        self.assertEqual(inherited[0]['historical_state']['attempts'], 1)
        self.assertTrue(inherited[0]['does_not_grant_current_paper_eligibility'])

    def test_only_a_superseded_budget_boundary_is_resumable(self):
        self.prepare()
        base = extension.load_runtime(self.root)
        state_path = self.root / extension.RUNTIME / 'state.json'
        state = extension.read(state_path)
        state['status'] = 'AUTHORIZED_CPU_BOUNDARY'
        extension.write(state_path, state)
        extension.write(self.root / extension.RUNTIME / 'budget_stop.json', {'status': 'AUTHORIZED_CPU_BOUNDARY'})
        resumed = extension.validate_resume_state(self.root, base)
        self.assertEqual(resumed['authorized_cpu_seconds'], 1090000)
        self.assertEqual(resumed['status'], 'AUTHORIZED_EXTENSION_RESUME_PENDING')
        self.assertFalse((self.root / extension.RUNTIME / 'budget_stop.json').exists())
        accounting = extension.read(self.root / extension.OUTPUT / 'cpu_accounting.json')
        self.assertGreaterEqual(accounting['observed_cpu_seconds'], 252378.76458)
        self.assertEqual(accounting['counter_epochs'][0]['last_counter_cpu_seconds'], 252378.76458)
        for forbidden in ('STOPPED_BY_USER', 'SCIENTIFIC_INTEGRITY_CHANGED'):
            extension.write(state_path, {**state, 'status': forbidden})
            with self.assertRaisesRegex(ValueError, 'cannot be bypassed'):
                extension.validate_resume_state(self.root, base)
        extension.write(state_path, state)
        extension.write(self.root / extension.RUNTIME / 'STOP.json', {'reason': 'User stop'})
        with self.assertRaisesRegex(ValueError, 'cannot be bypassed'):
            extension.validate_resume_state(self.root, base)

    def test_separate_wrapper_integrity_rejects_tampering(self):
        self.prepare()
        (self.root / extension.WRAPPER_COPY).write_text('modified wrapper')
        with self.assertRaisesRegex(ValueError, 'wrapper changed'):
            extension.load_runtime(self.root)


if __name__ == '__main__':
    unittest.main()
