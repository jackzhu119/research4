"""Small real subprocess tests; no research models or long background runs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('species_budget', ROOT / 'scripts/species_background_budget.py')
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def supervisor(self, name='state', **kwargs):
        options = {'budget_cpu_seconds': 5., 'poll_seconds': .01, 'heartbeat_seconds': .03,
                   'termination_grace_seconds': .05, 'cpu_reserve_seconds': .04}
        options.update(kwargs)
        return budget.BudgetSupervisor(self.root / name, **options)

    def job(self, name, code, **kwargs):
        return {'id': name, 'command': [sys.executable, '-c', code], 'cwd': str(self.root), **kwargs}

    def test_sleep_uses_cpu_not_wall_and_thread_environment_is_one(self):
        command = "import time,os,json;from pathlib import Path;time.sleep(.3);Path('done.json').write_text(json.dumps({k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')}))"
        started = time.monotonic()
        state = self.supervisor(budget_cpu_seconds=.5).run({'jobs': [self.job('sleep', command,
            outputs=['done.json'], env={'OPENBLAS_NUM_THREADS': '99'})]})
        self.assertEqual(state['status'], 'completed')
        self.assertGreater(time.monotonic() - started, .29)
        self.assertLess(state['spent_cpu_seconds'], .3)
        self.assertEqual(json.loads((self.root / 'done.json').read_text()), {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})

    def test_one_shared_budget_stops_later_job_and_preserves_unrelated_process(self):
        burn = "import time; s=time.process_time();\nwhile time.process_time()-s<.14: pass"
        external = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(10)'], start_new_session=True)
        try:
            state = self.supervisor(budget_cpu_seconds=.25).run({'jobs': [self.job('one', burn), self.job('two', burn), self.job('three', burn)]})
            self.assertEqual(state['status'], 'budget_exhausted')
            self.assertEqual(state['jobs']['one']['status'], 'completed')
            self.assertEqual(state['jobs']['two']['status'], 'interrupted')
            self.assertNotIn('three', state['jobs'])
            self.assertGreater(state['spent_cpu_seconds'], .14)
            self.assertLess(state['spent_cpu_seconds'], .3)
            self.assertIsNone(external.poll())
        finally:
            external.terminate()
            external.wait(timeout=3)

    def test_child_of_worker_cpu_is_counted(self):
        inner = 'import time; s=time.process_time();\nwhile time.process_time()-s<.16: pass'
        code = 'import subprocess,sys; subprocess.run([sys.executable,"-c",' + repr(inner) + '],check=True)'
        state = self.supervisor().run({'jobs': [self.job('descendant', code)]})
        self.assertEqual(state['status'], 'completed')
        self.assertGreater(state['spent_cpu_seconds'], .15)

    def test_descendant_new_session_is_counted_and_stopped(self):
        inner = 'import time; s=time.process_time();\nwhile time.process_time()-s<3: pass'
        code = 'import subprocess,sys; subprocess.run([sys.executable,"-c",' + repr(inner) + '],start_new_session=True,check=True)'
        state = self.supervisor(budget_cpu_seconds=.2).run({'jobs': [self.job('newsession', code)]})
        self.assertEqual(state['status'], 'budget_exhausted')
        self.assertEqual(state['jobs']['newsession']['status'], 'interrupted')
        self.assertIsNone(state['active'])
        self.assertLess(state['spent_cpu_seconds'], .3)

    def test_failed_job_continues_and_retry_is_explicit_and_bounded(self):
        state = self.supervisor().run({'jobs': [
            self.job('error', 'raise SystemExit(7)', max_retries=1),
            self.job('next', "from pathlib import Path;Path('ok').write_text('yes')", outputs=['ok'])]})
        self.assertEqual(state['status'], 'completed_with_errors')
        self.assertEqual(state['jobs']['error']['attempts'], 2)
        self.assertEqual(state['jobs']['next']['status'], 'completed')
        self.assertEqual(len(list((self.root / 'state/logs').glob('error.*.log'))), 2)

    def test_resume_keeps_cpu_counter_and_skips_only_intact_completion(self):
        manifest = {'jobs': [self.job('done', "from pathlib import Path;Path('out').write_text('correct')", outputs=['out'])]}
        first = self.supervisor().run(manifest)
        second = self.supervisor().run(manifest, resume=True)
        self.assertGreaterEqual(second['spent_cpu_seconds'], first['spent_cpu_seconds'])
        self.assertEqual(second['jobs']['done']['attempts'], 1)
        (self.root / 'out').write_text('corrupted')
        third = self.supervisor().run(manifest, resume=True)
        self.assertEqual(third['jobs']['done']['attempts'], 2)
        self.assertEqual((self.root / 'out').read_text(), 'correct')

    def test_all_inputs_are_frozen_before_first_job_and_drift_blocks_queue(self):
        (self.root / 'later_input').write_text('original')
        manifest = {'jobs': [self.job('mutate', "from pathlib import Path;Path('later_input').write_text('changed')"),
                             self.job('later', "raise RuntimeError('must never run')", inputs=['later_input'])]}
        state = self.supervisor().run(manifest)
        self.assertEqual(state['status'], 'blocked_provenance')
        self.assertNotIn('later', state['jobs'])
        self.assertEqual(state['input_hashes']['later'][str(self.root / 'later_input')],
                         __import__('hashlib').sha256(b'original').hexdigest())

    def test_changed_manifest_cannot_reset_budget_or_append_work(self):
        original = {'jobs': [self.job('first', 'pass')]}
        self.supervisor().run(original)
        with self.assertRaises(budget.ProvenanceError):
            self.supervisor().run({'jobs': [self.job('second', 'pass')]}, resume=True)
        with self.assertRaises(budget.ProvenanceError):
            self.supervisor(budget_cpu_seconds=10).run(original, resume=True)

    def test_budget_override_cannot_exceed_manifest_cap_but_may_lower_it(self):
        manifest = {'cpu_budget_seconds': .5, 'jobs': [self.job('bounded', 'pass')]}
        with self.assertRaises(ValueError):
            self.supervisor(budget_cpu_seconds=.6).run(manifest)
        self.assertFalse((self.root / 'state/state.json').exists())
        manifest_path = self.root / 'capped.json'; manifest_path.write_text(json.dumps(manifest))
        response = subprocess.run([sys.executable, str(ROOT / 'scripts/species_background_budget.py'),
            'run', '--manifest', str(manifest_path), '--state-dir', str(self.root / 'cli'),
            '--budget-cpu-seconds', '.6'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.assertNotEqual(response.returncode, 0)
        self.assertFalse((self.root / 'cli/state.json').exists())
        reduced = self.supervisor(budget_cpu_seconds=.4).run(manifest)
        self.assertEqual(reduced['status'], 'completed')
        self.assertEqual(reduced['budget_cpu_seconds'], .4)

    def test_stale_or_reused_pid_never_authorizes_signal(self):
        identity = budget.proc_stat(os.getpid())
        identity.update({'boot_id': budget.boot_id(), 'token': 'not-a-worker-token'})
        identity['start_ticks'] += 1
        with mock.patch.object(budget.os, 'killpg') as kill:
            self.assertFalse(budget.signal_owned_groups(identity, __import__('signal').SIGTERM))
            kill.assert_not_called()

    def test_stop_and_clean_resume_and_postprocess(self):
        state_dir = self.root / 'state'
        code = "import time;from pathlib import Path;time.sleep(.4);Path('longdone').write_text('yes')"
        manifest = {'cpu_budget_seconds': 3., 'poll_seconds': .01, 'heartbeat_seconds': .02,
                    'termination_grace_seconds': .05, 'jobs': [self.job('long', code, outputs=['longdone'])],
                    'postprocess_cmd': [sys.executable, '-c', "from pathlib import Path;Path('summary').write_text('done')"],
                    'postprocess_outputs': ['summary'], 'cwd': str(self.root)}
        manifest_path = self.root / 'manifest.json'; manifest_path.write_text(json.dumps(manifest))
        command = [sys.executable, str(ROOT / 'scripts/species_background_budget.py'), 'run', '--manifest', str(manifest_path), '--state-dir', str(state_dir)]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            until = time.monotonic() + 3
            while time.monotonic() < until:
                if (state_dir / 'state.json').exists() and json.loads((state_dir / 'state.json').read_text()).get('active'):
                    break
                time.sleep(.01)
            self.assertTrue(budget.request_stop(state_dir)['supervisor_identity_matches'])
            process.wait(timeout=3)
            stopped = json.loads((state_dir / 'state.json').read_text())
            self.assertEqual(stopped['status'], 'paused')
            self.assertIsNone(stopped['active'])
            resumed = self.supervisor(budget_cpu_seconds=3.).run(manifest, resume=True)
            self.assertEqual(resumed['status'], 'completed')
            self.assertGreaterEqual(resumed['spent_cpu_seconds'], stopped['spent_cpu_seconds'])
            self.assertEqual((self.root / 'summary').read_text(), 'done')
        finally:
            if process.poll() is None:
                process.terminate(); process.wait(timeout=3)
            if process.stderr:
                process.stderr.close()


if __name__ == '__main__':
    unittest.main()
