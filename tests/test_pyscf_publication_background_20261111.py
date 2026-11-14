"""Regression tests for budget reuse, process identity and publication scope."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'scripts/background_pyscf_publication_20261111.py'
if not MODULE.exists():
    MODULE = Path(__file__).with_name('background_pyscf_publication_20261111.py')
spec = importlib.util.spec_from_file_location('pyscf_publication_background_test_subject', MODULE)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = b.Paths(Path(self.directory.name))

    def test_cpu_ledger_reuses_prior_sessions_and_preflight(self):
        b.write(self.paths.output / 'preflight_cpu.json', {'actual_cpu_seconds': 40,
                'conservative_cpu_seconds': 1000, 'measurement': 'measured+reserve'})
        b.write(self.paths.runtime / 'sessions/a.json', {'session_id': 'a',
                'observed_cpu_seconds': 200, 'conservative_cpu_seconds': 250})
        b.write(self.paths.runtime / 'sessions/b.json', {'session_id': 'b',
                'observed_cpu_seconds': 80, 'conservative_cpu_seconds': 80})
        self.assertEqual(b.ledger_totals(self.paths), (320, 1330))
        self.assertEqual(b.ledger_totals(self.paths, excluding='b'), (240, 1250))
        # Reading/restarting the meter never consumes or resets the ledger.
        self.assertEqual(b.ledger_totals(self.paths), (320, 1330))

    def test_reaped_worker_cpu_is_counted_once(self):
        parent = {'pid': 11, 'boot_id': 'a', 'start_ticks': 1}
        worker = {'pid': 22, 'boot_id': 'a', 'start_ticks': 2}
        before = {'own_cpu': 3.0, 'waited_child_cpu': 0.0}
        after = {'own_cpu': 3.0, 'waited_child_cpu': 120.0}
        with patch.object(b, 'matching_process', side_effect=lambda x: bool(x)), \
             patch.object(b, 'proc_stat', return_value=before), \
             patch.object(b, 'group_cpu', return_value=120.0):
            self.assertEqual(b.session_cpu(parent, worker), 123.0)
        with patch.object(b, 'matching_process', side_effect=lambda x: bool(x)), \
             patch.object(b, 'proc_stat', return_value=after):
            self.assertEqual(b.session_cpu(parent, None), 123.0)

    def test_reaping_race_retries_instead_of_recording_double_cpu(self):
        values = iter([{'own_cpu': 3., 'waited_child_cpu': 0.},
                       {'own_cpu': 3., 'waited_child_cpu': 120.},
                       {'own_cpu': 3., 'waited_child_cpu': 120.},
                       {'own_cpu': 3., 'waited_child_cpu': 120.}])
        with patch.object(b, 'matching_process', side_effect=lambda x: bool(x)), \
             patch.object(b, 'proc_stat', side_effect=lambda _: next(values)), \
             patch.object(b, 'group_cpu', side_effect=[120., 0.]):
            self.assertEqual(b.session_cpu({'pid': 1}, {'pid': 2}), 123.)

    def test_uncertain_interval_is_separate_from_observed_cpu(self):
        last = {'observed_cpu_seconds': 80., 'boot_id': 'x', 'sample_monotonic': 100.}
        result = b.conservative_recovery_charge(last, 'x', 110., 2)
        self.assertEqual(result['observed_cpu_seconds'], 80.)
        self.assertEqual(result['uncertain_upper_cpu_seconds'], 20.)
        self.assertEqual(result['conservative_cpu_seconds'], 100.)

    def test_reaping_inside_worker_group_does_not_double_count_child(self):
        root = {'pid': 22, 'pgrp': 22, 'sid': 22}
        parent = {'start_ticks': 2, 'own_cpu': 3., 'waited_child_cpu': 0.}
        child = {'start_ticks': 3, 'own_cpu': 120., 'waited_child_cpu': 0.}
        reaped = {22: {**parent, 'waited_child_cpu': 120.}}
        snapshots = [{22: parent, 23: child}, reaped, reaped, reaped]
        with patch.object(b, 'matching_process', return_value=True), \
             patch.object(b, 'proc_stat', return_value=root), \
             patch.object(b, '_group_snapshot', side_effect=snapshots):
            self.assertEqual(b.group_cpu({'pid': 22}), 123.)

    def test_reboot_cannot_silently_reset_or_infer_cpu(self):
        last = {'observed_cpu_seconds': 80., 'boot_id': 'old', 'sample_monotonic': 100.}
        result = b.conservative_recovery_charge(last, 'new', 1., 2)
        self.assertEqual(result['observed_cpu_seconds'], 80.)
        self.assertEqual(result['conservative_cpu_seconds'], b.AUTHORIZED_CPU)

    def test_recycled_pid_or_other_boot_cannot_be_signalled(self):
        expected = {'pid': os.getpid(), 'boot_id': b.boot_id(), 'start_ticks': -1}
        with patch.object(b.os, 'killpg') as kill:
            self.assertFalse(b.terminate_group(expected))
            kill.assert_not_called()
        expected['start_ticks'] = b.proc_stat(os.getpid())['start_ticks']
        expected['boot_id'] = 'other-boot'
        with patch.object(b.os, 'killpg') as kill:
            self.assertFalse(b.terminate_group(expected))
            kill.assert_not_called()

    def test_non_session_group_leader_cannot_be_signalled(self):
        expected = {'pid': 100, 'boot_id': 'boot', 'start_ticks': 123}
        with patch.object(b, 'matching_process', return_value=True), \
             patch.object(b, 'proc_stat', return_value={'pid': 100, 'pgrp': 10, 'sid': 10}), \
             patch.object(b.os, 'killpg') as kill:
            with self.assertRaises(ValueError):
                b.terminate_group(expected)
            kill.assert_not_called()

    def test_publication_only_includes_this_task_and_no_local_cache(self):
        names = ['scripts/run_pyscf_publication_20261111.py', 'scripts/unrelated.py',
                 'scripts/pyscf_repair_core_20261111.py',
                 'tests/test_pyscf_publication_background_20261111.py',
                 'docs/PYSCF_PUBLICATION_PLAN_20261111.md',
                 'reports/pyscf_publication_20261111/progress.json',
                 'reports/pyscf_publication_20261111/.background/state.json',
                 'reports/pyscf_publication_20261111/jobs/a/.local/density.npy',
                 'reports/pyscf_publication_20261111/jobs/a/pyscf.log',
                 'reports/pyscf_publication_20261111/jobs/a/result.json',
                 'reports/pyscf_repair_20261111/result.json']
        for name in names:
            path = self.paths.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
        selected = b.eligible_files(self.paths)
        self.assertEqual(selected, sorted([names[i] for i in (0, 3, 4, 5, 9)]))

    def test_foreign_staging_prevents_any_commit_or_push(self):
        calls = []
        def fake_git(paths, *args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, 'unrelated.py\0', '')
        with patch.object(b, 'require_branch'), patch.object(b, 'git', side_effect=fake_git):
            with self.assertRaises(ValueError):
                b.publish(self.paths, 'checkpoint')
        self.assertEqual(calls, [('diff', '--cached', '--name-only', '-z')])

    def test_progress_ignores_heartbeat_but_tracks_checkpoint_token(self):
        b.write(self.paths.output / 'progress.json', {'heartbeat_at_utc': 'one'})
        first = b.progress_marker(self.paths)
        b.write(self.paths.output / 'progress.json', {'heartbeat_at_utc': 'two'})
        self.assertEqual(first, b.progress_marker(self.paths))
        b.write(self.paths.output / 'progress.json', {'checkpoint_token': [1, 5]})
        self.assertNotEqual(first, b.progress_marker(self.paths))

    def test_singleton_lock_rejects_second_independent_handle(self):
        import fcntl
        lockpath = self.paths.root / 'test.lock'
        with lockpath.open('a+') as first, lockpath.open('a+') as second:
            fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_real_process_group_meter_and_exact_identity_signal(self):
        child = subprocess.Popen([sys.executable, '-c',
            'import time; end=time.process_time()+.08\nwhile time.process_time()<end: pass\ntime.sleep(10)'],
            start_new_session=True)
        self.addCleanup(lambda: child.kill() if child.poll() is None else None)
        expected = b.identity(child.pid)
        deadline = time.monotonic() + 3
        measured = 0
        while time.monotonic() < deadline and measured < .04:
            measured = b.group_cpu(expected)
            time.sleep(.01)
        self.assertGreaterEqual(measured, .04)
        self.assertTrue(b.terminate_group(expected))
        child.wait(timeout=3)
        self.assertFalse(b.matching_process(expected))


if __name__ == '__main__':
    unittest.main()
