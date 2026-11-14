"""Meaningful supervision regressions: repeated failure, idling, CPU ownership."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import background_pyscf_ir_20261111 as bg


def test_cpu_ledger_uses_sessions_once_and_does_not_precharge(tmp_path):
    paths = bg.Paths(tmp_path)
    assert bg.session_totals(paths) == (0.0, 0.0)
    for name, actual, charged in [('a', 3.5, 3.5), ('b', 2.0, 2.2)]:
        bg.write(paths.runtime / 'sessions' / (name + '.json'), {
            'session_id': name, 'observed_cpu_seconds': actual,
            'conservative_cpu_seconds': charged})
    assert bg.session_totals(paths) == pytest.approx((5.5, 5.7))
    assert bg.session_totals(paths, excluding='a') == pytest.approx((2.0, 2.2))
    bg.record_preflight(paths, 12.5, 'Measured fixture CPU')
    assert bg.session_totals(paths) == pytest.approx((18.0, 18.2))
    bg.record_preflight(paths, 99.0, 'Must not duplicate existing preparation charge')
    assert bg.session_totals(paths) == pytest.approx((18.0, 18.2))


def test_same_supervisor_restart_subtracts_already_charged_counter(tmp_path):
    paths = bg.Paths(tmp_path)
    owner = {'pid': 17, 'start_ticks': 300, 'boot_id': 'same-boot'}
    bg.write(paths.runtime / 'sessions/a.json', {
        'session_id': 'a', 'supervisor_identity': owner,
        'observed_cpu_seconds': 200.0, 'supervisor_raw_counter_at_final': 200.0})
    baseline = bg.restart_baseline(paths, owner)
    delta = bg.session_cpu_delta(330.0, baseline)
    assert baseline == 200.0 and delta == 130.0
    bg.write(paths.runtime / 'sessions/b.json', {
        'session_id': 'b', 'supervisor_identity': owner, 'observed_cpu_seconds': delta})
    assert bg.session_totals(paths) == (330.0, 330.0)
    assert bg.restart_baseline(paths, {**owner, 'start_ticks': 301}) == 0.0


def test_sampler_retries_a_child_reaping_race_without_double_count(monkeypatch):
    calls = 0

    def snapshot(owner):
        nonlocal calls
        calls += 1
        if owner['pid'] == 10:
            return {10: {'start_ticks': 1, 'own_cpu': 1,
                         'waited_child_cpu': 0 if calls == 1 else 2}}
        return ({20: {'start_ticks': 2, 'own_cpu': 2, 'waited_child_cpu': 0}}
                if calls == 2 else {})

    monkeypatch.setattr(bg, 'matching', lambda _: True)
    monkeypatch.setattr(bg, 'group_snapshot', snapshot)
    assert bg.cpu_snapshot({'pid': 10}, {'pid': 20}) == 3
    assert calls == 8  # First incoherent transfer was rejected, not persisted.


def test_signal_rejects_pid_reuse_and_nonleader(monkeypatch):
    expected = {'pid': 50, 'start_ticks': 7, 'boot_id': 'boot'}
    monkeypatch.setattr(bg, 'boot_id', lambda: 'boot')
    monkeypatch.setattr(bg, 'proc_stat', lambda _: {
        'state': 'S', 'start_ticks': 8, 'pid': 50, 'pgrp': 50, 'sid': 50})
    assert bg.signal_group(expected) is False
    monkeypatch.setattr(bg, 'proc_stat', lambda _: {
        'state': 'S', 'start_ticks': 7, 'pid': 50, 'pgrp': 1, 'sid': 1})
    with pytest.raises(ValueError, match='own its session'):
        bg.signal_group(expected)


@pytest.mark.parametrize('measured,stopped', [(899999.8, False), (900000.0, True)])
def test_guard_has_no_early_reserve_but_enforces_authorization(tmp_path, monkeypatch, measured, stopped):
    paths = bg.Paths(tmp_path)
    bg.write(paths.state, {'status': 'RUNNING', 'session_id': 's',
             'supervisor_identity': {'pid': 1}, 'worker_identity': {'pid': 2},
             'publisher_identity': None})
    monkeypatch.setattr(bg, 'matching', lambda _: True)
    monkeypatch.setattr(bg, 'identity', lambda _: {'pid': 3})
    monkeypatch.setattr(bg, 'cpu_snapshot', lambda *a: measured)
    signalled = []
    monkeypatch.setattr(bg, 'signal_group', lambda p, sig: signalled.append(p))

    class EndTick(Exception):
        pass

    monkeypatch.setattr(bg.time, 'sleep', lambda _: (_ for _ in ()).throw(EndTick()))
    with pytest.raises(EndTick):
        bg.guard(paths, 's')
    assert (paths.runtime / 'budget_stop.json').exists() is stopped
    assert bool(signalled) is stopped
    assert bg.session_totals(paths)[0] == measured


def make_repository(tmp_path, exitcode):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    source = Path(bg.__file__).read_text().replace(
        'SCAN_INTERVAL_SECONDS = 60.0', 'SCAN_INTERVAL_SECONDS = 0.02').replace(
        'POLL_SECONDS = 0.25', 'POLL_SECONDS = 0.02')
    (scripts / bg.SELF_NAME).write_text(source)
    (scripts / bg.RUNNER_NAME).write_text('''import json, os, pathlib, sys
root=pathlib.Path(os.environ['PYSCF_IR_REPOSITORY_ROOT'])
p=root/'scans.json'
value=json.loads(p.read_text()) if p.exists() else {'scans':0}
value['scans'] += 1
value['args']=sys.argv[1:]
value['module_directory']=str(pathlib.Path(__file__).parent)
t=p.with_suffix('.tmp');t.write_text(json.dumps(value));t.replace(p)
raise SystemExit(EXIT_CODE)
'''.replace('EXIT_CODE', str(exitcode)))
    subprocess.run(['git', 'init', '-b', 'research/ir-test'], cwd=tmp_path,
                   check=True, capture_output=True)
    return bg.Paths(tmp_path)


def await_value(predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.03)
    raise AssertionError('Expected background state did not arrive')


@pytest.mark.parametrize('exitcode', [0, 2])
def test_real_detached_scans_keep_running_after_idle_or_five_failures(tmp_path, exitcode):
    paths = make_repository(tmp_path, exitcode)
    try:
        launched = bg.start(paths)
        assert launched['supervisor_alive'] and launched['guard_alive']
        owner = launched['state']['supervisor_identity']
        assert os.getsid(owner['pid']) == owner['pid']
        duplicate = bg.start(paths)
        assert duplicate['state']['supervisor_identity'] == owner
        await_value(lambda: bg.read(paths.state, {}).get('completed_scans', 0) >= 5)
        value = bg.verify(paths)
        assert value['verified'] and value['state']['status'] == 'RUNNING'
        assert value['state']['failed_scans'] >= 5 if exitcode else value['state']['failed_scans'] == 0
        scans = bg.read(tmp_path / 'scans.json')
        assert scans['args'] == ['--once']
        assert scans['module_directory'] == str(paths.frozen / 'scripts')
        assert value['observed_cpu_seconds'] > 0
    finally:
        bg.stop(paths)
        await_value(lambda: not bg.status(paths)['supervisor_alive'])
    assert bg.read(paths.state)['status'] == 'STOPPED_BY_USER'
    assert bg.session_totals(paths)[0] > 0


def test_frozen_input_tamper_is_detected_and_separate_outputs_are_excluded(tmp_path):
    paths = make_repository(tmp_path, 0)
    bg.freeze(paths)
    assert bg.check_integrity(paths)
    (paths.root / 'scripts' / bg.RUNNER_NAME).write_text('raise SystemExit(0)\n')
    assert bg.check_integrity(paths)  # Workspace changes cannot affect frozen execution.
    assert 'scripts/' + bg.RUNNER_NAME in bg.workspace_changes(paths)
    paths.frozen_runner.write_text('raise SystemExit(0)\n')
    with pytest.raises(ValueError, match='source changed'):
        bg.check_integrity(paths)
    bg.write(paths.output / 'result.json', {'ok': True})
    bg.write(paths.runtime / 'secret-runtime.json', {'runtime': True})
    elsewhere = tmp_path / 'reports/pyscf_publication_20261111'
    bg.write(elsewhere / 'result.json', {'other': True})
    files = bg.eligible_files(paths)
    assert 'reports/pyscf_ir_20261111/result.json' in files
    assert not any('.background' in name or 'pyscf_publication' in name for name in files)


def test_completion_requires_final_successful_publication(tmp_path):
    paths = make_repository(tmp_path, 0)
    runner = paths.root / 'scripts' / bg.RUNNER_NAME
    runner.write_text(runner.read_text().replace('raise SystemExit(0)',
        "q=root/'reports/pyscf_ir_20261111/summary.json';q.write_text(json.dumps({'status':'COMPLETE'}))\nraise SystemExit(0)"))
    origin = tmp_path / 'local-origin.git'
    subprocess.run(['git', 'init', '--bare', str(origin)], check=True, capture_output=True)
    for args in [('config', 'user.name', 'IR supervisor test'),
                 ('config', 'user.email', 'ir-test@example.invalid'),
                 ('remote', 'add', 'origin', str(origin))]:
        subprocess.run(['git', *args], cwd=tmp_path, check=True, capture_output=True)
    try:
        bg.start(paths)
        await_value(lambda: bg.read(paths.state, {}).get('status') == 'COMPLETE')
        await_value(lambda: not bg.status(paths)['supervisor_alive'])
        assert bg.verify(paths)['verified']
        remote = subprocess.run(['git', '--git-dir', str(origin), 'show',
            'refs/heads/research/ir-test:reports/pyscf_ir_20261111/summary.json'],
            check=True, text=True, capture_output=True)
        assert json.loads(remote.stdout)['status'] == 'COMPLETE'
    finally:
        if bg.status(paths)['supervisor_alive']:
            bg.stop(paths)
            await_value(lambda: not bg.status(paths)['supervisor_alive'])


def test_guard_replaces_a_lost_supervisor_and_retains_cpu_ledger(tmp_path):
    paths = make_repository(tmp_path, 2)
    try:
        initial = bg.start(paths)['state']['supervisor_identity']
        await_value(lambda: bg.read(paths.state, {}).get('completed_scans', 0) >= 1)
        os.kill(initial['pid'], signal.SIGKILL)
        await_value(lambda: (bg.read(paths.state, {}).get('supervisor_identity') != initial
                            and bg.verify(paths)['verified']))
        await_value(lambda: bg.read(paths.state, {}).get('completed_scans', 0) >= 1)
        assert len(list((paths.runtime / 'sessions').glob('*.json'))) == 2
        assert bg.session_totals(paths)[0] > 0
    finally:
        bg.stop(paths)
        await_value(lambda: not bg.status(paths)['supervisor_alive'])


def test_actual_same_process_supervisor_retry_keeps_counter_baseline(tmp_path):
    paths = make_repository(tmp_path, 0)
    runner = paths.root / 'scripts' / bg.RUNNER_NAME
    runner.write_text(runner.read_text().replace('raise SystemExit(0)',
        "q=root/'reports/pyscf_ir_20261111/summary.json'\nq.write_text('invalid JSON' if value['scans']==1 else json.dumps({'status':'RUNNING_OR_WAITING'}))\nraise SystemExit(0)"))
    try:
        initial = bg.start(paths)['state']['supervisor_identity']
        await_value(lambda: (bg.read(paths.state, {}).get('supervisor_counter_baseline', 0) > 0
                            and bg.read(paths.state, {}).get('completed_scans', 0) >= 1))
        state = bg.read(paths.state)
        assert state['supervisor_identity'] == initial
        entries = [bg.read(p) for p in (paths.runtime / 'sessions').glob('*.json')]
        closed = [p for p in entries if p.get('status') == 'SUPERVISOR_INTERRUPTED']
        assert len(closed) == 1
        assert state['supervisor_counter_baseline'] == closed[0]['supervisor_raw_counter_at_final']
    finally:
        bg.stop(paths)
        await_value(lambda: not bg.status(paths)['supervisor_alive'])
