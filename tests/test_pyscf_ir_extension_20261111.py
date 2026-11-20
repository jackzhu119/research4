"""Budget extension regression tests; isolated fixture processes only."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

REPO = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == 'tests' else Path('/workspace/research4')
SCRIPT = REPO/'scripts/extend_pyscf_ir_20261111.py'
spec = importlib.util.spec_from_file_location('ir_extension_under_review', SCRIPT)
e = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e
spec.loader.exec_module(e)


def await_true(predicate, seconds=12):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if predicate():return
        time.sleep(.03)
    raise AssertionError('Expected isolated extension transition did not arrive')


def fixture(tmp_path):
    scripts=tmp_path/'scripts';scripts.mkdir()
    source=(REPO/'scripts/background_pyscf_ir_20261111.py').read_text()
    source=source.replace('AUTHORIZED_CPU_SECONDS = 900000.0','AUTHORIZED_CPU_SECONDS = 1.0')
    source=source.replace('POLL_SECONDS = 0.25','POLL_SECONDS = 0.02')
    source=source.replace('SCAN_INTERVAL_SECONDS = 60.0','SCAN_INTERVAL_SECONDS = 0.03')
    p=scripts/'background_pyscf_ir_20261111.py';p.write_text(source)
    (scripts/'run_pyscf_ir_20261111.py').write_text('''import json, os, pathlib, time
p=pathlib.Path(os.environ['PYSCF_IR_REPOSITORY_ROOT'])/'worker_completed.json'
start=time.process_time()
while time.process_time()-start < 1.5: pass
value=json.loads(p.read_text()) if p.exists() else {'completed':0}
value['completed']+=1
p.write_text(json.dumps(value))
''')
    subprocess.run(['git','init','-b','research/ir-extension-test'],cwd=tmp_path,check=True,capture_output=True)
    spec=importlib.util.spec_from_file_location('fixture_ir_background_'+str(time.time_ns()),p)
    b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
    paths=b.Paths(tmp_path)
    b.freeze(paths)
    b.record_preflight(paths,0,'Isolated test fixture, not shared cgroup CPU')
    return b,paths


def test_grant_idempotent_and_rejects_conflicting_amount(tmp_path):
    b,p=fixture(tmp_path)
    old_manifest=(p.manifest).read_bytes()
    assert e.grant(tmp_path,20,'request-a')['cumulative_authorized_cpu_seconds']==21
    assert e.grant(tmp_path,20,'request-a')['cumulative_authorized_cpu_seconds']==21
    assert e.allowance(tmp_path)==21
    assert len(e.read(e.output(tmp_path)/'authorization.json')['grants'])==1
    with pytest.raises(ValueError,match='different amount'):e.grant(tmp_path,21,'request-a')
    assert p.manifest.read_bytes()==old_manifest
    assert b.check_integrity(p)


@pytest.mark.parametrize('amount',[True,0,-1,float('nan'),float('inf')])
def test_invalid_grants_rejected(tmp_path,amount):
    fixture(tmp_path)
    with pytest.raises(ValueError):e.grant(tmp_path,amount,'invalid')


def test_boundary_requires_quiescent_owned_processes_and_released_lock(tmp_path,monkeypatch):
    b,p=fixture(tmp_path)
    monkeypatch.setattr(b,'matching',lambda identity: identity=={'pid':4})
    assert not e.boundary_ready(b,p,{'status':'RUNNING'})
    assert not e.boundary_ready(b,p,{'status':'AUTHORIZED_CPU_BOUNDARY','guard_identity':{'pid':4}})
    monkeypatch.setattr(b,'matching',lambda identity:False)
    import fcntl
    with (p.runtime/'supervisor.lock').open('a+') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not e.boundary_ready(b,p,{'status':'AUTHORIZED_CPU_BOUNDARY'})
    assert e.boundary_ready(b,p,{'status':'AUTHORIZED_CPU_BOUNDARY'})


@pytest.mark.parametrize('terminal',['COMPLETE','STOPPED_BY_USER','SCIENTIFIC_INTEGRITY_CHANGED'])
def test_terminal_scientific_or_user_states_are_not_restarted(tmp_path,monkeypatch,terminal):
    b,p=fixture(tmp_path)
    e.grant(tmp_path,20,'next')
    target=e.extension_file(tmp_path);target.parent.mkdir(parents=True);target.write_bytes(SCRIPT.read_bytes())
    e.write(p.output/'authorization_extension_manifest.json',{'sha256':e.sha(target)})
    b.write(p.state,{'status':terminal})
    assert e.watch(tmp_path,interval=.01)==0
    assert e.read(p.output/'authorization_continuation.json')['status']=='NO_RESTART_'+terminal
    assert not b.status(p)['supervisor_alive']
    records=[b.read(q) for q in (p.runtime/'sessions').glob('*.json')]
    assert len(records)==1 and records[0]['status'].startswith('AUTHORIZATION_WATCHER_')
    assert records[0]['observed_cpu_seconds']>=0


def test_real_boundary_continuation_keeps_ledger_and_routes_all_background_children(tmp_path):
    b,p=fixture(tmp_path)
    initial=None
    try:
        initial=b.start(p)['state']['supervisor_identity']
        await_true(lambda:b.read(p.state,{}).get('worker_identity') is not None)
        old_state=b.read(p.state)
        old_worker=old_state['worker_identity']
        manifest=p.manifest.read_bytes()
        e.grant(tmp_path,20,'additional-cpu')
        armed=e.arm(tmp_path)
        assert armed['cumulative_authorized_cpu_seconds']==21
        await_true(lambda:e.read(p.output/'authorization_continuation.json',{}).get('status')=='ACTIVATED')
        state=b.read(p.state)
        assert state['supervisor_identity']!=initial
        assert state['authorized_cpu_seconds']==21
        assert '21 cumulative' in state['authorization']
        assert b.matching(state['supervisor_identity']) and b.matching(state['guard_identity'])
        for key in ('supervisor_identity','guard_identity'):
            args=Path(f"/proc/{state[key]['pid']}/cmdline").read_bytes()
            assert str(e.extension_file(tmp_path)).encode() in args
        await_true(lambda:(tmp_path/'worker_completed.json').exists())
        # Work continued after the old 1 CPU second boundary; no reset to zero.
        verified=e.verify(tmp_path)
        assert verified['verified']
        assert verified['current_phase_cpu_boundary']==21
        assert verified['observed_cpu_seconds']>1
        sessions=[b.read(q) for q in (p.runtime/'sessions').glob('*.json')]
        assert any(s['status']=='AUTHORIZED_CPU_BOUNDARY' for s in sessions)
        assert any(s['status'].startswith('AUTHORIZATION_WATCHER_') for s in sessions)
        assert len(list((p.output/'authorization').glob('previous_boundary_*.json')))==1
        assert p.manifest.read_bytes()==manifest
        assert b.check_integrity(p)
    finally:
        state=b.read(p.state,{})
        if b.matching(state.get('supervisor_identity')):
            b.stop(p)
            await_true(lambda:not b.status(p)['supervisor_alive'])
        continuation=e.read(p.output/'authorization_continuation.json',{})
        watch=continuation.get('watcher_identity')
        if b.matching(watch):
            b.signal_group(watch,signal.SIGTERM)
