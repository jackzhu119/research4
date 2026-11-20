"""Second authorization arriving after watcher launch is used at activation."""
from pathlib import Path
import importlib.util
import signal
import sys


REPO = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == 'tests' else Path('/workspace/research4')
spec = importlib.util.spec_from_file_location(
    'ir_extension_existing_test_helpers', REPO/'tests/test_pyscf_ir_extension_20261111.py')
t = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t
spec.loader.exec_module(t)
e = t.e


def test_second_grant_is_read_by_already_armed_watcher_and_successor_guard(tmp_path):
    b,p=t.fixture(tmp_path)
    # Assemble this isolated fixture before any processes are launched. The
    # worker waits on a gate, making the second grant timing deterministic.
    relative='scripts/run_pyscf_ir_20261111.py'
    runner=tmp_path/relative
    runner.write_text('''import json, os, pathlib, time
root=pathlib.Path(os.environ['PYSCF_IR_REPOSITORY_ROOT'])
while not (root/'release_worker').exists(): time.sleep(.01)
start=time.process_time()
while time.process_time()-start < 1.5: pass
p=root/'worker_completed.json'
value=json.loads(p.read_text()) if p.exists() else {'completed':0}
value['completed']+=1
p.write_text(json.dumps(value))
''')
    (p.frozen/relative).write_bytes(runner.read_bytes())
    (p.output/'frozen_source'/relative).write_bytes(runner.read_bytes())
    manifest=b.read(p.manifest)
    manifest['files'][relative]=b.sha(runner)
    b.write(p.manifest,manifest)
    manifest_before=p.manifest.read_bytes()
    assert b.check_integrity(p)
    try:
        initial=b.start(p)
        assert initial['supervisor_alive']
        t.await_true(lambda:b.read(p.state,{}).get('worker_identity') is not None)
        old_state=b.read(p.state)
        old_worker=old_state['worker_identity']
        assert e.grant(tmp_path,2,'first-grant')['cumulative_authorized_cpu_seconds']==3
        armed=e.arm(tmp_path)
        assert armed['continuation']['status']=='ARMED_WAITING'
        watcher_identity=armed['continuation']['watcher_identity']
        assert armed['continuation']['authorization_total_cpu_seconds']==3
        # The running watcher was created with total=3; the live ledger now
        # supplies total=23 without replacing watcher, worker or supervisor.
        assert e.grant(tmp_path,20,'second-grant')['cumulative_authorized_cpu_seconds']==23
        rearmed=e.arm(tmp_path)
        assert rearmed['continuation']['watcher_identity']==watcher_identity
        assert b.read(p.state)['worker_identity']==old_worker
        assert b.read(p.state)['supervisor_identity']==old_state['supervisor_identity']
        assert e.allowance(tmp_path)==23
        (tmp_path/'release_worker').write_text('go')
        t.await_true(lambda:e.read(p.output/'authorization_continuation.json',{}).get('status')=='ACTIVATED')
        state=b.read(p.state)
        assert state['authorized_cpu_seconds']==23
        assert state['supervisor_identity']!=old_state['supervisor_identity']
        assert b.matching(state['supervisor_identity'])
        assert b.matching(state['guard_identity'])
        for key in ('supervisor_identity','guard_identity'):
            args=Path(f"/proc/{state[key]['pid']}/cmdline").read_bytes()
            assert str(e.extension_file(tmp_path)).encode() in args
        # Crossing the watcher's original total=3 proves its newly launched
        # guard enforces the later total, rather than stale startup metadata.
        t.await_true(lambda:e.verify(tmp_path)['observed_cpu_seconds']>3.5)
        verified=e.verify(tmp_path)
        assert verified['current_phase_cpu_boundary']==23
        assert verified['supervisor_alive'] and verified['guard_alive']
        assert verified['state']['status']=='RUNNING'
        assert p.manifest.read_bytes()==manifest_before
        assert b.check_integrity(p)
    finally:
        state=b.read(p.state,{})
        if b.matching(state.get('supervisor_identity')):
            b.stop(p)
            t.await_true(lambda:not b.status(p)['supervisor_alive'])
        state=e.read(p.output/'authorization_continuation.json',{})
        if b.matching(state.get('watcher_identity')):
            b.signal_group(state['watcher_identity'],signal.SIGTERM)
