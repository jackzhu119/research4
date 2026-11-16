#!/usr/bin/env python3
"""Detached, resumable validation under one new 990000 CPU-second grant.

No score, elapsed-time, failed-attempt, or idle threshold terminates work.
Only explicit user stop, scientific source corruption, completed validation,
or the measured aggregate authorization boundary can end the supervisor.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

AUTHORIZED_CPU_SECONDS = 990000.0
RELATIVE_OUTPUT = 'reports/all_round_validation_20261112'
CONFIG = 'data/research/all_round_validation_20261112/tasks.json'
SELF = 'scripts/background_all_round_validation_20261112.py'
POLL_SECONDS = 0.5
RETRY_SECONDS = 60.0
PUBLISH_SECONDS = 600.0
TERMINAL = {'COMPLETE', 'AUTHORIZED_CPU_BOUNDARY', 'STOPPED_BY_USER', 'SCIENTIFIC_INTEGRITY_CHANGED'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Repository-relative path required: ' + str(value))
    return path


def output(root):
    return root / RELATIVE_OUTPUT


def runtime(root):
    return output(root) / '.background'


def frozen(root):
    return runtime(root) / 'frozen'


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def proc_stat(pid):
    try:
        raw = Path('/proc') / str(int(pid)) / 'stat'
        text = raw.read_text()
        fields = text[text.rfind(')') + 2:].split()
        return {'pid': int(pid), 'state': fields[0], 'pgrp': int(fields[2]),
                'sid': int(fields[3]), 'start_ticks': int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def identity(pid):
    value = proc_stat(pid)
    return {**value, 'boot_id': boot_id()} if value else None


def alive(expected):
    if not expected or expected.get('boot_id') != boot_id():
        return False
    point = proc_stat(expected.get('pid', -1))
    return bool(point and point['state'] != 'Z' and point['start_ticks'] == expected.get('start_ticks'))


def signal_session(expected, signum=signal.SIGKILL):
    if not alive(expected):
        return False
    point = proc_stat(expected['pid'])
    if point['pgrp'] != point['pid'] or point['sid'] != point['pid']:
        raise ValueError('Refusing to signal a process that does not own its session')
    try:
        os.killpg(point['pid'], signum)
        return True
    except ProcessLookupError:
        return False


def cgroup_sample():
    path = Path('/sys/fs/cgroup/cpu.stat')
    stats = dict(line.split() for line in path.read_text().splitlines())
    seconds = float(stats['usage_usec']) / 1e6
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid Linux cgroup CPU counter')
    capacity = float(len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1))
    try:
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota!='max':
            capacity=min(capacity,float(quota)/float(period))
    except (OSError,ValueError,ZeroDivisionError):
        pass
    return {'counter_cpu_seconds': seconds, 'boot_id': boot_id(), 'cpu_capacity':capacity,
            'sampled_monotonic':time.monotonic(),
            'cgroup_device': path.stat().st_dev, 'cgroup_inode': path.stat().st_ino,
            'counter_path': str(path), 'sampled_at_utc': utc()}


def counter_key(sample):
    return tuple(sample[k] for k in ('boot_id', 'cgroup_device', 'cgroup_inode'))


def account_sample(ledger, sample):
    """Carry observations across cgroup replacement; never reset the grant."""
    ledger = dict(ledger or {})
    epochs = list(ledger.get('counter_epochs', []))
    if epochs and counter_key(epochs[-1]) == counter_key(sample):
        if sample['counter_cpu_seconds'] < epochs[-1]['last_counter_cpu_seconds']:
            raise ValueError('Cgroup CPU counter moved backwards within one identity')
    else:
        if epochs:
            previous=epochs[-1]
            if previous['boot_id']==sample['boot_id'] and 'last_sampled_monotonic' in previous and 'sampled_monotonic' in sample:
                gap=max(0.0,sample['sampled_monotonic']-previous['last_sampled_monotonic'])
            else:
                gap=max(0.0,(datetime.fromisoformat(sample['sampled_at_utc'])-datetime.fromisoformat(previous['last_sampled_at_utc'])).total_seconds())
            previous['unobserved_tail_after_cgroup_loss'] = 'UNKNOWN actual CPU; separately charged conservative capacity-times-gap upper bound'
            previous['uncertain_upper_cpu_seconds']=gap*float(previous.get('cpu_capacity',os.cpu_count() or 1))
            previous['uncertain_tail_gap_seconds']=gap
        epochs.append({**sample, 'counter_baseline_cpu_seconds': 0.0,
                       'first_observed_counter_cpu_seconds': sample['counter_cpu_seconds'],
                       'scope': 'Entire dedicated environment cgroup, including this task preparation'})
    epochs[-1].update(last_counter_cpu_seconds=sample['counter_cpu_seconds'], last_sampled_at_utc=sample['sampled_at_utc'],
                      last_sampled_monotonic=sample.get('sampled_monotonic'))
    used = sum(item['last_counter_cpu_seconds'] - item['counter_baseline_cpu_seconds'] for item in epochs)
    uncertainty=sum(item.get('uncertain_upper_cpu_seconds',0.0) for item in epochs)
    charged=used+uncertainty
    ledger.update(schema='all-round-cgroup-cpu-v1', authorized_cpu_seconds=AUTHORIZED_CPU_SECONDS,
                  observed_cpu_seconds=used, conservative_cpu_seconds=charged,
                  uncertain_upper_cpu_seconds=uncertainty,
                  remaining_cpu_seconds=max(0.0, AUTHORIZED_CPU_SECONDS-charged),
                  counter_epochs=epochs, not_wall_time=True, early_cpu_reserve_seconds=0,
                  updated_at_utc=utc(), sampling_interval_seconds=POLL_SECONDS,
                  boundary_overshoot_cpu_seconds=max(0.0, charged-AUTHORIZED_CPU_SECONDS),
                  historical_other_chat_budgets_included=False,
                  lifetime_guarantee=False)
    return ledger


def sample_cpu(root):
    """Separate meter lock permits independent guard and supervisor sampling."""
    runtime(root).mkdir(parents=True, exist_ok=True)
    with (runtime(root) / 'cpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = account_sample(read(output(root)/'cpu_accounting.json', {}), cgroup_sample())
        write(output(root)/'cpu_accounting.json', value)
        return value


def config(root):
    value = read(frozen(root)/CONFIG) or read(root/CONFIG)
    if not value or not isinstance(value.get('tasks'), list) or not value['tasks']:
        raise ValueError('A nonempty frozen task configuration is required')
    ids = [item['id'] for item in value['tasks']]
    if len(set(ids)) != len(ids) or any(not item.replace('_', '').replace('-', '').isalnum() for item in ids):
        raise ValueError('Task IDs must be unique safe names')
    for task in value['tasks']:
        if not isinstance(task.get('command'), list) or not task['command'] or not task.get('completion'):
            raise ValueError('Each task needs an argv command and an explicit completion predicate')
        safe_relative(task['completion']['path'])
    return value


def freeze(root):
    manifest_path = output(root)/'source_manifest.json'
    if manifest_path.exists():
        check_integrity(root)
        return
    cfg = read(root/CONFIG)
    if not cfg:
        raise ValueError('Missing task configuration: ' + CONFIG)
    paths = set([Path(SELF), Path(CONFIG)])
    for name in cfg.get('freeze_paths', ['scripts', 'src']):
        relative = safe_relative(name)
        target = root/relative
        if target.is_dir():
            candidates=target.rglob('*.py') if relative.parts[0] in {'scripts','src'} else target.rglob('*')
            paths.update(p.relative_to(root) for p in candidates if p.is_file()
                         and not any(part.startswith('.') or part=='__pycache__' for part in p.relative_to(root).parts))
        elif target.is_file():
            paths.add(relative)
        else:
            raise FileNotFoundError('Frozen source is absent: ' + str(relative))
    files = {}
    for relative in sorted(paths):
        source = root/relative
        if source.is_symlink() or not source.is_file():
            raise ValueError('Frozen source must be a regular file: ' + str(relative))
        files[str(relative)] = sha(source)
        for destination in (frozen(root)/relative, output(root)/'frozen_source'/relative):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    write(manifest_path, {'schema':'all-round-frozen-v1', 'files':files, 'frozen_at_utc':utc(),
                         'python_executable':sys.executable, 'python_version':sys.version,
                         'configuration_path':CONFIG})
    check_integrity(root)


def check_integrity(root):
    manifest = read(output(root)/'source_manifest.json', {})
    if not manifest.get('files'):
        raise ValueError('Frozen source manifest absent')
    for relative, digest in manifest['files'].items():
        target = frozen(root)/safe_relative(relative)
        if not target.is_file() or target.is_symlink() or sha(target) != digest:
            raise ValueError('Frozen scientific code/input changed: ' + relative)


def completed(root, task):
    marker = task['completion']
    value = read(root/safe_relative(marker['path']))
    if not isinstance(value, dict):
        return False
    for field in marker.get('field', 'status').split('.'):
        if not isinstance(value, dict) or field not in value:
            return False
        value = value[field]
    return type(value) is type(marker.get('equals', 'COMPLETE')) and value == marker.get('equals', 'COMPLETE')


def substitutions(root, value):
    return str(value).replace('{python}', sys.executable).replace('{frozen}', str(frozen(root))).replace('{root}', str(root))


def spawn(root, command, name, extra_env=None):
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED='1', ALL_ROUND_VALIDATION_ROOT=str(root),
                       PYTHONPATH=os.pathsep.join([str(frozen(root)/'scripts'), str(frozen(root)/'src')]))
    for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        environment[variable] = '2'
    environment.update({k: substitutions(root,v) for k,v in (extra_env or {}).items()})
    argv = [substitutions(root,item) for item in command]
    with (runtime(root)/(name+'.log')).open('ab', buffering=0) as stream:
        return subprocess.Popen(argv, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, close_fds=True,
                                start_new_session=True)


def own_command(root, option, *args):
    return [sys.executable, str(frozen(root)/SELF), '--root', str(root), option, *args]


def git(root, argv, env=None, data=None):
    return subprocess.run(['git', *argv], cwd=root, env=env, input=data,
                          text=True, capture_output=True, check=True)


def publish(root):
    """Commit via a dedicated index/ref; leave the user's HEAD and index intact."""
    state = read(runtime(root)/'state.json', {})
    branch = state['branch']
    if git(root,['branch','--show-current']).stdout.strip() != branch:
        raise ValueError('Publication deferred because the workspace branch changed')
    cfg = config(root)
    roots = cfg.get('publish_paths', [RELATIVE_OUTPUT, SELF, CONFIG])
    selected = set()
    for item in roots:
        relative = safe_relative(item)
        target = root/relative
        files = target.rglob('*') if target.is_dir() else [target]
        for path in files:
            rel = path.relative_to(root)
            if not path.is_file() or path.is_symlink() or any(p.startswith('.') or p=='__pycache__' for p in rel.parts):
                continue
            if path.name.endswith(('.log','.pid','.lock','.chk','.h5','.pyc','.tmp')) or '.tmp-' in path.name:
                continue
            if path.name in {'density.npy','dm.npy'} or any(part in {'density-cache','density_cache'} for part in rel.parts):
                continue
            if any(rel.match(pattern) for pattern in cfg.get('publish_exclude_globs', [])):
                continue
            selected.add(str(rel))
    ref = 'refs/background/all-round-validation-20261112'
    try:
        parent = git(root,['rev-parse','--verify',ref]).stdout.strip()
    except subprocess.CalledProcessError:
        parent = state['initial_head']
    env = os.environ.copy()
    env['GIT_INDEX_FILE'] = str(runtime(root)/'publication.index')
    git(root,['read-tree',parent],env)
    if selected:
        git(root,['add','--pathspec-from-file=-','--pathspec-file-nul'],env,'\0'.join(sorted(selected))+'\0')
    tree = git(root,['write-tree'],env).stdout.strip()
    oldtree = git(root,['rev-parse',parent+'^{tree}']).stdout.strip()
    commit = parent
    if tree != oldtree:
        commit = git(root,['commit-tree',tree,'-p',parent],env,
                     'Checkpoint all-round validation and CPU accounting\n').stdout.strip()
        git(root,['update-ref',ref,commit])
    git(root,['push','origin',commit+':refs/heads/'+branch])
    write(output(root)/'publication.json', {'status':'PUSHED','commit':commit,'branch':branch,
          'published_at_utc':utc(),'workspace_head_and_index_unchanged':True,'background_local_ref':ref,
          'rebuildable_density_cache_excluded':True,
          'missing_density_does_not_authorize_byte_identical_cache_fingerprint_reuse':True})


def publish_process(root):
    try:
        publish(root)
        return 0
    except Exception as error:
        write(output(root)/'publication.json', {'status':'RETRY_PENDING','error':repr(error),'at_utc':utc()})
        return 1


def halt_workers(state):
    for task in state.get('tasks', {}).values():
        signal_session(task.get('identity'))
    signal_session(state.get('publisher_identity'))


def guard(root):
    while True:
        state = read(runtime(root)/'state.json', {})
        if state.get('status') in TERMINAL:
            return 0
        ledger = sample_cpu(root)
        if ledger['conservative_cpu_seconds'] >= AUTHORIZED_CPU_SECONDS:
            write(runtime(root)/'budget_stop.json', {'status':'AUTHORIZED_CPU_BOUNDARY', 'at_utc':utc(),
                  'observed_cpu_seconds':ledger['observed_cpu_seconds'],
                  'conservative_cpu_seconds':ledger['conservative_cpu_seconds'], 'early_cpu_reserve_seconds':0})
            halt_workers(state)
        if not alive(state.get('supervisor_identity')):
            halt_workers(state)
            if (runtime(root)/'budget_stop.json').exists() or (runtime(root)/'STOP.json').exists():
                state.update(status='AUTHORIZED_CPU_BOUNDARY' if (runtime(root)/'budget_stop.json').exists() else 'STOPPED_BY_USER')
                write(runtime(root)/'state.json',state)
                return 0
            try:
                check_integrity(root)
                replacement = spawn(root, own_command(root,'--supervise'),'supervisor')
                write(runtime(root)/'recovery.json', {'previous_supervisor':state.get('supervisor_identity'),
                      'replacement_identity':identity(replacement.pid),'at_utc':utc(),'automatic_retry_limit':None})
                return 0
            except Exception as error:
                write(runtime(root)/'guard_retry.json', {'error':repr(error),'at_utc':utc()})
        time.sleep(POLL_SECONDS)


def supervise(root):
    runtime(root).mkdir(parents=True,exist_ok=True)
    lock = (runtime(root)/'supervisor.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        return 0
    old = read(runtime(root)/'state.json',{})
    if old.get('status') in TERMINAL:
        return 0
    halt_workers(old)
    signal_session(old.get('guard_identity'))
    check_integrity(root)
    cfg = config(root)
    state = {**old,'status':'RUNNING','supervisor_identity':identity(os.getpid()),
             'guard_identity':None,'publisher_identity':None,'tasks':{},'heartbeat_at_utc':utc(),
             'authorized_cpu_seconds':AUTHORIZED_CPU_SECONDS,'no_wall_score_failure_idle_exit':True,
             'cloud_lifetime_guaranteed':False}
    write(runtime(root)/'state.json',state)
    children = {}
    for task in cfg['tasks']:
        previous = old.get('tasks',{}).get(task['id'],{})
        state['tasks'][task['id']] = {'identity':None,'attempts':previous.get('attempts',0),
                                    'next_launch_monotonic':0.0,'status':'PENDING'}
    guard_child = spawn(root, own_command(root,'--guard'),'guard')
    state['guard_identity'] = identity(guard_child.pid)
    publisher = None
    next_publish = time.monotonic()+PUBLISH_SECONDS
    next_integrity = 0.0
    final_publish = False
    publisher_is_final = False
    signal_received = False

    def on_signal(signum, frame):
        nonlocal signal_received
        signal_received = True
    signal.signal(signal.SIGTERM,on_signal)
    signal.signal(signal.SIGINT,on_signal)
    try:
        while True:
            now = time.monotonic()
            ledger = sample_cpu(root)
            if ledger['conservative_cpu_seconds'] >= AUTHORIZED_CPU_SECONDS or (runtime(root)/'budget_stop.json').exists():
                state['status']='AUTHORIZED_CPU_BOUNDARY'
                break
            if (runtime(root)/'STOP.json').exists():
                state['status']='STOPPED_BY_USER'
                break
            if signal_received:
                state['status']='SUPERVISOR_RESTART_PENDING'
                break
            if now >= next_integrity:
                check_integrity(root)
                next_integrity=now+30
            if guard_child.poll() is not None:
                guard_child.wait()
                guard_child = spawn(root, own_command(root,'--guard'),'guard')
                state['guard_identity']=identity(guard_child.pid)
            for task in cfg['tasks']:
                item = state['tasks'][task['id']]
                child = children.get(task['id'])
                if child is not None and child.poll() is not None:
                    item.update(identity=None,last_returncode=child.wait(),last_finished_at_utc=utc(),
                                next_launch_monotonic=now+RETRY_SECONDS,status='RETRY_PENDING')
                    del children[task['id']]
                    child=None
                if completed(root,task):
                    # Explicit completion is assessed only when no live worker
                    # can still be constructing its final scientific receipts.
                    if child is None:
                        item['status']='COMPLETE'
                    continue
                if child is None and now >= item['next_launch_monotonic']:
                    try:
                        child = spawn(root,task['command'],task['id'],task.get('env'))
                        children[task['id']]=child
                        item.update(identity=identity(child.pid),status='RUNNING',
                                    attempts=item['attempts']+1,last_started_at_utc=utc())
                    except OSError as error:
                        item.update(status='RETRY_PENDING',last_error=repr(error),next_launch_monotonic=now+RETRY_SECONDS)
            all_done = all(item.get('status')=='COMPLETE' for item in state['tasks'].values())
            if all_done and not final_publish:
                final_publish=True
                next_publish=now
            if publisher is not None and publisher.poll() is not None:
                code=publisher.wait()
                publisher=None
                state['publisher_identity']=None
                state['last_publication_returncode']=code
                if final_publish and publisher_is_final and all_done and code==0:
                    state['status']='COMPLETE'
                    break
                if final_publish:
                    next_publish=now+RETRY_SECONDS
            if publisher is None and now >= next_publish:
                publisher=spawn(root,own_command(root,'--publish'),'publication')
                publisher_is_final=final_publish
                state['publisher_identity']=identity(publisher.pid)
                next_publish=now+PUBLISH_SECONDS
            state['heartbeat_at_utc']=utc()
            write(runtime(root)/'state.json',state)
            write(output(root)/'status.json',{k:v for k,v in state.items() if k not in {'root'}})
            time.sleep(POLL_SECONDS)
    except Exception as error:
        try:
            check_integrity(root)
        except Exception:
            state.update(status='SCIENTIFIC_INTEGRITY_CHANGED',last_error=repr(error))
        else:
            state.update(status='SUPERVISOR_RESTART_PENDING',last_error=repr(error))
    finally:
        halt_workers(state)
        for child in children.values():
            child.wait()
        if publisher is not None:
            publisher.wait()
        state.update(heartbeat_at_utc=utc(),finished_at_utc=utc())
        write(runtime(root)/'state.json',state)
        write(output(root)/'status.json',state)
        sample_cpu(root)
        # A nonterminal failure leaves the guard alive to replace us. Source
        # corruption and explicit terminal causes are never silently retried.
        if state['status'] in TERMINAL:
            signal_session(state.get('guard_identity'))
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()
    return 0 if state['status'] in TERMINAL else 1


def status(root):
    state=read(runtime(root)/'state.json',{})
    return {'state':state,'supervisor_alive':alive(state.get('supervisor_identity')),
            'guard_alive':alive(state.get('guard_identity')),
            'tasks_alive':{key:alive(item.get('identity')) for key,item in state.get('tasks',{}).items()},
            'cpu_accounting':read(output(root)/'cpu_accounting.json',{}),
            'cloud_lifetime_guaranteed':False}


def start(root):
    runtime(root).mkdir(parents=True,exist_ok=True)
    with (runtime(root)/'launch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        previous=status(root)
        if previous['supervisor_alive'] or previous['state'].get('status') in TERMINAL:
            return previous
        if (runtime(root)/'STOP.json').exists():
            raise ValueError('Explicit stop marker prevents automatic start')
        freeze(root)
        config(root)
        ledger=sample_cpu(root)
        if ledger['conservative_cpu_seconds'] >= AUTHORIZED_CPU_SECONDS:
            raise ValueError('The cumulative task authorization is already exhausted')
        head=git(root,['rev-parse','HEAD']).stdout.strip()
        branch=git(root,['branch','--show-current']).stdout.strip()
        if not branch or branch=='main':
            raise ValueError('A dedicated named research branch is required')
        state=previous['state']
        state.update(root=str(root),branch=state.get('branch',branch),initial_head=state.get('initial_head',head),
                     status='STARTING',authorization_date='2026-11-12',authorization_cpu_seconds=AUTHORIZED_CPU_SECONDS,
                     authorization_scope='New task grant; historical jobs and ledgers are preserved separately')
        write(runtime(root)/'state.json',state)
        child=spawn(root,own_command(root,'--supervise'),'supervisor')
        # The handshake is an observation window, never a worker timeout.
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            value=status(root)
            if value['supervisor_alive'] and value['guard_alive']:
                return value
            if child.poll() is not None:
                return {**value,'start_returncode':child.returncode}
            time.sleep(.1)
        return {**status(root),'launch_handshake':'PENDING','launched_pid':child.pid}


def verify(root):
    value=status(root)
    try:
        check_integrity(root)
        integrity=True
    except Exception as error:
        integrity=False
        value['source_integrity_error']=repr(error)
    identities=[value['state'].get(key) for key in ('supervisor_identity','guard_identity','publisher_identity')]
    identities.extend(item.get('identity') for item in value['state'].get('tasks',{}).values())
    session_checks={str(item['pid']):bool((point:=proc_stat(item['pid'])) and point['pgrp']==point['pid'] and point['sid']==point['pid'])
                    for item in identities if alive(item)}
    detached=all(session_checks.values())
    value['verified']=bool(integrity and detached and ((value['supervisor_alive'] and value['guard_alive']) or value['state'].get('status')=='COMPLETE'))
    value['source_integrity_passed']=integrity
    value['live_processes_own_independent_sessions']=session_checks
    value['session_isolation_passed']=detached
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(os.environ.get('ALL_ROUND_VALIDATION_ROOT',Path(__file__).resolve().parents[1])))
    options=parser.add_mutually_exclusive_group(required=True)
    for flag in ('start','verify','status','supervise','guard','publish','stop'):
        options.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    if args.supervise:
        return supervise(root)
    if args.guard:
        return guard(root)
    if args.publish:
        return publish_process(root)
    if args.stop:
        write(runtime(root)/'STOP.json',{'reason':'Explicit user stop','at_utc':utc()})
        value=status(root)
    elif args.start:
        value=start(root)
    elif args.verify:
        value=verify(root)
    else:
        value=status(root)
    print(json.dumps(value,indent=2,ensure_ascii=False))
    return 1 if args.verify and not value['verified'] else 0


if __name__=='__main__':
    raise SystemExit(main())
