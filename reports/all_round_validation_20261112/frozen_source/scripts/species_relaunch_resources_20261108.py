#!/usr/bin/env python3
"""Finite resource envelope around the unchanged cumulative species supervisor.

Default is read-only validation. Root detaches --execute with Popen/setsid;
this process records its real identity before launching the reviewed supervisor.
Old grants and unclean states are never reset or inferred closed from a reboot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
BASE = ROOT / 'data/research/species_relaunch_20261108_v2'
SUPERVISOR = ROOT / 'scripts/species_background_budget.py'
SUPERVISOR_SHA = '3ca443016719c90697ed9e7f82db2a6205cdb8f07b1f78ad31db117869667874'
SCHEMA = 'species-relaunch-resources-v1'
TOKEN = 'SPECIES_RELAUNCH_OUTER_TOKEN'
LATEST_DEADLINE = datetime.fromisoformat('2026-11-10T02:50:37.289380+00:00')
CPU_CLEANUP_RESERVE = 64.


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def checked(ref):
    p = Path(ref['path']).resolve()
    need(p.is_relative_to(ROOT) and p.is_file() and sha(p) == ref['sha256'], 'Changed/missing pinned input: '+str(p))
    return p


def read(path):
    return json.loads(Path(path).read_text())


def load_supervisor():
    need(sha(SUPERVISOR) == SUPERVISOR_SHA, 'Unchanged reviewed inner supervisor required')
    spec = importlib.util.spec_from_file_location('relaunch_original_supervisor', SUPERVISOR)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # The helper instance observes our separate inherited nonce. The real inner
    # process imports the untouched source and retains its original job nonce.
    module.TOKEN_KEY = TOKEN
    return module


def limits(config):
    need(config['total_budget_cpu_seconds'] == 90000 and config['inner_budget_cpu_seconds'] == 88000
         and config['prep_budget_cpu_seconds'] == 1900 and config['controller_budget_cpu_seconds'] == 100,
         'The new grant is 88000 inner +1900 prep +100 controller, never renewed per job')
    preparation = float(config['preparation_cpu_seconds'])
    need(math.isfinite(preparation) and 0 <= preparation <= 1900, 'Measured preparation exceeds its allocation')
    need(0 < config['wall_cap_seconds'] <= 36*3600, 'At most36h aggregate wall time')
    end = datetime.fromisoformat(config['deadline_utc'])
    need(end.tzinfo is not None and end <= LATEST_DEADLINE, 'Deadline may not extend the authorization')
    need(config['rss_stop_bytes'] <= 10*1024**3 and config['rss_stop_bytes'] > 0
         and 0 < config['cgroup_stop_bytes'] <= 12*1024**3
         and config['minimum_disk_bytes'] >= 1024**3, 'Finite memory and disk safety floors required')
    return preparation, end


def validate(config, *, resume=False):
    need(config.get('schema') == SCHEMA and config['wrapper_source_sha256'] == sha(SELF), 'Exact wrapper source required')
    need(config['run_id']=='species-relaunch-20261108-v2', 'Exact new user-authorized run ID required')
    need(checked(config['supervisor_ref']) == SUPERVISOR and config['supervisor_ref']['sha256'] == SUPERVISOR_SHA,
         'Only the unchanged reviewed supervisor is allowed')
    manifest = checked(config['manifest_ref']); prep = read(checked(config['preparation_ref']))
    measured = prep.get('spent_cpu_seconds', prep.get('consumed_cpu_seconds', prep.get('actual_cpu_seconds', prep.get('preparation_cpu_seconds'))))
    need(measured == config['preparation_cpu_seconds'], 'Preparation measurement/reference differs')
    preparation, end = limits(config); data = read(manifest)
    need(data['cpu_budget_seconds'] == 88000 and data['run_id'] == config['run_id'], 'New manifest/grant identity differs')
    for name in ('state_dir', 'workroot'):
        p = Path(config[name]).resolve(); need(p.is_relative_to(BASE) and p != BASE, 'Distinct new state paths required')
    need(Path(config['state_dir']).resolve() != Path(config['workroot']).resolve(), 'Inner and outer state must be separate')
    need(Path(config['python']).is_absolute() and Path(config['python']).is_file(), 'Explicit installed Python required')
    inner = Path(config['state_dir'])/'state.json'; outer = Path(config['workroot'])/'state.json'
    need(resume == outer.exists(), 'Fresh run versus clean resume must be explicit')
    if resume:
        old = read(outer)
        need(old.get('active') is None and old.get('closure_confirmed') is True, 'Unclean outer identity/cost must be reconciled by root')
        need(old.get('status')=='PAUSED_RESOURCE', 'Resume only a genuinely clean resource pause, never burn completed jobs')
        need(old['config_sha256'] == config['_config_sha256'], 'Frozen resume config differs')
        for key in ('controller_spent_cpu_seconds','inner_actual_spent_cpu_seconds','wall_seconds'):
            need(key in old and isinstance(old[key], (int,float)) and math.isfinite(old[key]) and old[key]>=0,
                 'Missing/nonfinite historical resource cost: '+key)
        need(inner.exists() and read(inner).get('active') is None, 'Unclean inner job cannot be adopted')
    else:
        need(not inner.exists(), 'Do not copy, reinitialize or refund a previous inner state')
    need(datetime.now(timezone.utc) < end, 'Authorization deadline reached')
    return manifest, preparation, end


def usage(kind):
    u = resource.getrusage(kind); return u.ru_utime + u.ru_stime


def resource_reason(controller, inner, total, wall, cfg, *, rss, cgroup, free_disk, now):
    if controller >= cfg['controller_budget_cpu_seconds']-10: return 'controller_cpu_allocation'
    if inner >= cfg['inner_budget_cpu_seconds']-CPU_CLEANUP_RESERVE: return 'inner_actual_cpu_allocation'
    if total >= cfg['total_budget_cpu_seconds']-CPU_CLEANUP_RESERVE: return 'total_cpu_allocation'
    if wall >= cfg['wall_cap_seconds'] or now >= datetime.fromisoformat(cfg['deadline_utc']): return 'wall_or_deadline'
    if rss >= cfg['rss_stop_bytes'] or cgroup >= cfg['cgroup_stop_bytes']: return 'memory_limit'
    if free_disk < cfg['minimum_disk_bytes']: return 'disk_floor'
    return None


def run(config, config_path, b, *, resume):
    manifest, preparation, end = validate(config, resume=resume)
    work = Path(config['workroot']); work.mkdir(parents=True, exist_ok=True)
    statefile = work/'state.json'; innerfile = Path(config['state_dir'])/'state.json'
    with (work/'outer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A caller may have passed the first validation before another fresh
        # invocation finished. Recheck under the lock before reading any base.
        validate(config, resume=resume)
        old = read(statefile) if resume else {}
        controller_base = float(old.get('controller_spent_cpu_seconds', 0))
        inner_base = float(old.get('inner_actual_spent_cpu_seconds', 0)); wall_base = float(old.get('wall_seconds', 0))
        start_wall = time.monotonic(); child_start = usage(resource.RUSAGE_CHILDREN)
        controller_high, inner_high = controller_base, inner_base
        identity = b.proc_stat(os.getpid()); need(identity is not None, 'Actual wrapper identity unavailable')
        identity = {**identity, 'uid':os.getuid(), 'boot_id':b.boot_id()}
        attempt = str(uuid.uuid4()); receipt = work/('launch-'+attempt+'.json')
        b.atomic_json(receipt, {'status':'ACTUAL_RESOURCE_WRAPPER_STARTED_NOT_RESEARCH_COMPLETED','identity':identity,
            'source_sha256':sha(SELF),'config_sha256':config['_config_sha256'],'at_utc':datetime.now(timezone.utc).isoformat()})
        state = {'schema':SCHEMA,'config_sha256':config['_config_sha256'],'identity':identity,'run_id':config['run_id'],
            'status':'STARTING','active':None,'closure_confirmed':False,'preparation_cpu_seconds':preparation,
            'old_grants':'UNRECONCILED_NO_REFUND_NO_CARRY','attempt':attempt,'resource_monitor_is_best_effort':True}
        command = [config['python'], '-B', str(SUPERVISOR), 'resume' if resume else 'run',
            '--manifest',str(manifest),'--state-dir',config['state_dir'],'--budget-cpu-seconds','88000']
        nonce = uuid.uuid4().hex; env = dict(os.environ); env[TOKEN] = nonce; env.update(b.THREAD_ENV)
        state['launch_token'] = nonce
        stop = None; stop_at = None; term_sent = False; kill_sent = False; active = None; process = None
        handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda number, frame: state.setdefault('external_stop', signal.Signals(number).name))

        def members():
            result = b.owned_members(active) if active else []
            for row in result:
                try: need(Path('/proc',str(row['pid'])).stat().st_uid == os.getuid(), 'Owned identity changed UID')
                except FileNotFoundError: pass
            return result

        def sample(force=False):
            nonlocal controller_high, inner_high
            owned = members(); cpu = sum(x['cpu_seconds'] for x in owned)
            controller_high = max(controller_high, controller_base + usage(resource.RUSAGE_SELF))
            inner_high = max(inner_high, inner_base + max(0.,usage(resource.RUSAGE_CHILDREN)-child_start) + cpu)
            total = preparation + controller_high + inner_high
            rss = 0
            for row in [identity, *owned]:
                try:
                    for line in Path('/proc',str(row['pid']),'status').read_text().splitlines():
                        if line.startswith('VmRSS:'): rss += int(line.split()[1])*1024
                except FileNotFoundError: pass
            cg = int(Path('/sys/fs/cgroup/memory.current').read_text())
            free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
            wall = wall_base + time.monotonic()-start_wall
            state.update(controller_spent_cpu_seconds=controller_high,inner_actual_spent_cpu_seconds=inner_high,
                spent_cpu_seconds=total,remaining_cpu_seconds=max(0.,90000-total),wall_seconds=wall,
                active=active,heartbeat_utc=datetime.now(timezone.utc).isoformat(),rss_bytes=rss,cgroup_memory_bytes=cg,free_disk_bytes=free)
            b.atomic_json(statefile,state)
            return owned, resource_reason(controller_high,inner_high,total,wall,config,rss=rss,cgroup=cg,free_disk=free,now=datetime.now(timezone.utc))

        try:
            # Persist an unclean STARTING boundary before any child exists.
            # A crash in the Popen/identity-observation window must not permit
            # a second fresh run to adopt or duplicate an unknown process.
            _, initial_reason = sample()
            need(initial_reason is None, 'Prelaunch resource limit: '+str(initial_reason))
            console = work/('inner-'+attempt+'.log')
            with console.open('x') as log:
                process = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
                state['provisional_child_pid'] = process.pid
                b.atomic_json(statefile,state)
                child = b.proc_stat(process.pid)
                need(child is not None, 'Inner supervisor exited before real identity was observed')
                active = {k:child[k] for k in ('pid','start_ticks','pgrp','session')}
                active.update(boot_id=b.boot_id(),uid=os.getuid(),token=nonce)
                state.update(status='RUNNING',actual_command=command,active=active)
                while True:
                    code = process.poll(); owned, why = sample()
                    alive = [m for m in owned if m['state'] != 'Z']
                    if code is not None and not alive: break
                    if state.get('external_stop'): why = 'external_'+state['external_stop']
                    if sha(config_path) != config['_config_sha256'] or sha(SELF) != config['wrapper_source_sha256']:
                        why = 'wrapper_or_config_drift'
                    if code is not None and alive: why = 'owned_orphans_after_supervisor_exit'
                    if why and stop is None:
                        stop, stop_at = why, time.monotonic(); state.update(status='STOP_REQUESTED',stop_reason=why)
                    if stop:
                        if innerfile.exists():
                            inner = read(innerfile)
                            b.atomic_json(Path(config['state_dir'])/'stop_requested.json',
                                {'run_id':inner['run_id'],'requested_unix':time.time(),'outer_resource_reason':stop})
                        elapsed = time.monotonic()-stop_at
                        if elapsed >= 2 and not term_sent:
                            b.signal_owned_groups(active,signal.SIGTERM); term_sent=True
                        if elapsed >= 4 and not kill_sent:
                            b.signal_owned_groups(active,signal.SIGKILL); kill_sent=True
                        need(elapsed < 8, 'Verified owned processes did not close; root reconciliation required')
                    time.sleep(1)
                process.wait(); sample()
                active = None
                state.update(active=None,closure_confirmed=True,inner_exit_code=process.returncode,
                    status='PAUSED_RESOURCE' if stop else ('COMPLETED' if process.returncode==0 else 'INNER_FINISHED_WITH_ERRORS'))
        except BaseException as exc:
            state.update(status='BLOCKED_REQUIRES_ROOT_RECONCILIATION',error_type=type(exc).__name__,error=str(exc))
            # Fail closed on a monitor exception, retaining costs and identity.
            # No catch-all completion or automatic retry is permitted.
            if process is not None and active is not None:
                try:
                    if innerfile.exists():
                        inner=read(innerfile)
                        b.atomic_json(Path(config['state_dir'])/'stop_requested.json',
                            {'run_id':inner['run_id'],'requested_unix':time.time(),'outer_resource_reason':'monitor_exception'})
                    b.signal_owned_groups(active,signal.SIGTERM)
                    until=time.monotonic()+2
                    while time.monotonic()<until and any(m['state']!='Z' for m in members()):
                        process.poll();sample();time.sleep(.1)
                    if any(m['state']!='Z' for m in members()): b.signal_owned_groups(active,signal.SIGKILL)
                    process.wait(timeout=3)
                    if not any(m['state']!='Z' for m in members()):
                        sample();active=None;state.update(active=None,closure_confirmed=True)
                except BaseException as cleanup_error:
                    state['cleanup_error']=type(cleanup_error).__name__+': '+str(cleanup_error)
            elif process is not None:
                # Immediate child exit can precede /proc identity observation.
                # Reap when possible to retain CPU, but never signal a PID that
                # has not passed birth/boot ownership validation.
                try:
                    process.wait(timeout=.1)
                    state['provisional_child_exit_code']=process.returncode
                except subprocess.TimeoutExpired:
                    state['unverified_child_state']='UNKNOWN_NO_SIGNAL_NO_AUTORESTART'
        finally:
            try: sample()
            except BaseException as error: state['final_sample_error']=type(error).__name__+': '+str(error)
            state['updated_at_utc']=datetime.now(timezone.utc).isoformat();b.atomic_json(statefile,state)
            for sig,handler in handlers.items(): signal.signal(sig,handler)
        return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();need(sha(args.config)==args.config_sha256,'Changed wrapper config')
    config=read(args.config);config['_config_sha256']=args.config_sha256
    b=load_supervisor();validate(config,resume=args.resume)
    if not args.execute:
        print(json.dumps({'status':'READ_ONLY_PREFLIGHT_NOT_STARTED','new_total_CPU':90000,'inner_CPU':88000}));return
    result=run(config,args.config,b,resume=args.resume)
    print(json.dumps({'status':result['status'],'spent_cpu_seconds':result['spent_cpu_seconds']}))


if __name__=='__main__':main()
