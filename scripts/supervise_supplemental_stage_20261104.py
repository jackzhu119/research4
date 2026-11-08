"""Bound one supplemental ML stage; preserve wait4 cost across source/predict/score.

Uses the unchanged, previously tested owned-child monitor from the earlier ML
pilot. This guard never starts quantum calculations or changes their ledger.
"""
import argparse
import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/workspace/research4')
BASE = ORIGINAL/'scripts/run_conservative_cross_plant_pilot_20261104_v2.py'
BASE_SHA = 'd7eb20ac813185f777de8cd4ab63b360bbbecead6cf7219790a6df4e4209f78f'
PROTOCOL = 'data/splits/supplemental_spectra_20261104_v1/bound_protocol.json'
PROTOCOL_SHA = '981afb628b4bc7f6ac910d0ff2b9d8d9b74fa0117fb284e61ab6e813429ab02a'
RESOURCE_DIR = ROOT/'reports/supplemental-resources-20261104-v1'
OUTPUT = 'reports/supplemental-spectra-benchmark-20261104-v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, x):
    temporary = p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
    temporary.replace(p)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('source', 'predict', 'score'))
    parser.add_argument('--release')
    a = parser.parse_args()
    assert sha(BASE) == BASE_SHA and sha(ROOT/PROTOCOL) == PROTOCOL_SHA
    assert not any(p.get('status') == 'RUNNING' for p in json.loads(
        (ORIGINAL/'reports/pyscf-spinosin-resumable-20261104-v5/cpu_budget.json').read_text())['phases']), \
        'Complete auxiliary work before restarting quantum execution'
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = RESOURCE_DIR/'ledger.json'
    with (RESOURCE_DIR/'.owned.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
        else:
            ledger = {'schema':'research4-supplemental-actual-resource-v1',
                      'created_utc':now(), 'started_epoch':time.time(),
                      'cpu_cap_seconds':6000, 'wall_cap_seconds':14400,
                      'actual_closed_CPU_seconds':0., 'phases':[],
                      'quantum_budget_modified':False}
        assert all(p['status'] != 'RUNNING' and p.get('actual_CPU_seconds') is not None
                   for p in ledger['phases']), 'Unresolved accounting blocks new stages'
        assert not any(p['stage'] == a.stage for p in ledger['phases']), 'Preserve prior stage'
        remaining = 6000-ledger['actual_closed_CPU_seconds']
        wall = 14400-(time.time()-ledger['started_epoch'])
        # The frozen learner checks 5880 CPU / 14280 wall at fit boundaries.
        # Keep its planned 120-second cleanup window inside the global bounds;
        # do not terminate it before it can seal complete source tasks.
        cap = math.floor(remaining-1)
        wall = math.floor(wall)
        assert cap >= 12 and wall >= 8, 'Finite remaining resource budget insufficient'
        code = ast.parse(BASE.read_text())
        functions = [node for node in code.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'monitor_owned_child']
        assert len(functions) == 1
        namespace = {'os':os, 'signal':signal, 'time':time, 'Path':Path,
                     'CPU_CAP':cap, 'RSS_CAP':int(1.5*1024**3), 'WALL_CAP':wall}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(BASE), 'exec'), namespace)
        command = [str(ORIGINAL/'.runtime/bin/python'),
                   'scripts/run_supplemental_benchmark_20261104.py', '--protocol', PROTOCOL,
                   '--stage', a.stage, '--output', OUTPUT]
        if a.release:
            command += ['--release', a.release]
        environment = {**os.environ, 'PYTHONPATH':f'src:{ORIGINAL}/.runtime/quantum-pilot-site:{ORIGINAL}/.runtime/quantum-upgrade-site'}
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            environment[key] = '1'
        def limits():
            os.setsid()
            resource.setrlimit(resource.RLIMIT_CPU, (cap-2, cap))
            resource.setrlimit(resource.RLIMIT_AS, (int(1.5*1024**3),)*2)
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        log_path = RESOURCE_DIR/f'{a.stage}.log'
        phase = {'stage':a.stage, 'status':'RUNNING', 'started_utc':now(),
                 'command':command, 'CPU_grant_seconds':cap, 'wall_grant_seconds':wall,
                 'RSS_best_effort_guard_bytes':int(1.5*1024**3),
                 'AS_hard_limit_bytes':int(1.5*1024**3), 'threads':1,
                 'source_cooperative_CPU_gate_seconds':5880 if a.stage == 'source' else None,
                 'source_cooperative_wall_gate_seconds':14280 if a.stage == 'source' else None,
                 'monitor_source_sha256':BASE_SHA, 'guard_source_sha256':sha(__file__),
                 'protocol_sha256':PROTOCOL_SHA, 'release_sha256':sha(ROOT/a.release) if a.release else None}
        ledger['phases'].append(phase)
        save(ledger_path, ledger)
        start = time.monotonic()
        with log_path.open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT, preexec_fn=limits)
            phase['pid'] = child.pid
            save(ledger_path, ledger)
            receipt = namespace['monitor_owned_child'](child, start, 0.)
        usage = receipt.pop('usage')
        phase.update(receipt, ended_utc=now(), wall_seconds=time.monotonic()-start,
                     log_sha256=sha(log_path))
        if usage is None:
            phase.update(status='UNKNOWN_ACTUAL_CPU_BLOCKED', actual_CPU_seconds=None)
            save(ledger_path, ledger)
            raise RuntimeError('Unknown final wait4 cost; no automatic retries or scoring')
        actual = usage.ru_utime+usage.ru_stime
        phase.update(actual_CPU_seconds=actual, wait4_peak_RSS_KiB=usage.ru_maxrss,
                     status='DONE' if receipt['exit_code'] == 0 and receipt['stop_reason'] is None else 'PARTIAL_OR_FAILED')
        ledger['actual_closed_CPU_seconds'] += actual
        save(ledger_path, ledger)
        print(json.dumps(phase), flush=True)
        if phase['status'] != 'DONE':
            raise SystemExit(2)


if __name__ == '__main__':
    main()
