#!/usr/bin/env python3
"""Start or inspect two isolated, detached local Spinosin/IR compute lanes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from run_spinosin_ir_gap_20261112 import CASES, atomic_json, read_json, sha

ROOT = Path(__file__).resolve().parents[1]
WORKER = 'run_spinosin_ir_gap_20261112.py'


def process_identity(pid):
    try:
        pid = int(pid)
        if pid <= 0:
            return None
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': int(fields[19]), 'pgid': os.getpgid(pid),
                'command': Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode()}
    except (OSError, ValueError, TypeError):
        return None


def matches(pid, terms, expected_start_ticks=None):
    identity = process_identity(pid)
    return bool(identity and all(term in identity['command'] for term in terms)
                and (expected_start_ticks is None or identity['start_ticks'] == expected_start_ticks))


def affinities():
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
    if len(allowed) >= 4:
        return {CASES[0]: allowed[:2], CASES[1]: allowed[2:4]}
    # A smaller executor can still resume safely; record actual shared placement.
    return {name: allowed[:2] for name in CASES}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=Path('reports/spinosin_ir_gap_fill_20261112'))
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    args = parser.parse_args(argv)
    if len(set(args.cases)) != len(args.cases):
        parser.error('Conformer cases must be distinct')
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    ledger_path = root / 'data/research/meta_learning_20261111/compute_authorization_addenda.json'
    authorization = read_json(ledger_path, {}).get('total_authorized_cpu_hours')
    old_launch = read_json(output / 'launch_manifest.json', {})
    old_by_case = {row['case_id']: row for row in old_launch.get('launches', [])}
    if args.action == 'status':
        rows = []
        for name in args.cases:
            runtime = output / 'supervisors' / name
            state = read_json(runtime / 'status.json', {})
            execution = read_json(output / 'runs' / name / 'execution_state.json', {})
            birth = (old_by_case.get(name, {}).get('supervisor_identity') or {}).get('start_ticks')
            rows.append({'case_id': name, 'supervisor_pid': state.get('supervisor_pid'),
                'supervisor_alive': matches(state.get('supervisor_pid'), ['background_supervisor.py', str(runtime)], birth),
                'worker_alive': matches(state.get('worker_pid'), [WORKER, '--case', name]),
                'worker_pid': state.get('worker_pid'), 'manual_STOP_present': (runtime / 'STOP').exists(),
                'supervisor_heartbeat_at_utc': state.get('heartbeat_at'),
                'execution': execution, 'shared_authorized_cpu_hours': authorization,
                'completion_receipt_present': (output / 'runs' / name / 'ops/workflow_complete.json').exists(),
                'log_path': state.get('log_path')})
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'manager.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        python = root / '.venv-quantum-gap-20261112/bin/python'
        if not python.is_file():
            raise RuntimeError('The isolated PySCF/geomeTRIC backend must be prepared before launch')
        placement = affinities()
        environment = os.environ.copy()
        environment['PYTHONUNBUFFERED'] = '1'
        environment['OMP_MAX_ACTIVE_LEVELS'] = '1'
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            environment[key] = '2'
        environment['PYTHONPATH'] = os.pathsep.join([str(root / 'scripts'), environment.get('PYTHONPATH', '')])
        launches = []
        for name in args.cases:
            runtime = output / 'supervisors' / name
            runtime.mkdir(parents=True, exist_ok=True)
            if (runtime / 'STOP').exists():
                raise RuntimeError('An explicit STOP remains present: ' + str(runtime / 'STOP'))
            old = read_json(runtime / 'status.json', {})
            birth = (old_by_case.get(name, {}).get('supervisor_identity') or {}).get('start_ticks')
            if matches(old.get('supervisor_pid'), ['background_supervisor.py', str(runtime)], birth):
                launches.append({'case_id': name, 'status': 'ALREADY_RUNNING',
                    'supervisor_identity': process_identity(old['supervisor_pid'])})
                continue
            if matches(old.get('worker_pid'), [WORKER, '--case', name]):
                raise RuntimeError(f'{name}: live orphaned worker found; refusing duplicate molecular calculation')
            worker = [str(python), '-u', str(root / 'scripts' / WORKER), '--root', str(root),
                '--case', name, '--output', str(output / 'runs' / name), '--threads', '2', '--max-memory-mb', '3000']
            if placement[name]:
                worker += ['--cpu-affinity', *map(str, placement[name])]
            prepared = subprocess.run([*worker, '--prepare'], cwd=root, env=environment,
                                      text=True, capture_output=True)
            if prepared.returncode:
                raise RuntimeError('Scientific preflight refused startup: ' + prepared.stdout + prepared.stderr)
            protocol_path = output / 'runs' / name / 'ops/frozen_protocol.json'
            protocol = read_json(protocol_path)
            if not protocol:
                raise RuntimeError('Scientific preflight did not produce its frozen protocol')
            command = [str(python), '-u', str(root / 'scripts/background_supervisor.py'),
                '--cwd', str(root), '--state-dir', str(runtime), '--', *worker]
            with (runtime / 'launcher.log').open('ab', buffering=0) as log:
                process = subprocess.Popen(command, cwd=root, env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            launches.append({'case_id': name, 'status': 'LAUNCHED', 'supervisor_pid': process.pid,
                'supervisor_identity': process_identity(process.pid), 'command': command,
                'threads': 2, 'max_memory_mb': 3000, 'cpu_affinity': placement[name],
                'frozen_protocol_sha256': sha(protocol_path), 'input_hash': protocol['input_hash']})
        receipt = {'schema': 'local-spinosin-ir-gap-launch-v1',
            'launched_at_utc': datetime.now(timezone.utc).isoformat(), 'launch_id': uuid.uuid4().hex,
            'latest_additional_cpu_seconds_authorization': 900000,
            'shared_authorized_cpu_hours': authorization, 'authorization_ledger': str(ledger_path),
            'actual_reserved_cpu_hours': None, 'threads_per_lane': 2, 'max_memory_mb_per_lane': 3000,
            'cpu_wall_quality_retry_round_stop_limit': None,
            'persistence_scope': 'Current cloud host lifetime; no platform uptime guarantee',
            'completed_behavior': 'Remain live and idle; unresolved scientific qualification waits explicitly for a new protocol without recomputing cached work',
            'launches': launches}
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        atomic_json(output / 'launch_receipts' / f'{stamp}-{receipt["launch_id"]}.json', receipt)
        atomic_json(output / 'launch_manifest.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
