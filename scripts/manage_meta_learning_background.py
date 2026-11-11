#!/usr/bin/env python3
"""Start or inspect three detached TRAIN-only representation workers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def process_alive(pid, marker):
    if not isinstance(pid, int):
        return False
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode()
        return marker in command
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('reports/meta_learning_continuous_20261111'))
    parser.add_argument('--representations', nargs='+', choices=('alpha', 'n', 'fusion'),
                        default=['alpha', 'n', 'fusion'])
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output.is_absolute() else root / args.output
    if len(set(args.representations)) != len(args.representations):
        parser.error('duplicate representations')
    if args.action == 'start':
        output.mkdir(parents=True, exist_ok=True)
        feature_cache = root / 'reports/b_mechanism_20261110/.cache/features.npz'
        if not feature_cache.is_file():
            feature_cache = output / '.cache/features.npz'
        environment = os.environ.copy()
        for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                         'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
            environment[variable] = '1'
        environment['PYTHONPATH'] = os.pathsep.join(
            [str(root / 'src'), str(root / 'scripts'), environment.get('PYTHONPATH', '')])
        if not feature_cache.is_file():
            subprocess.run([sys.executable, str(root / 'scripts/prepare_species_revision_features.py'),
                            '--root', str(root), '--output', str(feature_cache.parent)],
                           cwd=root, env=environment, check=True)
        launches = []
        for name in args.representations:
            supervisor_dir = output / 'supervisors' / name
            supervisor_dir.mkdir(parents=True, exist_ok=True)
            if (supervisor_dir / 'STOP').exists():
                raise RuntimeError(f'Explicit STOP exists at {supervisor_dir}; remove it deliberately to resume')
            prior = read_json(supervisor_dir / 'status.json')
            if process_alive(prior.get('supervisor_pid'), 'background_supervisor.py'):
                launches.append({'representation': name, 'status': 'ALREADY_RUNNING',
                                 'supervisor_pid': prior['supervisor_pid']})
                continue
            if process_alive(prior.get('worker_pid'), 'run_meta_learning_continuous.py'):
                raise RuntimeError(f'{name}: existing worker is alive without its supervisor; refusing duplicate training')
            worker = [sys.executable, '-u', str(root / 'scripts/run_meta_learning_continuous.py'),
                      '--root', str(root), '--output', str(output / 'runs' / name),
                      '--feature-cache', str(feature_cache), '--representations', name,
                      '--outer-seeds', '20261031', '20261108', '42',
                      '--chunk-steps', '250', '--support-draws', '20']
            command = [sys.executable, '-u', str(root / 'scripts/background_supervisor.py'),
                       '--cwd', str(root), '--state-dir', str(supervisor_dir), '--', *worker]
            with (supervisor_dir / 'launcher.log').open('ab', buffering=0) as stream:
                process = subprocess.Popen(command, cwd=root, env=environment,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
            launches.append({'representation': name, 'status': 'LAUNCHED',
                             'supervisor_pid': process.pid, 'command': command})
        manifest = {'launched_at_utc': datetime.now(timezone.utc).isoformat(),
                    'authorized_cpu_hours': 900000, 'reserved_cpu_hours': None,
                    'cpu_budget_stop_limit': None, 'metric_stop_limit': None,
                    'runtime_stop_limit': None,
                    'persistence_scope': 'current cloud host lifetime; not a resource reservation',
                    'feature_cache': str(feature_cache), 'launches': launches}
        from background_supervisor import atomic_json
        atomic_json(output / 'launch_manifest.json', manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        states = []
        for name in args.representations:
            directory = output / 'supervisors' / name
            state = read_json(directory / 'status.json')
            training = read_json(output / 'runs' / name / 'execution_state.json')
            checkpoints = []
            for path in sorted((output / 'runs' / name / 'cells').glob('*/latest_training.json')):
                value = read_json(path)
                checkpoints.append({'cell': path.parent.name,
                    'training_steps': value.get('training_steps'),
                    'accumulated_training_cpu_seconds': value.get('accumulated_training_cpu_seconds'),
                    'at_utc': value.get('at_utc')})
            states.append({'representation': name,
                'supervisor_alive': process_alive(state.get('supervisor_pid'), 'background_supervisor.py'),
                'worker_alive': process_alive(state.get('worker_pid'), 'run_meta_learning_continuous.py'),
                'supervisor_pid': state.get('supervisor_pid'), 'worker_pid': state.get('worker_pid'),
                'phase': state.get('phase'), 'heartbeat_at_utc': state.get('heartbeat_at'),
                'attempt': state.get('attempt'), 'last_worker_returncode': state.get('last_worker_returncode'),
                'execution': training, 'checkpoints': checkpoints,
                'log_path': state.get('log_path')})
        print(json.dumps(states, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
