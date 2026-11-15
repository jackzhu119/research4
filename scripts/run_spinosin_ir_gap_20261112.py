#!/usr/bin/env python3
"""One local, resumable Spinosin conformer lane; no budget or quality early stop.

Only C003/C008 are eligible. Molecular qualification is owned by the scientific
workflow, never inferred from an exit code or fabricated by this operations layer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
CASES = ('C003', 'C008')
STOP_REQUESTED = False
INTEGRITY_ERROR = 42


def utc():
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, 'tolist'):
        return json_safe(value.tolist())
    if hasattr(value, 'item'):
        return value.item()
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(json_safe(value), stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


class IntegrityError(ValueError):
    pass


def configure_resources(args):
    # Apply affinity before importing NumPy/PySCF so native pool threads inherit it.
    if args.cpu_affinity and hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(args.cpu_affinity))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = str(args.threads)
    os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'


def runtime_snapshot():
    return {'python_version': platform.python_version(),
            'libraries': {name: importlib.metadata.version(name) for name in
                          ('pyscf', 'geometric', 'numpy', 'scipy', 'h5py', 'networkx')}}


def load_science(root):
    sys.path[:0] = [str(root / 'scripts'), str(root / 'src')]
    return importlib.import_module('spinosin_gap_science_20261112')


def validate_cases(root, manifest_path):
    registry = read_json(manifest_path)
    if not isinstance(registry, dict) or not isinstance(registry.get('cases'), list):
        raise IntegrityError('Authenticated conformer registry is missing')
    cases = registry['cases']
    if len(cases) != 2 or {case.get('case_id') for case in cases} != set(CASES):
        raise IntegrityError('Only the two authenticated C003/C008 inputs are eligible')
    import numpy as np
    for case in cases:
        if (case.get('charge') != 0 or case.get('spin') != 0 or case.get('formula') != 'C28H32O15'
                or Counter(case['symbols']) != Counter({'C': 28, 'H': 32, 'O': 15})):
            raise IntegrityError('Conformer formula/charge/spin/atoms differ')
        coords = np.asarray(case['initial_geometry_bohr'], dtype=float)
        if coords.shape != (75, 3) or not np.isfinite(coords).all():
            raise IntegrityError('A complete finite 75-atom Bohr geometry is required')
        payload = case['input_fingerprint_payload']
        if canonical_hash(payload) != case['input_hash']:
            raise IntegrityError('Conformer fingerprint differs')
        for key in ('case_id', 'formula', 'charge', 'spin', 'symbols', 'initial_geometry_bohr',
                    'bohr_in_angstrom', 'source_xyz_sha256', 'source_sdf_sha256', 'stereo_inchikey'):
            if payload[key] != case[key]:
                raise IntegrityError('Conformer fields differ from their authenticated fingerprint')
        for path_key, hash_key in (('source_xyz_path', 'source_xyz_sha256'),
                                   ('source_sdf_path', 'source_sdf_sha256'),
                                   ('initial_geometry_bohr_path', 'initial_geometry_bohr_npy_sha256')):
            path = resolve(root, case[path_key])
            if not path.is_file() or sha(path) != case[hash_key]:
                raise IntegrityError('Authenticated molecular input bytes changed: ' + path_key)
        saved = np.load(resolve(root, case['initial_geometry_bohr_path']), allow_pickle=False)
        if not np.array_equal(saved, coords):
            raise IntegrityError('Conformer JSON and binary coordinates differ')
        if case.get('initial_density_available') or case.get('historical_C001_density_or_hessian_reuse_allowed'):
            raise IntegrityError('No C001 density or Hessian may seed the new conformers')
    return registry, {case['case_id']: case for case in cases}


def scientific_snapshot(args):
    registry, cases = validate_cases(args.root, args.case_manifest)
    vendor = args.root / 'scripts/quantum_gap_vendor_20261112'
    scripts = [args.root / 'scripts' / name for name in
        ('run_spinosin_ir_gap_20261112.py', 'spinosin_gap_science_20261112.py', 'spinosin_gap_ir_20261112.py')]
    scripts += sorted(vendor.glob('*.py'))
    provenance = args.root / 'data/research/spinosin_ir_gap_fill_20261112/vendor_provenance.json'
    if not vendor.is_dir() or len(list(vendor.glob('*.py'))) < 7:
        raise IntegrityError('The fixed vendor source package is incomplete')
    scripts.append(provenance)
    paths = scripts + sorted(path for path in args.case_manifest.parent.rglob('*') if path.is_file())
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise IntegrityError('Frozen source/input file is missing or a symlink: ' + str(path))
    hashes = {path.relative_to(args.root).as_posix(): sha(path) for path in sorted(set(paths))}
    return registry, cases, hashes


def prepare(args, readonly=False):
    registry, cases, hashes = scientific_snapshot(args)
    science = load_science(args.root)
    source_fingerprints = {name: science.prepare_fingerprint(case, args.threads, args.max_memory_mb)['source_fingerprint']
                           for name, case in cases.items()}
    protocol = {'schema': 'local-spinosin-ir-gap-20261112-v1', 'case_id': args.case_id,
        'eligible_cases': list(CASES), 'case_input_hashes': {name: case['input_hash'] for name, case in cases.items()},
        'registry_id': registry.get('registry_id'), 'registry_path': str(args.case_manifest.relative_to(args.root)),
        'scientific_file_sha256': hashes, 'runtime': runtime_snapshot(),
        'resources': {'threads': args.threads, 'max_memory_mb': args.max_memory_mb},
        'science_design': json_safe(getattr(science, 'DESIGN', getattr(science, 'PROTOCOL', {}))),
        'science_source_fingerprints': source_fingerprints,
        'primary_scope': {'per_conformer_double_step_gradient_and_dipole_points': 900,
            'usual_additional_third_cartesian_dipole_points': 450,
            'initial_direct_normal_mode_points': 219 * 2 * 3,
            'optimization_centers_and_conditional_refinements': 'additional; actual counts recorded, no fixed global round cutoff'},
        'initial_state': 'Two verified MMFF conformers, no C001 density/Hessian reuse, no duplicate acid/calycosin/water jobs',
        'compute_policy': {'latest_additional_cpu_seconds_authorization': 900000,
            'authorization_is_execution_deadline': False, 'cpu_stop_limit': None, 'wall_stop_limit': None,
            'quality_stop_limit': None, 'retry_count_stop_limit': None, 'global_round_stop_limit': None,
            'authorization_ledger': 'data/research/meta_learning_20261111/compute_authorization_addenda.json',
            'cloud_uptime_guarantee': False},
        'result_policy': 'Scientific completion, minimum qualification and independent IR validation remain separate. Waiting for a new refinement design remains a live process.'}
    protocol['input_hash'] = canonical_hash(protocol)
    path = args.output / 'ops/frozen_protocol.json'
    if path.exists():
        if read_json(path) != protocol:
            raise IntegrityError('Frozen scientific code, inputs, runtime or numerical resource settings changed')
    elif readonly:
        raise IntegrityError('Read-only verification requires an existing frozen protocol')
    else:
        atomic_json(path, protocol)
    return protocol, cases[args.case_id], science


def check_frozen_files(args, protocol):
    for name, expected in protocol['scientific_file_sha256'].items():
        path = args.root / name
        if not path.is_file() or sha(path) != expected:
            raise IntegrityError('Frozen scientific code/input changed during execution: ' + name)


def phase_from(event, default='geometry_opt'):
    text = str(event.get('phase', event.get('stage', default))).lower()
    if 'hessian' in text or 'finite_difference' in text:
        return 'full_hessian'
    if any(word in text for word in ('infrared', 'ir_', 'normal_mode', 'third_cartesian', 'dipole')) or text == 'ir':
        return 'ir'
    if any(word in text for word in ('geometry', 'opt', 'center')):
        return 'geometry_opt'
    return default


def process_cpu_seconds():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


class Publisher:
    def __init__(self, args, protocol, heartbeat_seconds=None):
        self.args, self.protocol = args, protocol
        self.lock, self.done = threading.RLock(), threading.Event()
        self.interval = heartbeat_seconds or args.heartbeat_seconds
        self.attempt = uuid.uuid4().hex
        prior = read_json(args.output / 'ops/cpu_ledger.json', {})
        self.prior_cpu = float(prior.get('measured_cpu_seconds', 0.0))
        self.state = {'phase': 'starting', 'status': 'STARTING', 'case_id': args.case_id,
            'pid': os.getpid(), 'attempt_id': self.attempt, 'input_hash': protocol['input_hash'],
            'last_progress_at_utc': utc(), 'current_point': None,
            'threads': args.threads, 'max_memory_mb': args.max_memory_mb,
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'computed': False, 'minimum_verified': False, 'ir_verified': False,
            'cpu_wall_quality_retry_round_stop_limit': None}
        self.thread = threading.Thread(target=self._heartbeat, name='spinosin-lane-heartbeat', daemon=True)
        self.thread.start()

    def write(self):
        with self.lock:
            measured = self.prior_cpu + process_cpu_seconds()
            self.state.update(heartbeat_at_utc=utc(), measured_cpu_seconds=measured,
                              process_cpu_seconds=process_cpu_seconds(),
                              rss_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                              workspace_disk_free_bytes=shutil.disk_usage(self.args.output).free,
                              temporary_disk_free_bytes=shutil.disk_usage(tempfile.gettempdir()).free)
            try:
                atomic_json(self.args.output / 'execution_state.json', self.state)
                atomic_json(self.args.output / 'ops/cpu_ledger.json', {
                    'measured_cpu_seconds': measured, 'input_hash': self.protocol['input_hash'],
                    'attempt_id': self.attempt, 'pid': os.getpid(), 'observed_at_utc': utc(),
                    'accounting': 'Observed process user+system/all native threads plus reaped children; abrupt termination may leave an unmeasured tail, never assumed zero',
                    'budget_stop_limit': None})
            except OSError as error:
                print('Status persistence failed: ' + repr(error), file=sys.stderr, flush=True)

    def _heartbeat(self):
        self.write()
        while not self.done.wait(self.interval):
            self.write()

    def update(self, **fields):
        with self.lock:
            self.state.update(fields)
            if 'phase' in fields:
                self.state['status'] = fields['phase'].upper()
        self.write()

    def progress(self, event):
        event = json_safe(event)
        with self.lock:
            self.state.update(progress_event=event, last_progress_at_utc=utc(),
                              phase=phase_from(event, self.state['phase']))
            self.state['status'] = self.state['phase'].upper()
        if event.get('checkpoint_committed'):
            check_frozen_files(self.args, self.protocol)
        self.write()
        if STOP_REQUESTED and event.get('checkpoint_committed'):
            raise InterruptedError('Manual stop after a scientifically committed unit')

    def close(self):
        self.done.set()
        self.thread.join(timeout=2)
        self.write()


def scientific_artifacts(output):
    result = {}
    for path in sorted(output.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        relative_path = path.relative_to(output)
        if relative_path.parts[0] == 'ops' or any(part.startswith('.') for part in relative_path.parts):
            continue
        if any(part in ('local-only', 'local_only', 'cache', 'checkpoints', '__pycache__') for part in relative_path.parts):
            continue
        if path.name == 'execution_state.json' or path.name.endswith(('.tmp', '.partial', '.lock')):
            continue
        if path.suffix.lower() in ('.json', '.npy', '.npz', '.csv', '.xyz', '.sdf', '.log', '.txt', '.out', '.dat'):
            result[relative_path.as_posix()] = sha(path)
    return result


def science_flags(result):
    return {key: result.get(key) is True for key in
            ('computed', 'minimum_verified', 'ir_verified', 'paper_ready_molecular_harmonic_data')}


def seal_completion(args, protocol, result):
    flags = science_flags(result)
    if (result.get('status') != 'COMPLETE' or not all(flags.values())
            or result.get('case_id') != args.case_id
            or result.get('source_fingerprint') != protocol['science_source_fingerprints'][args.case_id]):
        raise IntegrityError('Operations layer cannot certify an unqualified scientific result')
    artifacts = scientific_artifacts(args.output)
    if not artifacts:
        raise IntegrityError('Completed workflow has no inspectable scientific artifacts')
    on_disk_result = read_json(args.output / 'result.json')
    if on_disk_result is not None and canonical_hash(on_disk_result) != canonical_hash(result):
        raise IntegrityError('Returned and saved scientific completion results differ')
    receipt = {'schema': 'local-spinosin-workflow-completion-v1', 'case_id': args.case_id,
        'input_hash': protocol['input_hash'], 'status': 'COMPLETE', 'at_utc': utc(),
        **flags, 'scientific_result': json_safe(result), 'artifact_sha256': artifacts}
    path = args.output / 'ops/workflow_complete.json'
    if path.exists():
        old = verify_completion(args, protocol)
        if old['scientific_result'] != receipt['scientific_result']:
            raise IntegrityError('Completed scientific result changed; original receipt preserved')
        return old
    atomic_json(path, receipt)
    return receipt


def verify_completion(args, protocol):
    receipt = read_json(args.output / 'ops/workflow_complete.json')
    if not receipt:
        return None
    if receipt['input_hash'] != protocol['input_hash'] or receipt['case_id'] != args.case_id:
        raise IntegrityError('Completed workflow input/case differs')
    if receipt.get('status') != 'COMPLETE' or not all(science_flags(receipt).values()):
        raise IntegrityError('Completed workflow qualification fields differ')
    result = receipt['scientific_result']
    if (result.get('case_id') != args.case_id
            or result.get('source_fingerprint') != protocol['science_source_fingerprints'][args.case_id]):
        raise IntegrityError('Completed scientific result belongs to a different conformer/model')
    for name, expected in receipt['artifact_sha256'].items():
        path = args.output / name
        if not path.is_file() or sha(path) != expected:
            raise IntegrityError('Completed scientific artifact changed: ' + name)
    return receipt


def verify_committed_record_hashes(output):
    """Read partial self-sealed scientific records without invoking any solver."""
    checked = 0
    for path in output.rglob('*.json'):
        relative_path = path.relative_to(output)
        if relative_path.parts[0] == 'ops' or any(part.startswith('.') for part in relative_path.parts):
            continue
        if path.name == 'execution_state.json':
            continue
        value = read_json(path)
        if isinstance(value, dict) and 'payload_sha256' in value:
            if value.get('payload_sha256') != canonical_hash(value.get('payload')):
                raise IntegrityError('Committed partial scientific record changed: ' + str(relative_path))
            checked += 1
    return checked


def run_lane(args, protocol, case, science, publisher):
    attempts = 0
    waiting = False
    while not STOP_REQUESTED:
        check_frozen_files(args, protocol)
        completed = verify_completion(args, protocol)
        if completed:
            publisher.update(phase='qualified_complete_idle', **science_flags(completed), active_computation=False)
        elif waiting:
            publisher.update(phase='waiting_refinement', active_computation=False,
                             waiting_for_new_protocol=True, continuation_eligible=False)
        else:
            publisher.update(phase='geometry_opt', active_computation=True, waiting_for_new_protocol=False)
            try:
                result = science.run_conformer(case, args.output, threads=args.threads,
                    max_memory_mb=args.max_memory_mb, progress=publisher.progress,
                    stop_requested=lambda: STOP_REQUESTED)
                if not isinstance(result, dict):
                    raise RuntimeError('Scientific workflow must return its explicit result dictionary')
                atomic_json(args.output / 'ops/last_science_return.json', result)
                publisher.update(**science_flags(result), scientific_status=result.get('status'))
                if result.get('status') == 'COMPLETE':
                    seal_completion(args, protocol, result)
                    publisher.update(phase='qualified_complete_idle', active_computation=False)
                else:
                    waiting = bool(result.get('waiting_for_new_protocol', False)) or not bool(result.get('continuation_eligible', True))
                    publisher.update(phase='waiting_refinement' if waiting else phase_from(result),
                                     active_computation=False, waiting_for_new_protocol=waiting,
                                     continuation_eligible=not waiting)
            except InterruptedError:
                publisher.update(phase='checkpointed_manual_stop', active_computation=False)
                return
            except IntegrityError:
                raise
            except Exception as error:
                scientific_integrity = getattr(science, 'ScientificIntegrityError', None)
                if scientific_integrity is not None and isinstance(error, scientific_integrity):
                    raise IntegrityError(str(error)) from error
                attempts += 1
                failure = {'attempt': attempts, 'at_utc': utc(), 'error': repr(error), 'traceback': traceback.format_exc()}
                atomic_json(args.output / 'ops/latest_recoverable_failure.json', failure)
                publisher.update(phase='retry_pending', active_computation=False, recoverable_failure=failure)
                if attempts & (attempts - 1) == 0:
                    print(json.dumps({'case_id': args.case_id, 'status': 'RETRY_PENDING', 'error': repr(error)}), flush=True)
        if args.once:
            return
        until = time.monotonic() + args.poll_seconds
        while not STOP_REQUESTED and time.monotonic() < until:
            time.sleep(min(.2, max(0.0, until - time.monotonic())))
    publisher.update(phase='manual_stop', active_computation=False)


def stop_handler(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--case', dest='case_id', choices=CASES, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case-manifest', type=Path, default=Path('data/research/spinosin_ir_gap_fill_20261112/inputs/cases.json'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-memory-mb', type=int, default=3000)
    parser.add_argument('--cpu-affinity', nargs='*', type=int)
    parser.add_argument('--heartbeat-seconds', type=float, default=5.0)
    parser.add_argument('--poll-seconds', type=float, default=10.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--verify', action='store_true')
    mode.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    args.case_manifest = resolve(args.root, args.case_manifest)
    args.output = resolve(args.root, args.output or Path('reports/spinosin_ir_gap_fill_20261112/runs') / args.case_id)
    if args.threads <= 0 or args.max_memory_mb <= 0 or args.heartbeat_seconds <= 0 or args.poll_seconds <= 0:
        parser.error('Positive numerical resource settings and heartbeat intervals required')
    return args


def main(argv=None):
    args = parse_args(argv)
    configure_resources(args)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop_handler)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    lock = None
    publisher = None
    try:
        if not args.verify:
            (args.output / 'ops').mkdir(parents=True, exist_ok=True)
            lock = (args.output / 'ops/worker.lock').open('a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, case, science = prepare(args, readonly=args.verify)
        if args.prepare:
            print(json.dumps({'case_id': args.case_id, 'phase': 'prepared_no_scientific_calculation',
                              'input_hash': protocol['input_hash']}), flush=True)
        elif args.verify:
            check_frozen_files(args, protocol)
            completed = verify_completion(args, protocol)
            partial_record_hashes = verify_committed_record_hashes(args.output)
            print(json.dumps({'case_id': args.case_id, 'completed': completed is not None,
                              'committed_partial_record_hashes_verified': partial_record_hashes,
                              'input_hash': protocol['input_hash'], 'read_only': True}), flush=True)
        else:
            publisher = Publisher(args, protocol)
            run_lane(args, protocol, case, science, publisher)
        return 0
    except IntegrityError as error:
        if not args.verify:
            atomic_json(args.output / 'ops/integrity_error.json', {'at_utc': utc(), 'error': repr(error)})
            if publisher:
                publisher.update(phase='integrity_blocked', active_computation=False, error=repr(error))
        print(str(error), file=sys.stderr, flush=True)
        return INTEGRITY_ERROR
    finally:
        if publisher:
            publisher.close()
        if lock:
            lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
