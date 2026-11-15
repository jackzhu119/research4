#!/usr/bin/env python3
"""New recovery namespace over unchanged operations/science, with bounded grids.

The original frozen source files are imported, never edited. A new protocol
binds this launcher, the numerical adapter, its block size and unqualified warm
densities. C008 waits for a sealed, real C003 gradient stress certificate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import run_spinosin_ir_gap_20261112 as base

ROOT = Path(__file__).resolve().parents[1]
_ORIGINAL_PREPARE = base.prepare
_ORIGINAL_SNAPSHOT = base.scientific_snapshot
_ORIGINAL_LOAD = base.load_science
_ORIGINAL_PUBLISHER = base.Publisher
_ORIGINAL_PHASE = base.phase_from


def memory_events():
    path = Path('/sys/fs/cgroup/memory.events')
    try:
        return {line.split()[0]: int(line.split()[1]) for line in path.read_text().splitlines()}
    except (OSError, ValueError, IndexError):
        return {}


def modules(root):
    sys.path[:0] = [str(root / 'scripts'), str(root / 'src')]
    return (importlib.import_module('spinosin_memory_repair_20261112'),
            importlib.import_module('spinosin_grid_response_streaming_20261112'))


def read_record(path):
    record = base.read_json(path)
    if not record or record.get('payload_sha256') != base.canonical_hash(record.get('payload')):
        raise base.IntegrityError('Recovery record seal differs: ' + str(path))
    return record['payload']


def point_arrays(result, leader):
    import numpy as np
    if (result.get('status') != 'DONE' or result.get('qualified') is not True
            or result.get('native_converged') is not True or not result.get('finished_at_utc')):
        raise ValueError('A finished native-qualified real gradient is required')
    arrays = {}
    for name, shape in (('coordinates_bohr', (75, 3)), ('gradient', (75, 3)), ('dipole_au', (3,))):
        reference = result['array_refs'][name]
        declared_path = Path(reference['path'])
        path = declared_path.resolve()
        if leader.resolve() not in path.parents or declared_path.is_symlink() or base.sha(path) != reference['sha256']:
            raise base.IntegrityError('Leader gradient arrays are not authenticated in the new namespace')
        value = np.load(path, allow_pickle=False)
        if value.shape != shape or not np.isfinite(value).all():
            raise base.IntegrityError('Leader gradient arrays have invalid shape or values')
        if not np.array_equal(value, np.asarray(result[name])):
            raise base.IntegrityError('Leader gradient JSON and binary arrays differ')
        arrays[name] = value
    return arrays


def verify_peer_protocol(leader, current):
    protocol = base.read_json(leader / 'ops/frozen_protocol.json')
    if not protocol:
        raise ValueError('Leader protocol is not prepared yet')
    expected = deepcopy(protocol)
    signature = expected.pop('input_hash')
    if (base.canonical_hash(expected) != signature or protocol['case_id'] != 'C003'
            or protocol['science_source_fingerprints']['C003'] != current['science_source_fingerprints']['C003']
            or protocol['scientific_file_sha256'] != current['scientific_file_sha256']
            or protocol['memory_repair'] != current['memory_repair']):
        raise base.IntegrityError('Leader protocol belongs to a different recovery experiment')
    science = read_record(leader / 'SCIENCE_PROTOCOL.json')
    if science['source_fingerprint'] != current['science_source_fingerprints']['C003']:
        raise base.IntegrityError('Leader scientific protocol has a different Source fingerprint')
    return protocol


def make_gradient_certificate(leader, point_path, protocol, audit):
    import hashlib
    import numpy as np
    leader_protocol = verify_peer_protocol(leader, protocol)
    declared_path = point_path
    point_path = point_path.resolve()
    if leader.resolve() not in point_path.parents or declared_path.is_symlink():
        raise base.IntegrityError('Gradient stress evidence is outside the new leader run')
    before = base.sha(point_path)
    result = base.read_json(point_path)
    arrays = point_arrays(result, leader)
    if result.get('symbols') != protocol['case_symbols']['C003']:
        raise ValueError('Gradient gate requires the authenticated Spinosin atom order')
    if result.get('gradient_response') != {'grid_response': True, 'auxbasis_response': True}:
        raise ValueError('Gradient gate requires full grid and auxiliary-basis response')
    geometry_hash = hashlib.sha256(np.ascontiguousarray(arrays['coordinates_bohr'], dtype='<f8').tobytes()).hexdigest()
    if (audit.get('completed') is not True or audit.get('atom_count') != 75 or audit.get('nao') != 762
            or audit.get('geometry_hash') != geometry_hash
            or audit.get('requested_block_size') != protocol['memory_repair']['block_size']
            or audit.get('owner_atoms_seen') != list(range(75))
            or audit.get('blocks_done', 0) <= 0 or not audit.get('finished_at_utc')):
        raise ValueError('A completed matching 75-atom blocked gradient audit is required')
    if base.sha(point_path) != before:
        raise ValueError('Gradient result is still being committed')
    return {'schema': 'spinosin-recovery-first-gradient-gate-v1', 'case_id': 'C003',
        'at_utc': base.utc(), 'leader_input_hash': leader_protocol['input_hash'],
        'leader_source_fingerprint': protocol['science_source_fingerprints']['C003'],
        'leader_science_protocol_sha256': base.sha(leader / 'SCIENCE_PROTOCOL.json'),
        'point_result_path': str(point_path), 'point_result_sha256': before,
        'gradient_geometry_hash': geometry_hash,
        'array_refs': {key: result['array_refs'][key] for key in arrays},
        'adapter_sha256': protocol['memory_repair']['adapter_sha256'],
        'block_size': protocol['memory_repair']['block_size'], 'adapter_audit': audit,
        'raw_gradient_native_qualified': True,
        'qualification_scope': 'Real 75-atom fresh electronic/gradient and blocked memory path only; not a stationary point, Hessian or IR qualification'}


def verify_gradient_certificate(leader, protocol):
    path = leader / 'ops/first_qualified_gradient_gate.json'
    certificate = read_record(path)
    leader_protocol = verify_peer_protocol(leader, protocol)
    if (certificate['leader_input_hash'] != leader_protocol['input_hash']
            or certificate['leader_source_fingerprint'] != protocol['science_source_fingerprints']['C003']
            or certificate['adapter_sha256'] != protocol['memory_repair']['adapter_sha256']
            or certificate['block_size'] != protocol['memory_repair']['block_size']
            or certificate['leader_science_protocol_sha256'] != base.sha(leader / 'SCIENCE_PROTOCOL.json')):
        raise base.IntegrityError('Gradient gate belongs to a different protocol/adapter')
    point = Path(certificate['point_result_path'])
    if base.sha(point) != certificate['point_result_sha256']:
        raise base.IntegrityError('Gradient stress result changed after its gate seal')
    current = make_gradient_certificate(leader, point, protocol, certificate['adapter_audit'])
    if current['gradient_geometry_hash'] != certificate['gradient_geometry_hash']:
        raise base.IntegrityError('Gradient stress geometry changed')
    return certificate


def prepare(args, readonly, block_size, recovery_root):
    registry, cases, hashes = _ORIGINAL_SNAPSHOT(args)
    bridge, adapter = modules(args.root)
    extra = [Path(__file__).resolve(), Path(__file__).with_name('manage_spinosin_ir_memory_repair_20261112.py'),
             args.root / 'scripts/manage_spinosin_ir_gap_20261112.py',
             args.root / 'scripts/background_supervisor.py',
             Path(bridge.__file__), Path(adapter.__file__)]
    dependencies = {}
    for case in cases.values():
        for path, expected in bridge.execution_dependencies(case, recovery_root=recovery_root).items():
            path = Path(path).resolve()
            if base.sha(path) != expected:
                raise base.IntegrityError('Warm-start recovery dependency bytes changed')
            dependencies[str(path)] = expected
            extra.append(path)
    for path in sorted(set(extra)):
        hashes[path.relative_to(args.root).as_posix()] = base.sha(path)
    source_fingerprints = {name: bridge.prepare_fingerprint(case, args.threads, args.max_memory_mb,
        block_size=block_size, recovery_root=recovery_root)['source_fingerprint'] for name, case in cases.items()}
    protocol = {'schema': 'local-spinosin-memory-repair-20261112-v1', 'case_id': args.case_id,
        'eligible_cases': list(base.CASES), 'case_input_hashes': {name: case['input_hash'] for name, case in cases.items()},
        'case_symbols': {name: case['symbols'] for name, case in cases.items()},
        'registry_id': registry.get('registry_id'), 'registry_path': str(args.case_manifest.relative_to(args.root)),
        'scientific_file_sha256': hashes, 'runtime': base.runtime_snapshot(),
        'resources': {'threads': args.threads, 'max_memory_mb': args.max_memory_mb},
        'science_design': base.json_safe(bridge.DESIGN), 'science_source_fingerprints': source_fingerprints,
        'memory_repair': {'adapter_sha256': base.sha(adapter.__file__), 'bridge_sha256': base.sha(bridge.__file__),
            'block_size': block_size, 'effective_block_size': adapter.effective_block_size(block_size),
            'warm_dependencies': dependencies,
            'density_scope': 'Previously unqualified electronic guesses only; all gradients, stationarity, Hessian and IR recomputed in the new namespace'},
        'parallel_gate': {'leader': 'C003', 'follower': 'C008',
            'condition': 'New-protocol 75-atom native-qualified real gradient and matching completed adapter audit with array SHA/shape validation',
            'timeout': None, 'RSS_stop_limit': None},
        'compute_policy': {'cpu_stop_limit': None, 'wall_stop_limit': None, 'quality_stop_limit': None,
            'retry_count_stop_limit': None, 'global_round_stop_limit': None,
            'authorization_ledger': 'data/research/meta_learning_20261111/compute_authorization_addenda.json',
            'prior_run_costs_remain_charged': True, 'cloud_uptime_guarantee': False}}
    protocol['input_hash'] = base.canonical_hash(protocol)
    path = args.output / 'ops/frozen_protocol.json'
    if path.exists():
        if base.read_json(path) != protocol:
            raise base.IntegrityError('Recovery code, adapter parameters, warm guess or original case changed')
    elif readonly:
        raise base.IntegrityError('Read-only recovery verification needs a frozen protocol')
    else:
        base.atomic_json(path, protocol)

    class BridgeProxy:
        ScientificIntegrityError = bridge.ScientificIntegrityError
        DESIGN = bridge.DESIGN
        @staticmethod
        def prepare_fingerprint(case, threads=2, max_memory_mb=3000):
            return bridge.prepare_fingerprint(case, threads, max_memory_mb,
                block_size=block_size, recovery_root=recovery_root)
        @staticmethod
        def run_conformer(case, output, threads=2, max_memory_mb=3000, progress=None, stop_requested=None):
            if case['case_id'] == 'C008':
                leader = Path(output).parent / 'C003'
                if progress:
                    progress({'phase': 'waiting_for_peer_gradient', 'checkpoint_committed': False,
                              'active_computation': False, 'leader_case': 'C003'})
                while True:
                    if stop_requested and stop_requested():
                        raise InterruptedError('Manual stop while waiting for leader gradient')
                    base.check_frozen_files(args, protocol)
                    if (leader / 'ops/first_qualified_gradient_gate.json').exists():
                        certificate = verify_gradient_certificate(leader, protocol)
                        base.atomic_json(Path(output) / 'ops/peer_gate_verified.json', certificate)
                        break
                    time.sleep(.5)
                if progress:
                    progress({'phase': 'geometry_opt', 'checkpoint_committed': False,
                              'active_computation': True, 'leader_gate_verified': True})
            return bridge.run_conformer(case, output, threads, max_memory_mb, progress, stop_requested,
                block_size=block_size, recovery_root=recovery_root)
    return protocol, cases[args.case_id], BridgeProxy


def publisher_type(block_size):
    class RecoveryPublisher(_ORIGINAL_PUBLISHER):
        def __init__(self, args, protocol, heartbeat_seconds=None):
            self.monitor_done = threading.Event()
            self.monitor_thread = None
            super().__init__(args, protocol, heartbeat_seconds)
            if args.case_id == 'C003':
                self.monitor_thread = threading.Thread(target=self._monitor_gradient, name='C003-first-gradient-gate', daemon=True)
                self.monitor_thread.start()
        def write(self):
            with self.lock:
                self.state.update(memory_events=memory_events(), grid_block_size=block_size,
                                  parallel_start_gate='C008 awaits a real qualified C003 gradient')
            super().write()
        def progress(self, event):
            if event.get('phase') == 'waiting_for_peer_gradient':
                self.update(phase='waiting_for_peer_gradient', active_computation=False)
            elif event.get('leader_gate_verified'):
                self.update(active_computation=True)
            super().progress(event)
        def _monitor_gradient(self):
            gate = self.args.output / 'ops/first_qualified_gradient_gate.json'
            while not self.monitor_done.is_set():
                try:
                    if gate.exists():
                        verify_gradient_certificate(self.args.output, self.protocol)
                        self.update(first_gradient_gate='PASSED', first_gradient_gate_path=str(gate))
                        return
                    _, adapter = modules(self.args.root)
                    audit = adapter.get_last_audit()
                    if audit.get('completed') is True:
                        for point in sorted(self.args.output.glob('generations/**/steps/**/result.json')):
                            try:
                                certificate = make_gradient_certificate(self.args.output, point, self.protocol, audit)
                            except (ValueError, KeyError, OSError, base.IntegrityError):
                                continue
                            base.atomic_json(gate, {'payload': certificate, 'payload_sha256': base.canonical_hash(certificate)})
                            self.update(first_gradient_gate='PASSED', first_gradient_gate_path=str(gate))
                            return
                except Exception as error:
                    self.update(first_gradient_gate='WAITING_FOR_VALID_EVIDENCE', gate_observation_error=repr(error))
                self.monitor_done.wait(.5)
        def close(self):
            self.monitor_done.set()
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2)
            super().close()
    return RecoveryPublisher


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--grid-block-size', type=int, default=2048)
    parser.add_argument('--recovery-input-root', type=Path)
    options, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if options.grid_block_size <= 0:
        parser.error('Grid blocks must be positive; this is a numerical tiling parameter, not a stopping threshold')
    parsed = base.parse_args(remaining)
    if parsed.output == parsed.root / 'reports/spinosin_ir_gap_fill_20261112/runs' / parsed.case_id:
        parsed.output = parsed.root / 'reports/spinosin_ir_memory_repair_20261112/runs' / parsed.case_id
        remaining += ['--output', str(parsed.output)]
    recovery_root = options.recovery_input_root or parsed.root / 'reports/spinosin_ir_memory_repair_20261112/recovery_inputs'
    recovery_root = base.resolve(parsed.root, recovery_root)
    previous_prepare, previous_publisher, previous_phase = base.prepare, base.Publisher, base.phase_from
    base.prepare = lambda args, readonly=False: prepare(args, readonly, options.grid_block_size, recovery_root)
    base.Publisher = publisher_type(options.grid_block_size)
    base.phase_from = lambda event, default='geometry_opt': (
        'waiting_for_peer_gradient' if event.get('phase') == 'waiting_for_peer_gradient'
        else _ORIGINAL_PHASE(event, default))
    try:
        return base.main(remaining)
    finally:
        base.prepare, base.Publisher, base.phase_from = previous_prepare, previous_publisher, previous_phase


if __name__ == '__main__':
    raise SystemExit(main())
