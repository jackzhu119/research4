#!/usr/bin/env python3
"""Export and verify a read-only snapshot of the running Spinosin experiment.

This does not change scientific code, stop workers, or certify a minimum.
Completed-point density caches may have been pruned by the frozen workflow;
coordinates, nuclear gradients and dipoles remain mandatory result artifacts.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

import numpy as np

RUN = Path('reports/spinosin_ir_memory_repair_20261112')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def staged_ref(root, stage, reference):
    path = Path(reference['path'])
    if path.is_absolute():
        path = path.relative_to(root)
    return stage / path


def copy_stable(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Bounded export retries do not impose any training stop/retry limit.
    for _ in range(3):
        before = source.stat()
        shutil.copyfile(source, destination)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
            return
    raise RuntimeError(f'File kept changing during export: {source}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    status_command = ['python', 'scripts/manage_spinosin_ir_memory_repair_20261112.py',
                      'status', '--root', str(root)]
    status = json.loads(subprocess.check_output(status_command, cwd=root, text=True))
    write_json(output / 'runtime_status.json', {'captured_at_utc': started, 'cases': status,
        'scope': 'Point-in-time process observation, not an uptime guarantee.'})
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    points = []
    summaries = {}
    with tempfile.TemporaryDirectory(prefix='spinosin-progress-export-') as folder:
        stage = Path(folder)
        excluded = []
        for source in sorted((root / RUN).rglob('*')):
            if not source.is_file():
                continue
            relative = source.relative_to(root)
            if (source.is_symlink() or source.name.endswith(('.lock', '.tmp', '.partial'))
                    or '__pycache__' in source.parts):
                excluded.append(relative.as_posix())
                continue
            copy_stable(source, stage / relative)
        for case in ('C003', 'C008'):
            case_root = stage / RUN / 'runs' / case
            case_points = []
            for path in sorted(case_root.glob('generations/*/optimization/chunk_*/steps/step-*/result.json')):
                result = json.loads(path.read_text())
                if result.get('status') != 'DONE' or result.get('qualified') is not True:
                    continue
                for key, shape in [('coordinates_bohr', (75, 3)), ('gradient', (75, 3)), ('dipole_au', (3,))]:
                    reference = result['array_refs'][key]
                    artifact = staged_ref(root, stage, reference)
                    if sha(artifact) != reference['sha256']:
                        raise ValueError(f'Array SHA mismatch: {artifact}')
                    array = np.load(artifact, allow_pickle=False)
                    if array.shape != shape or not np.isfinite(array).all():
                        raise ValueError(f'Invalid scientific array: {artifact}')
                density = result['array_refs'].get('density')
                row = {'case_id': case, 'path': path.relative_to(stage).as_posix(),
                    'result_sha256': sha(path),
                    'coordinates_sha256': result['array_refs']['coordinates_bohr']['sha256'],
                    'gradient_sha256': result['array_refs']['gradient']['sha256'],
                    'dipole_sha256': result['array_refs']['dipole_au']['sha256'],
                    'density_cache_present_in_snapshot': bool(density and staged_ref(root, stage, density).is_file()),
                    'optimizer_acceptance': 'Not inferred from electronic qualification or evaluation count',
                    **{key: result[key] for key in ('finished_at_utc', 'energy_hartree',
                        'nuclear_gradient_rms', 'nuclear_gradient_max')}}
                case_points.append(row)
            case_points.sort(key=lambda row: row['finished_at_utc'])
            points.extend(case_points)
            envelope = json.loads((case_root / 'workflow_state.json').read_text())
            state = envelope['payload']
            canonical = json.dumps(state, ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':'), allow_nan=False).encode()
            if hashlib.sha256(canonical).hexdigest() != envelope['payload_sha256']:
                raise ValueError('Workflow state envelope SHA mismatch')
            density_ref = state['density_ref']
            seed = staged_ref(root, stage, density_ref)
            density = np.load(seed, allow_pickle=False)
            if sha(seed) != density_ref['sha256'] or density.shape != (762, 762) or not np.isfinite(density).all():
                raise ValueError('Latest workflow warm density failed verification')
            rejections = []
            logs = list(case_root.glob('generations/*/optimization/chunk_*/*.log'))
            logs += list((stage / RUN / 'supervisors' / case / 'logs').glob('*.log'))
            for log in sorted(logs):
                for line_number, line in enumerate(log.read_text(errors='replace').splitlines(), 1):
                    if 'Rejecting step' in line:
                        rejections.append({'path': log.relative_to(stage).as_posix(),
                                           'line_number': line_number, 'line': line})
            summaries[case] = {'completed_qualified_gradient_evaluations': len(case_points),
                'distinct_serialized_geometry_sha_count': len({p['coordinates_sha256'] for p in case_points}),
                'counts_are_not_accepted_steps_or_independent_physical_conformers': True,
                'latest_completed_evaluation': case_points[-1],
                'lowest_observed_rms_evaluation': min(case_points, key=lambda p: p['nuclear_gradient_rms']),
                'lowest_observed_energy_evaluation': min(case_points, key=lambda p: p['energy_hartree']),
                'optimizer_rejection_evidence': rejections,
                'workflow_phase': state['phase'], 'geometry_converged': False,
                'production_hessian_sealed_points': len(list(case_root.glob('**/POINT_SEAL.json'))),
                'production_ir_verified': False,
                'workflow_density_sha256_verified': density_ref['sha256'],
                'workflow_state_path': (case_root / 'workflow_state.json').relative_to(stage).as_posix()}
        archive = output / 'spinosin_full_snapshot.tar.gz'
        with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
            for path in sorted(stage.rglob('*')):
                if path.is_file():
                    tar.add(path, arcname=path.relative_to(stage).as_posix(), recursive=False)
        files = {p.relative_to(stage).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
                 for p in sorted(stage.rglob('*')) if p.is_file()}
        # Read the archive back and verify every member independently of source files.
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            if len(members) != len(files):
                raise ValueError('Archive member count mismatch')
            for member in members:
                expected = files[member.name]
                blob = tar.extractfile(member).read()
                if len(blob) != expected['bytes'] or hashlib.sha256(blob).hexdigest() != expected['sha256']:
                    raise ValueError('Archive member verification failed: ' + member.name)
        write_json(output / 'archive_manifest.json', {'schema': 'spinosin-progress-snapshot-v1',
            'capture_started_at_utc': started, 'capture_finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_local_commit': source_commit, 'exporter_sha256': sha(__file__),
            'source_run_root': RUN.as_posix(), 'archive': archive.name,
            'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
            'member_count': len(files), 'members': files, 'excluded_operational_files': excluded,
            'snapshot_scope': 'Individually stable file copies while workers continue; mutable status/logs are historical observations. Completed arrays and current workflow density were validated in the copied snapshot.',
            'pruned_density_scope': 'Historical point density caches are reconstructible and were pruned by the workflow. Original result JSON references are preserved. Their absence does not remove coordinates, gradients or dipoles.',
            'verification': {'all_member_sha256_read_back': True,
                'all_completed_coordinates_gradient_dipole_sha_shape_finite': True,
                'both_workflow_state_envelopes_and_current_density_refs': True}})
    write_json(output / 'progress_summary.json', {'captured_at_utc': started, 'cases': summaries,
        'scientific_source_or_tolerances_changed': False,
        'stationarity_rms_target_eh_per_bohr': 3e-6, 'stationarity_max_target_eh_per_bohr': 1e-5,
        'known_restart_limitation': 'The frozen vendor progress records the latest evaluated geometry before optimizer acceptance. Following a host restart it can seed a fresh optimizer with a rejected trial and lose the previous trust radius/history. This affects recovery efficiency; no minimum/Hessian/IR qualification is bypassed. Scientific code was not changed in this export.',
        'archive_manifest': 'archive_manifest.json', 'runtime_status': 'runtime_status.json'})
    with (output / 'gradient_evaluations.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    print(json.dumps({'output': str(output), 'archive_bytes': archive.stat().st_size,
        'verified_members': len(files), 'complete_evaluations': {c: s['completed_qualified_gradient_evaluations']
        for c, s in summaries.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
