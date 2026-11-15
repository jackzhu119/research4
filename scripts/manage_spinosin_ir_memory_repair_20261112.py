#!/usr/bin/env python3
"""Launch the separately frozen, gradient-gated Spinosin memory recovery."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import uuid

import manage_spinosin_ir_gap_20261112 as original
from run_spinosin_ir_gap_20261112 import atomic_json, read_json, sha
from run_spinosin_ir_memory_repair_20261112 import memory_events

ROOT = Path(__file__).resolve().parents[1]
WORKER = 'run_spinosin_ir_memory_repair_20261112.py'
RUN_ROOT = Path('reports/spinosin_ir_memory_repair_20261112')


def prior_costs(root):
    """Retain measured prior cost; missing abrupt-kill tails are not zero."""
    values = {}
    for case in ('C003', 'C008'):
        path = root / 'reports/spinosin_ir_gap_fill_20261112/runs' / case / 'ops/cpu_ledger.json'
        record = read_json(path)
        values[case] = {'path': str(path), 'sha256': sha(path) if path.is_file() else None,
                       'record': record, 'abrupt_termination_tail_may_be_unmeasured': True}
    return values


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=RUN_ROOT)
    # One immutable production tiling design. A changed tiling needs its own
    # frozen protocol, not a silently modified running worker command.
    parser.add_argument('--grid-block-size', type=int, choices=(2048,), default=2048)
    options, _ = parser.parse_known_args(arguments)
    root = options.root.resolve()
    output = options.output if options.output.is_absolute() else root / options.output
    output = output.resolve()
    if not output.is_relative_to((root / RUN_ROOT).resolve()):
        parser.error('Recovery operations must use the new memory-repair namespace')
    # Original manager retains its process identity, locks, detached supervisor,
    # preflight and idempotence behavior. Its source bytes are never changed.
    filtered = []
    index = 0
    while index < len(arguments):
        if arguments[index] in ('--output', '--grid-block-size'):
            index += 2
        elif arguments[index].startswith(('--output=', '--grid-block-size=')):
            index += 1
        else:
            filtered.append(arguments[index])
            index += 1
    filtered += ['--output', str(output)]
    previous = original.WORKER
    original.WORKER = WORKER
    try:
        if not filtered or filtered[0] != 'start':
            return original.main(filtered)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            status = original.main(filtered)
        if status:
            print(captured.getvalue(), end='')
            return status
        receipt = read_json(output / 'launch_manifest.json')
        receipt.update(schema='spinosin-memory-repair-gated-launch-v1',
            recovery_launch_id=uuid.uuid4().hex,
            recovery_launcher_sha256=sha(__file__),
            recovery_worker_sha256=sha(root / 'scripts' / WORKER),
            grid_block_size=2048, effective_grid_block_size=2016,
            baseline_cgroup_memory_events=memory_events(),
            historical_initial_gradient_oom_kills=2,
            controlled_migration='Prior OOM workers manually stopped; unchanged old scientific artifacts retained. Only authenticated unqualified same-geometry densities seed fresh recovery.',
            prior_run_costs=prior_costs(root),
            parallel_start_gate={'C003': 'Starts immediately', 'C008': 'Live waiting until a new C003 real 75-atom native-qualified gradient and matching adapter audit pass authenticated arrays and SHA checks',
                                 'timeout': None, 'RSS_stop_limit': None},
            warm_start_scope='No old gradients, stationary point, Hessian or IR qualifications are reused')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        atomic_json(output / 'launch_receipts' / f'{stamp}-recovery-{receipt["recovery_launch_id"]}.json', receipt)
        atomic_json(output / 'launch_manifest.json', receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    finally:
        original.WORKER = previous


if __name__ == '__main__':
    raise SystemExit(main())
