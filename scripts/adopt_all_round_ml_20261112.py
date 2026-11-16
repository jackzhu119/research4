#!/usr/bin/env python3
"""Let an authenticated preflight audit finish before the detached retry queue."""
import argparse
import gzip
from pathlib import Path
import shutil
import time

import background_all_round_validation_20261112 as background
import verify_all_round_ml_20261112 as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    out = root / 'reports/all_round_validation_20261112/ml'
    lease = background.read(out / 'preflight_worker_identity.json', {})
    while any(background.alive(item) for item in lease.get('processes', [])):
        background.write(out / 'adoption.json', {
            'status': 'WAITING_FOR_AUTHENTICATED_PREFLIGHT',
            'processes': lease['processes'], 'at_utc': background.utc(),
            'duplicate_audit_started': False})
        time.sleep(2)
    summary = background.read(out / 'summary.json', {})
    if summary.get('all_requested_checks_passed') is not True:
        code = audit.main(['--root', str(root)])
        if code:
            return code
    # Detailed independently recounted settings stay inspectable without
    # exceeding Git's single-file transport limit. Original local bytes stay.
    compressed = []
    for source in sorted((out / 'species_predictions').glob('*.jsonl')):
        if source.name == 'jobs.jsonl':
            continue
        destination = source.with_suffix(source.suffix + '.gz')
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        with source.open('rb') as stream, temporary.open('wb') as target:
            with gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0) as packed:
                shutil.copyfileobj(stream, packed, length=1024 * 1024)
        temporary.replace(destination)
        compressed.append({'source': str(source.relative_to(root)),
                           'source_sha256': background.sha(source),
                           'source_bytes': source.stat().st_size,
                           'gzip': str(destination.relative_to(root)),
                           'gzip_sha256': background.sha(destination),
                           'gzip_bytes': destination.stat().st_size})
    background.write(out / 'adoption.json', {
        'status': 'COMPLETE', 'compressed_settings': compressed,
        'full_audit_passed': True, 'at_utc': background.utc()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
