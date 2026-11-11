#!/usr/bin/env python3
"""Verify and restore the initial checkpoint backup into a new run directory."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile


def restore(backup, output):
    backup, output = Path(backup).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Restore requires a new output directory: ' + str(output))
    manifest = json.loads((backup / 'manifest.json').read_text())
    prepared = {}
    for entry in manifest['files']:
        source = backup / entry['archive_path']
        relative = Path(entry['restore_relative_to_run_root'])
        if (not source.resolve().is_relative_to(backup) or relative.is_absolute()
                or '..' in relative.parts or relative in prepared):
            raise ValueError('Invalid or repeated backup path')
        compressed = source.read_bytes()
        if hashlib.sha256(compressed).hexdigest() != entry['compressed_sha256']:
            raise ValueError('Compressed checkpoint hash differs: ' + str(relative))
        content = gzip.decompress(compressed)
        if hashlib.sha256(content).hexdigest() != entry['uncompressed_sha256']:
            raise ValueError('Checkpoint content hash differs: ' + str(relative))
        prepared[relative] = content
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.meta-restore-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'run'
        stage.mkdir()
        for relative, content in prepared.items():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        os.rename(stage, output)
    return {'status': 'RESTORED_VERIFIED_INITIAL_BACKUP', 'files': len(prepared),
            'output': str(output), 'backup_observed_at_utc': manifest['observed_at_utc']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', type=Path, default=Path(__file__).resolve().parents[1] /
                        'data/research/meta_learning_20261111/launch_backup')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.backup, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
