"""Shared paths and evidence readers for the focused, corrected-data follow-up."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPORT_DIR = ROOT / 'reports/focused_followup_20261116'
CACHE_DIR = ROOT / '.cache/focused_followup_20261116'
BACKGROUND_DIR = CACHE_DIR / 'background'
PINNED_PAPER = '47f3f69a8cf6d0cce063da702e818f5e093537de'
PINNED_V4 = '71a303e089c7c7b7b2557d62372aaceda44e2f91'
PINNED_BACKGROUND = '93a607c8ce206a5bbbff22538233bdd0929bd04d'
PINNED_ARCHIVE = '7a145aaab9be8a8d969452352b9c8e4a9697c6f1'
INPUT_SHA256 = '76bde1b77584f9a6aba0e97cbb4fee2cb702f2da727fd26084c1f16209217b73'

def now():
    return datetime.now(timezone.utc).isoformat()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)

def load_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def load_split(job_dir):
    job_dir = Path(job_dir)
    for name in ('split_frozen.json', 'split_frozen.json.gz'):
        if (job_dir / name).is_file():
            return load_json(job_dir / name)
    raise FileNotFoundError('Missing actual frozen split: ' + str(job_dir))

def job_results(background_dir=BACKGROUND_DIR):
    for path in sorted((Path(background_dir) / 'jobs').glob('*/result.json')):
        yield load_json(path), path

def snv(x):
    import numpy as np
    return (x - x.mean(1, keepdims=True)) / np.maximum(
        x.std(1, ddof=1, keepdims=True), np.finfo(float).eps)

def alpha_indices(indices):
    """Background global alpha coordinates are 612..1223, including alpha-only."""
    import numpy as np
    ids = np.asarray(indices, dtype=int)
    return ids[(ids >= 612) & (ids < 1224)] - 612

# Conventional aliases retained for the small per-module entry points.
dump = atomic_json
filehash = sha256_file
digest = canonical_hash
