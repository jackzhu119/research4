"""Queue and provenance regressions; these tests do not perform SCF fits."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_all_round_quantum_20261112 as q


def test_target_failure_does_not_starve_later_molecules(tmp_path, monkeypatch):
    q.configure(tmp_path)
    monkeypatch.setattr(q, 'refresh', lambda root: None)
    monkeypatch.setattr(q, 'prepare_protocol', lambda root: None)
    monkeypatch.setattr(q.publication, 'water_control', lambda: None)
    called = []
    def recover(root, name):
        called.append(name)
        if name == 'calycosin':
            raise RuntimeError('SCF needs another checkpoint attempt')
    monkeypatch.setattr(q, 'publication_molecule', recover)
    monkeypatch.setattr(q, 'summarize', lambda root: {'publication': {'status': 'RUNNING_OR_WAITING'}})
    assert q.publication_once(tmp_path)['status'] == 'RUNNING_OR_WAITING'
    assert called == ['acid', 'calycosin', 'spinosin']
    receipt = q.read(tmp_path / q.TASK / 'publication_retry_state.json')
    assert receipt['errors'][0]['molecule'] == 'calycosin'


def test_selected_qualified_continuation_is_replayed_and_not_recomputed(tmp_path, monkeypatch):
    q.configure(tmp_path)
    selected = {'paper_ready': True, 'selected': {'path': 'source-case'}}
    monkeypatch.setattr(q, 'selected_publication', lambda root, name: selected)
    seen = []
    monkeypatch.setattr(q, 'replay_selected', lambda root, value: seen.append(value) or True)
    monkeypatch.setattr(q.publication, 'molecule_recovery', lambda name: pytest.fail('qualified result recomputed'))
    assert q.publication_molecule(tmp_path, 'acid') is selected
    assert seen == [selected]


def test_sampling_skips_large_step_and_continues_to_smaller_resolution():
    cycle, h = q.cycle_resolution(7, np.zeros((3, 3)))
    assert cycle == 8
    assert 0 < h <= .02
    assert q.cycle_resolution(cycle + 1, np.zeros((3, 3)))[0] > cycle


def test_incompatible_checkpoint_archive_is_outside_scientific_case(tmp_path):
    q.configure(tmp_path)
    point = tmp_path / q.TASK / 'publication/jobs/acid/case/finite-differences/h0/c000-plus'
    point.mkdir(parents=True)
    (point / 'result.json').write_text('{"source":true}')
    original_hash = q.sha(point / 'result.json')
    q._archive_incompatible(point, 'different exact density fingerprint')
    assert not point.exists()
    assert list(point.parent.glob('c*')) == []
    archived = next((tmp_path / q.TASK / 'archived_checkpoints').glob('*'))
    receipt = q.read(archived / 'consumer_cache_compatibility.json')
    assert receipt['original_files_sha256']['result.json'] == original_hash
    assert not archived.is_relative_to(tmp_path / q.TASK / 'publication')


class Reader:
    def __init__(self, blobs):
        self.blobs = blobs
        self.objects = {p: 'immutable-source-blob' for p in blobs}
    def read(self, path):
        return self.blobs[path]
    def read_many(self, paths):
        return {p: self.read(p) for p in paths}


def test_corrupt_original_complete_seal_rejected_before_relocation():
    blobs = {'reports/pyscf_ir_20261111/jobs/source/round000/result.json': b'{"paper_ready":true}',
             'reports/pyscf_ir_20261111/jobs/source/round000/COMPLETE.json':
                 b'{"result_sha256":"invalid"}'}
    with pytest.raises(ValueError, match='Original source COMPLETE seal mismatch'):
        q.authenticate_originals(Reader(blobs), list(blobs), 'ir')


def test_original_point_input_fingerprint_is_authenticated():
    prefix = 'reports/pyscf_ir_20261111/derivative_pools/source/center/'
    inputs = {'schema': q.dipoles.SCHEMA, 'geometry': [0, 1, 2]}
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, allow_nan=False).encode()).hexdigest()
    blobs = {prefix + 'input.json': json.dumps({**inputs, 'input_fingerprint': digest}).encode(),
             prefix + 'result.json': json.dumps({'status': 'DONE', 'array_refs': {},
                                                   'input_fingerprint': digest}).encode()}
    checks = q.authenticate_originals(Reader(blobs), list(blobs), 'ir')
    assert checks['partial_point_inputs'] == 1
    blobs[prefix + 'input.json'] = json.dumps({**inputs, 'geometry': [9], 'input_fingerprint': digest}).encode()
    with pytest.raises(ValueError, match='Original point input fingerprint mismatch'):
        q.authenticate_originals(Reader(blobs), list(blobs), 'ir')
