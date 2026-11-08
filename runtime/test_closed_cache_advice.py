"""Small filesystem fixtures with fadvise mocked: no real cache eviction."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('closed_cache_under_test', HERE/'closed_cache_advice.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CacheAdviceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='unit-cache-', dir=HERE)
        self.root = Path(self.temporary.name)
        self.old = self.root/'old'
        self.objects = self.root/'old-objects'
        self.backfill = self.root/'backfill'
        for p in [self.old/'control/receipts', self.old/'outputs', self.objects, self.backfill]:
            p.mkdir(parents=True)
        self.memory = self.root/'memory.current'
        self.memory.write_text('1234567\n')
        self.jobs = {}
        self.write_state()
        self.patches = [patch.object(m, 'OLD', self.old),
                        patch.object(m, 'OLD_OBJECTS', self.objects),
                        patch.object(m, 'BACKFILL', self.backfill),
                        patch.object(m, 'MEMORY_CURRENT', self.memory),
                        patch.object(m.os, 'posix_fadvise')]
        for p in self.patches:
            value = p.start()
            if p is self.patches[-1]:
                self.advice = value

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temporary.cleanup()

    def write_state(self):
        (self.old/'control/state.json').write_text(json.dumps({'jobs': self.jobs}))

    def large(self, name, root=None):
        p = (self.old/'outputs' if root is None else root)/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'x' * m.MIN_BYTES)
        return p

    def seal(self, job, outputs, status='completed'):
        payload = json.dumps({'output_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in outputs}}).encode()
        receipt = self.old/'control/receipts'/(job+'.json')
        receipt.write_bytes(payload)
        self.jobs[job] = {'status': status, 'receipt_sha256': hashlib.sha256(payload).hexdigest()}
        self.write_state()
        return receipt

    def test_completed_output_hash_chain_and_readonly_contents(self):
        p = self.large('result.json')
        self.seal('job1', [p])
        before = p.stat()
        checksum = hashlib.sha256(p.read_bytes()).hexdigest()
        result = m.advise_closed()
        self.assertEqual(result['files'], 1)
        self.assertEqual(result['bytes'], m.MIN_BYTES)
        self.assertEqual(result['verified_completed_receipts'], 1)
        self.assertEqual(result['memory_before_bytes'], 1234567)
        self.assertEqual(result['errors'], 0)
        self.assertEqual(self.advice.call_count, 1)
        self.assertEqual(self.advice.call_args.args[1:], (0, 0, m.os.POSIX_FADV_DONTNEED))
        after = p.stat()
        self.assertEqual((before.st_ino, before.st_size, before.st_mtime_ns),
                         (after.st_ino, after.st_size, after.st_mtime_ns))
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), checksum)
        self.assertFalse(result['file_bytes_written'])
        self.assertFalse(result['signals_sent'])

    def test_tampered_receipt_does_not_authorize_output(self):
        receipt = self.seal('job1', [self.large('result.json')])
        receipt.write_bytes(receipt.read_bytes()+b' ')
        result = m.advise_closed()
        self.assertEqual(result['files'], 0)
        self.assertEqual(result['error_counts']['receipt_read_or_seal'], 1)
        self.advice.assert_not_called()

    def test_running_job_outputs_are_not_advised(self):
        self.seal('job1', [self.large('result.json')], status='running')
        self.assertEqual(m.advise_closed()['files'], 0)
        self.advice.assert_not_called()

    def test_payload_extension_size_and_outside_path_exclusions(self):
        small = self.old/'outputs/small.json';small.write_text('{}')
        self.seal('job1', [small, self.large('raw.npz'), self.large('outside.json', self.root)])
        self.assertEqual(m.advise_closed()['files'], 0)

    def test_leaf_symlink_is_excluded(self):
        target = self.large('original.json')
        link = self.old/'outputs/link.json';link.symlink_to(target)
        self.seal('job1', [link])
        self.assertEqual(m.advise_closed()['files'], 0)

    def test_ancestor_directory_symlink_is_excluded(self):
        target = self.old/'real';target.mkdir()
        payload = self.large('result.json', target)
        link = self.old/'outputs/link';link.symlink_to(target, target_is_directory=True)
        self.seal('job1', [link/payload.name])
        self.assertEqual(m.advise_closed()['files'], 0)

    def test_final_git_object_name_allowlist_only(self):
        allowed = [self.large('b'*38, self.objects/'aa'),
                   self.large('pack-'+'c'*40+'.pack', self.objects/'pack'),
                   self.large('pack-'+'d'*40+'.idx', self.objects/'pack')]
        for name in ['tmp_pack_1', 'pack-'+'c'*40+'.pack.lock', 'pack-'+'c'*40+'.keep']:
            self.large(name, self.objects/'pack')
        self.large('not-an-object', self.objects/'aa')
        self.large('e'*38, self.objects/'nothex')
        result = m.advise_closed()
        self.assertEqual(result['files'], len(allowed))
        self.assertEqual(self.advice.call_count, len(allowed))

    def test_git_symlink_object_is_excluded(self):
        target = self.large('outside.json', self.root)
        p = self.objects/'aa'/('b'*38);p.parent.mkdir();p.symlink_to(target)
        self.assertEqual(m.advise_closed()['files'], 0)

    def test_verified_backfill_parts_and_git_objects(self):
        (self.backfill/'backfill_result.json').write_text(json.dumps({'status':'DONE_REMOTE_BACKFILL_VERIFIED'}))
        self.large('closed-results.tar.gz.part001', self.backfill/'archives')
        self.large('closed-results.tar.gz.part002.partial', self.backfill/'archives')
        self.large('b'*38, self.backfill/'.git/objects/aa')
        (self.backfill/'manifest.json').write_text(json.dumps({'parts':[
            {'path':'archives/closed-results.tar.gz.part001'},
            {'path':'archives/closed-results.tar.gz.part002.partial'},
            {'path':'archives/../../outside.json'}]}))
        self.assertEqual(m.advise_closed()['files'], 2)

    def test_unverified_backfill_does_not_authorize_parts_or_objects(self):
        (self.backfill/'backfill_result.json').write_text(json.dumps({'status':'PARTIAL'}))
        self.large('b'*38, self.backfill/'.git/objects/aa')
        self.assertEqual(m.advise_closed()['files'], 0)

    def test_git_gc_unlink_race_is_a_nonfatal_count(self):
        p = self.large('b'*38, self.objects/'aa')
        self.advice.side_effect = lambda *args: p.unlink()
        result = m.advise_closed()
        self.assertEqual(result['files'], 0)
        self.assertEqual(result['error_counts']['file_advice'], 1)

    def test_fadvise_filesystem_error_does_not_abort(self):
        self.large('b'*38, self.objects/'aa')
        self.advice.side_effect = OSError('not supported')
        result = m.advise_closed()
        self.assertEqual(result['files'], 0)
        self.assertEqual(result['error_counts']['file_advice'], 1)

    def test_metadata_cap_rejects_giant_json_without_payload_read(self):
        self.seal('job1', [self.large('result.json')])
        with patch.object(m, 'SMALL_JSON_CAP', 10):
            result = m.advise_closed()
        self.assertEqual(result['files'], 0)
        self.assertEqual(result['error_counts']['receipt_read_or_seal'], 1)

    def test_missing_memory_counter_is_reported_without_stopping(self):
        self.memory.unlink()
        result = m.advise_closed()
        self.assertIsNone(result['memory_before_bytes'])
        self.assertIsNone(result['memory_after_bytes'])
        self.assertEqual(result['error_counts']['memory_read'], 2)


if __name__ == '__main__':
    unittest.main()
