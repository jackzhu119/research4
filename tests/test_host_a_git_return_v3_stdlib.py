"""A-only exceptional failed-start guard tests, without Git or Q execution."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/research4_host_a_git_return_20261106_v3.py'
spec = importlib.util.spec_from_file_location('host_a_return_v3_stdlib', SCRIPT)
M = importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
LEDGER = ROOT / M.A_LEDGER_RELATIVE
ACTUAL = json.loads(LEDGER.read_text())
KNOWN = next(x for x in ACTUAL['phases'] if x.get('phase') == M.KNOWN_PREEXEC_PHASE)


class HistoricalPreexecException(unittest.TestCase):
    def test_exact_real_complete_row_and_log_without_proof_is_accepted_only_as_failed_start(self):
        self.assertEqual(M.canonical_row_sha256(KNOWN), M.KNOWN_PREEXEC_ROW_SHA256)
        self.assertTrue(M.known_historical_preexec_failure(KNOWN, LEDGER))
        self.assertFalse(Path(KNOWN['supervisor_proof_path']).exists())

    def test_any_altered_or_unknown_failure_is_not_exempt_and_still_requires_proof(self):
        for field,value in [('phase','unknown-preexec-attempt'),('failure','other failure'),
                            ('actual_cpu_seconds',0.1),('actual_cpu_seconds',0),('threads',1),
                            ('elapsed_wall_seconds',0.5)]:
            changed=deepcopy(KNOWN);changed[field]=value
            with self.subTest(field=field,value=value):
                self.assertFalse(M.known_historical_preexec_failure(changed, LEDGER))
                with self.assertRaises(M.SafetyError):M.strict_failed_start_proof(changed, LEDGER)

    def test_known_row_cannot_exempt_a_different_or_copied_ledger(self):
        with self.assertRaises(M.SafetyError):
            M.known_historical_preexec_failure(KNOWN,ROOT/'reports/copied-ledger.json')

    def test_only_exact_host_A_manifest_and_branch_are_accepted(self):
        original={'schema':'research4-root-host-a-result-return-v1','role':'A','run_id':'host-a-003',
                  'upload':{'enabled':True,'branch':M.BRANCH_A}}
        self.assertEqual(M.expected_branch(original),M.BRANCH_A)
        for change in [{'role':'B'},{'run_id':'other-run'},{'run_id':'host-a-001'},{'run_id':'host-a-002'},{'schema':'research4-remote-b-manifest-v1'},
                       {'upload':{'enabled':True,'branch':'main'}}]:
            with self.assertRaises(M.SafetyError):M.expected_branch({**original,**change})
        self.assertFalse(hasattr(M,'claim_result_branch'))

    def test_frozen_B_helper_and_A_ledger_remain_unmodified(self):
        source=ROOT/'scripts/research4_parallel_b_git_return_20261106.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),M.FORKED_B_HELPER_SHA256)
        self.assertEqual(M.KNOWN_PREEXEC_ROW_SHA256,'03a2bdade1ba879cb0e6c4ec0167c9b921432599016004f3b996c144a3e66517')


if __name__=='__main__':unittest.main()
