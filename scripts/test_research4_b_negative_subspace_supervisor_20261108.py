"""V7 closed history, no repeated extension and independent quantum chains."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest import mock

from scripts import test_research4_b_validation_supervisor_recovery_v6_20261107 as previous_fixture
from scripts import test_research4_b_diagnostic_supervisor_20261106 as initial_fixture

SOURCE = Path(__file__).with_name("research4_b_negative_subspace_supervisor_20261108.py")


class ClosedHistoryTests(unittest.TestCase):
    def setUp(self):
        original_setup = initial_fixture.SupervisorTests.setUp
        def epoch_complete_fixture(fixture):
            # The old test harness used a minimal CLOSED parent without an
            # identity. Populate this synthetic pre-boot record before any
            # higher-generation hashes are taken; real production data is
            # never changed and already contains its complete identities.
            original_setup(fixture)
            file = fixture.original / "state.json"; state = fixture.m.read(file)
            state["invocations"][0].update(identity={"pid": 999999, "start_ticks": 0, "uid": 1000,
                "pgrp": 999999, "session": 999999, "boot_id": "11111111-1111-1111-1111-111111111111"},
                ended_at_utc="2020-01-01T00:00:00+00:00")
            fixture.m.save(file, state); fixture.manifest["carry_state"] = fixture.m.ref(file)
        self.prior = previous_fixture.ExtensionTests()
        with mock.patch.object(initial_fixture.SupervisorTests, "setUp", epoch_complete_fixture):
            self.prior.setUp()
        completed = self.prior.execute(); self.assertEqual(completed.returncode, 0, completed.stderr)
        self.root = self.prior.root; self.python = self.prior.python
        self.source = self.root / "scripts" / SOURCE.name; shutil.copyfile(SOURCE, self.source)
        spec = importlib.util.spec_from_file_location("r4_v7_test", self.source)
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)
        self.worker = self.root / "scripts/research4_b_negative_subspace_worker_20261108.py"
        self.worker.write_text(self.prior.worker.read_text().replace(
            "from research4_b_validation_supervisor_recovery_v6_20261107 import", "from research4_b_negative_subspace_supervisor_20261108 import"))
        self.work = self.prior.work.parent / "followup-negative-subspace-v7"; self.work.mkdir()
        state = self.m.read(self.prior.work / "state.json")
        self.manifest = {key: self.prior.manifest[key] for key in ("run_id", "branch", "cpu_cap_seconds", "deadline_utc",
            "threads", "memory_gib", "rss_stop_gib", "reserve_cpu_seconds", "python_executable")}
        self.manifest.update(schema=self.m.MANIFEST_SCHEMA, work_directory=str(self.work.relative_to(self.root)),
            ledger=str((self.work / "cpu_budget.json").relative_to(self.root)),
            predecessor_manifest=self.m.ref(self.prior.work / "manifest.json"), predecessor_ledger=self.m.ref(self.prior.work / "cpu_budget.json"),
            predecessor_state=self.m.ref(self.prior.work / "state.json"), predecessor_supervisor=self.m.ref(self.prior.source),
            predecessor_allocation=self.m.ref(self.root / ".runtime/worker-b-validation-recovery-v6-allocation.json"),
            inherited_results=state["results"], recovery_boot_id=self.m.current_epoch()[0],
            historical_extension_claim=self.m.ref(self.root / self.m.EXTENSION_CLAIM),
            extension_authorization=self.prior.manifest["extension_authorization"],
            source_refs=[self.m.ref(self.source), self.m.ref(self.worker)], input_refs=[], smoke_job_id="acid-new-minus",
            fail_fast_quantum_groups={"acid-base": ["acid-new-minus", "acid-new-plus"], "caly-base": ["caly-new-minus", "caly-new-plus"]})
        receipt = self.work / "previous_return.json"; self.m.save(receipt, {"status": "TEST_RETURN_VERIFIED"})
        review = {"schema": "research4-b-v6-independent-completion-review-20261108-v1",
            "status": "PASS_COMPLETE_CLOSED_SEALED_AND_ACTUAL_REMOTE_RETURN_VERIFIED", "evidence": {
                "manifest": self.manifest["predecessor_manifest"], "ledger": self.manifest["predecessor_ledger"],
                "state": self.manifest["predecessor_state"], "source_supervisor": self.manifest["predecessor_supervisor"],
                "active_allocation": self.manifest["predecessor_allocation"],
                "single_use_extension_claim": self.manifest["historical_extension_claim"], "return_receipt": self.m.ref(receipt)}}
        review_path = self.work / "closure_review.json"; self.m.save(review_path, review)
        self.manifest["predecessor_closure_review"] = self.m.ref(review_path)
        argv = [str(self.python.relative_to(self.root)), str(self.worker.relative_to(self.root)), '--output', '{output}']
        jobs = []
        for case in ('acid', 'caly'):
            for sign in ('minus', 'plus'):
                job = {'id': case+'-new-'+sign, 'argv': list(argv), 'cpu_seconds': 20, 'wall_seconds': 61, 'max_attempts': 1,
                    'depends_on': ['v5-center'] + ([case+'-new-minus'] if sign == 'plus' else []), 'admission_group': case+'-pair',
                    'result_schema': 'test-v6-point', 'success_statuses': ['POINT_COMPUTED']}
                if case == 'acid' and sign == 'minus': job['smoke_success_gate'] = True
                jobs.append(job)
        self.manifest['jobs'] = jobs

    def tearDown(self): self.prior.tearDown()

    def execute(self, attempt='v7-test-001', execute=True):
        p=self.work/'manifest.json'; self.m.save(p,self.manifest)
        argv=[str(self.python),str(self.source),'--manifest',str(p),'--manifest-sha256',self.m.sha(p),'--attempt',attempt]
        if execute: argv.append('--execute')
        return subprocess.run(argv,cwd=self.root,text=True,capture_output=True,timeout=25)

    def test_full_history_audit_never_calls_old_live_validate(self):
        original=self.m.load_frozen
        def closed_only(reference,generation):
            module=original(reference,generation)
            module.validate=mock.Mock(side_effect=AssertionError('old live validate forbidden'))
            module.validate_carry=mock.Mock(side_effect=AssertionError('old live carry forbidden'))
            return module
        with mock.patch.object(self.m,'load_frozen',side_effect=closed_only):
            carry=self.m.validate(self.manifest,'0'*64)
        self.assertEqual(carry['new_cpu_authorization_seconds'],0)
        self.assertEqual(carry['conservative_debit_cpu_seconds'],1000)
        self.assertEqual(carry['inherited_success_count'],len(self.manifest['inherited_results']))

    def test_cross_boot_only_checks_closed_epoch_not_reused_numeric_pid(self):
        row={'status':'EXITED','ended_at_utc':'2020-01-01T00:00:00+00:00','child_identity':{'boot_id':'11111111-1111-1111-1111-111111111111'}}
        with mock.patch.object(self.m,'current_epoch',return_value=('22222222-2222-2222-2222-222222222222',1800000000)), \
             mock.patch.object(self.m,'is_live',side_effect=AssertionError('old PID lookup forbidden')):
            self.m.audit_closed_epochs([row],[])

    def test_unreviewed_unknown_is_not_a_cross_boot_exemption(self):
        with self.assertRaisesRegex(ValueError,'Unreviewed unknown'):
            self.m.audit_closed_epochs([{'status':'BLOCKED_UNACCOUNTED'}],[])

    def test_original_legacy_forensic_debit_cannot_be_refunded(self):
        node=self.m.read(self.manifest['predecessor_manifest']['path'])
        while 'diagnostic-recovery-manifest-v2' not in node['schema']:
            node=self.m.read(node['predecessor_manifest']['path'])
        resolution=self.m.read(node['forensic_resolution']['path']);resolution['conservative_debit_cpu_seconds']=0
        file=self.work/'altered_forensic_test.json';self.m.save(file,resolution);node['forensic_resolution']=self.m.ref(file)
        with self.assertRaisesRegex(ValueError,'permanently debited'):
            self.m.audit_legacy_v1_history(node)

    def test_success_reuses_all_results_and_never_rewrites_extension(self):
        claim=self.m.path(self.m.EXTENSION_CLAIM);before=claim.read_bytes()
        result=self.execute();self.assertEqual(result.returncode,0,result.stderr)
        state=self.m.read(self.work/'state.json');ledger=self.m.read(self.work/'cpu_budget.json')
        self.assertEqual(claim.read_bytes(),before)
        self.assertEqual(ledger['new_cpu_authorization_seconds'],0)
        for key,value in self.manifest['inherited_results'].items():self.assertEqual(state['results'][key],value)
        self.assertEqual(len(ledger['phases']),4)

    def test_acid_smoke_failure_does_not_block_independent_caly_chain(self):
        self.manifest['jobs'][0]['argv'].append('--bad-norm')
        result=self.execute();self.assertEqual(result.returncode,0,result.stderr)
        state=self.m.read(self.work/'state.json');rows=self.m.read(self.work/'cpu_budget.json')['phases']
        self.assertEqual(state['jobs']['acid-new-plus']['status'],'BLOCKED_DEPENDENCY')
        self.assertEqual(state['jobs']['caly-new-plus']['status'],'DONE_SEALED')
        self.assertEqual([row['job_id'] for row in rows],['acid-new-minus','caly-new-minus','caly-new-plus'])

    def test_lost_new_ledger_blocks_restart_without_reclaiming_old_extension(self):
        result=self.execute();self.assertEqual(result.returncode,0,result.stderr)
        claim=self.m.path(self.m.EXTENSION_CLAIM);before=claim.read_bytes();(self.work/'cpu_budget.json').unlink()
        result=self.execute('v7-test-002');self.assertNotEqual(result.returncode,0);self.assertIn('ledger is missing',result.stderr)
        self.assertEqual(claim.read_bytes(),before)

    def test_default_only_checks_and_does_not_allocate(self):
        result=self.execute(execute=False);self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['new_cpu_authorization_seconds'],0)
        self.assertFalse(self.m.path('.runtime/worker-b-negative-subspace-allocation.json').exists())

    def test_group_cannot_omit_a_quantum_point(self):
        self.manifest['fail_fast_quantum_groups']['caly-base'].pop()
        with self.assertRaisesRegex(ValueError,'cover each point exactly once'):
            self.m.validate(self.manifest,'0'*64)


if __name__=='__main__': unittest.main()
