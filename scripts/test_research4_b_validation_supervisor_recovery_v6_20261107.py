"""V6 explicit ceiling extension and complete smoke gate; no quantum work."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from scripts import test_research4_b_validation_supervisor_recovery_20261107 as prior_fixture

SOURCE = Path(__file__).with_name("research4_b_validation_supervisor_recovery_v6_20261107.py")


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.prior = prior_fixture.RecoveryTests(); self.prior.setUp()
        text = self.prior.worker.read_text().replace("'status':'TEST_COMPLETED'",
            "'status':('SCF_NOT_QUALIFIED' if 'smoke-' in a.output else 'TEST_COMPLETED')")
        self.prior.worker.write_text(text)
        self.prior.manifest["source_refs"] = [self.prior.m.ref(self.prior.source), self.prior.m.ref(self.prior.worker)]
        self.prior.manifest["jobs"] = [self.prior.job("v5-center"), self.prior.job("smoke-minus", max_attempts=2),
                                       self.prior.job("smoke-plus", max_attempts=2)]
        previous = self.prior.execute(); self.assertEqual(previous.returncode, 0, previous.stderr)
        self.root = self.prior.root; self.python = self.prior.python
        self.source = self.root / "scripts" / SOURCE.name; shutil.copyfile(SOURCE, self.source)
        spec = importlib.util.spec_from_file_location("r4_v6_test", self.source)
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)
        self.work = self.prior.work.parent / "followup-validation-recovery-v6"; self.work.mkdir()
        self.worker = self.root / "scripts/research4_b_validation_recovery_worker_v6_20261107.py"
        self.worker.write_text('''from pathlib import Path
import argparse,hashlib,json,struct
from research4_b_validation_supervisor_recovery_v6_20261107 import verify_resource_supervision
p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--bad-norm',action='store_true');p.add_argument('--missing-gradient',action='store_true');a=p.parse_args()
proof=verify_resource_supervision(expected_worker_path=__file__,expected_threads=2)
out=Path(a.output);out.mkdir(parents=True)
header=repr({'descr':'<f8','fortran_order':False,'shape':(1,3)})
header=header+' '*(117-len(header))+'\\n'
payload=b'\\x93NUMPY\\x01\\x00'+struct.pack('<H',len(header))+header.encode()+struct.pack('<ddd',.01,.02,.03)
arrays={}
for name in ('gradient.npy','geometry_bohr.npy','direction.npy','predicted_Hq.npy'):
 f=out/name;f.write_bytes(payload);arrays[name]={'path':str(f),'sha256':hashlib.sha256(payload).hexdigest()}
if a.missing_gradient: del arrays['gradient.npy']
result={'schema':'test-v6-point','status':'POINT_COMPUTED','operation':'fixed_grid_point','step_bohr':.001,'sign':-1,'scf_converged':True,'scf_qualified':True,'scf_qualification':{'qualified':True,'orbital_gradient_finite':True,'orbital_gradient_norm_Eh':1e-5 if a.bad_norm else 1e-10,'qualification_norm_threshold_Eh':1e-8},'arrays':arrays,'output_directory':str(out),'resource_supervision':proof}
(out/'result.json').write_text(json.dumps(result))
''')
        state = self.m.read(self.prior.work / "state.json"); ledger = self.m.read(self.prior.work / "cpu_budget.json")
        self.manifest = {key: self.prior.manifest[key] for key in ("run_id", "branch", "deadline_utc", "threads",
            "memory_gib", "rss_stop_gib", "reserve_cpu_seconds", "python_executable")}
        self.manifest.update(schema=self.m.MANIFEST_SCHEMA, cpu_cap_seconds=90000,
            work_directory=str(self.work.relative_to(self.root)), ledger=str((self.work / "cpu_budget.json").relative_to(self.root)),
            predecessor_manifest=self.m.ref(self.prior.work / "manifest.json"), predecessor_ledger=self.m.ref(self.prior.work / "cpu_budget.json"),
            predecessor_state=self.m.ref(self.prior.work / "state.json"), predecessor_supervisor=self.m.ref(self.prior.source),
            predecessor_allocation=self.m.ref(self.root / ".runtime/worker-b-validation-recovery-allocation.json"),
            inherited_results=state["results"], recovery_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            source_refs=[self.m.ref(self.source), self.m.ref(self.worker)], input_refs=[], jobs=[],
            smoke_job_id="smoke-minus", fail_fast_quantum_job_ids=["smoke-minus", "smoke-plus"])
        receipt = self.work / "previous_return.json"; self.m.save(receipt, {"status": "TEST_PREVIOUS_RETURN_VERIFIED"})
        review = {"schema": "research4-b-independent-v5-closure-remote-review-v1", "status": "PASS_CLOSED_PARTIAL_RESULTS_AND_PREVIOUS_REMOTE_VERIFICATION",
            "v5_manifest": self.manifest["predecessor_manifest"], "v5_ledger": self.manifest["predecessor_ledger"],
            "v5_state": self.manifest["predecessor_state"], "v5_supervisor": self.manifest["predecessor_supervisor"],
            "persistent_v5_allocation": self.manifest["predecessor_allocation"], "return_receipt": self.m.ref(receipt),
            "current_boot_observation": {"current_boot_id": self.manifest["recovery_boot_id"]}}
        review_path = self.work / "closure_review.json"; self.m.save(review_path, review)
        self.manifest["predecessor_closure_review"] = self.m.ref(review_path)
        self.grant_path = self.work / "extension_authorization.json"
        self.grant = {"schema": "research4-b-cpu-extension-authorization-v1", "grant_id": "test-single-use-90000",
            "run_id": self.m.RUN_ID, "approved": True, "single_use": True, "user_specified_numeric_total": 90000,
            "original_cpu_cap_seconds": 58800, "additional_cpu_seconds": 31200, "effective_cpu_cap_seconds": 90000,
            "deadline_utc": self.m.DEADLINE, "branch": self.m.BRANCH, "recovery_boot_id": self.manifest["recovery_boot_id"],
            "initial_actual_closed_cpu_seconds": ledger["actual_closed_cpu_seconds"], "permanent_conservative_debit_cpu_seconds": 1000,
            "initial_budget_charged_cpu_seconds": ledger["budget_charged_cpu_seconds"],
            "new_remaining_before_execution_cpu_seconds": 90000-ledger["budget_charged_cpu_seconds"],
            **{key: self.manifest[key] for key in ("predecessor_manifest", "predecessor_ledger", "predecessor_state", "predecessor_allocation")}}
        self.write_grant()
        argv = [str(self.python.relative_to(self.root)), str(self.worker.relative_to(self.root)), '--output', '{output}']
        self.manifest["jobs"] = [{"id": "smoke-minus", "argv": argv, "cpu_seconds": 20, "wall_seconds": 61,
            "max_attempts": 1, "depends_on": ["v5-center"], "admission_group": "smoke-pair", "smoke_success_gate": True,
            "result_schema": "test-v6-point", "success_statuses": ["POINT_COMPUTED"]},
            {"id": "smoke-plus", "argv": argv, "cpu_seconds": 20, "wall_seconds": 61, "max_attempts": 1,
             "depends_on": ["v5-center", "smoke-minus"], "admission_group": "smoke-pair", "result_schema": "test-v6-point",
             "success_statuses": ["POINT_COMPUTED"]}]

    def tearDown(self): self.prior.tearDown()

    def write_grant(self):
        self.m.save(self.grant_path, self.grant); self.manifest["extension_authorization"] = self.m.ref(self.grant_path)

    def execute(self, attempt="v6-test-001", execute=True):
        p = self.work / "manifest.json"; self.m.save(p, self.manifest)
        command = [str(self.python), str(self.source), '--manifest', str(p), '--manifest-sha256', self.m.sha(p), '--attempt', attempt]
        if execute: command.append('--execute')
        return subprocess.run(command, cwd=self.root, text=True, capture_output=True, timeout=20)

    def test_extension_adds_ceiling_without_charging_or_refunding_cost(self):
        carry = self.m.validate(self.manifest, '0'*64); old = self.m.read(self.prior.work/'cpu_budget.json')
        self.assertEqual(carry['actual_closed_cpu_seconds'], old['actual_closed_cpu_seconds'])
        self.assertEqual(carry['budget_charged_cpu_seconds'], old['budget_charged_cpu_seconds'])
        self.assertEqual(carry['additional_cpu_authorized_seconds'], 31200)
        self.assertEqual(carry['conservative_debit_cpu_seconds'], 1000)
        self.assertEqual(carry['inherited_success_count'], 2)

    def test_default_does_not_claim_extension_or_launch(self):
        result=self.execute(execute=False);self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['effective_cpu_cap_seconds'],90000)
        self.assertFalse(self.m.path(self.m.EXTENSION_CLAIM).exists())

    def test_complete_smoke_then_partner_seal(self):
        result=self.execute();self.assertEqual(result.returncode,0,result.stderr)
        state=self.m.read(self.work/'state.json');ledger=self.m.read(self.work/'cpu_budget.json')
        self.assertTrue(all(saved['status']=='DONE_SEALED' for saved in state['jobs'].values()))
        self.assertEqual(len(ledger['phases']),2)
        self.assertEqual(ledger['additional_cpu_authorized_seconds'],31200)
        self.assertAlmostEqual(ledger['actual_closed_cpu_seconds']-ledger['initial_actual_closed_cpu_seconds'],sum(row['actual_cpu_seconds'] for row in ledger['phases']))
        self.assertAlmostEqual(ledger['budget_charged_cpu_seconds']-ledger['actual_closed_cpu_seconds'],1000)

    def test_inadequate_fock_norm_blocks_all_later_quantum(self):
        self.manifest['jobs'][0]['argv']=[*self.manifest['jobs'][0]['argv'],'--bad-norm']
        result=self.execute();self.assertEqual(result.returncode,0,result.stderr)
        state=self.m.read(self.work/'state.json');ledger=self.m.read(self.work/'cpu_budget.json')
        self.assertEqual(state['jobs']['smoke-minus']['status'],'FAILED_RESULT_VALIDATION_CLOSED')
        self.assertEqual(state['jobs']['smoke-plus']['status'],'BLOCKED_DEPENDENCY')
        self.assertEqual(len(ledger['phases']),1)

    def test_missing_full_gradient_is_not_smoke_success(self):
        self.manifest['jobs'][0]['argv']=[*self.manifest['jobs'][0]['argv'],'--missing-gradient']
        result=self.execute();self.assertEqual(result.returncode,0,result.stderr)
        state=self.m.read(self.work/'state.json')
        self.assertNotIn('smoke-minus',state['results']);self.assertEqual(state['jobs']['smoke-plus']['status'],'BLOCKED_DEPENDENCY')

    def test_extension_marker_alone_with_lost_ledger_never_resets(self):
        result=self.execute();self.assertEqual(result.returncode,0,result.stderr)
        self.m.path('.runtime/worker-b-validation-recovery-v6-allocation.json').unlink()
        (self.work/'cpu_budget.json').unlink()
        result=self.execute('v6-test-002');self.assertNotEqual(result.returncode,0);self.assertIn('ledger is missing',result.stderr)

    def test_same_grant_cannot_bind_another_manifest(self):
        result=self.execute();self.assertEqual(result.returncode,0,result.stderr)
        self.m.path('.runtime/worker-b-validation-recovery-v6-allocation.json').unlink()
        carry=self.m.validate(self.manifest,'0'*64)
        with self.assertRaisesRegex(ValueError,'cannot be claimed again'):
            self.m.register_allocation(self.manifest,'1'*64,carry)

    def test_grant_cannot_silently_refund_conservative_debit(self):
        self.grant['permanent_conservative_debit_cpu_seconds']=0;self.write_grant()
        with self.assertRaisesRegex(ValueError,'all existing charges remain'):
            self.m.validate(self.manifest,'0'*64)

    def test_old_attempt_cannot_be_reset_despite_new_budget(self):
        self.manifest['jobs'][0]['max_attempts']=2
        with self.assertRaisesRegex(ValueError,'cannot reset historical attempt'):
            self.m.validate(self.manifest,'0'*64)

    def test_fail_fast_dependency_is_mandatory(self):
        self.manifest['jobs'][1]['depends_on']=['v5-center']
        with self.assertRaisesRegex(ValueError,'preceding quantum success'):
            self.m.validate(self.manifest,'0'*64)


if __name__=='__main__': unittest.main()
