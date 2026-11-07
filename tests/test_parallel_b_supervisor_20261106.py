"""Standard-library tests plus one real non-quantum wait4/proof fixture.

The fixture lives entirely in a temporary checkout, uses a Python JSON worker,
does not import numerical packages, run Git, or represent a molecular result.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'scripts/research4_parallel_b_supervisor_20261106.py'
spec=importlib.util.spec_from_file_location('b_supervisor_policy_under_test',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class PortableSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        for name in ('scripts','.runtime/test/bin','data','reports'):(self.root/name).mkdir(parents=True)
        self.source=self.root/'scripts'/SOURCE.name;shutil.copyfile(SOURCE,self.source)
        self.worker=self.root/'scripts/fixture.py';self.worker.write_text('# placeholder fixture, never quantum\n')
        self.input=self.root/'data/input.json';self.input.write_text('{"fixture":true}\n')
        self.allocation=self.root/'data/resource_allocation_ROOT_APPROVED.json';self.allocation.write_text(json.dumps({'schema':'research4-two-account-resource-allocation-v1','root_approved':True,'new_B_envelope':{'maximum_additional_CPU_seconds':60000,'local_package_validation_reserved_CPU_seconds':1200,'actual_second_account_CPU_cap_seconds':58800}}))
        self.python=self.root/'.runtime/test/bin/python';self.python.symlink_to(sys.executable)
        for key,value in {'ROOT':self.root,'SELF':self.source}.items():
            context=patch.object(b,key,value);context.start();self.addCleanup(context.stop)
        self.run_id='fixture-only'
        self.work=self.root/'reports/parallel-cloud-20261106/worker-b'/self.run_id
        self.job={'id':'fixture','case_id':'water','operation':'prepare','worker':b.ref(self.worker),'input_manifest_sha256':b.sha(self.input),'payload_argv':[str(self.python),str(self.worker),'--case','water','--operation','prepare','--execute','--output','{output}'],'depends_on':[],'result_schema':'fixture-only-no-quantum-v1','success_statuses':['FIXTURE_DONE_NOT_QUANTUM'],'required_result_fields':{'new_quantum_evaluations':0},'cpu_seconds':125,'wall_seconds':120,'optional':False,'retry_closed_resource_failure':False}
        self.manifest={'schema':b.MANIFEST_SCHEMA,'run_id':self.run_id,'work_directory':str(self.work.relative_to(self.root)),'ledger':str((self.work/'cpu_budget_B.json').relative_to(self.root)),'cpu_cap_seconds':300,'deadline_utc':b.DEADLINE,'reserve_cpu_seconds':125,'threads':1,'memory_gib':4,'rss_stop_gib':2,'python_executable':str(self.python.relative_to(self.root)),'source_refs':[b.ref(self.source),b.ref(self.worker)],'input_refs':[],'input_manifest':b.ref(self.input),'no_budget_transfer_from_A':True,'local_smoke_test_without_remote_B_claim':True,'jobs':[self.job],'upload':{'enabled':False,'branch':b.BRANCH_PREFIX+self.run_id,'allowlist':['*.json']}}
        self.auth={'schema':b.AUTH_SCHEMA,'enable_execution':True,'manifest_sha256':'FIXTURE','supervisor_sha256':b.sha(self.source),'cpu_cap_seconds':300,'deadline_utc':b.DEADLINE,'no_budget_transfer_from_A':True}
        self.manifest['allocation_ref']=b.ref(self.allocation);self.auth['allocation_ref']=b.ref(self.allocation)

    def validate(self):b.validate(self.manifest,self.auth,'FIXTURE')

    def test_valid_local_fixture_contract_and_system_python_symlink(self):
        self.validate();self.assertEqual(b.python_path(self.manifest['python_executable']),self.python)

    def test_same_preparation_reuses_original_hashes_without_resetting_running_ledger(self):
        release_file=self.root/'data/release_ROOT_APPROVED.json'
        release={'schema':'research4-parallel-b-release-v1','release_enabled':True,'supervisor_sha256':b.sha(self.source),'no_budget_transfer_from_A':True,'deadline_utc':b.DEADLINE,'worker':b.ref(self.worker),'input_manifest':b.ref(self.input),'source_refs':[b.ref(self.source),b.ref(self.worker)],'input_refs':[b.ref(self.allocation)],'cpu_cap_seconds':58800,'remote_run_id':'B20261106-01','local_smoke_run_id':self.run_id,'allow_local_smoke':True,'local_smoke_cpu_cap_seconds':1200}
        release_file.write_text(json.dumps(release))
        with patch.object(b,'RELEASE',release_file):
            first=b.prepare_manifest(self.run_id,str(self.python.relative_to(self.root)),True,True)
            saved=json.loads((self.work/'manifest.json').read_text())
            ledger={'schema':b.LEDGER_SCHEMA,'manifest_sha256':first['manifest_sha256'],'supervisor_sha256':b.sha(self.source),'run_id':self.run_id,'cpu_cap_seconds':1200,'deadline_utc':b.DEADLINE,'phases':[{'phase':'unit-fixture-live-status-not-a-quantum-run','status':'RUNNING'}],'consumed_cpu_seconds':0.,'remaining_cpu_seconds':1200.}
            ledger_file=self.work/'cpu_budget_B.json';ledger_file.write_text(json.dumps(ledger));before=ledger_file.read_bytes()
            second=b.prepare_manifest(self.run_id,str(self.python.relative_to(self.root)),True,True)
            self.assertTrue(second['existing_plan_reused']);self.assertEqual(first['manifest_sha256'],second['manifest_sha256']);self.assertEqual(first['authorization_sha256'],second['authorization_sha256']);self.assertEqual(ledger_file.read_bytes(),before)
            self.assertEqual(json.loads(ledger_file.read_text())['phases'][0]['status'],'RUNNING')

    def test_smaller_root_remaining_suballocation_is_used_without_recursive_carry(self):
        sub_file=self.root/'data/local_validation_remainder_003/resource_allocation_ROOT_APPROVED.json';sub_file.parent.mkdir()
        sub=json.loads(self.allocation.read_text());sub['new_B_envelope']['local_package_validation_reserved_CPU_seconds']=1199
        sub['parent_allocation_ref']=b.ref(self.allocation);sub['unique_closed_history_CPU_seconds']=0.196167
        sub_file.write_text(json.dumps(sub))
        release_file=self.root/'data/release_ROOT_APPROVED_v3.json'
        release={'schema':'research4-parallel-b-release-v1','release_enabled':True,'supervisor_sha256':b.sha(self.source),'no_budget_transfer_from_A':True,'deadline_utc':b.DEADLINE,'worker':b.ref(self.worker),'input_manifest':b.ref(self.input),'source_refs':[b.ref(self.source),b.ref(self.worker)],'input_refs':[b.ref(self.allocation),b.ref(sub_file)],'cpu_cap_seconds':58800,'remote_run_id':'B20261106-01','local_smoke_run_id':self.run_id,'allow_local_smoke':True,'local_smoke_cpu_cap_seconds':1199,'local_validation_allocation_ref':b.ref(sub_file),'local_validation_pythonpath':[]}
        release_file.write_text(json.dumps(release))
        with patch.object(b,'RELEASE',release_file):
            prepared=b.prepare_manifest(self.run_id,str(self.python.relative_to(self.root)),True,True)
        manifest=json.loads((self.root/prepared['manifest']).read_text());auth=json.loads((self.root/prepared['authorization']).read_text())
        self.assertEqual(manifest['cpu_cap_seconds'],1199)
        self.assertEqual(manifest['allocation_ref'],b.ref(sub_file))
        self.assertNotEqual(manifest['allocation_ref'],b.ref(self.allocation))
        self.assertNotIn('local_validation_prior_cost',manifest)
        self.assertNotIn('local_validation_prior_cost',auth)
        self.assertEqual(manifest['local_validation_pythonpath'],[])

    def test_A_budget_transfer_forbidden(self):
        self.auth['no_budget_transfer_from_A']=False
        with self.assertRaises(ValueError):self.validate()

    def test_over_cap_and_later_deadline_rejected(self):
        self.manifest['cpu_cap_seconds']=60001;self.auth['cpu_cap_seconds']=60001
        with self.assertRaises(ValueError):self.validate()
        self.manifest['cpu_cap_seconds']=300;self.auth['cpu_cap_seconds']=300
        self.manifest['deadline_utc']='2026-11-10T00:00:00+00:00';self.auth['deadline_utc']=self.manifest['deadline_utc']
        with self.assertRaises(ValueError):self.validate()

    def test_wrong_case_input_hash_and_future_dependency_rejected(self):
        self.job['input_manifest_sha256']='bad'
        with self.assertRaises(ValueError):self.validate()
        self.job['input_manifest_sha256']=b.sha(self.input);self.job['depends_on']=['self-or-future']
        with self.assertRaises(ValueError):self.validate()

    def test_main_or_wrong_B_branch_rejected(self):
        self.manifest['upload']['branch']='main'
        with self.assertRaises(ValueError):self.validate()

    def test_remote_claim_requires_local_nonce_and_verified_exact_branch(self):
        self.work.mkdir(parents=True)
        claim={'run_id':self.run_id,'nonce':'0123456789abcdef0123456789abcdef','release_sha256':'a'*64,'source_commit':'b'*40}
        (self.work/'owner_claim.json').write_text(json.dumps(claim))
        marker=self.root/'.runtime/worker-b-ownership'/(self.run_id+'.json');marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps({**claim,'remote_verified':True,'branch':b.BRANCH_PREFIX+self.run_id,'repository_root':str(self.root)}))
        actual=b.checked_owner_claim(self.run_id,'a'*64,self.work)
        self.assertEqual(actual['source_commit'],'b'*40)
        marker.write_text(json.dumps({**claim,'nonce':'other-not-owner','remote_verified':True,'branch':b.BRANCH_PREFIX+self.run_id,'repository_root':str(self.root)}))
        with self.assertRaises(ValueError):b.checked_owner_claim(self.run_id,'a'*64,self.work)

    def test_new_run_id_cannot_reissue_fixed_B_allocation(self):
        release={'remote_run_id':'B20261106-01','local_smoke_run_id':'B-water-package-validation-001'}
        b.validate_release_run_id(release,'B20261106-01',False)
        b.validate_release_run_id(release,'B-water-package-validation-001',True)
        with self.assertRaises(ValueError):b.validate_release_run_id(release,'B20261106-02',False)
        with self.assertRaises(ValueError):b.validate_release_run_id(release,'B-water-package-validation-002',True)

    def test_source_mutation_rejected(self):
        self.worker.write_text('# changed')
        with self.assertRaises(ValueError):self.validate()

    def test_local_smoke_cannot_claim_large_remote_B_allocation(self):
        self.manifest['cpu_cap_seconds']=60000;self.auth['cpu_cap_seconds']=60000
        with self.assertRaises(ValueError):self.validate()

    def test_output_path_and_unbounded_jobs_rejected(self):
        self.job['payload_argv'][-1]='reports/old-output'
        with self.assertRaises(ValueError):self.validate()
        self.job['payload_argv'][-1]='{output}';self.manifest['jobs']*=17
        with self.assertRaises(ValueError):self.validate()

    def test_PID_birth_boot_UID_reuse_is_not_same(self):
        value={'pid':5,'start_ticks':10,'uid':1000,'pgrp':5,'session':5,'boot_id':'a'}
        for key in value:
            changed=dict(value);changed[key]='different'
            self.assertFalse(b.same(value,changed))

    def test_water_gate_requires_true_independent_replay_and_gradient(self):
        p=self.root/'reports/water.json'
        result={'case':'water','operation':'analyze','status':'MODES_ANALYZED','normal_mode_count':3,'imaginary_frequency_count':0,'all_frequencies_real_positive':True,'optimizer_converged':True,'fresh_gradient_thresholds_passed':True,'final_gradient_thresholds_passed':True,'independent_mode_replay_passed':False,'stationary_minimum_verified':True}
        p.write_text(json.dumps(result));self.assertFalse(b.water_qualified({'water':b.ref(p)}))
        result['independent_mode_replay_passed']=True;p.write_text(json.dumps(result));self.assertTrue(b.water_qualified({'water':b.ref(p)}))

    def test_ledger_cannot_refund_failed_cost(self):
        row={'phase':'failed','status':'FAILED_CLOSED','actual_cpu_seconds':10.,'CPU_accounting':'ACTUAL_WAIT4_SELF_AND_WAITED_CHILDREN_SINGLE_TGID_ENFORCED'}
        ledger={'schema':b.LEDGER_SCHEMA,'manifest_sha256':'FIXTURE','supervisor_sha256':b.sha(self.source),'run_id':self.run_id,'cpu_cap_seconds':300,'deadline_utc':b.DEADLINE,'phases':[row],'consumed_cpu_seconds':0.,'remaining_cpu_seconds':300.}
        with self.assertRaises(ValueError):b.validate_ledger(ledger,self.manifest,'FIXTURE')
        ledger.update(consumed_cpu_seconds=10.,remaining_cpu_seconds=290.);b.validate_ledger(ledger,self.manifest,'FIXTURE')
        ledger['blocked_unknown_cpu_or_process_tree']=True
        with self.assertRaises(ValueError):b.validate_ledger(ledger,self.manifest,'FIXTURE')

    def prior_carry_fixture(self):
        bundle=self.root/'data/local_validation_failure_001';bundle.mkdir()
        historical_source=bundle/'supervisor_c2cb60_source.py';shutil.copyfile(self.source,historical_source)
        old_manifest=copy.deepcopy(self.manifest);old_manifest.update(run_id='old-local-water-fixture',cpu_cap_seconds=1200)
        mp=bundle/'manifest.json';mp.write_text(json.dumps(old_manifest))
        case_hash=hashlib.sha256(json.dumps(self.job,sort_keys=True).encode()).hexdigest()
        proof={'schema':b.PROOF_SCHEMA,'phase':'historical-closed-local-water-fixture','case_binding_sha256':case_hash,'manifest_sha256':b.sha(mp),'supervisor_source_sha256':b.sha(historical_source),'input_manifest_sha256':old_manifest['input_manifest']['sha256']}
        pp=bundle/'payload.proof.json';pp.write_text(json.dumps(proof));log=bundle/'payload.log';log.write_text('UNIT TEST FIXTURE: no quantum calculation, historical closure policy only\n')
        row={'phase':proof['phase'],'case_id':'water','operation':'prepare','case_binding_sha256':case_hash,'status':'FAILED_CLOSED','actual_cpu_seconds':0.101408,'CPU_accounting':'ACTUAL_WAIT4_SELF_AND_WAITED_CHILDREN_SINGLE_TGID_ENFORCED','proof_sha256':b.sha(pp),'log_sha256':b.sha(log)}
        old={'schema':b.LEDGER_SCHEMA,'run_id':old_manifest['run_id'],'manifest_sha256':b.sha(mp),'supervisor_sha256':b.sha(historical_source),'cpu_cap_seconds':1200,'consumed_cpu_seconds':0.101408,'remaining_cpu_seconds':1199.898592,'prior_A_budget_or_proof_reused':False,'phases':[row]}
        lp=bundle/'cpu_budget_B.json';lp.write_text(json.dumps(old))
        prior={'ledger_ref':b.ref(lp),'manifest_ref':b.ref(mp),'actual_closed_CPU':0.101408}
        release=bundle/'release_v2.json';release.write_text(json.dumps({'local_validation_prior_cost':prior}))
        current=copy.deepcopy(self.manifest);current.update(cpu_cap_seconds=1200,root_release=b.ref(release),local_validation_prior_cost=prior)
        return current,old,lp,release

    def test_root_pinned_closed_local_cost_is_carried_without_refund(self):
        current,_,_,_=self.prior_carry_fixture();carry=b.local_validation_carry(current)
        self.assertEqual(carry['actual_closed_CPU'],0.101408)
        historical=json.loads(b.checked(carry['manifest_ref']).read_text())
        registry=self.root/'.runtime/parallel-b-allocation-registry';registry.mkdir(parents=True)
        old_claim=registry/(current['allocation_ref']['sha256']+'-local-validation.json')
        old_claim.write_text(json.dumps({'allocation_ref':historical['allocation_ref'],'role':'local-validation','manifest_sha256':carry['manifest_ref']['sha256'],'ledger':historical['ledger'],'run_id':historical['run_id']}));old_claim_bytes=old_claim.read_bytes()
        b.register_allocation_claim(current,'NEW_RECOVERY_FIXTURE_MANIFEST')
        self.assertEqual(old_claim.read_bytes(),old_claim_bytes)
        self.assertTrue((registry/(current['allocation_ref']['sha256']+'-local-recovery-'+current['run_id']+'.json')).exists())
        ledger={'schema':b.LEDGER_SCHEMA,'manifest_sha256':'FIXTURE','supervisor_sha256':b.sha(self.source),'run_id':current['run_id'],'cpu_cap_seconds':1200,'deadline_utc':b.DEADLINE,'phases':[],'initial_consumed_cpu_seconds':0.101408,'carried_local_validation':carry,'consumed_cpu_seconds':0.101408,'remaining_cpu_seconds':1199.898592}
        b.validate_ledger(ledger,current,'FIXTURE')
        ledger.update(consumed_cpu_seconds=0.,remaining_cpu_seconds=1200.)
        with self.assertRaises(ValueError):b.validate_ledger(ledger,current,'FIXTURE')

    def test_running_or_changed_prior_cost_cannot_be_imported(self):
        current,old,lp,release=self.prior_carry_fixture();old['phases'][0]['status']='RUNNING';lp.write_text(json.dumps(old))
        current['local_validation_prior_cost']['ledger_ref']=b.ref(lp);release.write_text(json.dumps({'local_validation_prior_cost':current['local_validation_prior_cost']}));current['root_release']=b.ref(release)
        with self.assertRaises(ValueError):b.local_validation_carry(current)

    def test_local_carry_forbids_A_ledger_or_remote_budget_inheritance(self):
        current,old,lp,release=self.prior_carry_fixture();current['local_smoke_test_without_remote_B_claim']=False
        with self.assertRaises(ValueError):b.local_validation_carry(current)
        current['local_smoke_test_without_remote_B_claim']=True;old['schema']='research4-pyscf-continuation-budget-v8';lp.write_text(json.dumps(old));current['local_validation_prior_cost']['ledger_ref']=b.ref(lp);release.write_text(json.dumps({'local_validation_prior_cost':current['local_validation_prior_cost']}));current['root_release']=b.ref(release)
        with self.assertRaises(ValueError):b.local_validation_carry(current)

    def test_dynamic_reference_changed_bytes_rejected(self):
        p=self.root/'reports/prior.json';p.write_text('{}');value=b.ref(p);p.write_text('{"changed":true}')
        with self.assertRaises(ValueError):b.materialize([str(self.python),'{result:prior}'],self.work/'out',{'prior':value})

    def test_real_stdlib_payload_CPU_and_live_proof_end_to_end(self):
        self.worker.write_text('''import argparse,importlib.util,json,os\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument("--output",type=Path);p.add_argument("--case");p.add_argument("--operation");p.add_argument("--execute",action="store_true");a=p.parse_args()\ns=Path(__file__).parent/"research4_parallel_b_supervisor_20261106.py";sp=importlib.util.spec_from_file_location("real_fixture_b_supervisor",s);b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)\nproof=b.verify_b_resource_supervision(expected_threads=1,expected_worker_path=Path(__file__).resolve(),expected_case=a.case,expected_operation=a.operation)\na.output.mkdir(parents=True)\nr={"schema":"fixture-only-no-quantum-v1","status":"FIXTURE_DONE_NOT_QUANTUM","source_worker":b.ref(__file__),"output_directory":str(a.output),"resource_supervision":proof,"new_quantum_evaluations":0}\n(a.output/"result.json").write_text(json.dumps(r))\n''')
        self.job['worker']=b.ref(self.worker);self.manifest['source_refs']=[b.ref(self.source),b.ref(self.worker)]
        self.work.mkdir(parents=True);mp=self.work/'manifest.json';mp.write_text(json.dumps(self.manifest));self.auth['manifest_sha256']=b.sha(mp);ap=self.work/'authorization.json';ap.write_text(json.dumps(self.auth))
        completed=subprocess.run([sys.executable,str(self.source),'--manifest',str(mp),'--manifest-sha256',b.sha(mp),'--authorization',str(ap),'--authorization-sha256',b.sha(ap),'--execute','--attempt','fixture-actual-wait4'],cwd=self.root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=20)
        self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
        ledger=json.loads((self.work/'cpu_budget_B.json').read_text());state=json.loads((self.work/'state.json').read_text())
        self.assertGreater(ledger['consumed_cpu_seconds'],0)
        self.assertEqual(ledger['phases'][0]['CPU_accounting'],'ACTUAL_WAIT4_SELF_AND_WAITED_CHILDREN_SINGLE_TGID_ENFORCED')
        self.assertEqual(state['jobs']['fixture']['status'],'DONE_SEALED')
        self.assertFalse(any(x['status']=='RUNNING' for x in ledger['phases']))
        command=[sys.executable,str(self.source),'--manifest',str(mp),'--manifest-sha256',b.sha(mp),'--authorization',str(ap),'--authorization-sha256',b.sha(ap),'--execute','--attempt','fixture-reuse-with-no-new-payload']
        reused=subprocess.run(command,cwd=self.root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=20)
        self.assertEqual(reused.returncode,0,reused.stderr)
        reused_ledger=json.loads((self.work/'cpu_budget_B.json').read_text());self.assertEqual(len(reused_ledger['phases']),1);self.assertEqual(reused_ledger['consumed_cpu_seconds'],ledger['consumed_cpu_seconds'])
        output=self.work/'science/fixture-attempt001/result.json';output.write_text(output.read_text()+'\n')
        command[-1]='fixture-tampered-checkpoint-must-block'
        changed=subprocess.run(command,cwd=self.root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=20)
        self.assertNotEqual(changed.returncode,0)
        self.assertEqual(len(json.loads((self.work/'cpu_budget_B.json').read_text())['phases']),1)

if __name__=='__main__':unittest.main()
