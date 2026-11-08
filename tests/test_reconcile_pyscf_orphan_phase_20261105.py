"""Synthetic stdlib recovery tests; no real ledger/proc mutation or quantum IO."""
from argparse import Namespace
from contextlib import nullcontext,redirect_stdout
from copy import deepcopy
from datetime import datetime,timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/reconcile_pyscf_orphan_phase_20261105.py'
GUARD=ROOT/'scripts/run_pyscf_continuation_budget_20261105_v6b.py'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def proc_stat(pid,pgrp,session,start,user,system,state='Z',cutime=0,cstime=0):
    fields=['0']*50
    for index,value in {0:state,1:1,2:pgrp,3:session,11:user,12:system,13:cutime,14:cstime,
                        19:start,20:0,21:0,49:9}.items():fields[index]=str(value)
    return (str(pid)+' (python) '+' '.join(fields)+'\n').encode()


def proc_status(uid=1000,state='Z',fd=0,threads=1,noise='1/100'):
    return (f'Name:\tpython\nState:\t{state} (synthetic)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n'
        f'FDSize:\t{fd}\nThreads:\t{threads}\nSigQ:\t{noise}\n').encode()


class ReconcileOrphanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w=load(SOURCE,'orphan_reconcile_stdlib_tests')
        cls.g=load(GUARD,'frozen_guard_functions_only_synthetic')

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'root';self.root.mkdir();(self.root/'.runtime').mkdir()
        (self.root/'.runtime/pyscf-v4-global.lock').write_text('')
        self.base=self.root/'reports/run';self.base.mkdir(parents=True)
        self.forensic=self.base/'forensics';self.forensic.mkdir()
        self.ledger=self.base/'cpu_budget.json';self.ledger.with_suffix('.lock').write_text('')
        self.observation=self.forensic/'observation.json';self.proof_path=self.base/'execution/phase.log.supervisor-proof.json'
        self.proof_path.parent.mkdir();self.phase='v6b-spinosin-C001-A-001'
        self.scientific=self.base/'spinosin-rks-partial-001';self.scientific.mkdir()
        self.proc=self.root/'synthetic-proc';self.proc.mkdir()
        (self.proc/'sys/kernel/random').mkdir(parents=True)
        self.boot='50ca0010-b743-43c7-bc0f-b6bcbc47c68d'
        (self.proc/'sys/kernel/random/boot_id').write_text(self.boot+'\n')
        self.proof={'schema':self.g.PROOF_SCHEMA,'supervisor_script_sha256':self.w.GUARD_SHA,
            'supervisor_script_path':str(self.w.GUARD),'ledger_path':str(self.ledger),
            'phase':self.phase,'supervisor_pid':22618,'supervisor_start_ticks':5047560,
            'child_pid':22619,'child_start_ticks':5047571,'target_uid':1000,
            'target_pgid':22619,'target_session_id':22619,'threads':2,
            'actual_cpu_clock_ticks_per_second':100,'started_at_utc':'2026-11-04T20:37:15+00:00'}
        self.write(self.proof_path,self.proof);self.proof_sha=self.w.sha(self.proof_path)
        self.original={'schema':self.g.SCHEMA,'runner_sha256':self.w.GUARD_SHA,
            'quantum_cpu_cap_seconds':216000,'run_deadline_at_utc':'2026-11-05T23:20:50.391742+00:00',
            'created_at_utc':'2026-11-04T07:20:50.391742+00:00','unchanged_header':'WAIT4_INITIAL_PROTOCOL_HISTORY',
            'consumed_cpu_seconds':129542.777675,'remaining_quantum_cpu_seconds':86457.222325,
            'phases':[{'phase':'all_old_known_costs','status':'EXITED','exit_code':0,'actual_cpu_seconds':129542.777675,
                'ended_at_utc':'2026-11-04T20:30:00+00:00'},
                {'phase':self.phase,'status':'RUNNING','started_at_utc':'2026-11-04T20:37:15+00:00',
                 'command':['.runtime/bin/python',str(self.root/'scripts/pyscf_staged_hessian_worker_20261105_v6b.py'),
                    '--execute','--stage','rks-partial','--threads','2','--output',str(self.scientific)],
                 'pid':22619,'threads':2,'supervisor_proof_path':str(self.proof_path),
                 'supervisor_proof_sha256':self.proof_sha}]}
        self.write(self.ledger,self.original);self.before=self.ledger.read_bytes();self.original_sha=self.w.sha(self.ledger)
        (self.forensic/'cpu_budget.json').write_bytes(self.before)
        (self.forensic/'launch-proof.json').write_bytes(self.proof_path.read_bytes())
        raw_child=proc_stat(22619,22619,22619,5047571,2445491,40845)
        raw_parent=proc_stat(22618,22615,22615,5047560,17500,398)
        self.raw={}
        for pid,raw in [(22618,raw_parent),(22619,raw_child)]:
            (self.proc/str(pid)).mkdir()
            (self.proc/str(pid)/'stat').write_bytes(raw);(self.proc/str(pid)/'status').write_bytes(proc_status())
            for kind,data in [('stat',raw),('status',proc_status())]:
                p=self.forensic/f'proc_{pid}_{kind}';p.write_bytes(data)
                self.raw[f'/proc/{pid}/{kind}']={'file':str(p),'sha256':self.w.sha(p),'bytes':len(data)}
        self.obs={'schema':'research4-unclosed-phase-terminal-observation-v1',
            'phase':self.phase,'observed_at_utc':'2026-11-05T00:09:23+00:00',
            'clock_ticks_per_second':100,'terminal_process_cpu_seconds_tick_precision':24863.36,
            'final_utime_ticks':2445491,'final_stime_ticks':40845,'final_cutime_ticks':0,'final_cstime_ticks':0,
            'wait4_obtained':False,'exact_death_time_known':False,'result_json_exists':False,
            'raw_observations':self.raw,'original_ledger_bytes_preserved':{
                'cpu_budget.json':{'preserved_copy':str(self.forensic/'cpu_budget.json'),
                    'sha256':self.original_sha,'bytes':len(self.before)},
                'execution/'+self.phase+'.log.supervisor-proof.json':{
                    'preserved_copy':str(self.forensic/'launch-proof.json'),'sha256':self.proof_sha,
                    'bytes':self.proof_path.stat().st_size}}}
        self.write(self.observation,self.obs)
        self.run_input={'schema':'research4-staged-full-rks-hessian-result-v6b','stage':'rks-partial','phase':'rks-partial',
            'started_at_utc':'2026-11-04T20:37:16+00:00','boot_id':self.boot,
            'source_worker':{'sha256':self.w.SOURCE_SHAS['scripts/pyscf_staged_hessian_worker_20261105_v6b.py']},
            'resource_supervision':{'proof':self.proof,'proof_sha256':self.proof_sha}}
        self.write(self.scientific/'input.json',self.run_input)
        self.auth={'max_phase_cpu_seconds':122400,'max_phase_wall_seconds':43200,'max_threads':4,'max_memory_gib':24}
        def validate(ledger,allow_running=False):
            running=[r for r in ledger['phases'] if r['status']=='RUNNING']
            if running and not allow_running:raise ValueError('running')
            total=self.g.closed_costs([r for r in ledger['phases'] if r['status']!='RUNNING'])
            self.assertAlmostEqual(total,ledger['consumed_cpu_seconds'])
            self.assertAlmostEqual(ledger['remaining_quantum_cpu_seconds'],216000-total)
            return self.auth
        self.fake=SimpleNamespace(PROOF_SCHEMA=self.g.PROOF_SCHEMA,validate_ledger=Mock(side_effect=validate),
            closed_costs=self.g.closed_costs,healthy_closed_row=self.g.healthy_closed_row,
            phase_limits=Mock(return_value={'cpu_boundary_reserve_seconds':6,'threads':2}),
            lineage_locks=lambda auth:nullcontext())
        self.prepare_original=self.w.prepare;self.no_spawn_original=self.w.no_spawn_source_audit
        for key,value in [('ROOT',self.root),('guard_module',lambda:self.fake),
                          ('no_spawn_source_audit',lambda:{'synthetic_scope_only':True})]:
            p=patch.object(self.w,key,value);p.start();self.addCleanup(p.stop)
        self.addendum=self.base/'root-addendum.json'

    @staticmethod
    def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

    def args(self,execute=False,name='out'):
        return Namespace(execute=execute,ledger=self.ledger,ledger_sha256=self.original_sha,phase=self.phase,
            observation=self.observation,observation_sha256=self.w.sha(self.observation),
            output=self.base/name,holdback_seconds=1.,addendum=self.addendum,
            addendum_sha256=self.w.sha(self.addendum) if self.addendum.exists() else None)

    def prepare(self,args=None):return self.prepare_original(args or self.args(),self.fake,proc_root=self.proc)

    def approve(self):
        data=self.prepare()
        value={'schema':self.w.ADDENDUM_SCHEMA,'root_approved_for_reconciliation':True,
            'reconciler_source_sha256':self.w.sha(SOURCE),'original_ledger_path':str(self.ledger),
            'original_ledger_sha256':self.original_sha,'phase':self.phase,'guard_source_sha256':self.w.GUARD_SHA,
            'launch_proof_sha256':self.proof_sha,'observation_sha256':self.w.sha(self.observation),
            'raw_process_snapshots':data['records'],'scientific_input_sha256':self.w.sha(self.scientific/'input.json'),
            'accounting_source':'PROC_STAT_FINAL_TICKS','wait4_obtained':False,
            'required_non_actual_holdback_seconds':1.,'holdback_covered_by_unchanged_guard_boundary':True,
            'selected_future_threads':2,'no_resource_growth':True,'no_deadline_extension':True,
            'no_healthy_cache_release':True,'created_at_utc':datetime.now(timezone.utc).isoformat()}
        self.write(self.addendum,value);return value

    def run_reconcile(self,args):
        with patch.object(self.w,'prepare',side_effect=lambda a,g:self.prepare_original(a,g,proc_root=self.proc)),redirect_stdout(io.StringIO()):
            return self.w.reconcile(args)

    def test_terminal_measurement_charge_and_header_preservation(self):
        data=self.prepare();new=data['proposal'];row=new['phases'][-1]
        self.assertEqual(row['actual_cpu_seconds'],24863.36)
        self.assertEqual(row['supervisor_process_CPU_seconds_tick_precision_NOT_CHILD_LEDGER'],178.98)
        self.assertAlmostEqual(new['consumed_cpu_seconds'],154406.137675)
        self.assertAlmostEqual(new['remaining_quantum_cpu_seconds'],61593.862325)
        for key in self.original:
            if key not in ('phases','consumed_cpu_seconds','remaining_quantum_cpu_seconds'):
                self.assertEqual(new[key],self.original[key])
        self.assertEqual(set(new),set(self.original))
        self.assertEqual(row['status'],'PARTIAL_OR_FAILED');self.assertFalse(self.g.healthy_closed_row(row))
        self.assertFalse(row['wait4_obtained']);self.assertIsNone(row['actual_wait4_cpu_seconds'])
        self.assertEqual(row['actual_cpu_accounting_source'],'PROC_STAT_FINAL_TICKS')
        self.assertEqual(row['exit_code'],-9)
        self.assertIsNone(row['cpu_terminated']);self.assertIsNone(row['peak_rss_kib'])
        self.assertFalse((self.scientific/'result.json').exists())

    def test_dryrun_keeps_canonical_ledger_bytes_and_unapproved_row(self):
        report=self.run_reconcile(self.args())
        self.assertEqual(report['status'],'DRY_RUN_VALIDATED_NO_LEDGER_CHANGE')
        self.assertEqual(self.ledger.read_bytes(),self.before)
        proposal=json.loads((self.base/'out/proposed_FAILED_ledger.json').read_text())
        recovery=proposal['phases'][-1]['orphan_reconciliation']
        self.assertIsNone(recovery['root_addendum']);self.assertEqual(recovery['approval_status'],'DRY_RUN_NOT_GRANTED')
        self.assertEqual(recovery['reconciliation_receipt_path'],str(self.base/'out/result.json'))

    def test_execute_requires_actual_source_bound_addendum(self):
        with self.assertRaises(ValueError):self.run_reconcile(self.args(True))
        self.assertEqual(self.ledger.read_bytes(),self.before)

    def test_execute_atomic_update_is_approved_traceable_and_never_healthy(self):
        self.approve();args=self.args(True);report=self.run_reconcile(args)
        self.assertEqual(report['status'],'DONE_FAILED_PHASE_RECONCILED_NOT_A_HEALTHY_PRODUCER')
        new=json.loads(self.ledger.read_text());r=new['phases'][-1]['orphan_reconciliation']
        self.assertEqual(r['root_addendum']['sha256'],self.w.sha(self.addendum))
        self.assertEqual(r['approval_status'],'EXPLICIT_ROOT_APPROVED')
        self.assertEqual((self.base/'out/original_RUNNING_ledger.json').read_bytes(),self.before)
        self.assertEqual((self.forensic/'cpu_budget.json').read_bytes(),self.before)
        self.assertFalse(self.g.healthy_closed_row(new['phases'][-1]))
        self.assertEqual(self.w.sha(self.ledger),report['actual_new_ledger_sha256'])
        self.assertFalse((self.scientific/'result.json').exists())

    def test_repeat_execute_refuses_double_charge(self):
        self.approve();self.run_reconcile(self.args(True));after=self.ledger.read_bytes()
        with self.assertRaises(ValueError):self.run_reconcile(self.args(True,'again'))
        self.assertEqual(self.ledger.read_bytes(),after)

    def test_missing_or_corrupt_preserved_original_blocks(self):
        (self.forensic/'cpu_budget.json').write_text('changed')
        with self.assertRaises(ValueError):self.prepare()
        self.assertEqual(self.ledger.read_bytes(),self.before)

    def test_changed_source_ledger_CAS_blocks(self):
        self.ledger.write_text(self.ledger.read_text()+' ')
        with self.assertRaises(ValueError):self.prepare()

    def test_changed_pid_start_or_live_state_blocks(self):
        p=self.proc/'22619/stat';p.write_bytes(proc_stat(22619,22619,22619,5047572,2445491,40845))
        with self.assertRaisesRegex(ValueError,'identity differs'):self.prepare()
        p.write_bytes(proc_stat(22619,22619,22619,5047571,2445491,40845,state='S'))
        with self.assertRaises(ValueError):self.prepare()

    def test_nonzero_waited_child_cpu_or_FD_blocks(self):
        p=self.proc/'22619/stat';p.write_bytes(proc_stat(22619,22619,22619,5047571,2445491,40845,cutime=1))
        with self.assertRaisesRegex(ValueError,'waited-child CPU'):self.prepare()
        p.write_bytes(proc_stat(22619,22619,22619,5047571,2445491,40845))
        (self.proc/'22619/status').write_bytes(proc_status(fd=64))
        with self.assertRaisesRegex(ValueError,'no-files'):self.prepare()

    def test_UID_or_boot_mismatch_blocks(self):
        (self.proc/'22619/status').write_bytes(proc_status(uid=1001))
        with self.assertRaises(ValueError):self.prepare()
        (self.proc/'22619/status').write_bytes(proc_status())
        (self.proc/'sys/kernel/random/boot_id').write_text('00000000-0000-0000-0000-000000000000')
        with self.assertRaisesRegex(ValueError,'boot identity'):self.prepare()

    def test_live_other_member_in_child_or_parent_session_blocks(self):
        p=self.proc/'555';p.mkdir();(p/'stat').write_bytes(proc_stat(555,22619,22619,12,0,0,state='S'))
        with self.assertRaisesRegex(ValueError,'Live process remains'):self.prepare()

    def test_declared_observation_counter_mismatch_blocks(self):
        self.obs['final_stime_ticks']+=1;self.write(self.observation,self.obs)
        with self.assertRaisesRegex(ValueError,'Declared observation'):self.prepare()

    def test_observation_cannot_predate_launch(self):
        self.obs['observed_at_utc']='2026-11-04T19:00:00+00:00';self.write(self.observation,self.obs)
        with self.assertRaisesRegex(ValueError,'predate'):self.prepare()

    def test_irrelevant_status_text_changes_do_not_fake_identity_change(self):
        originals=self.w.process_snapshot;counter=[0]
        def noisy(pid,root):
            raw,status=originals(pid,root);counter[0]+=1
            return raw,status.replace(b'1/100',str(counter[0]).encode()+b'/100')
        with patch.object(self.w,'process_snapshot',side_effect=noisy):self.prepare()

    def test_exact_death_time_and_wait4_never_invented(self):
        row=self.prepare()['proposal']['phases'][-1]
        self.assertEqual(row['ended_at_utc'],self.obs['observed_at_utc'])
        self.assertEqual(row['ended_at_utc_semantics'],'TERMINAL_OBSERVED_UPPER_BOUND_NOT_EXACT_DEATH_TIME')
        self.assertFalse(row['exact_death_time_known']);self.assertIsNone(row['elapsed_wall_seconds'])

    def test_reserve_is_not_extra_actual_cost_or_phase(self):
        data=self.prepare();proposal=data['proposal']
        self.assertEqual(len(proposal['phases']),len(self.original['phases']))
        self.assertEqual(proposal['phases'][-1]['actual_cpu_seconds'],24863.36)
        recovery=proposal['phases'][-1]['orphan_reconciliation']
        self.assertEqual(recovery['conservative_non_actual_holdback_seconds'],1.)
        self.assertFalse(recovery['holdback_is_actual_cost'])
        self.assertEqual(data['future_phase_limits_metadata_only_NOT_A_LAUNCH']['cpu_boundary_reserve_seconds'],6)

    def test_expired_or_insufficient_future_phase_does_not_block_accounting(self):
        self.fake.phase_limits.side_effect=ValueError('Insufficient original CPU/wall budget for a safe bounded phase')
        data=self.prepare()
        self.assertEqual(data['future_phase_limits_metadata_only_NOT_A_LAUNCH']['status'],
            'BLOCKED_INSUFFICIENT_REMAINING_CPU_OR_WALL_NO_NEW_PHASE')

    def test_authorization_binding_corruption_blocks_execute(self):
        value=self.approve();value['wait4_obtained']=True;self.write(self.addendum,value)
        with self.assertRaisesRegex(ValueError,'binding/policy'):self.run_reconcile(self.args(True))
        self.assertEqual(self.ledger.read_bytes(),self.before)

    def test_atomic_CAS_refuses_change_and_preserves_owner_mode(self):
        expected=self.w.sha(self.ledger);mode=self.ledger.stat().st_mode
        self.w.atomic_cas_ledger(self.ledger,expected,b'{"fixture":true}\n')
        self.assertEqual(self.ledger.stat().st_mode,mode)
        with self.assertRaises(ValueError):self.w.atomic_cas_ledger(self.ledger,expected,b'bad')

    def test_source_no_spawn_pins_and_top_imports_are_stdlib(self):
        with patch.object(self.w,'ROOT',ROOT):audit=self.no_spawn_original()
        self.assertEqual(len(audit['files']),5)
        text=SOURCE.read_text()
        self.assertNotIn('os.wait4(',text);self.assertNotIn('os.kill(',text)
        self.assertNotIn('np.load(',text);self.assertNotIn('subprocess.Popen(',text)


if __name__=='__main__':unittest.main()
