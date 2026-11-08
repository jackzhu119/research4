"""Pure stdlib fixtures: no subprocess, signals, PySCF, or real ledger mutation."""
import ast
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/run_pyscf_continuation_budget_20261105_v8.py'
OLD=ROOT/'scripts/run_pyscf_continuation_budget_20261105_v6b.py'
spec=importlib.util.spec_from_file_location('v8_under_test',SOURCE)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');return g.sha(path)

def fixture(root):
 olddir=root/'old';oldpath=olddir/'cpu_budget.json';newpath=root/'new'/'cpu_budget.json';newpath.parent.mkdir()
 resource={'guard_semantics':g.GUARD_SEMANTICS,'rss_stop_bytes':g.RSS_STOP_BYTES,'cgroup_kill_bytes':g.CGROUP_KILL_BYTES,
  'monitor_interval_seconds':g.MONITOR_INTERVAL_SECONDS,'memory_kill_grace_seconds':g.MEMORY_KILL_GRACE_SECONDS,
  'expected_cgroup_memory_max_bytes':g.CGROUP_MEMORY_MAX_BYTES,'actual_cpu_cleanup_reserve_seconds':120,
  'cpu_guard_kill_grace_seconds':1.0,'actual_cpu_guard_source':g.ACTUAL_CPU_GUARD_SOURCE}
 phases=[{'phase':'old_ok','status':'EXITED','exit_code':0,'ended_at_utc':'2026-11-05T11:00:00+00:00','actual_cpu_seconds':2.75},
  {'phase':'old_failed','status':'PARTIAL_OR_FAILED','exit_code':2,'ended_at_utc':'2026-11-05T11:14:00+00:00','actual_cpu_seconds':7.125,'cpu_terminated':True}]
 old={'schema':g.OLD_SCHEMA,'runner_sha256':g.OLD_GUARD_SHA,'phases':phases,'consumed_cpu_seconds':9.875,
  'remaining_quantum_cpu_seconds':216000-9.875,'quantum_cpu_cap_seconds':216000,'created_at_utc':'2026-11-04T07:20:50+00:00',
  'run_deadline_at_utc':'2026-11-05T23:20:50.391742+00:00','authorization_path':'old_auth.json','authorization_sha256':'a'*64,
  'source_checkpoint_sha256':'b'*64,'source_base_commit':'c'*40,
  'continuation_source':{'source_v5_path':str(root/'v5'/'cpu.json')},'migration':{'source_v4_path':str(root/'v4'/'cpu.json')},
  'prior_closed_ledger':{'path':str(root/'v3'/'cpu.json')}}
 oldsha=write(oldpath,old)
 for key in ('v5','v4','v3'):write(root/key/'cpu.json',{})
 observed={'schema':'research4-root-observed-new-96h-authorization-v1','observation_not_invented_exact_user_message_timestamp':True,
  'user_instruction_verbatim':'New96huserinstruction','authorization_observed_at_utc':'2026-11-05T12:33:48+00:00',
  'conservative_deadline_at_utc':'2026-11-09T12:30:00+00:00','old_ledger':{'path':str(oldpath),'sha256':oldsha},
  'root_additional_quantum_CPU_seconds':180000,'root_total_cap_with_legacy_actual_cost':180009.875}
 rec=root/'new'/'user.json';recsha=write(rec,observed)
 auth={'schema':g.AUTH_SCHEMA,'root_approved_for_execution':True,'no_budget_reset':True,'no_automatic_budget_growth':True,
  'new_user_96h_authorization':True,'v8_supervisor_source_sha256':g.sha(SOURCE),'continuation_ledger_path':str(newpath),
  'source_v6b_ledger_path':str(oldpath),'source_v6b_ledger_sha256':oldsha,'source_v6b_runner_sha256':g.OLD_GUARD_SHA,
  'research_authorization_record':{'path':str(rec),'sha256':recsha},'authorization_observed_at_utc':observed['authorization_observed_at_utc'],
  'created_at_utc':'2026-11-05T12:40:00+00:00','research_authorization_deadline':observed['conservative_deadline_at_utc'],
  'additional_cpu_seconds':180000,'max_quantum_cpu_seconds':180009.875,'source_v6b_consumed_cpu_seconds':9.875,
  'prior_closed_ledger_path':str(oldpath),'prior_closed_ledger_sha256':oldsha,'source_base_commit':'d'*40,
  'max_threads':2,'max_memory_gib':24,'max_phase_cpu_seconds':122400,'max_phase_wall_seconds':43200,**resource}
 ap=root/'new'/'auth.json';ash=write(ap,auth)
 def check(value,allow_running=False):
  if value.get('schema')!=g.OLD_SCHEMA:raise ValueError('not old')
  if any(x.get('status')=='RUNNING' for x in value.get('phases',[])):raise RuntimeError('old running')
  return resource
 prior=SimpleNamespace(validate_ledger=check)
 args=SimpleNamespace(ledger=newpath,authorization=ap,authorization_sha256=ash,source_v6b_ledger=oldpath,
  source_v6b_sha256=oldsha,source_base_commit='d'*40)
 return SimpleNamespace(root=root,old=old,oldpath=oldpath,oldsha=oldsha,newpath=newpath,auth=auth,ap=ap,ash=ash,args=args,prior=prior,resource=resource,rec=rec,observed=observed)

class V8GuardTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.fx=fixture(Path(self.tmp.name));self.frozen_now=datetime(2026,11,5,13,tzinfo=timezone.utc)
  self.old=patch.object(g,'old_guard',return_value=self.fx.prior);self.old.start();self.addCleanup(self.old.stop)
  self.clock=patch.object(g,'utc',return_value=self.frozen_now);self.clock.start();self.addCleanup(self.clock.stop)
  self.reg=patch.object(g,'CONTINUATION_REGISTRY',self.fx.root/'registry');self.reg.start();self.addCleanup(self.reg.stop)
 def authorize(self,change=None):
  if change:self.fx.auth.update(change)
  self.fx.ash=write(self.fx.ap,self.fx.auth);self.fx.args.authorization_sha256=self.fx.ash
  return g.validate_authorization(self.fx.ap,self.fx.ash)
 def register(self):
  with patch.object(g,'lineage_locks',return_value=nullcontext()):return g.register_continuation(self.fx.args)
 def test_native_OS_CPU_RSS_PID_monitor_AST_is_unchanged(self):
  old={n.name:n for n in ast.parse(OLD.read_text()).body if isinstance(n,ast.FunctionDef)}
  new={n.name:n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef)}
  for name in ['parse_proc_stat','read_process_identity','identity_matches','owned_identity_matches','terminal_reap_observation','bounded_owned_reap','signal_bound_group','actual_cpu_observation','process_group_snapshot','read_cgroup_memory','resource_runtime_snapshot','phase_limits','_run_phase_with_prior_lock','healthy_closed_row']:
   self.assertEqual(ast.dump(old[name],include_attributes=False),ast.dump(new[name],include_attributes=False),name)
 def test_new_authorization_accepts_fractional_cap_without_rounding_or_refund(self):
  a=self.authorize();self.assertEqual(a['max_quantum_cpu_seconds'],180009.875)
 def test_disabled_root_grant_rejected(self):
  with self.assertRaises(ValueError):self.authorize({'root_approved_for_execution':False})
 def test_automatic_budget_growth_rejected(self):
  with self.assertRaises(ValueError):self.authorize({'no_automatic_budget_growth':False})
 def test_not_new_user_authorization_rejected(self):
  with self.assertRaises(ValueError):self.authorize({'new_user_96h_authorization':False})
 def test_new_increment_above_explicit_root_decision_rejected(self):
  with self.assertRaises(ValueError):self.authorize({'additional_cpu_seconds':180001})
 def test_old_unused_budget_cannot_be_added_again(self):
  with self.assertRaises(ValueError):self.authorize({'max_quantum_cpu_seconds':216000+180000})
 def test_cap_rounding_is_not_accepted(self):
  with self.assertRaises(ValueError):self.authorize({'max_quantum_cpu_seconds':180009})
 def test_source_actual_cost_changed_rejected(self):
  with self.assertRaises(ValueError):self.authorize({'source_v6b_consumed_cpu_seconds':9})
 def test_changed_old_ledger_bytes_rejected(self):
  self.fx.oldpath.write_text(self.fx.oldpath.read_text()+' ')
  with self.assertRaises(ValueError):self.authorize()
 def test_missing_authorization_record_rejected(self):
  self.fx.rec.unlink()
  with self.assertRaises(ValueError):self.authorize()
 def test_registration_time_cannot_reset_user96h_clock(self):
  with self.assertRaises(ValueError):self.authorize({'authorization_observed_at_utc':'2026-11-05T12:40:00+00:00'})
 def test_future_root_decision_rejected(self):
  with self.assertRaises(ValueError):self.authorize({'created_at_utc':'2026-11-06T12:40:00+00:00'})
 def test_resource_envelope_cannot_silently_expand(self):
  with self.assertRaises(ValueError):self.authorize({'rss_stop_bytes':11*1024**3})
 def test_more_than_two_threads_is_rejected(self):
  with self.assertRaises(ValueError):self.authorize({'max_threads':4})
 def test_alias_or_old_tree_new_ledger_rejected(self):
  with self.assertRaises(ValueError):self.authorize({'continuation_ledger_path':str(self.fx.oldpath.parent/'new.json')})
 def test_register_preserves_failed_and_healthy_prefix_original_creation_and_old_bytes(self):
  value=self.register();self.assertEqual(value['phases'],self.fx.old['phases']);self.assertEqual(value['consumed_cpu_seconds'],9.875);self.assertEqual(value['remaining_quantum_cpu_seconds'],180000);self.assertEqual(value['quantum_cpu_cap_seconds'],180009.875);self.assertEqual(g.sha(self.fx.oldpath),self.fx.oldsha);self.assertEqual(value['created_at_utc'],self.fx.old['created_at_utc']);g.validate_ledger(value)
 def test_existing_ledger_cannot_be_overwritten(self):
  self.register()
  with self.assertRaises(FileExistsError):self.register()
 def test_expired_authorization_cannot_register_but_closed_ledger_remains_auditable(self):
  value=self.register()
  with patch.object(g,'utc',return_value=datetime(2026,11,10,tzinfo=timezone.utc)):
   g.validate_ledger(value)
   self.fx.args.ledger=self.fx.root/'expired.json';self.fx.auth['continuation_ledger_path']=str(self.fx.args.ledger);self.authorize()
   with self.assertRaises(ValueError):self.register()
 def test_symlink_registry_is_rejected_before_publication(self):
  target=self.fx.root/'outside';target.mkdir();link=self.fx.root/'registry-link';link.symlink_to(target,target_is_directory=True)
  with patch.object(g,'CONTINUATION_REGISTRY',link):
   with self.assertRaises(ValueError):self.register()
  self.assertFalse(self.fx.newpath.exists());self.assertEqual(list(target.iterdir()),[])
 def test_registry_forbids_second_allocation_with_another_target(self):
  self.register();self.fx.auth['continuation_ledger_path']=str(self.fx.root/'another'/'ledger.json');self.fx.args.ledger=Path(self.fx.auth['continuation_ledger_path']);self.fx.args.ledger.parent.mkdir();self.authorize()
  with self.assertRaises(FileExistsError):self.register()
 def test_dropped_failed_prefix_rejected(self):
  value=self.register();value['phases']=value['phases'][:1]
  with self.assertRaises(ValueError):g.validate_ledger(value)
 def test_new_cost_cannot_be_omitted(self):
  value=self.register();value['phases'].append({'phase':'new','status':'PARTIAL_OR_FAILED','ended_at_utc':'2026-11-05T13:30:00+00:00','actual_cpu_seconds':1.5})
  with self.assertRaises(ValueError):g.validate_ledger(value)
 def test_all_actual_new_failed_cost_is_counted(self):
  value=self.register();value['phases'].append({'phase':'new','status':'PARTIAL_OR_FAILED','ended_at_utc':'2026-11-05T13:30:00+00:00','actual_cpu_seconds':1.5});value['consumed_cpu_seconds']+=1.5;value['remaining_quantum_cpu_seconds']-=1.5;g.validate_ledger(value)
 def test_unresolved_new_phase_blocks_another_launch(self):
  value=self.register();value['phases'].append({'phase':'new','status':'RUNNING'})
  with self.assertRaises(RuntimeError):g.validate_ledger(value)
  g.validate_ledger(value,allow_running=True)
 def test_allocation_metadata_tamper_rejected(self):
  value=self.register();value['allocation']['new_fixed_CPU_seconds']=0
  with self.assertRaises(ValueError):g.validate_ledger(value)
 def test_full96h_deadline_cannot_be_extended(self):
  value=self.register();value['run_deadline_at_utc']='2026-11-10T12:30:00+00:00'
  with self.assertRaises(ValueError):g.validate_ledger(value)
 def test_legacy_live_launch_rejected_even_when_old_ledger_is_valid(self):
  with self.assertRaises(ValueError):g.run_phase(SimpleNamespace(),self.fx.old)
 def test_legacy_ledger_validation_is_explicit_dispatch(self):
  self.assertEqual(g.validate_ledger(self.fx.old),self.fx.resource)
 def test_closed_legacy_proof_is_delegated_without_relabel(self):
  proof={'schema':g.OLD_PROOF_SCHEMA,'supervisor_script_path':str(g.OLD_GUARD),'supervisor_script_sha256':g.OLD_GUARD_SHA};p=self.fx.root/'proof.json';sh=write(p,proof);e={'resource_supervision':{'proof_path':str(p),'proof_sha256':sh}};callee=Mock(return_value={'legacy':'EXACT_OLD_RETURN'})
  with patch.object(g,'old_guard',return_value=SimpleNamespace(verify_closed_phase=callee)):
   self.assertEqual(g.verify_closed_phase(e,expected_worker_path=OLD,expected_worker_sha256='f'*64,expected_result_schema='old',allowed_result_statuses=('STAGE_COMPLETE',)),{'legacy':'EXACT_OLD_RETURN'})
  callee.assert_called_once();self.assertEqual(json.loads(p.read_text())['schema'],g.OLD_PROOF_SCHEMA)
 def test_legacy_proof_wrong_supervisor_path_rejected(self):
  p=self.fx.root/'proof.json';sh=write(p,{'schema':g.OLD_PROOF_SCHEMA,'supervisor_script_path':str(SOURCE),'supervisor_script_sha256':g.OLD_GUARD_SHA})
  with self.assertRaises(ValueError):g.verify_closed_phase({'resource_supervision':{'proof_path':str(p),'proof_sha256':sh}},expected_worker_path=OLD,expected_worker_sha256='f'*64,expected_result_schema='old',allowed_result_statuses=('STAGE_COMPLETE',))
 def test_unknown_closed_proof_rejected(self):
  p=self.fx.root/'proof.json';sh=write(p,{'schema':'UNKNOWN'})
  with self.assertRaises(ValueError):g.verify_closed_phase({'resource_supervision':{'proof_path':str(p),'proof_sha256':sh}},expected_worker_path=OLD,expected_worker_sha256='f'*64,expected_result_schema='old',allowed_result_statuses=('STAGE_COMPLETE',))
 def test_legacy_proof_never_accepted_for_live_supervision(self):
  p=self.fx.root/'proof.json';write(p,{'schema':g.OLD_PROOF_SCHEMA,'guard_semantics':g.GUARD_SEMANTICS})
  with self.assertRaises(ValueError):g.verify_resource_supervision(p)
 def test_fractional_batch_cap_is_safely_floored_for_phase_grant(self):
  ledger={'quantum_cpu_cap_seconds':390160.8308359999,'consumed_cpu_seconds':210160.8308359999,'run_deadline_at_utc':'2026-11-09T12:30:00+00:00'};auth={**self.fx.resource,'max_threads':2,'max_memory_gib':24,'max_phase_cpu_seconds':122400,'max_phase_wall_seconds':43200};args=SimpleNamespace(threads=2,memory_gib=24,cpu_seconds=21600,wall_seconds=14400);v=g.phase_limits(ledger,auth,args,now=self.frozen_now);self.assertEqual(v['granted_cpu_seconds'],21600);self.assertEqual(v['actual_cpu_batch_cap_seconds'],ledger['quantum_cpu_cap_seconds']);self.assertEqual(v['actual_cpu_stop_threshold_seconds'],21480)
 def test_expired_deadline_refuses_new_phase(self):
  ledger={'quantum_cpu_cap_seconds':390160.8308359999,'consumed_cpu_seconds':210160.8308359999,'run_deadline_at_utc':'2026-11-05T12:30:00+00:00'};auth={**self.fx.resource,'max_threads':2,'max_memory_gib':24,'max_phase_cpu_seconds':122400,'max_phase_wall_seconds':43200};args=SimpleNamespace(threads=2,memory_gib=24,cpu_seconds=21600,wall_seconds=14400)
  with self.assertRaises(ValueError):g.phase_limits(ledger,auth,args,now=self.frozen_now)
 def test_exhausted_cumulative_cap_refuses_even_with_phase_grant(self):
  ledger={'quantum_cpu_cap_seconds':390160.8308359999,'consumed_cpu_seconds':390100.8308359999,'run_deadline_at_utc':'2026-11-09T12:30:00+00:00'};auth={**self.fx.resource,'max_threads':2,'max_memory_gib':24,'max_phase_cpu_seconds':122400,'max_phase_wall_seconds':43200};args=SimpleNamespace(threads=2,memory_gib=24,cpu_seconds=21600,wall_seconds=14400)
  with self.assertRaises(ValueError):g.phase_limits(ledger,auth,args,now=self.frozen_now)

if __name__=='__main__':unittest.main()
