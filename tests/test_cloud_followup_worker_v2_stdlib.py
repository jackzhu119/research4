"""Adversarial source-seal and append-policy tests; no numerical/Q imports."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/research4_cloud_followup_worker_20261106_v2.py'
SPEC=importlib.util.spec_from_file_location('cloud_followup_v2_stdlib',SCRIPT)
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class QualificationSealSafety(unittest.TestCase):
    def test_import_and_qualification_are_stdlib_only(self):
        self.assertFalse(any(x.split('.')[0] in {'numpy','pyscf','scipy','torch'} for x in sys.modules))
        self.assertEqual(M.sha(M.ANALYZER),M.ANALYZER_SHA)
        self.assertEqual(M.sha(ROOT/'scripts/research4_cloud_followup_worker_20261106.py'),
                         'b241714d38f989c1bf28321e2ff9d8957c303de1a546a5f4fa01e05c003131cc')
        functions={x.name:x for x in ast.parse(SCRIPT.read_text()).body if isinstance(x,ast.FunctionDef)}
        source=ast.get_source_segment(SCRIPT.read_text(),functions['source_gate'])
        self.assertNotIn('analyzer.qualify(',source)
        self.assertIn('analyzer.qualify(',ast.get_source_segment(SCRIPT.read_text(),functions['qualify_source']))

    def test_original_numerical_functions_are_ast_identical(self):
        def functions(path):return {x.name:x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef)}
        old=functions(ROOT/'scripts/research4_cloud_followup_worker_20261106.py');new=functions(SCRIPT)
        for name in ('direction_and_hessian','analyze','validate_displacement'):
            self.assertEqual(ast.dump(old[name],include_attributes=False),ast.dump(new[name],include_attributes=False))
        for node in ast.walk(new['execute_closed']):
            if isinstance(node,ast.Dict):
                for i,key in enumerate(list(node.keys)):
                    if isinstance(key,ast.Constant) and key.value=='v2_source_qualification':
                        node.keys.pop(i);node.values.pop(i);break
        self.assertEqual(ast.dump(old['execute_closed'],include_attributes=False),ast.dump(new['execute_closed'],include_attributes=False))

    def test_sealed_prefix_rejects_running_duplicate_or_changed_failed_cost(self):
        old=[{'phase':'old-success','status':'EXITED','actual_cpu_seconds':42.},
             {'phase':'old-real-failure','status':'PARTIAL_OR_FAILED','actual_cpu_seconds':.355263}]
        sealed=M.row_prefix({'phases':old})
        self.assertEqual(sealed[1]['canonical_sha256'],M.canonical_sha(old[1]))
        for rows in ([{**old[0],'status':'RUNNING'}],[old[0],old[0]]):
            with self.assertRaises(ValueError):M.row_prefix({'phases':rows})
        self.assertNotEqual(M.canonical_sha({**old[1],'actual_cpu_seconds':0.}),sealed[1]['canonical_sha256'])

    def make_fixture(self,root):
        reports=root/'reports';reports.mkdir()
        worker=root/'worker.py';worker.write_text('# synthetic worker identity only\n')
        ledger=reports/'ledger.json';ledger.write_text('{}')
        phase='cloudhosta20261106v2-water-plus-attempt001'
        output=reports/'science'/'water-plus-attempt001'
        log=reports/'water-plus.log';log.write_text('synthetic guard log\n')
        proof={'phase':phase,'ledger_path':str(ledger),'supervisor_script_sha256':M.GUARD_SHA}
        proof_path=reports/'proof.json';proof_path.write_text(json.dumps(proof))
        policy={'phase':phase,'worker':M.ref(worker),'output_directory':str(output),'log':str(log)}
        row={'phase':phase,'status':'RUNNING','command':['python',str(worker),'--execute','--output',str(output)],
             'log':str(log),'supervisor_proof_path':str(proof_path),'supervisor_proof_sha256':M.sha(proof_path)}
        prefix=[{'phase':'legacy-real-failure','status':'PARTIAL_OR_FAILED','actual_cpu_seconds':.355263}]
        actual={'phases':prefix+[row]};schedule={'schema':M.SCHEDULE_SCHEMA,'ledger_path':str(ledger),'phases':[policy]}
        guard=SimpleNamespace(root_path=lambda x:Path(x).resolve(),read=lambda x:json.loads(Path(x).read_text()))
        return actual,M.row_prefix({'phases':prefix}),schedule,proof,guard,worker

    def test_only_approved_current_phase_after_unchanged_prefix_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);actual,sealed,schedule,proof,guard,worker=self.make_fixture(root)
            with patch.object(M,'ROOT',root):
                result=M.validate_appended_rows(actual,sealed,schedule,proof,guard,worker)
            self.assertEqual(result['sealed_original_rows_verified'],1)
            self.assertEqual(result['scheduled_appended_rows_verified'],1)

    def test_prefix_mutation_unknown_phase_worker_or_guard_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);actual,sealed,schedule,proof,guard,worker=self.make_fixture(root)
            cases=[]
            value=deepcopy(actual);value['phases'][0]['actual_cpu_seconds']=0.;cases.append(value)
            value=deepcopy(actual);value['phases'][-1]['phase']='unknown';cases.append(value)
            value=deepcopy(actual);value['phases'][-1]['command'][1]='/different.py';cases.append(value)
            value=deepcopy(actual);value['phases'][-1]['supervisor_proof_sha256']='0'*64;cases.append(value)
            value=deepcopy(actual);value['phases'].append(deepcopy(value['phases'][-1]));cases.append(value)
            with patch.object(M,'ROOT',root):
                for value in cases:
                    with self.subTest(value=value),self.assertRaises(ValueError):
                        M.validate_appended_rows(value,sealed,schedule,proof,guard,worker)

    def test_prior_appended_failure_requires_actual_resource_cause(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);actual,sealed,schedule,proof,guard,worker=self.make_fixture(root)
            prior=deepcopy(actual['phases'][-1]);prior.update(phase='approved-prior',status='PARTIAL_OR_FAILED',
                ended_at_utc='2026-11-06T00:00:00Z',actual_cpu_seconds=2.,exit_code=2,log_sha256=M.sha(prior['log']))
            priorproof=Path(d)/'reports/prior-proof.json';priorproof.write_text(json.dumps({**proof,'phase':prior['phase']}))
            prior.update(supervisor_proof_path=str(priorproof),supervisor_proof_sha256=M.sha(priorproof))
            schedule['phases'].insert(0,{**schedule['phases'][0],'phase':prior['phase']})
            actual['phases'].insert(1,prior)
            with patch.object(M,'ROOT',root):
                with self.assertRaises(ValueError):M.validate_appended_rows(actual,sealed,schedule,proof,guard,worker)
                prior['cpu_terminated']=True
                result=M.validate_appended_rows(actual,sealed,schedule,proof,guard,worker)
                self.assertEqual(result['scheduled_appended_rows_verified'],2)
                prior['monitor_failed']=True
                with self.assertRaises(ValueError):M.validate_appended_rows(actual,sealed,schedule,proof,guard,worker)

    def test_schedule_requires_exact_finite_unique_worker_output_bindings(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);actual,sealed,schedule,proof,guard,worker=self.make_fixture(root)
            p=root/'reports/schedule.json'
            with patch.object(M,'ROOT',root):
                p.write_text(json.dumps(schedule));M.read_schedule(M.ref(p))
                for change in ('duplicate','extra','outside','wildcard'):
                    value=deepcopy(schedule)
                    if change=='duplicate':value['phases'].append(value['phases'][0])
                    if change=='extra':value['phases'][0]['wildcard']=True
                    if change=='outside':value['phases'][0]['output_directory']='/tmp/outside'
                    if change=='wildcard':value['phases'][0]['phase']='allow-any-*'
                    p.write_text(json.dumps(value))
                    with self.subTest(change=change),self.assertRaises(ValueError):M.read_schedule(M.ref(p))


if __name__=='__main__':unittest.main()
