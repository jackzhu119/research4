"""Metadata compatibility/anti-alias tests only; no Q, Git or array loads."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/research4_cloud_followup_queue_20261106_v2.py'
spec=importlib.util.spec_from_file_location('followup_queue_v2_metadata_tests',SCRIPT)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class OutputBindingCompatibility(unittest.TestCase):
    def fixture(self,root):
        script=root/'scripts/audit_pyscf_saved_cphf_residual_20261105.py'
        script.parent.mkdir();script.write_bytes((ROOT/'scripts/audit_pyscf_saved_cphf_residual_20261105.py').read_bytes())
        output=root/'reports/fixture-output';output.mkdir(parents=True)
        source={'path':str(script),'sha256':M.LEGACY_RESIDUAL_WORKER_SHA,'bytes':script.stat().st_size}
        supervision={'fixture':'not a scientific proof; only tests the metadata helper'}
        record={'schema':M.LEGACY_RESIDUAL_SCHEMA,'source_worker':source,
                'resource_supervision':supervision,'arguments':{'output':str(output)}}
        p=output/'input.json';p.write_text(json.dumps(record))
        result={'schema':M.LEGACY_RESIDUAL_SCHEMA,'status':M.LEGACY_RESIDUAL_SUCCESS,
                'source_worker':source,'resource_supervision':supervision,'input_json_sha256':M.sha(p)}
        job={'worker':source,'result_schema':M.LEGACY_RESIDUAL_SCHEMA}
        row={'command':['python',str(script),'--execute','--output',str(output)]}
        return output,job,row,result,p

    def test_exact_legacy_missing_field_is_bound_without_rewriting_result(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output,job,row,result,_=self.fixture(root);original=deepcopy(result)
            with patch.object(M,'ROOT',root):binding=M.validated_output_binding(job,output,row,result)
            self.assertIn('PINNED_6FF',binding['scope']);self.assertEqual(result,original)

    def test_existing_wrong_or_null_output_is_never_legacy_aliased(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output,job,row,result,_=self.fixture(root)
            for value in [None,str(root/'reports/wrong')]:
                with patch.object(M,'ROOT',root),self.assertRaises(ValueError):
                    M.validated_output_binding(job,output,row,{**result,'output_directory':value})
            with patch.object(M,'ROOT',root):
                self.assertIsNone(M.validated_output_binding(job,output,row,{**result,'output_directory':str(output)}))

    def test_duplicate_or_wrong_guard_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output,job,row,result,_=self.fixture(root)
            for command in [row['command']+['--output',str(output)],row['command'][:-1]+[str(root/'reports/other')]]:
                with patch.object(M,'ROOT',root),self.assertRaises(ValueError):
                    M.validated_output_binding(job,output,{'command':command},result)

    def test_changed_input_hash_source_supervision_or_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output,job,row,result,p=self.fixture(root)
            with patch.object(M,'ROOT',root),self.assertRaises(ValueError):
                M.validated_output_binding(job,output,row,{**result,'input_json_sha256':'0'*64})
            base=json.loads(p.read_text())
            for change in [{'arguments':{'output':str(root/'reports/wrong')}},
                           {'source_worker':{**base['source_worker'],'sha256':'0'*64}},
                           {'resource_supervision':{'fixture':'different'}}]:
                p.write_text(json.dumps({**base,**change}));result['input_json_sha256']=M.sha(p)
                with patch.object(M,'ROOT',root),self.assertRaises(ValueError):
                    M.validated_output_binding(job,output,row,result)

    def test_unknown_worker_hash_or_schema_cannot_use_missing_field_alias(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);output,job,row,result,_=self.fixture(root)
            for change in [{'source_worker':{**result['source_worker'],'sha256':'0'*64}},
                           {'source_worker':{**result['source_worker'],'path':str(root/'scripts/other.py')}},
                           {'schema':'unknown'}, {'status':'PARTIAL'}]:
                with patch.object(M,'ROOT',root),self.assertRaises(ValueError):
                    M.validated_output_binding(job,output,row,{**result,**change})

    def test_all_original_functions_except_seal_have_identical_AST(self):
        old=ROOT/'scripts/research4_cloud_followup_queue_20261106.py'
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(),'6c1c716878107bb4ef69a7f5e57d9831d42f1ac07be50d33f847fd8ccda37e70')
        def functions(path):return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        original,new=functions(old),functions(SCRIPT)
        for name,node in original.items():
            if name!='quantum_seal':self.assertEqual(ast.dump(node,include_attributes=False),ast.dump(new[name],include_attributes=False),name)


if __name__=='__main__':unittest.main()
