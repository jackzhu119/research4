import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec=importlib.util.spec_from_file_location('run_all',Path(__file__).resolve().parents[1]/'scripts/run_all_algorithms.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
PACKAGES={'statistics':'1.7.3','signal':'1.4.6','io':'2.6.4','image':'2.16.0','optim':'1.6.2'}


class UnifiedRunnerTests(unittest.TestCase):
    def write_rows(self,folder,rows):
        path=Path(folder)/'results.json';path.write_text(json.dumps(rows));return path

    def test_zero_process_exit_does_not_hide_case_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.write_rows(tmp,[{'id':'missing_model','status':'fail'}])
            with self.assertRaises(ValueError):runner.read_success_rows(path)

    def test_defect_reproduction_is_not_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.write_rows(tmp,[{'id':'find_wrong_metrics','status':'pass','stage':'isolated_defect'}])
            with self.assertRaises(ValueError):runner.read_success_rows(path)

    def test_empty_results_are_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):runner.read_success_rows(self.write_rows(tmp,[]))

    def test_partial_positive_results_are_not_all_algorithms(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.write_rows(tmp,[{'id':'one_algorithm','status':'pass'}])
            with self.assertRaises(ValueError):runner.read_success_rows(path,{'one_algorithm','missing_algorithm'})

    def fixture(self,root):
        root=Path(root);e=root/'reports/runtime/2026-10-30';e.mkdir(parents=True)
        archive=root/'references/ssa.zip';archive.parent.mkdir();archive.write_bytes(b'original archive')
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        row={'phase':'ssa-full','id':'compat_ssa_full_budget_long','status':'pass','archive':'ssa.zip',
             'source_member':'SSA.m','source_sha256':'member-hash','saved_log':'ssa.log',
             'started_at_shanghai':'2026-10-30T16:25:54+08:00','duration_seconds':512.9}
        (e/'cases.json').write_text(json.dumps([row]))
        (e/'source_hashes.json').write_text(json.dumps([{'archive':'references/ssa.zip','source_sha256':digest,'members':[{'name':'SSA.m','sha256':'member-hash'}]}]))
        log=e/'ssa.log';log.write_text('100\nCHECK_PASSED\n')
        script=e/'harnesses/ssa-full/compat_ssa_full_budget_long/check.m';script.parent.mkdir(parents=True);script.write_text('run SSA.m')
        (e/'octave-environment.log').write_text('GNU Octave Version: 9.4.0 (verified)\n'+''.join(f' {k} *| {v} | path\n' for k,v in PACKAGES.items()))
        manifest=[{'path':str(p.relative_to(e)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [log,script,e/'cases.json',e/'source_hashes.json',e/'octave-environment.log']]
        (e/'evidence_sha256.json').write_text(json.dumps(manifest))
        return e

    @patch.object(runner.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='9.4.0\n'))
    def test_full_ssa_cache_is_explicitly_historical(self,probe):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp);result=runner.verify_cached_ssa(Path(tmp),Path('octave'),PACKAGES)
            self.assertEqual(result['status'],'cached_verified')
            self.assertIn('not reexecuted',result['scope'])

    @patch.object(runner.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='9.4.0\n'))
    def test_tampered_cache_is_rejected(self,probe):
        with tempfile.TemporaryDirectory() as tmp:
            evidence=self.fixture(tmp);(evidence/'ssa.log').write_text('changed result')
            with self.assertRaises(ValueError):runner.verify_cached_ssa(Path(tmp),Path('octave'),PACKAGES)

    @patch.object(runner.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='9.4.0\n'))
    def test_changed_metadata_invalidates_cache(self,probe):
        with tempfile.TemporaryDirectory() as tmp:
            evidence=self.fixture(tmp)
            path=evidence/'cases.json';rows=json.loads(path.read_text());rows[0]['duration_seconds']=1;path.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):runner.verify_cached_ssa(Path(tmp),Path('octave'),PACKAGES)

    @patch.object(runner.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='10.0.0\n'))
    def test_changed_octave_version_does_not_silently_reuse_cache(self,probe):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            with self.assertRaises(ValueError):runner.verify_cached_ssa(Path(tmp),Path('octave'),PACKAGES)


if __name__=='__main__':unittest.main()
