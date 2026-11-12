"""Regression checks for actual detached lifetime, CPU carry-over and publication isolation."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'scripts/background_all_round_validation_20261112.py'
spec = importlib.util.spec_from_file_location('all_round_background',SOURCE)
background=importlib.util.module_from_spec(spec)
spec.loader.exec_module(background)


def sample(counter, boot='boot1', inode=10):
    return {'counter_cpu_seconds':counter,'boot_id':boot,'cgroup_device':1,
            'cgroup_inode':inode,'counter_path':'/sys/fs/cgroup/cpu.stat','sampled_at_utc':'2026-11-12T00:00:00+00:00'}


class AccountingTests(unittest.TestCase):
    def test_preparation_is_included_and_restarts_do_not_reset(self):
        first=background.account_sample({},sample(123.5))
        self.assertEqual(first['observed_cpu_seconds'],123.5)
        resumed=background.account_sample(first,sample(200.0))
        self.assertEqual(resumed['observed_cpu_seconds'],200.0)
        replacement=background.account_sample(resumed,sample(10.0,boot='boot2'))
        self.assertEqual(replacement['observed_cpu_seconds'],210.0)
        self.assertEqual(replacement['remaining_cpu_seconds'],989790.0)
        self.assertIn('UNKNOWN',replacement['counter_epochs'][0]['unobserved_tail_after_cgroup_loss'])
        self.assertEqual(replacement['early_cpu_reserve_seconds'],0)

    def test_counter_rollback_is_an_error(self):
        ledger=background.account_sample({},sample(20))
        with self.assertRaisesRegex(ValueError,'backwards'):
            background.account_sample(ledger,sample(10))

    def test_lost_counter_tail_is_charged_as_uncertainty_not_measured_cpu(self):
        initial=sample(100)
        initial['cpu_capacity']=4
        ledger=background.account_sample({},initial)
        replacement=sample(10,boot='boot2')
        replacement['sampled_at_utc']='2026-11-12T00:00:02+00:00'
        updated=background.account_sample(ledger,replacement)
        self.assertEqual(updated['observed_cpu_seconds'],110)
        self.assertEqual(updated['uncertain_upper_cpu_seconds'],8)
        self.assertEqual(updated['conservative_cpu_seconds'],118)
        self.assertEqual(updated['remaining_cpu_seconds'],989882)

    def test_boundary_has_no_early_reserve_and_reports_overshoot(self):
        before=background.account_sample({},sample(989999.9))
        self.assertAlmostEqual(before['remaining_cpu_seconds'],0.1)
        after=background.account_sample(before,sample(990000.2))
        self.assertAlmostEqual(after['boundary_overshoot_cpu_seconds'],0.2)
        self.assertEqual(after['remaining_cpu_seconds'],0)

    def test_missing_or_blocked_marker_cannot_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            task={'completion':{'path':'result.json','field':'status','equals':'COMPLETE'}}
            self.assertFalse(background.completed(root,task))
            background.write(root/'result.json',{'status':'BLOCKED'})
            self.assertFalse(background.completed(root,task))
            background.write(root/'result.json',{'status':'COMPLETE'})
            self.assertTrue(background.completed(root,task))
            task['completion'].update(field='qualified',equals=True)
            background.write(root/'result.json',{'qualified':1})
            self.assertFalse(background.completed(root,task))


class DetachedIntegrationTests(unittest.TestCase):
    def test_explicit_scientific_input_directory_freezes_non_python_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'scripts').mkdir()
            shutil.copyfile(SOURCE,root/background.SELF)
            inputs=root/'data/research/fixture_inputs'
            inputs.mkdir(parents=True)
            (inputs/'coordinates.npy').write_bytes(b'array evidence')
            (inputs/'symbols.json').write_text('["H","O","H"]')
            background.write(root/background.CONFIG,{'tasks':[{'id':'fixture','command':['unused'],
                'completion':{'path':'done.json','equals':'COMPLETE'}}],
                'freeze_paths':['scripts','data/research/fixture_inputs']})
            background.freeze(root)
            self.assertEqual((background.frozen(root)/'data/research/fixture_inputs/coordinates.npy').read_bytes(),b'array evidence')
            self.assertEqual((background.frozen(root)/'data/research/fixture_inputs/symbols.json').read_text(),'["H","O","H"]')
            background.check_integrity(root)

    def test_detached_work_completes_and_publisher_preserves_user_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'work'
            root.mkdir()
            remote=Path(directory)/'remote.git'
            def git(*args):
                return subprocess.run(['git',*args],cwd=root,text=True,capture_output=True,check=True).stdout.strip()
            git('init','-b','research/test-validation')
            git('config','user.name','Validation Test')
            git('config','user.email','validation@example.invalid')
            subprocess.run(['git','init','--bare',str(remote)],text=True,capture_output=True,check=True)
            git('remote','add','origin',str(remote))
            (root/'scripts').mkdir()
            shutil.copyfile(SOURCE,root/background.SELF)
            worker=root/'scripts/fixture_worker.py'
            worker.write_text('import json, os, pathlib, sys\n'
                              'root=pathlib.Path(sys.argv[1])\n'
                              'output=root/"reports/all_round_validation_20261112/fixture_result.json"\n'
                              'output.parent.mkdir(parents=True,exist_ok=True)\n'
                              'output.write_text(json.dumps({"status":"COMPLETE","pid":os.getpid(),"sid":os.getsid(0)}))\n')
            task={'id':'fixture','command':['{python}','{frozen}/scripts/fixture_worker.py','{root}'],
                  'completion':{'path':'reports/all_round_validation_20261112/fixture_result.json','field':'status','equals':'COMPLETE'}}
            background.write(root/background.CONFIG,{'tasks':[task],'freeze_paths':['scripts'],
                             'publish_paths':[background.RELATIVE_OUTPUT]})
            (root/'unrelated.txt').write_text('original\n')
            git('add','.')
            git('commit','-m','fixture source')
            original_head=git('rev-parse','HEAD')
            git('push','origin','HEAD')
            (root/'unrelated.txt').write_text('user staged change\n')
            git('add','unrelated.txt')
            original_index=git('diff','--cached','--binary')
            state=None
            try:
                launched=background.start(root)
                self.assertTrue(launched['supervisor_alive'])
                self.assertTrue(launched['guard_alive'])
                supervisor=launched['state']['supervisor_identity']
                self.assertEqual(supervisor['pid'],supervisor['sid'])
                self.assertNotEqual(supervisor['sid'],os.getsid(0))
                deadline=time.monotonic()+20
                while time.monotonic()<deadline:
                    state=background.read(background.runtime(root)/'state.json',{})
                    if state.get('status')=='COMPLETE':
                        break
                    time.sleep(.1)
                self.assertEqual(state.get('status'),'COMPLETE',state)
                fixture=background.read(root/task['completion']['path'])
                self.assertEqual(fixture['pid'],fixture['sid'])
                self.assertEqual(git('rev-parse','HEAD'),original_head)
                self.assertEqual(git('diff','--cached','--binary'),original_index)
                published=json.loads(subprocess.run(['git','--git-dir',str(remote),'show',
                    'refs/heads/research/test-validation:'+task['completion']['path']],
                    text=True,capture_output=True,check=True).stdout)
                self.assertEqual(published['status'],'COMPLETE')
                self.assertTrue(background.verify(root)['verified'])
                (background.frozen(root)/'scripts/fixture_worker.py').write_text('corrupt\n')
                self.assertFalse(background.verify(root)['verified'])
            finally:
                background.write(background.runtime(root)/'STOP.json',{'reason':'Test cleanup'})
                state=background.read(background.runtime(root)/'state.json',{})
                background.halt_workers(state)
                background.signal_session(state.get('supervisor_identity'))
                background.signal_session(state.get('guard_identity'))


if __name__=='__main__':
    unittest.main()
