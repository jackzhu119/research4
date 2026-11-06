"""Offline bootstrap integration: fake Git/bash, never install, push or run Q."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/bootstrap_research4_worker_b_20261106.sh'
PIN = 'c2284ad4755b223830aeb2283d7491a3a3f35a28'

FAKE_GIT = r'''#!PYTHON
import json,os,pathlib,shutil,sys
args=sys.argv[1:]
with open(os.environ['BOOT_TEST_LOG'],'a') as f:f.write(json.dumps(['git',*args])+'\n')
if args[:1]==['clone']:
 p=pathlib.Path(args[-1]);(p/'.git').mkdir(parents=True);(p/'scripts').mkdir();shutil.copyfile(os.environ['BOOT_LAUNCHER_FIXTURE'],p/'scripts/run_research4_worker_b_20261106.sh');sys.exit()
if args[:1]==['-C']:args=args[2:]
if args[:3]==['remote','get-url','origin']:print('https://github.com/jackzhu119/research4.git')
elif args[:2]==['branch','--show-current']:print('research/parallel-worker-b-20261106')
elif args[:2]==['rev-parse','HEAD']:print(os.environ.get('BOOT_TEST_HEAD','c2284ad4755b223830aeb2283d7491a3a3f35a28'))
elif args[:1]==['config']:
 if '--name-only' in args:sys.exit(1)
 if '--get' in args:print('Research4 Test')
elif args[:1]==['push'] and os.environ.get('BOOT_TEST_DENY'):
 print('remote: Permission denied (FAKE)',file=sys.stderr);sys.exit(128)
'''
FAKE_BASH = r'''#!PYTHON
import json,os,sys
with open(os.environ['BOOT_TEST_LOG'],'a') as f:f.write(json.dumps(['child-bash',*sys.argv[1:]])+'\n')
'''

class BootstrapOfflineTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name);self.bin=self.base/'bin';self.bin.mkdir();self.repo=self.base/'repo';self.log=self.base/'calls.jsonl';self.env=dict(os.environ)
  self.env.update(PATH=str(self.bin)+os.pathsep+os.environ['PATH'],BOOT_TEST_LOG=str(self.log),BOOT_LAUNCHER_FIXTURE=str(ROOT/'scripts/run_research4_worker_b_20261106.sh'),TMPDIR=str(self.base))
  for name,source in [('git',FAKE_GIT),('bash',FAKE_BASH)]:
   p=self.bin/name;p.write_text(source.replace('#!PYTHON','#!'+sys.executable));p.chmod(0o700)
 def run_boot(self,*flags):return subprocess.run(['/usr/bin/bash',str(SCRIPT),'--repo-directory',str(self.repo),*flags],env=self.env,text=True,capture_output=True,timeout=20)
 def calls(self):return [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.exists() else []
 def test_check_only_has_no_git_clone_or_install(self):
  p=self.run_boot('--check-only');self.assertEqual(p.returncode,0,p.stderr);self.assertFalse(self.repo.exists());self.assertEqual(self.calls(),[])
 def test_unknown_argument_stops(self):
  p=self.run_boot('--unknown');self.assertEqual(p.returncode,2);self.assertEqual(self.calls(),[])
 def test_mock_full_setup_delegates_check_then_detach(self):
  p=self.run_boot();self.assertEqual(p.returncode,0,p.stderr);children=[x for x in self.calls() if x[0]=='child-bash'];self.assertEqual(len(children),2);self.assertIn('--check-only',children[0]);self.assertIn('--detach',children[1]);self.assertIn('B20261106-01',children[1]);self.assertTrue(any('push' in x and '--dry-run' in x for x in self.calls()))
 def test_write_failure_does_not_start_science(self):
  self.env['BOOT_TEST_DENY']='1';p=self.run_boot();self.assertEqual(p.returncode,2);self.assertIn('Git write preflight failed',p.stderr);self.assertFalse(any(x[0]=='child-bash' for x in self.calls()));self.assertNotIn('Permission denied (FAKE)',p.stderr)
 def test_full_sha_required_not_seven_digit_prefix(self):
  self.env['BOOT_TEST_HEAD']='c2284ad'+'0'*33;p=self.run_boot();self.assertNotEqual(p.returncode,0);self.assertIn('differs from frozen B commit',p.stderr);self.assertFalse(any(x[0]=='child-bash' for x in self.calls()))
 def test_existing_unknown_directory_is_preserved(self):
  self.repo.mkdir();sentinel=self.repo/'original';sentinel.write_text('KEEP');p=self.run_boot();self.assertEqual(p.returncode,2);self.assertEqual(sentinel.read_text(),'KEEP');self.assertFalse(any('clone' in x for x in self.calls()))
 def test_device_login_does_not_override_inherited_auth(self):
  self.env.update(BOOT_TEST_DENY='1',GIT_ASKPASS='/not-used-test-auth-marker');p=self.run_boot('--device-login');self.assertEqual(p.returncode,2);self.assertIn('existing authentication is preserved',p.stderr);self.assertFalse(any(x[0]=='child-bash' for x in self.calls()))

if __name__ == '__main__':unittest.main()
