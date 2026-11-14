"""Authorizations extend permission without rewriting scientific or CPU records."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'scripts/background_pyscf_publication_20261111.py'
if not MODULE.exists():
    MODULE = Path(__file__).with_name('background_pyscf_publication_20261111.py')
spec = importlib.util.spec_from_file_location('publication_budget_extension_subject', MODULE)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = b.Paths(Path(self.temp.name))

    def add(self, amount=90000, identifier='additional-20261111-01', text='追加90000CPU秒'):
        return b.extend_budget(self.paths, amount, identifier, text)

    def test_empty_ledger_retains_original_permission(self):
        self.assertEqual(b.effective_authorized_cpu(self.paths), 900000)

    def test_extension_retains_usage_bytes_and_reports_990000(self):
        source=self.paths.runtime/'sessions/previous.json'
        b.write(source,{'session_id':'previous','observed_cpu_seconds':32100,
                        'conservative_cpu_seconds':32150})
        b.write(self.paths.output/'preflight_cpu.json',{'actual_cpu_seconds':100,
                'conservative_cpu_seconds':1000})
        saved=source.read_bytes(); before=b.ledger_totals(self.paths)
        result=self.add()
        self.assertEqual(result['status'],'RECORDED')
        self.assertEqual(source.read_bytes(),saved)
        self.assertEqual(b.ledger_totals(self.paths),before)
        status=b.status(self.paths)
        self.assertEqual(status['authorized_cpu_seconds'],990000)
        self.assertEqual(status['remaining_conservative_cpu_seconds'],990000-before[1])

    def test_repeat_and_concurrent_same_id_are_idempotent(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda _:self.add(),range(4)))
        self.assertEqual(sum(x['status']=='RECORDED'for x in results),1)
        self.assertEqual(b.authorization_ledger(self.paths)['record_count'],1)
        self.assertEqual(b.effective_authorized_cpu(self.paths),990000)
        with self.assertRaises(ValueError):
            self.add(amount=90001)

    def test_later_authorization_links_to_prior_digest(self):
        first=self.add()['authorization']
        second=self.add(1000,'additional-20261111-02','再追加1000CPU秒')['authorization']
        self.assertEqual(second['previous_record_sha256'],first['record_sha256'])
        self.assertEqual(second['previous_total_cpu_seconds'],990000)
        self.assertEqual(b.effective_authorized_cpu(self.paths),991000)

    def test_tamper_and_truncation_fail_closed(self):
        self.add()
        path=next((self.paths.output/'authorizations').glob('*.json'))
        saved=path.read_bytes();record=json.loads(saved)
        record['new_total_cpu_seconds']=99999999
        b.write(path,record)
        with self.assertRaises(ValueError):
            b.effective_authorized_cpu(self.paths)
        path.write_bytes(saved);path.unlink()
        with self.assertRaises(ValueError):
            b.effective_authorized_cpu(self.paths)

    def test_rehashed_wrong_chain_or_unit_is_rejected(self):
        self.add()
        path=next((self.paths.output/'authorizations').glob('*.json'))
        record=json.loads(path.read_text());record['unit']='CPU_HOURS'
        record.pop('record_sha256');record['record_sha256']=b._canonical_digest(record)
        b.write(path,record)
        with self.assertRaises(ValueError):
            b.effective_authorized_cpu(self.paths)

    def test_bad_amount_and_units_cannot_create_permission(self):
        for amount in [0,-1,float('nan'),float('inf'),True,None,'90000']:
            with self.subTest(amount=amount),self.assertRaises(ValueError):
                self.add(amount)
        with self.assertRaises(ValueError):
            b.extend_budget(self.paths,90000,'good-id','authorized',unit='CPU_HOURS')
        self.assertEqual(b.effective_authorized_cpu(self.paths),900000)

    def test_reboot_conservative_recovery_uses_extended_permission(self):
        self.add()
        actual=2000
        charge=b.conservative_recovery_charge({'observed_cpu_seconds':actual,'boot_id':'old'},
                'new',1,2,b.effective_authorized_cpu(self.paths))
        self.assertEqual(charge['observed_cpu_seconds'],actual)
        self.assertEqual(charge['conservative_cpu_seconds'],990000)


if __name__=='__main__':
    unittest.main()
