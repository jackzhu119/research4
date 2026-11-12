"""Identity/archive/native-fallback tests. No electronic or gradient calls."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('resource_adapter_tested',ROOT/'resource_worker.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)


class ResourceAdapterTests(unittest.TestCase):
    def payload(self):
        settings={'xc':'PBE','basis':'def2-SVP','profile':'df-grid5','grid_level':5,'charge':0,'spin':0,
                  'grid_response':True,'auxbasis_response':True,'max_memory_mb':2200}
        unsigned={'operation':'point','source_sha256':'source','pyscf_version':'2.13.0',
                  'settings':settings,'symbols':['O','H','H'],
                  'coordinates_bohr':[[0.,0.,0.],[1.,0.,1.],[-1.,0.,1.]]}
        return unsigned|{'input_fingerprint':adapter.digest_json(unsigned)}

    def check(self,payload,**changes):
        return adapter.assert_exact_point_binding(payload,
            changes.get('symbols',payload['symbols']),changes.get('coordinates',payload['coordinates_bohr']),
            changes.get('settings',payload['settings']),payload['input_fingerprint'],
            changes.get('source_sha','source'),changes.get('pyscf_version','2.13.0'))

    def test_exact_geometry_method_binding(self):self.check(self.payload())

    def test_geometry_and_atom_order_changes_rejected(self):
        p=self.payload();xyz=np.array(p['coordinates_bohr']);xyz[1,0]+=.001
        with self.assertRaises(adapter.AdapterIntegrityError):self.check(p,coordinates=xyz)
        with self.assertRaises(adapter.AdapterIntegrityError):self.check(p,symbols=['H','O','H'])

    def test_profile_basis_response_charge_or_version_changes_rejected(self):
        p=self.payload()
        for key,value in [('profile','df-grid7'),('basis','def2-TZVP'),('grid_response',False),('auxbasis_response',False),('charge',1)]:
            settings=dict(p['settings']);settings[key]=value
            with self.assertRaises(adapter.AdapterIntegrityError):self.check(p,settings=settings)
        with self.assertRaises(adapter.AdapterIntegrityError):self.check(p,pyscf_version='different')
        with self.assertRaises(adapter.AdapterIntegrityError):self.check(p,source_sha='different')

    def test_modified_payload_with_old_fingerprint_rejected(self):
        p=self.payload();p['coordinates_bohr'][0][0]=1.
        with self.assertRaises(adapter.AdapterIntegrityError):self.check(p)

    def test_archive_survives_removal_of_original(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=root/'point/attempt-002/.local/rolling.chk';source.parent.mkdir(parents=True)
            source.write_bytes(b'bound-orbital-state');digest=adapter.sha(source)
            result=adapter.archive_checkpoint(root,{'checkpoint_path':str(source),'checkpoint_sha256':digest})
            source.unlink();archive=Path(result['archived_checkpoint_path'])
            self.assertEqual(adapter.sha(archive),digest)
            self.assertEqual(archive.parent,root/'recovery_checkpoints')

    def test_archive_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as t:
            source=Path(t)/'rolling.chk';source.write_bytes(b'changed')
            with self.assertRaises(adapter.AdapterIntegrityError):
                adapter.archive_checkpoint(Path(t),{'checkpoint_path':str(source),'checkpoint_sha256':'wrong'})

    def fake_core(self,root,fresh_qualified=True):
        from types import SimpleNamespace
        settings={'profile':'df-grid5'}
        mf=SimpleNamespace(make_rdm1=lambda:np.eye(2))
        calls=[]
        def native(*args):
            calls.append(args[4]);return mf,np.eye(2),{'qualified':True},[{'solver':'NATIVE_RECOMPUTED'}]
        def atomic(path,value):
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        core=SimpleNamespace(__file__=str(ROOT/'worker.py'),_run_electronic=native,
            settings=lambda grid:settings,new_molecule=lambda *a:None,make_scf=lambda *a:mf,
            _audit=lambda obj:(np.eye(2),{'qualified':fresh_qualified,'native_converged':True}),atomic_json=atomic)
        return core,calls

    def test_verified_native_state_requires_new_fresh_audit(self):
        for accepted in [True,False]:
            with self.subTest(fresh_qualified=accepted),tempfile.TemporaryDirectory() as t:
                root=Path(t);(root/'worker.py').write_text('original')
                checkpoint=root/'old.chk';checkpoint.write_bytes(b'bound')
                state={'mo_coeff':np.eye(2),'mo_occ':np.array([2.,0.]),'mo_energy':np.array([-1.,1.]),'e_tot':-1.}
                prior={'actual_conv_tol_hartree':1e-12,'actual_conv_tol_grad':2e-9}
                proof={'checkpoint_path':str(checkpoint),'checkpoint_sha256':adapter.sha(checkpoint)}
                core,calls=self.fake_core(root,accepted)
                with patch.object(adapter,'install_lifecycle_patch',return_value={}),patch.object(adapter,'load_bound_checkpoint',return_value=(state,prior,proof)):
                    adapter.install_electronic_adapter(root,core)
                    mf,density,audit,trace=core._run_electronic(['H'],np.zeros((1,3)),root/'new',None,None,None)
                receipt=json.loads((root/'new/resume_provenance.json').read_text())
                self.assertEqual(len(calls),0 if accepted else 1)
                self.assertTrue(audit['qualified'])
                self.assertNotIn('gradient',receipt)
                self.assertNotIn('complete',receipt)
                self.assertEqual(trace[0]['solver'],'AUTHENTICATED_NATIVE_CHECKPOINT_FRESH_AUDIT' if accepted else 'AUTHENTICATED_CHECKPOINT_DENSITY_WARM_START')
                if not accepted:self.assertTrue(np.array_equal(calls[0],np.eye(2)))

    def test_two_real_checkpoint_identity_catalogs_without_scf(self):
        source=ROOT/'source'/adapter.PUB_COMMIT/'scripts';sys.path.insert(0,str(source))
        core=__import__('pyscf_publication_core_20261111');core.configure('df-grid5',threads=1);core.SCF_SETTINGS['max_memory_mb']=2200
        for fingerprint,catalog in adapter.CHECKPOINT_CATALOG.items():
            point=ROOT/catalog['point_relative'];p=json.loads((point/'input.json').read_text())
            state,prior,proof=adapter.load_bound_checkpoint(ROOT,core,p['symbols'],np.array(p['coordinates_bohr']),point/'new-test-attempt',None,None)
            self.assertEqual(state['mo_coeff'].shape,(762,762));self.assertEqual(float(state['mo_occ'].sum()),320.)
            self.assertEqual(proof['input_fingerprint'],fingerprint)


if __name__=='__main__':unittest.main(verbosity=2)
