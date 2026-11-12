"""Meaningful orchestration/identity tests: no electronic or force-field calls."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('chemworker_tested', HERE/'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerTests(unittest.TestCase):
    def test_rolling_cleanup_preserves_arrays_and_rejects_bad_seal(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);array=root/'density.npy';np.save(array,np.eye(2))
            rolling=root/'attempt-001/.local/rolling.chk';rolling.parent.mkdir(parents=True);rolling.write_bytes(b'checkpoint')
            result={'status':'DONE','qualified':True,'array_refs':{'density':{'path':str(array),'sha256':worker.sha(array)}}}
            self.assertEqual(worker.cleanup_rolling_checkpoint(root,result),10)
            self.assertTrue(array.is_file());self.assertFalse(rolling.exists())
            rolling.write_bytes(b'checkpoint');result['array_refs']['density']['sha256']='wrong'
            with self.assertRaises(worker.IntegrityError):worker.cleanup_rolling_checkpoint(root,result)
            self.assertTrue(rolling.exists())

    def test_checkpoint_seed_seals_and_coordinates_verified_on_resume(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);coords=np.zeros((75,3));density=root/'density.npy';np.save(density,np.eye(2))
            seed={'coordinates_bohr':coords.tolist(),'density_path':str(density)}
            previous=root/'case/result.json';worker.atomic(previous,{'status':'REFINEMENT_REQUIRED'})
            source=root/'case/negative-mode-recovery/selection.json';worker.atomic(source,{'selected':seed})
            state={'round':1,'profile_index':0,'seed_coordinates_bohr':coords.tolist(),'seed_density_path':str(density),
                'seed_density_sha256':worker.sha(density),'previous_case':str(previous.parent),
                'previous_case_result_sha256':worker.sha(previous),'seed_source_path':str(source),
                'seed_source_sha256':worker.sha(source),'seed_kind':'supported_negative_branch'}
            got,dm=worker.verified_checkpoint_seed(root,state,coords);self.assertTrue(np.array_equal(got,coords))
            tampered=copy.deepcopy(state);tampered['seed_coordinates_bohr'][0][0]=1
            with self.assertRaises(worker.IntegrityError):worker.verified_checkpoint_seed(root,tampered,coords)
            np.save(density,np.ones((2,2)))
            with self.assertRaises(worker.IntegrityError):worker.verified_checkpoint_seed(root,state,coords)

    def test_alignment_ignores_rigid_motion_but_not_reflection(self):
        rng=np.random.default_rng(42);xyz=rng.normal(size=(75,3))
        theta=.7;r=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        delta=worker.aligned_geometry_difference(xyz,xyz@r+np.array([2,3,4]))
        self.assertLess(delta['maximum_atom_deviation_bohr'],1e-12)
        mirror=xyz.copy();mirror[:,0]*=-1
        self.assertGreater(worker.aligned_geometry_difference(xyz,mirror)['all_atom_rmsd_bohr'],.1)

    def test_local_basin_waits_only_for_earlier_eligible_job(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);coords=np.random.default_rng(2).normal(size=(75,3))
            jobs=[{'id':'earlier','ordinal':0},{'id':'later','ordinal':1}]
            worker.atomic(root/'jobs.json',{'jobs':jobs})
            settings={'profile':'df-grid5'}
            opt={'optimizer_converged':True,'minimum_stationarity_passed':True,'settings':settings,
                 'coordinates_bohr':coords.tolist(),'final_point':{'energy_hartree':-5.}}
            other=root/'jobs/earlier';case=other/'science/df-grid5/round000000'
            worker.atomic(other/'job_frozen.json',{'identity':{'inchikey':worker.KEY},'job':{'conformer':'C002'}})
            worker.atomic(case/'optimize/result.json',opt)
            own=root/'jobs/later/science/df-grid5/round000000';worker.atomic(own/'optimize/result.json',opt)
            core=types.SimpleNamespace(settings=lambda:settings)
            runner=types.SimpleNamespace(sealed=lambda p:None)
            evidence=worker.local_duplicate_evidence(root,{'id':'later','ordinal':1},coords,None,own/'finite-differences',core,runner)
            self.assertEqual(evidence['canonical_job_id'],'earlier')
            self.assertEqual(evidence['canonical_hessian_status'],'AWAITING_COMPLETE_QUALIFIED_EVIDENCE')
            self.assertIsNone(worker.local_duplicate_evidence(root,{'id':'earlier','ordinal':0},coords,None,case/'finite-differences',core,runner))
            worker.atomic(other/'result.json',{'status':'scientific_blocked'})
            self.assertIsNone(worker.local_duplicate_evidence(root,{'id':'later','ordinal':1},coords,None,own/'finite-differences',core,runner))

    def test_reused_optimizer_step_has_separate_scientific_point(self):
        p=Path('/tmp/optimize/steps/step-00005')
        self.assertNotEqual(worker.optimizer_point_path(p,'first'),worker.optimizer_point_path(p,'second'))
        fd=Path('/tmp/finite-differences/h0/c005-plus')
        self.assertEqual(worker.optimizer_point_path(fd,'first'),fd)

    def test_two_authenticated_distinct_geometries(self):
        results=[]
        for c in ('C002','C003'):
            symbols,coords,metadata=worker.authenticate_inputs(HERE,{'conformer':c})
            self.assertEqual(len(symbols),75)
            self.assertEqual(metadata['inchikey'],worker.KEY)
            results.append(coords)
        # All-pair distances are invariant to rotation/translation. Different
        # matrices therefore prove these are more than rigid-body duplicates.
        d=[np.linalg.norm(x[:,None]-x[None,:],axis=2) for x in results]
        self.assertGreater(float(np.max(np.abs(d[0]-d[1]))),1.)

    def test_unreviewed_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'changed.xyz';p.write_text('75\nchanged\n')
            with self.assertRaises(worker.IntegrityError):
                worker.authenticate_inputs(HERE,{'conformer':'C002','input_xyz':str(p)})

    def test_profile_advancement_and_more_than_six_negative_rounds(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); density=root/'density.npy';np.save(density,np.eye(2))
            seen=[]
            def stationary(name,profile,coords,dm,out):
                seen.append((profile,int(out.name.replace('round',''))))
                out.mkdir(parents=True,exist_ok=True)
                n=seen[-1][1]
                case={'status':'ELECTRONIC_STATE_UNRESOLVED','paper_ready':False}
                if profile=='df-grid7':
                    case={'status':'REFINEMENT_REQUIRED','paper_ready':False,'analysis':{},
                          'coordinates_bohr':coords.tolist(),'density_path':str(density)}
                    if n==8:
                        center=out/'point.json';worker.atomic(center,{'status':'DONE'})
                        case.update(status='QUALIFIED',paper_ready=True,center_point_path=str(center),
                            electronic_stability={'tested_electronic_stability_passed':True},method={'profile':profile},
                            energy_hartree=-1.,analysis={'steps':[{'vibrational_mode_count':219}]})
                worker.atomic(out/'result.json',case);return case
            def branches(name,profile,case,out):
                value={'status':'BOTH_SIGNS_RELAXED','selected':{
                    'coordinates_bohr':case['coordinates_bohr'],'density_path':str(density)}}
                worker.atomic(out/'selection.json',value);return value
            runner=types.SimpleNamespace(stationary_case=stationary,negative_branches=branches,
                quality=types.SimpleNamespace(audit_publication=lambda *a,**k:{'paper_ready_molecular_harmonic_data':True}),
                modes=types.SimpleNamespace(verify_analysis=lambda p:True))
            result=worker.scientific_loop({'profile':'df-grid5'},root,runner,None,np.zeros((75,3)),lambda **v:None)
            self.assertEqual(result['profile'],'df-grid7')
            self.assertEqual(result['scientific_round'],8)
            self.assertEqual(len(seen),10)
            self.assertTrue(all(p=='df-grid7' for p,n in seen[1:]))
            self.assertFalse(result['independent_conformer_sample'])

    def test_exhausted_profile_is_blocked_not_empty_retry(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);seen=[]
            def stationary(name,profile,coords,dm,out):
                seen.append(profile);out.mkdir(parents=True,exist_ok=True)
                value={'status':'ELECTRONIC_STATE_UNRESOLVED','paper_ready':False}
                worker.atomic(out/'result.json',value);return value
            runner=types.SimpleNamespace(stationary_case=stationary)
            result=worker.scientific_loop({'profile':'df-grid5'},root,runner,None,np.zeros((75,3)),lambda **v:None)
            self.assertEqual(seen,['df-grid5','df-grid7','direct-grid7'])
            self.assertTrue(result['scientific_blocked'])
            self.assertFalse(result['complete'])
            self.assertFalse((root/'done.json').exists())

    def test_identity_failure_does_not_escalate_into_other_molecule(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);seen=[]
            def stationary(name,profile,coords,dm,out):
                seen.append(profile);out.mkdir(parents=True,exist_ok=True)
                value={'status':'IDENTITY_GUARD_FAILED','paper_ready':False}
                worker.atomic(out/'result.json',value);return value
            result=worker.scientific_loop({'profile':'df-grid5'},root,
                types.SimpleNamespace(stationary_case=stationary),None,np.zeros((75,3)),lambda **v:None)
            self.assertTrue(result['scientific_blocked']);self.assertEqual(len(seen),1)


if __name__=='__main__':unittest.main(verbosity=2)
