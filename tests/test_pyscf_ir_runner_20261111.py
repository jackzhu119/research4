"""Persisted refinement must replace noisy source matrices and keep claim gates."""
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_pyscf_ir_20261111 as r


def test_new_continuous_triples_can_replace_noisy_originals():
    modes=np.eye(3)
    good=np.eye(3)*.1
    pool={.002:good+1,.001:good-1,.0005:good,.00025:good,.000125:good}
    hs,ds,check=r.choose_derivatives(pool,modes,np.array([1.]))
    assert check['qualified']
    assert hs==[.0005,.00025,.000125]
    np.testing.assert_array_equal(ds,np.stack([good]*3))


def test_one_tight_sample_does_not_reuse_loose_samples_as_three_step_proof():
    hs,ds,check=r.choose_derivatives({.001:np.eye(3)},np.eye(3),[1.])
    assert not check['qualified']
    assert check['failures']==['three_consecutive_resolutions_pending']


def test_refinement_plan_never_cycles_fixed_same_precision_steps():
    plans=[r.sampling_plan([.002,.001],i) for i in range(24)]
    assert len(set(plans))==len(plans)
    assert plans[:3]==[(0,.0005),(0,.00025),(0,.004)]
    assert plans[3:6]==[(1,.001),(1,.0005),(1,.00025)]


def test_invalid_larger_step_does_not_block_new_small_resolutions():
    plan=r.resolvable_sampling_plan([.002,.001],15,np.zeros((1,3)))
    assert plan is not None
    assert plan[1]<=.02
    assert plan[2]>15
    assert r.resolvable_sampling_plan([.002,.001],180,np.zeros((1,3))) is None


def test_precision_reset_and_physical_model_guard():
    r.configure_precision('df-grid5',1)
    assert r.electronic.settings()['fresh_orbital_gradient_acceptance']==1e-9
    r.configure_precision('direct-grid5',0)
    assert r.electronic.settings()['fresh_orbital_gradient_acceptance']==1e-8
    case={'method':r.electronic.settings()}
    r.require_same_physical_model(case)
    case['method']['basis']='def2-TZVP'
    with pytest.raises(ValueError,match='basis'):
        r.require_same_physical_model(case)


def test_profile_summary_cannot_hide_failed_direct_water(tmp_path):
    descriptors=[{'id':'a'},{'id':'b'}]
    for identifier in ('a','b'):
        folder=tmp_path/'jobs'/identifier/'round00000'
        folder.mkdir(parents=True)
        (folder/'result.json').write_text('{}')
    results={str(tmp_path/'jobs/a/round00000'):{'molecule':'water','profile':'df-grid5','paper_ready':True},
             str(tmp_path/'jobs/b/round00000'):{'molecule':'water','profile':'direct-grid5','paper_ready':False}}
    with mock.patch.object(r,'OUTPUT',tmp_path),mock.patch.object(r,'_verify_completed',side_effect=lambda p:results[str(p)]):
        summary=r.summarize(descriptors)
    assert summary['water_control_profiles']=={'direct-grid5':False,'df-grid5':True}
    assert not summary['controls_qualified']
    assert summary['status']=='RUNNING_OR_WAITING'


def test_completed_result_or_scientific_array_tampering_fails(tmp_path):
    r.write(tmp_path/'result.json',{'array_sha256':'bad','csv_sha256':'bad'})
    r.write(tmp_path/'COMPLETE.json',{'result_sha256':r.electronic.sha256(tmp_path/'result.json')})
    (tmp_path/'ir_arrays.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='scientific artifact'):
        r._verify_completed(tmp_path)
