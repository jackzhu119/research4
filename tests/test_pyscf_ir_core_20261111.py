"""Catch displacement, unit, resume and provenance faults without costly SCFs."""
from pathlib import Path
import sys
import tempfile
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pyscf_ir_core_20261111 as c


def test_normal_mode_path_differentiates_in_mass_normalized_coordinate():
    symbols=['H','He']
    coords=np.array([[0.,0.,0.],[1.,0.,0.]])
    vectors=np.eye(6)[:,[3,4]]/2  # mass 4; no Euclidean renormalization
    jacobian=np.arange(18,dtype=float).reshape(3,6)/10
    visited=[]
    def point(symbols,r,out,density):
        visited.append(np.array(r))
        return {'dipole_au':(jacobian@r.ravel()).tolist(),
                'input_fingerprint':'synthetic','qualified':True}
    with tempfile.TemporaryDirectory() as temp, mock.patch.object(c,'qualified_dipole',point):
        direct,ledger=c.directional_derivatives(symbols,coords,vectors,temp,
                                               [.004,.002,.001],None)
    np.testing.assert_allclose(direct,np.broadcast_to(jacobian@vectors,(3,3,2)),atol=1e-12)
    assert ledger[0]['maximum_cartesian_displacement_bohr']==.002
    assert len(visited)==12
    np.testing.assert_allclose(visited[0]-coords,.004*vectors[:,0].reshape(2,3))


def test_cartesian_atom_major_layout_and_odd_nonlinear_contamination():
    coords=np.zeros((2,3))
    jacobian=np.arange(18,dtype=float).reshape(3,6)/7
    def point(symbols,r,out,density):
        return {'dipole_au':(jacobian@r.ravel()+r.ravel().sum()**3).tolist()}
    with tempfile.TemporaryDirectory() as temp, mock.patch.object(c,'qualified_dipole',point):
        d=c.cartesian_derivative(['H','H'],coords,temp,.001,None)
    np.testing.assert_allclose(d,jacobian+1e-6,rtol=1e-12,atol=1e-12)


def test_unqualified_point_is_retryable_and_cannot_enter_derivative():
    with mock.patch.object(c,'calculate_dipole',return_value={'qualified':False}):
        with pytest.raises(RuntimeError,match='requires retry'):
            c.qualified_dipole(['H'],np.zeros((1,3)),'.')


def test_fingerprint_covers_density_geometry_and_electronic_model():
    c.electronic.configure('df-grid5')
    coords=np.zeros((1,3))
    base,_=c.fingerprint(['He'],coords,None)
    assert base!=c.fingerprint(['He'],coords+.001,None)[0]
    assert base!=c.fingerprint(['He'],coords,np.eye(2))[0]
    c.electronic.configure('direct-grid5')
    assert base!=c.fingerprint(['He'],coords,None)[0]


def test_cache_rejects_changed_array_before_solver_is_called(tmp_path):
    c.electronic.configure('df-grid5')
    coords=np.zeros((1,3))
    digest,_=c.fingerprint(['He'],coords,None)
    a=tmp_path/'dipole.npy'
    np.save(a,np.ones(3))
    c.electronic.atomic_json(tmp_path/'result.json',{
        'input_fingerprint':digest,'status':'DONE','qualified':True,
        'array_refs':{'dipole':{'path':str(a),'sha256':'wrong'}}})
    with mock.patch.object(c.electronic,'_run_electronic') as solver:
        with pytest.raises(ValueError,match='array fingerprint changed'):
            c.calculate_dipole(['He'],coords,tmp_path)
        solver.assert_not_called()
