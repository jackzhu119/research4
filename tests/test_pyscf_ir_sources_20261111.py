"""Scientific source authentication regressions without external services."""
import copy
import io
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

import pyscf_ir_sources_20261111 as source

CASE = 'reports/pyscf_publication_20261111/jobs/acid/df-grid5/round000'


def encode(value):
    return json.dumps(value,sort_keys=True,allow_nan=False).encode()


def npy(value):
    stream=io.BytesIO();np.save(stream,np.asarray(value),allow_pickle=False)
    return stream.getvalue()


class FakeReader:
    def __init__(self,blobs):
        self.blobs=blobs
        self.objects={key:'fake-oid' for key in blobs}
    def read(self,path):
        return self.blobs[path]
    def read_many(self,paths):
        return {path:self.blobs[path] for path in paths}


def fixture(*,ir_ready=True,point_qualified=True):
    symbols=['H','H'];coords=np.array([[0.,0.,0.],[0.,0.,1.]])
    ncart=6;steps=[0.002,0.001];method={'xc':'PBE','profile':'df-grid5','charge':0}
    audit={'native_converged':True,'qualified':True,'fresh_orbital_gradient_norm':1e-10,
           'fresh_energy_difference_hartree':0.,'electron_count_error':0.}
    center={'symbols':symbols,'coordinates_bohr':coords.tolist(),'settings':method,
        'qualified':True,'native_converged':True,'energy_hartree':-1.,
        'dipole_au':[0.,0.,0.],'gradient':np.zeros_like(coords).tolist(),'electronic_audit':audit}
    blobs={};arrays={'coordinates_bohr':coords}
    direction=np.array([[0.],[0.],[1/np.sqrt(2)],[0.],[0.],[-1/np.sqrt(2)]])
    point_i=0
    for hi,h in enumerate(steps):
        hs,ds=[],[]
        for j in range(ncart):
            pair=[]
            for sign,label in ((1,'plus'),(-1,'minus')):
                r=coords.copy();r.flat[j]+=sign*h
                point={**copy.deepcopy(center),'coordinates_bohr':r.tolist(),
                    'gradient':(r-coords).tolist(),'dipole_au':(r.sum(axis=0)-coords.sum(axis=0)).tolist(),
                    'input_fingerprint':f'h{hi}-c{j}-{label}'}
                if not point_qualified and point_i==0:
                    point['qualified']=False
                point_i+=1
                base=CASE+f'/finite-differences/h{hi}/c{j:03d}-{label}/'
                blobs[base+'result.json']=encode(point)
                for key,file in [('coordinates_bohr','coordinates_bohr.npy'),('gradient','gradient.npy'),('dipole_au','dipole_au.npy')]:
                    blobs[base+file]=npy(point[key])
                pair.append(point)
            hs.append((np.asarray(pair[0]['gradient'])-np.asarray(pair[1]['gradient'])).ravel()/(2*h))
            ds.append((np.asarray(pair[0]['dipole_au'])-np.asarray(pair[1]['dipole_au']))/(2*h))
        arrays.update({f'step_{hi}_raw_hessian':np.stack(hs,axis=1),
            f'step_{hi}_dipole_derivative_au':np.stack(ds,axis=1),
            f'step_{hi}_masses_amu':np.ones(2),
            f'step_{hi}_cartesian_mass_normalized_modes':direction,
            f'step_{hi}_mass_weighted_modes':direction,
            f'step_{hi}_signed_frequencies_cm1':np.array([100.])})
    stream=io.BytesIO();np.savez_compressed(stream,**arrays)
    blobs[CASE+'/modes/analysis_arrays.npz']=stream.getvalue()
    blobs[CASE+'/modes/all_modes.csv']=b'mode,frequency\n1,100\n'
    provenance={'symbols':symbols,'steps_bohr':steps}
    mode={'input_provenance':provenance,'input_fingerprint':source._sha(source._canonical(provenance)),
        'symbols':symbols,'matrix_count':len(steps),
        'artifacts':{name:{'path':name,'sha256':source._sha(blobs[CASE+'/modes/'+name])}
            for name in ('analysis_arrays.npz','all_modes.csv')}}
    blobs[CASE+'/modes/result.json']=encode(mode)
    center_path=CASE+'/optimize/final-center/result.json'
    blobs[center_path]=encode(center)
    gate={'at_utc':'earlier','minimum_verified':True,
        'paper_ready_molecular_harmonic_data':ir_ready,
        'gates':{'source_artifacts_authenticated':True,'ir_step_convergence_passed':ir_ready},
        'thresholds':{'quality':1e-8},'failed_gates':[] if ir_ready else ['ir_step_convergence_passed'],
        'center_point_sha256':source._sha(source._canonical(center)),
        'original_displacement_point_count':ncart*2*len(steps),
        'authenticated_artifact_sha256':{'/workspace/research4/'+key:source._sha(value)
            for key,value in blobs.items() if key!=center_path}}
    case={'molecule':'acid','profile':'df-grid5','coordinates_bohr':coords.tolist(),
          'method':method,'analysis':mode,'center_point_path':'/workspace/research4/'+center_path,
          'publication_gate':gate}
    blobs[CASE+'/modes/publication_quality.json']=encode({**gate,'at_utc':'later'})
    blobs[CASE+'/result.json']=encode(case)
    blobs[CASE+'/COMPLETE.json']=encode({'result_sha256':source._sha(blobs[CASE+'/result.json'])})
    return FakeReader(blobs)


def reseal(reader):
    """Keep crypto seals internally valid while testing semantic corruption."""
    case=json.loads(reader.blobs[CASE+'/result.json'])
    gate=case['publication_gate']
    for path in list(gate['authenticated_artifact_sha256']):
        relative=source._relative(path)
        gate['authenticated_artifact_sha256'][path]=source._sha(reader.blobs[relative])
    reader.blobs[CASE+'/modes/publication_quality.json']=encode(gate)
    reader.blobs[CASE+'/result.json']=encode(case)
    reader.blobs[CASE+'/COMPLETE.json']=encode({'result_sha256':source._sha(reader.blobs[CASE+'/result.json'])})


def test_roundtrip_authenticates_every_original_point(tmp_path):
    descriptor=source.import_case(fixture(),CASE,tmp_path,'known-branch','a'*40)
    value=source.load_source(descriptor)
    assert value['source_minimum_qualified'] and value['source_ir_qualified']
    assert value['original_displacement_point_count']==24
    assert value['cartesian_dipole_derivatives'].shape==(2,3,6)
    assert value['cartesian_modes'].shape==(6,1)
    assert value['source_center_energy_hartree']==-1.


def test_ir_failure_does_not_disqualify_verified_minimum():
    value=source.authenticate_case(fixture(ir_ready=False),CASE)
    assert value['source_minimum_qualified']
    assert not value['source_ir_qualified']


def test_actual_unqualified_point_overrules_positive_upstream_flag():
    value=source.authenticate_case(fixture(point_qualified=False),CASE)
    assert not value['source_minimum_qualified']
    assert not value['source_ir_qualified']


def test_missing_or_changed_raw_array_is_rejected():
    reader=fixture();path=CASE+'/finite-differences/h0/c000-plus/gradient.npy'
    reader.blobs[path]+=b'corruption'
    with pytest.raises(ValueError,match='hash mismatch'):
        source.authenticate_case(reader,CASE)


def test_hash_valid_wrong_geometry_is_rejected():
    reader=fixture();path=CASE+'/finite-differences/h0/c000-plus/result.json'
    point=json.loads(reader.blobs[path]);point['coordinates_bohr'][0][0]+=0.01
    reader.blobs[path]=encode(point);reseal(reader)
    with pytest.raises(ValueError,match='geometry differs'):
        source.authenticate_case(reader,CASE)


def test_hash_valid_json_npy_disagreement_is_rejected():
    reader=fixture();path=CASE+'/finite-differences/h0/c000-plus/result.json'
    point=json.loads(reader.blobs[path]);point['gradient'][0][0]+=0.01
    reader.blobs[path]=encode(point);reseal(reader)
    with pytest.raises(ValueError,match='JSON/NPY'):
        source.authenticate_case(reader,CASE)


def test_hash_valid_duplicate_fingerprint_is_rejected():
    reader=fixture();base=CASE+'/finite-differences/h0/c000-'
    plus=json.loads(reader.blobs[base+'plus/result.json'])
    minus=json.loads(reader.blobs[base+'minus/result.json'])
    minus['input_fingerprint']=plus['input_fingerprint']
    reader.blobs[base+'minus/result.json']=encode(minus);reseal(reader)
    with pytest.raises(ValueError,match='Duplicate'):
        source.authenticate_case(reader,CASE)


def test_timestamp_and_progress_commit_do_not_duplicate_scientific_job(tmp_path):
    a=source.import_case(fixture(),CASE,tmp_path,'known-branch','a'*40)
    b=source.import_case(fixture(),CASE,tmp_path,'known-branch','b'*40)
    assert a['id']==b['id'] and a['snapshot_path']==b['snapshot_path']
    assert b['source']['commit']=='a'*40


def test_snapshot_manifest_tampering_is_detected(tmp_path):
    descriptor=source.import_case(fixture(),CASE,tmp_path,'known-branch','a'*40)
    path=Path(descriptor['snapshot_path'])/'source.json'
    value=json.loads(path.read_text());value['source_ir_qualified']=False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='manifest changed'):
        source.load_source(descriptor)


def test_absolute_paths_cannot_escape_report_case():
    reader=fixture();case=json.loads(reader.blobs[CASE+'/result.json'])
    case['publication_gate']['authenticated_artifact_sha256']['/etc/passwd']='a'*64
    reader.blobs[CASE+'/modes/publication_quality.json']=encode(case['publication_gate'])
    reader.blobs[CASE+'/result.json']=encode(case)
    reader.blobs[CASE+'/COMPLETE.json']=encode({'result_sha256':source._sha(reader.blobs[CASE+'/result.json'])})
    with pytest.raises(ValueError,match='outside reports'):
        source.authenticate_case(reader,CASE)


def test_frequency_adapter_uses_fresh_audit_and_preserves_raw_seal(monkeypatch,tmp_path):
    reader=fixture()
    original_case=json.loads(reader.blobs[CASE+'/result.json'])
    template_gate=copy.deepcopy(original_case['publication_gate'])
    center_path=source._relative(original_case['center_point_path'])
    center=json.loads(reader.blobs[center_path])
    center['gradient_path']='/workspace/research4/'+center_path.rsplit('/',1)[0]+'/gradient.npy'
    center['settings']['grid_level']=5
    # All original displacement settings must match the center model exactly.
    for path,blob in list(reader.blobs.items()):
        if '/finite-differences/' in path and path.endswith('/result.json'):
            point=json.loads(blob);point['settings']=center['settings']
            reader.blobs[path]=encode(point)
    reader.blobs[center_path]=encode(center)
    optimization_path=CASE+'/grid5-optimize/result.json'
    reader.blobs[optimization_path]=encode({'final_point':center})
    legacy_case={'molecule':'acid','analysis':original_case['analysis'],
        'optimization_path':'/workspace/research4/'+optimization_path,
        'center_quality':{'qualified':False},
        'electronic_stability':{'status':'DONE','tested_electronic_stability_passed':True}}
    reader.blobs[CASE+'/result.json']=encode(legacy_case)
    raw_result=reader.blobs[CASE+'/result.json']
    reader.blobs[CASE+'/COMPLETE.json']=encode({'result_sha256':source._sha(raw_result)})
    raw_seal=reader.blobs[CASE+'/COMPLETE.json']
    reader.objects.update({optimization_path:'fake-oid'})
    audit_calls=[]
    def fresh_audit(modes_dir,fd,actual_center,*,electronic_stability_passed):
        audit_calls.append((modes_dir,fd))
        assert electronic_stability_passed is True
        assert actual_center==center
        assert (fd/'h0/c000-plus/result.json').read_bytes()==reader.blobs[CASE+'/finite-differences/h0/c000-plus/result.json']
        gate=copy.deepcopy(template_gate)
        gate['center_point_sha256']=source._sha(source._canonical(center))
        gate['authenticated_artifact_sha256']={str(modes_dir.parent/source._relative(path)[len(CASE)+1:]):source._sha(reader.blobs[source._relative(path)])
            for path in template_gate['authenticated_artifact_sha256']}
        return gate
    monkeypatch.setitem(sys.modules,'pyscf_publication_quality_20261111',
        types.SimpleNamespace(audit_publication=fresh_audit,__file__=source.__file__))
    descriptor=source.import_case(reader,CASE,tmp_path,'frequency-branch','a'*40,frequency=True)
    assert len(audit_calls)==1
    assert descriptor['source_minimum_qualified'] is True
    assert descriptor['source']['upstream_historical_qualification_flags_used'] is False
    evidence=Path(descriptor['snapshot_path'])/'evidence'
    assert (evidence/CASE/'result.json').read_bytes()==raw_result
    assert (evidence/CASE/'COMPLETE.json').read_bytes()==raw_seal
    assert (evidence/CASE/'consumer-generated/publication_quality.json').exists()


def preview_fixture():
    reader=fixture()
    case=json.loads(reader.blobs[CASE+'/result.json'])
    center_path=source._relative(case['center_point_path'])
    point=json.loads(reader.blobs[center_path])
    point['gradient_path']='/workspace/research4/'+center_path.rsplit('/',1)[0]+'/gradient.npy'
    point['settings']['grid_level']=5
    point['array_refs']={}
    for key,filename in [('coordinates_bohr','coordinates_bohr.npy'),('gradient','gradient.npy'),('dipole_au','dipole_au.npy')]:
        path=center_path.rsplit('/',1)[0]+'/'+filename
        reader.blobs[path]=npy(point[key])
        point['array_refs'][key]={'path':'/workspace/research4/'+path,'sha256':source._sha(reader.blobs[path])}
    reader.blobs[center_path]=encode(point)
    optimization_path=CASE+'/optimize/result.json'
    reader.blobs[optimization_path]=encode({'status':'DONE','optimizer_converged':True,
        'minimum_stationarity_passed':True,'final_point':point,'coordinates_bohr':point['coordinates_bohr']})
    return reader,optimization_path


def test_stationary_preview_never_grants_minimum_or_ir_qualification():
    reader,path=preview_fixture()
    record=source.authenticate_preview(reader,path,'acid')
    assert record['status']=='PRECURSOR_ONLY'
    assert record['source_minimum_qualified'] is False
    assert record['source_ir_qualified'] is False
    assert record['source_center_energy_hartree']==-1.


def test_unfinished_optimization_cannot_supply_preview():
    reader,path=preview_fixture()
    value=json.loads(reader.blobs[path]);value['optimizer_converged']=False
    reader.blobs[path]=encode(value)
    with pytest.raises(ValueError,match='not a completed stationary'):
        source.authenticate_preview(reader,path,'acid')


def test_preview_checks_actual_stationarity_and_array_seals():
    reader,path=preview_fixture()
    point_path=CASE+'/optimize/final-center/result.json'
    raw_path=point_path.rsplit('/',1)[0]+'/dipole_au.npy'
    reader.blobs[raw_path]=npy([1.,2.,3.])
    with pytest.raises(ValueError,match='seal mismatch'):
        source.authenticate_preview(reader,path,'acid')
