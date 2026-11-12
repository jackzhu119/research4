#!/usr/bin/env python3
"""Resource-only adapter around the immutable distinct-conformer worker.

The scientific profiles, grids, response terms and acceptance thresholds are
unchanged. No scientific calculation starts on import. The external resource
supervisor must serialize the large-molecule workers and owns all CPU limits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

PUB_COMMIT='2eb6b6e1dfe6627f24c8f97ea268bdec8f636bcb'
EXPECTED_WORKER_SHA='88cb93b7854e645601ba390616067a7631619a156b873cf818dc3356e528d745'
EXPECTED_CORE_SHA='69a178f2d2a0311698af72724221c13deaefb65ac8b192b8f63c5491c602029d'
EXPECTED_BECKE_SOURCE_SHA='9f20d3a392a8c23499b643da6f0bf72e73d33185802b0d87d22c497c8616efc6'
EXPECTED_RKS_MODULE_SHA='cffc3e9efa10cf1b547d757e9d165c1000377e9159c794f87464a79c8c93ab46'
CHECKPOINT_CATALOG={
 '2357916c3000fba5d9e9f162759ac5443c92785f035b3fdfd8e9f3fc03b2a7e3':{
  'job_id':'spinosin_C002_df_grid5',
  'point_relative':'jobs/spinosin_C002_df_grid5/science/df-grid5/round000000/optimize/steps/step-00000/2357916c3000fba5d9e9f162759ac5443c92785f035b3fdfd8e9f3fc03b2a7e3',
  'input_sha256':'8c563f0448efc67b9cffc4a58e1ad080edb5075bc37081c9eda1dcc75ff780f3',
  'audit_sha256':'04d62f4d5c7ddeab31da7613e7192785919650e993d3b9fea9dd844ca7f25e5c',
  'checkpoint_sha256':'7a0202df40c6fccb6ed909e5699801429e637e745a5673d5f66ad7c34fe05317'},
 'f19cd1f79fd09894c7d24dddd642ebcca7342dd9b703028ecd31f1e11d28d8ee':{
  'job_id':'spinosin_C003_df_grid5',
  'point_relative':'jobs/spinosin_C003_df_grid5/science/df-grid5/round000000/optimize/steps/step-00000/f19cd1f79fd09894c7d24dddd642ebcca7342dd9b703028ecd31f1e11d28d8ee',
  'input_sha256':'3d8f5d40d788f35eafcae9f54e30fc950d6e475f0fc01978a8848131aae70291',
  'audit_sha256':'d838d6b825c0574e537d58db4dc8adec3531bd18bf9370d8f832826f6c22fe3c',
  'checkpoint_sha256':'8b07c5f2e90aa9a201170abb31a8be443ae366f9c0341e6960b73a9bd0887dd8'},
}


class AdapterIntegrityError(RuntimeError):
    def __init__(self,message): super().__init__('Integrity error: '+message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def patched_becke_function(module):
    """Delete only dead intermediates; all original arithmetic is unchanged."""
    original=module.grids_response_becke
    source=inspect.getsource(original)
    source_sha=hashlib.sha256(source.encode()).hexdigest()
    if source_sha!=EXPECTED_BECKE_SOURCE_SHA:
        raise AdapterIntegrityError('Unexpected PySCF Becke generator source')
    module_file=Path(inspect.getsourcefile(original))
    if sha(module_file)!=EXPECTED_RKS_MODULE_SHA:
        raise AdapterIntegrityError('Unexpected PySCF RKS gradient module bytes')
    marker='        yield coords, w0, w1\n'
    if source.count(marker)!=1:
        raise AdapterIntegrityError('Becke lifecycle patch anchor is not unique')
    modified=source.replace(marker,'        del pbecke, dpbecke, z\n'+marker)
    namespace=dict(module.__dict__)
    exec(compile(modified,'<resource-only-becke-lifecycle-v1>','exec'),namespace)
    patched=namespace['grids_response_becke']
    proof={'schema':'becke-intermediate-lifecycle-v1','pyscf_version_required':'2.13.0',
        'source_function_sha256':source_sha,'source_module_sha256':sha(module_file),
        'patched_function_source_sha256':hashlib.sha256(modified.encode()).hexdigest(),
        'only_change':'del pbecke, dpbecke, z immediately before yield coords,w0,w1',
        'formulas_grid_weights_response_and_arithmetic_order_changed':False,
        'point_chunking':False,'worker_concurrency_required':1}
    return original,patched,proof


def install_lifecycle_patch():
    import pyscf
    from pyscf.grad import rks
    if pyscf.__version__!='2.13.0': raise AdapterIntegrityError('PySCF 2.13.0 required')
    existing=getattr(rks,'_resource_lifecycle_proof',None)
    if existing is not None: return dict(existing)
    original,patched,proof=patched_becke_function(rks)
    rks.grids_response_becke=patched
    rks._resource_lifecycle_original=original
    rks._resource_lifecycle_proof=proof
    return dict(proof)


def assert_exact_point_binding(payload,symbols,coordinates,settings,fingerprint,source_sha,pyscf_version):
    """Pure identity checks, also used by no-fit rejection tests."""
    import numpy as np
    if (payload.get('input_fingerprint')!=fingerprint or payload.get('source_sha256')!=source_sha
            or payload.get('pyscf_version')!=pyscf_version or payload.get('operation')!='point'
            or payload.get('settings')!=settings or payload.get('symbols')!=list(symbols)):
        raise AdapterIntegrityError('Checkpoint point/model identity differs')
    expected=np.asarray(payload.get('coordinates_bohr'),dtype=float)
    actual=np.asarray(coordinates,dtype=float)
    if expected.shape!=actual.shape or actual.ndim!=2 or actual.shape[1]!=3 or not np.isfinite(actual).all() or not np.array_equal(expected,actual):
        raise AdapterIntegrityError('Checkpoint geometry differs, including atom ordering')
    unsigned={k:v for k,v in payload.items() if k!='input_fingerprint'}
    if digest_json(unsigned)!=fingerprint:
        raise AdapterIntegrityError('Stored original point fingerprint does not match its payload')


def load_bound_checkpoint(root,core,symbols,coordinates,out,grid_level,dm0):
    """Only the two reviewed, qualified attempt-002 states may be restored."""
    import h5py
    import numpy as np
    import pyscf
    from pyscf import lib
    out=Path(out).resolve();point=out.parent
    catalog=CHECKPOINT_CATALOG.get(point.name)
    if catalog is None: return None
    if point!=(root/catalog['point_relative']).resolve():
        raise AdapterIntegrityError('Checkpoint path is not its authorized job/point')
    payload_path=point/'input.json'
    prior=point/'attempt-002';original_checkpoint=prior/'.local/rolling.chk';audit_path=prior/'electronic-attempts.json'
    checkpoint=original_checkpoint
    if not checkpoint.exists():
        # The immutable original worker may clean rolling.chk after a qualified
        # point. The separately authenticated content-addressed archive persists.
        checkpoint=root/'recovery_checkpoints'/(catalog['checkpoint_sha256']+'.chk')
    for path,expected in ((payload_path,catalog['input_sha256']),(checkpoint,catalog['checkpoint_sha256']),(audit_path,catalog['audit_sha256'])):
        if not path.is_file() or sha(path)!=expected:
            raise AdapterIntegrityError('Authorized checkpoint evidence SHA changed: '+str(path))
    payload=read(payload_path)
    fingerprint,new_payload=core._fingerprint(list(symbols),np.asarray(coordinates,dtype=float),grid_level,dm0)
    assert_exact_point_binding(payload,symbols,coordinates,core.settings(grid_level),fingerprint,sha(core.__file__),pyscf.__version__)
    if fingerprint!=point.name or new_payload!={k:v for k,v in payload.items() if k!='input_fingerprint'}:
        raise AdapterIntegrityError('Live scientific point is not the authorized checkpoint point')
    calls=read(audit_path)
    last=calls[-1] if isinstance(calls,list) and calls else {}
    audit=last.get('audit',{})
    if (last.get('solver')!='TIGHT_NATIVE_POLISH' or last.get('native_converged') is not True
            or audit.get('native_converged') is not True or audit.get('qualified') is not True
            or audit.get('quality_warnings')!=[]):
        raise AdapterIntegrityError('Checkpoint has no qualified native-polish provenance')
    # The prior strict audit must independently meet current numerical thresholds.
    if (abs(audit['fresh_energy_difference_hartree'])>core.SCF_SETTINGS['fresh_energy_difference_acceptance_hartree']
            or audit['fresh_orbital_gradient_norm']>core.SCF_SETTINGS['fresh_orbital_gradient_acceptance']
            or abs(audit['electron_count_error'])>core.SCF_SETTINGS['electron_count_acceptance']):
        raise AdapterIntegrityError('Prior electronic audit no longer meets the unchanged thresholds')
    mol=lib.chkfile.load_mol(str(checkpoint))
    if (list(mol.elements)!=list(symbols) or not np.array_equal(mol.atom_coords(),np.asarray(coordinates))
            or mol.basis!=core.SCF_SETTINGS['basis'] or mol.charge!=core.SCF_SETTINGS['charge']
            or mol.spin!=core.SCF_SETTINGS['spin']):
        raise AdapterIntegrityError('HDF molecular geometry/basis/charge/spin mismatch')
    with h5py.File(checkpoint,'r') as stream:
        state={name:stream['scf/'+name][()] for name in ('mo_coeff','mo_occ','mo_energy','e_tot')}
    nao=mol.nao_nr()
    c,o,e=np.asarray(state['mo_coeff']),np.asarray(state['mo_occ']),np.asarray(state['mo_energy'])
    if (c.shape!=(nao,nao) or o.shape!=(nao,) or e.shape!=(nao,)
            or any(np.iscomplexobj(x) or not np.isfinite(x).all() for x in (c,o,e))
            or not np.isfinite(state['e_tot']) or abs(float(o.sum())-mol.nelectron)>1e-12
            or not np.all((o==0)|(o==2))):
        raise AdapterIntegrityError('Checkpoint orbital state is incomplete or not real closed-shell RKS')
    if abs(float(state['e_tot'])-audit['native_energy_hartree'])>1e-12:
        raise AdapterIntegrityError('Checkpoint energy differs from its qualified native audit')
    proof={'schema':'exact-point-native-scf-restore-v1','checkpoint_path':str(checkpoint),
        'original_checkpoint_path':str(original_checkpoint),
        'checkpoint_sha256':sha(checkpoint),'original_point_input_sha256':sha(payload_path),
        'original_native_audit_path':str(audit_path),'original_native_audit_sha256':sha(audit_path),
        'input_fingerprint':fingerprint,'settings':core.settings(grid_level),
        'prior_native_audit':audit,'atom_geometry_basis_charge_spin_verified':True,
        'native_converged_origin':'SHA-bound prior TIGHT_NATIVE_POLISH; HDF alone is not a convergence certificate',
        'arrays_sha256':{n:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for n,v in state.items()}}
    return state,last,proof


def archive_checkpoint(root,proof):
    """Keep the authenticated orbital state outside old point cleanup paths."""
    source=Path(proof['checkpoint_path'])
    expected=proof['checkpoint_sha256']
    if sha(source)!=expected: raise AdapterIntegrityError('Checkpoint changed before recovery archive')
    directory=root/'recovery_checkpoints';directory.mkdir(parents=True,exist_ok=True)
    destination=directory/(expected+'.chk')
    if destination.exists():
        if sha(destination)!=expected: raise AdapterIntegrityError('Recovery archive SHA mismatch')
    else:
        temporary=destination.with_name(destination.name+'.'+str(os.getpid())+'.tmp')
        try:
            try: os.link(source,temporary)
            except OSError: shutil.copyfile(source,temporary)
            if sha(temporary)!=expected: raise AdapterIntegrityError('Recovery archive copy SHA mismatch')
            with temporary.open('rb') as stream: os.fsync(stream.fileno())
            os.replace(temporary,destination)
        finally:
            if temporary.exists(): temporary.unlink()
    return {'archived_checkpoint_path':str(destination.resolve()),
            'archived_checkpoint_sha256':expected,'archive_excluded_from_remote_wavefunction_publication':True}


def install_electronic_adapter(root,core):
    if getattr(core,'_resource_electronic_adapter_installed',False): return
    original=core._run_electronic
    def run_electronic(symbols,coordinates,out,grid_level,dm0,log):
        import numpy as np
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        lifecycle=install_lifecycle_patch()  # worker already set its disk TMPDIR
        provenance={'schema':'resource-point-provenance-v1','started_at_utc':utc(),
            'adapter_sha256':sha(__file__),'original_worker_sha256':sha(root/'worker.py'),
            'original_core_sha256':sha(core.__file__),'lifecycle_patch':lifecycle,
            'scientific_settings':core.settings(grid_level),'prior_results_modified':False,
            'method_threshold_grid_or_response_changed':False}
        core.atomic_json(out/'resource_provenance.json',provenance)
        try:
            bound=load_bound_checkpoint(root,core,symbols,coordinates,out,grid_level,dm0)
        except Exception as error:
            core.atomic_json(out/'resume_provenance.json',dict(provenance,status='REJECTED_IDENTITY_OR_EVIDENCE',error=str(error)))
            raise
        if bound is None:
            core.atomic_json(out/'resume_provenance.json',dict(provenance,status='NO_AUTHORIZED_EXACT_POINT_CHECKPOINT'))
            return original(symbols,coordinates,out,grid_level,dm0,log)
        state,prior,proof=bound
        proof.update(archive_checkpoint(root,proof))
        proof.update(provenance,status='FRESH_AUDIT_PENDING')
        core.atomic_json(out/'resume_provenance.json',proof)
        mol=core.new_molecule(symbols,coordinates,log)
        mf=core.make_scf(mol,out,grid_level)
        mf.mo_coeff=np.asarray(state['mo_coeff']);mf.mo_occ=np.asarray(state['mo_occ'])
        mf.mo_energy=np.asarray(state['mo_energy']);mf.e_tot=float(state['e_tot'])
        mf.conv_tol=prior['actual_conv_tol_hartree'];mf.conv_tol_grad=prior['actual_conv_tol_grad'];mf.diis_space=12
        # Native convergence is taken only from the exact authenticated native
        # solver record, then independently tested against a newly built Fock.
        mf.converged=True
        seed=np.asarray(mf.make_rdm1())
        try:
            density,audit=core._audit(mf)
            proof.update(fresh_audit=audit,fresh_audit_at_utc=utc())
        except Exception as error:
            audit={'qualified':False,'fresh_audit_error':str(error)}
            proof.update(fresh_audit=audit,fresh_audit_at_utc=utc())
        if audit.get('qualified') is True:
            proof.update(status='EXACT_NATIVE_STATE_RESTORED_AND_FRESH_AUDIT_PASSED')
            core.atomic_json(out/'resume_provenance.json',proof)
            call={'solver':'AUTHENTICATED_NATIVE_CHECKPOINT_FRESH_AUDIT','native_converged':True,
                'actual_conv_tol_hartree':mf.conv_tol,'actual_conv_tol_grad':mf.conv_tol_grad,
                'audit':audit,'checkpoint_sha256':proof['checkpoint_sha256'],
                'resume_provenance_path':str((out/'resume_provenance.json').resolve())}
            return mf,density,audit,[call]
        proof.update(status='FRESH_AUDIT_FAILED_NATIVE_WARM_START_REQUIRED')
        core.atomic_json(out/'resume_provenance.json',proof)
        # A failed fresh check is not relabelled converged. The immutable native
        # recovery loop receives only a verified real density as its new guess.
        restored=original(symbols,coordinates,out,grid_level,seed,log)
        mf,density,final_audit,calls=restored
        proof.update(status='NATIVE_WARM_START_RETURNED_QUALIFIED' if final_audit.get('qualified') else 'NATIVE_WARM_START_RETURNED_UNQUALIFIED',
                     native_warm_start_audit=final_audit,finished_at_utc=utc())
        core.atomic_json(out/'resume_provenance.json',proof)
        entry={'solver':'AUTHENTICATED_CHECKPOINT_DENSITY_WARM_START','fresh_restore_accepted':False,
               'checkpoint_sha256':proof['checkpoint_sha256'],'fresh_audit':audit}
        return mf,density,final_audit,[entry]+calls
    core._run_electronic=run_electronic
    core._resource_electronic_adapter_installed=True


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root',type=Path,required=True)
    args,_=parser.parse_known_args()
    root=args.root.resolve()
    for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[variable]='1'
    worker_path=root/'worker.py';core_path=root/'source'/PUB_COMMIT/'scripts/pyscf_publication_core_20261111.py'
    try:
        if sha(worker_path)!=EXPECTED_WORKER_SHA or sha(core_path)!=EXPECTED_CORE_SHA:
            raise AdapterIntegrityError('Immutable worker/core source SHA differs')
        spec=importlib.util.spec_from_file_location('_immutable_chem_worker_resource',worker_path)
        worker=importlib.util.module_from_spec(spec);sys.modules[spec.name]=worker;spec.loader.exec_module(worker)
        sys.path.insert(0,str(core_path.parent))
        core=importlib.import_module('pyscf_publication_core_20261111')
        install_electronic_adapter(root,core)
        return worker.main()
    except AdapterIntegrityError as error:
        print(json.dumps({'status':'integrity_error','complete':False,'error':str(error)}),file=sys.stderr)
        return 78


if __name__=='__main__':
    raise SystemExit(main())
