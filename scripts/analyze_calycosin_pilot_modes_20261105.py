"""NOT_RUN Calycosin-only closed-result NumPy replay; never a quantum run.

The old Spinosin analyzer entrypoint is NOT used or declared compatible. Only its
frozen generic NumPy TR/mode-audit helper is reused after this new wrapper has
verified the new Calycosin source, actual V6b closure and33/93 identity contract.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'scripts/pyscf_calycosin_optfreq_pilot_20261105.py'
WORKER_SHA='02491114cf2d7c8a7bd027a85db29c53964c5205297816e25708a621570e9963'
GUARD=ROOT/'scripts/run_pyscf_continuation_budget_20261105_v6b.py'
GUARD_SHA='b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5'
MATH_HELPER=ROOT/'scripts/analyze_spinosin_v5_modes_20261104.py'
MATH_HELPER_SHA='a2e3588c6dea121be58e4a69afb137bc6e5092199ca4180937fcdb34f6f57154'
REPLAY=ROOT/'reports/pyscf-spinosin-resumable-20261104-v5/real-restart-validation-independent-audit/replay_restarts_v5.py'
REPLAY_SHA='49a12e894e0a1afa48ebffbb80391dc4f414a8168a4368f8b807b9769ec46ad5'
RESULT_SCHEMA='research4-calycosin-optfreq-pilot-result-v1'
SCHEMA='research4-calycosin-independent-numpy-modes-analysis-v1'
MASS={'H':1.008,'C':12.011,'O':15.999}
CM1_TO_THZ=.0299792458
BOHR_TO_ANGSTROM=.52917721092


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def frozen(path,expected,name):
    require(sha(path)==expected,'Frozen analysis dependency changed')
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def result_header(result):
    require(result.get('schema')==RESULT_SCHEMA and result.get('status')=='FREQUENCIES_COMPUTED'
        and result.get('identity')=='Calycosin_CID5280448_user_supplied_conformer_001'
        and result.get('atoms')==33 and result.get('electrons')==148 and result.get('basis_functions')==354
        and result.get('normal_mode_count')==result.get('expected_normal_mode_count')==93,
        'Only the actual Calycosin33/148/354/93 pilot result is accepted')
    for key in ('optimizer_converged','scf_converged','geometry_optimized','final_gradient_thresholds_passed',
                'stable_MF_state_unchanged','complete_global_Hessian','normal_modes_complete',
                'full_system_CPHF_returned_normally','mode_geometry_and_Hessian_same_final_SCF'):
        require(result.get(key) is True,'Incomplete scientific source: '+key)
    require(result.get('IR_intensities_status')=='NOT_COMPUTED'
        and result.get('Hessian_method')=='ORIGINAL_STOCK_FULL_RKS_KERNEL_NO_INCREMENTAL_OR_STAGE_CACHE',
        'Unsupported Hessian/IR scope')
    settings=result.get('settings',{})
    require(settings.get('xc')=='PBE' and settings.get('basis')=='def2-svp'
        and settings.get('grid_level')==1 and settings.get('density_fitting') is False
        and settings.get('dispersion')=='NONE_EXPLICIT_PBE_PILOT_NOT_D3'
        and settings.get('threads') in (1,2) and 1<=settings.get('optimizer_maxsteps',0)<=35,
        'Pilot method/resources differ')


def qualify(result_path,expected_sha):
    """Only stdlib and closed JSON; executed before NumPy or any array read."""
    worker=frozen(WORKER,WORKER_SHA,'calycosin_closed_pilot_metadata')
    guard=frozen(GUARD,GUARD_SHA,'calycosin_closed_v6b_parent')
    path=Path(result_path)
    result=worker.read_json(path,expected_sha);path=path.resolve()
    result_header(result)
    require(result.get('output_directory')==str(path.parent),'Producer output path differs')
    closure=guard.verify_closed_phase({'source_worker':result['source_worker'],
        'resource_supervision':result['resource_supervision'],'output_directory':str(path.parent)},
        expected_worker_path=WORKER,expected_worker_sha256=WORKER_SHA,
        expected_result_schema=RESULT_SCHEMA,allowed_result_statuses=('FREQUENCIES_COMPUTED',),
        expected_output=path.parent)
    require(closure.get('producer_result_sha256')==expected_sha,'Actual closed producer result SHA differs')
    identity=worker.identity_metadata()
    require(result.get('chemical_identity')==identity,'Exact original input and later identity overlay changed')
    decision=result['root_post_spinosin_decision']['authorization']
    args=SimpleNamespace(authorization=Path(decision['path']),authorization_sha256=decision['sha256'],
        output=path.parent,threads=result['settings']['threads'],maxsteps=result['settings']['optimizer_maxsteps'])
    worker.authorize_after_spinosin(args,result['resource_supervision'],guard)
    records=result.get('arrays',{})
    expected={'hessian_hartree_per_bohr2.npy':(33,33,3,3),'hessian_used_for_analysis.npy':(33,33,3,3),
        'normal_modes.npy':(93,33,3),'frequencies_cm1.npy':(93,),
        'optimized_coordinates_angstrom.npy':(33,3),'normal_mode_geometry_bohr.npy':(33,3),
        'masses_amu.npy':(33,),'gradient_hartree_per_bohr.npy':(33,3)}
    for name in expected:
        record=records.get(name,{})
        require(record.get('path')==str(path.parent/name) and len(record.get('sha256',''))==64
            and type(record.get('bytes')) is int and 0<record['bytes']<2*1024**2,
            'Bound closed array artifact metadata missing: '+name)
    require(sha(path)==expected_sha,'Closed source result changed during metadata qualification')
    return worker,guard,result,identity,closure,expected


@contextmanager
def exclusive_closed_quantum():
    path=ROOT/'.runtime/pyscf-v4-global.lock'
    with path.open('a') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def analyze(result_path,expected_sha,output):
    with exclusive_closed_quantum():
        worker,guard,result,identity,closure,shapes=qualify(result_path,expected_sha)
        guard.validate_ledger(worker.read_json(result['resource_supervision']['proof']['ledger_path']))
        require(not any(k=='pyscf' or k.startswith('pyscf.') for k in sys.modules),'NumPy replay must not import a quantum engine')
        output=Path(output).resolve()
        require(output.is_relative_to(ROOT/'reports') and not output.exists(),'Fresh report directory required')
        # The actual closed parent and method/input contract are now qualified.
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
        os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
        resource.setrlimit(resource.RLIMIT_CPU,(120,121))
        def stop(_sig,_frame):raise TimeoutError('Finite120second Calycosin analysis bound')
        old=signal.signal(signal.SIGALRM,stop);signal.alarm(120)
        try:
            worker.install_sites()
            import numpy as np
            math_helper=frozen(MATH_HELPER,MATH_HELPER_SHA,'generic_numpy_TR_math_only_not_old_analyzer_entrypoint')
            require(math_helper.MASS==MASS and math_helper.CM1_TO_THZ==CM1_TO_THZ,'Validated mass/unit convention changed')
            require(math_helper.REPLAY==REPLAY and math_helper.REPLAY_SHA==REPLAY_SHA,'Independent projection source differs')
            arrays={}
            for name,shape in shapes.items():
                record=result['arrays'][name];path=Path(record['path'])
                require(not path.is_symlink() and path.stat().st_size==record['bytes']
                    and sha(path)==record['sha256'],'Closed array bytes changed: '+name)
                value=np.load(path,allow_pickle=False)
                require(value.dtype==np.float64 and value.shape==shape and np.isfinite(value).all(),'Array dimension/type/finite check failed')
                arrays[name]=value
            cm=arrays['frequencies_cm1.npy'];symbols=identity['input_document']['atom_symbols']
            require(np.array_equal(arrays['masses_amu.npy'],np.asarray([MASS[s] for s in symbols])),
                'Recorded masses differ from independent isotope-average convention')
            require(np.allclose(arrays['normal_mode_geometry_bohr.npy']*BOHR_TO_ANGSTROM,
                arrays['optimized_coordinates_angstrom.npy'],atol=1e-11,rtol=0),'Two saved geometry units differ')
            require(np.array_equal(cm,np.asarray(result['frequencies_cm1'],dtype=np.float64))
                and np.allclose(cm*CM1_TO_THZ,result['frequencies_thz'],atol=1e-12,rtol=0),'JSON frequencies differ from producer-hashed NPY')
            replay=math_helper.numpy_mode_audit(arrays['hessian_hartree_per_bohr2.npy'],
                arrays['hessian_used_for_analysis.npy'],arrays['normal_modes.npy'],cm,
                arrays['normal_mode_geometry_bohr.npy'],symbols)
            gradient=arrays['gradient_hartree_per_bohr.npy'];lengths=np.linalg.norm(gradient,axis=1)
            grms=float(np.sqrt(np.mean(lengths**2)));gmax=float(lengths.max())
            require(grms<=3e-4 and gmax<=4.5e-4,'Independent saved fresh gradient fails stationarity')
            positive=bool(np.all(cm>0));imaginary=int(np.sum(cm<0));zero=int(np.sum(cm==0))
            require(result['imaginary_frequency_count']==imaginary and result['all_frequencies_real_positive']==positive
                and result['stationary_minimum_verified']==positive and result['frequency_validation_passed']==positive,
                'Minimum flags disagree with the actual signed modes')
            require(sha(result_path)==expected_sha and all(sha(result['arrays'][name]['path'])==result['arrays'][name]['sha256'] for name in shapes),
                'Closed input changed during independent replay')
            require(not any(k=='pyscf' or k.startswith('pyscf.') for k in sys.modules),'Independent replay imported a quantum engine')
            output.mkdir(parents=True)
            fields=['mode','frequency_cm1','frequency_THz','imaginary','qualified_positive_minimum','IR_intensity_status']
            rows=[dict(zip(fields,[i,float(value),float(value*CM1_TO_THZ),bool(value<0),positive,'NOT_COMPUTED'])) for i,value in enumerate(cm,1)]
            for name,chosen in [('all93_signed_frequencies.csv',rows),('normal_modes_0p1_to_1p5_THz.csv',[r for r in rows if .1<=r['frequency_THz']<=1.5])]:
                with (output/name).open('x',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(chosen)
            checks={'schema':SCHEMA,'status':'DONE_ACTUAL_CALYCOSIN93_NUMPY_REPLAY',
                'created_at_utc':datetime.now(timezone.utc).isoformat(),'case':'Calycosin','atoms':33,'normal_modes':93,
                'source_result':{'path':str(Path(result_path).resolve()),'sha256':expected_sha},'producer_closure':closure,
                'analysis_source_sha256':sha(__file__),'worker_source_sha256':WORKER_SHA,'guard_source_sha256':GUARD_SHA,
                'generic_math_helper_source_sha256':MATH_HELPER_SHA,'independent_projector_source_sha256':REPLAY_SHA,
                'old_Spinosin_analyzer_entrypoint_used':False,'old_analyzer_claimed_to_accept_Calycosin':False,
                'identity':{k:identity[k] for k in ('name','pubchem_cid','chebi_id','full_inchikey','formula')},
                'independent_numpy_replay':replay,'fresh_gradient_atom_RMS_Eh_per_Bohr':grms,'fresh_gradient_atom_max_Eh_per_Bohr':gmax,
                'imaginary_frequency_count':imaginary,'zero_frequency_count':zero,'qualified_positive_minimum':positive,
                'minimum_qualification_separate_from_frequency_completion':True,
                'mechanism_pairing_status':'NO_EXPERIMENTAL_PAIRING_PERFORMED' if positive else 'BLOCKED_NONPOSITIVE_MINIMUM',
                'global_conformer_search_performed':False,'chemical_assignment_verified':False,
                'IR_intensities_status':'NOT_COMPUTED','theory_enters_classifier':False,
                'new_quantum_evaluations':0,'PySCF_imported':False,'numpy_version':np.__version__}
            worker.save_json(output/'checks.json',checks,exclusive=True)
            (output/'README.md').write_text('Calycosin:93 actual signed modes numerically replayed from a healthy closed pilot. '
                'Only one user-supplied conformer was optimized; this is not a global minimum search. '
                'Negative/zero modes remain in the complete table and block minimum qualification. '
                'IR NOT_COMPUTED. Isolated-molecule harmonic modes are not the solid herbal THz spectrum. '
                'No chemical peak assignment or classifier change is claimed.\n')
            worker.save_json(output/'artifact_index.json',{'schema':SCHEMA+'-index','files':[
                worker.artifact(p) for p in sorted(output.iterdir()) if p.is_file()]},exclusive=True)
            return checks
        finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)


def preflight(output):
    output=Path(output).resolve()
    require(output.is_relative_to(ROOT/'reports') and not output.exists(),'Fresh preparation directory required')
    for path,expected in ((WORKER,WORKER_SHA),(GUARD,GUARD_SHA),(MATH_HELPER,MATH_HELPER_SHA),(REPLAY,REPLAY_SHA)):
        require(sha(path)==expected,'Unfrozen Calycosin analysis dependency')
    output.mkdir(parents=True)
    value={'schema':SCHEMA,'status':'NOT_RUN','actual_Calycosin_Opt_Freq_or_analysis_run':False,
        'new_quantum_evaluations':0,'numerical_imports':False,'source_worker_sha256':WORKER_SHA,
        'guard_source_sha256':GUARD_SHA,'expected_modes_NOT_ACTUAL_RESULT':93,
        'old_analyzer_entrypoint_used':False,'actual_source_and_V6b_closure_required_before_numpy':True}
    with (output/'preflight.json').open('x') as handle:json.dump(value,handle,indent=2);handle.write('\n')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--result',type=Path);p.add_argument('--result-sha256');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.execute:require(args.result and args.result_sha256,'Actual closed result path/SHA required')
    value=analyze(args.result,args.result_sha256,args.output) if args.execute else preflight(args.output)
    print(json.dumps({k:value.get(k) for k in ('status','case','normal_modes','qualified_positive_minimum')}),flush=True)


if __name__=='__main__':main()
