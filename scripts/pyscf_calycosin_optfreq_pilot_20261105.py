"""NOT_RUN Calycosin33 pilot, eligible only after the main Spinosin result closes.

Numerical libraries and coordinates are accessed only after the frozen V6b live
parent, a separate explicit root decision, and actual completed Spinosin C/replay
are verified. No budget registration or growth is implemented here. This is a
single supplied conformer, not a global conformer search or a plant assay.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
GUARD=ROOT/'scripts/run_pyscf_continuation_budget_20261105_v6b.py'
GUARD_SHA='b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5'
STATE=ROOT/'scripts/pyscf_molecular_state_worker_20261104_v5.py'
STATE_SHA='be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8'
BASE=ROOT/'scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py'
BASE_SHA='acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d'
V5_NUMERICAL=ROOT/'scripts/pyscf_resumable_hessian_worker_20261104_v5.py'
V5_NUMERICAL_SHA='b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190'
MAIN_WORKER=ROOT/'scripts/pyscf_staged_hessian_worker_20261105_v6b.py'
MAIN_WORKER_SHA='f13701964cd8a5fbda25a19ca27ec91b4a414f6b5cb49c14abacf6c82eaa448d'
MAIN_ANALYZER=ROOT/'scripts/analyze_pyscf_staged_modes_20261105_v6b_v2.py'
MAIN_ANALYZER_SHA='38dde5e185c4fc48d58732982e0164b3a2f5430a26bdfc1076b21103d1d75c6a'
INPUT=ROOT/'reports/pyscf-sdf-continuation-20261103-v1/state-worker-preflight/inputs/cid5280448_optimize_input.json'
INPUT_SHA='842ebe92dad9e7d6d5a56b4c27c515fb10580966e0dd9495bc932fd3bfa76f49'
OVERLAY=ROOT/'reports/external-evidence-20261104-v1/chemistry/uploaded_identity_overlay.json'
OVERLAY_SHA='e00dd59f2aa058bd4a65e326ebc1bb041c3576c9650ff57c661385a12133acc1'
SDF_SHA='1bf68cb55ed29ac48017aa13b967b12922581290e89667efbd69bd42d9ac480f'
XYZ_SHA='8fda598d70ff9eb70085022d42fb4922283e5d54e5d2fabc37e72d2f060f7236'
COORDINATES_SHA='793776316ecf1a6f34b4f80fbd2467a2864d1be7eb5769c92bd96c05ef47b5fa'
INCHIKEY='ZZAJQOPSWWVMBI-UHFFFAOYSA-N'
SCHEMA='research4-calycosin-optfreq-pilot-result-v1'
AUTH_SCHEMA='research4-post-spinosin-calycosin-pilot-authorization-v1'
IDENTITY='Calycosin_CID5280448_user_supplied_conformer_001'
CM1_TO_THZ=.0299792458
BOHR_TO_ANGSTROM=.52917721092
MAX_CPU=7200
MAX_WALL=7200
MAX_STEPS=35


class PilotStop(RuntimeError):
    pass


def require(value,message):
    if not value:raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def root_path(value):
    path=Path(value)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def artifact(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}


def read_json(path,expected=None):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=2*1024**2,'Small closed JSON artifact required')
    if expected:require(sha(path)==expected,'Metadata artifact SHA differs: '+str(path))
    return json.loads(path.read_text())


def recorded_json(record):
    require(isinstance(record,dict) and set(record)=={'path','sha256','bytes'},'Explicit metadata artifact path/SHA/size required')
    path=root_path(record['path'])
    require(path.stat().st_size==record['bytes'],'Recorded metadata size changed')
    return path,read_json(path,record['sha256'])


def save_json(path,value,*,exclusive=False):
    path=Path(path)
    if exclusive:
        with path.open('x') as handle:
            json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')
        return
    temporary=path.with_suffix(path.suffix+'.writing')
    with temporary.open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')
    os.replace(temporary,path)


def frozen(path,expected,name):
    require(sha(path)==expected,'Frozen dependency source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def install_sites():
    """Use existing verified installations only; upgrade site takes precedence."""
    for relative in ('.runtime/quantum-pilot-site','.runtime/quantum-upgrade-site'):
        location=str(ROOT/relative)
        if location not in sys.path:sys.path.insert(0,location)


def identity_metadata():
    """JSON only; deliberately does not read SDF/XYZ/NPY payloads."""
    document=read_json(INPUT,INPUT_SHA);overlay=read_json(OVERLAY,OVERLAY_SHA)
    require(document.get('schema')=='research4-molecular-input-v1','Unsupported original input')
    require(document.get('atom_symbols')==['O']*5+['C']*16+['H']*12,'Exact original atom order required')
    require(document.get('composition')=={'C':16,'H':12,'O':5} and document.get('formula')=='C16H12O5','Calycosin composition differs')
    require((document.get('charge'),document.get('spin_2S'),document.get('multiplicity'),document.get('radical_electrons'),document.get('electron_count'))==(0,0,1,0,148),'Only neutral148-electron singlet supported')
    require(document.get('connectivity_verified') is True and document.get('stereochemistry_checked') is True,'Original graph check missing')
    require(document['input_xyz']['sha256']==XYZ_SHA and document['coordinates_angstrom_npy']['sha256']==COORDINATES_SHA,'Original supplied conformer coordinates changed')
    records=[r for r in overlay.get('records',[]) if r.get('pubchem_cid')==5280448]
    require(len(records)==1,'Unique later neutral Calycosin identity overlay required')
    row=records[0]
    require(row.get('name')=='Calycosin' and row.get('chebi_id')=='17793'
        and row.get('source_inchikey')==row.get('uploaded_sdf_inchikey')==row.get('source_smiles_inchikey')==INCHIKEY
        and row.get('canonical_isomeric_smiles_equal') is True
        and row.get('identity_status')=='PRIMARY_DATABASE_GRAPH_AND_STEREO_MATCH'
        and row.get('uploaded_sdf_sha256')==SDF_SHA
        and row.get('formula')=='C16H12O5' and row.get('explicit_atom_count')==33
        and row.get('charge')==0 and row.get('radical_electrons')==0,'Later primary graph/name verification differs')
    sdf=[r for r in document['source_artifacts'] if r.get('sha256')==SDF_SHA]
    require(len(sdf)==1 and root_path(sdf[0]['path'])==root_path(row['uploaded_sdf_path']),'Source SDF provenance does not match original derived coordinates')
    return {'identity':IDENTITY,'name':'Calycosin','pubchem_cid':5280448,'chebi_id':'17793',
        'full_inchikey':INCHIKEY,'formula':'C16H12O5','atoms':33,'electrons':148,
        'charge':0,'spin_2S':0,'multiplicity':1,'input_manifest':artifact(INPUT),
        'identity_overlay':artifact(OVERLAY),'input_document':document,'source_SDF_record':sdf[0],
        'old_input_name_pending_flag_is_historical':document.get('name_CID_mapping_independently_verified') is False,
        'later_name_graph_evidence_applied_without_overwriting_history':True,
        'coordinate_download_bytes_independently_verified':False,
        'conformer_scope':'ONE_USER_SUPPLIED_CONFORMER_LOCAL_OPTIMIZATION_NOT_GLOBAL_SEARCH'}


def argument_limits(args):
    require(type(args.threads) is int and args.threads in (1,2),'Pilot supports1or2 supervised threads')
    require(type(args.maxsteps) is int and 1<=args.maxsteps<=MAX_STEPS,'Pilot maxsteps must remain1..35')


def authorize_after_spinosin(args,supervision,guard):
    """Only metadata/real parent checks; executes before ANY numerical import."""
    argument_limits(args)
    require(args.authorization and args.authorization_sha256,'Root must explicitly authorize this optional pilot after Spinosin')
    auth=read_json(args.authorization,args.authorization_sha256)
    require(auth.get('schema')==AUTH_SCHEMA and auth.get('enable_execution') is True
        and auth.get('after_spinosin_only') is True and auth.get('no_budget_reset') is True
        and auth.get('worker_sha256')==sha(__file__) and auth.get('supervisor_sha256')==GUARD_SHA,
        'Explicit source-bound post-Spinosin root decision required; prep is not an allocation')
    require(auth.get('input_manifest_sha256')==INPUT_SHA and auth.get('identity_overlay_sha256')==OVERLAY_SHA,'Root decision must bind exact verified Calycosin input')
    require(auth.get('output_directory')==str(args.output.resolve()) and auth.get('selected_threads')==args.threads,'Root decision output/threads differ')
    require(auth.get('optimizer_maxsteps')==args.maxsteps,'Optimizer limit differs from root decision')
    cpu,wall=auth.get('maximum_cpu_seconds'),auth.get('maximum_wall_seconds')
    require(type(cpu) is int and 125<=cpu<=MAX_CPU and type(wall) is int and 61<=wall<=MAX_WALL,'Optional pilot must remain within7200actualCPU/7200wallseconds')
    proof=supervision['proof']
    require(proof['cpu_hard_limit_seconds']==cpu and 61<=proof['wall_limit_seconds']<=wall
        and proof['threads']==args.threads,'Actual phase grant differs or insufficient resources remain')
    require(root_path(auth['existing_ledger_path'])==root_path(proof['ledger_path']),'Use existing cumulative V6b ledger, never a fresh pool')
    main_path,main=recorded_json(auth['completed_spinosin_result'])
    require(main.get('schema')=='research4-staged-full-rks-hessian-result-v6b'
        and main.get('stage')==main.get('phase')=='global' and main.get('status')=='FREQUENCIES_COMPUTED'
        and main.get('case')=='Spinosin' and main.get('atoms')==75 and main.get('normal_mode_count')==219
        and main.get('complete_global_Hessian') is True and main.get('normal_modes_complete') is True,
        'Main Spinosin219 must genuinely finish first; no resource diversion while incomplete')
    require(root_path(main['resource_supervision']['proof']['ledger_path'])==root_path(proof['ledger_path']),
        'Main completion must belong to the same cumulative V6b budget lineage')
    evidence={'source_worker':main['source_worker'],'resource_supervision':main['resource_supervision'],
        'output_directory':str(main_path.parent),'snapshot_manifest_sha256':main['snapshot_manifest_sha256'],'stage':'global'}
    closure=guard.verify_closed_phase(evidence,expected_worker_path=MAIN_WORKER,
        expected_worker_sha256=MAIN_WORKER_SHA,expected_result_schema='research4-staged-full-rks-hessian-result-v6b',
        allowed_result_statuses=('FREQUENCIES_COMPUTED',),expected_output=main_path.parent,
        expected_snapshot_sha256='3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311')
    require(closure.get('producer_result_sha256')==auth['completed_spinosin_result']['sha256'],
        'Actual healthy Spinosin producer result SHA differs from authorized input')
    analysis_path,analysis=recorded_json(auth['completed_spinosin_analysis_checks'])
    require(sha(MAIN_ANALYZER)==MAIN_ANALYZER_SHA and analysis.get('schema')=='research4-V6b-independent-staged-modes-analysis-v1'
        and analysis.get('status')=='DONE_COMPUTED_MODES_AUDITED' and analysis.get('case')=='Spinosin'
        and analysis.get('atoms')==75 and analysis.get('normal_modes')==219
        and analysis.get('normal_mode_count_complete') is True
        and analysis.get('source_result_sha256')==auth['completed_spinosin_result']['sha256']
        and analysis.get('analysis_source_sha256')==MAIN_ANALYZER_SHA
        and analysis.get('actual_V6b_worker_sha256')==MAIN_WORKER_SHA
        and analysis.get('actual_V6b_parent_sha256')==GUARD_SHA
        and analysis.get('independent_numpy_replay',{}).get('status')=='PASS_INDEPENDENT_NUMPY_TR_REPLAY'
        and analysis.get('new_quantum_evaluations')==0,'Actual Spinosin independent replay must complete before this optional pilot')
    decision=datetime.fromisoformat(auth.get('created_at_utc',''))
    require(decision.tzinfo is not None and decision<=datetime.now(timezone.utc),'Actual timezone-aware root decision timestamp required')
    main_ledger=read_json(root_path(main['resource_supervision']['proof']['ledger_path']))
    main_rows=[r for r in main_ledger['phases'] if r['phase']==main['resource_supervision']['proof']['phase']]
    require(len(main_rows)==1 and decision>=datetime.fromisoformat(main_rows[0]['ended_at_utc'])
        and decision>=datetime.fromisoformat(analysis['created_at_utc']),
        'Root decision must actually follow main producer closure and independent analysis')
    require(sha(main_path)==auth['completed_spinosin_result']['sha256']
        and sha(analysis_path)==auth['completed_spinosin_analysis_checks']['sha256']
        and sha(args.authorization)==args.authorization_sha256,'Post-main authorization/input metadata changed during qualification')
    return {'authorization':artifact(args.authorization),'decision':auth,'main_result':artifact(main_path),
        'main_analysis':artifact(analysis_path),'main_closed_producer':closure,
        'main_minimum_qualification_separate_from_computation':analysis.get('qualified_for_mechanism'),
        'existing_budget_only_no_registration':True}


def settings_document(threads,maxsteps):
    return {'xc':'PBE','basis':'def2-svp','grid_level':1,'density_fitting':False,
        'dispersion':'NONE_EXPLICIT_PBE_PILOT_NOT_D3','direct_scf':True,'direct_scf_tol':1e-13,
        'scf_conv_tol_hartree':1e-8,'scf_max_cycle':80,'hessian_conv_tol_cpscf':1e-8,
        'max_memory_mb':8000,'hessian_max_memory_mb':3000,'hessian_max_cycle':50,'hessian_level_shift':0,
        'threads':threads,'native_direct_driver_threads':threads,'charge':0,'spin_2S':0,'multiplicity':1,
        'optimizer_maxsteps':maxsteps,'gradient_grid_response':False,
        'optimizer_thresholds':{'convergence_energy':1e-6,'convergence_grms':3e-4,'convergence_gmax':4.5e-4,
            'convergence_drms':1.2e-3,'convergence_dmax':1.8e-3},
        'scope':'SINGLE_SUPPLIED_CONFORMER_PLAIN_PBE_LOCAL_OPT_FREQ_PILOT'}


def execute(args):
    guard=frozen(GUARD,GUARD_SHA,'calycosin_v6b_live_parent')
    supervision=guard.verify_resource_supervision(expected_threads=args.threads)
    decision=authorize_after_spinosin(args,supervision,guard)
    identity=identity_metadata()
    output=args.output.resolve()
    require(output.is_relative_to(ROOT/'reports') and not output.exists(),'Fresh report directory required')
    output.mkdir(parents=True)
    started_wall,started_cpu=time.perf_counter(),time.process_time()
    state={'schema':SCHEMA,'phase':'calycosin-optfreq','status':'PARTIAL','started_at_utc':utc(),
        'identity':IDENTITY,'chemical_identity':identity,'source_worker':artifact(__file__),
        'output_directory':str(output),'resource_supervision':supervision,'root_post_spinosin_decision':decision,
        'settings':settings_document(args.threads,args.maxsteps),'optimizer_converged':False,
        'scf_converged':False,'geometry_optimized':False,'final_gradient_thresholds_passed':False,
        'stable_MF_state_unchanged':False,'frequencies_status':'NOT_RUN','IR_intensities_status':'NOT_COMPUTED',
        'isolated_molecule_not_solid_herbal_spectrum':True,'global_minimum_search_performed':False,
        'chemical_peak_assignment_verified':False,'theory_enters_classifier':False}
    save_json(output/'input.json',state,exclusive=True)
    log=(output/'pyscf.log').open('x',buffering=1)
    native=None
    def emit(event,details=None,**extra):
        row={'utc':utc(),'event':event,'wall_seconds':time.perf_counter()-started_wall,
            'process_cpu_seconds':time.process_time()-started_cpu,**(details or {}),**extra}
        with (output/'events.jsonl').open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
        save_json(output/'progress.json',{**state,**row})
    def stop(signum,_frame):raise PilotStop('Actual bounded V6b parent signal '+str(signum))
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGXCPU)}
    try:
        # This is the FIRST numerical import. The main task and live parent have
        # already passed. Only trusted numerical helper functions are reused;
        # no old V5 execute(), supervisor, snapshot producer or case label runs.
        install_sites()
        import numpy as np
        from pyscf import gto,lib
        from pyscf.scf import _vhf
        from pyscf.grad.rks import Gradients
        from pyscf.geomopt import geometric_solver
        from pyscf.hessian import thermo
        w=frozen(STATE,STATE_SHA,'calycosin_trusted_numerical_state_helpers')
        base=frozen(BASE,BASE_SHA,'calycosin_trusted_direct_rks_helpers')
        numerical=frozen(V5_NUMERICAL,V5_NUMERICAL_SHA,'calycosin_stock_gate_and_physical_state_only')
        manifest,symbols,coordinates=w.validate_input(INPUT,INPUT_SHA)
        lib.num_threads(args.threads)
        state['versions']=w.environment_identity()
        settings=w.method_settings(args.threads,args.maxsteps)
        settings.update(hessian_max_memory_mb=3000,hessian_max_cycle=50,hessian_level_shift=0,
            engineering_worker_version='CALYCOSIN33_SINGLE_PROVIDED_CONFORMER_STOCK_FULL_OPT_FREQ')
        state['settings']=settings
        require(settings['optimizer_thresholds']==base.THRESHOLDS,'Trusted optimizer thresholds changed')
        shutil.copy2(w.checked_artifact(manifest['input_xyz']),output/'input.xyz')
        shutil.copy2(w.checked_artifact(identity['source_SDF_record']),output/'user_supplied_CID5280448.sdf')
        w.save_immutable_npy(output/'input_coordinates_angstrom.npy',coordinates)
        native=base.NativeDirectDriverThreadGuard(_vhf,lib.with_omp_threads,lib.num_threads,
            native_threads=args.threads,emit=lambda details:emit('native_driver_signature',details))
        native.install()
        mol=gto.Mole();mol.stdout=log
        mol.build(atom=list(zip(symbols,coordinates)),unit='Angstrom',basis='def2-svp',charge=0,spin=0,
            symmetry=False,max_memory=8000,verbose=4)
        mol.incore_anyway=False
        require(mol.natm==33 and mol.nelectron==148 and mol.nao_nr()==354,'Actual built Calycosin dimensions differ')
        state.update(atoms=int(mol.natm),electrons=int(mol.nelectron),basis_functions=int(mol.nao_nr()))
        gradient_class=base.direct_gradient_type(Gradients,emit,memory_mb=lambda:lib.current_memory()[0])
        def gradient_method(mf):
            value=gradient_class(mf);value.grid_response=False;return value
        mf=w.make_scf(mol,settings,output/'optimization.chk')
        mf.callback=lambda env:emit('optimization_scf_cycle',cycle=int(env.get('cycle',-1)),energy_hartree=float(env['e_tot']))
        (output/'geometry_steps').mkdir()
        def geometry_callback(env):
            cycle=int(env['self'].cycle);step=output/'geometry_steps'/f'step_{cycle:04d}'
            xyz=env['mol'].atom_coords(unit='Angstrom');gradient=np.asarray(env['gradients'])
            step.with_suffix('.xyz').write_text(base.xyz_text(symbols,xyz,'Optimization intermediate; not a minimum certificate'))
            w.save_immutable_npy(step.with_suffix('.coordinates_angstrom.npy'),xyz)
            w.save_immutable_npy(step.with_suffix('.gradient.npy'),gradient)
            info={'cycle':cycle,'energy_hartree':float(env['energy']),
                'scf_converged':bool(env['g_scanner'].converged),**base.stationary_checks(gradient)}
            save_json(step.with_suffix('.json'),info,exclusive=True);emit('geometry_cycle',info)
        emit('geometry_optimization_started')
        converged,mol=geometric_solver.kernel(gradient_method(mf),maxsteps=args.maxsteps,
            assert_convergence=True,callback=geometry_callback,**base.THRESHOLDS)
        state['optimizer_converged']=bool(converged)
        require(converged,'Optimizer did not converge within bounded steps')
        xyz=mol.atom_coords(unit='Angstrom')
        (output/'optimized.xyz').write_text(base.xyz_text(symbols,xyz,'Single supplied conformer; inspect actual fresh gradient and frequencies'))
        w.save_immutable_npy(output/'optimized_coordinates_angstrom.npy',xyz)
        w.save_immutable_npy(output/'normal_mode_geometry_bohr.npy',mol.atom_coords(unit='Bohr'))
        w.save_immutable_npy(output/'masses_amu.npy',mol.atom_mass_list(isotope_avg=True))
        state['optimized_xyz_sha256']=sha(output/'optimized.xyz')
        state['optimized_coordinates_angstrom_npy_sha256']=sha(output/'optimized_coordinates_angstrom.npy')
        # A separate NEW same-method SCF, not the optimizer's last gradient or
        # old orbitals. This new checkpoint has no pre-existing initial guess.
        mf=w.make_scf(mol,settings,output/'final_scf.chk')
        mf.callback=lambda env:emit('fresh_final_scf_cycle',cycle=int(env.get('cycle',-1)),energy_hartree=float(env['e_tot']))
        state['energy_hartree']=float(mf.kernel());state['scf_converged']=bool(mf.converged)
        require(mf.converged and np.isfinite(mf.e_tot),'Fresh final SCF not converged')
        gradient=np.asarray(gradient_method(mf).kernel())
        w.save_immutable_npy(output/'gradient_hartree_per_bohr.npy',gradient)
        state.update(base.stationary_checks(gradient))
        state['geometry_optimized']=bool(state['optimizer_converged'] and state['final_gradient_thresholds_passed'])
        require(state['geometry_optimized'],'Fresh same-method final gradient failed; Hessian NOT_RUN')
        for name,value in [('final_mo_coeff',mf.mo_coeff),('final_mo_energy',mf.mo_energy),('final_mo_occ',mf.mo_occ)]:
            w.save_immutable_npy(output/(name+'.npy'),value)
        before=numerical.physical_state(mf)
        state['final_SCF_physical_binding_before_Hessian']=before
        hobj=mf.Hessian();hobj.max_cycle=50;hobj.level_shift=0
        state['stock_hamiltonian_gate']=numerical.stock_operator_gate(mf,hobj)
        state['frequencies_status']='PARTIAL_HESSIAN_NOT_COMPLETE'
        emit('stock_full_system_Hessian_started')
        with base.hessian_memory_context(hobj,settings,emit=lambda event,details:emit(event,details)):
            raw=np.asarray(hobj.kernel(atmlst=list(range(33))))
        w.save_immutable_npy(output/'hessian_hartree_per_bohr2.npy',raw)
        state['full_system_CPHF_returned_normally']=True
        state['normal_return_is_not_a_measured_CPHF_residual']=True
        state['measured_CPHF_residual_norm']=None
        state['stable_MF_state_unchanged']=numerical.physical_state(mf)==before
        require(state['stable_MF_state_unchanged'],'Final physical MF changed across Hessian')
        used,asymmetry=base.prepare_hessian_for_analysis(raw,33)
        w.save_immutable_npy(output/'hessian_used_for_analysis.npy',used)
        analysis=thermo.harmonic_analysis(mol,used,exclude_trans=True,exclude_rot=True,imaginary_freq=False)
        info=base.frequencies_payload(analysis['freq_wavenumber'],analysis['norm_mode'],33)
        require(info['normal_mode_count']==93,'All93 signed nonlinear modes required')
        w.save_immutable_npy(output/'normal_modes.npy',analysis['norm_mode'])
        w.save_immutable_npy(output/'frequencies_cm1.npy',np.asarray(info['frequencies_cm1'],dtype=np.float64))
        w.save_immutable_npy(output/'reduced_mass_amu.npy',analysis['reduced_mass'])
        with (output/'frequencies.csv').open('x',newline='') as handle:
            writer=csv.writer(handle,lineterminator='\n');writer.writerow(['mode','frequency_cm1','frequency_THz','imaginary','IR_intensity_status'])
            for index,value in enumerate(info['frequencies_cm1'],1):
                writer.writerow([index,value,value*CM1_TO_THZ,value<0,'NOT_COMPUTED'])
        state.update(info,**base.frequency_minimum_checks(info),hessian_max_symmetry_difference=asymmetry,
            frequencies_status='DONE',status='FREQUENCIES_COMPUTED',complete_global_Hessian=True,
            mode_geometry_and_Hessian_same_final_SCF=True,
            mass_convention='PySCF isotope-average masses H1.008/C12.011/O15.999',
            harmonic_analysis_excludes_translations_and_rotations=True,
            Hessian_method='ORIGINAL_STOCK_FULL_RKS_KERNEL_NO_INCREMENTAL_OR_STAGE_CACHE',
            positive_minimum_separate_from_frequency_completion=True)
        state['arrays']={p.name:artifact(p) for p in output.glob('*.npy')}
        emit('signed_frequencies_complete',normal_modes=93,imaginary=info['imaginary_frequency_count'])
    except BaseException as error:
        state.update(status='PARTIAL',stop_reason=str(error),exception_type=type(error).__name__)
        (output/'exception_traceback.txt').write_text(traceback.format_exc())
    finally:
        if native is not None:
            try:native.restore()
            except BaseException as error:state.update(status='PARTIAL',stop_reason='Native driver restoration failed: '+str(error))
            state['native_direct_driver_control']=native.summary()
            if (state['native_direct_driver_control'].get('restoration_failures')!=0
                    or state['native_direct_driver_control'].get('original_driver_restored') is not True):
                state.update(status='PARTIAL',stop_reason='Native driver restoration failures')
        else:state['native_direct_driver_control']={'original_driver_restored':True,'restoration_failures':0,'installation_status':'NOT_REACHED'}
        for sig,handler in handlers.items():signal.signal(sig,handler)
        state.update(completed_at_utc=utc(),wall_seconds=time.perf_counter()-started_wall,
            process_cpu_seconds=time.process_time()-started_cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        state['checkpoint_files']=[artifact(p) for p in output.glob('*.chk')]
        log.flush();log.close();save_json(output/'result.json',state,exclusive=True)
    return state


def preflight(output):
    identity=identity_metadata()
    output=Path(output).resolve()
    require(output.is_relative_to(ROOT/'reports') and not output.exists(),'Fresh preparation report directory required')
    dependencies=[artifact(path) for path in (GUARD,STATE,BASE,V5_NUMERICAL,MAIN_WORKER,MAIN_ANALYZER)]
    require([r['sha256'] for r in dependencies]==[GUARD_SHA,STATE_SHA,BASE_SHA,V5_NUMERICAL_SHA,MAIN_WORKER_SHA,MAIN_ANALYZER_SHA],'Frozen dependency changed')
    output.mkdir(parents=True)
    value={'schema':SCHEMA,'status':'NOT_RUN','source_worker':artifact(__file__),'chemical_identity':identity,
        'numerical_imports':False,'SDF_XYZ_NPY_payloads_read':False,'new_quantum_evaluations':0,
        'actual_authorization_or_allocation_created':False,'gate':'ACTUAL_SPINOSIN219_HEALTHY_C_AND_NUMPY_REPLAY_THEN_NEW_ROOT_DECISION',
        'expected_atoms':33,'expected_electrons':148,'expected_modes_NOT_ACTUAL_RESULT':93,
        'basis_function_count_NOT_ACTUAL_RESULT':{'status':'FORMULA_ESTIMATE_ONLY','value':354},
        'policy_max_cpu_seconds':MAX_CPU,'policy_max_wall_seconds':MAX_WALL,'policy_maxsteps':MAX_STEPS,
        'conformer_search':'NOT_RUN_ONE_SUPPLIED_CONFORMER_ONLY','settings':settings_document(2,35),
        'frozen_dependencies':dependencies,'IR_intensities_status':'NOT_COMPUTED'}
    save_json(output/'preflight.json',value,exclusive=True)
    return value


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--authorization',type=Path);p.add_argument('--authorization-sha256')
    p.add_argument('--threads',type=int,choices=(1,2),default=2)
    p.add_argument('--maxsteps',type=int,default=35)
    return p


def main():
    args=parser().parse_args();argument_limits(args)
    result=execute(args) if args.execute else preflight(args.output)
    print(json.dumps({k:result.get(k) for k in ('status','identity','geometry_optimized','normal_mode_count','stop_reason')}),flush=True)
    if args.execute and result['status']!='FREQUENCIES_COMPUTED':raise SystemExit(2)


if __name__=='__main__':main()
