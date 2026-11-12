#!/usr/bin/env python3
"""Checkpointed distinct-conformer PySCF worker; the supervisor owns budgets.

No calculation starts on import or --preflight. Scientific source modules are
immutable pinned copies. A completed case requires the full two-step numerical
Hessian/dipole ledger and its independent publication gate, never an old flag.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import traceback

PUB_COMMIT = '2eb6b6e1dfe6627f24c8f97ea268bdec8f636bcb'
INPUT_COMMIT = 'cd4b50c37c1e26064f9f15f340d30e20ac386458'
INPUT_PREFIX = 'reports/pyscf-spinosin-pilot-20261031/conformers'
KEY = 'VGGSULWDCMWZPO-ODEMIOGVSA-N'
CANDIDATES = {
    'C002': {'xyz': 'b651dedc1e52a91937d3a9c88911ba2f583f55bd589c07ccaddf21a5ff6647b3',
             'sdf': '2e35121ad2c40a9b31e4773861ed197bcda286033a2bf74c4541e5359c7a411c'},
    'C003': {'xyz': 'b89423175036d35a32dc669c89a2dd8e4929f0290c934e2eea48922372f613d0',
             'sdf': 'f5b6a57ec88fc08f3922557a400bd9d2477d9c5ac77399fed63b43dc9e70b806'},
}
SCRIPTS = ('pyscf_publication_core_20261111.py', 'pyscf_publication_quality_20261111.py',
           'pyscf_repair_modes_20261111.py', 'run_pyscf_publication_20261111.py')
BOHR_ANGSTROM = 0.529177210903


class IntegrityError(RuntimeError):
    pass


class CheckpointRequested(BaseException):
    pass


class BasinPending(BaseException):
    def __init__(self,evidence): self.evidence=evidence


class BasinDuplicate(BaseException):
    def __init__(self,evidence): self.evidence=evidence


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def job_spec(root, argument):
    path = Path(argument)
    if path.is_file():
        value = read(path)
    else:
        matches = [v for v in read(root/'jobs.json')['jobs'] if v['id'] == argument]
        if len(matches) != 1:
            raise IntegrityError('Job ID must identify exactly one frozen job')
        value = matches[0]
    identifier = value.get('id', '')
    if not identifier or Path(identifier).name != identifier or identifier in ('.', '..'):
        raise IntegrityError('Invalid job ID')
    candidate = value.get('conformer', value.get('candidate'))
    if candidate is None:
        candidate = next((c for c in CANDIDATES if f'_{c}_' in identifier), None)
    if candidate not in CANDIDATES or value.get('molecule', 'spinosin') != 'spinosin':
        raise IntegrityError('Only authenticated distinct Spinosin C002/C003 inputs are eligible')
    profile = value.get('profile', 'df-grid5')
    if profile not in ('df-grid5', 'df-grid7', 'direct-grid7'):
        raise IntegrityError('Unsupported frozen profile')
    if value.get('threads', 1) != 1 or value.get('max_memory_mb', 2200) not in (2200, 2500):
        raise IntegrityError('Worker requires one thread and 2200/2500 MB model limit')
    value = dict(value, conformer=candidate, molecule='spinosin', profile=profile,
                 threads=1, max_memory_mb=value.get('max_memory_mb', 2200))
    return value


def authenticate_inputs(root, spec):
    """Authenticate archived stereo graph and exact high precision XYZ/SDF pair.

    This replays byte seals, graph adjacency and source-relative stereochemistry;
    it does not claim a new experimental or database retrieval.
    """
    import numpy as np
    manifest = read(root/'source/source_manifest.json')
    index = {(r['commit'], r['path']): r for r in manifest['files']}
    base = root/'source'/INPUT_COMMIT/INPUT_PREFIX
    candidate = spec['conformer']
    inputs = {}
    for extension in ('xyz', 'sdf'):
        default = base/f'{candidate}_mmff94s.{extension}'
        path = Path(spec.get('input_'+extension, default))
        if not path.is_absolute(): path = root/path
        digest = sha(path)
        if digest != CANDIDATES[candidate][extension]:
            raise IntegrityError('Unreviewed conformer bytes: '+str(path))
        if spec.get('input_'+extension+'_sha256', digest) != digest:
            raise IntegrityError('Job input SHA mismatch')
        inputs[extension] = path
    required = [base/'conformer_summary.json', base/'source_graph_verification.json',
                base/'parameters.json', base/'heavy_atom_pair_RMSD.json']
    source = root/'source'/PUB_COMMIT/'scripts'
    required += [source/name for name in SCRIPTS]
    for path in required:
        relative = path.relative_to(root/'source')
        record = index.get((relative.parts[0], '/'.join(relative.parts[1:])))
        if record is None or sha(path) != record['sha256']:
            raise IntegrityError('Pinned source bytes changed: '+str(path))
    summary = read(base/'conformer_summary.json')
    rows = [r for r in summary['all_rows'] if r['candidate'] == candidate]
    if len(rows) != 1 or not rows[0]['eligible_for_DFT'] or rows[0]['MMFF_status'] != 0:
        raise IntegrityError('Conformer did not pass archived MMFF/stereo eligibility')
    row = rows[0]
    if row['stereo_inchikey'] != KEY or any(row[e+'_sha256'] != sha(inputs[e]) for e in ('xyz','sdf')):
        raise IntegrityError('Archived stereo or coordinate seal mismatch')
    graph = read(base/'source_graph_verification.json')
    if (graph['computed_inchikey'] != KEY or graph['formula'] != 'C28H32O15'
            or graph['formal_charge'] != 0 or graph['radical_electrons'] != 0
            or graph['electron_count'] != 320 or len(graph['source_stereocenters']) != 10):
        raise IntegrityError('Spinosin valence/stereo identity record mismatch')
    lines = inputs['xyz'].read_text().splitlines()
    symbols = [line.split()[0] for line in lines[2:]]
    angstrom = np.array([[float(x) for x in line.split()[1:]] for line in lines[2:]])
    if int(lines[0]) != 75 or Counter(symbols) != {'C':28,'H':32,'O':15} or angstrom.shape != (75,3) or not np.isfinite(angstrom).all():
        raise IntegrityError('Incomplete 75-atom Spinosin geometry')
    sdf = inputs['sdf'].read_text().splitlines()
    n, nb = int(sdf[3][:3]), int(sdf[3][3:6])
    sdf_symbols = [line[31:34].strip() for line in sdf[4:4+n]]
    sdf_xyz = np.array([[float(line[a:b]) for a,b in ((0,10),(10,20),(20,30))] for line in sdf[4:4+n]])
    if n != 75 or symbols != sdf_symbols or np.max(np.abs(sdf_xyz-angstrom)) > 5.01e-5:
        raise IntegrityError('SDF graph/XYZ atom order or rounded coordinates mismatch')
    bonds = np.zeros((n,n), dtype=bool)
    for line in sdf[4+n:4+n+nb]:
        i,j = int(line[:3])-1,int(line[3:6])-1
        bonds[i,j] = bonds[j,i] = True
    radii = np.array([{'H':.31,'C':.76,'O':.66}[s] for s in symbols])
    distances = np.linalg.norm(angstrom[:,None]-angstrom[None,:],axis=2)
    neighbors = (distances < 1.25*(radii[:,None]+radii[None,:])) & ~np.eye(n,dtype=bool)
    if not np.array_equal(bonds, neighbors):
        raise IntegrityError('XYZ distance graph differs from authenticated SDF bond graph')
    minimum = float(distances[np.triu_indices(n,1)].min())
    if minimum < .45:
        raise IntegrityError('Unphysical coincident atoms')
    references = {str(p.relative_to(root)):sha(p) for p in required+list(inputs.values())}
    metadata = {'molecule':'Spinosin','conformer':candidate,'formula':'C28H32O15',
        'pubchem_cid':155692,'chebi_id':'CHEBI:81360','inchikey':KEY,
        'atoms':75,'charge':0,'spin_2S':0,'multiplicity':1,'electron_count':320,
        'input_geometry_origin':'Archived ETKDGv3/MMFF94s candidate; not measured or DFT optimized',
        'MMFF_energy_kcal_mol':row['MMFF_energy_kcal_mol'],
        'MMFF_relative_to_C001_kcal_mol':row['MMFF_energy_kcal_mol']-summary['selected_for_initial_DFT']['MMFF_energy_kcal_mol'],
        'source_hashes':references,'source_publication_commit':PUB_COMMIT,'source_input_commit':INPUT_COMMIT,
        'profile_refinement_ladder':['df-grid5','df-grid7','direct-grid7'],
        'scientific_blocked_exit_code':76,
        'geometry_units':'Bohr after exact division of high precision Angstrom XYZ',
        'source_relative_identity_guard':True,'new_independent_database_identity_validation':False}
    return symbols, angstrom/BOHR_ANGSTROM, metadata


def integrity_exception(error):
    return isinstance(error, IntegrityError) or any(term in str(error).lower() for term in (
        'integrity error', 'fingerprint changed', 'scientific inputs changed', 'array hash changed'))


def optimizer_point_path(path, fingerprint):
    """GeomeTRIC can reuse an evaluation index after a partially saved step.

    Keep each original molecular point, keyed by the complete upstream input
    fingerprint, rather than confuse a reused index with scientific tampering.
    Numerical derivative paths remain unchanged for independent raw-data audit.
    """
    path = Path(path)
    if path.parent.name == 'steps' and path.name.startswith('step-'):
        return path/fingerprint
    return path


def cleanup_rolling_checkpoint(point_directory,result):
    """Only after durable qualified arrays; never remove referenced evidence."""
    if result.get('status')!='DONE' or result.get('qualified') is not True: return 0
    protected=set()
    for reference in result.get('array_refs',{}).values():
        path=Path(reference['path']).resolve()
        if not path.is_file() or sha(path)!=reference['sha256']:
            raise IntegrityError('Point evidence seal failed before checkpoint cleanup')
        protected.add(path)
    if not protected: raise IntegrityError('Qualified point has no sealed durable arrays')
    removed=0
    for path in Path(point_directory).glob('attempt-*/.local/rolling.chk'):
        if path.resolve() in protected: raise IntegrityError('Rolling checkpoint is referenced evidence')
        if path.is_file(): removed+=path.stat().st_size;path.unlink()
    return removed


def aligned_geometry_difference(left,right):
    """Same authenticated atom mapping, proper rigid rotation, no reflection."""
    import numpy as np
    a,b=np.asarray(left,dtype=float),np.asarray(right,dtype=float)
    if a.shape!=b.shape or a.shape!=(75,3) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise IntegrityError('Incomplete same-basin geometry comparison')
    a=a-a.mean(axis=0);b=b-b.mean(axis=0)
    u,s,vt=np.linalg.svd(a.T@b)
    correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt)
    residual=a@(u@correction@vt)-b
    return {'all_atom_rmsd_bohr':float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
            'maximum_atom_deviation_bohr':float(np.linalg.norm(residual,axis=1).max())}


def local_duplicate_evidence(root,spec,coordinates,density,current_out,core,runner):
    """Only earlier job ordinals can own a basin: no pairwise waiting cycle.

    This conservative test uses the authenticated same atom order, all atoms,
    identical numerical settings, strict gradients, and equal electronic energy.
    Symmetry permutations are not guessed; missing a duplicate is preferable to
    merging distinct structures. Unqualified sealed earlier cases never block.
    """
    if (Path(current_out)/'h0/c000-plus/result.json').exists(): return None
    jobs=read(root/'jobs.json',{}).get('jobs',[])
    ordinal=spec.get('ordinal',next((i for i,v in enumerate(jobs) if v['id']==spec['id']),0))
    current_opt=read(Path(current_out).parent/'optimize/result.json',{})
    # After an orbital-stability reoptimization, use that final optimum energy.
    own_opts=[read(p,{}) for p in Path(current_out).parent.glob('reoptimized-stable-state-*/result.json')]
    own_opts=[v for v in own_opts if v.get('optimizer_converged') and v.get('minimum_stationarity_passed')]
    if own_opts: current_opt=own_opts[-1]
    own_energy=current_opt.get('final_point',{}).get('energy_hartree')
    if own_energy is None: return None
    for i,other in enumerate(jobs):
        if other['id']==spec['id'] or other.get('ordinal',i)>=ordinal: continue
        other_dir=root/'jobs'/other['id']
        overall=read(other_dir/'result.json',{})
        if overall.get('status')=='scientific_blocked': continue
        frozen=read(other_dir/'job_frozen.json',{})
        if frozen.get('identity',{}).get('inchikey')!=KEY: continue
        # C002/C003 originate from one pinned graph/atom ordering. An external
        # geometry with an unreviewed mapping is deliberately ineligible.
        if frozen.get('job',{}).get('conformer') not in CANDIDATES: continue
        for path in sorted((other_dir/'science'/core.settings()['profile']).glob('round*/optimize/result.json')):
            opt=read(path,{})
            if (opt.get('optimizer_converged') is not True or opt.get('minimum_stationarity_passed') is not True
                    or opt.get('settings')!=core.settings()): continue
            case_dir=path.parent.parent;case=runner.sealed(case_dir)
            if case is not None and not case.get('paper_ready'): continue
            other_coords=case.get('coordinates_bohr') if case else opt.get('coordinates_bohr')
            energy=case.get('energy_hartree') if case else opt.get('final_point',{}).get('energy_hartree')
            if energy is None or other_coords is None: continue
            difference=aligned_geometry_difference(coordinates,other_coords)
            de=abs(float(own_energy)-float(energy))
            if (difference['all_atom_rmsd_bohr']>1e-5 or difference['maximum_atom_deviation_bohr']>3e-5 or de>1e-7): continue
            evidence={'task_outcome':'basin_duplicate','canonical_job_id':other['id'],
                'canonical_case':str(case_dir),'profile':core.settings()['profile'],
                'atom_mapping':'Same pinned explicit-H graph and atom indices; no symmetry permutation or reflection',
                'geometry_difference':difference,'absolute_energy_difference_hartree':de,
                'comparison_tolerances':{'all_atom_rmsd_bohr':1e-5,'maximum_atom_deviation_bohr':3e-5,'energy_hartree':1e-7},
                'new_hessian_computed':False,'independent_conformer_sample':False,
                'scope':'Duplicate stationary basin within the explicit numerical tolerances; reuse canonical evidence without adding a chemical sample.'}
            if case is None: return dict(evidence,canonical_hessian_status='AWAITING_COMPLETE_QUALIFIED_EVIDENCE')
            point=read(case['center_point_path'])
            gate=runner.quality.audit_publication(case_dir/'modes',case_dir/'finite-differences',point,
                electronic_stability_passed=case['electronic_stability']['tested_electronic_stability_passed'])
            if not gate['paper_ready_molecular_harmonic_data']: raise IntegrityError('Canonical basin qualification replay failed')
            runner.modes.verify_analysis(case_dir/'modes')
            return dict(evidence,canonical_hessian_status='QUALIFIED',
                        canonical_case_result_sha256=sha(case_dir/'result.json'),publication_gate=gate,
                        coordinates_bohr=coordinates.tolist(),energy_hartree=own_energy,method=core.settings())
    return None


def verified_checkpoint_seed(directory,state,default_coordinates):
    import numpy as np
    if not state.get('round',0) and 'seed_coordinates_bohr' not in state:
        return default_coordinates,None
    def evidence_path(value):
        if not value: raise IntegrityError('Checkpoint evidence path is missing')
        path=Path(value).resolve()
        try: path.relative_to(directory.resolve())
        except ValueError as error: raise IntegrityError('Checkpoint evidence escapes this job') from error
        if not path.is_file(): raise IntegrityError('Checkpoint evidence file is missing: '+str(path))
        return path
    previous=evidence_path(str(Path(state.get('previous_case',''))/'result.json'))
    if sha(previous)!=state.get('previous_case_result_sha256'):
        raise IntegrityError('Checkpoint previous case SHA changed')
    source=evidence_path(state.get('seed_source_path'))
    if sha(source)!=state.get('seed_source_sha256'):
        raise IntegrityError('Checkpoint seed source SHA changed')
    record=read(source)
    if state.get('seed_kind')=='supported_negative_branch':
        record=record.get('selected')
    if not isinstance(record,dict): raise IntegrityError('Checkpoint has no authenticated selected seed')
    coordinates=np.asarray(state.get('seed_coordinates_bohr'),dtype=float)
    reference=np.asarray(record.get('coordinates_bohr'),dtype=float)
    if (coordinates.shape!=(75,3) or not np.isfinite(coordinates).all()
            or reference.shape!=coordinates.shape or not np.array_equal(reference,coordinates)):
        raise IntegrityError('Checkpoint seed coordinates differ from their authenticated source')
    density_path=state.get('seed_density_path')
    if density_path!=record.get('density_path'):
        raise IntegrityError('Checkpoint density path differs from its authenticated source')
    density=None
    if density_path:
        path=evidence_path(density_path)
        if sha(path)!=state.get('seed_density_sha256'):
            raise IntegrityError('Checkpoint seed density SHA changed')
        density=np.load(path,allow_pickle=False)
        if (density.ndim!=2 or density.shape[0]!=density.shape[1]
                or np.iscomplexobj(density) or not np.isfinite(density).all()):
            raise IntegrityError('Checkpoint seed density is not a finite real square matrix')
    elif state.get('seed_density_sha256') is not None:
        raise IntegrityError('Checkpoint density seal exists without a density')
    return coordinates,density


def scientific_loop(spec, directory, runner, core, coordinates, update):
    """Follow supported negative directions without a fixed round-count cutoff."""
    import numpy as np
    ladder = ['df-grid5','df-grid7','direct-grid7']
    state = read(directory/'checkpoint.json', {'round':0,'profile_index':ladder.index(spec['profile'])})
    if not isinstance(state.get('round'),int) or state['round']<0 or state.get('profile_index') not in range(3):
        raise IntegrityError('Invalid scientific checkpoint round/profile')
    coordinates,density=verified_checkpoint_seed(directory,state,coordinates)
    while True:
        round_id = state['round']
        profile = ladder[state['profile_index']]
        case_dir = directory/'science'/profile/f'round{round_id:06d}'
        update(stage='STATIONARY_CASE', profile=profile, scientific_round=round_id, case_path=str(case_dir))
        case = runner.stationary_case('spinosin', profile, coordinates, density, case_dir)
        if case.get('paper_ready') is True:
            # Replay original rawpoint SHAs, complete two-step derivatives and IR
            # gate before supervisor-facing completion, including after restart.
            point = read(case['center_point_path'])
            try:
                gate = runner.quality.audit_publication(case_dir/'modes', case_dir/'finite-differences', point,
                    electronic_stability_passed=case['electronic_stability']['tested_electronic_stability_passed'])
                runner.modes.verify_analysis(case_dir/'modes')
            except Exception as error:
                raise IntegrityError('Completed scientific evidence failed independent replay: '+str(error)) from error
            if not gate['paper_ready_molecular_harmonic_data']:
                raise IntegrityError('Previously qualified full derivative gate failed replay')
            return {'selected_case':str(case_dir),'case_result_sha256':sha(case_dir/'result.json'),
                    'method':case['method'],'coordinates_bohr':case['coordinates_bohr'],
                    'energy_hartree':case['energy_hartree'],'publication_gate':gate,
                    'profile':profile,'scientific_round':round_id,
                    'mode_count':case['analysis']['steps'][-1]['vibrational_mode_count'],
                    'independent_conformer_sample':False,
                    'basin_identity_status':'REQUIRES_COMPARISON_WITH_OTHER_FINAL_GEOMETRIES',
                    'scope':'Isolated molecule local harmonic and IR prediction; distinct starting conformers may converge to the same basin.'}
        selected = None
        follow = {'status':case.get('status')}
        if 'analysis' in case:
            follow = runner.negative_branches('spinosin',profile,case,case_dir/'negative-mode-recovery')
            selected = follow.get('selected')
        if selected is not None:
            coordinates = np.asarray(selected['coordinates_bohr'],dtype=float)
            density_path = selected['density_path']
            density = np.load(density_path,allow_pickle=False)
            next_round, next_profile = round_id+1,state['profile_index']
            seed_source=case_dir/'negative-mode-recovery/selection.json'
            seed_kind='supported_negative_branch'
        elif case.get('status') == 'IDENTITY_GUARD_FAILED' or state['profile_index'] == len(ladder)-1:
            reason = {'case_status':case.get('status'),'branch_status':follow.get('status'),
                      'publication_gate':case.get('publication_gate')}
            update(stage='SCIENTIFIC_BLOCKED', reason=reason, complete=False)
            return {'scientific_blocked':True,'complete':False,'profile':profile,
                    'case_path':str(case_dir),'reason':reason,
                    'scope':'No remaining predeclared refinement. This job must be skipped until a reviewed new scientific action is authorized; do not loop cached unresolved results.'}
        else:
            # A failed full Hessian/IR gate without a supported downhill branch
            # advances the numerical model. Never retry a sealed failure as an
            # empty checkpoint. These are distinct-conformer profile refinements.
            seed = case if 'coordinates_bohr' in case else read(case_dir/'optimize/result.json',{})
            if seed.get('coordinates_bohr') is not None:
                coordinates = np.asarray(seed['coordinates_bohr'],dtype=float)
            density_path = seed.get('density_path')
            density = np.load(density_path,allow_pickle=False) if density_path else None
            seed_source=case_dir/'result.json' if 'coordinates_bohr' in case else case_dir/'optimize/result.json'
            seed_kind='numerical_profile_refinement'
            if not seed_source.exists() or seed.get('coordinates_bohr') is None:
                # A failure before a closed optimization can still escalate from
                # the already-authenticated seed; bind that earlier source.
                if state.get('seed_source_path'):
                    seed_source=Path(state['seed_source_path']);seed_kind=state['seed_kind']
                else:
                    seed_source=case_dir/'refinement_seed.json'
                    atomic(seed_source,{'coordinates_bohr':coordinates.tolist(),'density_path':density_path,
                                        'origin':'Authenticated original job input; no successful optimizer geometry available'})
            next_round,next_profile = 0,state['profile_index']+1
            update(stage='ADVANCING_NUMERICAL_PROFILE',previous_profile=profile,
                   next_profile=ladder[next_profile],reason=follow.get('status'),complete=False)
        state = {'round':next_round,'profile_index':next_profile,
                 'seed_coordinates_bohr':coordinates.tolist(),
                 'seed_density_path':density_path,'previous_case':str(case_dir),
                 'seed_density_sha256':sha(density_path) if density_path else None,
                 'seed_source_path':str(seed_source),'seed_source_sha256':sha(seed_source),'seed_kind':seed_kind,
                 'previous_case_result_sha256':sha(case_dir/'result.json'),'updated_at_utc':utc()}
        atomic(directory/'checkpoint.json',state)


def run(root, argument, preflight=False):
    spec = job_spec(root, argument)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    if not preflight:
        temporary_root=(root/'tmpfiles').resolve()
        supplied=Path(os.environ.get('PYSCF_TMPDIR',os.environ.get('TMPDIR','/tmp'))).resolve()
        try: supplied.relative_to(temporary_root);scratch=supplied
        except ValueError:
            label=hashlib.sha256(os.environ.get('CHEM_ATTEMPT_ID',str(os.getpid())).encode()).hexdigest()[:20]
            scratch=temporary_root/(label+'-'+str(os.getpid()))
        scratch.mkdir(parents=True,exist_ok=True,mode=0o700)
        os.environ['TMPDIR']=os.environ['PYSCF_TMPDIR']=str(scratch)
        tempfile.tempdir=str(scratch)
    symbols, coordinates, metadata = authenticate_inputs(root,spec)
    if preflight:
        return {'status':'preflight_passed','quantum_calculations':0,'spec':spec,'identity':metadata,
                'profile_refinement_ladder':['df-grid5','df-grid7','direct-grid7'],
                'finite_difference_steps_bohr':[.002,.001],'planned_displaced_points_per_case':900}
    source = root/'source'/PUB_COMMIT/'scripts'
    sys.path.insert(0,str(source))
    core = importlib.import_module('pyscf_publication_core_20261111')
    runner = importlib.import_module('run_pyscf_publication_20261111')
    versions = {p:importlib.metadata.version(p) for p in ('pyscf','geometric','numpy','scipy','h5py')}
    if versions['pyscf'] != '2.13.0':
        raise IntegrityError('Pinned PySCF 2.13.0 required')
    original_configure = core.configure
    def configure(profile=spec['profile'],threads=1):
        original_configure(profile,threads=1)
        core.SCF_SETTINGS['max_memory_mb'] = spec['max_memory_mb']
        return core.settings()
    core.configure = configure
    configure()
    directory = root/'jobs'/spec['id']; directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen = {'schema':'distinct-conformer-chem-worker-v1','job':spec,'identity':metadata,
                  'versions':versions,'settings':core.settings(), 'worker_sha256':sha(__file__),
                  'finite_difference_steps_bohr':[.002,.001],
                  'profile_refinement_ladder':['df-grid5','df-grid7','direct-grid7'],
                  'completion_requires':'Strict stable stationary geometry + all 900 gradient/dipole points + independent Hessian and IR publication gate',
                  'budget_owned_by':'external_supervisor','unlimited_supported_negative_mode_rounds':True}
        previous = read(directory/'job_frozen.json')
        if previous is not None and previous != frozen:
            raise IntegrityError('Frozen job or source fingerprint changed')
        if previous is None: atomic(directory/'job_frozen.json',frozen)
        mutex=threading.RLock(); stopped=threading.Event(); finished=threading.Event()
        attempt=os.environ.get('CHEM_ATTEMPT_ID','unmanaged-'+str(os.getpid()))
        try:
            stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
            start_ticks=int(stat[19]);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except (OSError,ValueError,IndexError): start_ticks=None;boot=None
        status={'schema':'chem-worker-progress-v1','job_id':spec['id'],'attempt_id':attempt,
                'pid':os.getpid(),'start_ticks':start_ticks,'boot_id':boot,'complete':False}
        def update(**values):
            with mutex:
                status.update(values,updated_at_utc=utc(),process_cpu_seconds=time.process_time())
                status['checkpoint_token']=int(status.get('checkpoint_token',0))+1
                atomic(directory/'progress.json',status)
            if stopped.is_set(): raise CheckpointRequested()
        def heartbeat(closed=False):
            receipt={'job_id':spec['id'],'attempt_id':attempt,'pid':os.getpid(),
                     'start_ticks':start_ticks,'boot_id':boot,'process_cpu_seconds':time.process_time(),
                     'cpu_seconds':time.process_time(),'closed':closed,'updated_at_utc':utc(),
                     'accounting_scope':'own process including imports; no child processes'}
            atomic(directory/'heartbeat.json',receipt);atomic(directory/'cpu_receipt.json',receipt)
        def ticker():
            while not finished.wait(5): heartbeat()
        def on_signal(signum,frame):
            stopped.set()
            raise CheckpointRequested('Signal '+str(signum))
        for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGXCPU): signal.signal(signum,on_signal)
        runner.progress=update
        runner.inputs=lambda name:(symbols,coordinates.copy(),None) if name=='spinosin' else (_ for _ in ()).throw(IntegrityError('Unexpected molecule'))
        runner.OUTPUT=directory
        original_displacement=runner.displacement_set
        def displacement(name,point_symbols,point_coords,density,out):
            duplicate=local_duplicate_evidence(root,spec,point_coords,density,out,core,runner)
            if duplicate:
                atomic(directory/'basin_comparison.json',duplicate)
                if duplicate['canonical_hessian_status']=='QUALIFIED': raise BasinDuplicate(duplicate)
                raise BasinPending(duplicate)
            return original_displacement(name,point_symbols,point_coords,density,out)
        runner.displacement_set=displacement
        original_point=core.calculate_point
        original_atomic=core.atomic_json
        resolved_optimizer_points={}
        def scientific_atomic(path,value):
            if isinstance(value,dict) and value.get('last_point_path') in resolved_optimizer_points:
                value=dict(value,last_point_path=resolved_optimizer_points[value['last_point_path']])
            return original_atomic(path,value)
        core.atomic_json=scientific_atomic
        def point(*args,**kwargs):
            args=list(args);kwargs=dict(kwargs)
            target=Path(args[2] if len(args)>2 else kwargs['out'])
            if target.parent.name=='steps' and target.name.startswith('step-'):
                point_symbols=args[0] if args else kwargs['symbols']
                point_coords=args[1] if len(args)>1 else kwargs['coords']
                grid=args[3] if len(args)>3 else kwargs.get('grid_level')
                seed=args[4] if len(args)>4 else kwargs.get('dm0')
                fingerprint,_=core._fingerprint(point_symbols,__import__('numpy').asarray(point_coords),grid,seed)
                resolved=optimizer_point_path(target,fingerprint)
                resolved_optimizer_points[str(target/'result.json')]=str(resolved/'result.json')
                if len(args)>2: args[2]=resolved
                else: kwargs['out']=resolved
                target=resolved
            update(stage='ELECTRONIC_POINT',point_path=str(target))
            result=original_point(*args,**kwargs)
            reclaimed=cleanup_rolling_checkpoint(target,result)
            # Releasing unreachable PySCF objects closes/removes their own DF
            # temporary files. Scientific arrays and densities stay under jobs.
            gc.collect()
            if reclaimed: atomic(target/'temporary_cleanup.json',{
                'rolling_checkpoint_bytes_removed':reclaimed,'all_array_refs_verified':True,
                'density_gradient_dipole_geometry_retained':True,'updated_at_utc':utc()})
            update(stage='ELECTRONIC_POINT_RECORDED',point_path=result.get('attempt_path'),point_qualified=result.get('qualified'))
            return result
        core.calculate_point=point
        thread=threading.Thread(target=ticker,name='cpu-heartbeat',daemon=True)
        thread.start();heartbeat()
        try:
            update(stage='STARTING_OR_RESUMING',profile=spec['profile'],conformer=spec['conformer'])
            try:
                result=scientific_loop(spec,directory,runner,core,coordinates,update)
            except BasinDuplicate as duplicate:
                result=duplicate.evidence
            if result is None:
                atomic(directory/'result.json',{'status':'checkpoint','complete':False,'job_id':spec['id'],
                                               'progress':dict(status),'updated_at_utc':utc()})
                return 75
            if result.get('scientific_blocked'):
                result.update(status='scientific_blocked',complete=False,testing=False,job_id=spec['id'],
                              job_frozen_sha256=sha(directory/'job_frozen.json'),updated_at_utc=utc())
                atomic(directory/'result.json',result)
                atomic(directory/'scientific_blocked.json',result)
                return 76
            result.update(status='complete',complete=True,testing=False,job_id=spec['id'],
                          conformer=spec['conformer'],identity=metadata,job_frozen_sha256=sha(directory/'job_frozen.json'),
                          completed_at_utc=utc())
            atomic(directory/'result.json',result)
            atomic(directory/'done.json',{'status':'complete','complete':True,'testing':False,
                   'result_sha256':sha(directory/'result.json'),'job_frozen_sha256':sha(directory/'job_frozen.json'),
                   'completed_at_utc':utc()})
            update(stage='COMPLETE',complete=True)
            return 0
        except CheckpointRequested as error:
            with mutex:
                status.update(stage='CHECKPOINTED',complete=False,reason=str(error),updated_at_utc=utc())
                atomic(directory/'progress.json',status)
            return 75
        except BasinPending as pending:
            update(stage='WAITING_CANONICAL_BASIN_EVIDENCE',canonical_job_id=pending.evidence['canonical_job_id'],complete=False)
            atomic(directory/'result.json',{'status':'dependency_wait','complete':False,'job_id':spec['id'],
                   'canonical_job_id':pending.evidence['canonical_job_id'],'minimum_recheck_seconds':600,
                   'basin_comparison':pending.evidence,'updated_at_utc':utc()})
            return 77
        except Exception as error:
            code=78 if integrity_exception(error) else 75
            atomic(directory/'last_error.json',{'status':'integrity_blocked' if code==78 else 'retry_needed',
                   'complete':False,'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
                   'attempt_id':attempt,'updated_at_utc':utc()})
            return code
        finally:
            finished.set();thread.join(timeout=1);heartbeat(closed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    try:
        result=run(args.root.resolve(),args.job,args.preflight)
        if isinstance(result,dict): print(json.dumps(result,ensure_ascii=False,indent=2));return 0
        return result
    except Exception as error:
        print(json.dumps({'status':'error','complete':False,'type':type(error).__name__,'error':str(error)}),file=sys.stderr)
        return 78 if integrity_exception(error) else 75


if __name__=='__main__':
    raise SystemExit(main())
