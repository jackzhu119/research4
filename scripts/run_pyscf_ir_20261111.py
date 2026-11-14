#!/usr/bin/env python3
"""Consume sealed molecular minima and independently refine their IR strengths.

A single invocation scans a persistent queue. Waiting, scientific rejection and
point errors never exhaust the queue. The detached supervisor owns resource
accounting. Other PySCF processes and their branches are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

import pyscf_publication_core_20261111 as electronic
import pyscf_ir_core_20261111 as core
import pyscf_ir_quality_20261111 as quality
import pyscf_ir_sources_20261111 as sources

ROOT = Path(os.environ.get('PYSCF_IR_REPOSITORY_ROOT', Path(__file__).resolve().parents[1]))
OUTPUT = ROOT / 'reports/pyscf_ir_20261111'
TARGETS = ('acid', 'calycosin', 'spinosin')
BASE_ELECTRONIC_SETTINGS = dict(electronic.SCF_SETTINGS)


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    electronic.atomic_json(path, value)


def progress(**values):
    previous = read(OUTPUT/'progress.json', {})
    previous.update(values, heartbeat_at_utc=electronic.utc())
    write(OUTPUT/'progress.json', previous)


def protocol():
    return {
        'schema': 'pyscf-independent-ir-v1',
        'molecules': list(TARGETS), 'authorized_cpu_seconds': 900000,
        'scope': 'isolated molecule local harmonic integrated IR, standard km/mol',
        'primary_model': 'exact upstream PBE/def2-SVP/noD3 profile, atom order and geometry',
        'independent_checks': ['SI and atomic-constant prefactors', 'mass Gram matrix',
            'same-frame derivative vectors and strengths', 'three-step derivative sensitivity',
            'all-mode simultaneous-displacement SCF dipoles', 'total-dipole rigid-body identities'],
        'quality_failures_end_execution': False,
        'maximum_refinement_rounds': None, 'maximum_retries': None,
        'arbitrary_runtime_timeout': None,
        'resource_boundary': '900000 measured cumulative CPU seconds; preserve partial results',
        'source_publication': 'Read-only upstream fetch; source scientific arrays and original points authenticated',
        'nuclear_gradients_at_ir_points': False,
        'source_mass_and_mode_convention': 'Cartesian L columns; L.T M_amu L=I',
        'never_delete_imaginary_modes_or_take_absolute_frequencies': True,
    }


def configure_precision(profile, precision):
    electronic.SCF_SETTINGS.update(BASE_ELECTRONIC_SETTINGS)
    electronic.configure(profile, threads=2)
    if precision == 1:
        electronic.SCF_SETTINGS.update(conv_tol_grad=5e-10,
            polish_conv_tol_grad=2e-10, fresh_orbital_gradient_acceptance=1e-9)


def require_same_physical_model(case):
    source=case['method'];effective=electronic.settings()
    defaults={'density_fitting':False,'density_fitting_auxbasis':None}
    for key in ('xc','basis','charge','spin','dispersion','density_fitting','grid_level','density_fitting_auxbasis'):
        if source.get(key,defaults.get(key)) != effective.get(key):
            raise ValueError('IR source physical model differs: '+key)


def sampling_plan(original, attempt):
    """New resolutions accumulate; a noisy source is replaced by new triples.

    Source precision explores two smaller and one larger step before strict
    fresh-Fock precision. After strict initial h,h/2,h/4, new factor-two
    resolutions alternate smaller/larger. Each (geometry,model,precision,h)
    point set has one reusable cache. No round count ends the global queue.
    """
    h=min(float(x) for x in original)
    if attempt < 3:
        return 0, [h/2,h/4,4*h][attempt]
    index=attempt-3
    if index < 3:
        return 1,[h,h/2,h/4][index]
    index-=3
    exponent=3+index//2 if index%2==0 else -(1+index//2)
    return 1,h*2.0**(-exponent)


def geometry_model_key(case):
    # Match the electronic point cache's implementation and runtime identity.
    return core.fingerprint(case['symbols'],case['coordinates_bohr'],None)[0]


def resolvable_sampling_plan(original,index,coordinates):
    maxcoord=max(float(np.max(np.abs(coordinates))),1.0)
    floor=128*np.finfo(float).eps*maxcoord
    unavailable=set()
    while True:
        precision,h=sampling_plan(original,index)
        if floor < h <= .02:
            return precision,h,index
        unavailable.add('small' if h <= floor else 'large')
        # Each side grows monotonically beyond the local/roundoff bounds.
        if len(unavailable)==2:
            return None
        index+=1


def continuous_triples(pool):
    steps=sorted(pool,reverse=True)
    triples=[]
    for h in steps:
        trio=[h,h/2,h/4]
        if all(x in pool for x in trio):
            triples.append(trio)
    return triples


def choose_derivatives(pool,vectors,masses):
    trials=[]
    for hs in continuous_triples(pool):
        ds=np.stack([pool[h] for h in hs])
        check=quality.common_frame_derivative_comparison(ds.transpose(0,2,1),hs,vectors.T,masses)
        projected=np.einsum('dk,skc->sdc',vectors.T,ds.transpose(0,2,1))
        error=float(np.max(np.linalg.norm(np.diff(projected,axis=0),axis=-1)))
        trials.append((not check['qualified'],error,hs,ds,check))
    if trials:
        _,_,hs,ds,check=min(trials,key=lambda x:(x[0],x[1]))
        return hs,ds,check
    hs=sorted(pool,reverse=True)
    ds=np.stack([pool[h] for h in hs])
    check={'qualified':False,'valid':True,'failures':['three_consecutive_resolutions_pending']}
    return hs,ds,check


def process_case(descriptor, attempt=0):
    case = sources.load_source(descriptor)
    identifier = case['id']
    out = OUTPUT/'jobs'/identifier/f'round{attempt:05d}'
    existing = read(out/'result.json')
    if existing and existing.get('status') in ('QUALIFIED', 'DIAGNOSTIC_COMPLETE', 'REFINEMENT_REQUIRED'):
        _verify_completed(out)
        return existing
    name, profile = case['molecule'], case['profile']
    previous=[read(p) for p in (OUTPUT/'jobs'/identifier).glob('round*/result.json')]
    sampling_index=max([attempt]+[int(p.get('sampling_index',p.get('attempt',0)))+1 for p in previous])
    plan=resolvable_sampling_plan(case['steps_bohr'],sampling_index,case['coordinates_bohr'])
    if plan is None:
        write(out/'waiting.json',{'status':'WAITING_FOR_NEW_SOURCE_OR_MODEL',
            'reason':'both_smaller_and_larger_new_steps_are_outside_resolvable_local_range',
            'actual_round':attempt})
        return {'status':'WAITING_FOR_NEW_SOURCE_OR_MODEL','paper_ready':False}
    precision,extra_step,sampling_index=plan
    configure_precision(profile,precision)
    require_same_physical_model(case)
    symbols = case['symbols']
    coords = np.asarray(case['coordinates_bohr'], dtype=float)
    masses = np.asarray(case['masses_amu'], dtype=float)
    vectors = np.asarray(case['cartesian_modes'], dtype=float)
    original = np.asarray(case['cartesian_dipole_derivatives'], dtype=float)
    original_steps = list(case['steps_bohr'])
    if vectors.shape[0] != coords.size or original.shape[1:] != (3, coords.size):
        raise ValueError('Unexpected authenticated source mode/derivative layout')
    norm = quality.validate_mode_normalization(vectors.T, masses)
    if not norm['qualified']:
        raise ValueError('Authenticated source mass normalization failed')
    # Keep all prior accepted matrices from this exact geometry and model.
    pool_root=OUTPUT/'derivative_pools'/geometry_model_key(case)/f'precision{precision}'
    pool={float(h):d for h,d in zip(original_steps,original)} if precision==0 else {}
    for receipt_path in sorted(pool_root.glob('h*/COMPLETE.json')):
        receipt=read(receipt_path)
        array_path=receipt_path.parent/'dipole_derivative.npy'
        if electronic.sha256(array_path)!=receipt['array_sha256']:
            raise ValueError('Derivative pool array seal mismatch')
        pool[float(receipt['step_bohr'])]=np.load(array_path,allow_pickle=False)
    direct_steps=None
    input_spec = {
        'source_id': identifier, 'source': case.get('source'), 'molecule': name,
        'profile': profile, 'attempt': attempt,'sampling_index':sampling_index,
        'coordinates_bohr': coords.tolist(),
        'symbols': symbols, 'masses_amu': masses.tolist(),
        'original_steps_bohr': original_steps, 'additional_cartesian_step_bohr': extra_step,
        'electronic_precision_level':precision,'effective_electronic_settings':electronic.settings(),
        'source_minimum_qualified': bool(case['source_minimum_qualified']),
        'source_ir_qualified': bool(case['source_ir_qualified']),
    }
    write(out/'input.json', input_spec)
    def report(**values):
        progress(molecule=name, profile=profile, source_id=identifier, round=attempt, **values)
    report(stage='CENTER_DIPOLE')
    center = core.qualified_dipole(symbols, coords, pool_root/'center',require_density=True)
    energy_error = abs(float(center['energy_hartree'])-float(case['source_center_energy_hartree']))
    dipole_error = float(np.linalg.norm(np.asarray(center['dipole_au'])-np.asarray(case['source_center_dipole_au'])))
    if not np.isfinite(energy_error) or not np.isfinite(dipole_error) or energy_error > 1e-8 or dipole_error > 1e-5:
        raise RuntimeError('IR center does not reproduce the authenticated source electronic state')
    density = np.load(center['density_path'], allow_pickle=False)
    step_path=pool_root/('h'+extra_step.hex())
    derivative=core.cartesian_derivative(symbols,coords,step_path,extra_step,density,report)
    write(step_path/'COMPLETE.json',{'step_bohr':extra_step,
        'array_sha256':electronic.sha256(step_path/'dipole_derivative.npy'),
        'electronic_precision_level':precision,'all_point_electronic_audits_qualified':True})
    pool[extra_step]=derivative
    hs,ds,common=choose_derivatives(pool,vectors,masses)
    finest = ds[int(np.argmin(hs))]
    projected = (finest @ vectors).T
    if common['qualified']:
        direct_steps=[8*min(hs),4*min(hs),2*min(hs)]
        # Each normal-coordinate resolution is a distinct cache; across
        # rounds the same mode/geometry/precision never needs re-evaluation.
        collected=[];ledger=[]
        mode_digest=hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest()
        for h in direct_steps:
            part,points=core.directional_derivatives(symbols,coords,vectors,
                pool_root/'directional'/mode_digest/('q'+h.hex()),[h],density,report)
            collected.append(part[0]);ledger.extend(points)
        direct=np.stack(collected)
        independent=quality.directional_comparison(projected,direct.transpose(0,2,1),direct_steps)
    else:
        direct=np.empty((0,3,vectors.shape[1]));ledger=[];direct_steps=[]
        independent={'qualified':False,'valid':True,'failures':['awaiting_common_frame_convergence']}
    input_spec['normal_coordinate_steps_bohr_sqrt_amu']=direct_steps
    rigid = quality.rigid_body_residuals(finest.T.reshape(len(symbols),3,3), coords,
                                       center['dipole_au'], electronic.settings()['charge'])
    factor_relative = abs(quality.prefactor_si()/quality.prefactor_atomic()-1)
    factor_passed = factor_relative < 2e-9
    ir_ok = bool(norm['qualified'] and common['qualified'] and independent['qualified']
                 and rigid['qualified'] and factor_passed)
    minimum_ok = bool(case['source_minimum_qualified'])
    ready = bool(ir_ok and minimum_ok)
    status = 'QUALIFIED' if ready else ('REFINEMENT_REQUIRED' if minimum_ok else 'DIAGNOSTIC_COMPLETE')
    arrays_path = out/'ir_arrays.npz'
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = arrays_path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, coordinates_bohr=coords, masses_amu=masses,
            cartesian_modes=vectors, cartesian_derivative_series=ds, steps_bohr=np.asarray(hs),
            direct_mode_derivative_series=direct, normal_coordinate_steps=np.asarray(direct_steps),
            signed_frequencies_cm1=case['frequencies_cm1'], projected_dipole_derivatives=projected)
    os.replace(temporary, arrays_path)
    intensities = quality.prefactor_si()*np.einsum('ij,ij->i', projected, projected)
    with (out/'ir_modes.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['mode_index_one_based','signed_frequency_cm1','signed_frequency_THz',
                  'IR_intensity_km_mol','imaginary','paper_ready','source_id']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mi, (freq, intensity) in enumerate(zip(case['frequencies_cm1'], intensities)):
            writer.writerow(dict(zip(fields, [mi+1,float(freq),float(freq)*0.0299792458,
                float(intensity),bool(freq<0),ready,identifier])))
    result = {**input_spec, 'status':status, 'paper_ready':ready,
        'ir_numerically_qualified':ir_ok, 'source_minimum_qualified':minimum_ok,
        'common_frame':common, 'direct_normal_mode':independent, 'rigid_body':rigid,
        'mass_normalization':norm, 'conversion_factor_relative_disagreement':factor_relative,
        'electronic_points_qualified':all(p['qualified'] for p in ledger),
        'directional_point_count':len(ledger), 'all_vibrational_modes_checked':bool(ledger),
        'source_center_energy_difference_hartree':energy_error,
        'source_center_dipole_difference_au':dipole_error,
        'array_sha256':electronic.sha256(arrays_path),
        'csv_sha256':electronic.sha256(out/'ir_modes.csv'), 'finished_at_utc':electronic.utc(),
        'uncertainty_scope':'Empirical displacement sensitivity; no claim of rigorous error bound, basis or solid-state validation',
        'qualification_does_not_relax_automatically':True}
    write(out/'result.json', result)
    write(out/'COMPLETE.json', {'result_sha256':electronic.sha256(out/'result.json'),
                              'qualification_is_separate':True})
    return result


def process_preview(case):
    """Calculate D while the other branch builds H; never publish an IR claim."""
    configure_precision(case['profile'],0)
    require_same_physical_model(case)
    pool_root=OUTPUT/'derivative_pools'/geometry_model_key(case)/'precision0'
    final=read(pool_root/'precursor_complete.json')
    if final:
        for item in final['derivatives']:
            if electronic.sha256(pool_root/item['path'])!=item['sha256']:
                raise ValueError('Precursor derivative seal mismatch')
        return final
    coords=np.asarray(case['coordinates_bohr'],dtype=float)
    def report(**values):
        progress(molecule=case['molecule'],profile=case['profile'],precursor_id=case['id'],
                 source_minimum_qualified=False,**values)
    report(stage='STATIONARY_PRECURSOR_IR_DERIVATIVES')
    center=core.qualified_dipole(case['symbols'],coords,pool_root/'center',require_density=True)
    if abs(center['energy_hartree']-case['source_center_energy_hartree'])>1e-8 or np.linalg.norm(
        np.asarray(center['dipole_au'])-case['source_center_dipole_au'])>1e-5:
        raise RuntimeError('Stationary precursor electronic state differs')
    density=np.load(center['density_path'],allow_pickle=False)
    derivatives=[]
    for h in (.002,.001,.0005):
        step_path=pool_root/('h'+h.hex())
        core.cartesian_derivative(case['symbols'],coords,step_path,h,density,report)
        digest=electronic.sha256(step_path/'dipole_derivative.npy')
        write(step_path/'COMPLETE.json',{'step_bohr':h,'array_sha256':digest,
            'electronic_precision_level':0,'all_point_electronic_audits_qualified':True})
        derivatives.append({'step_bohr':h,'path':str((step_path/'dipole_derivative.npy').relative_to(pool_root)),
                            'sha256':digest})
    record={'status':'DERIVATIVES_ONLY_AWAITING_FULL_HESSIAN','paper_ready':False,
        'molecule':case['molecule'],'profile':case['profile'],'source':case['source'],
        'source_minimum_qualified':False,'derivatives':derivatives,'at_utc':electronic.utc()}
    write(pool_root/'precursor_complete.json',record)
    return record


def _verify_completed(folder):
    folder=Path(folder)
    receipt=read(folder/'COMPLETE.json')
    value=read(folder/'result.json')
    if not receipt or not value or receipt.get('result_sha256') != electronic.sha256(folder/'result.json'):
        raise ValueError('IR case result seal mismatch: '+str(folder))
    for filename, key in [('ir_arrays.npz','array_sha256'),('ir_modes.csv','csv_sha256')]:
        if electronic.sha256(folder/filename) != value.get(key):
            raise ValueError('IR case scientific artifact seal mismatch: '+str(folder/filename))
    return value


def summarize(descriptors):
    entries=[]
    for descriptor in descriptors:
        identifier = descriptor['id']
        for path in sorted((OUTPUT/'jobs'/identifier).glob('round*/result.json')):
            value=_verify_completed(path.parent)
            entries.append({k:value.get(k) for k in ['molecule','profile','source_id','attempt','status',
                'paper_ready','ir_numerically_qualified','source_minimum_qualified','finished_at_utc']})
    qualified={name:any(x['molecule']==name and x['paper_ready'] for x in entries) for name in TARGETS}
    water_profiles={profile:any(x['molecule']=='water' and x['profile']==profile and x['paper_ready']
        for x in entries) for profile in ('direct-grid5','df-grid5')}
    water_ok=all(water_profiles.values())
    status='COMPLETE' if water_ok and all(qualified.values()) else 'RUNNING_OR_WAITING'
    value={'schema':'independent-ir-summary-v1','status':status,
        'controls_qualified':water_ok,'water_control_profiles':water_profiles,
        'target_qualification':qualified,'cases':entries,
        'source_count':len(descriptors),'updated_at_utc':electronic.utc(),
        'quality_thresholds_end_execution':False,'all_attempted_is_not_complete':True}
    write(OUTPUT/'summary.json',value)
    rows=['# IR 强度独立验证：后台结果','',
          f"状态：{status}。原儿茶酸、毛蕊异黄酮和斯皮诺素分别验收。",'',
          '| 分子 | 方法 | 最新状态 | IR 数值合格 | 上游极小值合格 | 本轮数据资格 |',
          '|---|---|---|---|---|---|']
    for e in entries:
        rows.append(f"| {e['molecule']} | {e['profile']} | {e['status']} | {e['ir_numerically_qualified']} | {e['source_minimum_qualified']} | {e['paper_ready']} |")
    for name,qualified_flag in qualified.items():
        if not qualified_flag:
            rows.append(f'\n{name}：尚未获得本轮已通过全部验收的数据；继续处理或等待上游完整核矩阵。')
    rows+=['','每个已完成 case 保存带符号频率、km/mol 强度 CSV、全部导数和三步长/逐模式核验。',
           '科学不合格不终止后台；900000 CPU 秒为本轮资源授权边界。',
           '用途限于实际合格的孤立分子谐振 IR；不据此确认药材固体 THz 化学峰归属。','']
    destination=OUTPUT/'RESULTS_ZH.md'
    temporary=destination.with_suffix('.tmp')
    temporary.write_text('\n'.join(rows),encoding='utf-8');os.replace(temporary,destination)
    return value


def once(refresh=True):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    frozen=read(OUTPUT/'protocol.json')
    if frozen is None:
        write(OUTPUT/'protocol.json',protocol())
    elif frozen != protocol():
        raise ValueError('Frozen IR protocol changed')
    descriptors=sources.sync_sources(ROOT, OUTPUT, refresh=refresh)
    # Controls first; the source prerequisites govern publication. A bad control
    # remains in the refinement queue without terminating other computations.
    descriptors=sorted(descriptors,key=lambda d:({'water':0,'acid':1,'calycosin':2,'spinosin':3}.get(d['molecule'],4),d['id']))
    errors=[]
    for descriptor in descriptors:
        paths=sorted((OUTPUT/'jobs'/descriptor['id']).glob('round*/result.json'))
        results=[_verify_completed(p.parent) for p in paths]
        if any(r.get('paper_ready') or r.get('status')=='DIAGNOSTIC_COMPLETE' for r in results):
            continue
        attempt=len(results)
        try:
            process_case(descriptor,attempt)
        except Exception as error:
            record={'molecule':descriptor['molecule'],'source_id':descriptor['id'],
                'attempt':attempt,'type':type(error).__name__,'error':str(error),
                'recorded_at_utc':electronic.utc(),'traceback':traceback.format_exc()}
            errors.append(record)
            write(OUTPUT/'jobs'/descriptor['id']/f'round{attempt:05d}'/'retry_error.json',record)
    preview_records=sources.preview_centers(ROOT,OUTPUT)
    latest={}
    for candidate in preview_records:
        key=(candidate['molecule'],candidate['profile'])
        if key not in latest or candidate['source']['optimization_path']>latest[key]['source']['optimization_path']:
            latest[key]=candidate
    preview_states=[]
    for candidate in latest.values():
        # Completed mode sources already own their derivative work.
        if any(d['molecule']==candidate['molecule'] and d['profile']==candidate['profile'] and np.array_equal(
            sources.load_source(d)['coordinates_bohr'],np.asarray(candidate['coordinates_bohr'])) for d in descriptors):
            continue
        try:
            preview_states.append(process_preview(candidate))
        except Exception as error:
            preview_states.append({'molecule':candidate['molecule'],'status':'RETRY_PENDING','error':str(error)})
    write(OUTPUT/'precursor_progress.json',{'cases':preview_states,'updated_at_utc':electronic.utc(),
                                         'precursors_never_grant_paper_qualification':True})
    result=summarize(descriptors)
    progress(stage='COMPLETE' if result['status']=='COMPLETE' else 'WAITING_OR_REFINING',
             source_count=len(descriptors),errors_in_last_scan=errors)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--no-refresh',action='store_true')
    parser.add_argument('--summary',action='store_true')
    args=parser.parse_args()
    if args.summary:
        print(json.dumps(read(OUTPUT/'summary.json',{}),indent=2,ensure_ascii=False))
    else:
        print(json.dumps(once(refresh=not args.no_refresh),indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
