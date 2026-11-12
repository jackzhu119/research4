#!/usr/bin/env python3
"""Budget-supervised recovery, with explicit scientific states and raw evidence.

Historical runs are immutable. RI-J results are explicitly a different numerical
model; no approximate result is relabelled as the earlier direct calculation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

import pyscf_publication_core_20261111 as core
import pyscf_repair_modes_20261111 as modes
import pyscf_publication_quality_20261111 as quality

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'reports/pyscf_publication_20261111'
INPUTS = ROOT / 'data/research/pyscf_repair_20261111/inputs'
STEPS = (0.002, 0.001)
MOLECULES = ('acid', 'calycosin', 'spinosin')
PROFILES = ('df-grid5', 'df-grid7', 'direct-grid7')
MAX_ROUNDS = 6
AMPLITUDES = (.01, .03, .08, .2)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    core.atomic_json(Path(path), value)


def digest_array(value):
    value = np.ascontiguousarray(value, dtype='<f8')
    return hashlib.sha256(str(value.shape).encode()+value.tobytes()).hexdigest()


def progress(**values):
    value = read(OUTPUT/'progress.json', {})
    value['checkpoint_token'] = int(value.get('checkpoint_token', 0)) + 1
    value.update(values, heartbeat_at_utc=utc())
    write(OUTPUT/'progress.json', value)


def frozen_protocol():
    files = sorted(p for p in INPUTS.iterdir() if p.is_file())
    scripts = [ROOT/'scripts'/name for name in (
        'pyscf_publication_core_20261111.py', 'pyscf_publication_quality_20261111.py',
        'run_pyscf_publication_20261111.py', 'pyscf_repair_modes_20261111.py')]
    value = {
        'schema': 'research4-publication-recovery-v1',
        'source_branch': 'research/pyscf-frequency-repair-20261111',
        'source_commit': 'e8b53b593d97381654372dcfdfd6a1a456c3f342',
        'molecules': list(MOLECULES), 'profiles': list(PROFILES),
        'steps_bohr': list(STEPS), 'negative_mode_amplitudes_bohr': list(AMPLITUDES),
        'maximum_geometry_rounds_per_profile': MAX_ROUNDS,
        'profile_escalation': 'Numerical failure is retained and escalates to denser grid or exact-J cross-check; it is never a qualified completion.',
        'density_fitting': 'Explicit RI-J with def2-universal-jkfit, auxiliary-basis and quadrature response in gradients; direct controls before target molecules.',
        'inputs': {str(p.relative_to(ROOT)): core.sha256(p) for p in files},
        'code': {str(p.relative_to(ROOT)): core.sha256(p) for p in scripts},
        'versions': {k: importlib.metadata.version(k) for k in ('pyscf','geometric','numpy','scipy','h5py')},
        'scope': 'Gas-phase local minima and harmonic/IR predictions only; no global minimum or herb-solid chemical assignment.',
        'budget_authorization_cpu_seconds': 900000,
        'budget_interpretation': 'User 2026-11-11: 授权你900000cpu时; interpreted in the ongoing CPU-second convention, i.e. 250 CPU hours.',
        'original_results_modified': False,
    }
    old = read(OUTPUT/'frozen_protocol.json')
    if old is not None and old != value:
        raise ValueError('Integrity error: frozen publication scientific inputs changed')
    if old is None:
        write(OUTPUT/'frozen_protocol.json', value)
    return value


def inputs(name):
    if name == 'water':
        return ['O','H','H'], np.array([[0.,0.,0.],[1.43,0.,1.11],[-1.43,0.,1.11]]), None
    symbols = read(INPUTS/f'{name}_symbols.json')
    coords = np.load(INPUTS/f'{name}_coordinates_bohr.npy', allow_pickle=False)
    density = INPUTS/f'{name}_density.npy'
    return symbols, coords, np.load(density, allow_pickle=False) if density.exists() else None


def dm(point):
    return np.load(point['density_path'], allow_pickle=False)


def sealed(out):
    receipt = read(Path(out)/'COMPLETE.json')
    if receipt is None:
        return None
    path = Path(out)/'result.json'
    if not path.is_file() or receipt.get('result_sha256') != core.sha256(path):
        raise ValueError('Integrity error: publication result seal mismatch')
    return read(path)


def seal(out, value):
    write(Path(out)/'result.json', value)
    write(Path(out)/'COMPLETE.json', {'result_sha256': core.sha256(Path(out)/'result.json'),
          'paper_qualification_is_separate': True, 'sealed_at_utc': utc()})
    return value


def identity_check(symbols, reference, candidate):
    """Conservative atom-order, connectivity and tetrahedral-parity guard.

    This checks preservation relative to the source geometry, not independent
    experimental authentication of a stereoisomer or a bond-order assignment.
    """
    from pyscf.data import elements
    ref, xyz = np.asarray(reference), np.asarray(candidate)
    if xyz.shape != ref.shape or not np.isfinite(xyz).all():
        return {'passed': False, 'reason': 'incomplete coordinates'}
    # Covalent radii in Angstrom; convert to Bohr. Only source elements needed.
    radii = {'H':.31,'C':.76,'N':.71,'O':.66,'S':1.05,'F':.57,'P':1.07}
    radius = np.array([radii[s]/.529177210903 for s in symbols])
    d0 = np.linalg.norm(ref[:,None,:]-ref[None,:,:],axis=2)
    d1 = np.linalg.norm(xyz[:,None,:]-xyz[None,:,:],axis=2)
    eye = np.eye(len(symbols),dtype=bool)
    bonds = (d0 < 1.25*(radius[:,None]+radius[None,:])) & ~eye
    candidate_bonds = (d1 < 1.25*(radius[:,None]+radius[None,:])) & ~eye
    graph_preserved = bool(np.array_equal(candidate_bonds, bonds))
    heavy = np.array([s != 'H' for s in symbols])
    hydrogen_owners = []
    if np.any(heavy):
        heavy_indices = np.flatnonzero(heavy)
        for index, symbol in enumerate(symbols):
            if symbol == 'H':
                old_owner = int(heavy_indices[np.argmin(d0[index, heavy])])
                new_owner = int(heavy_indices[np.argmin(d1[index, heavy])])
                hydrogen_owners.append({'hydrogen_zero_based': index,
                    'source_heavy_atom_zero_based': old_owner,
                    'candidate_heavy_atom_zero_based': new_owner,
                    'preserved': old_owner == new_owner})
    upper = np.triu(bonds,1)
    ratios = d1[upper]/d0[upper]
    bond_ok = bool(len(ratios) and np.all((ratios>.65)&(ratios<1.40)))
    new_contact = np.triu((~bonds)&(~eye)&(d1<.80*(radius[:,None]+radius[None,:])),1)
    parity = []
    for center, symbol in enumerate(symbols):
        neighbors = np.flatnonzero(bonds[center])
        if symbol != 'C' or len(neighbors) != 4:
            continue
        def volume(c):
            points = c[neighbors]
            return float(np.linalg.det(points[:3]-points[3]))
        before, after = volume(ref), volume(xyz)
        if abs(before) > .1:
            parity.append({'atom_zero_based':center, 'source_signed_volume':before,
                           'candidate_signed_volume':after, 'preserved':bool(before*after>0)})
    return {'passed':bool(bond_ok and graph_preserved and not np.any(new_contact)
                          and all(p['preserved'] for p in parity)
                          and all(p['preserved'] for p in hydrogen_owners)),
            'source_neighbor_graph_preserved':graph_preserved,
            'hydrogen_ownership':hydrogen_owners,
            'source_bond_count':int(np.count_nonzero(upper)),
            'bond_length_ratio_min':float(np.min(ratios)) if len(ratios) else None,
            'bond_length_ratio_max':float(np.max(ratios)) if len(ratios) else None,
            'new_unphysical_close_contacts':int(np.count_nonzero(new_contact)),
            'tetrahedral_parity':parity,
            'scope':'Source-relative geometric preservation, not independent chemical identity validation.'}


def optimize(symbols, coords, density, out):
    # Segments are restart intervals; the independent supervisor owns CPU limits.
    while True:
        value = core.optimize_geometry(symbols, coords, out, dm0=density)
        if value['minimum_stationarity_passed'] and value['optimizer_converged']:
            return value
        progress(stage='OPTIMIZATION_CONTINUATION', optimization=str(out))


def assemble(points, step):
    pg = np.array([p['gradient'] for p in points[1]])
    mg = np.array([p['gradient'] for p in points[-1]])
    pd = np.array([p['dipole_au'] for p in points[1]])
    md = np.array([p['dipole_au'] for p in points[-1]])
    n = pg.shape[0]
    if pg.size != n*n or mg.shape != pg.shape or pd.shape != (n,3) or md.shape != (n,3):
        raise ValueError('Integrity error: incomplete Cartesian derivatives')
    return (pg.reshape(n,n)-mg.reshape(n,n)).T/(2*step), (pd-md).T/(2*step)


def displacement_set(name, symbols, coords, density, out):
    matrices, dipoles, ledger = [], [], []
    for hi, h in enumerate(STEPS):
        points = {1:[], -1:[]}
        for coordinate in range(coords.size):
            for sign in (1,-1):
                path = out/f'h{hi}'/f'c{coordinate:03d}-{"plus" if sign==1 else "minus"}'
                displaced = np.array(coords,copy=True)
                displaced.flat[coordinate] += sign*h
                progress(molecule=name, stage='FULL_NUMERICAL_DERIVATIVES', step_bohr=h,
                         coordinate=coordinate, sign=sign,
                         molecule_displacements_completed=hi*coords.size*2+coordinate*2+(sign==-1),
                         molecule_displacements_planned=4*coords.size)
                point = None
                for attempt in range(3):
                    point = core.calculate_point(symbols,displaced,path,dm0=density)
                    if point['qualified']:
                        break
                if not point['qualified']:
                    raise RuntimeError('Electronic displacement remains unqualified: '+str(path))
                points[sign].append(point)
                audit = point['electronic_audit']
                ledger.append({'qualified':point['qualified'], 'native_converged':point['native_converged'],
                    'fresh_orbital_gradient_norm':audit['fresh_orbital_gradient_norm'],
                    'energy_recheck_difference_hartree':audit['fresh_energy_difference_hartree'],
                    'electron_count_error':audit['electron_count_error']})
                progress(molecule_displacements_completed=len(ledger))
        matrix, dipole = assemble(points,h)
        core.atomic_array(out/f'hessian_h{hi}.npy',matrix)
        core.atomic_array(out/f'dipole_derivative_h{hi}.npy',dipole)
        matrices.append(matrix); dipoles.append(dipole)
    return matrices,dipoles,ledger


def center_quality(point, stability):
    audit = point['electronic_audit']
    return {'qualified':point['qualified'], 'native_converged':point['native_converged'],
            'fresh_orbital_gradient_norm':audit['fresh_orbital_gradient_norm'],
            'energy_recheck_difference_hartree':audit['fresh_energy_difference_hartree'],
            'electron_count_error':audit['electron_count_error'],
            'nuclear_gradient_max':point['nuclear_gradient_max'],
            'nuclear_gradient_rms':point['nuclear_gradient_rms'],
            'orbital_stability_passed':stability['tested_electronic_stability_passed']}


def stationary_case(name, profile, initial_coords, initial_density, out):
    old = sealed(out)
    if old is not None:
        return old
    core.configure(profile)
    symbols, reference, _ = inputs(name)
    progress(molecule=name, profile=profile, stage='STRICT_GEOMETRY_OPTIMIZATION', case=str(out))
    opt = optimize(symbols,initial_coords,initial_density,out/'optimize')
    coords = np.array(opt['coordinates_bohr'])
    identity = identity_check(symbols,reference,coords)
    if not identity['passed']:
        return seal(out,{'status':'IDENTITY_GUARD_FAILED','paper_ready':False,'identity':identity})
    point = opt['final_point']
    for electronic_round in range(4):
        stability = core.check_electronic_stability(symbols,coords,
                    out/f'electronic-stability-{electronic_round:02d}',dm0=dm(opt))
        if not stability['tested_electronic_stability_passed']:
            return seal(out,{'status':'ELECTRONIC_STATE_UNRESOLVED','paper_ready':False,
                             'identity':identity,'electronic_stability':stability})
        if stability.get('supported_for_primary_hessian'):
            break
        if electronic_round == 3:
            return seal(out,{'status':'ELECTRONIC_REPAIR_NOT_CLOSED','paper_ready':False,
                             'identity':identity,'electronic_stability':stability})
        # Every altered electronic state must close a fresh nuclear optimization
        # and a stable-state test before its center can qualify any Hessian.
        repaired = np.load(stability['density_path'],allow_pickle=False)
        opt = optimize(symbols,coords,repaired,out/f'reoptimized-stable-state-{electronic_round:02d}')
        coords = np.array(opt['coordinates_bohr']); point = opt['final_point']
        identity = identity_check(symbols,reference,coords)
        if not identity['passed']:
            return seal(out,{'status':'IDENTITY_GUARD_FAILED','paper_ready':False,'identity':identity})
    matrices,dipoles,ledger = displacement_set(name,symbols,coords,dm(opt),out/'finite-differences')
    analysis = modes.analyze_modes(symbols,coords,matrices,dipoles,out/'modes',
              center_quality=center_quality(point,stability),point_quality=ledger,steps_bohr=STEPS)
    gate = quality.audit_publication(out/'modes',out/'finite-differences',point,
                electronic_stability_passed=stability['tested_electronic_stability_passed'])
    ready = gate['paper_ready_molecular_harmonic_data']
    value = {'status':'QUALIFIED' if ready else 'REFINEMENT_REQUIRED',
             'molecule':name,'profile':profile,'method':core.settings(),
             'paper_ready':bool(ready),
             'center_point_path':str(Path(point['gradient_path']).parent/'result.json'),
             'optimization_path':str(Path(opt['array_refs']['coordinates_bohr']['path']).parent/'result.json'),
             'coordinates_bohr':coords.tolist(),'energy_hartree':point['energy_hartree'],
             'density_path':opt['density_path'],'identity':identity,'electronic_stability':stability,
             'analysis':analysis,'publication_gate':gate,'finished_at_utc':utc(),
             'scope':'Isolated-molecule local harmonic/IR result at the explicitly recorded numerical model.'}
    return seal(out,value)


def negative_branches(name,profile,case,out):
    old = read(out/'selection.json')
    if old is not None:
        return old
    coords = np.array(case['coordinates_bohr']); symbols,reference,_ = inputs(name)
    final = case['analysis']['steps'][-1]
    first = case['analysis']['steps'][0]
    error = case['analysis']['step_comparison']['projected_spectral_step_difference_hartree_per_amu_bohr2']
    low = final['lowest_signed_eigenvalue_hartree_per_amu_bohr2']
    if not (first['imaginary_count']>0 and final['imaginary_count']>0 and low < -error):
        return {'status':'NO_SUPPORTED_NEGATIVE_MODE','selected':None}
    core.configure(profile)
    arrays = np.load(Path(case['optimization_path']).parents[1]/'modes/analysis_arrays.npz',allow_pickle=False)
    q = arrays['step_1_cartesian_unit_display_modes'][:,0].reshape(coords.shape)
    q /= np.max(np.abs(q))
    trials=[]; optimized=[]
    for sign in (1,-1):
        signed=[]
        for amplitude in AMPLITUDES:
            xyz=coords+sign*amplitude*q
            guard=identity_check(symbols,reference,xyz)
            if not guard['passed']:
                continue
            try:
                p=core.calculate_point(symbols,xyz,out/f'sign{sign:+d}/a{amplitude}/point',dm0=dm(case))
            except Exception as error:
                if any(t in str(error).lower() for t in ('integrity error','fingerprint changed','array hash changed')):
                    raise
                trials.append({'sign':sign,'amplitude_bohr':amplitude,'qualified':False,
                               'error_type':type(error).__name__,'error':str(error)})
                continue
            record={'sign':sign,'amplitude_bohr':amplitude,'qualified':p['qualified'],
                    'energy_hartree':p['energy_hartree'],'energy_change_hartree':p['energy_hartree']-case['energy_hartree']}
            trials.append(record)
            if p['qualified']:
                signed.append((p,amplitude))
        if not signed:
            continue
        seed,amplitude=min(signed,key=lambda item:item[0]['energy_hartree'])
        # Both signs receive unconstrained relaxation even if a finite trial was
        # not downhill. Anharmonicity at one amplitude cannot reject a basin.
        try:
            opt=optimize(symbols,np.array(seed['coordinates_bohr']),dm(seed),out/f'sign{sign:+d}/relax')
        except Exception as error:
            if any(t in str(error).lower() for t in ('integrity error','fingerprint changed','array hash changed')):
                raise
            optimized.append({'sign':sign,'qualified':False,'error_type':type(error).__name__,
                              'error':str(error)})
            continue
        guard=identity_check(symbols,reference,np.array(opt['coordinates_bohr']))
        optimized.append({'sign':sign,'amplitude_bohr':amplitude,
              'qualified':bool(opt['minimum_stationarity_passed'] and guard['passed']),
              'identity':guard,'coordinates_bohr':opt['coordinates_bohr'],'density_path':opt['density_path'],
              'energy_hartree':opt['final_point']['energy_hartree']})
    candidates=[p for p in optimized if p['qualified'] and p['energy_hartree']<case['energy_hartree']-1e-9]
    best=min(candidates,key=lambda p:p['energy_hartree']) if candidates else None
    value={'status':'BOTH_SIGNS_RELAXED','trials':trials,'relaxed_branches':optimized,'selected':best,
           'scope':'Finite local basin search only; not a global conformer search.'}
    write(out/'selection.json',value)
    return value


def direct_soft_checks(name,profile,case,out):
    """Validate fitting and quadrature sensitivity without duplicating full H."""
    old=read(out/'result.json')
    if old is not None:
        return old
    coords=np.array(case['coordinates_bohr']);symbols,_,_=inputs(name)
    case_dir=Path(case['optimization_path']).parents[1]
    arrays=np.load(case_dir/'modes/analysis_arrays.npz',allow_pickle=False)
    directions=arrays['step_1_cartesian_unit_display_modes'][:,:min(3,coords.size-6)]
    rows=[]
    check_profiles=['direct-grid5','df-grid7'] if profile=='df-grid5' else ['direct-grid5']
    if profile=='df-grid7':
        check_profiles=['direct-grid7']
    if profile.startswith('direct-'):
        check_profiles=['df-grid7']
    for check_profile in check_profiles:
        core.configure(check_profile)
        center=core.calculate_point(symbols,coords,out/check_profile/'center')
        for index in range(directions.shape[1]):
            q=directions[:,index].reshape(coords.shape)
            p=core.calculate_point(symbols,coords+.002*q,out/check_profile/f'm{index}/plus',dm0=dm(center))
            m=core.calculate_point(symbols,coords-.002*q,out/check_profile/f'm{index}/minus',dm0=dm(center))
            fd=(np.array(p['gradient'])-np.array(m['gradient']))/.004
            curvature=float(np.sum(q*fd))
            base_h=arrays['step_1_raw_hessian']; base=float(q.ravel()@base_h@q.ravel())
            dipole=(np.array(p['dipole_au'])-np.array(m['dipole_au']))/.004
            base_d=arrays['step_1_dipole_derivative_au']@q.ravel()
            rows.append({'profile':check_profile,'mode_zero_based':index,
                'all_points_qualified':all(z['qualified'] for z in (center,p,m)),
                'gradient_curvature':curvature,'base_gradient_curvature':base,
                'relative_curvature_change':abs(curvature-base)/max(abs(base),1e-12),
                'energy_curvature':(p['energy_hartree']+m['energy_hartree']-2*center['energy_hartree'])/.002**2,
                'dipole_derivative_absolute_change':float(np.linalg.norm(dipole-base_d)),
                'reference_center_gradient_max':center['nuclear_gradient_max']})
    # A local directional test is only sensitivity evidence. It cannot certify
    # the full exact-J Hessian, a denser-grid stationary point, or a global basin.
    passed=all(r['all_points_qualified'] and r['gradient_curvature']>0 and
               r['relative_curvature_change']<=.10 for r in rows)
    value={'status':'DONE','directional_sensitivity_passed':bool(rows and passed),
           'maximum_allowed_relative_curvature_change':.10,'rows':rows,
           'scope':'Three soft directions only, not a full independent minimum certification.',
           'exact_J_method_equivalence_claimed':False}
    core.configure(profile)
    write(out/'result.json',value)
    return value


def water_control():
    out=OUTPUT/'controls/water-final'
    previous=sealed(out)
    if previous is not None:
        if not previous['control_passed']:
            raise ValueError('Scientific water control failed; target queue remains blocked')
        return previous
    symbols,coords,density=inputs('water')
    cases={}
    for profile in ('direct-grid5','df-grid5'):
        cases[profile]=stationary_case('water',profile,coords,density,out/profile)
    direct, fitted=cases['direct-grid5'],cases['df-grid5']
    passed=bool(direct.get('paper_ready') and fitted.get('paper_ready'))
    difference=None
    if passed:
        da=direct['analysis']['steps'][-1];fa=fitted['analysis']['steps'][-1]
        difference=float(np.max(np.abs(np.array(da['signed_frequencies_cm1'])-fa['signed_frequencies_cm1'])))
        passed=passed and difference<=2.0 and abs(direct['energy_hartree']-fitted['energy_hartree'])<=1e-3
    value={'status':'DONE','control_passed':bool(passed),'cases':cases,
           'max_frequency_difference_cm1':difference,'frequency_difference_tolerance_cm1':2.0,
           'energy_difference_tolerance_hartree':1e-3,'finished_at_utc':utc()}
    seal(out,value)
    if not passed:
        raise ValueError('Scientific water control failed; inspect actual evidence before target calculations')
    return value


def molecule_recovery(name):
    root=OUTPUT/'jobs'/name
    previous=sealed(root)
    if previous is not None:
        return previous
    _,reference,density=inputs(name)
    cases=[];selected=None
    for profile in PROFILES:
        coords=np.array(reference);guess=density
        for round_id in range(MAX_ROUNDS):
            out=root/profile/f'round{round_id:03d}'
            try:
                case=stationary_case(name,profile,coords,guess,out)
            except Exception as error:
                if 'Integrity error' in str(error) or 'fingerprint changed' in str(error):
                    raise
                write(out/'failure.json',{'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'utc':utc()})
                cases.append({'profile':profile,'round':round_id,'path':str(out),'status':'COMPUTATION_FAILED','error':str(error)})
                break
            cases.append({'profile':profile,'round':round_id,'path':str(out),'status':case['status'],'paper_ready':case.get('paper_ready',False)})
            if case.get('paper_ready'):
                try:
                    sensitivity=direct_soft_checks(name,profile,case,out/'sensitivity')
                except Exception as error:
                    if any(t in str(error).lower() for t in ('integrity error','fingerprint changed','array hash changed')):
                        raise
                    cases[-1].update(status='SENSITIVITY_CONTROL_FAILED',error=str(error))
                    write(out/'sensitivity/failure.json',{'type':type(error).__name__,
                          'error':str(error),'traceback':traceback.format_exc(),'utc':utc()})
                    break
                if sensitivity['directional_sensitivity_passed']:
                    selected={'profile':profile,'round':round_id,'path':str(out),'case_result_sha256':core.sha256(out/'result.json'),
                              'sensitivity':sensitivity}
                    break
                # Positive but sensitive results are retained; next profile
                # performs complete derivatives at a refined numerical model.
                cases[-1]['status']='REFINE_GRID_OR_FITTING_REQUIRED'
                break
            if 'analysis' not in case:
                break
            follow=negative_branches(name,profile,case,out/'negative-mode-recovery')
            if follow.get('selected') is None:
                break
            coords=np.array(follow['selected']['coordinates_bohr']);guess=dm(follow['selected'])
        if selected:
            break
    return seal(root,{'status':'QUALIFIED_LOCAL_HARMONIC_DATA' if selected else 'UNRESOLVED_AFTER_RECORDED_REFINEMENTS',
                     'molecule':name,'paper_ready':selected is not None,'selected':selected,'attempts':cases,
                     'finished_at_utc':utc(),'scope':'Local isolated-molecule evidence only; no solid THz assignment.'})


def verify():
    checks=[]
    for file in sorted(OUTPUT.rglob('COMPLETE.json')):
        value=sealed(file.parent)
        if value is None:
            raise ValueError('Integrity error: unreadable result seal')
        checks.append(str(file.relative_to(OUTPUT)))
    for path in OUTPUT.glob('**/modes/result.json'):
        modes.verify_analysis(path.parent)
    # Re-audit original point coordinates, complete matrices and IR gates.
    for path in OUTPUT.glob('**/modes/result.json'):
        case_dir=path.parent.parent
        case=read(case_dir/'result.json',{})
        if case.get('center_point_path'):
            point=read(case['center_point_path'])
            gate=quality.audit_publication(path.parent,case_dir/'finite-differences',point,
                       electronic_stability_passed=case['electronic_stability']['tested_electronic_stability_passed'])
            if bool(gate['paper_ready_molecular_harmonic_data']) != bool(case.get('paper_ready')):
                raise ValueError('Integrity error: replayed publication gate differs')
    result={'status':'VERIFIED_AVAILABLE_ARTIFACTS','sealed_results':len(checks),
            'all_target_molecules_completed':all((OUTPUT/'jobs'/n/'COMPLETE.json').exists() for n in MOLECULES),
            'at_utc':utc(),'completion_does_not_imply_qualification':True}
    write(OUTPUT/'verification.json',result)
    return result


def summarize():
    rows=[]
    for name in MOLECULES:
        value=read(OUTPUT/'jobs'/name/'result.json',{})
        selected=value.get('selected')
        row={'molecule':name,'status':value.get('status','PENDING_OR_CHECKPOINTED'),
             'paper_ready':value.get('paper_ready',False),'selected_profile':selected.get('profile') if selected else None}
        if selected:
            case=read(Path(selected['path'])/'result.json')
            row.update(mode_count=case['analysis']['steps'][-1]['vibrational_mode_count'],
                       imaginary_count=case['analysis']['steps'][-1]['imaginary_count'],
                       lowest_cm1=case['analysis']['steps'][-1]['lowest_signed_frequency_cm1'])
        rows.append(row)
    result={'schema':'publication-recovery-summary-v1','updated_at_utc':utc(),
            'molecules':rows,'qualified_count':sum(r['paper_ready'] for r in rows),
            'all_molecules_attempted':all(r['status']!='PENDING_OR_CHECKPOINTED' for r in rows),
            'scope':'No measured chemical composition, global conformer optimum, solid-state THz assignment or ML gain is established.'}
    write(OUTPUT/'summary.json',result)
    lines=['# PySCF 论文数据修复结果','',
           '计算执行状态与论文数据资格分别报告。原始实验、负频率和所有失败记录保留。',
           'RI-J 近似、辅助基组、网格及步长按实际方法记录，不改称旧 direct 方法。','',
           '|分子|执行状态|分子谐振数据资格|实际方法|','|---|---|---|---|']
    for r in rows:
        lines.append(f"|{r['molecule']}|{r['status']}|{'通过' if r['paper_ready'] else '尚未通过'}|{r['selected_profile'] or '待核验'}|")
    lines+=['','可用范围：通过核验的孤立分子局部极小值及谐振/IR 预测。',
            '不能据此认定全局最低构象、药材固体 THz 化学峰归属或成分辅助分类收益。',
            '云环境休眠或销毁会停止本地后台；逐点检查点可在恢复环境后续跑。','']
    (OUTPUT/'RESULTS_ZH.md').write_text('\n'.join(lines))
    return result


def run():
    frozen_protocol()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'.runner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        water_control()
        for name in MOLECULES:
            molecule_recovery(name)
            summarize()
        verify()
        progress(stage='ALL_PLANNED_REFINEMENTS_FINISHED',status='DONE')
        summarize()
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    for name in ('resume','pilot','verify','summarize','prepare'):
        group.add_argument('--'+name,action='store_true')
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    try:
        if args.prepare:
            result=frozen_protocol()
        elif args.summarize:
            result=summarize()
        elif args.verify:
            frozen_protocol();result=verify()
        elif args.pilot:
            frozen_protocol();result=water_control()
        else:
            return run()
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except BlockingIOError:
        print('Runner lock is held by another process',file=sys.stderr)
        return 42
    except Exception as error:
        write(OUTPUT/'last_error.json',{'type':type(error).__name__,'error':str(error),
              'traceback':traceback.format_exc(),'utc':utc()})
        print(str(error),file=sys.stderr)
        return 42 if any(t in str(error).lower() for t in ('integrity error','fingerprint changed','scientific inputs changed','array hash changed')) else 2


if __name__=='__main__':
    raise SystemExit(main())
