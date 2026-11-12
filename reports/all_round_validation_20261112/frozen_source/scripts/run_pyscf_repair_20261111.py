#!/usr/bin/env python3
"""A resumable, source-bound repair of the historical imaginary-frequency route.

Quality flags qualify evidence; they do not terminate the experiment queue.
Each displacement is a real native electronic calculation, never a filled row.
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
import sys
import traceback

import numpy as np

import pyscf_repair_core_20261111 as core

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'reports/pyscf_repair_20261111'
INPUTS = ROOT / 'data/research/pyscf_repair_20261111/inputs'
HISTORY = INPUTS.parent / 'history_controls'
MOLECULES = ('water', 'acid', 'calycosin', 'spinosin')
STEPS = (.001, .0005)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return core.sha256(path)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write(path, value):
    core.atomic_json(path, value)


def progress(**values):
    old = read(OUTPUT / 'progress.json', {})
    old.update(values, heartbeat_at_utc=utc())
    write(OUTPUT / 'progress.json', old)


class IntegrityError(ValueError):
    """A changed input or saved artifact, distinct from scientific failure."""


def is_integrity_error(error):
    return isinstance(error, IntegrityError) or any(fragment in str(error).lower() for fragment in
        ('fingerprint changed', 'scientific inputs changed', 'array hash changed',
         'seal mismatch', 'frozen scientific', 'analysis inputs changed',
         'artifact hash mismatch', 'saved analysis array'))


def completed_result(out):
    """Reuse a sealed completion only after checking its result bytes."""
    out = Path(out)
    seal = read(out / 'COMPLETE.json')
    if seal is None:
        return None
    result_path = out / 'result.json'
    if not result_path.is_file() or seal.get('result_sha256') != sha(result_path):
        raise IntegrityError('Completed result seal mismatch: ' + str(out))
    return read(result_path)


def optional_diagnostic_error(out, error):
    """Keep a failed comparison visible without blocking primary derivatives."""
    if is_integrity_error(error):
        raise error
    value = {'status': 'DIAGNOSTIC_ERROR', 'error_type': type(error).__name__,
             'error': str(error), 'at_utc': utc(),
             'primary_numerical_calculation_blocked': False}
    write(Path(out) / 'diagnostic_error.json', value)
    return value


def build_protocol():
    files = [p for d in (INPUTS, HISTORY) for p in d.rglob('*') if p.is_file()]
    scripts = [ROOT / 'scripts' / name for name in (
        'pyscf_repair_core_20261111.py', 'pyscf_repair_modes_20261111.py',
        'run_pyscf_repair_20261111.py')]
    return {
        'schema': 'research4-pyscf-repair-frozen-v1',
        'molecules': list(MOLECULES),
        'settings': core.settings(5),
        'optimization_settings': core.OPTIMIZATION_SETTINGS,
        'finite_difference_steps_bohr': list(STEPS),
        'planned_primary_displaced_points': {k: 12*n for k, n in
            [('water', 3), ('acid', 17), ('calycosin', 33), ('spinosin', 75)]},
        'planned_primary_displaced_points_total': 1536,
        'inputs': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
        'code': {str(p.relative_to(ROOT)): sha(p) for p in scripts},
        'versions': {k: importlib.metadata.version(k) for k in
                     ('pyscf', 'geometric', 'numpy', 'scipy', 'h5py')},
        'quality_flags_stop_queue': False,
        'cpu_time_score_retry_cutoffs': None,
        'electronic_stability': 'Native internal, spin breaking and complex orbital checks, explicitly reported.',
        'imaginary_followup': 'A negative lowest mode at both steps, with the fine-step negative eigenvalue exceeding the empirical spectral step difference in magnitude, generates both signed candidates at a stationary electronically qualified center. A new optimization round is queued only if a qualified candidate is at least 1e-9 Eh below the center. Ambiguous curvature and non-downhill trials are retained without false saddle following. Historical and all previous rounds retained.',
        'publication_scope': 'Isolated-molecule PBE/def2-SVP/noD3 computation; neither solid-state peak identification nor measured sample chemistry.',
        'old_software_confound': 'Some old analytic records used PySCF2.14.0, while recent full numerical records and this repair use2.13.0; no unique attribution to a PySCF version or one implementation defect.',
    }


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol()
    old = read(OUTPUT / 'frozen_protocol.json')
    if old is not None and old != protocol:
        raise IntegrityError('Frozen scientific protocol or inputs changed')
    if old is None:
        write(OUTPUT / 'frozen_protocol.json', protocol)
    return protocol


def assemble_fd(plus_gradients, minus_gradients, plus_dipoles, minus_dipoles, step):
    """Assemble columns d(g)/d(R_j), with no missing-displacement substitution."""
    pg, mg = np.asarray(plus_gradients, dtype=float), np.asarray(minus_gradients, dtype=float)
    pd, md = np.asarray(plus_dipoles, dtype=float), np.asarray(minus_dipoles, dtype=float)
    if not np.isfinite(step) or step <= 0:
        raise ValueError('Positive finite difference step required')
    if pg.shape != mg.shape or pg.ndim not in (2, 3):
        raise ValueError('Paired full gradient arrays required')
    d = pg.shape[0]
    if d % 3 or pg.size != d*d or pd.shape != (d, 3) or md.shape != (d, 3):
        raise ValueError('Every Cartesian displacement and dipole must be present')
    if not all(np.isfinite(a).all() for a in (pg, mg, pd, md)):
        raise ValueError('Nonfinite or missing displacement; never fill a Hessian')
    return (pg.reshape(d, d)-mg.reshape(d, d)).T/(2*step), (pd-md).T/(2*step)


def molecule_inputs(name):
    if name == 'water':
        return ['O', 'H', 'H'], np.array([[0,0,0],[1.43,0,1.11],[-1.43,0,1.11]]), None
    symbols = read(INPUTS / f'{name}_symbols.json')
    coords = np.load(INPUTS / f'{name}_coordinates_bohr.npy', allow_pickle=False)
    dm_path = INPUTS / f'{name}_density.npy'
    return symbols, coords, np.load(dm_path, allow_pickle=False) if dm_path.exists() else None


def density(point):
    return np.load(point['density_path'], allow_pickle=False)


def point_quality(point):
    audit = point.get('electronic_audit', {})
    return {
        'qualified': point.get('qualified', False),
        'native_converged': point.get('native_converged', False),
        'fresh_orbital_gradient_norm': audit.get('fresh_orbital_gradient_norm'),
        'energy_recheck_difference_hartree': audit.get('fresh_energy_difference_hartree'),
        'electron_count_error': audit.get('electron_count_error'),
        'nuclear_gradient_max': point.get('nuclear_gradient_max'),
        'nuclear_gradient_rms': point.get('nuclear_gradient_rms'),
    }


def complete_displacements(name, symbols, coords, dm0, out):
    hessians, dipoles, qualities = [], [], []
    for hi, step in enumerate(STEPS):
        points = {1: [], -1: []}
        for column in range(coords.size):
            for sign in (1, -1):
                path = out / f'h{hi}' / f'c{column:03d}-{"plus" if sign == 1 else "minus"}'
                progress(molecule=name, stage='full_numerical_hessian_and_dipole',
                         step_bohr=step, coordinate=column, sign=sign,
                         molecule_displacements_completed=hi*coords.size*2+column*2+(sign == -1),
                         molecule_displacements_planned=coords.size*4)
                displaced = np.array(coords, copy=True)
                displaced.flat[column] += sign*step
                point = core.calculate_point(symbols, displaced, path, grid_level=5, dm0=dm0)
                points[sign].append(point)
                qualities.append(point_quality(point))
                progress(molecule_displacements_completed=hi*coords.size*2+column*2+(sign == -1)+1)
        h, dip = assemble_fd([p['gradient'] for p in points[1]], [p['gradient'] for p in points[-1]],
                             [p['dipole_au'] for p in points[1]], [p['dipole_au'] for p in points[-1]], step)
        core.atomic_array(out / f'hessian_h{hi}.npy', h)
        core.atomic_array(out / f'dipole_derivative_h{hi}.npy', dip)
        hessians.append(h); dipoles.append(dip)
    return hessians, dipoles, qualities


def old_acid_control():
    """Matched geometry/current implementation; old/new software confound retained."""
    out = OUTPUT / 'history_controls/acid-old-negative-direction'
    previous = completed_result(out)
    if previous is not None:
        return previous
    xyz = np.load(HISTORY / 'acid_old_geometry_bohr.npy', allow_pickle=False)
    q = np.load(HISTORY / 'acid_old_negative_mode01_unit_cartesian.npy', allow_pickle=False)
    symbols = read(HISTORY / 'acid_old_optimized_geometry.json')['atom_symbols']
    old_h = np.load(HISTORY / 'acid_old_hessian_used.npy', allow_pickle=False)
    old_h = old_h.transpose(0,2,1,3).reshape(xyz.size, xyz.size)
    rows = []
    for grid in (1, 5):
        center = core.calculate_point(symbols, xyz, out / f'grid{grid}/center', grid_level=grid)
        dm = density(center)
        for step in (.005, .002, .001):
            p = core.calculate_point(symbols, xyz+step*q, out / f'grid{grid}/h{step}/plus', grid_level=grid, dm0=dm)
            m = core.calculate_point(symbols, xyz-step*q, out / f'grid{grid}/h{step}/minus', grid_level=grid, dm0=dm)
            rows.append({'grid_level': grid, 'step_bohr': step,
                         'gradient_curvature': float(np.sum(q*(np.array(p['gradient'])-np.array(m['gradient'])))/(2*step)),
                         'energy_curvature': (p['energy_hartree']+m['energy_hartree']-2*center['energy_hartree'])/step**2,
                         'all_three_points_qualified': all(z['qualified'] for z in (center,p,m))})
        progress(molecule='acid', stage='historical_derivative_control', control_grid=grid)
    native = None
    native_curvature = None
    native_error = None
    try:
        native = core.native_analytic_hessian(symbols, xyz, out / 'grid1/native-analytic', grid_level=1)
        h = np.load(native['hessian_path'], allow_pickle=False).transpose(0,2,1,3).reshape(xyz.size, xyz.size)
        native_curvature = float(q.ravel()@h@q.ravel())
    except Exception as error:
        native_error = optional_diagnostic_error(out / 'grid1/native-analytic', error)
    value = {'status': 'DONE', 'old_analytic_qHq': float(q.ravel()@old_h@q.ravel()),
             'current_native_grid1_qHq': native_curvature,
             'matched_geometry_current_fd': rows, 'native_analytic': native,
             'native_analytic_error': native_error,
             'causal_limit': 'Identical old geometry and Cartesian direction. Old software/assembly differs; this isolates derivative-route behavior within the current native implementation, not one uniquely identified historical bug.',
             'finished_at_utc': utc()}
    write(out / 'result.json', value)
    write(out / 'COMPLETE.json', {'result_sha256':sha(out/'result.json')})
    return value


def follow_negative_mode(name, symbols, coords, dm, hessian, out, *, center_energy):
    """Record both signed trials and select only an actually downhill candidate."""
    # The projected Cartesian direction is saved by the analysis module. Derive
    # it directly here with its own external-space projection to avoid selecting
    # a rigid-body numerical mode.
    from scipy.linalg import null_space
    from pyscf.data import elements
    masses = np.array([elements.MASSES[elements.charge(s)] for s in symbols])
    centered = coords-np.average(coords, axis=0, weights=masses)
    ext = []
    for axis in np.eye(3):
        ext.append((np.sqrt(masses)[:,None]*np.broadcast_to(axis,coords.shape)).ravel())
        ext.append((np.sqrt(masses)[:,None]*np.cross(np.broadcast_to(axis,coords.shape),centered)).ravel())
    basis = null_space(np.array(ext))
    mass = np.repeat(np.sqrt(masses),3)
    hw = (hessian+hessian.T)/2/np.outer(mass,mass)
    vals, vectors = np.linalg.eigh(basis.T@hw@basis)
    q = (basis@vectors[:,0]/mass).reshape(coords.shape)
    q /= np.max(np.abs(q))
    trials = []
    for sign in (1,-1):
        displaced = coords+sign*.05*q
        p=core.calculate_point(symbols,displaced,out/f'candidate-{sign}',grid_level=5,dm0=dm)
        trials.append(p)
    qualified = [p for p in trials if p['qualified']]
    best = min(qualified, key=lambda p:p['energy_hartree']) if qualified else None
    downhill = bool(best is not None and best['energy_hartree'] < center_energy-1e-9)
    selection = {'lowest_eigenvalue':float(vals[0]),'amplitude_max_component_bohr':.05,
          'criterion':'lowest qualified native electronic energy of two signed candidates',
          'center_energy_hartree':float(center_energy), 'required_energy_decrease_hartree':1e-9,
          'candidate_energies':[p['energy_hartree'] for p in trials],
          'candidate_energy_changes_hartree':[p['energy_hartree']-center_energy for p in trials],
          'qualified_candidate_count':len(qualified),'selected_for_new_round':downhill,
          'status':'DOWNHILL_CANDIDATE_SELECTED' if downhill else 'NO_QUALIFIED_DOWNHILL_CANDIDATE'}
    if best is not None:
        selection.update(coordinates_bohr=best['coordinates_bohr'],density_path=best['density_path'])
    write(out/'selection.json',selection)
    return best if downhill else None


def run_molecule(name):
    from pyscf_repair_modes_20261111 import analyze_modes
    out = OUTPUT/'jobs'/name
    state = read(out/'repair_state.json', {'round':0})
    symbols, coords, dm = molecule_inputs(name)
    round_id = state['round']
    if round_id:
        selected = read(out/f'round{round_id-1:03d}'/'negative-mode-follow'/'selection.json')
        coords = np.array(selected['coordinates_bohr']); dm = np.load(selected['density_path'],allow_pickle=False)
    current = out if round_id == 0 else out/f'round{round_id:03d}'
    previous = completed_result(current)
    if previous is not None:
        return previous
    progress(molecule=name,stage='strict_geometry_optimization',repair_round=round_id)
    # This matched grid3 stage measures the old stopping-criteria repair before
    # separately changing to the denser grid5 primary calculation.
    if name in ('acid','calycosin') and round_id==0:
        reference=core.optimize_geometry(symbols,coords,out/'grid3-optimize',grid_level=3,dm0=dm)
        if not reference['minimum_stationarity_passed']:
            raise RuntimeError(name+': tighter grid3 optimization needs another checkpoint segment')
        coords=np.array(reference['coordinates_bohr']);dm=density(reference)
    opt=core.optimize_geometry(symbols,coords,current/'grid5-optimize',grid_level=5,dm0=dm)
    if not opt['minimum_stationarity_passed']:
        raise RuntimeError(name+': tighter grid5 optimization needs another checkpoint segment')
    coords=np.array(opt['coordinates_bohr']);dm=density(opt)
    progress(molecule=name,stage='electronic_stability',repair_round=round_id)
    stability=core.check_electronic_stability(symbols,coords,current/'electronic-stability',grid_level=5,dm0=dm)
    hessians,dipoles,qualities=complete_displacements(name,symbols,coords,dm,current/'finite-differences')
    cq=point_quality(opt['final_point'])
    cq['orbital_stability_passed']=stability['tested_electronic_stability_passed']
    analysis=analyze_modes(symbols,coords,hessians,dipoles,current/'modes',center_quality=cq,point_quality=qualities)
    # Save native Hessian comparison on the two tractable controls. Larger
    # molecules rely on the complete native-gradient finite differences.
    native_comparison=None
    if name in ('water','acid'):
        progress(molecule=name,stage='native_analytic_comparison')
        try:
            native=core.native_analytic_hessian(symbols,coords,current/'native-analytic',grid_level=5,dm0=dm)
            nh=np.load(native['hessian_path'],allow_pickle=False).transpose(0,2,1,3).reshape(coords.size,coords.size)
            native_comparison={'status':'DONE',
                               'relative_matrix_difference':float(np.linalg.norm(nh-hessians[-1])/np.linalg.norm(hessians[-1])),
                               'native_result':native}
        except Exception as error:
            native_comparison=optional_diagnostic_error(current/'native-analytic',error)
    value={'status':'DONE','molecule':name,'repair_round':round_id,
           'result_directory':str(current.relative_to(ROOT)),
           'optimization_path':str(current/'grid5-optimize/result.json'),
           'center_quality':cq,'electronic_stability':stability,'analysis':analysis,
           'native_analytic_comparison':native_comparison,'finished_at_utc':utc(),
           'scope':'Gas-phase isolated molecule, same PBE/def2-SVP/noD3; no herb-solid peak assignment.'}
    imag=[item['imaginary_count'] for item in analysis['steps']]
    step_comparison=analysis.get('step_comparison') or {}
    spectral_difference=step_comparison.get('projected_spectral_step_difference_hartree_per_amu_bohr2')
    fine_lowest=analysis['steps'][-1].get('lowest_signed_eigenvalue_hartree_per_amu_bohr2')
    supported_negative=bool(len(imag)>=2 and imag[0]>0 and imag[-1]>0
                            and spectral_difference is not None and fine_lowest is not None
                            and fine_lowest < -spectral_difference)
    any_negative=bool(imag and max(imag)>0)
    value['negative_mode_followup']={
        'any_negative_mode':any_negative,'same_sign_negative_beyond_empirical_step_difference':supported_negative,
        'status':'NUMERICALLY_AMBIGUOUS_NEGATIVE_MODE_RETAINED' if any_negative else 'NO_NEGATIVE_MODE',
        'geometry_round_advanced':False}
    write(current/'result.json',value)
    if supported_negative and cq['qualified'] and all(q['qualified'] for q in qualities):
        followpath=current/'negative-mode-follow'
        selected=follow_negative_mode(name,symbols,coords,dm,hessians[-1],followpath,
                                      center_energy=opt['final_point']['energy_hartree'])
        value['negative_mode_followup'].update(
            status='DOWNHILL_OPTIMIZATION_ROUND_QUEUED' if selected is not None else 'NO_QUALIFIED_DOWNHILL_CANDIDATE_RETAINED',
            geometry_round_advanced=selected is not None)
        write(current/'result.json',value)
        if selected is not None:
            # Round0 resides directly in jobs/name, later rounds in roundNNN.
            if round_id==0:
                source=read(followpath/'selection.json')
                write(out/'round000/negative-mode-follow/selection.json',source)
            state.update(round=round_id+1,status='NEGATIVE_MODE_FOLLOWUP_QUEUED')
            write(out/'repair_state.json',state)
            raise RuntimeError(name+': supported numerical negative mode; downhill geometry round queued')
    elif supported_negative:
        value['negative_mode_followup']['status']='NEGATIVE_MODE_RETAINED_ELECTRONIC_QUALITY_NOT_QUALIFIED'
        write(current/'result.json',value)
    write(current/'COMPLETE.json',{'result_sha256':sha(current/'result.json'),
           'qualification_is_separate_from_completion':True})
    state.update(status='DONE');write(out/'repair_state.json',state)
    return value


def make_report():
    rows=[]
    for name in MOLECULES:
        root=OUTPUT/'jobs'/name; state=read(root/'repair_state.json',{'round':0})
        current=root if state['round']==0 else root/f"round{state['round']:03d}"
        value=read(current/'result.json')
        if value and (current/'COMPLETE.json').exists():rows.append(value)
    report={'status':'DONE' if len(rows)==len(MOLECULES) else 'PARTIAL',
            'completed_molecules':len(rows),'planned_molecules':len(MOLECULES),
            'molecules':{r['molecule']:{'center_quality':r['center_quality'],'analysis':r['analysis']} for r in rows},
            'historical_control':read(OUTPUT/'history_controls/acid-old-negative-direction/result.json',
                                      read(OUTPUT/'history_controls/acid-old-negative-direction/diagnostic_error.json')),
            'at_utc':utc(),'quality_flags_stop_queue':False}
    write(OUTPUT/'results_summary.json',report)
    text=['# PySCF 虚频修复：实时计算结果','',f"状态：{report['status']}；完成 {len(rows)}/{len(MOLECULES)} 个分子。",
          '', '所有新计算为孤立分子 PBE/def2-SVP、明确无 D3、PySCF2.13.0、原生直接 SCF、grid5、完整网格响应梯度。',
          '完整双步长数值 Hessian 与偶极导数来自逐坐标实际位移，保留有符号频率。完成状态与论文资格分别记录。','',
          '旧结果、模型结构来源及已知误差见 [虚频诊断](PYSCF_IMAGINARY_FREQUENCY_DIAGNOSIS_20261111.md)。','',
          '| 分子 | 细步长模式数 | 两步长虚频数 | 最低频率 cm⁻¹ | 最低频率 THz | 本模型谐振数据资格 |',
          '|---|---:|---|---:|---:|---|']
    for r in rows:
        a=r['analysis'];fine=a['steps'][-1];f=fine['signed_frequencies_cm1'];t=fine['signed_frequencies_THz']
        text.append(f"| {r['molecule']} | {len(f)} | {[s['imaginary_count'] for s in a['steps']]} | {f[0]:.6f} | {t[0]:.8f} | {a['qualification']['paper_ready_molecular_harmonic_data']} |")
    text.append('')
    for r in rows:
        name=r['molecule'];a=r['analysis'];text.extend([f'## {name}','',
            f"新鲜核梯度最大分量：{r['center_quality']['nuclear_gradient_max']:.8g} Eh/Bohr；RMS：{r['center_quality']['nuclear_gradient_rms']:.8g}。",
            f"轨道稳定性实际检查通过：{r['center_quality']['orbital_stability_passed']}。",'',
            f"机器可读完整核验：[该分子结果](../{r['result_directory']}/result.json)；[全部有符号频率与 IR 强度 CSV](../{r['result_directory']}/modes/all_modes.csv)。",'',
            '数值资格（原样保留）：', '```json',json.dumps(a.get('qualification',a.get('qualification_flags',{})),ensure_ascii=False,indent=2),'```',''])
    text.extend(['## 论文使用边界','',
       '只有通过新鲜电子收敛、核驻点、轨道稳定性、全核空间正曲率及双步长一致性核验的结果，才标为本计算模型下的合格最低点。未通过项及负模式保留。',
       'IR 强度为真实偶极导数计算的孤立分子谐振预测；频率不截负、不取绝对值。没有纯品或晶体模型验证，不能据此确认药材固体 THz 吸收峰，也不作为逐样品化学标签。',''])
    (ROOT/'docs/PYSCF_REPAIR_RESULTS_20261111.md').write_text('\n'.join(text),encoding='utf-8')
    return report


def verify():
    prepare()
    checks=0
    for category in ('jobs','history_controls'):
        for path in (OUTPUT/category).rglob('result.json'):
            result=read(path,{})
            if result.get('status') in ('DONE','UNQUALIFIED'):
                for key,ref in result.get('array_refs',{}).items():
                    # Electronic densities/checkpoints are local restart caches;
                    # published gradients, dipoles and modes do not depend on them.
                    if '/.local/' in ref['path']:continue
                    target=Path(ref['path'])
                    if not target.exists() or sha(target)!=ref['sha256']:
                        raise IntegrityError('Published array seal mismatch: '+str(target))
                    if key in result and isinstance(result[key],list):
                        if not np.array_equal(np.asarray(result[key]),np.load(target,allow_pickle=False)):
                            raise IntegrityError('Published JSON/array seal mismatch: '+str(target))
                    checks+=1
        for path in (OUTPUT/category).rglob('COMPLETE.json'):
            completed_result(path.parent)
            checks+=1
    from pyscf_repair_modes_20261111 import verify_analysis
    for name in MOLECULES:
        root=OUTPUT/'jobs'/name; state=read(root/'repair_state.json',{'round':0})
        current=root if state['round']==0 else root/f"round{state['round']:03d}"
        if (current/'COMPLETE.json').exists():
            opt=read(current/'grid5-optimize/result.json')
            xyz=np.array(opt['coordinates_bohr']);plus={};minus={}
            for hi,step in enumerate(STEPS):
                for sign,store in ((1,plus),(-1,minus)):
                    store[sign]=[read(current/'finite-differences'/f'h{hi}'/f'c{j:03d}-{"plus" if sign==1 else "minus"}'/'result.json') for j in range(xyz.size)]
                    if any(p is None for p in store[sign]):raise IntegrityError('Missing saved Cartesian point')
                    for j,p in enumerate(store[sign]):
                        expected=np.array(xyz,copy=True);expected.flat[j]+=sign*step
                        if not np.array_equal(np.array(p['coordinates_bohr']),expected):
                            raise IntegrityError('Displacement coordinate seal mismatch')
                        checks+=1
                hp,dp=assemble_fd([p['gradient'] for p in plus[1]],[p['gradient'] for p in minus[-1]],
                                  [p['dipole_au'] for p in plus[1]],[p['dipole_au'] for p in minus[-1]],step)
                if not np.array_equal(hp,np.load(current/'finite-differences'/f'hessian_h{hi}.npy')):
                    raise IntegrityError('Hessian point replay seal mismatch')
                if not np.array_equal(dp,np.load(current/'finite-differences'/f'dipole_derivative_h{hi}.npy')):
                    raise IntegrityError('Dipole point replay seal mismatch')
                checks+=2
            replay=verify_analysis(current/'modes')
            if replay is False:raise ValueError('Independent modes replay failed: '+name)
            checks+=1
    report=make_report()
    value={'status':report['status'],'completed_molecules':report['completed_molecules'],
           'checks':checks,'all_completed_arrays_and_modes_verified':True,'at_utc':utc()}
    write(OUTPUT/'verification.json',value)
    return value


def run(selected=None):
    prepare()
    errors=[]
    requested=list(selected or MOLECULES)
    for name in requested:
        if name=='acid':
            try:
                old_acid_control()
            except Exception as error:
                optional_diagnostic_error(OUTPUT/'history_controls/acid-old-negative-direction',error)
        try:
            run_molecule(name)
        except Exception as error:
            if is_integrity_error(error):raise
            item={'molecule':name,'error_type':type(error).__name__,'error':str(error),'at_utc':utc()}
            errors.append(item);write(OUTPUT/'errors'/f'{name}-{utc().replace(":", "-")}.json',item)
            progress(status='RUNNING',last_recoverable_error=item)
        make_report()
    report=make_report()
    progress(status=report['status'],stage='computed_queue_pass',completed_molecules=report['completed_molecules'],errors_this_pass=errors)
    selected_complete=all(name in report['molecules'] for name in requested)
    return 0 if selected_complete and not errors else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--verify',action='store_true');parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'.runner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'status':'ALREADY_RUNNING'}));return 1
        try:
            if args.dry_run:print(json.dumps(build_protocol(),ensure_ascii=False,indent=2));return 0
            if args.prepare:prepare();print(json.dumps({'status':'PREPARED'}));return 0
            if args.verify:print(json.dumps(verify(),ensure_ascii=False));return 0
            return run(['water'] if args.pilot else None)
        except Exception as error:
            integrity=is_integrity_error(error)
            write(OUTPUT/('integrity_error.json' if integrity else 'retryable_error.json'),
                  {'error':str(error),'traceback':traceback.format_exc(),'at_utc':utc()})
            print(str(error),file=sys.stderr);return 42 if integrity else 1


if __name__=='__main__':
    raise SystemExit(main())
