"""Traceable protocatechuic-acid conformers and optional supervised Opt/Freq.

prepare performs RDKit force-field modelling only. compute never runs without
the existing V8 live resource guardian and a source/input/output-bound root
authorization. An isolated small molecule is not a substitute for Spinosin,
solid-state THz measurements, or sample-specific chemical measurements.
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
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'reports/external-evidence-20261104-v1/chemistry/chemical_candidates.csv'
REGISTRY_SHA = '06288dafda597cc9d4d8d726430473608425480d8a7953921d77cc617683fefb'
SOURCES = ROOT / 'reports/external-evidence-20261104-v1/chemistry/source_registry.json'
SOURCES_SHA = '761394441c43c78fc06ae1fd34e64ebebf91ee408bf371bd000d6f0b3f44f253'
GRAPH_REPLAY = ROOT / 'reports/external-evidence-20261104-v1/chemistry/graph_and_triage_replay.json'
GRAPH_REPLAY_SHA = '559ac1407376d3a2306b4b4fbc3a465bc6b66e99a551076841d33129b1b471bc'
GUARD = ROOT / 'scripts/run_pyscf_continuation_budget_20261105_v8.py'
GUARD_SHA = '7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65'
AUDIT = ROOT / 'scripts/audit_pyscf_saved_cphf_residual_20261105.py'
AUDIT_SHA = '6ff1df9930257f33637a34cfc58ba01b0fe38c246458ab704891663efbef800c'
CONSUMER = ROOT / 'scripts/pyscf_staged_consumer_20261105_v8.py'
CONSUMER_SHA = '3a107671d9c8a5a47c25f631fd09ed70148dc82b529448d9bafc67179801a585'
STATE = ROOT / 'scripts/pyscf_molecular_state_worker_20261104_v5.py'
STATE_SHA = 'be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8'
BASE = ROOT / 'scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py'
BASE_SHA = 'acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d'
NUMERICAL = ROOT / 'scripts/pyscf_resumable_hessian_worker_20261104_v5.py'
NUMERICAL_SHA = 'b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190'
LEDGER = ROOT / 'reports/pyscf-spinosin-96h-continuation-20261105-v8/cpu_budget.json'
CAP = 390160.8308359999
DEADLINE = '2026-11-09T12:30:00+00:00'
KEY = 'YQUVCSBJEUQKSH-UHFFFAOYSA-N'
SMILES = 'O=C(O)c1ccc(O)c(O)c1'
SCHEMA = 'research4-cloud-protocatechuic-optfreq-result-v1'
PREP_SCHEMA = 'research4-cloud-protocatechuic-conformer-preparation-v1'
AUTH_SCHEMA = 'research4-cloud-small-molecule-authorization-v1'
CM1_TO_THZ = .0299792458


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            d.update(block)
    return d.hexdigest()


def artifact(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def load(path, expected=None):
    p = Path(path).resolve()
    require(p.is_file() and not p.is_symlink() and p.stat().st_size < 2 * 1024 ** 2,
            'Small non-symlink metadata artifact required')
    if expected:
        require(sha(p) == expected, 'Changed metadata: ' + str(p))
    return json.loads(p.read_text())


def save(path, data, exclusive=False):
    p = Path(path)
    temporary = p if exclusive else p.with_suffix(p.suffix + '.writing')
    with temporary.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        os.replace(temporary, p)


def frozen(path, expected, name):
    require(sha(path) == expected, 'Frozen helper changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sites():
    for relative in ('.runtime/quantum-pilot-site', '.runtime/quantum-upgrade-site'):
        p = str(ROOT / relative)
        if p not in sys.path:
            sys.path.insert(0, p)


def source_identity():
    require(sha(REGISTRY) == REGISTRY_SHA and sha(SOURCES) == SOURCES_SHA
            and sha(GRAPH_REPLAY) == GRAPH_REPLAY_SHA, 'Registered graph sources changed')
    with REGISTRY.open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['cid'] == '72']
    require(len(rows) == 1, 'Unique CID72 registry graph required')
    r = rows[0]
    require((r['smiles'], r['full_inchikey'], r['formula']) == (SMILES, KEY, 'C7H6O4')
            and r['graph_source'] == 'https://www.ebi.ac.uk/chebi/CHEBI:36062',
            'Verified registered neutral graph differs')
    return {'name': 'Protocatechuic acid', 'pubchem_cid': 72, 'chebi_id': 36062,
            'smiles': SMILES, 'full_inchikey': KEY, 'formula': 'C7H6O4',
            'charge': 0, 'spin_2S': 0, 'multiplicity': 1, 'electrons': 80,
            'registry': artifact(REGISTRY), 'sources': artifact(SOURCES),
            'prior_graph_replay': artifact(GRAPH_REPLAY),
            'downloaded_experimental_3D': False,
            'scope': 'ISOLATED_MODELED_COMPOUND_NOT_PROJECT_SAMPLE_OR_SPINOSIN'}


def fresh_output(path):
    p = Path(path).resolve()
    require(p.is_relative_to(ROOT / 'reports') and not p.exists(), 'Fresh reports output required')
    p.mkdir(parents=True)
    return p


def prepare(args):
    require(1 <= args.conformers <= 16, 'Bounded1..16 force-field conformers required')
    identity = source_identity()
    output = fresh_output(args.output)
    sites()
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    graph = Chem.MolFromSmiles(SMILES)
    require(graph is not None and Chem.MolToInchiKey(graph) == KEY, 'RDKit graph identity differs')
    mol = Chem.AddHs(graph)
    require(Counter(a.GetSymbol() for a in mol.GetAtoms()) == {'C': 7, 'H': 6, 'O': 4}
            and mol.GetNumAtoms() == 17 and rdMolDescriptors.CalcMolFormula(mol) == 'C7H6O4'
            and Chem.GetFormalCharge(mol) == 0
            and sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 0,
            'Actual formula/charge/radicals differ')
    params = AllChem.ETKDGv3()
    params.randomSeed = args.seed
    params.numThreads = 1
    params.pruneRmsThresh = .10
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=args.conformers, params=params))
    require(ids and AllChem.MMFFHasAllMoleculeParams(mol), 'No supported ETKDG/MMFF candidates')
    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    records = []
    for cid in ids:
        field = AllChem.MMFFGetMoleculeForceField(mol, properties, confId=cid)
        require(field is not None, 'MMFF field unavailable')
        energy_before = float(field.CalcEnergy())
        status = int(field.Minimize(maxIts=500))
        energy = float(field.CalcEnergy())
        coords = [[float(v) for v in p] for p in mol.GetConformer(cid).GetPositions()]
        require(all(math.isfinite(v) for xyz in coords for v in xyz) and math.isfinite(energy), 'Nonfinite conformer')
        stem = output / f'conformer_{cid:03d}'
        stem.with_suffix('.xyz').write_text('17\nETKDGv3/MMFF94s model, not experimental3D or DFT\n'
            + ''.join(f'{symbol} {xyz[0]:.14f} {xyz[1]:.14f} {xyz[2]:.14f}\n'
                      for symbol, xyz in zip(symbols, coords)))
        conf = {'conformer_id': cid, 'atom_symbols': symbols, 'coordinates_angstrom': coords,
                'MMFF94s_status': status, 'MMFF94s_converged': status == 0,
                'MMFF94s_energy_before_kcal_mol': energy_before, 'MMFF94s_energy_kcal_mol': energy,
                'DFT_energy': 'NOT_RUN', 'xyz': artifact(stem.with_suffix('.xyz'))}
        save(stem.with_suffix('.json'), conf, exclusive=True)
        records.append({**conf, 'conformer_json': artifact(stem.with_suffix('.json'))})
    converged = [r for r in records if r['MMFF94s_converged']]
    require(converged, 'No converged force-field candidate; no DFT input selected')
    selected = min(converged, key=lambda r: (r['MMFF94s_energy_kcal_mol'], r['conformer_id']))
    writer = Chem.SDWriter(str(output / 'model_conformers.sdf'))
    for cid in ids:
        mol.SetProp('_Name', 'Protocatechuic acid ETKDG/MMFF model')
        writer.write(mol, confId=cid)
    writer.close()
    manifest = {'schema': PREP_SCHEMA, 'status': 'PREPARED_FORCE_FIELD_MODELS_DFT_NOT_RUN',
                'created_at_utc': utc(), 'source_worker': artifact(__file__), 'identity': identity,
                'versions': {'rdkit': rdkit.__version__}, 'requested_conformers': args.conformers,
                'embedded_conformers': len(ids), 'seed': args.seed, 'num_threads': 1,
                'embedding': 'ETKDGv3', 'force_field': 'MMFF94s', 'force_field_max_steps': 500,
                'DFT_calculations': 0, 'frequencies': 'NOT_RUN', 'IR_intensities_status': 'NOT_COMPUTED',
                'global_minimum_claim': False, 'conformers': records,
                'selected_conformer': selected['conformer_json'], 'model_SDF': artifact(output / 'model_conformers.sdf')}
    save(output / 'prepared_manifest.json', manifest, exclusive=True)
    return manifest


def authorize(args):
    guard = frozen(GUARD, GUARD_SHA, 'small_molecule_v8_live_guard')
    supervision = guard.verify_resource_supervision(expected_threads=args.threads)
    require(args.authorization and args.authorization_sha256, 'Root source-bound authorization required')
    auth = load(args.authorization, args.authorization_sha256)
    require(auth.get('schema') == AUTH_SCHEMA and auth.get('enable_execution') is True
            and auth.get('worker_sha256') == sha(__file__)
            and auth.get('supervisor_sha256') == GUARD_SHA
            and auth.get('prepared_manifest_sha256') == args.prepared_sha256
            and auth.get('output_directory') == str(args.output.resolve())
            and auth.get('selected_threads') == args.threads
            and auth.get('optimizer_maxsteps') == args.maxsteps, 'Root binding differs')
    require(auth.get('no_budget_reset') is True and auth.get('no_deadline_extension') is True,
            'Existing fixed resource pool only')
    require(args.main_pipeline_state is not None and args.main_pipeline_state_sha256
            and auth.get('main_pipeline_state') == str(args.main_pipeline_state.resolve())
            and auth.get('main_pipeline_state_sha256') == args.main_pipeline_state_sha256,
            'Explicit actual closed Spinosin main-state binding required')
    proof = supervision['proof']
    require(Path(proof['ledger_path']).resolve() == LEDGER.resolve(), 'Exact registeredV8 ledger required')
    ledger = load(LEDGER)
    require(ledger['quantum_cpu_cap_seconds'] == CAP and ledger['run_deadline_at_utc'] == DEADLINE,
            'Original cap or deadline changed')
    require(proof['cpu_hard_limit_seconds'] == auth.get('maximum_cpu_seconds')
            and proof['wall_limit_seconds'] <= auth.get('maximum_wall_seconds', 0), 'Actual grant differs/clamped')
    require(120 <= proof['cpu_hard_limit_seconds'] <= 10000 and 61 <= proof['wall_limit_seconds'] <= 14400,
            'Finite small-molecule pilot only')
    not_after = datetime.fromisoformat(auth.get('not_after_utc', ''))
    require(not_after.tzinfo is not None and not_after <= datetime.fromisoformat(DEADLINE)
            and datetime.now(timezone.utc).timestamp() + proof['wall_limit_seconds'] <= not_after.timestamp(),
            'Full finite pilot window must fit the stricter follow-up authorization deadline')
    active = [row for row in ledger['phases'] if row.get('status') == 'RUNNING']
    require(len(active) == 1 and active[0]['phase'] == proof['phase']
            and str(Path(__file__).resolve()) in active[0]['command'],
            'Only this exact source-bound live quantum phase may be active')
    require(1 <= args.maxsteps <= 35, 'Optimizer maxsteps1..35')
    return supervision


def compute(args):
    supervision = authorize(args)  # No numerical import or coordinate access before this live-parent gate.
    audit = frozen(AUDIT, AUDIT_SHA, 'small_molecule_frozen_main_queue_boundary')
    consumer = frozen(CONSUMER, CONSUMER_SHA, 'small_molecule_frozen_metadata_checker')
    with audit.closed_main_queue(args, consumer) as main_state_ref:
        return compute_closed(args, supervision, main_state_ref)


def compute_closed(args, supervision, main_state_ref):
    identity = source_identity()
    prepared = load(args.prepared, args.prepared_sha256)
    require(prepared['schema'] == PREP_SCHEMA and prepared['identity'] == identity,
            'Prepared neutral graph/source identity differs')
    chosen = prepared['selected_conformer']
    conformer = load(chosen['path'], chosen['sha256'])
    require(conformer['MMFF94s_converged'] is True, 'Selected force-field model not converged')
    symbols, coords = conformer['atom_symbols'], conformer['coordinates_angstrom']
    require(Counter(symbols) == {'C': 7, 'H': 6, 'O': 4} and len(coords) == 17
            and all(len(c) == 3 and all(math.isfinite(v) for v in c) for c in coords), 'Geometry shape/composition differs')
    output = fresh_output(args.output)
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    state = {'schema': SCHEMA, 'status': 'PARTIAL', 'started_at_utc': utc(),
             'source_worker': artifact(__file__), 'resource_supervision': supervision,
             'actual_closed_main_pipeline_state': main_state_ref,
             'output_directory': str(output), 'chemical_identity': identity,
             'prepared_manifest': artifact(args.prepared), 'selected_force_field_conformer': chosen,
             'optimizer_converged': False, 'geometry_optimized': False, 'frequencies_status': 'NOT_RUN',
             'IR_intensities_status': 'NOT_COMPUTED', 'global_minimum_search_performed': False,
             'chemical_peak_assignment_verified': False, 'theory_enters_classifier': False,
             'new_Spinosin_calculations': 0, 'scope': 'INDEPENDENT_SMALL_COMPOUND_PILOT'}
    save(output / 'input.json', state, exclusive=True)
    native = None
    log = (output / 'pyscf.log').open('x', buffering=1)
    def emit(event, details=None, **extra):
        row = {'utc': utc(), 'event': event, 'wall_seconds': time.perf_counter() - started_wall,
               'process_cpu_seconds': time.process_time() - started_cpu, **(details or {}), **extra}
        with (output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        save(output / 'progress.json', {**state, **row})
    def stop(signum, _frame):
        raise RuntimeError('Actual liveV8 guardian signal ' + str(signum))
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU)}
    try:
        sites()
        import numpy as np
        from pyscf import gto, lib
        from pyscf.scf import _vhf
        from pyscf.grad.rks import Gradients
        from pyscf.geomopt import geometric_solver
        from pyscf.hessian import thermo
        w = frozen(STATE, STATE_SHA, 'small_compound_frozen_state_helpers')
        base = frozen(BASE, BASE_SHA, 'small_compound_frozen_direct_helpers')
        numerical = frozen(NUMERICAL, NUMERICAL_SHA, 'small_compound_frozen_gate_helpers')
        lib.num_threads(args.threads)
        settings = w.method_settings(args.threads, args.maxsteps)
        settings.update(hessian_max_memory_mb=3000, hessian_max_cycle=50, hessian_level_shift=0,
                        engineering_worker_version='PROTOCATECHUIC17_STOCK_FULL_OPT_FREQ')
        state.update(settings=settings, versions=w.environment_identity())
        native = base.NativeDirectDriverThreadGuard(_vhf, lib.with_omp_threads, lib.num_threads,
                    native_threads=args.threads, emit=lambda details: emit('native_driver_signature', details))
        native.install()
        mol = gto.Mole(); mol.stdout = log
        mol.build(atom=list(zip(symbols, coords)), unit='Angstrom', basis='def2-svp', charge=0,
                  spin=0, symmetry=False, max_memory=8000, verbose=4)
        mol.incore_anyway = False
        require(mol.natm == 17 and mol.nelectron == 80 and mol.nao_nr() == 184, 'Actual molecule dimensions differ')
        state.update(atoms=17, electrons=80, basis_functions=184)
        gradient_class = base.direct_gradient_type(Gradients, emit, memory_mb=lambda: lib.current_memory()[0])
        def gradient_method(mf):
            value = gradient_class(mf); value.grid_response = False; return value
        mf = w.make_scf(mol, settings, output / 'optimization.chk')
        mf.callback = lambda env: emit('optimization_scf_cycle', cycle=int(env.get('cycle', -1)), energy_hartree=float(env['e_tot']))
        (output / 'geometry_steps').mkdir()
        def callback(env):
            cycle = int(env['self'].cycle); stem = output / 'geometry_steps' / f'step_{cycle:04d}'
            xyz = env['mol'].atom_coords(unit='Angstrom'); gradient = np.asarray(env['gradients'])
            stem.with_suffix('.xyz').write_text(base.xyz_text(symbols, xyz, 'Optimization intermediate, not minimum certificate'))
            w.save_immutable_npy(stem.with_suffix('.coordinates_angstrom.npy'), xyz)
            w.save_immutable_npy(stem.with_suffix('.gradient.npy'), gradient)
            info = {'cycle': cycle, 'energy_hartree': float(env['energy']),
                    'scf_converged': bool(env['g_scanner'].converged), **base.stationary_checks(gradient)}
            save(stem.with_suffix('.json'), info, exclusive=True); emit('geometry_cycle', info)
        emit('geometry_optimization_started')
        converged, mol = geometric_solver.kernel(gradient_method(mf), maxsteps=args.maxsteps,
                assert_convergence=True, callback=callback, **base.THRESHOLDS)
        state['optimizer_converged'] = bool(converged)
        require(converged, 'Optimizer did not converge within boundedsteps; HessianNOT_RUN')
        xyz = mol.atom_coords(unit='Angstrom')
        (output / 'optimized.xyz').write_text(base.xyz_text(symbols, xyz, 'Single selected model conformer, inspect signed frequencies'))
        for name, value in [('optimized_coordinates_angstrom', xyz), ('normal_mode_geometry_bohr', mol.atom_coords(unit='Bohr')),
                            ('masses_amu', mol.atom_mass_list(isotope_avg=True))]:
            w.save_immutable_npy(output / (name + '.npy'), value)
        mf = w.make_scf(mol, settings, output / 'final_scf.chk')
        mf.callback = lambda env: emit('fresh_final_scf_cycle', cycle=int(env.get('cycle', -1)), energy_hartree=float(env['e_tot']))
        state['energy_hartree'] = float(mf.kernel()); state['scf_converged'] = bool(mf.converged)
        require(mf.converged and np.isfinite(mf.e_tot), 'Fresh finalSCF not converged')
        gradient = np.asarray(gradient_method(mf).kernel())
        w.save_immutable_npy(output / 'gradient_hartree_per_bohr.npy', gradient)
        state.update(base.stationary_checks(gradient))
        state['geometry_optimized'] = bool(state['optimizer_converged'] and state['final_gradient_thresholds_passed'])
        require(state['geometry_optimized'], 'Fresh gradient failed; HessianNOT_RUN')
        for name, value in [('final_mo_coeff', mf.mo_coeff), ('final_mo_energy', mf.mo_energy), ('final_mo_occ', mf.mo_occ)]:
            w.save_immutable_npy(output / (name + '.npy'), value)
        before = numerical.physical_state(mf)
        hobj = mf.Hessian(); hobj.max_cycle = 50; hobj.level_shift = 0
        state['stock_hamiltonian_gate'] = numerical.stock_operator_gate(mf, hobj)
        state['frequencies_status'] = 'PARTIAL_HESSIAN_NOT_COMPLETE'
        emit('stock_full_system_Hessian_started')
        with base.hessian_memory_context(hobj, settings, emit=lambda event, details: emit(event, details)):
            raw = np.asarray(hobj.kernel(atmlst=list(range(17))))
        w.save_immutable_npy(output / 'hessian_hartree_per_bohr2.npy', raw)
        state['full_system_CPHF_returned_normally'] = True
        state['measured_CPHF_residual_norm'] = 'NOT_COMPUTED'
        state['stable_MF_state_unchanged'] = numerical.physical_state(mf) == before
        require(state['stable_MF_state_unchanged'], 'PhysicalMF changed across Hessian')
        used, asymmetry = base.prepare_hessian_for_analysis(raw, 17)
        w.save_immutable_npy(output / 'hessian_used_for_analysis.npy', used)
        analysis = thermo.harmonic_analysis(mol, used, exclude_trans=True, exclude_rot=True, imaginary_freq=False)
        info = base.frequencies_payload(analysis['freq_wavenumber'], analysis['norm_mode'], 17)
        require(info['normal_mode_count'] == 45, 'All45signed modes required')
        w.save_immutable_npy(output / 'normal_modes.npy', analysis['norm_mode'])
        w.save_immutable_npy(output / 'reduced_mass_amu.npy', analysis['reduced_mass'])
        with (output / 'frequencies.csv').open('x', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['mode', 'frequency_cm1', 'frequency_THz', 'imaginary', 'IR_intensity_status'])
            for index, value in enumerate(info['frequencies_cm1'], 1):
                writer.writerow([index, value, value * CM1_TO_THZ, value < 0, 'NOT_COMPUTED'])
        state.update(info, **base.frequency_minimum_checks(info), hessian_max_symmetry_difference=asymmetry,
                     frequencies_status='DONE', status='FREQUENCIES_COMPUTED', complete_global_Hessian=True,
                     normal_modes_complete=True, Hessian_method='ORIGINAL_STOCK_FULL_RKS_KERNEL',
                     positive_minimum_separate_from_frequency_completion=True)
        state['arrays'] = {p.name: artifact(p) for p in output.glob('*.npy')}
        emit('signed_frequencies_complete', normal_modes=45, imaginary=info['imaginary_frequency_count'])
    except BaseException as error:
        state.update(status='PARTIAL', stop_reason=str(error), exception_type=type(error).__name__)
        (output / 'exception_traceback.txt').write_text(traceback.format_exc())
    finally:
        if native is not None:
            try:
                native.restore()
                state['native_direct_driver_control'] = native.summary()
                require(state['native_direct_driver_control'].get('restoration_failures') == 0
                        and state['native_direct_driver_control'].get('original_driver_restored') is True,
                        'Native driver restoration failed')
            except BaseException as error:
                state.update(status='PARTIAL', stop_reason='Native driver restoration: ' + str(error))
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        state.update(completed_at_utc=utc(), wall_seconds=time.perf_counter() - started_wall,
                     process_cpu_seconds=time.process_time() - started_cpu,
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        state['checkpoint_files'] = [artifact(p) for p in output.glob('*.chk')]
        log.flush(); log.close(); save(output / 'result.json', state, exclusive=True)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--conformers', type=int, default=8); prep.add_argument('--seed', type=int, default=20261106)
    run = sub.add_parser('compute'); run.add_argument('--execute', action='store_true')
    run.add_argument('--prepared', type=Path, required=True); run.add_argument('--prepared-sha256', required=True)
    run.add_argument('--output', type=Path, required=True); run.add_argument('--threads', type=int, choices=(1, 2), default=2)
    run.add_argument('--maxsteps', type=int, default=25); run.add_argument('--authorization', type=Path)
    run.add_argument('--authorization-sha256')
    run.add_argument('--main-pipeline-state', type=Path); run.add_argument('--main-pipeline-state-sha256')
    args = p.parse_args()
    if args.action == 'prepare':
        r = prepare(args)
    else:
        require(args.execute, 'compute requires explicit--execute under the liveV8 guard')
        r = compute(args)
    print(json.dumps({'status': r['status'], 'output': str(args.output.resolve()),
                      'normal_mode_count': r.get('normal_mode_count'), 'new_quantum_calculations': args.action == 'compute'}), flush=True)
    if args.action == 'compute' and r['status'] != 'FREQUENCIES_COMPUTED':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
