"""One-time audit of closed water/ethanol gates; NumPy and stdlib only.

No PySCF import, SCF, gradient evaluation, integral, Hessian derivative or active
checkpoint read occurs here. The existing Hessians are small matrix inputs.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "reports/pyscf-spinosin-continuation-20261031-v3"
OUT = Path(__file__).resolve().parent
INPUTS = {}
MASS = {"H": 1.008, "C": 12.011, "O": 15.999}
BOHR = .52917721092
PLANCK = 6.626070040e-34
E_MASS = 9.10938356e-31
AVOGADRO = 6.022140857e23
HBAR = PLANCK/(2*np.pi)
HARTREE2J = HBAR**2/(E_MASS*(BOHR*1e-10)**2)
AMU = 1e-3/AVOGADRO
FREQUENCY_FACTOR = np.sqrt(HARTREE2J/(AMU*(BOHR*1e-10)**2))/(2*np.pi*299792458)*1e-2
CM1_TO_THZ = .0299792458


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path).resolve()
    INPUTS[str(path.relative_to(ROOT))] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return path


def read(path):
    return json.loads(record(path).read_text())


def load(path, shape):
    result = np.load(record(path), allow_pickle=False)
    assert result.shape == shape and result.dtype == np.float64 and np.isfinite(result).all(), str(path)
    return result


def external_projector(coordinates, masses):
    centered = coordinates - np.sum(coordinates*masses[:,None], axis=0)/masses.sum()
    assert np.linalg.matrix_rank(centered, tol=1e-10) >= 2
    basis = np.eye(3)
    translations = [(np.sqrt(masses)[:,None]*axis).reshape(-1) for axis in basis]
    rotations = [(np.sqrt(masses)[:,None]*np.cross(centered, axis)).reshape(-1) for axis in basis]
    tr = np.stack(translations+rotations, axis=1)
    assert np.linalg.matrix_rank(tr, tol=1e-10) == 6
    q, _ = np.linalg.qr(tr)
    projector = np.eye(coordinates.size) - q@q.T
    values, vectors = np.linalg.eigh(projector)
    vib = vectors[:, values > 1e-7]
    return projector, vib


def inspect_modes(folder, geometry, symbols):
    atom_count, modes_count = len(symbols), 3*len(symbols)-6
    raw = load(folder/"hessian_hartree_per_bohr2.npy", (atom_count, atom_count, 3, 3))
    used = load(folder/"hessian_used_for_analysis.npy", raw.shape)
    expected_used = (raw+raw.transpose(1,0,3,2))*.5
    assert np.array_equal(used, expected_used)
    asym = float(np.max(np.abs(raw-raw.transpose(1,0,3,2))))
    assert asym <= 1e-5
    masses = np.asarray([MASS[item] for item in symbols])
    mass_hessian = np.einsum("pqxy,p,q->pqxy", used, masses**-.5, masses**-.5)
    mass_matrix = mass_hessian.transpose(0,2,1,3).reshape(atom_count*3, atom_count*3)
    projector, vib = external_projector(geometry, masses)
    eigenvalues, eigenvectors = np.linalg.eigh(vib.T@mass_matrix@vib)
    assert eigenvalues.shape == (modes_count,) and np.all(eigenvalues > 0)
    frequencies = np.sqrt(eigenvalues)*FREQUENCY_FACTOR
    rows = list(csv.DictReader(record(folder/"frequencies.csv").open()))
    saved_freq = np.asarray([float(row["frequency_cm1"]) for row in rows])
    assert saved_freq.shape == frequencies.shape and np.all(saved_freq > 0)
    assert all(row["IR_intensity_status"] == "NOT_COMPUTED" for row in rows)
    freq_error = float(np.max(np.abs(saved_freq-frequencies)))
    assert freq_error < 1e-7
    for row, frequency in zip(rows, saved_freq):
        assert abs(float(row["frequency_thz"])-frequency*CM1_TO_THZ) < 1e-10
    modes = load(folder/"normal_modes.npy", (modes_count, atom_count, 3))
    reduced = load(folder/"reduced_mass_amu.npy", (modes_count,))
    z = (modes*np.sqrt(masses)[None,:,None]).reshape(modes_count, atom_count*3).T
    gram_error = float(np.max(np.abs(z.T@z-np.eye(modes_count))))
    external_error = float(np.max(np.abs(projector@z-z)))
    saved_lambdas = (saved_freq/FREQUENCY_FACTOR)**2
    residual = float(np.max(np.abs(projector@mass_matrix@projector@z-z*saved_lambdas[None,:])))
    mass_error = float(np.max(np.abs(reduced-1/np.sum(modes*modes, axis=(1,2)))))
    assert gram_error < 1e-10 and external_error < 1e-10 and residual < 1e-10 and mass_error < 1e-10
    with (OUT/(folder.name+"_replayed_frequencies.csv")).open("w", newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["mode","saved_cm1","numpy_replayed_cm1","difference_cm1","THz","IR_intensity_status"])
        writer.writerows((i+1, saved, actual, actual-saved, actual*CM1_TO_THZ,"NOT_COMPUTED") for i,(saved,actual) in enumerate(zip(saved_freq,frequencies)))
    summary = {"mode_count": modes_count, "all_frequencies_positive": True,
        "raw_max_asymmetry": asym, "analysis_Hessian_exactly_symmetrized": True,
        "max_frequency_replay_difference_cm1": freq_error,
        "max_mass_weighted_mode_orthogonality_error": gram_error,
        "max_external_motion_projection_error": external_error,
        "max_projected_eigen_equation_residual": residual,
        "max_reduced_mass_replay_difference_amu": mass_error,
        "minimum_frequency_cm1": float(frequencies.min()), "maximum_frequency_cm1": float(frequencies.max()),
        "IR_intensities_status": "NOT_COMPUTED"}
    return raw, used, saved_freq, summary


def gradient_stats(path, atom_count):
    gradient = load(path, (atom_count,3))
    lengths = np.linalg.norm(gradient,axis=1)
    stats = {"atom_rms_hartree_per_bohr": float(np.sqrt(np.mean(lengths**2))),
        "atom_max_hartree_per_bohr": float(lengths.max())}
    assert stats["atom_rms_hartree_per_bohr"] < 3e-4 and stats["atom_max_hartree_per_bohr"] < 4.5e-4
    return gradient, stats


def main():
    started, clock = now(), time.perf_counter()
    ledger_bytes = (BASE/"cpu_budget.json").read_bytes()
    ledger = json.loads(ledger_bytes)
    (OUT/"ledger_snapshot_at_matrix_audit.json").write_bytes(ledger_bytes)
    for phase in ("water_native4_validation_001", "ethanol_native4_validation_001"):
        rows = [row for row in ledger["phases"] if row["phase"] == phase]
        assert len(rows) == 1 and rows[0]["status"] == "EXITED" and rows[0]["exit_code"] == 0
    water = BASE/"water-native4-validation-001"
    ethanol = BASE/"validation-ethanol-native4-001"
    report = {"status":"RUNNING_NUMPY_REPLAY", "started_at_utc": started,
        "scope":"CLOSED_WATER_AND_ETHANOL_ENGINEERING_GATES_ONLY_NOT_SPINOSIN",
        "SCF_or_integral_or_Hessian_derivative_evaluations":0, "PySCF_imported":False,
        "active_checkpoint_reads":0, "independent_projection":"Cartesian translations and cross-product rotations, NumPy QR/eigh",
        "masses_amu": MASS, "cm1_frequency_factor": float(FREQUENCY_FACTOR),
        "ledger_snapshot_sha256":hashlib.sha256(ledger_bytes).hexdigest()}
    for name in ("pyscf/data/nist.py","pyscf/data/elements.py","pyscf/hessian/thermo.py"):
        record(ROOT/".runtime/quantum-upgrade-site"/name)
    for path in ("scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py",
                 "scripts/validate_pyscf_native4_large_block_engineering_20261031.py",
                 "scripts/run_pyscf_monitored_budget_20261031.py"):
        record(ROOT/path)
    wr=read(water/"result.json"); er=read(ethanol/"result.json")
    for result in (wr,er):
        assert all(result.get(key) is True for key in ("optimizer_converged","geometry_optimized","scf_converged","final_gradient_thresholds_passed","stationary_minimum_verified"))
        assert result["status"] == "DONE"
        assert result["settings"]["density_fitting"] is False
        assert result["settings"]["xc"] == "PBE" and result["settings"]["basis"] == "def2-svp" and result["settings"]["grid_level"] == 1
    water_geometry = load(water/"normal_mode_geometry_bohr.npy",(3,3))
    symbols = [line.split()[0] for line in record(water/"normal_mode_geometry.xyz").read_text().splitlines()[2:] if line.strip()]
    water_raw, _, water_f, water_modes = inspect_modes(water, water_geometry, symbols)
    _, water_grad = gradient_stats(water/"gradient_hartree_per_bohr.npy",3)
    old=ROOT/"reports/pyscf-spinosin-continuation-20261031-v2/water-direct-v4-single-thread-optfreq"
    old_raw=load(old/"hessian_hartree_per_bohr2.npy",(3,3,3,3))
    old_result=read(old/"result.json")
    difference_h=float(np.max(np.abs(water_raw-old_raw)))
    difference_f=float(np.max(np.abs(water_f-np.asarray(old_result["frequencies_cm1"]))))
    assert difference_h <= 1e-6 and difference_f <= 1e-2
    report["water"]={"normal_modes":water_modes,"fresh_gradient":water_grad,
        "comparison_to_earlier_separately_optimized_native1_control":{"max_abs_raw_Hessian_difference":difference_h,"max_abs_frequency_difference_cm1":difference_f,"predeclared_Hessian_tolerance":1e-6,"predeclared_frequency_tolerance_cm1":.01,"different_optimization_runs_not_same_MF_control":True,"passed":True}}
    ethanol_geometry=load(ethanol/"optimized_coordinates_angstrom.npy",(9,3))/BOHR
    symbols=[line.split()[0] for line in record(ethanol/"optimized.xyz").read_text().splitlines()[2:] if line.strip()]
    original_grad, ethanol_grad=gradient_stats(ethanol/"gradient_hartree_per_bohr.npy",9)
    reference=None; comparisons=[]
    for label in ("standard_full_native1","response_block4_native4","response_block8_native4"):
        folder=ethanol/label; result=read(folder/"result.json")
        raw, used, freq, modes=inspect_modes(folder,ethanol_geometry,symbols)
        grad, gs=gradient_stats(folder/"post_Hessian_gradient_hartree_per_bohr.npy",9)
        actual={"raw":raw,"used":used,"freq":freq,"grad":grad,"energy":result["energy_hartree"]}
        if reference is None: reference=actual
        delta={key:float(np.max(np.abs(value-reference[key]))) for key,value in actual.items()}
        assert delta["raw"]<=1e-6 and delta["freq"]<=.01 and delta["grad"]<=1e-8 and delta["energy"]<=1e-8
        assert np.max(np.abs(grad-original_grad)) <= 1e-8
        saved=read(folder/"comparison_to_standard.json")
        assert abs(delta["raw"]-saved["max_abs_raw_hessian_difference_hartree_per_bohr2"])<1e-15
        assert abs(delta["freq"]-saved["max_abs_frequency_difference_cm1"])<1e-12
        assert result["stationary_minimum_verified"] is True
        comparisons.append({"case":label,"all_81_atom_pairs_and_all_729_raw_components_compared":True,"max_abs_differences":delta,"normal_modes":modes,"fresh_post_Hessian_gradient":gs,"passed":True})
    report["ethanol"]={"fresh_initial_gradient":ethanol_grad,"cases":comparisons,"all_comparisons_passed":True}
    report.update(status="PASS_CLOSED_GATES_NUMPY_REPLAY",ended_at_utc=now(),wall_seconds=time.perf_counter()-clock,
                  input_files=INPUTS, source_script_sha256=digest(__file__))
    (OUT/"matrix_replay.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"water":report["water"],"ethanol_comparisons":comparisons},indent=2))


if __name__ == "__main__":
    main()
