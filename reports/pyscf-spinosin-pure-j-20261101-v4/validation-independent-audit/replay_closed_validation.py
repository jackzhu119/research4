"""Independent NumPy-only replay of root-released closed validation fixtures.

Reads only named water/ethanol artifacts and immutable closed ledger snapshots.
No PySCF import, integral, SCF, gradient/Hessian evaluation, or checkpoint parse.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "reports/pyscf-spinosin-pure-j-20261101-v4"
OUT = Path(__file__).resolve().parent
MASS = {"H": 1.008, "C": 12.011, "O": 15.999}
BOHR = .52917721092
PLANCK, E_MASS, AVOGADRO = 6.626070040e-34, 9.10938356e-31, 6.022140857e23
HBAR = PLANCK / (2*np.pi)
HARTREE2J = HBAR**2/(E_MASS*(BOHR*1e-10)**2)
AMU = 1e-3/AVOGADRO
FACTOR = np.sqrt(HARTREE2J/(AMU*(BOHR*1e-10)**2))/(2*np.pi*299792458)*1e-2
INPUTS = {}


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path, expected=None):
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink(), "Missing/symlink source")
    digest = sha(path)
    require(expected is None or digest == expected, "Input SHA mismatch: " + str(path))
    INPUTS[str(path.relative_to(ROOT))] = {"sha256": digest, "bytes": path.stat().st_size}
    return path


def read(path, expected=None):
    return json.loads(record(path, expected).read_text())


def array(path, shape=None):
    value = np.load(record(path), allow_pickle=False)
    require(value.dtype == np.float64 and (shape is None or value.shape == shape) and np.isfinite(value).all(), "Incomplete/nonfinite/nonfloat64 saved array")
    return value


def check_saved_arrays(value):
    count = 0
    if isinstance(value, dict):
        if all(key in value for key in ("path", "file_sha256", "payload_sha256_C_order")):
            path = record(value["path"], value["file_sha256"])
            actual = array(path, tuple(value["shape"]))
            require(path.stat().st_size == value["file_bytes"] and actual.nbytes == value["payload_bytes"], "Cache size differs")
            require(hashlib.sha256(actual.tobytes(order="C")).hexdigest() == value["payload_sha256_C_order"], "Cache payload SHA differs")
            return 1
        count += sum(check_saved_arrays(item) for item in value.values())
    elif isinstance(value, list):
        count += sum(check_saved_arrays(item) for item in value)
    return count


def projector(coordinates, masses):
    centered = coordinates-(coordinates*masses[:,None]).sum(0)/masses.sum()
    external = np.stack([(np.sqrt(masses)[:,None]*v).ravel() for v in np.eye(3)] +
        [(np.sqrt(masses)[:,None]*np.cross(centered,v)).ravel() for v in np.eye(3)],axis=1)
    require(np.linalg.matrix_rank(external,tol=1e-10)==6, "Nonlinear translation/rotation rank differs")
    q,_ = np.linalg.qr(external)
    p = np.eye(coordinates.size)-q@q.T
    w,v = np.linalg.eigh(p)
    return p,v[:,w>1e-7]


def inspect_case(folder, coordinates, symbols, expected_fingerprint):
    natom,nmode = len(symbols),3*len(symbols)-6
    result = read(folder/"result.json")
    raw = array(folder/"hessian_hartree_per_bohr2.npy",(natom,natom,3,3))
    used = array(folder/"hessian_used_for_analysis.npy",raw.shape)
    require(sha(folder/"hessian_hartree_per_bohr2.npy")==result["raw_hessian_sha256"], "Raw Hessian binding differs")
    require(sha(folder/"hessian_used_for_analysis.npy")==result["analysis_hessian_sha256"], "Analysis Hessian binding differs")
    require(np.array_equal(used,(raw+raw.transpose(1,0,3,2))*.5), "Analysis is not exactly the prescribed symmetrization")
    asym=float(np.max(np.abs(raw-raw.transpose(1,0,3,2))))
    require(asym<=1e-5, "Raw asymmetry gate failed")
    modes=array(folder/"normal_modes.npy",(nmode,natom,3))
    freq=array(folder/"frequencies_cm1.npy",(nmode,))
    require(sha(folder/"normal_modes.npy")==result["normal_modes_sha256"], "Mode file binding differs")
    rows=list(csv.DictReader(record(folder/"frequencies.csv").open()))
    require(len(rows)==nmode and np.all(freq>0), "Missing/nonpositive actual frequencies")
    require(np.max(np.abs(freq-np.asarray([float(row["frequency_cm1"]) for row in rows])))<1e-10,"CSV frequencies differ")
    require(np.max(np.abs(freq-np.asarray(result["frequencies_cm1"])))<1e-12,"Metadata frequencies differ")
    require(all(row["IR_intensity_status"]=="NOT_COMPUTED" for row in rows), "IR not computed scope differs")
    require(np.max(np.abs(freq*.0299792458-np.asarray([float(row["frequency_THz"]) for row in rows])))<1e-12,"THz unit conversion differs")
    masses=np.asarray([MASS[s] for s in symbols])
    p,vib=projector(coordinates,masses)
    mass_h=np.einsum("pqxy,p,q->pqxy",used,masses**-.5,masses**-.5).transpose(0,2,1,3).reshape(natom*3,natom*3)
    eig,_=np.linalg.eigh(vib.T@mass_h@vib)
    require(eig.shape==(nmode,) and np.all(eig>0), "Independent projected Hessian is not a complete positive spectrum")
    replay=np.sqrt(eig)*FACTOR
    replay_error=float(np.max(np.abs(replay-freq)))
    require(replay_error<1e-7,"Independent NumPy frequencies differ")
    z=(modes*np.sqrt(masses)[None,:,None]).reshape(nmode,natom*3).T
    gram=float(np.max(np.abs(z.T@z-np.eye(nmode))))
    external=float(np.max(np.abs(p@z-z)))
    residual=float(np.max(np.abs(p@mass_h@p@z-z*((freq/FACTOR)**2)[None,:])))
    require(max(gram,external,residual)<1e-10,"Saved modes fail independent orthogonality/projection/eigenvector checks")
    helper=read(folder/"rhf_helper_cache_manifest.json",result["raw_e1_ej_ek_cache_manifest_sha256"])
    require(helper["status"]=="DONE_RHF_HELPER_ONLY_BEFORE_RKS_XC" and helper["actual_helper_MO_arguments_verified_equal_bound_MF"],"RHF cache not complete/bound")
    global_cache=read(folder/"hessian_cache_manifest.json")
    cache_count=check_saved_arrays(helper)+check_saved_arrays(global_cache)
    require(global_cache["native_global_kernel_returned_normally"] and global_cache["response_complete_global_atom_ids"]==list(range(natom)),"Full Hessian response incomplete")
    require(result["same_MF_fingerprint_after"]==expected_fingerprint,"Same-MF fingerprint drift")
    for key in ("molecule_identity","coordinates_bohr","MO_coeff","MO_energy","MO_occ","SCF_energy_hartree","grid_coordinates","grid_weights"):
        require(helper["binding"][key]==global_cache["binding"][key],"RHF/global cache binding differs: "+key)
    parts={name:array(helper["arrays"][name]["path"],raw.shape) for name in ("e1","ej","ek")}
    nonzero=int(np.count_nonzero(parts["ek"]))
    require(nonzero==result["RHF_ek_nonzero_entries"]==helper["ek_nonzero_entries"],"ek count differs")
    require((nonzero==0)==result["pure_j_helper_enabled"],"Stock ek must be nonzero and candidate exactly zero")
    gradient=array(folder/"post_Hessian_gradient_hartree_per_bohr.npy",(natom,3))
    stats={"raw_max_asymmetry":asym,"modes":nmode,"all_frequencies_positive":True,
        "observed_case_CPU_seconds":result["CPU_seconds"],"observed_case_wall_seconds":result["wall_seconds"],
        "frequency_cm1":freq.tolist(),"independent_replay_cm1":replay.tolist(),"max_replay_difference_cm1":replay_error,
        "mass_weighted_orthogonality_error":gram,"external_projection_error":external,"projected_eigen_equation_residual":residual,
        "ek_nonzero_entries":nonzero,"ek_max_abs":float(np.max(np.abs(parts["ek"]))),"cache_arrays_SHA_shape_finite_verified":cache_count}
    return {"raw":raw,"sym":used,"freq":freq,"gradient":gradient,**parts,"energy":result["energy_hartree"],"stats":stats,"binding":helper["binding"]}


def audit(identity):
    phase={"water":"water-pure-j-equivalence-001","validation_ethanol":"ethanol-pure-j-equivalence-001"}[identity]
    label="water" if identity=="water" else "ethanol"
    output=OUT/label
    output.mkdir(parents=True,exist_ok=True)
    require(not any((output/name).exists() for name in ("matrix_replay.json","README.md","index.json")),"Existing audit outputs refused")
    ledger=read(BASE/"gate-prefix-snapshots/both-validations-closed-ledger.json","97c75a4bdd68e2a4d78499d896ceb5898f3e930dcc02b7621173dc701140f465")
    rows=[r for r in ledger["phases"] if r["phase"]==phase]
    require(len(rows)==1 and rows[0]["status"]=="EXITED" and rows[0]["exit_code"]==0,"Closed EXITED0 parent required")
    row=rows[0]
    require(not any(row.get(k) for k in ("cpu_terminated","wall_terminated","memory_terminated","hard_killed","monitor_failed")),"Parent has stop flag")
    proof=read(row["supervisor_proof_path"],row["supervisor_proof_sha256"])
    record(row["log"],row["log_sha256"])
    folder=BASE/phase; result=read(folder/"result.json")
    require(result["identity"]==identity and result["validation_only"] and result["status"]=="DONE" and result["validation_passed"],"Fixture result not qualified")
    require(result["resource_supervision"]["proof"]==proof and proof["child_pid"]==row["pid"],"Proof/parent binding differs")
    for entry in result["source_identity"].values():
        if isinstance(entry,dict) and "path" in entry and "sha256" in entry:record(entry["path"],entry["sha256"])
    for entry in result["source_identity"]["pinned_PySCF"].values():record(entry["path"],entry["sha256"])
    record(ROOT/".runtime/quantum-upgrade-site/pyscf/data/nist.py")
    record(ROOT/".runtime/quantum-upgrade-site/pyscf/data/elements.py")
    fingerprint=read(folder/"same_MF_fingerprint.json")
    coordinates=array(folder/"actual_geometry_angstrom.npy")
    command=row["command"];xyz=ROOT/command[command.index("--input-xyz")+1]
    symbols=[line.split()[0] for line in record(xyz).read_text().splitlines()[2:] if line.strip()]
    initial=array(folder/"fresh_gradient_hartree_per_bohr.npy",(len(symbols),3))
    norms=np.linalg.norm(initial,axis=1);grms=float(np.sqrt(np.mean(norms**2)));gmax=float(norms.max())
    require(grms<3e-4 and gmax<4.5e-4,"Fresh gradient threshold failed")
    stock=inspect_case(folder/"stock_partial_B8_native4",coordinates,symbols,fingerprint)
    candidate=inspect_case(folder/"pure_j_partial_B8_native4",coordinates,symbols,fingerprint)
    difference={key:float(np.max(np.abs(stock[key]-candidate[key]))) for key in ("raw","sym","freq","gradient","e1","ej")}
    difference["energy"]=abs(stock["energy"]-candidate["energy"])
    tolerances={"raw":1e-6,"sym":1e-6,"freq":.01,"gradient":1e-8,"e1":1e-6,"ej":1e-6,"energy":1e-8}
    require(all(difference[k]<=v for k,v in tolerances.items()),"Fixed whole-matrix/mode gate failed")
    for key in ("molecule_identity","coordinates_bohr","MO_coeff","MO_energy","MO_occ","SCF_energy_hartree","grid_coordinates","grid_weights"):
        require(stock["binding"][key]==candidate["binding"][key],"Stock/pure binding differs: "+key)
    for case in (stock,candidate):require(np.max(np.abs(case["gradient"]-initial))<=1e-8,"Post-Hessian gradient drift")
    require("pyscf" not in sys.modules,"Unexpected quantum engine import")
    report={"status":"PASS_INDEPENDENT_CLOSED_VALIDATION_NUMPY_REPLAY","identity":identity,"validation_only":True,
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"parent_phase":phase,"parent_exit_code":0,"parent_actual_wait4_CPU_seconds":row["actual_cpu_seconds"],
        "all_raw_and_sym_Hessian_components_compared":int(stock["raw"].size),"all_atom_pair_couplings_compared":True,
        "max_abs_differences":difference,"fixed_tolerances":tolerances,"fresh_gradient_atom_RMS":grms,"fresh_gradient_atom_max":gmax,
        "stock":stock["stats"],"pure_j":candidate["stats"],"same_MF_geometry_MO_grid_bindings_equal":True,
        "stock_ek_nonzero_pure_ek_exact_zero":True,"projection_method":"Independent Cartesian translation/cross-product rotation basis, NumPy QR/eigh",
        "timing_interpretation":{"single_paired_observation_only":True,"case_order":"stock then pure-J on the same MF",
            "general_or_Spinosin_speedup_proven":False,"cannot_compare_failed_V3_duration_to_future_completed_V4_as_speedup":True,
            "Spinosin_launch_basis":"Numerical equivalence plus early RHF SAVE_ONLY checkpoints, independent actual-CPU control and a separately finite resource budget; not a proven speed gain."},
        "masses_amu":MASS,"frequency_factor_cm1":float(FACTOR),"IR_intensities_status":"NOT_COMPUTED",
        "new_Spinosin_frequencies":0,"new_SCF_gradient_Hessian_or_integral_evaluations":0,"PySCF_imported":False,"checkpoint_reads":0,
        "only_explicitly_closed_fixture_files_read":True,"input_files":dict(INPUTS),"audit_source_sha256":sha(__file__)}
    for relative,item in report["input_files"].items():require(sha(ROOT/relative)==item["sha256"],"Input changed during replay")
    (output/"matrix_replay.json").write_text(json.dumps(report,indent=2)+"\n")
    with (output/"frequencies_replayed.csv").open("w",newline="") as handle:
        w=csv.writer(handle);w.writerow(["case","mode","saved_cm1","independent_NumPy_cm1","IR_intensity_status"])
        for name,case in (("stock",stock),("pure_j",candidate)):
            for i,(saved,replay) in enumerate(zip(case["freq"],case["stats"]["independent_replay_cm1"]),1):w.writerow([name,i,saved,replay,"NOT_COMPUTED"])
    (output/"README.md").write_text(f"# {label}: independent closed-fixture validation\n\nPASS: NumPy replay compared all {stock['raw'].size} raw and symmetrized Hessian entries, all {len(stock['freq'])} modes and every atom-pair coupling. Maximum raw difference {difference['raw']:.15g} Eh/Bohr², frequency difference {difference['freq']:.15g} cm⁻¹. Both full spectra are positive; independent projected-matrix frequencies, mode normalization and eigen-equation residuals pass. Stock ek has {stock['stats']['ek_nonzero_entries']} nonzero entries; pure-J ek is exactly zero. e1/ej, fresh/post gradients, same-MF geometry/orbital/grid bindings and saved cache file/payload hashes were checked.\n\nParent is closed EXITED0 without resource-stop flags; actual wait4 CPU {row['actual_cpu_seconds']} s. This is a software/numerical fixture, not a medicinal-plant or Spinosin result. IR NOT_COMPUTED. No SCF, integral, gradient/Hessian evaluation, PySCF import or checkpoint read occurred in this audit. Existing data and frozen source were unchanged. Matrix replay uses independent Cartesian translation/rotation QR projection and NumPy eigendecomposition. Source-file hashes and all numeric checks are in matrix_replay.json; frequency rows are in frequencies_replayed.csv. Physical accuracy on Spinosin and general speedups are not established by these small controls.\n")
    with (output/"README.md").open("a") as handle:
        handle.write(f"\n## Single paired timing observation\n\n| Case | CPU seconds | Wall seconds |\n|---|---:|---:|\n| Stock | {stock['stats']['observed_case_CPU_seconds']:.12f} | {stock['stats']['observed_case_wall_seconds']:.12f} |\n| Pure J | {candidate['stats']['observed_case_CPU_seconds']:.12f} | {candidate['stats']['observed_case_wall_seconds']:.12f} |\n\nThese are single, sequential paired measurements (stock first), not a general benchmark. Ethanol's observed CPU can increase while wall time falls slightly; neither this nor the water observation proves large-molecule acceleration. A future Spinosin attempt is justified by numerical equivalence, earlier RHF checkpoints, stricter independently monitored actual CPU and a separately finite budget. V3 did not finish a full Spinosin Hessian, so its failed duration cannot be used as a completed baseline to calculate speedup.\n")
    files=[output/"README.md",output/"matrix_replay.json",output/"frequencies_replayed.csv"]
    index={"status":"FROZEN_CLOSED_VALIDATION_AUDIT","audit_source_path":str(Path(__file__).relative_to(ROOT)),"audit_source_sha256":sha(__file__),"files":[{"path":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in files]}
    (output/"index.json").write_text(json.dumps(index,indent=2)+"\n")
    print(json.dumps({"identity":identity,"status":report["status"],"difference":difference,"replay_max_cm1":[stock['stats']['max_replay_difference_cm1'],candidate['stats']['max_replay_difference_cm1']],"index_sha256":sha(output/"index.json")}))


if __name__=="__main__":
    for identity in ("water","validation_ethanol"):
        INPUTS.clear();audit(identity)
