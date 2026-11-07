"""Bounded fixed-original-grid CENTRAL ENERGY CURVATURE with bounded SCF polish.

Only fresh geometry energies E0, Eplus and Eminus are evaluated, in that order,
with at most three bounded SCF calls at each geometry in one V8 phase.
Final energies use two different-density full native nonincremental audits.
This is an energy-only local diagnostic,
not a full Hessian, minimum, frequency, IR, geometry or classifier calculation.
The original 75-atom source and all historical worker/results stay immutable.
"""
from __future__ import annotations

import argparse
import ast
import copy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
from numbers import Real
import os
from pathlib import Path
import signal
import struct
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v8.py"
GUARD_SHA = "7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65"
ANALYZER = ROOT / "scripts/analyze_pyscf_staged_modes_20261105_v8.py"
ANALYZER_SHA = "da2a53d85015e0ced6445e7766e18ae4fbe58e098e0eb077719759be7b18cab6"
AUDITOR = ROOT / "scripts/audit_pyscf_saved_cphf_residual_20261105.py"
AUDITOR_SHA = "6ff1df9930257f33637a34cfc58ba01b0fe38c246458ab704891663efbef800c"
POINT_SCHEMA = "research4-fixed-original-grid-energy-point-v2"
POINT_SUCCESS = "FIXED_ORIGINAL_GRID_POLISHED_SCF_ENERGY_COMPUTED"
ENERGY_SCHEMA = "research4-fixed-original-grid-central-energy-curvature-v2"
ENERGY_SUCCESS = "ENERGY_CURVATURE_COMPUTED"
CONTROL_WORKER = ROOT / "scripts/research4_fixed_grid_water_convergence_20261107_v3.py"
CONTROL_POINT_SCHEMA = "research4-fixed-grid-water-convergence-point-v3"
CONTROL_POINT_SUCCESS = "WATER_FIXED_GRID_CERTIFIED_SCF_GRADIENT_COMPUTED"
CONTROL_SCHEMA = "research4-fixed-grid-water-step-convergence-v3"
CONTROL_SUCCESS = "WATER_FIXED_GRID_STEP_CONVERGENCE_FULL_ENDPOINT_MEASURED"
QUALIFICATION_SCHEMA = "research4-fixed-original-grid-energy-source-qualification-v2"
QUALIFICATION_SUCCESS = "SOURCE_QUALIFIED_OUTSIDE_QUANTUM_PHASE"
SCHEDULE_SCHEMA = "research4-cloud-followup-phase-schedule-v1"
SOURCE_SNAPSHOT = "3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311"
METHOD = "FIXED_ORIGINAL_QUADRATURE"
DIAGNOSTIC_STATUS = "ENERGY_ONLY_LOCAL_DIAGNOSTIC_NOT_FULL_HESSIAN_MINIMUM_FREQUENCY"
SPINOSIN_GRADIENT_SHA = "9418727d662aaada4548cf27bbfdec365627126eb1836703096f737a5ad94ad4"
ENERGY_WORK_DIRECTORY = ROOT / "reports/fixed-quadrature-convergence-20261107-v2/queue-host-a-006"
MODE_INDEX = 1
STEP_BOHR = 0.0003125
SCF_CONV_TOL = 1e-12
SCF_CONV_TOL_GRAD = 1e-9
SCF_MAIN_ORBITAL_GRADIENT_TRIGGER = 1e-10
MAX_SCF_KERNEL_CALLS = 3
SCF_MAX_CYCLES_PER_CALL = 100
SCF_CERTIFICATION_POLICY = "FULL_NATIVE_VEFF_ENDPOINT_CERTIFICATION_NOT_NATIVE_KERNEL_FLAG"
CONTROL_COORDINATES = (0, 4, 8)
CONTROL_STEPS = (0.00125, 0.000625, 0.0003125, 0.00015625, 0.000078125)
CONTROL_THRESHOLDS = {"FD_Hq_max_abs_error_Eh_per_Bohr2": 1e-4,
    "h_halving_FD_Hq_max_abs_difference_Eh_per_Bohr2": 5e-5,
    "central_energy_curvature_abs_error_Eh_per_Bohr2": 1e-4}
FIXED_ARRAY_NAMES = ("coords", "weights", "non0tab", "screen_index", "atm_idx", "quadrature_weights")
STRICT_DEADLINE = datetime.fromisoformat("2026-11-09T08:25:17.730193+00:00")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}


def checked(record, *, max_bytes=32 * 1024**2):
    original = Path(record["path"])
    require(not original.is_symlink(), "Symlink source rejected")
    p = original.resolve()
    require(p.is_file() and p.is_relative_to(ROOT), "Existing repository source required")
    require(p.stat().st_size <= max_bytes and sha(p) == record["sha256"], "Source hash or size differs")
    return p


def read(path):
    p = Path(path)
    require(p.is_file() and p.stat().st_size <= 32 * 1024**2, "Bounded JSON required")
    return json.loads(p.read_text())


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def ledger_header(value):
    return {k:v for k,v in value.items() if k not in ("phases","consumed_cpu_seconds","remaining_quantum_cpu_seconds")}


def row_prefix(value):
    rows=value.get("phases")
    require(isinstance(rows,list) and all(isinstance(row,dict) and row.get("status")!="RUNNING" for row in rows),
            "Sealed source ledger must contain closed original rows only")
    names=[row.get("phase") for row in rows]
    require(len(names)==len(set(names)) and all(isinstance(name,str) and name for name in names),
            "Unique sealed source phase names required")
    return [{"phase":row["phase"],"canonical_sha256":canonical_sha(row)} for row in rows]


def read_schedule(record):
    schedule=read(checked(record))
    require(schedule.get("schema")==SCHEDULE_SCHEMA,"Explicit phase schedule schema required")
    rows=schedule.get("phases")
    require(isinstance(rows,list) and rows,"A finite nonempty phase schedule required")
    names=[]
    for row in rows:
        name=row.get("phase")
        require(isinstance(name,str) and name and all(ch.isalnum() or ch in "-_" for ch in name),"Exact safe phase name required")
        names.append(name)
        require(set(row)=={"phase","worker","output_directory","log"},"Phase schedule has unexpected/missing fields")
        checked(row["worker"],max_bytes=1024**2)
        for key in ("output_directory","log"):
            p=Path(row[key]).resolve()
            require(p.is_relative_to(ROOT/"reports"),"Scheduled output and log must stay in reports")
    require(len(names)==len(set(names)),"Duplicate scheduled phases rejected")
    return schedule


def validate_appended_rows(ledger, sealed_rows, schedule, proof, guard, expected_script=None):
    """Pure metadata gate: preserve every old row and whitelist finite new ones."""
    actual=ledger.get("phases",[])
    require(len(actual)>=len(sealed_rows),"Source ledger prefix was truncated")
    for row,seal in zip(actual,sealed_rows):
        require(row.get("phase")==seal["phase"] and canonical_sha(row)==seal["canonical_sha256"],
                "An original sealed closed ledger row changed")
    allowed={row["phase"]:row for row in schedule["phases"]}
    appended=actual[len(sealed_rows):]
    require(appended,"Actual protected phase must be appended after qualification")
    running=[]
    for row in appended:
        require(row.get("phase") in allowed,"Unknown extra current-ledger phase is not authorized by the seal")
        policy=allowed[row["phase"]]
        worker=checked(policy["worker"],max_bytes=1024**2)
        command=row.get("command",[])
        scripts=[Path(x).resolve() for x in command if isinstance(x,str) and x.endswith(".py")]
        require(worker in scripts and "--output" in command,"Appended phase command lacks the scheduled worker/output")
        idx=command.index("--output")
        require(idx+1<len(command) and Path(command[idx+1]).resolve()==Path(policy["output_directory"]).resolve()
                and Path(row.get("log","")).resolve()==Path(policy["log"]).resolve(),
                "Appended phase command/output/log differs from its exact schedule")
        if row.get("status")=="RUNNING":
            running.append(row)
            require(row["phase"]==proof["phase"] and worker==Path(expected_script or __file__).resolve(),
                    "An unrelated scheduled phase is running instead of this worker")
            require(row.get("supervisor_proof_sha256")==canonical_proof_sha(proof, row, guard),
                    "Current phase proof differs from its actual guardian")
        else:
            require(row.get("status") in ("EXITED","PARTIAL_OR_FAILED") and row.get("ended_at_utc")
                    and isinstance(row.get("actual_cpu_seconds"),(float,int))
                    and not isinstance(row.get("actual_cpu_seconds"),bool) and row["actual_cpu_seconds"]>=0,
                    "A preceding scheduled phase lacks its actual closed cost")
            require(not row.get("residual_processes_require_review_after_main_payload_exit")
                    and not row.get("monitor_failed"),"Unresolved or unknown preceding phase requires root forensics")
            if row["status"]!="EXITED":
                require(any(row.get(key) is True for key in ("wall_terminated","cpu_terminated","memory_terminated")),
                        "Non-resource scientific/startup failure in appended phase requires a new root-reviewed seal")
            else:
                require(row.get("exit_code")==0,"Closed EXITED phase must have actual exit code zero")
            log=Path(row["log"])
            require(log.is_file() and sha(log)==row.get("log_sha256"),"Closed appended phase log changed")
            p=Path(row.get("supervisor_proof_path",""))
            require(p.is_file() and sha(p)==row.get("supervisor_proof_sha256"),"Closed appended guardian proof changed")
            closed=read(p)
            require(closed.get("phase")==row["phase"] and closed.get("supervisor_script_sha256")==GUARD_SHA
                    and Path(closed["ledger_path"]).resolve()==Path(proof["ledger_path"]).resolve(),
                    "Closed appended guardian source or lineage differs")
    require(len(running)==1 and running[0]["phase"]==proof["phase"],"Exactly this one protected appended phase may be running")
    return {"sealed_original_rows_verified":len(sealed_rows),"scheduled_appended_rows_verified":len(appended),
            "unique_running_phase":proof["phase"]}


def canonical_proof_sha(proof,row,guard):
    path=guard.root_path(row.get("supervisor_proof_path",""))
    require(path.is_file() and guard.read(path)==proof,"Actual running proof record is not identical")
    return sha(path)


def save(path, value, *, exclusive=False):
    p = Path(path)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with p.open("x") as handle:
            handle.write(content)
    else:
        tmp = p.with_name(p.name + ".tmp-" + str(os.getpid()))
        with tmp.open("x") as handle:
            handle.write(content)
        os.replace(tmp, p)


def load(path, expected, name):
    require(sha(path) == expected, "Frozen dependency source differs")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_displacement(mode_index, coordinate_index, step_bohr, sign, atoms=None):
    require((mode_index is not None) != (coordinate_index is not None), "Choose one mode OR one Cartesian coordinate")
    if mode_index is not None:
        require(type(mode_index) is int and mode_index >= 1, "Mode numbers are positive and one based")
        if atoms is not None:
            require(mode_index <= 3 * atoms - 6, "Mode number exceeds complete nonlinear mode space")
    else:
        require(type(coordinate_index) is int and coordinate_index >= 0, "Cartesian coordinate numbers are zero based")
        if atoms is not None:
            require(coordinate_index < 3 * atoms, "Coordinate number exceeds complete molecule")
    require(type(sign) is int and sign in (-1, 1), "A point has one explicitly signed displacement")
    require(math.isfinite(step_bohr) and 1e-4 <= step_bohr <= 0.02, "Bounded 0.0001--0.02 Bohr Euclidean displacement required")


def fresh_output(path):
    p = Path(path).resolve()
    require(p.is_relative_to(ROOT / "reports") and not p.exists(), "Fresh separate reports directory required")
    p.mkdir(parents=True)
    return p


def sole_phase(guard, supervision):
    proof = supervision["proof"]
    ledger = guard.read(guard.root_path(proof["ledger_path"]))
    guard.validate_ledger(ledger, allow_running=True)
    running = [p for p in ledger["phases"] if p.get("status") == "RUNNING"]
    require(len(running) == 1 and running[0]["phase"] == proof["phase"], "Exactly this one supervised Q phase may run")
    scripts = [guard.root_path(x) for x in running[0]["command"] if isinstance(x, str) and x.endswith(".py")]
    require(Path(__file__).resolve() in scripts, "Actual guardian command does not bind this worker")


@contextmanager
def closed_main(args, analyzer):
    record = {"path": str(args.main_pipeline_state.resolve()), "sha256": args.main_pipeline_state_sha256}
    path = checked(record)
    auditor = load(AUDITOR, AUDITOR_SHA, "cloud_followup_main_closure_validator")
    lock = ROOT / ".runtime/research4-materialized-quantum-queue.lock"
    require(lock.is_file() and not lock.is_symlink(), "Existing main quantum queue lock required")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        auditor.validate_main_closed_state(read(path))
        yield record
        checked(record)


def source_identity(result, control_validation):
    if control_validation:
        require(result["case"] == "water_software_control" and result["atoms"] == 3,
                "Only authenticated water software control is authorized")
    else:
        require(result["case"] == "Spinosin" and result["atoms"] == 75 and result["basis_functions"] == 762
                and result["snapshot_manifest_sha256"] == SOURCE_SNAPSHOT, "Only original complete Spinosin C001 source is allowed")


def qualify_source(args):
    """Run the ORIGINAL strict metadata analyzer only while all Q is closed."""
    guard=load(GUARD,GUARD_SHA,"cloud_followup_external_source_qualification_guard")
    analyzer=load(ANALYZER,ANALYZER_SHA,"cloud_followup_external_original_analyzer")
    schedule_ref={"path":str(args.allowed_phase_schedule.resolve()),"sha256":args.allowed_phase_schedule_sha256}
    schedule=read_schedule(schedule_ref)
    ledger_path=Path(schedule["ledger_path"]).resolve()
    ledger=guard.read(ledger_path)
    guard.validate_ledger(ledger,allow_running=False)
    require(ledger_path==guard.root_path(ledger["continuation_ledger_path"]),"Schedule must bind the registered cumulative ledger")
    initial_ledger_sha=sha(ledger_path)
    started_cpu,started_wall=time.process_time(),time.perf_counter()
    with closed_main(args,analyzer) as main_ref:
        gate=analyzer.qualify(args.source_result.resolve(),args.source_result_sha256,control=args.control_validation)
        source_identity(gate["result"],args.control_validation)
        require(Path(gate["result"]["resource_supervision"]["proof"]["ledger_path"]).resolve()==ledger_path,
                "Original global producer and explicit schedule use different current-ledger lineages")
        guard.validate_ledger(guard.read(ledger_path),allow_running=False)
        require(sha(ledger_path)==initial_ledger_sha,"Current ledger changed during external qualification")
        immutable={str(Path(path).resolve()):dict(record) for path,record in gate["hashes"].items()}
        require(str(ledger_path) in immutable,"Full original qualifier did not capture the registered ledger")
        # The original frozen payload restorer will also verify every array at
        # actual execution. Record its declared byte hashes without importing NumPy.
        snapshot=Path(gate["result"]["snapshot_directory"])
        for record in gate["source"]["snapshot_manifest"]["arrays"].values():
            path=(snapshot/record["file"]).resolve()
            require(path.is_file() and sha(path)==record["sha256"],"Original snapshot array file hash differs")
            immutable[str(path)]={"sha256":record["sha256"],"bytes":path.stat().st_size,
                "verified_by":"EXTERNAL_ORIGINAL_SNAPSHOT_ARRAY_HASH_NO_NUMERICAL_IMPORT"}
        gradient_result = read(checked(gate["result"]["gradient_gate"]["result"]))
        gradient_path = Path(gate["result"]["gradient_gate"]["result"]["path"]).parent / "gradient_hartree_per_bohr.npy"
        gradient_record = {"path":str(gradient_path), "sha256":gradient_result["gradient_npy_sha256"]}
        checked(gradient_record)
        if not args.control_validation:
            require(gradient_record["sha256"] == SPINOSIN_GRADIENT_SHA,
                    "The actual historical 75x3 Spinosin reference gradient is required")
        immutable[str(gradient_path.resolve())] = {"sha256":gradient_record["sha256"],
            "bytes":gradient_path.stat().st_size,"verified_by":"ORIGINAL_GRADIENT_GATE_SIBLING_ARRAY_HASH"}
        ledgers=[]
        for path,record in immutable.items():
            p=Path(path)
            require(p.is_file() and not p.is_symlink() and p.is_relative_to(ROOT),"Qualification source outside repository")
            require(p.stat().st_size==record["bytes"] and sha(p)==record["sha256"],"Source bytes changed during external qualification")
            if p.suffix==".json" and p.stat().st_size<=32*1024**2:
                value=read(p)
                if isinstance(value,dict) and isinstance(value.get("phases"),list) and "consumed_cpu_seconds" in value:
                    ledgers.append({"path":str(p),"sha256_at_qualification":record["sha256"],
                        "immutable_header":ledger_header(value),"closed_rows":row_prefix(value)})
        require(any(Path(x["path"])==ledger_path for x in ledgers),"Registered ledger rows missing from source seal")
        existing={row["phase"] for row in ledger["phases"]}
        require(not existing.intersection(row["phase"] for row in schedule["phases"]),"Scheduled future phase reuses an existing phase name")
        # Validate complete serializability now; no modules/callables may leak in.
        gate=json.loads(json.dumps(gate,allow_nan=False))
        out=fresh_output(args.output)
        value={"schema":QUALIFICATION_SCHEMA,"status":QUALIFICATION_SUCCESS,"created_at_utc":utc(),
            "source_worker":ref(__file__),"frozen_analyzer":ref(ANALYZER),"source_result":ref(args.source_result),
            "control_validation":args.control_validation,"closed_main_pipeline_state":main_ref,
            "allowed_phase_schedule":schedule_ref,"mutable_registered_ledger_path":str(ledger_path),
            "source_ledgers":ledgers,"immutable_source_hashes":immutable,"complete_original_gate":gate,
            "qualification_performed_with_all_ledgers_closed":True,"new_quantum_evaluations":0,
            "numerical_packages_imported":False,"target_numerical_arrays_loaded":False,
            "process_CPU_seconds":time.process_time()-started_cpu,"wall_seconds":time.perf_counter()-started_wall}
        save(out/"qualification.json",value,exclusive=True)
    return value


def source_gate(args, analyzer, guard, supervision):
    """Reuse a root-approved strict external gate, with immutable-prefix checks."""
    seal_ref={"path":str(args.source_qualification.resolve()),"sha256":args.source_qualification_sha256}
    seal=read(checked(seal_ref))
    require(seal.get("schema")==QUALIFICATION_SCHEMA and seal.get("status")==QUALIFICATION_SUCCESS
            and seal.get("qualification_performed_with_all_ledgers_closed") is True
            and seal.get("new_quantum_evaluations")==0,"A real externally qualified source seal is required")
    require(seal["source_worker"]==ref(__file__) and seal["frozen_analyzer"]==ref(ANALYZER),
            "Qualification producer or frozen analyzer changed")
    source_ref={"path":str(args.source_result.resolve()),"sha256":args.source_result_sha256}
    require(seal["source_result"]["path"]==source_ref["path"] and seal["source_result"]["sha256"]==source_ref["sha256"]
            and seal["control_validation"]==args.control_validation,"Qualification belongs to a different source or control scope")
    checked(source_ref)
    schedule=read_schedule(seal["allowed_phase_schedule"])
    proof=supervision["proof"]
    mutable=Path(seal["mutable_registered_ledger_path"]).resolve()
    require(mutable==guard.root_path(proof["ledger_path"])==Path(schedule["ledger_path"]).resolve(),
            "Source seal, exact phase schedule and live guardian must share one registered ledger")
    ledger=guard.read(mutable);guard.validate_ledger(ledger,allow_running=True)
    prefix_verification=None
    for binding in seal["source_ledgers"]:
        path=Path(binding["path"]).resolve();actual=guard.read(path)
        require(ledger_header(actual)==binding["immutable_header"],"A sealed source-ledger header changed")
        if path==mutable:
            prefix_verification=validate_appended_rows(actual,binding["closed_rows"],schedule,proof,guard)
        else:
            require(sha(path)==binding["sha256_at_qualification"] and row_prefix(actual)==binding["closed_rows"],
                    "A historical immutable source ledger or closed row changed")
    require(prefix_verification is not None,"Mutable current ledger has no sealed closed-row prefix")
    immutable=seal["immutable_source_hashes"]
    for path,record in immutable.items():
        p=Path(path).resolve()
        if p==mutable:
            continue  # Its WHOLE hash changes only by the exact approved append policy above.
        require(p.is_file() and not p.is_symlink() and p.is_relative_to(ROOT)
                and p.stat().st_size==record["bytes"] and sha(p)==record["sha256"],
                "An immutable original source input changed after qualification")
    gate=seal["complete_original_gate"]
    require(gate.get("metadata_gate_passed_before_numerical_import") is True and gate["result_path"]==source_ref["path"]
            and gate["result"]==read(checked(source_ref)),"Sealed original gate does not bind the actual current source result")
    result=gate["result"]
    source_identity(result,args.control_validation)
    closure=guard.verify_closed_phase({"source_worker":result["source_worker"],"resource_supervision":result["resource_supervision"],
            "output_directory":str(args.source_result.resolve().parent),"snapshot_manifest_sha256":result["snapshot_manifest_sha256"],
            "stage":"global-assemble"},expected_worker_path=analyzer.WORKER,expected_worker_sha256=analyzer.WORKER_SHA,
        expected_result_schema=analyzer.SCHEMA,allowed_result_statuses=("FREQUENCIES_COMPUTED",),
        expected_output=args.source_result.resolve().parent,expected_snapshot_sha256=result["snapshot_manifest_sha256"])
    require(closure==gate["closure"],"Real original global producer closure changed after external qualification")
    if not args.control_suite:
        validate_displacement(args.mode_index,args.coordinate_index,args.step_bohr,args.sign,result["atoms"])
    gate["v2_source_qualification"]={"seal":seal_ref,"prefix_verification":prefix_verification,
        "whole_mutable_ledger_hash_exception_scope":"EXACT_SEALED_PREFIX_AND_EXPLICIT_SCHEDULE_ONLY"}
    return gate


def direction_and_hessian(args, gate, np):
    result = gate["result"]
    n = result["atoms"]
    directory = args.source_result.resolve().parent
    raw_file = checked({"path": str(directory / "hessian_hartree_per_bohr2.npy"), "sha256": result["raw_hessian_sha256"]})
    raw = np.load(raw_file, allow_pickle=False)
    require(raw.dtype == np.float64 and raw.shape == (n, n, 3, 3) and np.isfinite(raw).all(), "Complete original raw Hessian required")
    references = {"raw_hessian": ref(raw_file)}
    if args.mode_index is None:
        q = np.zeros((n, 3), dtype=float)
        q.reshape(-1)[args.coordinate_index] = 1.
    else:
        require(isinstance(args.mode_array_sha256, str) and len(args.mode_array_sha256) == 64,
                "Root protocol must explicitly bind the previously audited mode array SHA")
        mode_file = checked({"path": str(directory / "normal_modes.npy"), "sha256": args.mode_array_sha256},
                            max_bytes=8 * 1024**2)
        # The global result records all actual modes. The independent analyzer
        # additionally verifies the matrix/mode replay, but no mode is re-fit here.
        mode_array = np.load(mode_file, allow_pickle=False)
        require(mode_array.dtype == np.float64
                and mode_array.shape == (3*n-6, n, 3) and np.isfinite(mode_array).all(), "Complete bounded saved mode vectors required")
        references["normal_modes"] = ref(mode_file)
        vector = mode_array[args.mode_index - 1]
        norm = float(np.linalg.norm(vector))
        require(norm > 0 and math.isfinite(norm), "Finite nonzero mode vector required")
        q = vector / norm
    require(np.isclose(np.sum(q*q), 1., atol=1e-12, rtol=0), "Euclidean Cartesian unit direction required")
    hq = np.einsum("abxy,by->ax", raw, q)
    curvature = float(np.sum(q * hq))
    return q, hq, curvature, references



def frozen_grid_type(Grids, protected_attributes):
    """Keep stock Grids setters during configuration; fail closed once attached."""
    protected = set(protected_attributes) | set(FIXED_ARRAY_NAMES) | {"mol", "symmetry", "_research4_fixed_attached"}
    class FixedOriginalGrids(Grids):
        def __setattr__(self, name, value):
            if getattr(self, "_research4_fixed_attached", False) and name in protected:
                raise RuntimeError("Fixed original quadrature mutation forbidden: " + name)
            super().__setattr__(name, value)
    def guarded(name):
        original = getattr(Grids, name)
        def call(self, *args, **kwargs):
            if getattr(self, "_research4_fixed_attached", False):
                raise RuntimeError("Fixed original quadrature operation forbidden: " + name)
            return original(self, *args, **kwargs)
        call.__name__ = name
        return call
    for name in ("build", "kernel", "reset", "gen_atomic_grids", "get_partition", "gen_partition",
                 "prune_by_density_", "_select_grids", "_add_padding"):
        if hasattr(Grids, name):
            setattr(FixedOriginalGrids, name, guarded(name))
    return FixedOriginalGrids


def grid_payload_fingerprints(grids, base):
    return {name: base.array_fingerprint(getattr(grids, name)) for name in FIXED_ARRAY_NAMES
            if getattr(grids, name, None) is not None}


def original_mf_fingerprints(original, base):
    return {"energy_hartree":float(original.e_tot), "converged":bool(original.converged),
            "small_rho_cutoff":original.small_rho_cutoff, "dispersion":original.disp,
            "MOs": {key: base.array_fingerprint(getattr(original, key))
                    for key in ("mo_coeff", "mo_energy", "mo_occ")},
            "grids": grid_payload_fingerprints(original.grids, base)}


def attach_fixed_quadrature(mf, original, w, base, np, gen_grid):
    """Configure every source grid option before attaching copied fixed arrays.

    A screening mask depends on the displaced AO basis. It is recomputed on
    the original coordinates for mf.mol, never copied from the original mask.
    Original padding/order/weights are retained byte for byte.
    """
    source = original.grids
    FixedGrids = frozen_grid_type(gen_grid.Grids, w.GRID_ATTRIBUTES)
    grids = FixedGrids(mf.mol)
    for name in w.GRID_ATTRIBUTES:
        if hasattr(source, name):
            setattr(grids, name, copy.deepcopy(getattr(source, name)))
    grids.symmetry = copy.deepcopy(source.symmetry)
    grids.stdout = mf.mol.stdout
    grids.verbose = source.verbose
    require(w.grid_settings(grids) == w.grid_settings(source), "Fixed grid configuration differs before attachment")
    coordinates = np.array(source.coords, copy=True, order="C")
    weights = np.array(source.weights, copy=True, order="C")
    # This zero-displacement consistency check is independent of the moved mask.
    zero_mask = gen_grid.make_mask(original.mol, coordinates, cutoff=source.cutoff)
    require(source.non0tab is not None and source.screen_index is not None
            and np.array_equal(zero_mask, source.non0tab)
            and np.array_equal(zero_mask, source.screen_index),
            "Recomputed source-geometry screening mask differs from saved source masks")
    moved_mask = gen_grid.make_mask(mf.mol, coordinates, cutoff=grids.cutoff)
    grids.coords, grids.weights = coordinates, weights
    grids.non0tab = moved_mask
    grids.screen_index = moved_mask
    for name in ("atm_idx", "quadrature_weights"):
        value = getattr(source, name, None)
        if value is not None:
            setattr(grids, name, np.array(value, copy=True, order="C"))
    for name in FIXED_ARRAY_NAMES:
        value = getattr(grids, name, None)
        if value is not None:
            value.flags.writeable = False
    grids._research4_fixed_attached = True
    mf.grids = grids
    mf.small_rho_cutoff = 0
    mf.nlc, mf.disp = original.nlc, original.disp
    mf._numint.cutoff, mf._numint.omega = original._numint.cutoff, original._numint.omega
    require(mf.mol is grids.mol and mf.grids is not source
            and not np.shares_memory(grids.coords, source.coords)
            and not np.shares_memory(grids.weights, source.weights),
            "A separate new molecule-owned copied quadrature is required")
    require(np.array_equal(grids.coords, source.coords) and np.array_equal(grids.weights, source.weights),
            "Fixed original quadrature coordinates/order/padding/weights changed")
    require(mf.disp is None and not mf.do_nlc(), "Frozen PBE must have no D3 or nonlocal correlation")
    proof = {"method": METHOD, "configuration": w.grid_settings(grids),
        "arrays_before": grid_payload_fingerprints(grids, base),
        "source_zero_mask_recomputed_and_equal_to_both_saved_masks": True,
        "moved_mask_recomputed_at_actual_geometry_on_original_coordinates": True,
        "non0tab_and_screen_index_share_actual_recomputed_mask": grids.non0tab is grids.screen_index,
        "grid_owns_actual_MF_molecule": grids.mol is mf.mol,
        "source_coordinates_and_weights_copied_without_reordering_or_padding_change": True,
        "small_rho_cutoff": mf.small_rho_cutoff, "grid_cutoff": grids.cutoff,
        "numint_cutoff": mf._numint.cutoff, "numint_omega": mf._numint.omega,
        "grid_rebuild_repartition_pruning_and_settings_mutation_forbidden": True,
        "all_attached_grid_arrays_readonly": True, "gradient_grid_response": False}
    return proof


def verify_fixed_quadrature(mf, proof, w, base):
    require(mf.grids.mol is mf.mol and mf.grids._research4_fixed_attached
            and mf.small_rho_cutoff == 0 and mf._numint.cutoff == proof["numint_cutoff"]
            and mf._numint.omega == proof["numint_omega"]
            and w.grid_settings(mf.grids) == proof["configuration"], "Fixed numerical settings changed")
    after = grid_payload_fingerprints(mf.grids, base)
    require(after == proof["arrays_before"] and mf.grids.non0tab is mf.grids.screen_index,
            "Attached fixed quadrature or geometry-specific screening masks changed")
    require(all(not getattr(mf.grids, name).flags.writeable for name in after), "Fixed array write protection changed")
    proof["arrays_after"] = after
    proof["fingerprints_unchanged_after_SCF"] = True


def finite_number(value):
    return isinstance(value,Real) and not isinstance(value,bool) and math.isfinite(value)


def strict_scf_decision(env, *, energy_tolerance=SCF_CONV_TOL, gradient_tolerance=SCF_CONV_TOL_GRAD):
    """Legacy nominal check calculator; v2 certifies full native endpoints."""
    energy, previous, gradient = env.get("e_tot"), env.get("last_hf_e"), env.get("norm_gorb")
    valid = all(finite_number(v) for v in (energy, previous, gradient)) and gradient >= 0
    drift = abs(energy-previous) if valid else None
    return {"accepted":bool(valid and drift <= energy_tolerance and gradient <= gradient_tolerance),
        "energy_hartree":float(energy) if finite_number(energy) else None,
        "previous_energy_hartree":float(previous) if finite_number(previous) else None,
        "absolute_energy_drift_hartree":float(drift) if drift is not None else None,
        "orbital_gradient_norm":float(gradient) if finite_number(gradient) else None,
        "energy_ulp_hartree":math.ulp(energy) if finite_number(energy) else None,
        "energy_drift_tolerance_hartree":energy_tolerance,"orbital_gradient_tolerance":gradient_tolerance,
        "requires_BOTH_energy_drift_AND_orbital_gradient":True,
        "extra_cycle_relaxation_allowed":False,"energy_drift_is_not_a_rigorous_energy_error_bound":True}



def polish_convergence_check(env, *, main_energy_tolerance, main_orbital_gradient_trigger,
                             final_energy_tolerance, final_orbital_gradient_tolerance):
    """Stock hf.kernel identifies its extra cycle by its local tolerance scaling."""
    final_extra = (env["conv_tol"] == 10*main_energy_tolerance
                   and env["conv_tol_grad"] == 3*main_orbital_gradient_trigger)
    energy_limit = final_energy_tolerance if final_extra else main_energy_tolerance
    gradient_limit = final_orbital_gradient_tolerance if final_extra else main_orbital_gradient_trigger
    drift = abs(float(env["e_tot"])-float(env["last_hf_e"]))
    gradient = float(env["norm_gorb"])
    return {"phase":"FINAL_EXTRA_CYCLE" if final_extra else "SCF_MAIN_CYCLE",
        "energy_delta_abs_hartree":drift,"orbital_gradient_norm":gradient,
        "energy_tolerance_hartree":energy_limit,"orbital_gradient_tolerance":gradient_limit,
        "accepted":bool(math.isfinite(drift) and math.isfinite(gradient)
            and drift < energy_limit and gradient < gradient_limit),
        "acceptance_rule":"ENERGY_AND_ORBITAL_GRADIENT_NO_RELAXED_EXTRA_CYCLE_OR"}



def finite_native_scf_check(check):
    keys = ("energy_delta_abs_hartree","orbital_gradient_norm","energy_tolerance_hartree","orbital_gradient_tolerance")
    return all(isinstance(check.get(key),(int,float)) and not isinstance(check[key],bool)
               and math.isfinite(check[key]) and check[key] >= 0 for key in keys)



def audited_endpoint_qualification(native_converged, last_check, previous_energy, current_energy,
                                   orbital_gradient_norm, *, preceding_main_check, native_checks_finite,
                                   density_change_norm, energy_tolerance, orbital_gradient_tolerance):
    drift = abs(current_energy-previous_energy)
    same_kernel = (type(last_check.get("kernel_call_1based")) is int
                   and 1 <= last_check["kernel_call_1based"] <= MAX_SCF_KERNEL_CALLS
                   and last_check.get("kernel_call_1based") == preceding_main_check.get("kernel_call_1based"))
    native_lineage = bool(native_checks_finite and same_kernel
        and last_check.get("phase") == "FINAL_EXTRA_CYCLE" and finite_native_scf_check(last_check)
        and preceding_main_check.get("phase") == "SCF_MAIN_CYCLE"
        and preceding_main_check.get("accepted") is True and finite_native_scf_check(preceding_main_check)
        and preceding_main_check["energy_delta_abs_hartree"] < preceding_main_check["energy_tolerance_hartree"]
        and preceding_main_check["orbital_gradient_norm"] < preceding_main_check["orbital_gradient_tolerance"])
    return {"certification_policy":SCF_CERTIFICATION_POLICY,
        "native_kernel_converged":bool(native_converged),"native_checks_finite":bool(native_checks_finite),
        "last_native_check":last_check,"preceding_accepted_main_check":preceding_main_check,
        "actual_native_final_extra_after_accepted_main_same_kernel":native_lineage,
        "previous_full_energy_hartree":previous_energy,
        "current_full_energy_hartree":current_energy,"full_consecutive_density_energy_delta_abs_hartree":drift,
        "full_current_orbital_gradient_norm":orbital_gradient_norm,
        "consecutive_density_Frobenius_difference":density_change_norm,
        "energy_tolerance_hartree":energy_tolerance,"orbital_gradient_tolerance":orbital_gradient_tolerance,
        "qualified":bool(native_lineage and math.isfinite(previous_energy)
            and math.isfinite(current_energy) and math.isfinite(orbital_gradient_norm)
            and math.isfinite(density_change_norm) and density_change_norm >= 0
            and drift < energy_tolerance and orbital_gradient_norm < orbital_gradient_tolerance)}



def bounded_polished_scf(mf, dm0, *, np, base, out, emit, state,
                         final_energy_tolerance, final_orbital_gradient_tolerance,
                         main_orbital_gradient_trigger, max_calls=MAX_SCF_KERNEL_CALLS,
                         max_cycles=SCF_MAX_CYCLES_PER_CALL, storage_out=None):
    """Select the first qualified SCF endpoint; no Hessian/error data enters.

    Every continuation starts from the immediately preceding actual final
    density, with a fresh local DIIS object. Every final audit independently
    evaluates the two consecutive densities from the actual last kernel step.
    No cached/incremental VEFF is passed to either full evaluation.
    """
    require(type(max_calls) is int and 1 <= max_calls <= 3 and type(max_cycles) is int
            and 1 <= max_cycles <= 100, "Bounded at most three SCF calls of at most 100 cycles required")
    require(mf.diis is True, "Each kernel call requires its own fresh native DIIS object")
    require(mf.grids.mol is mf.mol and mf.small_rho_cutoff == 0 and mf.direct_scf is True
            and mf.direct_scf_tol == 1e-13 and mf._numint.cutoff == 1e-13
            and mf.grids.cutoff == 1e-15, "Original native cutoffs and fixed-grid ownership required")
    storage_out = out if storage_out is None else Path(storage_out)
    require(storage_out.is_dir(), "Existing computed local raw-state directory required")
    state["large_audit_state_directory"] = str(storage_out)
    state["large_audit_states_storage_policy"] = "LOCAL_ONLY_NOT_FOR_GIT_ADD_BY_DEFAULT"
    hcore,overlap = mf.get_hcore(mf.mol),mf.get_ovlp(mf.mol)
    state["SCF_polishing_attempts"] = []
    state["strict_SCF_convergence_checks"] = []
    starting_density = np.array(dm0,copy=True)
    for call in range(1,max_calls+1):
        mf.conv_tol,mf.conv_tol_grad,mf.max_cycle = final_energy_tolerance,main_orbital_gradient_trigger,max_cycles
        mf.chkfile = str(storage_out/("scf-call-%d.chk" % call))
        attempt = {"kernel_call_1based":call,"maximum_cycles":max_cycles,
            "start_density_fingerprint":base.array_fingerprint(starting_density),
            "initial_guess_policy":"PROJECTED_SOURCE_DENSITY" if call == 1 else "IMMEDIATELY_PRECEDING_ACTUAL_FINAL_DENSITY",
            "main_energy_tolerance_hartree":final_energy_tolerance,
            "main_orbital_gradient_trigger":main_orbital_gradient_trigger,
            "final_energy_tolerance_hartree":final_energy_tolerance,
            "final_orbital_gradient_tolerance":final_orbital_gradient_tolerance,
            "qualified":False,"VEFF_audit_policy":"TWO_FULL_NATIVE_EVALUATIONS_DM_LAST_NONE_VHF_LAST_NONE"}
        state["SCF_polishing_attempts"].append(attempt)
        last_pair = {}
        def check_convergence(env):
            row = polish_convergence_check(env,main_energy_tolerance=final_energy_tolerance,
                main_orbital_gradient_trigger=main_orbital_gradient_trigger,
                final_energy_tolerance=final_energy_tolerance,
                final_orbital_gradient_tolerance=final_orbital_gradient_tolerance)
            row["kernel_call_1based"] = call
            state["strict_SCF_convergence_checks"].append(row)
            last_pair["previous"] = np.array(env["dm_last"],copy=True)
            last_pair["current"] = np.array(env["dm"],copy=True)
            last_pair["check"] = row
            if row["phase"] == "SCF_MAIN_CYCLE":
                last_pair["main_check"] = row
            emit("strict_SCF_convergence_check",row)
            return row["accepted"]
        def callback(env):
            emit("SCF_cycle",{"kernel_call_1based":call,"cycle_0based":int(env.get("cycle",-1)),
                "energy_hartree":float(env["e_tot"]) if "e_tot" in env else None})
        mf.check_convergence,mf.callback = check_convergence,callback
        emit("SCF_polishing_call_started",attempt)
        mf.kernel(dm0=starting_density)
        attempt["native_kernel_converged"] = bool(mf.converged)
        attempt["native_kernel_energy_hartree"] = float(mf.e_tot)
        require(last_pair and math.isfinite(mf.e_tot), "Actual consecutive final SCF densities and finite kernel energy required")
        current_dm = np.asarray(mf.make_rdm1())
        nao = mf.mol.nao_nr()
        require(np.array_equal(current_dm,last_pair["current"])
                and not np.shares_memory(last_pair["previous"],last_pair["current"])
                and current_dm.shape == last_pair["previous"].shape == (nao,nao)
                and np.isfinite(current_dm).all() and np.isfinite(last_pair["previous"]).all(),
                "Audited current density must equal the actual returned MO density; independent prior-step copy required")
        density_change_norm = float(np.linalg.norm(current_dm-last_pair["previous"]))
        previous_veff = mf.get_veff(mf.mol,last_pair["previous"],dm_last=None,vhf_last=None)
        previous_energy = float(mf.energy_tot(last_pair["previous"],hcore,previous_veff))
        current_veff = mf.get_veff(mf.mol,current_dm,dm_last=None,vhf_last=None)
        current_energy = float(mf.energy_tot(current_dm,hcore,current_veff))
        current_fock = mf.get_fock(h1e=hcore,s1e=overlap,vhf=current_veff,dm=current_dm,
            cycle=-1,diis=None,level_shift_factor=0,damp_factor=0)
        gradient_norm = float(np.linalg.norm(mf.get_grad(mf.mo_coeff,mf.mo_occ,current_fock)))
        audit = audited_endpoint_qualification(mf.converged,last_pair["check"],previous_energy,current_energy,
            gradient_norm,preceding_main_check=last_pair.get("main_check",{}),
            native_checks_finite=all(finite_native_scf_check(row) for row in state["strict_SCF_convergence_checks"]
                if row["kernel_call_1based"] == call),
            density_change_norm=density_change_norm,energy_tolerance=final_energy_tolerance,
            orbital_gradient_tolerance=final_orbital_gradient_tolerance)
        attempt.update(audit)
        attempt["full_current_minus_kernel_energy_hartree"] = current_energy-attempt["native_kernel_energy_hartree"]
        attempt["previous_density_fingerprint"] = base.array_fingerprint(last_pair["previous"])
        attempt["current_density_fingerprint"] = base.array_fingerprint(current_dm)
        attempt["density_change_Frobenius_norm"] = density_change_norm
        attempt["arrays"] = {}
        for name,value in (("start-density",starting_density),("previous-density",last_pair["previous"]),
                           ("current-density",current_dm),("MO-coeff",mf.mo_coeff),
                           ("MO-energy",mf.mo_energy),("MO-occ",mf.mo_occ)):
            path = storage_out/("scf-call-%d-%s.npy" % (call,name))
            base.atomic_npy(path,value)
            attempt["arrays"][name] = ref(path)
        if Path(mf.chkfile).is_file():
            attempt["checkpoint"] = {**ref(Path(mf.chkfile)),"storage_policy":"LOCAL_ONLY_NOT_FOR_GIT_ADD_BY_DEFAULT"}
        emit("SCF_polishing_call_completed",attempt)
        if audit["qualified"]:
            # Keep the exact audited MOs/density. The reported energy is the
            # full native evaluation of that density, not an incremental value.
            state["native_kernel_converged_at_certified_endpoint"] = bool(mf.converged)
            state["SCF_convergence_scope"] = SCF_CERTIFICATION_POLICY
            mf.converged = True  # Custom full-native certification; native flag remains in every raw record.
            mf.e_tot = current_energy
            mf.mo_energy = np.einsum("pi,pi->i",mf.mo_coeff,current_fock @ mf.mo_coeff)
            state["qualified_MF_array_files"] = {}
            for name,value in (("MO_coeff",mf.mo_coeff),("MO_energy",mf.mo_energy),
                               ("MO_occ",mf.mo_occ),("density",current_dm),("full_Fock",current_fock)):
                path = storage_out/("qualified-%s.npy" % name)
                base.atomic_npy(path,value)
                state["qualified_MF_array_files"][name] = ref(path)
            state["native_checkpoint_scope"] = "NATIVE_KERNEL_ENDPOINT_BEFORE_FULL_ENERGY_AND_MO_ENERGY_REFRESH"
            state["qualified_SCF_kernel_call_1based"] = call
            state["final_SCF_endpoint_audit"] = audit
            state["reported_energy_evaluation"] = "FULL_NONINCREMENTAL_NATIVE_VEFF_AT_AUDITED_RETURNED_MO_DENSITY"
            state["final_MO_energy_evaluation"] = "DIAGONAL_FULL_CURRENT_FOCK_IN_SAME_MO_BASIS_WITHOUT_ORBITAL_ROTATION"
            return audit
        starting_density = np.array(current_dm,copy=True)
    raise ValueError("No qualified final SCF endpoint after the bounded three-call polishing protocol")



def central_energy_curvature(zero, plus, minus, step=STEP_BOHR):
    require(all(finite_number(e) for e in (zero,plus,minus)), "Three finite real fresh SCF energies required")
    require(finite_number(step) and step > 0, "Positive finite central energy step required")
    numerator = math.fsum([plus,minus,-2*zero])
    curvature = numerator/(step*step)
    # This is a binary64 scale estimate, not a rigorously bounded DFT/SCF error.
    rounding_scale = (math.ulp(plus)+math.ulp(minus)+2*math.ulp(zero))/(step*step)
    cancellation = (abs(plus)+abs(minus)+2*abs(zero))/abs(numerator) if numerator else None
    return {"formula":"math.fsum([Eplus,Eminus,-2*FRESH_Ezero]) / h**2",
        "Ezero_hartree":zero,"Eplus_hartree":plus,"Eminus_hartree":minus,"step_bohr":step,
        "compensated_energy_numerator_hartree":numerator,"central_energy_curvature_Eh_per_Bohr2":curvature,
        "binary64_ulp_scale_Eh_per_Bohr2":rounding_scale,"energy_cancellation_ratio":cancellation,
        "nominal_4_times_SCF_drift_tolerance_over_h2_Eh_per_Bohr2":4*SCF_CONV_TOL/(step*step),
        "floating_point_and_drift_scales_are_not_rigorous_SCF_energy_error_bounds":True,
        "fresh_zero_energy_used":True,"archived_source_energy_used_in_numerator":False,
        "diagnostic_status":DIAGNOSTIC_STATUS,
        "Spinosin_step_convergence_verified":False,"scientific_PASS_FAIL_threshold":"NOT_SET"}


def raw_vector(value, *, rows=3, columns=3):
    require(isinstance(value,list) and len(value)==rows and all(isinstance(row,list) and len(row)==columns
        and all(finite_number(x) for x in row) for row in value), "Complete finite raw control vector required")
    return value


def matrix_close(left,right, *, atol=1e-12):
    raw_vector(left);raw_vector(right)
    return all(math.isclose(a,b,rel_tol=0,abs_tol=atol) for lrow,rrow in zip(left,right) for a,b in zip(lrow,rrow))


def checked_control_array(record, expected_shape):
    """Inspect bounded NPY float64 source arrays without numerical imports."""
    path=checked(record,max_bytes=8*1024**2)
    raw=path.read_bytes()
    require(raw[:6]==b"\x93NUMPY" and tuple(raw[6:8]) in ((1,0),(2,0),(3,0)),"Control array requires a real NPY file")
    version=raw[6];offset=10 if version==1 else 12
    length=struct.unpack("<H" if version==1 else "<I",raw[8:offset])[0]
    require(0<length<=65536 and offset+length<=len(raw),"Bounded complete NPY header required")
    header=ast.literal_eval(raw[offset:offset+length].decode("utf-8" if version==3 else "latin1").strip())
    shape=tuple(expected_shape)
    require(len(shape) in (1,2) and all(type(v) is int and 1<=v<=128 for v in shape)
        and header.get("shape")==shape and type(header.get("fortran_order")) is bool
        and header.get("descr") in ("<f8",">f8"),"Complete bounded float64 control array required")
    count=math.prod(shape);payload=raw[offset+length:]
    require(len(payload)==8*count,"Exact complete control array byte count required")
    flat=struct.unpack(("<" if header["descr"]=="<f8" else ">")+str(count)+"d",payload)
    require(all(math.isfinite(x) for x in flat),"Finite raw control array required")
    if len(shape)==1:return list(flat)
    rows,columns=shape
    if header["fortran_order"]:
        return [[flat[row+column*rows] for column in range(columns)] for row in range(rows)]
    return [list(flat[row*columns:(row+1)*columns]) for row in range(rows)]


def checked_control_matrix(record):
    return checked_control_array(record,(3,3))


def control_array_fingerprint(value):
    matrix=isinstance(value[0],list)
    shape=[len(value),len(value[0])] if matrix else [len(value)]
    flat=[x for row in value for x in row] if matrix else value
    payload=struct.pack("="+str(len(flat))+"d",*flat)
    return {"shape":shape,"dtype":"float64","payload_bytes":len(payload),
        "payload_sha256_C_order":hashlib.sha256(payload).hexdigest()}


def verify_control_endpoint(point, point_path):
    """Verify the new bounded-polish endpoint rather than old v1 booleans."""
    n=point.get("basis_functions")
    require(type(n) is int and 1<=n<=128,"Complete bounded water AO dimension required")
    decisions=point.get("strict_SCF_convergence_checks",[])
    call=point.get("qualified_SCF_kernel_call_1based")
    require(type(call) is int and 1<=call<=3 and isinstance(decisions,list) and decisions,
        "Qualified bounded SCF call and actual native check history required")
    last=decisions[-1]
    require(last.get("phase")=="FINAL_EXTRA_CYCLE" and last.get("kernel_call_1based")==call
        and type(last.get("accepted")) is bool and last.get("energy_tolerance_hartree")==1e-13
        and last.get("orbital_gradient_tolerance")==1e-9 and finite_native_scf_check(last),
        "Actual finite final native extra-cycle record must retain its original thresholds and raw result")
    current_checks=[row for row in decisions if row.get("kernel_call_1based")==call]
    mains=[row for row in current_checks if row.get("phase")=="SCF_MAIN_CYCLE"]
    require(mains and mains[-1].get("accepted") is True and mains[-1].get("energy_tolerance_hartree")==1e-13
        and mains[-1].get("orbital_gradient_tolerance")==1e-10 and finite_native_scf_check(mains[-1])
        and mains[-1]["energy_delta_abs_hartree"]<1e-13 and mains[-1]["orbital_gradient_norm"]<1e-10
        and all(finite_native_scf_check(row) for row in current_checks),
        "Final extra cycle must follow an actually accepted tight main cycle in the same native call")
    audit=point.get("final_SCF_endpoint_audit",{})
    require(audit.get("qualified") is True and audit.get("certification_policy")==SCF_CERTIFICATION_POLICY
        and type(audit.get("native_kernel_converged")) is bool
        and audit["native_kernel_converged"] is last["accepted"]
        and audit.get("native_checks_finite") is True
        and audit.get("actual_native_final_extra_after_accepted_main_same_kernel") is True
        and audit.get("preceding_accepted_main_check")==mains[-1]
        and point.get("SCF_convergence_scope")==SCF_CERTIFICATION_POLICY
        and point.get("native_kernel_converged_at_certified_endpoint") is audit["native_kernel_converged"]
        and audit.get("last_native_check")==last and audit.get("energy_tolerance_hartree")==1e-13
        and audit.get("orbital_gradient_tolerance")==1e-9
        and all(finite_number(audit.get(k)) for k in ("previous_full_energy_hartree","current_full_energy_hartree",
            "full_consecutive_density_energy_delta_abs_hartree","full_current_orbital_gradient_norm",
            "consecutive_density_Frobenius_difference"))
        and audit["current_full_energy_hartree"]==point.get("energy_hartree")
        and audit["full_consecutive_density_energy_delta_abs_hartree"]==
            abs(audit["current_full_energy_hartree"]-audit["previous_full_energy_hartree"])
        and 0<=audit["full_consecutive_density_energy_delta_abs_hartree"]<1e-13
        and 0<=audit["full_current_orbital_gradient_norm"]<1e-9
        and audit["consecutive_density_Frobenius_difference"]>=0
        and point.get("reported_energy_evaluation")=="FULL_NONINCREMENTAL_NATIVE_VEFF_AT_AUDITED_RETURNED_MO_DENSITY",
        "Reported water energy must be the strict full-native actual-consecutive-state certified endpoint")
    attempts=point.get("SCF_polishing_attempts",[])
    require(isinstance(attempts,list) and len(attempts)==call,"All bounded SCF attempts must be preserved")
    shapes={"start-density":(n,n),"previous-density":(n,n),"current-density":(n,n),
        "MO-coeff":(n,n),"MO-energy":(n,),"MO-occ":(n,)}
    final_arrays=None;previous_current=None
    for index,attempt in enumerate(attempts,1):
        require(attempt.get("kernel_call_1based")==index and attempt.get("maximum_cycles")==100
            and attempt.get("main_energy_tolerance_hartree")==1e-13
            and attempt.get("main_orbital_gradient_trigger")==1e-10
            and attempt.get("final_energy_tolerance_hartree")==1e-13
            and attempt.get("final_orbital_gradient_tolerance")==1e-9
            and attempt.get("VEFF_audit_policy")=="TWO_FULL_NATIVE_EVALUATIONS_DM_LAST_NONE_VHF_LAST_NONE"
            and attempt.get("qualified") is (index==call),"Actual finite per-call SCF audit policy differs")
        require(set(attempt.get("arrays",{}))==set(shapes),"Every raw SCF audit state must be retained")
        values={}
        for name,shape in shapes.items():
            record=attempt["arrays"][name]
            require(Path(record["path"]).resolve()==point_path.parent/("scf-call-%d-%s.npy"%(index,name)),
                "Water audit array must be its actual call sibling")
            values[name]=checked_control_array(record,shape)
        if previous_current is not None:
            require(values["start-density"]==previous_current,"Continuation must use immediately preceding actual density")
        previous_current=values["current-density"]
        require(attempt.get("start_density_fingerprint")==control_array_fingerprint(values["start-density"])
            and attempt.get("previous_density_fingerprint")==control_array_fingerprint(values["previous-density"])
            and attempt.get("current_density_fingerprint")==control_array_fingerprint(values["current-density"]),
            "SCF density fingerprints must match their complete raw arrays")
        checkpoint=attempt.get("checkpoint",{})
        require(Path(checkpoint.get("path","")).resolve()==point_path.parent/("scf-call-%d.chk"%index)
            and checkpoint.get("storage_policy")=="LOCAL_ONLY_NOT_FOR_GIT_ADD_BY_DEFAULT",
            "Every actual SCF call must retain its correctly scoped local checkpoint")
        checked(checkpoint)
        difference=math.sqrt(math.fsum((a-b)**2 for ar,br in zip(values["previous-density"],values["current-density"])
            for a,b in zip(ar,br)))
        require(finite_number(attempt.get("density_change_Frobenius_norm"))
            and math.isclose(difference,attempt["density_change_Frobenius_norm"],rel_tol=1e-12,abs_tol=1e-14),
            "Actual consecutive-state density change differs from raw audit arrays")
        if index==call:
            require(all(attempt.get(k)==v for k,v in audit.items()),"Final attempt must bind every qualified full-audit field")
            require(math.isclose(difference,audit["consecutive_density_Frobenius_difference"],rel_tol=1e-12,abs_tol=1e-14),
                "Final density-change audit differs from its raw consecutive states")
            require(finite_number(attempt.get("native_kernel_energy_hartree"))
                and finite_number(attempt.get("full_current_minus_kernel_energy_hartree"))
                and attempt["full_current_minus_kernel_energy_hartree"]==
                    audit["current_full_energy_hartree"]-attempt["native_kernel_energy_hartree"],
                "Full-versus-native-kernel energy difference must be retained accurately")
            final_arrays=values
    qualified=point.get("qualified_MF_array_files",{})
    qualified_shapes={"MO_coeff":(n,n),"MO_energy":(n,),"MO_occ":(n,),"density":(n,n),"full_Fock":(n,n)}
    require(set(qualified)==set(qualified_shapes),"All qualified full MF raw arrays must be retained")
    values={}
    for name,shape in qualified_shapes.items():
        require(Path(qualified[name]["path"]).resolve()==point_path.parent/("qualified-%s.npy"%name),
            "Qualified MF array must be its actual water-point sibling")
        values[name]=checked_control_array(qualified[name],shape)
    require(values["density"]==final_arrays["current-density"] and values["MO_coeff"]==final_arrays["MO-coeff"]
        and values["MO_occ"]==final_arrays["MO-occ"],"Qualified MOs/density must retain the actually audited native endpoint")
    occupied=[index for index,occupation in enumerate(values["MO_occ"]) if occupation>0]
    virtual=[index for index,occupation in enumerate(values["MO_occ"]) if occupation<=0]
    require(occupied and virtual and all(occupation in (0.,2.) for occupation in values["MO_occ"])
        and len(occupied)==point.get("occupied_orbitals"),"Complete closed-shell occupied/virtual water space required")
    coefficients,fock=values["MO_coeff"],values["full_Fock"]
    fock_occupied=[[math.fsum(fock[row][k]*coefficients[k][column] for k in range(n))
        for column in occupied] for row in range(n)]
    gradient=[2*math.fsum(coefficients[k][v]*fock_occupied[k][column] for k in range(n))
        for v in virtual for column in range(len(occupied))]
    full_gradient_norm=math.sqrt(math.fsum(x*x for x in gradient))
    require(math.isclose(full_gradient_norm,audit["full_current_orbital_gradient_norm"],rel_tol=1e-5,abs_tol=1e-11),
        "Full native orbital-gradient norm differs from qualified Fock/MO arrays")
    require(point.get("native_checkpoint_scope")=="NATIVE_KERNEL_ENDPOINT_BEFORE_FULL_ENERGY_AND_MO_ENERGY_REFRESH"
        and point.get("final_MO_energy_evaluation")=="DIAGONAL_FULL_CURRENT_FOCK_IN_SAME_MO_BASIS_WITHOUT_ORBITAL_ROTATION",
        "Native checkpoint and refreshed qualified MF scopes must remain explicit")
    return audit


def verify_control_release(args, guard, supervision):
    """Authenticate the real closed new suite, then recompute its raw gates."""
    require(isinstance(args.control_worker_sha256,str) and len(args.control_worker_sha256)==64,
            "Explicit root-bound water-control worker hash required")
    worker = checked({"path":str(CONTROL_WORKER),"sha256":args.control_worker_sha256},max_bytes=1024**2)
    record={"path":str(args.control_result.resolve()),"sha256":args.control_result_sha256}
    path=checked(record);result=read(path)
    require(result.get("schema")==CONTROL_SCHEMA and result.get("status")==CONTROL_SUCCESS
        and result.get("finest_pair_passed") is True
        and result.get("same_spinosin_step_energy_passed") is True
        and result.get("same_spinosin_energy_step_bohr")==STEP_BOHR and result.get("method")==METHOD
        and result.get("thresholds")==CONTROL_THRESHOLDS and result.get("case")=="water_software_control"
        and result.get("source_worker")==ref(worker), "Authentic new passing water step-convergence suite required")
    require(result.get("current_execution_boot_id")==Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "Passing water suite must belong to the current execution boot")
    proof=result["resource_supervision"]["proof"]
    require(type(proof.get("cpu_hard_limit_seconds")) is int and proof["cpu_hard_limit_seconds"]<=300
        and Path(proof["ledger_path"]).resolve()==Path(supervision["proof"]["ledger_path"]).resolve(),
        "Controls must share the cumulative ledger and one bounded <=300 CPU-second phase")
    closure=guard.verify_closed_phase(result,expected_worker_path=worker,expected_worker_sha256=args.control_worker_sha256,
        expected_result_schema=CONTROL_SCHEMA,allowed_result_statuses=(CONTROL_SUCCESS,),expected_output=path.parent)
    require(result.get("predetermined_coordinates_0based")==list(CONTROL_COORDINATES)
        and result.get("predetermined_steps_bohr")==list(CONTROL_STEPS) and result.get("predetermined_signs")==[-1,1]
        and result.get("all_31_prescribed_points_completed") is True
        and result.get("finest_pair_selection")==list(CONTROL_STEPS[-2:])
        and result.get("software_control_SCF_conv_tol_hartree")==1e-13
        and result.get("software_control_SCF_conv_tol_grad")==1e-9
        and result.get("all_points_including_zero_perform_fresh_SCF") is True
        and result.get("full_native_energy_AND_orbital_gradient_required_at_final_extra_endpoint") is True
        and result.get("native_kernel_flags_are_preserved_separate_from_custom_full_native_certification") is True
        and result.get("SCF_main_orbital_gradient_trigger")==1e-10
        and result.get("maximum_SCF_kernel_calls_per_point")==3
        and result.get("maximum_SCF_cycles_per_call")==100
        and result.get("final_energy_evaluation")=="FRESH_NONINCREMENTAL_NATIVE_VEFF_AT_QUALIFIED_CURRENT_DENSITY"
        and result.get("SCF_endpoint_certification_policy")==SCF_CERTIFICATION_POLICY
        and len(result.get("points",[]))==31
        and len(result.get("comparisons",[]))==15 and len(result.get("h_halving_comparisons",[]))==12,
        "Exact prescribed 31-point water coverage required")
    require(len({item["path"] for item in result["points"]})==31,"Duplicated control subpoint references rejected")
    points={}
    for point_ref in result["points"]:
        point=read(checked(point_ref))
        require(point.get("schema")==CONTROL_POINT_SCHEMA and point.get("status")==CONTROL_POINT_SUCCESS
            and point.get("method")==METHOD and point.get("gradient_completed") is True
            and point.get("SCF_performed") is True and point.get("scf_converged") is True
            and point.get("source_worker")==ref(worker)
            and point.get("source_snapshot_manifest_sha256")==result["snapshot_manifest_sha256"]
            and Path(point.get("parent_output_directory","")).resolve()==path.parent,
            "Control child must be a completed actual fresh SCF/gradient artifact")
        settings=point["settings"]
        require(settings.get("actual_conv_tol_hartree")==1e-13 and settings.get("actual_conv_tol_grad")==1e-10,
            "All control SCFs including zero require actual 1e-13 and the 1e-10 main orbital trigger")
        verify_control_endpoint(point,Path(point_ref["path"]).resolve())
        require(finite_number(point.get("energy_hartree")),"Control child energy must be finite")
        raw_vector(point.get("gradient_hartree_per_bohr"))
        artifacts=point.get("computed_array_files",{})
        names=("direction_cartesian_unit.npy","analytic_H_times_direction.npy","source_coordinates_bohr.npy",
            "displaced_coordinates_bohr.npy","gradient_hartree_per_bohr.npy")
        require(set(artifacts)==set(names),"Every completed water child must bind all five raw matrices")
        for name in names:
            matrix=checked_control_matrix(artifacts[name])
            require(Path(artifacts[name]["path"]).resolve()==Path(point_ref["path"]).resolve().parent/name,
                "Child array reference must be its actual sibling matrix")
            if name=="gradient_hartree_per_bohr.npy":
                require(matrix_close(matrix,point["gradient_hartree_per_bohr"],atol=0),
                    "Control child JSON gradient differs from its hash-bound raw matrix")
            if name=="analytic_H_times_direction.npy":
                point["_replayed_analytic_Hq"]=matrix
        displacement=point["displacement"]
        require(displacement.get("mode_index_1based") is None,"Only prescribed Cartesian water directions allowed")
        key=(displacement.get("coordinate_index_0based"),displacement.get("step_bohr"),displacement.get("sign"))
        require(key not in points,"Duplicate physical water subpoint rejected")
        points[key]=point
    expected={(None,0.,0)}|{(c,h,s) for c in CONTROL_COORDINATES for h in CONTROL_STEPS for s in (-1,1)}
    require(set(points)==expected,"Child artifacts do not cover the exact 31 actual fresh SCFs")
    zero=points[(None,0.,0)];comparisons={}
    for item in result["comparisons"]:
        coordinate,step=item.get("coordinate_index_0based"),item.get("step_bohr")
        require((coordinate,step) not in comparisons and coordinate in CONTROL_COORDINATES and step in CONTROL_STEPS,
            "Control comparisons must cover each prescribed direction/step once")
        plus,minus=points[(coordinate,step,1)],points[(coordinate,step,-1)]
        pg,mg=plus["gradient_hartree_per_bohr"],minus["gradient_hartree_per_bohr"]
        fd=[[(a-b)/(2*step) for a,b in zip(pr,mr)] for pr,mr in zip(pg,mg)]
        analytic=raw_vector(item.get("analytic_Hq_hartree_per_bohr2"))
        require(matrix_close(analytic,plus["_replayed_analytic_Hq"],atol=0)
            and matrix_close(analytic,minus["_replayed_analytic_Hq"],atol=0),
            "Parent analytic Hq differs from hash-bound child raw matrices")
        require(item.get("pair_completed") is True and item.get("fresh_zero_completed") is True
            and item.get("plus_energy_hartree")==plus["energy_hartree"]
            and item.get("minus_energy_hartree")==minus["energy_hartree"]
            and item.get("fresh_zero_energy_hartree")==zero["energy_hartree"],
            "Parent comparison must bind actual completed fresh child energies")
        require(matrix_close(fd,item.get("FD_Hq_hartree_per_bohr2")),"Control FD full vector differs from raw children")
        error=max(abs(a-b) for fr,ar in zip(fd,analytic) for a,b in zip(fr,ar))
        ec=central_energy_curvature(zero["energy_hartree"],plus["energy_hartree"],minus["energy_hartree"],step)
        curvature=analytic[coordinate//3][coordinate%3]
        energy_error=abs(ec["central_energy_curvature_Eh_per_Bohr2"]-curvature)
        require(item.get("analytic_directional_curvature_Eh_per_Bohr2")==curvature
            and finite_number(item.get("central_energy_curvature_Eh_per_Bohr2"))
            and math.isclose(item["central_energy_curvature_Eh_per_Bohr2"],
                ec["central_energy_curvature_Eh_per_Bohr2"],rel_tol=0,abs_tol=1e-12),
            "Parent directional/central curvature differs from raw controls")
        require(finite_number(item.get("FD_Hq_max_abs_error_Eh_per_Bohr2"))
            and math.isclose(error,item["FD_Hq_max_abs_error_Eh_per_Bohr2"],rel_tol=0,abs_tol=1e-12)
            and finite_number(item.get("central_energy_curvature_abs_error_Eh_per_Bohr2"))
            and math.isclose(energy_error,item["central_energy_curvature_abs_error_Eh_per_Bohr2"],rel_tol=0,abs_tol=1e-10),
            "Control reported errors differ from raw vector/energy recomputation")
        if step in CONTROL_STEPS[-2:]:
            require(error<=CONTROL_THRESHOLDS["FD_Hq_max_abs_error_Eh_per_Bohr2"]
                and energy_error<=CONTROL_THRESHOLDS["central_energy_curvature_abs_error_Eh_per_Bohr2"],
                "Finest two full-vector or energy-curvature water checks failed")
        if step==STEP_BOHR:
            require(energy_error<=CONTROL_THRESHOLDS["central_energy_curvature_abs_error_Eh_per_Bohr2"],
                "Water central-energy curvature failed at the exact intended Spinosin step")
        comparisons[(coordinate,step)]={"fd":fd,"analytic":analytic,"error":error,"energy_error":energy_error}
    require(len(comparisons)==15,"Missing prescribed water comparison")
    halvings=set()
    for item in result["h_halving_comparisons"]:
        coordinate,coarse,fine=item.get("coordinate_index_0based"),item.get("coarser_step_bohr"),item.get("finer_step_bohr")
        key=(coordinate,coarse,fine)
        require(coordinate in CONTROL_COORDINATES and (coarse,fine) in tuple(zip(CONTROL_STEPS[:-1],CONTROL_STEPS[1:]))
            and key not in halvings,"Halving records must cover exact adjacent control steps once")
        left,right=comparisons[(coordinate,coarse)]["fd"],comparisons[(coordinate,fine)]["fd"]
        difference=max(abs(a-b) for lr,rr in zip(left,right) for a,b in zip(lr,rr))
        require(finite_number(item.get("max_abs_difference_Eh_per_Bohr2")) and math.isclose(difference,
            item["max_abs_difference_Eh_per_Bohr2"],rel_tol=0,abs_tol=1e-12),"Halving error differs from raw children")
        if (coarse,fine)==CONTROL_STEPS[-2:]:
            require(difference<=CONTROL_THRESHOLDS["h_halving_FD_Hq_max_abs_difference_Eh_per_Bohr2"],
                "Finest-pair halving water check failed")
        halvings.add(key)
    require(len(halvings)==12,"Missing required water step-halving record")
    return {"result":ref(path),"control_worker":ref(worker),"closed_producer":closure,"control_passed":True,
        "all_31_child_artifacts_and_raw_comparisons_recomputed":True,"finest_two_steps_verified":list(CONTROL_STEPS[-2:]),
        "additional_water_energy_curvature_step_bohr":STEP_BOHR,"Spinosin_step_convergence_verified":False}


def validate_execution_envelope(proof, *, now=None):
    now=now or datetime.now(timezone.utc)
    require(type(proof.get("cpu_hard_limit_seconds")) is int and 0<proof["cpu_hard_limit_seconds"]<=7500,
        "Energy diagnostic must fit its one <=7500 CPU-second phase")
    require(finite_number(proof.get("actual_cpu_stop_threshold_seconds"))
        and 0<proof["actual_cpu_stop_threshold_seconds"]<=7380,
        "Actual-CPU stop must reserve cleanup and occur by 7380 CPU seconds")
    start=datetime.fromisoformat(proof["started_at_utc"])
    require(start.tzinfo is not None and start.utcoffset()==timedelta(0)
        and now.tzinfo is not None and start<=now<STRICT_DEADLINE
        and type(proof.get("wall_limit_seconds")) is int and 61<=proof["wall_limit_seconds"]<=43200
        and start+timedelta(seconds=proof["wall_limit_seconds"])<=STRICT_DEADLINE,
        "Real started UTC proof and entire finite wall envelope must fit the earlier fixed deadline")


def execute(args):
    guard=load(GUARD,GUARD_SHA,"fixed_energy_curvature_v8_guard")
    supervision=guard.verify_resource_supervision(expected_threads=2)
    require(args.threads==2,"Exactly two native/direct-driver threads required")
    validate_execution_envelope(supervision["proof"])
    sole_phase(guard,supervision)
    control_release=verify_control_release(args,guard,supervision)
    analyzer=load(ANALYZER,ANALYZER_SHA,"fixed_energy_curvature_source_analyzer")
    with closed_main(args,analyzer) as main_ref:
        gate=source_gate(args,analyzer,guard,supervision)
        return execute_closed(args,analyzer,gate,supervision,main_ref,control_release)

def ensure_original_unchanged(original, fingerprint, payload, w, base):
    require(w.molecule_physical_fingerprint(original.mol)==fingerprint,
        "Original source molecule changed during energy diagnostic")
    require(original_mf_fingerprints(original,base)==payload,
        "Original source MF/MO/energy/quadrature changed during energy diagnostic")


def local_state_directory(job_output, point_name):
    job=Path(job_output).resolve()
    require(job.parent==ENERGY_WORK_DIRECTORY/"science" and point_name in ("fresh-zero","plus","minus"),
        "Large state storage requires the exact A006 science job and predetermined point")
    path=ENERGY_WORK_DIRECTORY/"local-checkpoints"/job.name/point_name
    require(not any(p.is_symlink() for p in (path,*path.parents)),"Local state namespace cannot contain symlinks")
    require(not path.exists(),"Preserve prior local raw audit states; a fresh job/point is required")
    return path


def compute_energy_point(args, gate, original, binding, w, base, np, gen_grid, addons,
                         out, q, sign, *, fresh_zero=None, check_budget=lambda:None):
    """One real fresh SCF. Neither nuclear gradients nor Hessians are evaluated."""
    out.mkdir()
    storage=local_state_directory(args.output,out.name)
    storage.mkdir(parents=True)
    zero=sign==0
    state={"schema":POINT_SCHEMA,"status":"PARTIAL","method":METHOD,"source_worker":ref(__file__),
        "output_directory":str(out),"parent_output_directory":str(args.output.resolve()),
        "nested_point_under_parent_guardian_phase":True,
        "source_snapshot_manifest_sha256":gate["result"]["snapshot_manifest_sha256"],
        "displacement":{"mode_index_1based":MODE_INDEX,"coordinate_index_0based":None,
            "step_bohr":0. if zero else STEP_BOHR,"sign":sign,
            "direction_normalization":"EUCLIDEAN_CARTESIAN_UNIT_NORM"},
        "SCF_performed":False,"scf_converged":False,"energy_completed":False,
        "nuclear_gradient_performed":False,"Hessian_performed":False,
        "diagnostic_status":DIAGNOSTIC_STATUS,"IR_intensities_status":"NOT_COMPUTED",
        "local_state_storage":{"path":str(storage),"storage_policy":"LOCAL_ONLY_NOT_FOR_GIT_ADD_BY_DEFAULT",
            "all_raw_states_retained":True,"outside_recursive_scientific_job_artifact_collection":True}}
    cpu,wall=time.process_time(),time.perf_counter()
    log=(out/"pyscf.log").open("x",buffering=1)
    mf=None
    def emit(event,detail=None):
        row={"utc":utc(),"event":event,"CPU_seconds":time.process_time()-cpu,
             "wall_seconds":time.perf_counter()-wall,**(detail or {})}
        with (out/"events.jsonl").open("a") as handle:
            handle.write(json.dumps(row,allow_nan=False)+"\n")
        save(out/"progress.json",{**state,**row})
    try:
        check_budget()
        n,nao=original.mol.natm,original.mol.nao_nr()
        source_xyz=original.mol.atom_coords(unit="Bohr")
        xyz=source_xyz+sign*STEP_BOHR*q
        mol=original.mol.copy();mol.set_geom_(xyz,unit="Bohr",inplace=True);mol.stdout=log
        require(n==75 and nao==762 and mol.nao_nr()==nao and mol.charge==0 and mol.spin==0
            and np.array_equal(mol.atom_coords(unit="Bohr"),xyz),"Exact full-space Spinosin geometry required")
        settings=dict(binding["method_settings"])
        require(settings["xc"]=="PBE" and settings["basis"].lower()=="def2-svp" and settings["grid_level"]==1
            and settings["dispersion"]=="NONE_EXPLICIT_PBE_PILOT_NOT_D3"
            and settings["gradient_grid_response"] is False,"Original complete PBE/def2-SVP/grid1/no-D3 method required")
        settings.update(threads=2,native_direct_driver_threads=2,scf_conv_tol_hartree=SCF_CONV_TOL)
        mf=w.make_scf(mol,settings,storage/"scf-initial.chk");mf.conv_tol_grad=SCF_MAIN_ORBITAL_GRADIENT_TRIGGER
        mf.max_cycle=SCF_MAX_CYCLES_PER_CALL
        fixed=attach_fixed_quadrature(mf,original,w,base,np,gen_grid)
        settings.update(scientific_quadrature_method=METHOD,small_rho_cutoff=0,
            actual_conv_tol_hartree=mf.conv_tol,actual_conv_tol_grad=mf.conv_tol_grad,
            native_grid_cutoff=mf.grids.cutoff,native_numint_cutoff=mf._numint.cutoff,
            strict_energy_AND_orbital_gradient_check_including_extra_cycle=True,
            final_energy_tolerance_hartree=SCF_CONV_TOL,final_orbital_gradient_tolerance=SCF_CONV_TOL_GRAD,
            SCF_main_orbital_gradient_trigger=SCF_MAIN_ORBITAL_GRADIENT_TRIGGER,
            maximum_SCF_kernel_calls_per_point=MAX_SCF_KERNEL_CALLS,
            maximum_SCF_cycles_per_call=SCF_MAX_CYCLES_PER_CALL,
            gradient_grid_response=False,nuclear_gradient_performed=False)
        require(mf.conv_tol==SCF_CONV_TOL and mf.conv_tol_grad==SCF_MAIN_ORBITAL_GRADIENT_TRIGGER,
            "Predetermined strict Spinosin SCF thresholds changed")
        if zero:
            require(fresh_zero is None,"Fresh zero must initialize only from original source density")
            dm0=np.array(original.make_rdm1(),copy=True)
            guess="EXACT_ORIGINAL_SOURCE_MF_DENSITY_FRESH_ZERO_SCF"
        else:
            require(fresh_zero is not None and fresh_zero.converged,"Both signs require successful fresh ZERO density")
            dm0=addons.project_dm_nr2nr(fresh_zero.mol,fresh_zero.make_rdm1(),mol)
            guess="PROJECTED_SAME_FRESH_ZERO_DENSITY_TO_DISPLACED_FULL_AO_BASIS"
        require(dm0.shape==(nao,nao) and np.isfinite(dm0).all(),"Complete finite initial density required")
        state.update(settings=settings,atoms=n,basis_functions=nao,
            occupied_orbitals=int(np.count_nonzero(original.mo_occ>0)),initial_guess=guess,
            initial_density_fingerprint=base.array_fingerprint(dm0),fixed_quadrature_proof=fixed,
            archived_source_energy_hartree=float(original.e_tot),
            archived_source_energy_used_in_numerator=False,
            fresh_zero_energy_hartree=None if zero else float(fresh_zero.e_tot))
        for name,value in (("direction_cartesian_unit.npy",q),("source_coordinates_bohr.npy",source_xyz),
                           ("actual_coordinates_bohr.npy",xyz)):
            base.atomic_npy(out/name,value)
        base.atomic_npy(storage/"initial_full_density.npy",dm0)
        state["initial_full_density_local_file"]={**ref(storage/"initial_full_density.npy"),
            "storage_policy":"LOCAL_ONLY_NOT_FOR_GIT_ADD_BY_DEFAULT"}
        save(out/"input.json",state,exclusive=True);state["input_json_sha256"]=sha(out/"input.json")
        state["SCF_performed"]=True
        emit("fresh_SCF_started",{"sign":sign,"no_nuclear_gradient":True,
            "bounded_internal_SCF_polishing_is_not_another_geometry_or_job_retry":True})
        def polish_emit(event,detail=None):
            check_budget()
            emit(event,detail)
        bounded_polished_scf(mf,dm0,np=np,base=base,out=out,storage_out=storage,emit=polish_emit,state=state,
            final_energy_tolerance=SCF_CONV_TOL,final_orbital_gradient_tolerance=SCF_CONV_TOL_GRAD,
            main_orbital_gradient_trigger=SCF_MAIN_ORBITAL_GRADIENT_TRIGGER)
        require(mf.converged and finite_number(mf.e_tot) and state["final_SCF_endpoint_audit"]["qualified"],
            "SCF endpoint lacks the strict native and full nonincremental consecutive-state audit")
        require(mf.mo_coeff.shape==(nao,nao) and mf.mo_energy.shape==(nao,) and mf.mo_occ.shape==(nao,)
            and np.count_nonzero(mf.mo_occ>0)==state["occupied_orbitals"]
            and int(mf.mo_occ.sum())==original.mol.nelectron
            and np.isfinite(mf.mo_coeff).all() and np.isfinite(mf.mo_energy).all() and np.isfinite(mf.mo_occ).all(),
            "Full finite AO/MO/electron space changed")
        verify_fixed_quadrature(mf,fixed,w,base)
        state.update(scf_converged=True,energy_completed=True,energy_hartree=float(mf.e_tot),status=POINT_SUCCESS,
            full_AO_MO_space_preserved=True,
            final_full_density_fingerprint=base.array_fingerprint(mf.make_rdm1()))
        emit("energy_point_completed",{"energy_hartree":float(mf.e_tot),"strict_convergence_passed":True})
    except BaseException as exc:
        state["status"]="PARTIAL_RESOURCE_STOP" if isinstance(exc,base.BudgetStop) else "FAILED_ENERGY_POINT"
        state["error"]={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        if mf is not None and finite_number(getattr(mf,"e_tot",None)):
            state["last_iterate_energy_hartree_NOT_COMPLETED"]=float(mf.e_tot)
        emit("energy_point_failed",{"error":state["error"]})
        if isinstance(exc,base.BudgetStop):
            raise
    finally:
        state.update(ended_at_utc=utc(),worker_CPU_seconds=time.process_time()-cpu,worker_wall_seconds=time.perf_counter()-wall)
        log.close()
        state["saved_raw_array_files"]={name:ref(out/name) for name in
            ("direction_cartesian_unit.npy","source_coordinates_bohr.npy","actual_coordinates_bohr.npy")
            if (out/name).is_file()}
        if mf is not None and Path(mf.chkfile).is_file():
            state["checkpoint"]={**ref(Path(mf.chkfile)),"storage_policy":"LOCAL_ONLY_NOT_FOR_GIT_ADD_BY_DEFAULT"}
        save(out/"native_result.json",state,exclusive=True)
    return state,mf


def completed_energy_points(out):
    result={}
    for name in ("fresh-zero","plus","minus"):
        path=out/name/"native_result.json"
        if path.is_file():
            value=read(path)
            result[name]={"result":ref(path),"status":value.get("status"),
                "energy_completed":value.get("energy_completed") is True,
                "energy_hartree":value.get("energy_hartree") if value.get("energy_completed") is True else None}
    return result


def execute_closed(args, analyzer, gate, supervision, main_ref, control_release):
    import numpy as np
    from pyscf import lib
    from pyscf.dft import gen_grid
    from pyscf.scf import _vhf, addons
    lib.num_threads(2)
    w=load(analyzer.STATE,analyzer.STATE_SHA,"fixed_energy_curvature_original_state_restorer")
    base=w.frozen_base();out=fresh_output(args.output);source=gate["result"]
    state={"schema":ENERGY_SCHEMA,"status":"PARTIAL","stage":"fixed-original-grid-central-energy-curvature",
        "method":METHOD,"diagnostic_status":DIAGNOSTIC_STATUS,"started_at_utc":utc(),
        "source_worker":ref(__file__),"resource_supervision":supervision,"output_directory":str(out),
        "source_result":ref(args.source_result),"snapshot_manifest_sha256":source["snapshot_manifest_sha256"],
        "case":source["case"],"v2_source_qualification":gate["v2_source_qualification"],
        "closed_main_pipeline_state":main_ref,"passing_control_release":control_release,
        "source_boot_id":source["boot_id"],
        "current_execution_boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "historical_source_gate_validated_against_its_own_boot":True,"stable_MF_state_unchanged":False,
        "stationary_minimum_verified":False,"chemical_assignment_verified":False,
        "IR_intensities_status":"NOT_COMPUTED","new_full_Hessian":"NOT_COMPUTED",
        "nuclear_gradients":"NOT_COMPUTED","new_frequencies":"NOT_COMPUTED",
        "old_scientific_results_modified":False,"new_classifier_training":False,
        "predetermined_mode_index_1based":MODE_INDEX,"predetermined_step_bohr":STEP_BOHR,
        "predetermined_fresh_SCF_order":["fresh-zero","plus","minus"],
        "actual_conv_tol_hartree":SCF_CONV_TOL,"actual_conv_tol_grad":SCF_MAIN_ORBITAL_GRADIENT_TRIGGER,
        "final_energy_tolerance_hartree":SCF_CONV_TOL,"final_orbital_gradient_tolerance":SCF_CONV_TOL_GRAD,
        "maximum_SCF_kernel_calls_per_point":MAX_SCF_KERNEL_CALLS,
        "maximum_SCF_cycles_per_call":SCF_MAX_CYCLES_PER_CALL,
        "energy_evaluation":"FULL_NONINCREMENTAL_NATIVE_VEFF_AT_AUDITED_RETURNED_MO_DENSITY",
        "SCF_endpoint_certification_policy":SCF_CERTIFICATION_POLICY,
        "Spinosin_step_convergence_verified":False,"automatic_retry_allowed":False,"points":{}}
    save(out/"input.json",{**state,"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},exclusive=True)
    state["input_json_sha256"]=sha(out/"input.json");cpu,wall=time.process_time(),time.perf_counter()
    def emit(event,detail=None):
        row={"utc":utc(),"event":event,"CPU_seconds":time.process_time()-cpu,
             "wall_seconds":time.perf_counter()-wall,**(detail or {})}
        with (out/"events.jsonl").open("a") as handle:
            handle.write(json.dumps(row,allow_nan=False)+"\n")
        save(out/"progress.json",{**state,**row})
    def check_budget():
        require(datetime.now(timezone.utc)<STRICT_DEADLINE,"Strict earlier deadline reached")
        if time.process_time()>=supervision["proof"]["actual_cpu_stop_threshold_seconds"]:
            raise base.BudgetStop("Finite single-phase actual-CPU cleanup threshold reached")
    def stop(signum,frame):
        raise base.BudgetStop("Bounded energy curvature diagnostic received resource signal "+str(signum))
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGXCPU)}
    native=base.NativeDirectDriverThreadGuard(_vhf,lib.with_omp_threads,lib.num_threads,
        native_threads=2,emit=lambda detail:emit("native_direct_driver_signature",detail))
    original=None;restoration_log=(out/"restoration.log").open("x",buffering=1)
    try:
        native.install();check_budget()
        original,binding=w.restore_snapshot(Path(source["snapshot_directory"]),
            expected_manifest_sha256=source["snapshot_manifest_sha256"],expected_source_sha256=analyzer.STATE_SHA,
            stdout=restoration_log)
        require(binding==source["source_state_binding"],"Restored original binding differs")
        fingerprint=w.molecule_physical_fingerprint(original.mol);payload=original_mf_fingerprints(original,base)
        require(original.mol.natm==75 and original.mol.nao_nr()==762 and original.mo_coeff.shape==(762,762)
            and original.mol.charge==0 and original.mol.spin==0
            and original.mol.nelectron==int(original.mo_occ.sum()),"Complete original 75-atom 762-AO/MO source required")
        state.update(atoms=75,basis_functions=762,reference_state_binding=binding,
            source_geometry_fingerprint=fingerprint,original_MF_payload_fingerprints=payload,
            archived_source_energy_hartree=float(original.e_tot),archived_source_energy_used_in_numerator=False)
        q,hq,analytic,references=direction_and_hessian(args,gate,np)
        state.update(source_arrays=references,analytic_directional_curvature_Eh_per_Bohr2=analytic,
            source_signed_frequency_cm1=source["frequencies_cm1"][MODE_INDEX-1])
        base.atomic_npy(out/"mode1_cartesian_unit_direction.npy",q)
        def point(name,sign,fresh_zero=None):
            check_budget();emit("energy_point_starting",{"point":name,"sign":sign})
            value,mf=compute_energy_point(args,gate,original,binding,w,base,np,gen_grid,addons,
                out/name,q,sign,fresh_zero=fresh_zero,check_budget=check_budget)
            state["points"]=completed_energy_points(out)
            ensure_original_unchanged(original,fingerprint,payload,w,base);checked(state["source_result"])
            require(value["status"]==POINT_SUCCESS,"A required fresh energy point failed; no automatic retry")
            return value,mf
        e0,zero_mf=point("fresh-zero",0)
        zero_fingerprint=w.molecule_physical_fingerprint(zero_mf.mol)
        zero_payload=original_mf_fingerprints(zero_mf,base)
        state["fresh_zero_MF_payload_fingerprints"]=zero_payload
        ep,_=point("plus",1,fresh_zero=zero_mf)
        ensure_original_unchanged(zero_mf,zero_fingerprint,zero_payload,w,base)
        em,_=point("minus",-1,fresh_zero=zero_mf)
        ensure_original_unchanged(zero_mf,zero_fingerprint,zero_payload,w,base)
        state["same_fresh_zero_MF_and_density_preserved_for_both_signs"]=True
        state["central_energy_analysis"]=central_energy_curvature(e0["energy_hartree"],ep["energy_hartree"],em["energy_hartree"])
        state["central_energy_analysis"].update(analytic_directional_curvature_Eh_per_Bohr2=analytic,
            central_energy_minus_analytic_curvature_Eh_per_Bohr2=
            state["central_energy_analysis"]["central_energy_curvature_Eh_per_Bohr2"]-analytic)
        ensure_original_unchanged(original,fingerprint,payload,w,base)
        state["stable_MF_state_unchanged"]=True;state["status"]=ENERGY_SUCCESS
        emit("central_energy_curvature_completed",state["central_energy_analysis"])
    except BaseException as exc:
        state["status"]="PARTIAL_RESOURCE_STOP" if isinstance(exc,base.BudgetStop) else "FAILED_ENERGY_DIAGNOSTIC"
        state["error"]={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        emit("energy_diagnostic_stopped",{"status":state["status"],"error":state["error"]})
    finally:
        state["points"]=completed_energy_points(out)
        state["all_three_fresh_energies_completed"]=len(state["points"])==3 and all(p["energy_completed"] for p in state["points"].values())
        state["completed_points"]=[name for name,point in state["points"].items() if point["energy_completed"]]
        state["energies_hartree"]={name:point["energy_hartree"] for name,point in state["points"].items()
            if point["energy_completed"]}
        state["energies"]=dict(state["energies_hartree"])
        if state["status"]==ENERGY_SUCCESS and not state["all_three_fresh_energies_completed"]:
            state["status"]="FAILED_INCOMPLETE_ENERGY_TRIPLET";state.pop("central_energy_analysis",None)
        if original is not None and "payload" in locals():
            try:
                ensure_original_unchanged(original,fingerprint,payload,w,base)
                state["stable_MF_state_unchanged"]=True
            except ValueError as exc:
                state["stable_MF_state_unchanged"]=False;state["status"]="FAILED_ORIGINAL_SOURCE_MUTATED"
                state["source_mutation_error"]=str(exc)
            state["original_MF_payload_after"]=original_mf_fingerprints(original,base)
        native.restore();state["native_direct_driver_control"]=native.summary()
        if state["native_direct_driver_control"].get("restoration_failures")!=0 or not state["native_direct_driver_control"].get("original_driver_restored"):
            state["status"]="FAILED_NATIVE_DRIVER_RESTORATION"
        state.update(ended_at_utc=utc(),worker_CPU_seconds=time.process_time()-cpu,worker_wall_seconds=time.perf_counter()-wall)
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
        restoration_log.close();save(out/"result.json",state,exclusive=True)
    return state


def validate_actual_arguments(args):
    require(args.source_result is not None and args.source_result_sha256 and args.main_pipeline_state is not None
        and args.main_pipeline_state_sha256 and args.source_qualification is not None and args.source_qualification_sha256,
        "Authenticated source/main state and own outside-Q source qualification required")
    require(args.allowed_phase_schedule is None and args.allowed_phase_schedule_sha256 is None,
        "Execution obtains exact finite phase schedule only from its sealed qualification")
    require(args.threads==2 and args.mode_index==MODE_INDEX and args.step_bohr==STEP_BOHR
        and args.sign==1 and args.coordinate_index is None and not args.control_suite and not args.control_validation,
        "Only the fixed complete mode-1 central-energy triplet is authorized")
    require(isinstance(args.mode_array_sha256,str) and len(args.mode_array_sha256)==64
        and args.control_result is not None and args.control_result_sha256
        and isinstance(args.control_worker_sha256,str) and len(args.control_worker_sha256)==64,
        "Root-bound full saved modes and authentic new water control/worker hashes required")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    action=p.add_mutually_exclusive_group();action.add_argument("--execute",action="store_true")
    action.add_argument("--qualify-source-only",action="store_true")
    p.add_argument("--source-result",type=Path);p.add_argument("--source-result-sha256")
    p.add_argument("--source-qualification",type=Path);p.add_argument("--source-qualification-sha256")
    p.add_argument("--allowed-phase-schedule",type=Path);p.add_argument("--allowed-phase-schedule-sha256")
    p.add_argument("--main-pipeline-state",type=Path);p.add_argument("--main-pipeline-state-sha256")
    p.add_argument("--mode-array-sha256");p.add_argument("--threads",type=int,default=2)
    p.add_argument("--control-result",type=Path);p.add_argument("--control-result-sha256")
    p.add_argument("--control-worker-sha256")
    # Subpoints cannot be chosen from the command line. The fixed internal
    # values also allow reuse of the fully preserved original source gate.
    p.set_defaults(mode_index=MODE_INDEX,coordinate_index=None,step_bohr=STEP_BOHR,sign=1,
        control_suite=False,control_validation=False)
    args=p.parse_args()
    if args.execute:
        validate_actual_arguments(args);value=execute(args)
    elif args.qualify_source_only:
        require(args.source_result is not None and args.source_result_sha256 and args.main_pipeline_state is not None
            and args.main_pipeline_state_sha256 and args.allowed_phase_schedule is not None
            and args.allowed_phase_schedule_sha256,"Outside-Q qualification requires exact source/main and finite phase schedule")
        require(all(v is None for v in (args.mode_array_sha256,args.source_qualification,args.source_qualification_sha256,
            args.control_result,args.control_result_sha256,args.control_worker_sha256)),
            "Source qualification does not load modes or consume future control/seal results")
        value=qualify_source(args)
    else:
        require(all(getattr(args,key) is None for key in ("source_result","source_result_sha256","source_qualification",
            "source_qualification_sha256","allowed_phase_schedule","allowed_phase_schedule_sha256","main_pipeline_state",
            "main_pipeline_state_sha256","mode_array_sha256","control_result","control_result_sha256","control_worker_sha256")),
            "Preparation does not read target numerical inputs")
        out=fresh_output(args.output)
        value={"schema":ENERGY_SCHEMA,"status":"NOT_RUN","prepared_at_utc":utc(),"source_worker":ref(__file__),
            "method":METHOD,"diagnostic_status":DIAGNOSTIC_STATUS,"new_quantum_evaluations":0,
            "numerical_packages_imported":False,"target_arrays_read":False,"automatic_retry_allowed":False,
            "predetermined_mode_index_1based":MODE_INDEX,"predetermined_step_bohr":STEP_BOHR,
            "actual_conv_tol_hartree":SCF_CONV_TOL,"actual_conv_tol_grad":SCF_MAIN_ORBITAL_GRADIENT_TRIGGER,
        "final_energy_tolerance_hartree":SCF_CONV_TOL,"final_orbital_gradient_tolerance":SCF_CONV_TOL_GRAD,
        "maximum_SCF_kernel_calls_per_point":MAX_SCF_KERNEL_CALLS,
        "maximum_SCF_cycles_per_call":SCF_MAX_CYCLES_PER_CALL,
        "energy_evaluation":"FULL_NONINCREMENTAL_NATIVE_VEFF_AT_AUDITED_RETURNED_MO_DENSITY",
        "SCF_endpoint_certification_policy":SCF_CERTIFICATION_POLICY,
            "IR_intensities_status":"NOT_COMPUTED"}
        save(out/"preflight.json",value,exclusive=True)
    print(json.dumps({key:value.get(key) for key in ("schema","status","case","all_three_fresh_energies_completed")},ensure_ascii=False))
    return 0 if value["status"] in ("NOT_RUN",QUALIFICATION_SUCCESS,ENERGY_SUCCESS) else 2


if __name__=="__main__":
    raise SystemExit(main())
