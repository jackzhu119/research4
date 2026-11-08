"""Water-only fixed-grid convergence with authoritative full-native endpoint certification.

This separate protocol retains every coarse failed measurement. It performs
31 fresh tight SCF/gradient points, including zero, on copied exact original
quadrature. Completion and predetermined finest-pair numerical acceptance are
separate results. It cannot execute or release the old Spinosin point.
The original workers, historical arrays/results, and cumulative budget remain
unchanged. Source qualification is the complete original outside-Q gate.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts/run_pyscf_continuation_budget_20261105_v8.py"
GUARD_SHA = "7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65"
ANALYZER = ROOT / "scripts/analyze_pyscf_staged_modes_20261105_v8.py"
ANALYZER_SHA = "da2a53d85015e0ced6445e7766e18ae4fbe58e098e0eb077719759be7b18cab6"
AUDITOR = ROOT / "scripts/audit_pyscf_saved_cphf_residual_20261105.py"
AUDITOR_SHA = "6ff1df9930257f33637a34cfc58ba01b0fe38c246458ab704891663efbef800c"
POINT_SCHEMA = "research4-fixed-grid-water-convergence-point-v3"
POINT_SUCCESS = "WATER_FIXED_GRID_CERTIFIED_SCF_GRADIENT_COMPUTED"
CONTROL_SCHEMA = "research4-fixed-grid-water-step-convergence-v3"
CONTROL_SUCCESS = "WATER_FIXED_GRID_STEP_CONVERGENCE_FULL_ENDPOINT_MEASURED"
QUALIFICATION_SCHEMA = "research4-fixed-grid-water-convergence-source-qualification-v3"
QUALIFICATION_SUCCESS = "SOURCE_QUALIFIED_OUTSIDE_QUANTUM_PHASE"
SCHEDULE_SCHEMA = "research4-cloud-followup-phase-schedule-v1"
SOURCE_SNAPSHOT = "3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311"
METHOD = "FIXED_ORIGINAL_QUADRATURE"
DIAGNOSTIC_STATUS = "NOT_HESSIAN_OR_MINIMUM"
SPINOSIN_GRADIENT_SHA = "9418727d662aaada4548cf27bbfdec365627126eb1836703096f737a5ad94ad4"
CONTROL_COORDINATES = (0, 4, 8)
CONTROL_STEPS = (0.00125, 0.000625, 0.0003125, 0.00015625, 0.000078125)
CONTROL_THRESHOLDS = {"FD_Hq_max_abs_error_Eh_per_Bohr2": 1e-4,
    "h_halving_FD_Hq_max_abs_difference_Eh_per_Bohr2": 5e-5,
    "central_energy_curvature_abs_error_Eh_per_Bohr2": 1e-4}
TIGHT_SCF_CONV_TOL = 1e-13
TIGHT_SCF_CONV_TOL_GRAD = 1e-9
SCF_MAIN_ORBITAL_GRADIENT_TRIGGER = 1e-10
MAX_SCF_KERNEL_CALLS = 3
SCF_MAX_CYCLES_PER_CALL = 100
SCF_CERTIFICATION_POLICY = "FULL_NATIVE_VEFF_ENDPOINT_CERTIFICATION_NOT_NATIVE_KERNEL_FLAG"
EXPECTED_POINT_COUNT = 31
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
    require(control_validation is True and result["case"] == "water_software_control" and result["atoms"] == 3,
            "This worker only accepts the authenticated water software control")


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
    return {"MOs": {key: base.array_fingerprint(getattr(original, key))
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
    proof["fingerprints_unchanged_after_SCF_and_gradient"] = True


def historical_gradient(gate, np):
    source = gate["result"]
    receipt = checked(source["gradient_gate"]["result"])
    value = read(receipt)
    record = {"path":str(receipt.parent / "gradient_hartree_per_bohr.npy"),
              "sha256":value["gradient_npy_sha256"]}
    path = checked(record)
    if source["case"] == "Spinosin":
        require(record["sha256"] == SPINOSIN_GRADIENT_SHA, "Historical Spinosin reference gradient differs")
    gradient = np.load(path, allow_pickle=False)
    require(gradient.dtype == np.float64 and gradient.shape == (source["atoms"], 3)
            and np.isfinite(gradient).all(), "Actual complete historical gradient required")
    return gradient, {**ref(path), "gradient_gate_result":ref(receipt),
        "reference_scope":"HISTORICAL_OLD_BOOT_RESTORED_GRADIENT_NOT_NEW_BOOT_FRESH_ZERO_OR_MINIMUM",
        "historical_boot_id":source["boot_id"], "zero_gradient_assumed":False}


def compute_point(args, gate, original, binding, w, base, np, gen_grid, Gradients, addons,
                  out, q, hq, step, sign, *, zero=False, coordinate_index=None, mode_index=None):
    """Nested numerical point; consumes its one already-qualified parent phase."""
    out.mkdir()
    source = gate["result"]
    state = {"schema":POINT_SCHEMA,"status":"PARTIAL","method":METHOD,"output_directory":str(out),
        "source_worker":ref(__file__),"parent_output_directory":str(args.output.resolve()),
        "nested_point_under_parent_guardian_phase":True,"source_snapshot_manifest_sha256":source["snapshot_manifest_sha256"],
        "displacement":{"mode_index_1based":mode_index,"coordinate_index_0based":coordinate_index,
            "step_bohr":step,"sign":sign,"direction_normalization":"EUCLIDEAN_CARTESIAN_UNIT_NORM"},
        "SCF_performed":True,"fresh_zero_reference_point":zero,"scf_converged":False,"gradient_completed":False,
        "diagnostic_status":DIAGNOSTIC_STATUS,"IR_intensities_status":"NOT_COMPUTED"}
    log = (out/"pyscf.log").open("x",buffering=1)
    cpu, wall = time.process_time(),time.perf_counter()
    def emit(event, detail=None):
        row = {"utc":utc(),"event":event,"CPU_seconds":time.process_time()-cpu,
            "wall_seconds":time.perf_counter()-wall,**(detail or {})}
        with (out/"events.jsonl").open("a") as handle:
            handle.write(json.dumps(row,allow_nan=False)+"\n")
        save(out/"progress.json",{**state,**row})
    try:
        n, nao = original.mol.natm, original.mol.nao_nr()
        source_xyz = original.mol.atom_coords(unit="Bohr")
        new_xyz = source_xyz + sign*step*q
        mol = original.mol.copy()
        mol.set_geom_(new_xyz,unit="Bohr",inplace=True)
        mol.stdout = log
        require(mol.nao_nr() == nao and mol.charge == 0 and mol.spin == 0
                and np.array_equal(mol.atom_coords(unit="Bohr"),new_xyz), "Full AO space or exact displaced geometry changed")
        settings = dict(binding["method_settings"])
        require(settings["xc"] == "PBE" and settings["basis"].lower() == "def2-svp"
                and settings["grid_level"] == 1 and settings["dispersion"] == "NONE_EXPLICIT_PBE_PILOT_NOT_D3"
                and settings["gradient_grid_response"] is False, "Frozen PBE/def2-SVP/grid1/no-D3 model required")
        settings.update(threads=2,native_direct_driver_threads=2)
        settings["scf_conv_tol_hartree"] = TIGHT_SCF_CONV_TOL
        mf = w.make_scf(mol,settings,out/"scf.chk")
        mf.conv_tol_grad = SCF_MAIN_ORBITAL_GRADIENT_TRIGGER
        mf.max_cycle = SCF_MAX_CYCLES_PER_CALL
        fixed = attach_fixed_quadrature(mf,original,w,base,np,gen_grid)
        settings.update(scientific_quadrature_method=METHOD,small_rho_cutoff=0,
            actual_conv_tol_hartree=mf.conv_tol,actual_conv_tol_grad=mf.conv_tol_grad,
            actual_scf_max_cycles_per_call=mf.max_cycle,maximum_SCF_kernel_calls=MAX_SCF_KERNEL_CALLS,
            final_acceptance_energy_tolerance_hartree=TIGHT_SCF_CONV_TOL,
            final_acceptance_orbital_gradient_tolerance=TIGHT_SCF_CONV_TOL_GRAD,
            native_grid_cutoff=mf.grids.cutoff,native_numint_cutoff=mf._numint.cutoff,
            gradient_grid_response=False)
        state.update(settings=settings,atoms=n,basis_functions=nao,
            occupied_orbitals=int(np.count_nonzero(original.mo_occ > 0)),fixed_quadrature_proof=fixed,
            reference_energy_hartree=float(original.e_tot))
        for name,value in (("direction_cartesian_unit.npy",q),("analytic_H_times_direction.npy",hq),
            ("source_coordinates_bohr.npy",source_xyz),("displaced_coordinates_bohr.npy",new_xyz)):
            base.atomic_npy(out/name,value)
        save(out/"input.json",state,exclusive=True)
        state["input_json_sha256"] = sha(out/"input.json")
        dm0 = addons.project_dm_nr2nr(original.mol,original.make_rdm1(),mol)
        state["initial_guess"] = "PROJECTED_EXACT_SOURCE_DENSITY_TO_ACTUAL_AO_BASIS_FRESH_SCF_INCLUDING_ZERO"
        emit("SCF_started",{"actual_new_geometry":not zero,"fresh_zero_SCF":zero,"quadrature_method":METHOD})
        bounded_polished_scf(mf,dm0,np=np,base=base,out=out,emit=emit,state=state,
            final_energy_tolerance=TIGHT_SCF_CONV_TOL,
            final_orbital_gradient_tolerance=TIGHT_SCF_CONV_TOL_GRAD,
            main_orbital_gradient_trigger=SCF_MAIN_ORBITAL_GRADIENT_TRIGGER)
        state.update(scf_converged=bool(mf.converged),energy_hartree=float(mf.e_tot))
        require(mf.mo_coeff.shape == (nao,nao) and np.count_nonzero(mf.mo_occ>0) == state["occupied_orbitals"],
                "Full AO/MO or occupied space differs")
        verify_fixed_quadrature(mf,fixed,w,base)
        cls = base.direct_gradient_type(Gradients,emit,memory_mb=lambda:__import__("pyscf").lib.current_memory()[0])
        gradient_obj = cls(mf)
        gradient_obj.grid_response = False
        gradient = np.asarray(gradient_obj.kernel())
        require(gradient_obj.grid_response is False and gradient_obj.base.grids is mf.grids,
                "Gradient response or actual fixed grids changed")
        require(gradient.shape == (n,3) and np.isfinite(gradient).all(), "Complete finite gradient required")
        base.atomic_npy(out/"gradient_hartree_per_bohr.npy",gradient)
        state.update(gradient_completed=True,gradient_hartree_per_bohr=gradient.tolist(),
            gradient_projected_Eh_per_Bohr=float(np.sum(q*gradient)),status=POINT_SUCCESS)
        verify_fixed_quadrature(mf,fixed,w,base)
        state["computed_array_files"] = {name:ref(out/name) for name in
            ("direction_cartesian_unit.npy","analytic_H_times_direction.npy","source_coordinates_bohr.npy",
             "displaced_coordinates_bohr.npy","gradient_hartree_per_bohr.npy")}
        state["checkpoint"] = {**ref(Path(mf.chkfile)),"storage_policy":"LOCAL_ONLY_NOT_FOR_GIT_ADD_BY_DEFAULT"}
        emit("point_completed",{"gradient_completed":True})
    except BaseException as exc:
        state["status"] = "PARTIAL_RESOURCE_STOP" if isinstance(exc,base.BudgetStop) else "FAILED_DIAGNOSTIC"
        state["error"] = {"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        emit("point_failed",{"error":state["error"]})
        if isinstance(exc,base.BudgetStop):
            raise
    finally:
        state.update(ended_at_utc=utc(),worker_CPU_seconds=time.process_time()-cpu,worker_wall_seconds=time.perf_counter()-wall)
        log.close()
        state["saved_raw_array_files"] = {name:ref(out/name) for name in
            ("direction_cartesian_unit.npy","analytic_H_times_direction.npy","source_coordinates_bohr.npy",
             "displaced_coordinates_bohr.npy","gradient_hartree_per_bohr.npy") if (out/name).is_file()}
        save(out/"native_result.json",state,exclusive=True)
    return state


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
                         max_cycles=SCF_MAX_CYCLES_PER_CALL):
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
    hcore,overlap = mf.get_hcore(mf.mol),mf.get_ovlp(mf.mol)
    state["SCF_polishing_attempts"] = []
    state["strict_SCF_convergence_checks"] = []
    starting_density = np.array(dm0,copy=True)
    for call in range(1,max_calls+1):
        mf.conv_tol,mf.conv_tol_grad,mf.max_cycle = final_energy_tolerance,main_orbital_gradient_trigger,max_cycles
        mf.chkfile = str(out/("scf-call-%d.chk" % call))
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
            path = out/("scf-call-%d-%s.npy" % (call,name))
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
                path = out/("qualified-%s.npy" % name)
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


def finest_pair_gate(comparisons, halvings, *, measurements_complete):
    """Only the preselected last two steps govern this numerical gate."""
    if not measurements_complete:
        return False
    selected = {(c,h) for c in CONTROL_COORDINATES for h in CONTROL_STEPS[-2:]}
    fine = [row for row in comparisons if (row["coordinate_index_0based"],row["step_bohr"]) in selected]
    halving = [row for row in halvings if row["coarser_step_bohr"] == CONTROL_STEPS[-2]
               and row["finer_step_bohr"] == CONTROL_STEPS[-1]]
    if len(fine) != 6 or {(row["coordinate_index_0based"],row["step_bohr"]) for row in fine} != selected:
        return False
    if len(halving) != 3 or {row["coordinate_index_0based"] for row in halving} != set(CONTROL_COORDINATES):
        return False
    return bool(all(row.get("pair_completed") is True and row.get("fresh_zero_completed") is True
        and math.isfinite(row.get("FD_Hq_max_abs_error_Eh_per_Bohr2",float("nan")))
        and row["FD_Hq_max_abs_error_Eh_per_Bohr2"] <= CONTROL_THRESHOLDS["FD_Hq_max_abs_error_Eh_per_Bohr2"]
        and math.isfinite(row.get("central_energy_curvature_abs_error_Eh_per_Bohr2",float("nan")))
        and row["central_energy_curvature_abs_error_Eh_per_Bohr2"] <= CONTROL_THRESHOLDS["central_energy_curvature_abs_error_Eh_per_Bohr2"]
        for row in fine)
        and all(row.get("pair_completed") is True
            and math.isfinite(row.get("max_abs_difference_Eh_per_Bohr2",float("nan")))
            and row["max_abs_difference_Eh_per_Bohr2"] <= CONTROL_THRESHOLDS["h_halving_FD_Hq_max_abs_difference_Eh_per_Bohr2"]
            for row in halving))


def point_name(coordinate, step, sign):
    return "c%d-h%s-%s" % (coordinate,format(step,".8f"),"plus" if sign == 1 else "minus")


def central_energy_curvature(plus_energy, minus_energy, fresh_zero_energy, step):
    return math.fsum((plus_energy,minus_energy,-2*fresh_zero_energy))/(step*step)


def same_spinosin_step_energy_gate(comparisons, *, measurements_complete):
    if not measurements_complete:
        return False
    selected = [row for row in comparisons if row["step_bohr"] == 0.0003125]
    return bool(len(selected) == 3 and {row["coordinate_index_0based"] for row in selected} == set(CONTROL_COORDINATES)
        and all(row.get("pair_completed") is True and row.get("fresh_zero_completed") is True
            and math.isfinite(row.get("central_energy_curvature_abs_error_Eh_per_Bohr2",float("nan")))
            and row["central_energy_curvature_abs_error_Eh_per_Bohr2"]
                <= CONTROL_THRESHOLDS["central_energy_curvature_abs_error_Eh_per_Bohr2"] for row in selected))


def execute(args):
    require(args.control_suite and args.control_validation, "Only the water convergence control suite is authorized")
    guard = load(GUARD,GUARD_SHA,"fixed_grid_water_convergence_v8_guard")
    supervision = guard.verify_resource_supervision(expected_threads=args.threads)
    require(args.threads == 2, "Exactly two threads are authorized")
    proof = supervision["proof"]
    require(proof["cpu_hard_limit_seconds"] <= 300, "Water convergence phase exceeds its 300 CPU-second contract")
    require(datetime.now(timezone.utc) < STRICT_DEADLINE
            and datetime.fromisoformat(proof["started_at_utc"]) + timedelta(seconds=proof["wall_limit_seconds"]) <= STRICT_DEADLINE,
            "The entire guardian wall envelope must fit the earlier Nov-9 08:25:17.730193Z deadline")
    sole_phase(guard,supervision)
    analyzer = load(ANALYZER,ANALYZER_SHA,"fixed_grid_water_convergence_original_analyzer")
    with closed_main(args,analyzer) as main_ref:
        gate = source_gate(args,analyzer,guard,supervision)
        return execute_closed(args,analyzer,gate,supervision,main_ref)


def execute_closed(args, analyzer, gate, supervision, main_ref):
    import numpy as np
    from pyscf import lib
    from pyscf.dft import gen_grid
    from pyscf.scf import _vhf, addons
    from pyscf.grad.rks import Gradients
    lib.num_threads(2)
    w = load(analyzer.STATE,analyzer.STATE_SHA,"fixed_grid_water_convergence_exact_state_restorer")
    base = w.frozen_base()
    out = fresh_output(args.output)
    source = gate["result"]
    state = {"schema":CONTROL_SCHEMA,"status":"PARTIAL","stage":"water-fixed-grid-step-convergence",
        "method":METHOD,"diagnostic_status":DIAGNOSTIC_STATUS,"started_at_utc":utc(),
        "source_worker":ref(__file__),"resource_supervision":supervision,"output_directory":str(out),
        "source_result":ref(args.source_result),"snapshot_manifest_sha256":source["snapshot_manifest_sha256"],
        "case":source["case"],"v2_source_qualification":gate["v2_source_qualification"],
        "closed_main_pipeline_state":main_ref,"source_boot_id":source["boot_id"],
        "current_execution_boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "historical_source_gate_validated_against_its_own_boot":True,"stable_MF_state_unchanged":False,
        "stationary_minimum_verified":False,"chemical_assignment_verified":False,
        "IR_intensities_status":"NOT_COMPUTED","new_full_Hessian":"NOT_COMPUTED",
        "old_scientific_results_modified":False,"new_classifier_training":False,"automatic_retry_allowed":False,
        "old_single_gradient_Spinosin_release":False,"Spinosin_execution_in_this_worker":"FORBIDDEN",
        "finest_pair_passed":False,"same_spinosin_step_energy_passed":False,
        "same_spinosin_energy_step_bohr":0.0003125,"all_31_prescribed_points_completed":False,
        "thresholds":dict(CONTROL_THRESHOLDS),"predetermined_coordinates_0based":list(CONTROL_COORDINATES),
        "predetermined_steps_bohr":list(CONTROL_STEPS),"predetermined_signs":[-1,1],
        "finest_pair_selection":list(CONTROL_STEPS[-2:]),"expected_point_count":EXPECTED_POINT_COUNT,
        "software_control_SCF_conv_tol_hartree":TIGHT_SCF_CONV_TOL,
        "software_control_SCF_conv_tol_grad":TIGHT_SCF_CONV_TOL_GRAD,
        "all_points_including_zero_perform_fresh_SCF":True,
        "full_native_energy_AND_orbital_gradient_required_at_final_extra_endpoint":True,
        "SCF_endpoint_certification_policy":SCF_CERTIFICATION_POLICY,
        "native_kernel_flags_are_preserved_separate_from_custom_full_native_certification":True,
        "SCF_main_orbital_gradient_trigger":SCF_MAIN_ORBITAL_GRADIENT_TRIGGER,
        "maximum_SCF_kernel_calls_per_point":MAX_SCF_KERNEL_CALLS,
        "maximum_SCF_cycles_per_call":SCF_MAX_CYCLES_PER_CALL,
        "final_energy_evaluation":"FRESH_NONINCREMENTAL_NATIVE_VEFF_AT_QUALIFIED_CURRENT_DENSITY",
        "previous_failed_protocol_preserved":True,
        "archived_gradient_comparison_scope":"MEASURED_ONLY_NO_ARCHIVED_ZERO_GRADIENT_ACCEPTANCE_GATE",
        "acceptance_uses_absolute_full_vector_and_energy_errors":True,"points":[],"comparisons":[],"h_halving_comparisons":[]}
    save(out/"input.json",{**state,"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},exclusive=True)
    state["input_json_sha256"] = sha(out/"input.json")
    cpu,wall = time.process_time(),time.perf_counter()
    def emit(event,detail=None):
        row = {"utc":utc(),"event":event,"CPU_seconds":time.process_time()-cpu,
            "wall_seconds":time.perf_counter()-wall,**(detail or {})}
        with (out/"events.jsonl").open("a") as handle:
            handle.write(json.dumps(row,allow_nan=False)+"\n")
        save(out/"progress.json",{**state,**row})
    def stop(signum,frame):
        raise base.BudgetStop("Bounded water convergence received resource signal "+str(signum))
    handlers = {sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGXCPU)}
    native = base.NativeDirectDriverThreadGuard(_vhf,lib.with_omp_threads,lib.num_threads,
        native_threads=2,emit=lambda detail:emit("native_direct_driver_signature",detail))
    original = None
    restoration_log = (out/"restoration.log").open("x",buffering=1)
    try:
        native.install()
        original,binding = w.restore_snapshot(Path(source["snapshot_directory"]),
            expected_manifest_sha256=source["snapshot_manifest_sha256"],expected_source_sha256=analyzer.STATE_SHA,stdout=restoration_log)
        require(binding == source["source_state_binding"], "Restored original water source binding differs")
        source_fingerprint = w.molecule_physical_fingerprint(original.mol)
        source_payload = original_mf_fingerprints(original,base)
        n,nao = original.mol.natm,original.mol.nao_nr()
        require(n == 3 == source["atoms"] and nao == source["basis_functions"] and original.mo_coeff.shape == (nao,nao)
            and original.mol.charge == 0 and original.mol.spin == 0
            and original.mol.nelectron == int(original.mo_occ.sum()), "Complete neutral closed-shell water AO/MO required")
        state.update(atoms=n,basis_functions=nao,symbols=[original.mol.atom_symbol(i) for i in range(n)],
            occupied_orbitals=int(np.count_nonzero(original.mo_occ>0)),reference_state_binding=binding,
            source_geometry_fingerprint=source_fingerprint,original_MF_payload_fingerprints=source_payload)
        archived_g0,g0_record = historical_gradient(gate,np)
        state["historical_gradient_reference"] = g0_record
        base.atomic_npy(out/"historical_gradient_hartree_per_bohr.npy",archived_g0)
        state["historical_gradient_local_copy"] = ref(out/"historical_gradient_hartree_per_bohr.npy")
        def unchanged():
            require(w.molecule_physical_fingerprint(original.mol) == source_fingerprint,
                "Original source molecule changed during calculation")
            require(original_mf_fingerprints(original,base) == source_payload,
                "Original source MF/MO/quadrature changed during calculation")
            checked(state["source_result"])
            checked(g0_record)
        def point(name,q,hq,step,sign,**kwargs):
            value = compute_point(args,gate,original,binding,w,base,np,gen_grid,Gradients,addons,
                out/name,q,hq,step,sign,**kwargs)
            unchanged()
            state["points"].append(ref(out/name/"native_result.json"))
            return value
        q0 = np.zeros((n,3),dtype=float)
        zero = point("fresh-zero-scf-gradient",q0,q0,0.,0,zero=True)
        zero_ok = zero["status"] == POINT_SUCCESS
        state["fresh_zero_completed"] = bool(zero_ok)
        if zero_ok:
            fresh_zero = np.asarray(zero["gradient_hartree_per_bohr"],dtype=float)
            state["fresh_zero_energy_hartree"] = float(zero["energy_hartree"])
            state["fresh_zero_gradient_hartree_per_bohr"] = fresh_zero.tolist()
            state["fresh_zero_minus_archived_gradient_hartree_per_bohr"] = (fresh_zero-archived_g0).tolist()
            state["fresh_zero_minus_archived_gradient_max_abs_Eh_per_Bohr"] = float(np.max(np.abs(fresh_zero-archived_g0)))
        all_points = zero_ok
        for coordinate in CONTROL_COORDINATES:
            probe = argparse.Namespace(**vars(args))
            probe.mode_index,probe.coordinate_index = None,coordinate
            q,hq,curvature,references = direction_and_hessian(probe,gate,np)
            comparisons = []
            for step in CONTROL_STEPS:
                pair = {}
                for sign in (-1,1):
                    name = point_name(coordinate,step,sign)
                    value = point(name,q,hq,step,sign,coordinate_index=coordinate)
                    pair[sign] = value
                    all_points = all_points and value["status"] == POINT_SUCCESS
                    emit("convergence_subpoint_completed",{"coordinate_index_0based":coordinate,"step_bohr":step,
                        "sign":sign,"point_status":value["status"]})
                comparison = {"coordinate_index_0based":coordinate,"step_bohr":step,
                    "analytic_Hq_hartree_per_bohr2":hq.tolist(),"analytic_directional_curvature_Eh_per_Bohr2":curvature,
                    "source_arrays":references,"pair_completed":all(value["status"] == POINT_SUCCESS for value in pair.values()),
                    "fresh_zero_completed":zero_ok,"included_in_finest_pair_gate":step in CONTROL_STEPS[-2:]}
                if comparison["pair_completed"]:
                    plus = np.asarray(pair[1]["gradient_hartree_per_bohr"],dtype=float)
                    minus = np.asarray(pair[-1]["gradient_hartree_per_bohr"],dtype=float)
                    fd = (plus-minus)/(2*step)
                    error = fd-hq
                    maximum = float(np.max(np.abs(error)))
                    comparison.update(plus_gradient_hartree_per_bohr=plus.tolist(),minus_gradient_hartree_per_bohr=minus.tolist(),
                        plus_energy_hartree=float(pair[1]["energy_hartree"]),minus_energy_hartree=float(pair[-1]["energy_hartree"]),
                        FD_Hq_hartree_per_bohr2=fd.tolist(),FD_minus_analytic_Hq_hartree_per_bohr2=error.tolist(),
                        FD_Hq_max_abs_error_Eh_per_Bohr2=maximum,
                        FD_Hq_absolute_error_check_passed=maximum <= CONTROL_THRESHOLDS["FD_Hq_max_abs_error_Eh_per_Bohr2"],
                        finite_gradient_directional_curvature_Eh_per_Bohr2=float(np.sum(q*fd)))
                    if zero_ok:
                        energy_curvature = central_energy_curvature(comparison["plus_energy_hartree"],
                            comparison["minus_energy_hartree"],state["fresh_zero_energy_hartree"],step)
                        energy_error = energy_curvature-curvature
                        comparison.update(fresh_zero_energy_hartree=state["fresh_zero_energy_hartree"],
                            central_energy_curvature_formula="(E_plus+E_minus-2*FRESH_SCF_E0)/(h*h)",
                            central_energy_curvature_Eh_per_Bohr2=energy_curvature,
                            central_energy_curvature_minus_analytic_Eh_per_Bohr2=energy_error,
                            central_energy_curvature_abs_error_Eh_per_Bohr2=abs(energy_error),
                            central_energy_curvature_absolute_error_check_passed=abs(energy_error)
                                <= CONTROL_THRESHOLDS["central_energy_curvature_abs_error_Eh_per_Bohr2"])
                comparisons.append(comparison)
                state["comparisons"].append(comparison)
                emit("full_vector_and_energy_comparison",comparison)
            for coarser,finer in zip(comparisons,comparisons[1:]):
                halving = {"coordinate_index_0based":coordinate,"coarser_step_bohr":coarser["step_bohr"],
                    "finer_step_bohr":finer["step_bohr"],"pair_completed":coarser["pair_completed"] and finer["pair_completed"],
                    "included_in_finest_pair_gate":coarser["step_bohr"] == CONTROL_STEPS[-2],"check_passed":False}
                if halving["pair_completed"]:
                    delta = np.asarray(coarser["FD_Hq_hartree_per_bohr2"])-np.asarray(finer["FD_Hq_hartree_per_bohr2"])
                    maximum = float(np.max(np.abs(delta)))
                    halving.update(FD_Hq_h_minus_h_over_2_hartree_per_bohr2=delta.tolist(),
                        max_abs_difference_Eh_per_Bohr2=maximum,
                        check_passed=maximum <= CONTROL_THRESHOLDS["h_halving_FD_Hq_max_abs_difference_Eh_per_Bohr2"])
                    if zero_ok:
                        halving["central_energy_curvature_difference_Eh_per_Bohr2"] = (
                            coarser["central_energy_curvature_Eh_per_Bohr2"]-finer["central_energy_curvature_Eh_per_Bohr2"])
                state["h_halving_comparisons"].append(halving)
                emit("adjacent_h_halving_comparison",halving)
        state["all_31_prescribed_points_completed"] = bool(all_points and len(state["points"]) == EXPECTED_POINT_COUNT)
        state["finest_pair_passed"] = finest_pair_gate(state["comparisons"],state["h_halving_comparisons"],
            measurements_complete=state["all_31_prescribed_points_completed"])
        state["same_spinosin_step_energy_passed"] = same_spinosin_step_energy_gate(state["comparisons"],
            measurements_complete=state["all_31_prescribed_points_completed"])
        state["status"] = CONTROL_SUCCESS if state["all_31_prescribed_points_completed"] else "WATER_FIXED_GRID_STEP_CONVERGENCE_INCOMPLETE"
        unchanged()
        state["stable_MF_state_unchanged"] = True
        emit("water_convergence_completed",{"status":state["status"],"finest_pair_passed":state["finest_pair_passed"],
            "same_spinosin_step_energy_passed":state["same_spinosin_step_energy_passed"]})
    except BaseException as exc:
        state["status"] = "PARTIAL_RESOURCE_STOP" if isinstance(exc,base.BudgetStop) else "FAILED_DIAGNOSTIC"
        state["error"] = {"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        emit("water_convergence_failed",{"error":state["error"]})
    finally:
        if original is not None and "source_payload" in locals():
            state["original_MF_payload_after"] = original_mf_fingerprints(original,base)
            if state["original_MF_payload_after"] != source_payload or w.molecule_physical_fingerprint(original.mol) != source_fingerprint:
                state["stable_MF_state_unchanged"] = False
                state["status"] = "FAILED_ORIGINAL_SOURCE_MUTATED"
        native.restore()
        state["native_direct_driver_control"] = native.summary()
        if state["native_direct_driver_control"].get("restoration_failures") != 0 or not state["native_direct_driver_control"].get("original_driver_restored"):
            state["status"] = "FAILED_NATIVE_DRIVER_RESTORATION"
        if state["status"] != CONTROL_SUCCESS:
            state["finest_pair_passed"] = False
            state["same_spinosin_step_energy_passed"] = False
        state.update(ended_at_utc=utc(),worker_CPU_seconds=time.process_time()-cpu,worker_wall_seconds=time.perf_counter()-wall)
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
        restoration_log.close()
        save(out/"result.json",state,exclusive=True)
    return state


def validate_actual_arguments(args):
    require(args.control_suite and args.control_validation and args.threads == 2,
        "Only the fixed water convergence control suite with two threads is authorized")
    require(args.source_result is not None and args.source_result_sha256 and args.main_pipeline_state is not None
        and args.main_pipeline_state_sha256 and args.source_qualification is not None and args.source_qualification_sha256,
        "Authenticated water source/main state and outside-Q source qualification are required")
    require(args.allowed_phase_schedule is None and args.allowed_phase_schedule_sha256 is None,
        "Execution obtains the exact finite schedule only from its sealed qualification")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--control-suite",action="store_true")
    mode.add_argument("--qualify-source-only",action="store_true")
    p.add_argument("--source-result",type=Path);p.add_argument("--source-result-sha256")
    p.add_argument("--source-qualification",type=Path);p.add_argument("--source-qualification-sha256")
    p.add_argument("--allowed-phase-schedule",type=Path);p.add_argument("--allowed-phase-schedule-sha256")
    p.add_argument("--main-pipeline-state",type=Path);p.add_argument("--main-pipeline-state-sha256")
    p.add_argument("--control-validation",action="store_true");p.add_argument("--threads",type=int,default=2)
    args = p.parse_args()
    if args.control_suite:
        validate_actual_arguments(args)
        value = execute(args)
    elif args.qualify_source_only:
        require(args.control_validation and args.threads == 2 and args.source_result is not None and args.source_result_sha256
            and args.main_pipeline_state is not None and args.main_pipeline_state_sha256
            and args.allowed_phase_schedule is not None and args.allowed_phase_schedule_sha256,
            "Water-only outside-Q qualification requires exact source/main state and explicit schedule")
        require(args.source_qualification is None and args.source_qualification_sha256 is None,
            "Outside-Q qualification cannot consume its own future seal")
        # The original qualifier rejects target-numerical arguments; no such CLI exists here.
        value = qualify_source(args)
    else:
        require(all(value is None for key,value in vars(args).items() if key not in
            ("output","threads","control_suite","qualify_source_only","control_validation"))
            and not args.control_validation,"Metadata preparation does not read target inputs")
        out = fresh_output(args.output)
        value = {"schema":CONTROL_SCHEMA,"status":"NOT_RUN","prepared_at_utc":utc(),"source_worker":ref(__file__),
            "method":METHOD,"diagnostic_status":DIAGNOSTIC_STATUS,"new_quantum_evaluations":0,
            "numerical_packages_imported":False,"target_arrays_read":False,"IR_intensities_status":"NOT_COMPUTED",
            "control_thresholds":CONTROL_THRESHOLDS,"predetermined_steps_bohr":list(CONTROL_STEPS),
            "expected_point_count":EXPECTED_POINT_COUNT,"finest_pair_selection":list(CONTROL_STEPS[-2:]),
            "automatic_retry_allowed":False,"old_single_gradient_Spinosin_release":False}
        save(out/"preflight.json",value,exclusive=True)
    print(json.dumps({key:value.get(key) for key in ("schema","status","case","finest_pair_passed")},ensure_ascii=False))
    return 0 if value["status"] in ("NOT_RUN",CONTROL_SUCCESS,QUALIFICATION_SUCCESS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
