"""Finite full-native-VEFF controls, checkpoint restart and energy diagnostics.

No quantum package is imported until the actual new72h guardian and sealed
original source have passed.  Old sources/checkpoints are read-only.  Native
Coulomb builds use one thread; the worker/BLAS envelope uses two threads.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import shutil
import struct
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/"scripts/research4_fixed_grid_energy_curvature_20261107_v2.py"
OLD_SHA = "5f746f94a792b6d653c724774236410dbfea768a382462c6f6b375b79757b592"
HELPERS = ROOT/"scripts/research4_full_native_veff_restart_helpers_20261107.py"
GUARD = ROOT/"scripts/run_research4_research72h_budget_20261107.py"
GUARD_SHA = "643a431b7229eca94ac50645d467e6c3472f976b8ac48cc2f36be68241319e68"
LEDGER = ROOT/"reports/research72h-20261107-v1/cpu_budget.json"
WORK = ROOT/"reports/research72h-20261107-v1"
SCHEMA = "research4-full-native-veff-research-worker-v1"
POINT_SCHEMA = "research4-full-native-veff-energy-point-v1"
QUALIFICATION_SCHEMA = "research4-full-native-veff-source-qualification-v1"
STEPS = (.005, .0025)
MODES = (1, 13, 25, 26)
ENERGY_TOLERANCE = 1e-12
GRADIENT_TOLERANCE = 1e-9
REPEAT_SPREAD_TOLERANCE = 1e-12
POINT_STORAGE_RESERVE_BYTES = 12*1024**2
GLOBAL_FREE_FLOOR_BYTES = 64*1024**2
STORAGE_POLICY = "COMPACT_RESTART_AND_AUDIT_HASHES"
LOCAL_STATE_MAX_BYTES = 4*POINT_STORAGE_RESERVE_BYTES
CHECKPOINT = ROOT/"reports/fixed-quadrature-convergence-20261107-v2/queue-host-a-006/local-checkpoints/spino-mode1-energy-attempt001/fresh-zero/scf-call-2.chk"
CHECKPOINT_SHA = "8c2993d5198c95a79d584f5f41d3be6ce4ccd45281dcb58f58d03e51c8cc4fb0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path=Path(path).resolve()
    return {"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size}


def checked(record):
    path=Path(record["path"])
    require(not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(ROOT),"Repository regular artifact required")
    require(sha(path)==record["sha256"],"Artifact byte hash differs: "+str(path))
    return path.resolve()


def read(path):
    require(Path(path).stat().st_size<=32*1024**2,"Bounded JSON required")
    return json.loads(Path(path).read_text())


def save(path, value):
    path=Path(path);temp=path.with_name(path.name+".tmp-"+str(os.getpid()))
    with temp.open("x") as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write("\n");handle.flush();os.fsync(handle.fileno())
    os.replace(temp,path)


def load(path, expected, name):
    require(sha(path)==expected,"Frozen source changed: "+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def old():
    return load(OLD,OLD_SHA,"fullnative_frozen_energy_helpers")


def fresh(path):
    path=Path(path).absolute()
    require(not any(p.is_symlink() for p in (path,*path.parents)),"Output symlinks forbidden")
    path=path.resolve()
    require(path.is_relative_to(WORK) and not path.exists(),"Fresh new72h output namespace required")
    require(not any(p.is_symlink() for p in (path,*path.parents)),"Output symlinks forbidden")
    path.mkdir(parents=True)
    return path


def free_space_preflight(path, remaining_points):
    require(type(remaining_points) is int and 0<=remaining_points<=4,"Finite remaining point count required")
    required=GLOBAL_FREE_FLOOR_BYTES+remaining_points*POINT_STORAGE_RESERVE_BYTES
    free=shutil.disk_usage(path).free
    require(free>=required,"Insufficient persistent free space for remaining compact points and64MiB floor")
    return {"free_bytes":free,"required_bytes":required,"remaining_points":remaining_points,
            "per_point_reserve_bytes":POINT_STORAGE_RESERVE_BYTES,"global_free_floor_bytes":GLOBAL_FREE_FLOOR_BYTES}


def source_identity(gate):
    result=gate["result"]
    if result["case"]=="Spinosin":
        require(result["atoms"]==75 and result["basis_functions"]==762 and result["snapshot_manifest_sha256"]==old().SOURCE_SNAPSHOT,
                "Original complete Spinosin source required")
    else:
        require(result["case"] in ("water_software_control","ethanol_software_control"),"Only original water/ethanol controls accepted")


def qualify(args):
    """Strict old analyzer while new and old ledgers are closed; no NumPy."""
    frozen=old();guard=load(GUARD,GUARD_SHA,"fullnative_new_guard_qualifier")
    ledger=read(LEDGER);guard.validate_ledger(ledger,allow_running=False)
    before=sha(LEDGER)
    analyzer=load(frozen.ANALYZER,frozen.ANALYZER_SHA,"fullnative_original_analyzer")
    source=checked({"path":str(args.source_result.resolve()),"sha256":args.source_result_sha256})
    schedule_ref={"path":str(args.allowed_phase_schedule.resolve()),"sha256":args.allowed_phase_schedule_sha256}
    schedule=frozen.read_schedule(schedule_ref)
    require(Path(schedule["ledger_path"]).resolve()==LEDGER,"Schedule must use new72h ledger")
    with frozen.closed_main(args,analyzer) as main_ref:
        gate=analyzer.qualify(source,args.source_result_sha256,control=args.control_validation)
        source_identity(gate)
        require((gate["result"]["case"]!="Spinosin")==args.control_validation,"Control scope differs")
        immutable={str(Path(path).resolve()):dict(record) for path,record in gate["hashes"].items()}
        snapshot=Path(gate["result"]["snapshot_directory"])
        for record in gate["source"]["snapshot_manifest"]["arrays"].values():
            path=snapshot/record["file"]
            require(sha(path)==record["sha256"],"Snapshot array changed")
            immutable[str(path.resolve())]={"sha256":record["sha256"],"bytes":path.stat().st_size}
        for path,record in immutable.items():
            require(sha(path)==record["sha256"] and Path(path).stat().st_size==record["bytes"],"Original immutable source changed")
        require(sha(LEDGER)==before,"New ledger changed during outside-Q qualification")
        guard.validate_ledger(read(LEDGER),allow_running=False)
        out=fresh(args.output)
        value={"schema":QUALIFICATION_SCHEMA,"status":"SOURCE_QUALIFIED_OUTSIDE_QUANTUM_PHASE",
            "source_worker":ref(__file__),"helper_source":ref(HELPERS),"new_guard_source":ref(GUARD),
            "source_result":ref(source),"original_analyzer":ref(frozen.ANALYZER),
            "original_strict_gate_receipt":{"result_path":gate["result_path"],
                "metadata_gate_passed_before_numerical_import":gate["metadata_gate_passed_before_numerical_import"],
                "canonical_complete_gate_sha256":frozen.canonical_sha(gate),
                "canonical_global_closure_sha256":frozen.canonical_sha(gate["closure"]),
                "old_gate_tree_retained_in_this_receipt":False},"immutable_original_inputs":immutable,
            "new_ledger_header":frozen.ledger_header(ledger),"new_ledger_closed_rows":frozen.row_prefix(ledger),
            "allowed_phase_schedule":schedule_ref,"closed_main_pipeline_state":main_ref,
            "all_ledgers_closed_at_qualification":True,"new_quantum_evaluations":0,
            "numerical_packages_imported":False,"control_validation":args.control_validation}
        save(out/"qualification.json",value)
    return value


def source_gate(args, guard, supervision):
    frozen=old();seal_ref={"path":str(args.source_qualification.resolve()),"sha256":args.source_qualification_sha256}
    seal=read(checked(seal_ref))
    require(seal.get("schema")==QUALIFICATION_SCHEMA and seal.get("status")=="SOURCE_QUALIFIED_OUTSIDE_QUANTUM_PHASE"
        and seal.get("all_ledgers_closed_at_qualification") is True and seal.get("new_quantum_evaluations")==0,
        "Outside-Q original-source qualification required")
    require(seal["source_worker"]==ref(__file__) and seal["helper_source"]==ref(HELPERS)
        and seal["new_guard_source"]==ref(GUARD),"Source seal producer/helper/guardian changed")
    for path,record in seal["immutable_original_inputs"].items():
        checked({"path":path,"sha256":record["sha256"]})
    source=checked(seal["source_result"]);receipt=seal["original_strict_gate_receipt"]
    require(receipt["result_path"]==str(source) and receipt.get("metadata_gate_passed_before_numerical_import") is True,
        "Original strict metadata gate differs")
    result=read(source)
    analyzer=load(frozen.ANALYZER,frozen.ANALYZER_SHA,"fullnative_original_global_closure_gate")
    closure=guard.verify_closed_phase({"source_worker":result["source_worker"],"resource_supervision":result["resource_supervision"],
        "output_directory":str(source.parent),"snapshot_manifest_sha256":result["snapshot_manifest_sha256"],"stage":"global-assemble"},
        expected_worker_path=analyzer.WORKER,expected_worker_sha256=analyzer.WORKER_SHA,
        expected_result_schema=analyzer.SCHEMA,allowed_result_statuses=("FREQUENCIES_COMPUTED",),
        expected_output=source.parent,expected_snapshot_sha256=result["snapshot_manifest_sha256"])
    require(frozen.canonical_sha(closure)==receipt["canonical_global_closure_sha256"],"Original global guardian closure changed")
    gate={"result":result,"result_path":str(source),"metadata_gate_passed_before_numerical_import":True}
    source_identity(gate)
    schedule=frozen.read_schedule(seal["allowed_phase_schedule"])
    ledger=read(LEDGER);guard.validate_ledger(ledger,allow_running=True)
    require(frozen.ledger_header(ledger)==seal["new_ledger_header"],"New ledger immutable header changed")
    prefix=seal["new_ledger_closed_rows"];require(len(ledger["phases"])>=len(prefix),"New ledger truncated")
    for row,binding in zip(ledger["phases"],prefix):
        require(row["phase"]==binding["phase"] and frozen.canonical_sha(row)==binding["canonical_sha256"],"Sealed closed phase changed")
    policies={p["phase"]:p for p in schedule["phases"]};running=[]
    for row in ledger["phases"][len(prefix):]:
        require(row["phase"] in policies,"Appended phase outside finite sealed schedule")
        policy=policies[row["phase"]];worker=checked(policy["worker"]);cmd=row["command"]
        require(worker in [Path(t).resolve() for t in cmd if isinstance(t,str) and t.endswith(".py")]
            and "--output" in cmd and Path(cmd[cmd.index("--output")+1]).resolve()==Path(policy["output_directory"]).resolve()
            and Path(row["log"]).resolve()==Path(policy["log"]).resolve(),"Scheduled command/output/log differs")
        if row["status"]=="RUNNING":
            running.append(row)
            require(worker==Path(__file__).resolve() and row["phase"]==supervision["proof"]["phase"]
                and row["supervisor_proof_sha256"]==sha(supervision["proof_path"]),"Unrelated or unauthenticated running phase")
            require(Path(policy["output_directory"]).resolve()==args.output.resolve(),"Actual CLI output differs from live phase")
        else:
            guard.verify_closed_row(LEDGER,row["phase"],expected_worker_path=worker,
                expected_worker_sha256=policy["worker"]["sha256"],expected_output=policy["output_directory"],allow_failure=True)
    require(len(running)==1 and Path(supervision["proof"]["ledger_path"]).resolve()==LEDGER,"Exactly one new72h live phase required")
    return gate,{"seal":seal_ref,"original_input_count":len(seal["immutable_original_inputs"]),
                 "sealed_closed_rows":len(prefix),"scheduled_appended_rows":len(ledger["phases"])-len(prefix)}


def validate_stage_args(args):
    require(args.threads==2,"Root-selected worker/BLAS thread envelope is two")
    if args.stage=="controls":
        require(args.max_cycles==80 and args.source_zero_result is None,"Controls use one bounded80cycle SCF per variant")
    elif args.stage=="zero":
        require(args.max_cycles in (30,80),"Only initial30cycle pilot or one80cycle continuation declared")
        require((args.max_cycles==80)==(args.source_zero_result is not None),"80cycle continuation must bind pilot result")
    else:
        require(args.mode_index in MODES and tuple(args.steps)==STEPS and args.source_zero_result is not None
            and args.max_cycles==80,"Energy job requires fixed anchors/two halving steps/certifiedzero/80cycles")


def dependency(args, guard, *, certified):
    record={"path":str(args.source_zero_result.resolve()),"sha256":args.source_zero_result_sha256}
    path=checked(record);value=read(path)
    require(value.get("schema")==SCHEMA and value.get("stage")=="zero" and value.get("source_worker")==ref(__file__)
        and value.get("case")=="Spinosin" and value.get("snapshot_manifest_sha256")==old().SOURCE_SNAPSHOT
        and value.get("status") in ("ZERO_PILOT_COMPLETE","ZERO_CONTINUATION_COMPLETE","REUSED_CERTIFIED_ZERO"),"Authentic completed zero measurement required")
    guard.verify_closed_phase(value,expected_worker_path=Path(__file__),expected_worker_sha256=sha(__file__),
        expected_result_schema=SCHEMA,allowed_result_statuses=("ZERO_PILOT_COMPLETE","ZERO_CONTINUATION_COMPLETE","REUSED_CERTIFIED_ZERO"),
        expected_output=path.parent)
    if certified:
        require(value.get("zero_certified") is True,"Displacements require a physically certified zero")
    point=read(checked(value["points"]["fresh-zero"]["result"]))
    require(point.get("schema")==POINT_SCHEMA and point.get("source_worker")==ref(__file__),"Zero point source differs")
    if certified:
        verify_certified_point(point)
        require(value.get("energies_hartree",{}).get("fresh-zero")==point["energy_hartree"]
            and "fresh-zero" in value.get("completed_points",[])
            and value["points"]["fresh-zero"].get("energy_completed") is True
            and value["points"]["fresh-zero"].get("energy_hartree")==point["energy_hartree"],
            "Certifiedzero parent/child completed energy maps differ")
    return value,point,record


def verify_certified_point(point, *, require_full_every_cycle=True):
    """Replay scalar endpoint conditions before accepting any completed energy."""
    audit=point.get("endpoint_audit",{})
    energy=point.get("energy_hartree")
    finite=lambda value:isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    repeats=audit.get("full_same_density_energy_repeats_hartree",[])
    require(point.get("energy_completed") is True and point.get("SCF_endpoint_certified") is True
        and audit.get("qualified") is True and finite(energy)
        and audit.get("current_full_energy_hartree")==energy
        and len(repeats)==3 and all(finite(value) for value in repeats)
        and repeats[-1]==energy and max(repeats)-min(repeats)<=REPEAT_SPREAD_TOLERANCE
        and audit.get("full_same_density_energy_spread_hartree")==max(repeats)-min(repeats)
        and finite(audit.get("previous_full_energy_hartree"))
        and abs(energy-audit["previous_full_energy_hartree"])<ENERGY_TOLERANCE
        and finite(audit.get("full_current_orbital_gradient_norm"))
        and 0<=audit["full_current_orbital_gradient_norm"]<GRADIENT_TOLERANCE
        and audit.get("actual_native_final_extra_after_accepted_main_same_kernel") is True
        and (not require_full_every_cycle or audit.get("full_VEFF_applied_every_cycle") is True),
        "Certified point lacks actual full finite energy/gradient/lineage evidence")
    require(point.get("storage_policy")==STORAGE_POLICY and point.get("all_intermediate_raw_matrices_archived") is False,
            "Explicit compact-state retention policy required")
    checked(point["checkpoint"])
    for key in ("previous-density","full-orbital-gradient","refreshed-MO-energy"):
        checked(point["audit_state_files"][key])
    for key in ("previous-density","current-density","MO_coeff","MO_occ","MO_energy","full_Fock","full-orbital-gradient"):
        fingerprint=point["audit_state_fingerprints"][key]
        require(isinstance(fingerprint,dict) and isinstance(fingerprint.get("payload_sha256_C_order"),str)
                and len(fingerprint["payload_sha256_C_order"])==64,"Full computed audit matrix fingerprint missing")
    nao=point.get("basis_functions");electrons=point.get("electrons")
    require(type(nao) is int and 1<=nao<=762 and type(electrons) is int and 0<electrons<=2*nao and electrons%2==0,
            "Complete point AO/electron dimensions required")
    vector=read_finite_vector(point["audit_state_files"]["full-orbital-gradient"],(electrons//2)*(nao-electrons//2),
        expected_fingerprint=point["audit_state_fingerprints"]["full-orbital-gradient"])
    read_finite_vector(point["audit_state_files"]["refreshed-MO-energy"],nao,
        expected_fingerprint=point["audit_state_fingerprints"]["MO_energy"])
    norm=math.sqrt(math.fsum(value*value for value in vector))
    require(norm<GRADIENT_TOLERANCE and math.isclose(norm,audit["full_current_orbital_gradient_norm"],rel_tol=1e-9,abs_tol=1e-13),
            "Archived full orbital-gradient vector does not reproduce certified norm")
    return True


def read_finite_vector(record, count, *, expected_fingerprint=None):
    path=checked(record);require(type(count) is int and 0<count<=762**2,"Bounded complete vector length required")
    raw=path.read_bytes();require(raw[:6]==b"\x93NUMPY" and tuple(raw[6:8]) in ((1,0),(2,0),(3,0)),"Real NPY vector required")
    version=raw[6];offset=10 if version==1 else 12
    length=struct.unpack("<H" if version==1 else "<I",raw[8:offset])[0]
    require(0<length<=65536 and offset+length<=len(raw),"Bounded complete NPY header required")
    header=ast.literal_eval(raw[offset:offset+length].decode("utf-8" if version==3 else "latin1").strip())
    require(header.get("shape")== (count,) and header.get("descr") in ("<f8",">f8")
        and type(header.get("fortran_order")) is bool,"Full float64 vector required")
    payload=raw[offset+length:];require(len(payload)==count*8,"Complete vector byte count required")
    values=struct.unpack(("<" if header["descr"]=="<f8" else ">")+str(count)+"d",payload)
    require(all(math.isfinite(value) for value in values),"Finite archived vector required")
    if expected_fingerprint is not None:
        actual={"shape":[count],"dtype":"float64","payload_bytes":count*8,
            "payload_sha256_C_order":hashlib.sha256(struct.pack("="+str(count)+"d",*values)).hexdigest()}
        require(actual==expected_fingerprint,"Archived vector payload differs from computed audit fingerprint")
    return values


def verify_control_release(args, guard):
    """Authenticate both real control jobs and replay their raw small gradients."""
    frozen=old();release={}
    for name,atoms in (("water",3),("ethanol",9)):
        path=getattr(args,name+"_control_result",None);digest=getattr(args,name+"_control_result_sha256",None)
        require(path is not None and isinstance(digest,str),"Hash-bound water AND ethanol control results required")
        record={"path":str(path.resolve()),"sha256":digest};result=read(checked(record))
        require(result.get("schema")==SCHEMA and result.get("source_worker")==ref(__file__)
            and result.get("stage")=="controls" and result.get("case")==name+"_software_control"
            and result.get("status")=="FULL_NATIVE_CONTROL_PARITY_PASSED","Actual selected control result required")
        closure=guard.verify_closed_phase(result,expected_worker_path=Path(__file__),expected_worker_sha256=sha(__file__),
            expected_result_schema=SCHEMA,allowed_result_statuses=("FULL_NATIVE_CONTROL_PARITY_PASSED",),expected_output=path.parent)
        points={label:read(checked(result["points"][label]["result"])) for label in ("full-native","default-incremental")}
        for label,point in points.items():
            require(point.get("source_worker")==ref(__file__) and point.get("schema")==POINT_SCHEMA,"Control point source/schema differs")
            verify_certified_point(point,require_full_every_cycle=label=="full-native")
        parity=result["control_parity"]
        gradients={label:frozen.checked_control_array(parity[key],(atoms,3)) for label,key in
            (("full","full_gradient"),("default","default_gradient"),("native2","native2_full_gradient"))}
        difference=lambda left,right:max(abs(a-b) for ra,rb in zip(left,right) for a,b in zip(ra,rb))
        energy_gap=abs(points["full-native"]["energy_hartree"]-points["default-incremental"]["energy_hartree"])
        thread_gap=abs(points["full-native"]["energy_hartree"]-parity["native2_full_energy_hartree"])
        gradient_gap=difference(gradients["full"],gradients["default"])
        thread_gradient_gap=difference(gradients["full"],gradients["native2"])
        require(parity.get("passed") is True and energy_gap<=1e-10 and gradient_gap<=1e-8
            and thread_gap<=1e-10 and thread_gradient_gap<=1e-8
            and parity.get("energy_abs_difference_hartree")==energy_gap
            and parity.get("gradient_max_abs_difference_hartree_per_bohr")==gradient_gap
            and parity.get("same_density_native1_vs_native2_energy_abs_difference_hartree")==thread_gap
            and parity.get("native1_vs_native2_gradient_max_abs_difference_hartree_per_bohr")==thread_gradient_gap,
            "Raw water/ethanol native/default/thread parity does not pass fixed limits")
        release[name]={"result":record,"closure":closure,"raw_gradient_and_energy_parity_replayed":True}
    return release


def replay_zero_arrays(point, original, *, np):
    verify_certified_point(point)
    import h5py
    with h5py.File(checked(point["checkpoint"]),"r") as handle:
        arrays={key:handle["scf/"+name][()] for key,name in
            (("MO_coeff","mo_coeff"),("MO_occ","mo_occ"),("MO_energy","mo_energy"))}
        checkpoint_molecule=json.loads(handle["mol"][()])
    for key in ("_atm","_bas","_env"):
        require(np.array_equal(np.asarray(checkpoint_molecule[key]),getattr(original.mol,key)),"Certifiedzero checkpoint molecule differs from exactsource")
    arrays["previous-density"]=np.load(checked(point["audit_state_files"]["previous-density"]),allow_pickle=False)
    arrays["full-orbital-gradient"]=np.load(checked(point["audit_state_files"]["full-orbital-gradient"]),allow_pickle=False)
    arrays["current-density"]=original.make_rdm1(arrays["MO_coeff"],arrays["MO_occ"])
    for key in ("MO_coeff","MO_occ","previous-density","full-orbital-gradient"):
        digest=hashlib.sha256(np.ascontiguousarray(arrays[key]).tobytes()).hexdigest()
        require(digest==point["audit_state_fingerprints"][key]["payload_sha256_C_order"],"Compact checkpoint/audit payload fingerprint differs")
    for key in ("previous-density","current-density","MO_coeff"):
        require(arrays[key].shape==(762,762) and arrays[key].dtype==np.float64 and np.isfinite(arrays[key]).all(),"Certifiedzero fullAO matrix differs")
    for key in ("MO_occ","MO_energy"):
        require(arrays[key].shape==(762,) and arrays[key].dtype==np.float64 and np.isfinite(arrays[key]).all(),"Certifiedzero fullMO vector differs")
    c,occ,dm=(arrays[key] for key in ("MO_coeff","MO_occ","current-density"))
    require(np.isin(occ,[0.,2.]).all() and float(occ.sum())==320
        and np.max(np.abs((c*occ)@c.T-dm))<1e-12,"Certifiedzero density/MO/electron replay failed")
    overlap=original.get_ovlp();trace=float(np.einsum("ij,ji->",dm,overlap))
    orthogonality=float(np.max(np.abs(c.T@overlap@c-np.eye(762))))
    vector=arrays["full-orbital-gradient"]
    require(vector.shape==(160*(762-160),) and vector.dtype==np.float64 and np.isfinite(vector).all(),"Complete true orbital-gradient vector required")
    gradient=float(np.linalg.norm(vector))
    require(abs(trace-320)<1e-7 and orthogonality<1e-7 and gradient<GRADIENT_TOLERANCE
        and math.isclose(gradient,point["endpoint_audit"]["full_current_orbital_gradient_norm"],rel_tol=1e-9,abs_tol=1e-13),
        "Certifiedzero archived orbital-gradient/metric scalar replay failed")
    arrays["replay_scope"]="CHECKPOINT_MO_AND_SAVED_ORBITAL_GRADIENT_REPLAY_NO_NEW_FOCK_EVALUATION"
    return arrays


def endpoint_audit(mf, pair, checks, *, np, frozen):
    current=np.array(mf.make_rdm1(),copy=True);previous=pair["previous"]
    require(np.array_equal(current,pair["current"]) and not np.shares_memory(previous,pair["current"]),
        "Actual consecutive copied native densities required")
    hcore=mf.get_hcore();overlap=mf.get_ovlp()
    prior_veff=mf.get_veff(mf.mol,previous,dm_last=None,vhf_last=None)
    prior_energy=float(mf.energy_tot(previous,hcore,prior_veff))
    repeat=[];veff=None
    current_digest=hashlib.sha256(current.tobytes()).hexdigest()
    for _ in range(3):
        veff=mf.get_veff(mf.mol,current,dm_last=None,vhf_last=None)
        repeat.append(float(mf.energy_tot(current,hcore,veff)))
        require(hashlib.sha256(current.tobytes()).hexdigest()==current_digest,"Fixed repeat density changed")
    fock=mf.get_fock(h1e=hcore,s1e=overlap,vhf=veff,dm=current,cycle=-1,diis=None,level_shift_factor=0,damp_factor=0)
    gradient_vector=mf.get_grad(mf.mo_coeff,mf.mo_occ,fock)
    gradient=float(np.linalg.norm(gradient_vector))
    audit=frozen.audited_endpoint_qualification(mf.converged,pair["check"],prior_energy,repeat[-1],gradient,
        preceding_main_check=pair.get("main_check",{}),native_checks_finite=all(frozen.finite_native_scf_check(c) for c in checks),
        density_change_norm=float(np.linalg.norm(current-previous)),energy_tolerance=ENERGY_TOLERANCE,
        orbital_gradient_tolerance=GRADIENT_TOLERANCE)
    trace=float(np.einsum("ij,ji->",current,overlap));ortho=float(np.max(np.abs(mf.mo_coeff.T@overlap@mf.mo_coeff-np.eye(mf.mol.nao_nr()))))
    audit.update(full_same_density_energy_repeats_hartree=repeat,
        full_same_density_energy_spread_hartree=max(repeat)-min(repeat),
        same_density_repeat_payload_sha256=current_digest,energy_ulp_hartree=math.ulp(repeat[-1]),
        density_overlap_electron_trace=trace,MO_overlap_orthonormality_max_abs_error=ortho,
        native_kernel_energy_hartree=float(mf.e_tot),
        full_minus_native_kernel_energy_hartree=repeat[-1]-float(mf.e_tot),
        energy_drift_is_rigorous_energy_error_bound=False,
        full_VEFF_applied_every_cycle=getattr(mf,"_research4_full_native_veff_policy",None) is not None)
    audit["qualified"]=bool(audit["qualified"] and max(repeat)-min(repeat)<=REPEAT_SPREAD_TOLERANCE
        and abs(trace-mf.mol.nelectron)<1e-7 and ortho<1e-7)
    return audit,{"previous-density":previous,"current-density":current,"MO_coeff":mf.mo_coeff,
        "MO_occ":mf.mo_occ,"MO_energy":np.einsum("pi,pi->i",mf.mo_coeff,fock@mf.mo_coeff),"full_Fock":fock,
        "full-orbital-gradient":gradient_vector}


def compute_point(name, dm0, mol, original, binding, context, *, default_incremental=False,
                  operator_probe_reference=None,displacement=None):
    """Exactly one bounded native DIIS kernel; a resource kill leaves rolling state."""
    args,out,w,base,np,gen_grid,rks,helper,frozen,emit,check_budget=context
    pointout=out/name;pointout.mkdir();local=out/"local-only"/name;local.mkdir(parents=True)
    state={"schema":POINT_SCHEMA,"status":"PARTIAL","source_worker":ref(__file__),"name":name,
        "energy_completed":False,"SCF_performed":False,"SCF_endpoint_certified":False,
        "atoms":mol.natm,"basis_functions":mol.nao_nr(),"electrons":mol.nelectron,
        "maximum_cycles":args.max_cycles,"SCF_kernel_calls":1,"native_direct_driver_threads":1,"worker_threads":2,
        "VEFF_cycle_policy":"DEFAULT_NATIVE_INCREMENTAL_CONTROL" if default_incremental else helper.FULL_BUILD_POLICY,
        "storage_policy":STORAGE_POLICY,"all_intermediate_raw_matrices_archived":False,
        "omitted_raw_matrix_files":["initial-density","current-density","MO_coeff-duplicate","full_Fock"],
        "omitted_matrices_have_fingerprints_only_not_surviving_file_references":True,
        "later_Fock_reconstruction_requires_new_guarded_quantum_evaluation":True,
        "original_source_not_mutated":True,"strict_SCF_convergence_checks":[],"output_directory":str(pointout)}
    state["displacement"]=displacement or {"mode_index_1based":None,"step_bohr":0.,"sign":0}
    state["coordinate_arrays"]={}
    for key,value in (("source_coordinates_bohr",original.mol.atom_coords(unit="Bohr")),("actual_coordinates_bohr",mol.atom_coords(unit="Bohr"))):
        base.atomic_npy(pointout/(key+".npy"),value);state["coordinate_arrays"][key]=ref(pointout/(key+".npy"))
    log=(pointout/"pyscf.log").open("x",buffering=1);mol.stdout=log
    settings=dict(binding["method_settings"]);settings.update(threads=2,native_direct_driver_threads=1,
        scf_conv_tol_hartree=ENERGY_TOLERANCE,scf_max_cycle=args.max_cycles)
    mf=w.make_scf(mol,settings,local/"rolling.chk");mf.conv_tol_grad=GRADIENT_TOLERANCE
    fixed=frozen.attach_fixed_quadrature(mf,original,w,base,np,gen_grid)
    if not default_incremental:
        mf.__class__=helper.full_native_veff_type(type(mf),rks.get_veff,emit=emit)
    state.update(settings=settings,fixed_quadrature=fixed)
    import h5py
    from pyscf.scf import chkfile
    # A real initial density/molecule checkpoint before the first Coulomb build.
    with h5py.File(mf.chkfile,"w") as handle:
        handle["mol"]=mol.dumps();handle["restart/initial_density"]=dm0
        handle["restart/scope"]="INITIAL_DENSITY_ONLY_NOT_CERTIFIED_SCF"
    with Path(mf.chkfile).open("rb") as handle:os.fsync(handle.fileno())
    state["before_first_J_initial_density_checkpoint_sha256"]=sha(mf.chkfile)
    state["initial_density_fingerprint"]=base.array_fingerprint(dm0)
    state["initial_checkpoint_retention_scope"]="ROLLING_RESTART_OVERWRITTEN_BY_NEXT_ACTUAL_NATIVE_SCF_ENDPOINT"
    pair={};started=time.process_time()
    def checkpoint(self,env):
        target=Path(self.chkfile);temporary=target.with_name("rolling-next.chk")
        require(not temporary.exists(),"Unexpected pending checkpoint requires review")
        chkfile.dump_scf(self.mol,str(temporary),env["e_tot"],env["mo_energy"],env["mo_coeff"],env["mo_occ"],overwrite_mol=True)
        with temporary.open("rb") as handle:os.fsync(handle.fileno())
        os.replace(temporary,target)
        directory=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
        emit("rolling_checkpoint_committed",{"point":name,"cycle_0based":int(env.get("cycle",-1)),"bytes":target.stat().st_size})
        return self
    mf.dump_chk=types.MethodType(checkpoint,mf)
    def check(env):
        row=frozen.polish_convergence_check(env,main_energy_tolerance=ENERGY_TOLERANCE,
            main_orbital_gradient_trigger=GRADIENT_TOLERANCE,final_energy_tolerance=ENERGY_TOLERANCE,
            final_orbital_gradient_tolerance=GRADIENT_TOLERANCE)
        row.update(kernel_call_1based=1,cycle_0based=int(env.get("cycle",-1)),energy_hartree=float(env["e_tot"]))
        state["strict_SCF_convergence_checks"].append(row)
        pair.update(previous=np.array(env["dm_last"],copy=True),current=np.array(env["dm"],copy=True),check=row)
        if row["phase"]=="SCF_MAIN_CYCLE":pair["main_check"]=row
        emit("SCF_convergence_check",{"point":name,**row});save(pointout/"progress.json",state)
        return row["accepted"]
    mf.check_convergence=check
    mf.callback=lambda env:check_budget()  # Native kernel has committed this cycle's checkpoint first.
    try:
        check_budget()
        if operator_probe_reference is not None:
            state["same_density_operator_probe"]=helper.compare_same_density_veff(mf,dm0,operator_probe_reference,
                native_get_veff=rks.get_veff,np=np)
            state["same_density_operator_probe"]["scope"]="ONE_NEW_INCREMENTAL_STEP_NOT_HISTORICAL_ACCUMULATED_OPERATOR_REPLAY"
            emit("Spinosin_checkpoint_same_density_operator_probe",state["same_density_operator_probe"])
            save(pointout/"progress.json",state);check_budget()
        state["SCF_performed"]=True;mf.kernel(dm0=np.array(dm0,copy=True))
        require(pair and math.isfinite(float(mf.e_tot)),"Actual complete native endpoint required")
        audit,arrays=endpoint_audit(mf,pair,state["strict_SCF_convergence_checks"],np=np,frozen=frozen)
        state["endpoint_audit"]=audit;state["native_kernel_converged"]=bool(mf.converged)
        state["audit_state_fingerprints"]={key:base.array_fingerprint(value) for key,value in arrays.items()}
        state["audit_state_files"]={}
        for key,filename in (("previous-density","previous-density.npy"),
                             ("full-orbital-gradient","full-orbital-gradient.npy"),
                             ("MO_energy","refreshed-MO-energy.npy")):
            base.atomic_npy(local/filename,arrays[key])
            state["audit_state_files"]["refreshed-MO-energy" if key=="MO_energy" else key]=ref(local/filename)
        if audit["qualified"]:
            mf.e_tot=audit["current_full_energy_hartree"];mf.mo_energy=arrays["MO_energy"];mf.converged=True
            state.update(status="FULL_NATIVE_ENDPOINT_CERTIFIED",energy_completed=True,SCF_endpoint_certified=True,
                energy_hartree=float(mf.e_tot),SCF_convergence_scope=("DEFAULT_NATIVE_INCREMENTAL_WITH_FULL_ENDPOINT_AUDIT"
                    if default_incremental else "FULL_NATIVE_EVERY_CYCLE_AND_INDEPENDENT_ENDPOINT_AUDIT"))
        else:
            state["status"]="BOUNDED_SCF_MEASUREMENT_COMPLETE_UNCERTIFIED"
        frozen.verify_fixed_quadrature(mf,fixed,w,base)
        state["checkpoint"]=ref(mf.chkfile)
        state["rolling_checkpoint_scope"]="LATEST_ACTUAL_NATIVE_KERNEL_ENDPOINT_NOT_POSTAUDIT_MO_ENERGY_REFRESH"
        state["current_density_reconstruction"]="NATIVE_OCCUPIED_ONLY_MAKE_RDM1_FROM_EXACT_CHECKPOINT_MOS; FLOATING_REPLAY_NOT_BITWISE_PROMISED"
        state["current_density_reconstruction_numeric_tolerance"]=1e-12
        state["local_audit_bytes"]=sum(p.stat().st_size for p in local.iterdir() if p.is_file())
        require(state["local_audit_bytes"]<=POINT_STORAGE_RESERVE_BYTES,"Compact point exceeded12MiB storage envelope")
        save(pointout/"native_result.json",state)
        return state,mf
    finally:
        state["worker_CPU_seconds"]=time.process_time()-started
        if Path(mf.chkfile).exists():state["checkpoint"]=ref(mf.chkfile)
        save(pointout/"progress.json",state);log.close()


def execute(args):
    validate_stage_args(args);frozen=old();guard=load(GUARD,GUARD_SHA,"fullnative_actual_guard")
    supervision=guard.verify_resource_supervision(expected_threads=2)
    gate,seal=source_gate(args,guard,supervision)
    source=gate["result"]
    require((args.stage=="controls")== (source["case"]!="Spinosin"),"Stage/source identity differs")
    release=verify_control_release(args,guard) if args.stage=="zero" else None
    analyzer=load(frozen.ANALYZER,frozen.ANALYZER_SHA,"fullnative_bound_analyzer")
    with frozen.closed_main(args,analyzer) as main_ref:
        return execute_closed(args,guard,supervision,gate,seal,main_ref,release)


def execute_closed(args,guard,supervision,gate,seal,main_ref,control_release=None):
    frozen=old();source=gate["result"]
    import numpy as np
    import h5py
    from pyscf import lib
    from pyscf.dft import gen_grid,rks
    from pyscf.scf import _vhf,addons
    lib.num_threads(2)
    helper=load(HELPERS,sha(HELPERS),"fullnative_restart_helpers")
    w=load(frozen.load(frozen.ANALYZER,frozen.ANALYZER_SHA,"fullnative_analyzer_state").STATE,
           "be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8","fullnative_original_restorer")
    base=w.frozen_base();out=fresh(args.output);started=time.process_time();wall=time.perf_counter()
    state={"schema":SCHEMA,"status":"PARTIAL","stage":args.stage,"source_worker":ref(__file__),
        "helper_source":ref(HELPERS),"resource_supervision":supervision,"output_directory":str(out),
        "source_result":{"path":gate["result_path"],"sha256":sha(gate["result_path"])},
        "snapshot_manifest_sha256":source["snapshot_manifest_sha256"],"case":source["case"],
        "source_qualification":seal,"closed_main_pipeline_state":main_ref,"zero_certified":False,"points":{},
        "passing_water_and_ethanol_release":control_release,
        "stable_MF_state_unchanged":False,
        "completed_points":[],"energies_hartree":{},"old_scientific_results_modified":False,
        "full_Hessian_performed":False,"optimization_performed":False,"IR_intensities_performed":False,
        "diagnostic_scope":"LOCAL_FINITE_ENERGY_DIAGNOSTIC_NOT_FULL_HESSIAN_MINIMUM_FREQUENCY",
        "native_direct_driver_threads":1,"worker_threads":2,"maximum_cycles_per_point":args.max_cycles,
        "main_and_final_energy_tolerance_hartree":ENERGY_TOLERANCE,
        "main_and_final_orbital_gradient_tolerance":GRADIENT_TOLERANCE,
        "same_density_repeat_energy_spread_tolerance_hartree":REPEAT_SPREAD_TOLERANCE,
        "local_only_state_maximum_bytes":LOCAL_STATE_MAX_BYTES,"automatic_retry":False}
    save(out/"input.json",{**state,"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}})
    def emit(event,detail=None):
        row={"event":event,"utc":datetime.now(timezone.utc).isoformat(),"CPU_seconds":time.process_time()-started,
             "wall_seconds":time.perf_counter()-wall,**(detail or {})}
        with (out/"events.jsonl").open("a") as handle:handle.write(json.dumps(row,allow_nan=False)+"\n")
        save(out/"progress.json",{**state,"latest_event":row})
    def check_budget():
        proof=supervision["proof"]
        if time.process_time()>=proof["actual_cpu_stop_threshold_seconds"]:
            raise base.BudgetStop("Actual CPU cleanup threshold reached")
        require(datetime.now(timezone.utc)<datetime.fromisoformat(read(LEDGER)["effective_run_deadline_at_utc"]),"New authorization deadline reached")
        size=sum(p.stat().st_size for p in (out/"local-only").rglob("*") if p.is_file()) if (out/"local-only").exists() else 0
        require(size<=LOCAL_STATE_MAX_BYTES,"Local raw audit state disk envelope exceeded")
    def stop(signum,frame):raise base.BudgetStop("Resource signal "+str(signum))
    handlers={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGXCPU)}
    native=base.NativeDirectDriverThreadGuard(_vhf,lib.with_omp_threads,lib.num_threads,native_threads=1,
        emit=lambda detail:emit("native_direct_driver_signature",detail))
    original=None;log=(out/"restoration.log").open("x",buffering=1);extra_logs=[]
    try:
        planned_points=2 if args.stage=="controls" else 1 if args.stage=="zero" else 4
        state["compact_storage_preflight"]=free_space_preflight(out,planned_points)
        native.install();check_budget()
        original,binding=w.restore_snapshot(Path(source["snapshot_directory"]),expected_manifest_sha256=source["snapshot_manifest_sha256"],
            expected_source_sha256="be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8",stdout=log)
        require(binding==source["source_state_binding"],"Original restored state binding differs")
        fingerprint=w.molecule_physical_fingerprint(original.mol);payload=frozen.original_mf_fingerprints(original,base)
        context=(args,out,w,base,np,gen_grid,rks,helper,frozen,emit,check_budget)
        points_started=0
        def point(name,dm,mol,default=False,operator_probe_reference=None,displacement=None):
            nonlocal points_started
            emit("compact_storage_before_point",free_space_preflight(out,planned_points-points_started))
            points_started+=1
            value,mf=compute_point(name,dm,mol,original,binding,context,default_incremental=default,
                operator_probe_reference=operator_probe_reference,displacement=displacement)
            state["points"][name]={"result":ref(out/name/"native_result.json"),"status":value["status"],
                "energy_completed":value["energy_completed"]}
            if value["energy_completed"]:
                state["points"][name]["energy_hartree"]=value["energy_hartree"]
                state["completed_points"].append(name);state["energies_hartree"][name]=value["energy_hartree"]
            frozen.ensure_original_unchanged(original,fingerprint,payload,w,base);check_budget()
            return value,mf
        if args.stage=="controls":
            initial=np.array(original.make_rdm1(),copy=True)
            full,fullmf=point("full-native",initial,original.mol.copy())
            default,defaultmf=point("default-incremental",initial,original.mol.copy(),True)
            require(full["SCF_endpoint_certified"] and default["SCF_endpoint_certified"],"Both control variants require full endpoint certification")
            control_log=(out/"control-gradients-and-operator-probe.log").open("x",buffering=1);extra_logs.append(control_log)
            fullmf.stdout=fullmf.mol.stdout=control_log;defaultmf.stdout=defaultmf.mol.stdout=control_log
            probe=helper.compare_same_density_veff(fullmf,fullmf.make_rdm1(),initial,native_get_veff=rks.get_veff,np=np)
            fullgrad=fullmf.nuc_grad_method();fullgrad.grid_response=False;fg=fullgrad.kernel()
            defaultgrad=defaultmf.nuc_grad_method();defaultgrad.grid_response=False;dg=defaultgrad.kernel()
            require(np.isfinite(fg).all() and np.isfinite(dg).all(),"Finite full control gradients required")
            base.atomic_npy(out/"full-gradient.npy",fg);base.atomic_npy(out/"default-gradient.npy",dg)
            # Small controls also verify the declared native1/native2 change.
            native.restore()
            thread2=base.NativeDirectDriverThreadGuard(_vhf,lib.with_omp_threads,lib.num_threads,native_threads=2,
                emit=lambda detail:emit("control_native2_driver_signature",detail))
            try:
                thread2.install();same_dm=np.array(fullmf.make_rdm1(),copy=True)
                thread2_veff=fullmf.get_veff(fullmf.mol,same_dm,dm_last=None,vhf_last=None)
                thread2_energy=float(fullmf.energy_tot(same_dm,fullmf.get_hcore(),thread2_veff))
                thread2_gradient=fullmf.nuc_grad_method();thread2_gradient.grid_response=False;tg=thread2_gradient.kernel()
            finally:
                thread2.restore();state["control_native2_driver_control"]=thread2.summary();native.install()
            require(np.isfinite(tg).all(),"Finite native2 control gradient required")
            require(state["control_native2_driver_control"]["original_driver_restored"]
                and state["control_native2_driver_control"]["restoration_failures"]==0,"Control native2 restoration failed")
            base.atomic_npy(out/"native2-full-gradient.npy",tg)
            parity={"energy_abs_difference_hartree":abs(full["energy_hartree"]-default["energy_hartree"]),
                "gradient_max_abs_difference_hartree_per_bohr":float(np.max(np.abs(fg-dg))),
                "same_density_native1_vs_native2_energy_abs_difference_hartree":abs(full["energy_hartree"]-thread2_energy),
                "native1_vs_native2_gradient_max_abs_difference_hartree_per_bohr":float(np.max(np.abs(fg-tg))),
                "native2_full_energy_hartree":thread2_energy,"native2_full_gradient":ref(out/"native2-full-gradient.npy"),
                "energy_tolerance_hartree":1e-10,"gradient_tolerance_hartree_per_bohr":1e-8,
                "same_density_operator_probe":probe,"full_gradient":ref(out/"full-gradient.npy"),"default_gradient":ref(out/"default-gradient.npy")}
            parity["passed"]=parity["energy_abs_difference_hartree"]<=1e-10 and parity["gradient_max_abs_difference_hartree_per_bohr"]<=1e-8 \
                and parity["same_density_native1_vs_native2_energy_abs_difference_hartree"]<=1e-10 \
                and parity["native1_vs_native2_gradient_max_abs_difference_hartree_per_bohr"]<=1e-8
            state["control_parity"]=parity;require(parity["passed"],"Control full/default parity failed")
            state["status"]="FULL_NATIVE_CONTROL_PARITY_PASSED"
        elif args.stage=="zero":
            if args.source_zero_result:
                prior,priorpoint,priorref=dependency(args,guard,certified=False);state["source_zero_result"]=priorref
                require(prior.get("maximum_cycles_per_point")==30 and prior.get("source_zero_result") is None,
                    "Only one declared80cycle continuation of the original30cycle pilot is permitted")
                if prior["zero_certified"]:
                    verify_certified_point(priorpoint)
                    replay_zero_arrays(priorpoint,original,np=np)
                    state.update(status="REUSED_CERTIFIED_ZERO",zero_certified=True,points=prior["points"],
                        completed_points=prior["completed_points"],energies_hartree=prior["energies_hartree"],
                        no_new_SCF_because_prior_zero_already_certified=True)
                    return state
                checkpoint=checked(priorpoint["checkpoint"])
            else:
                checkpoint=checked({"path":str(CHECKPOINT),"sha256":CHECKPOINT_SHA})
            with h5py.File(checkpoint,"r") as handle:
                cp={"molecule":handle["mol"][()],"energy_hartree":handle["scf/e_tot"][()],
                    **{k:handle["scf/"+k][()] for k in ("mo_coeff","mo_energy","mo_occ")}}
            arrays,metadata=helper.validate_checkpoint_restart_payload(cp,read(Path(source["snapshot_directory"])/"molecule.json"),
                np=np,atoms=75,basis_functions=762,electrons=320)
            # Actual overlap checks occur only under this guardian, before SCF.
            overlap=original.get_ovlp();trace=float(np.einsum("ij,ji->",arrays["density"],overlap))
            orthogonality=float(np.max(np.abs(arrays["mo_coeff"].T@overlap@arrays["mo_coeff"]-np.eye(762))))
            require(abs(trace-320)<1e-7 and orthogonality<1e-7,"Checkpoint AO metric/electron check failed")
            metadata.update(density_overlap_electron_trace=trace,MO_overlap_orthonormality_max_abs_error=orthogonality)
            state.update(restart_checkpoint=ref(checkpoint),restart_validation=metadata)
            zero,zeromf=point("fresh-zero",arrays["density"],original.mol.copy(),
                operator_probe_reference=np.array(original.make_rdm1(),copy=True) if args.source_zero_result is None else None)
            state["zero_certified"]=zero["SCF_endpoint_certified"]
            state["status"]="ZERO_PILOT_COMPLETE" if args.max_cycles==30 else "ZERO_CONTINUATION_COMPLETE"
        else:
            prior,priorpoint,priorref=dependency(args,guard,certified=True);state["source_zero_result"]=priorref
            require(prior["snapshot_manifest_sha256"]==source["snapshot_manifest_sha256"],"Certified zero uses different source")
            require(priorpoint["SCF_endpoint_certified"] is True and priorpoint["endpoint_audit"]["qualified"] is True,"Zero audit required")
            dmzero=replay_zero_arrays(priorpoint,original,np=np)["current-density"]
            require(dmzero.shape==(762,762) and np.isfinite(dmzero).all(),"Complete zero density required")
            state.update(zero_certified=True,shared_certified_zero_checkpoint=priorpoint["checkpoint"],
                shared_certified_zero_density_fingerprint=base.array_fingerprint(dmzero),
                certified_zero_energy_hartree=priorpoint["energy_hartree"],predetermined_mode_index_1based=args.mode_index,predetermined_steps_bohr=list(STEPS))
            directionargs=argparse.Namespace(source_result=Path(gate["result_path"]),mode_index=args.mode_index,
                coordinate_index=None,mode_array_sha256=args.mode_array_sha256)
            q,hq,analytic,refs=frozen.direction_and_hessian(directionargs,gate,np)
            base.atomic_npy(out/"direction.npy",q);state["analytic_directional_curvature_Eh_per_Bohr2"]=analytic
            state["source_arrays"]=refs;analyses=[]
            for step in STEPS:
                values={}
                for sign,label in ((1,"plus"),(-1,"minus")):
                    name="h%.4f-%s"%(step,label);mol=original.mol.copy()
                    mol.set_geom_(original.mol.atom_coords(unit="Bohr")+sign*step*q,unit="Bohr",inplace=True)
                    dm=addons.project_dm_nr2nr(original.mol,dmzero,mol)
                    value,mf=point(name,dm,mol,displacement={"mode_index_1based":args.mode_index,
                        "step_bohr":step,"sign":sign,"direction_normalization":"EUCLIDEAN_CARTESIAN_UNIT_NORM"})
                    require(value["SCF_endpoint_certified"],"A displacement is uncertified; no further geometry attempted")
                    values[label]=value["energy_hartree"]
                analysis=frozen.central_energy_curvature(priorpoint["energy_hartree"],values["plus"],values["minus"],step)
                analysis.update(step_bohr=step,analytic_directional_curvature_Eh_per_Bohr2=analytic,
                    central_energy_minus_analytic_curvature_Eh_per_Bohr2=analysis["central_energy_curvature_Eh_per_Bohr2"]-analytic)
                analyses.append(analysis);state["central_energy_analyses"]=analyses
            state["h_halving_curvature_differences_Eh_per_Bohr2"]=[analyses[1]["central_energy_curvature_Eh_per_Bohr2"]-analyses[0]["central_energy_curvature_Eh_per_Bohr2"]]
            state["status"]="FOUR_DISPLACEMENT_ENERGIES_COMPUTED"
        frozen.ensure_original_unchanged(original,fingerprint,payload,w,base)
        state.update(original_MF_unchanged=True,stable_MF_state_unchanged=True)
    except BaseException as exc:
        state["status"]="PARTIAL_RESOURCE_STOP" if isinstance(exc,base.BudgetStop) else "FAILED_FULL_NATIVE_RESEARCH"
        state["error"]={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        emit("worker_stopped",{"status":state["status"],"error":state["error"]})
    finally:
        if original is not None and "payload" in locals():
            try:
                frozen.ensure_original_unchanged(original,fingerprint,payload,w,base)
                state.update(original_MF_unchanged=True,stable_MF_state_unchanged=True)
            except ValueError as exc:
                state.update(status="FAILED_ORIGINAL_SOURCE_MUTATED",original_MF_unchanged=False,
                             stable_MF_state_unchanged=False,source_mutation_error=str(exc))
        native.restore();state["native_direct_driver_control"]=native.summary()
        if not state["native_direct_driver_control"].get("original_driver_restored") or state["native_direct_driver_control"].get("restoration_failures")!=0:
            state["status"]="FAILED_NATIVE_DRIVER_RESTORATION"
        state.update(ended_at_utc=datetime.now(timezone.utc).isoformat(),worker_CPU_seconds=time.process_time()-started,
                     worker_wall_seconds=time.perf_counter()-wall,energies=dict(state["energies_hartree"]))
        for sig,handler in handlers.items():signal.signal(sig,handler)
        for handle in extra_logs:handle.close()
        log.close();save(out/"result.json",state)
    return state


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    action=p.add_mutually_exclusive_group(required=True);action.add_argument("--execute",action="store_true");action.add_argument("--qualify-source-only",action="store_true")
    p.add_argument("--stage",choices=("controls","zero","energies"),default="zero")
    p.add_argument("--source-result",type=Path);p.add_argument("--source-result-sha256")
    p.add_argument("--control-validation",action="store_true")
    p.add_argument("--source-qualification",type=Path);p.add_argument("--source-qualification-sha256")
    p.add_argument("--allowed-phase-schedule",type=Path);p.add_argument("--allowed-phase-schedule-sha256")
    p.add_argument("--main-pipeline-state",type=Path,required=True);p.add_argument("--main-pipeline-state-sha256",required=True)
    p.add_argument("--max-cycles",type=int,default=30);p.add_argument("--threads",type=int,default=2)
    p.add_argument("--source-zero-result",type=Path);p.add_argument("--source-zero-result-sha256")
    for case in ("water","ethanol"):
        p.add_argument("--"+case+"-control-result",type=Path);p.add_argument("--"+case+"-control-result-sha256")
    p.add_argument("--mode-index",type=int);p.add_argument("--mode-array-sha256")
    p.add_argument("--steps",type=float,nargs="+",default=list(STEPS))
    return p


def main():
    args=parser().parse_args()
    if args.qualify_source_only:
        require(args.source_result is not None and args.source_result_sha256 and args.allowed_phase_schedule is not None
            and args.allowed_phase_schedule_sha256,"Qualification requires hash-bound source and schedule")
        value=qualify(args)
    else:
        require(args.source_qualification is not None and args.source_qualification_sha256,"Execution requires sealed original source")
        require((args.source_zero_result is None)==(args.source_zero_result_sha256 is None),"Dependency result and SHA must be paired")
        value=execute(args)
    print(json.dumps({"schema":value["schema"],"status":value["status"],"output":str(args.output)},allow_nan=False))
    successful=value["status"] in ("SOURCE_QUALIFIED_OUTSIDE_QUANTUM_PHASE","FULL_NATIVE_CONTROL_PARITY_PASSED",
        "ZERO_PILOT_COMPLETE","ZERO_CONTINUATION_COMPLETE","REUSED_CERTIFIED_ZERO","FOUR_DISPLACEMENT_ENERGIES_COMPUTED")
    return 0 if successful else 2


if __name__=="__main__":
    raise SystemExit(main())
