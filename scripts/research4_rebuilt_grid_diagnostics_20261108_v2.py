"""Fresh moving-grid diagnostics with two bounded native SCF polishing calls.

All quantum operations require the actual new72h guardian.  The immutable
fixed-grid worker supplies authenticated input states and numerical audit
helpers only; this file is the producer of every new result and point.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback
import types

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/"scripts/research4_full_native_checkpoint_worker_20261107.py"
FROZEN_SHA="f93bc6eab66d1d3d1909edff25bc01b2231e1cfa49b23d45bba51e2e6706b03a"
GUARD=ROOT/"scripts/run_research4_research72h_budget_20261107.py"
GUARD_SHA="643a431b7229eca94ac50645d467e6c3472f976b8ac48cc2f36be68241319e68"
HELPERS=ROOT/"scripts/research4_full_native_veff_restart_helpers_20261107.py"
HELPERS_SHA="ea2e3b6c130cbc68f189fc7c721b93aeb5600ec197a3df591e7f3b464f0d9d6b"
SCHEMA="research4-rebuilt-moving-grid-diagnostics-v2"
POINT_SCHEMA="research4-rebuilt-moving-grid-energy-point-v2"
CONTROL_SUCCESS="REBUILT_GRID_GRADIENT_CONTROLS_PASSED"
SPINOSIN_SUCCESS="REBUILT_GRID_SPINOSIN_DIAGNOSTIC_COMPUTED"
CONTROL_SHIFT_BOHR=.01
CONTROL_STEPS=(.002,.001)
SPINOSIN_STEP=.005
GRADIENT_FD_TOLERANCE=1e-5
HALVING_TOLERANCE=1e-5
TOTAL_GRADIENT_TOLERANCE=1e-6
ENERGY_TOLERANCE=1e-12
GRADIENT_TOLERANCE=1e-9
MAIN_GRADIENT_TRIGGER=1e-10
MAX_SCF_KERNEL_CALLS=2
STORAGE_POLICY="COMPACT_RESTART_AND_AUDIT_HASHES"
POINT_STORAGE_RESERVE_BYTES=12*1024**2


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def load(path,expected,name):
    require(sha(path)==expected,"Frozen source changed: "+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def frozen_worker():
    return load(FROZEN,FROZEN_SHA,"rebuilt_grid_frozen_input_worker")


def ref(path):
    p=Path(path).resolve();return {"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size}


def save(path,value):
    return frozen_worker().save(path,value)


def utc():return datetime.now(timezone.utc).isoformat()


def validate_stage_args(args):
    require(args.stage in ("controls","spinosin") and args.threads==2 and args.max_cycles==80,
            "At most two finite80cycle native DIIS calls per point, outer2/native1 required")
    require(args.grid_level in (1,3),"Only predeclared fresh grid levels1 or3 allowed")
    if args.stage=="controls":
        require(args.grid_level==1 and not args.with_displacements,"Controls use grid1 and exactlyfive points per molecule")
    else:
        require(args.source_zero_result is not None and args.source_zero_result_sha256
                and args.grid_control_result is not None and args.grid_control_result_sha256,
                "Spinosin requires authenticated fixed-grid zero input and actual moving-grid controls")
        if args.with_displacements:require(isinstance(args.mode_array_sha256,str) and len(args.mode_array_sha256)==64,"Mode1 array SHA required")


def attach_rebuilt_quadrature(mf,original,level,w,base,np,gen_grid):
    """A new standard atom-centered grid on the actual geometry, never fixed."""
    grids=gen_grid.Grids(mf.mol)
    for name in w.GRID_ATTRIBUTES:
        if hasattr(original.grids,name):setattr(grids,name,copy.deepcopy(getattr(original.grids,name)))
    grids.level=level;grids.stdout=mf.mol.stdout;grids.verbose=original.grids.verbose
    mf.grids=grids;mf.small_rho_cutoff=0;mf.disp=None;mf.nlc=original.nlc
    mf._numint.cutoff=original._numint.cutoff;mf._numint.omega=original._numint.omega
    grids.build(with_non0tab=True)
    require(mf.grids.mol is mf.mol and mf.direct_scf is True and mf._eri is None
            and getattr(mf,"with_df",None) is None and mf.direct_scf_tol==1e-13
            and mf._numint.cutoff==1e-13 and grids.cutoff==1e-15 and not mf.do_nlc(),"Original full native PBE direct cutoffs required")
    require(grids.coords.ndim==2 and grids.coords.shape[1]==3 and grids.weights.shape==(len(grids.coords),)
            and np.isfinite(grids.coords).all() and np.isfinite(grids.weights).all(),"Fresh finite complete quadrature required")
    return {"method":"FRESH_REBUILT_MOVING_ATOM_CENTERED_QUADRATURE","grid_level":level,
        "grid_settings":w.grid_settings(grids),"grid_points":len(grids.coords),
        "coordinate_fingerprint":base.array_fingerprint(grids.coords),"weight_fingerprint":base.array_fingerprint(grids.weights),
        "small_rho_cutoff":0,"grid_cutoff":grids.cutoff,"numint_cutoff":mf._numint.cutoff,
        "geometry_specific_masks_built":grids.non0tab is not None,"dense_grid_arrays_archived":False,
        "original_fixed_grid_coordinates_or_weights_attached":False}


def verify_rebuilt_quadrature(mf,proof,w,base):
    require(mf.grids.mol is mf.mol and mf.small_rho_cutoff==0 and mf.grids.level==proof["grid_level"]
            and w.grid_settings(mf.grids)==proof["grid_settings"]
            and base.array_fingerprint(mf.grids.coords)==proof["coordinate_fingerprint"]
            and base.array_fingerprint(mf.grids.weights)==proof["weight_fingerprint"],"Fresh-grid functional changed within one geometry")
    proof["unchanged_within_geometry"]=True


def endpoint_audit(*args,**kwargs):
    return frozen_worker().endpoint_audit(*args,**kwargs)


def bounded_kernel_sequence(mf,dm0,*,np,pair,checks,emit,check_budget,state,fingerprint,auditor):
    """First qualified actual endpoint, with a fresh native DIIS per kernel.

    A second call uses only the immediately preceding actual final density.
    Both unsuccessful calls and their independent audits remain in the new
    result. This does not relabel any old point or relax its final thresholds.
    """
    require(mf.diis is True,"Fresh native DIIS object per actual kernel call required")
    require(state["maximum_cycles"]==80,"Finite80cycle call limit required")
    require(0<MAIN_GRADIENT_TRIGGER<GRADIENT_TOLERANCE and MAX_SCF_KERNEL_CALLS==2,
            "Tighter main trigger and exactly two-call ceiling required")
    state["SCF_polishing_attempts"]=[]
    start=np.array(dm0,copy=True)
    for call in range(1,MAX_SCF_KERNEL_CALLS+1):
        check_budget();pair.clear()
        mf.conv_tol=ENERGY_TOLERANCE;mf.conv_tol_grad=MAIN_GRADIENT_TRIGGER;mf.max_cycle=80
        state["SCF_kernel_calls"]=call
        attempt={"kernel_call_1based":call,"maximum_cycles":80,"started_at_utc":utc(),
            "start_density_fingerprint":fingerprint(start),
            "initial_guess_policy":"DECLARED_POINT_INITIAL_DENSITY" if call==1 else "IMMEDIATELY_PRECEDING_ACTUAL_FINAL_DENSITY",
            "main_energy_tolerance_hartree":ENERGY_TOLERANCE,"main_orbital_gradient_trigger":MAIN_GRADIENT_TRIGGER,
            "final_energy_tolerance_hartree":ENERGY_TOLERANCE,"final_orbital_gradient_tolerance":GRADIENT_TOLERANCE,
            "qualified":False,"status":"RUNNING"}
        state["SCF_polishing_attempts"].append(attempt)
        emit("SCF_polishing_call_started",attempt)
        cpu=time.process_time();wall=time.perf_counter()
        try:
            state["SCF_performed"]=True;mf.kernel(dm0=np.array(start,copy=True))
            require(pair and math.isfinite(float(mf.e_tot)),"Actual complete native endpoint required")
            audit,arrays=auditor(mf,pair,checks)
            current=np.array(mf.make_rdm1(),copy=True)
            require(np.array_equal(current,arrays["current-density"]),"Actual endpoint density must seed continuation")
            attempt.update(endpoint_audit=audit,native_kernel_converged=bool(mf.converged),
                native_kernel_energy_hartree=float(mf.e_tot),
                actual_final_density_fingerprint=fingerprint(current),
                final_check=pair["check"],main_check=pair.get("main_check"),
                qualified=bool(audit["qualified"]),status="QUALIFIED" if audit["qualified"] else "UNQUALIFIED")
            emit("SCF_polishing_call_completed",attempt)
            if audit["qualified"]:return audit,arrays
            start=np.array(current,copy=True)
        except BaseException as exc:
            attempt.update(status="EXCEPTION",error={"type":type(exc).__name__,"message":str(exc)})
            raise
        finally:
            attempt.update(ended_at_utc=utc(),actual_CPU_seconds=time.process_time()-cpu,
                actual_wall_seconds=time.perf_counter()-wall)
    return audit,arrays


def control_comparison(center_gradient,energies):
    require(len(center_gradient)>0 and all(len(row)==3 for row in center_gradient),"Full finite center gradient required")
    require(all(math.isfinite(value) for row in center_gradient for value in row),"Finite true gradient required")
    component=center_gradient[0][0];rows=[]
    for step in CONTROL_STEPS:
        plus,minus=energies[step]
        require(all(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) for value in (plus,minus)),"Two actual finite energies per step required")
        derivative=math.fsum([plus,-minus])/(2*step)
        rows.append({"step_bohr":step,"Eplus_hartree":plus,"Eminus_hartree":minus,
            "central_energy_first_derivative_hartree_per_bohr":derivative,
            "true_center_gradient_component_hartree_per_bohr":component,
            "absolute_gradient_FD_error_hartree_per_bohr":abs(derivative-component)})
    total=[math.fsum(row[axis] for row in center_gradient) for axis in range(3)]
    totalnorm=math.sqrt(math.fsum(value*value for value in total))
    halfdifference=abs(rows[0]["central_energy_first_derivative_hartree_per_bohr"]-rows[1]["central_energy_first_derivative_hartree_per_bohr"])
    passed=all(row["absolute_gradient_FD_error_hartree_per_bohr"]<=GRADIENT_FD_TOLERANCE for row in rows)
    passed=passed and halfdifference<=HALVING_TOLERANCE and totalnorm<=TOTAL_GRADIENT_TOLERANCE and abs(component)>1e-8
    return {"comparisons":rows,"halving_derivative_abs_difference_hartree_per_bohr":halfdifference,
        "sum_atomic_gradient_vector_hartree_per_bohr":total,"sum_atomic_gradient_vector_norm_hartree_per_bohr":totalnorm,
        "force_balance_scope":"TRANSLATION_BALANCE_NOT_ATOMIC_STATIONARITY_OR_MINIMUM",
        "gradient_FD_tolerance":GRADIENT_FD_TOLERANCE,"halving_tolerance":HALVING_TOLERANCE,
        "total_gradient_vector_tolerance":TOTAL_GRADIENT_TOLERANCE,"nontrivial_component_threshold":1e-8,"passed":bool(passed)}


def validate_live_phase(args,guard,supervision):
    ledger=guard.read(guard.root_path(supervision["proof"]["ledger_path"]));guard.validate_ledger(ledger,allow_running=True)
    running=[row for row in ledger["phases"] if row.get("status")=="RUNNING"]
    require(len(running)==1 and running[0]["phase"]==supervision["proof"]["phase"],"Exactly this actual guarded phase required")
    command=running[0]["command"]
    require(Path(__file__).resolve() in [Path(token).resolve() for token in command if isinstance(token,str) and token.endswith(".py")]
            and "--output" in command and Path(command[command.index("--output")+1]).resolve()==args.output.resolve(),"Live guardian does not bind new worker/output")


def grid_control_release(args,guard):
    m=frozen_worker();record={"path":str(args.grid_control_result.resolve()),"sha256":args.grid_control_result_sha256}
    path=m.checked(record);value=m.read(path)
    require(value.get("schema")==SCHEMA and value.get("source_worker")==ref(__file__)
            and value.get("status")==CONTROL_SUCCESS and value.get("stage")=="controls", "Actual new moving-grid control producer required")
    closure=guard.verify_closed_phase(value,expected_worker_path=Path(__file__),expected_worker_sha256=sha(__file__),
        expected_result_schema=SCHEMA,allowed_result_statuses=(CONTROL_SUCCESS,),expected_output=path.parent)
    for case,atoms in (("water",3),("ethanol",9)):
        result=value["controls"][case]
        gradient=m.old().checked_control_array(result["true_gradient"],(atoms,3))
        energies={float(row["step_bohr"]):(row["Eplus_hartree"],row["Eminus_hartree"]) for row in result["comparison"]["comparisons"]}
        replay=control_comparison(gradient,energies)
        require(replay==result["comparison"] and replay["passed"],"Raw moving-grid controls do not pass fixed release gates")
        expected=[case+"-center"]+[case+"-h%.3f-%s"%(step,label) for step in CONTROL_STEPS for label in ("plus","minus")]
        require(result["point_names"]==expected and len(set(result["point_names"]))==5,"Five unique prescribed control geometries required")
        actual={}
        for name in result["point_names"]:
            point=m.read(m.checked(value["points"][name]["result"]))
            require(point.get("schema")==POINT_SCHEMA and point.get("source_worker")==ref(__file__)
                    and point.get("SCF_performed") is True and point["rebuilt_quadrature"]["grid_level"]==1
                    and point["rebuilt_quadrature"]["original_fixed_grid_coordinates_or_weights_attached"] is False,
                    "Actual fresh grid1 point identity required")
            m.verify_certified_point(point)
            require(value["points"][name].get("energy_hartree")==point["energy_hartree"],"Control parent/child energy differs")
            actual[name]=point["energy_hartree"]
        for row in result["comparison"]["comparisons"]:
            for label,key in (("plus","Eplus_hartree"),("minus","Eminus_hartree")):
                require(actual[case+"-h%.3f-%s"%(row["step_bohr"],label)]==row[key],"FD comparison energy differs from actual certified point")
    return {"result":record,"closure":closure,"raw_control_comparisons_replayed":True}


def compute_point(name, dm0, mol, original, binding, context, *, default_incremental=False,
                  operator_probe_reference=None,displacement=None):
    """At most two bounded native DIIS kernels; preserve all failed audits."""
    args,out,w,base,np,gen_grid,rks,helper,frozen,emit,check_budget=context
    pointout=out/name;pointout.mkdir();local=out/"local-only"/name;local.mkdir(parents=True)
    state={"schema":POINT_SCHEMA,"status":"PARTIAL","source_worker":ref(__file__),"name":name,
        "energy_completed":False,"SCF_performed":False,"SCF_endpoint_certified":False,
        "atoms":mol.natm,"basis_functions":mol.nao_nr(),"electrons":mol.nelectron,
        "maximum_cycles":args.max_cycles,"maximum_SCF_kernel_calls":MAX_SCF_KERNEL_CALLS,"SCF_kernel_calls":0,
        "main_orbital_gradient_trigger":MAIN_GRADIENT_TRIGGER,
        "final_energy_tolerance_hartree":ENERGY_TOLERANCE,"final_orbital_gradient_tolerance":GRADIENT_TOLERANCE,
        "native_direct_driver_threads":1,"worker_threads":2,
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
    settings=dict(binding["method_settings"]);settings.update(threads=2,native_direct_driver_threads=1,grid_level=args.grid_level,
        scf_conv_tol_hartree=ENERGY_TOLERANCE,scf_max_cycle=args.max_cycles)
    mf=w.make_scf(mol,settings,local/"rolling.chk");mf.conv_tol_grad=MAIN_GRADIENT_TRIGGER
    fixed=attach_rebuilt_quadrature(mf,original,args.grid_level,w,base,np,gen_grid)
    if not default_incremental:
        mf.__class__=helper.full_native_veff_type(type(mf),rks.get_veff,emit=emit)
    state.update(settings=settings,rebuilt_quadrature=fixed)
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
        emit("rolling_checkpoint_committed",{"point":name,"kernel_call_1based":state["SCF_kernel_calls"],
            "cycle_0based":int(env.get("cycle",-1)),"bytes":target.stat().st_size})
        return self
    mf.dump_chk=types.MethodType(checkpoint,mf)
    def check(env):
        row=frozen.polish_convergence_check(env,main_energy_tolerance=ENERGY_TOLERANCE,
            main_orbital_gradient_trigger=MAIN_GRADIENT_TRIGGER,final_energy_tolerance=ENERGY_TOLERANCE,
            final_orbital_gradient_tolerance=GRADIENT_TOLERANCE)
        row.update(kernel_call_1based=state["SCF_kernel_calls"],cycle_0based=int(env.get("cycle",-1)),energy_hartree=float(env["e_tot"]))
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
        audit,arrays=bounded_kernel_sequence(mf,dm0,np=np,pair=pair,checks=state["strict_SCF_convergence_checks"],
            emit=emit,check_budget=check_budget,state=state,fingerprint=base.array_fingerprint,
            auditor=lambda obj,pair,checks:endpoint_audit(obj,pair,checks,np=np,frozen=frozen))
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
        verify_rebuilt_quadrature(mf,fixed,w,base)
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
    validate_stage_args(args);m=frozen_worker();guard=load(GUARD,GUARD_SHA,"rebuilt_actual_guard")
    supervision=guard.verify_resource_supervision(expected_threads=2);validate_live_phase(args,guard,supervision)
    inputs=m.verify_control_release(args,guard)
    zero_input=m.dependency(args,guard,certified=True) if args.stage=="spinosin" else None
    gridrelease=grid_control_release(args,guard) if args.stage=="spinosin" else None
    old=m.old();analyzer=load(old.ANALYZER,old.ANALYZER_SHA,"rebuilt_original_analyzer_metadata")
    with old.closed_main(args,analyzer) as mainref:
        return execute_closed(args,m,guard,supervision,inputs,zero_input,gridrelease,mainref)


def execute_closed(args,m,guard,supervision,inputs,zero_input,gridrelease,mainref):
    import numpy as np
    import h5py
    from pyscf import lib
    from pyscf.dft import gen_grid,rks
    from pyscf.scf import _vhf,addons
    lib.num_threads(2);old=m.old()
    w=load(ROOT/"scripts/pyscf_molecular_state_worker_20261104_v5.py",
        "be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8","rebuilt_original_snapshot_restorer")
    base=w.frozen_base();helper=load(HELPERS,HELPERS_SHA,"rebuilt_full_native_helper");out=m.fresh(args.output)
    started=time.process_time();wall=time.perf_counter()
    state={"schema":SCHEMA,"status":"PARTIAL","stage":args.stage,"source_worker":ref(__file__),
        "resource_supervision":supervision,"output_directory":str(out),"input_fixed_grid_worker":ref(FROZEN),
        "input_validation_scope":"READ_ONLY_F93_PRODUCER_IS_INPUT_NOT_NEW_RESULT_PRODUCER",
        "fixed_grid_control_inputs":inputs,"moving_grid_control_release":gridrelease,
        "closed_main_pipeline_state":mainref,"grid_level":args.grid_level,"points":{},"completed_points":[],
        "energies_hartree":{},"controls":{},"stable_MF_state_unchanged":False,
        "nuclear_stationary_minimum_verified":False,"full_Hessian_performed":False,"frequencies_performed":False,
        "IR_performed":False,"optimization_performed":False,"grid_definition":"FRESH_REBUILT_MOVING_ATOM_CENTERED",
        "native_direct_driver_threads":1,"worker_threads":2,"storage_policy":STORAGE_POLICY,
        "maximum_SCF_kernel_calls_per_point":MAX_SCF_KERNEL_CALLS,"main_orbital_gradient_trigger":MAIN_GRADIENT_TRIGGER,
        "final_energy_tolerance_hartree":ENERGY_TOLERANCE,"final_orbital_gradient_tolerance":GRADIENT_TOLERANCE,
        "all_intermediate_raw_matrices_archived":False,"new_results_initial_status":"NOT_RUN"}
    def emit(event,detail=None):
        row={"event":event,"utc":utc(),"CPU_seconds":time.process_time()-started,"wall_seconds":time.perf_counter()-wall,**(detail or {})}
        with (out/"events.jsonl").open("a") as handle:handle.write(json.dumps(row,allow_nan=False)+"\n")
        save(out/"progress.json",{**state,"latest_event":row})
    def budget():
        if time.process_time()>=supervision["proof"]["actual_cpu_stop_threshold_seconds"]:raise base.BudgetStop("ActualCPU cleanup threshold")
        require(datetime.now(timezone.utc)<datetime.fromisoformat(m.read(m.LEDGER)["effective_run_deadline_at_utc"]),"Authorized deadline reached")
    def stop(signum,frame):raise base.BudgetStop("Resource signal "+str(signum))
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGXCPU)}
    native=base.NativeDirectDriverThreadGuard(_vhf,lib.with_omp_threads,lib.num_threads,native_threads=1,
        emit=lambda detail:emit("native_direct_driver_signature",detail))
    restorelog=(out/"restoration.log").open("x",buffering=1);gradientlog=(out/"nuclear-gradients.log").open("x",buffering=1)
    originals=[];planned=10 if args.stage=="controls" else 3 if args.with_displacements else 1;attempted=0
    def preflight(remaining):
        import shutil
        free=shutil.disk_usage(out).free;required=64*1024**2+remaining*POINT_STORAGE_RESERVE_BYTES
        require(free>=required,"Insufficient persistent space for all remaining compact points plus64MiB floor")
        emit("compact_storage_preflight",{"free_bytes":free,"required_bytes":required,"remaining_points":remaining})
    def restore(source_result):
        source=m.read(m.checked(source_result));original,binding=w.restore_snapshot(Path(source["snapshot_directory"]),
            expected_manifest_sha256=source["snapshot_manifest_sha256"],
            expected_source_sha256="be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8",stdout=restorelog)
        require(binding==source["source_state_binding"],"Immutable original source binding differs")
        originals.append((original,w.molecule_physical_fingerprint(original.mol),old.original_mf_fingerprints(original,base)))
        return source,original,binding
    def checkpoint_density(point,original):
        with h5py.File(m.checked(point["checkpoint"]),"r") as handle:
            payload={"molecule":handle["mol"][()],"energy_hartree":handle["scf/e_tot"][()],
                **{key:handle["scf/"+key][()] for key in ("mo_coeff","mo_energy","mo_occ")}}
        arrays,metadata=helper.validate_checkpoint_restart_payload(payload,original.mol.__dict__,np=np,
            atoms=original.mol.natm,basis_functions=original.mol.nao_nr(),electrons=original.mol.nelectron)
        overlap=original.get_ovlp();trace=float(np.einsum("ij,ji->",arrays["density"],overlap))
        orth=float(np.max(np.abs(arrays["mo_coeff"].T@overlap@arrays["mo_coeff"]-np.eye(original.mol.nao_nr()))))
        require(abs(trace-original.mol.nelectron)<1e-7 and orth<1e-7,"Input checkpoint fullAO metric check failed")
        metadata.update(density_overlap_electron_trace=trace,MO_overlap_orthonormality_max_abs_error=orth)
        return original.make_rdm1(arrays["mo_coeff"],arrays["mo_occ"]),metadata
    def point(name,dm,mol,original,binding,displacement=None):
        nonlocal attempted
        preflight(planned-attempted);attempted+=1;budget()
        context=(args,out,w,base,np,gen_grid,rks,helper,old,emit,budget)
        value,mf=compute_point(name,dm,mol,original,binding,context,displacement=displacement)
        state["points"][name]={"result":ref(out/name/"native_result.json"),"status":value["status"],"energy_completed":value["energy_completed"]}
        require(value["SCF_endpoint_certified"] is True,"Required new-grid SCF endpoint uncertified; no automatic retry")
        state["points"][name]["energy_hartree"]=value["energy_hartree"];state["completed_points"].append(name)
        state["energies_hartree"][name]=value["energy_hartree"]
        mf.stdout=mf.mol.stdout=gradientlog
        return value,mf
    def gradients(mf,name):
        result={}
        for response in (True,False):
            budget();gradient=mf.nuc_grad_method();gradient.grid_response=response;array=gradient.kernel()
            require(array.shape==(mf.mol.natm,3) and np.isfinite(array).all(),"Finite full nuclear gradient required")
            key="true" if response else "false";base.atomic_npy(out/(name+"-grid-response-"+key+"-gradient.npy"),array)
            total=np.sum(array,axis=0)
            result[key]={"array":ref(out/(name+"-grid-response-"+key+"-gradient.npy")),
                "grid_response":response,"definition":"TOTAL_MOVING_GRID_DERIVATIVE" if response else "OMITTED_GRID_RESPONSE_DERIVATIVE",
                "nuclear_gradient_max_abs_hartree_per_bohr":float(np.max(np.abs(array))),
                "nuclear_gradient_RMS_hartree_per_bohr":float(np.sqrt(np.mean(array*array))),
                "sum_atomic_gradient_vector_hartree_per_bohr":total.tolist(),"sum_atomic_gradient_vector_norm_hartree_per_bohr":float(np.linalg.norm(total))}
        return result
    try:
        preflight(planned);native.install();budget()
        if args.stage=="controls":
            for case in ("water","ethanol"):
                old_result=m.read(m.checked(inputs[case]["result"]))
                source,original,binding=restore(old_result["source_result"])
                oldpoint=m.read(m.checked(old_result["points"]["full-native"]["result"]))
                initial,metadata=checkpoint_density(oldpoint,original)
                xyz=original.mol.atom_coords(unit="Bohr").copy();xyz[0,0]+=CONTROL_SHIFT_BOHR
                center=original.mol.copy();center.set_geom_(xyz,unit="Bohr",inplace=True)
                name=case+"-center";center_state,center_mf=point(name,addons.project_dm_nr2nr(original.mol,initial,center),center,original,binding)
                zero_dm=np.array(center_mf.make_rdm1(),copy=True);zero_fp=base.array_fingerprint(zero_dm)
                gs=gradients(center_mf,name);energies={};names=[name]
                for step in CONTROL_STEPS:
                    signs=[]
                    for sign,label in ((1,"plus"),(-1,"minus")):
                        mol=center.copy()
                        displaced=xyz.copy();displaced[0,0]+=sign*step;mol.set_geom_(displaced,unit="Bohr",inplace=True)
                        pointname=case+"-h%.3f-%s"%(step,label)
                        value,_=point(pointname,addons.project_dm_nr2nr(center_mf.mol,zero_dm,mol),mol,original,binding,
                            {"coordinate_index_0based":0,"step_bohr":step,"sign":sign,"relative_to_shifted_center":True})
                        signs.append(value["energy_hartree"]);names.append(pointname)
                    energies[step]=tuple(signs)
                require(base.array_fingerprint(zero_dm)==zero_fp,"Shared new center density changed")
                raw=old.checked_control_array(gs["true"]["array"],(original.mol.natm,3));comparison=control_comparison(raw,energies)
                state["controls"][case]={"input_initial_guess_validation":metadata,"purpose_shift_bohr":CONTROL_SHIFT_BOHR,
                    "coordinate_index_0based":0,"point_names":names,"true_gradient":gs["true"]["array"],
                    "nuclear_gradients":gs,"comparison":comparison}
                require(comparison["passed"],"Moving-grid true-gradient finite-energy control failed")
            state["status"]=CONTROL_SUCCESS
        else:
            prior,priorpoint,priorref=zero_input;source,original,binding=restore(prior["source_result"])
            initial,metadata=checkpoint_density(priorpoint,original)
            state.update(input_fixed_grid_zero_result=priorref,input_zero_initial_guess_validation=metadata,
                snapshot_manifest_sha256=source["snapshot_manifest_sha256"],case="Spinosin")
            zero,zero_mf=point("fresh-zero",initial,original.mol.copy(),original,binding)
            zero_dm=np.array(zero_mf.make_rdm1(),copy=True);zero_fp=base.array_fingerprint(zero_dm)
            state["rebuilt_zero_nuclear_gradients"]=gradients(zero_mf,"fresh-zero")
            state["rebuilt_zero_electronically_certified"]=True
            if args.with_displacements:
                direction_args=argparse.Namespace(source_result=Path(prior["source_result"]["path"]),mode_index=1,
                    coordinate_index=None,mode_array_sha256=args.mode_array_sha256)
                q,hq,analytic,refs=old.direction_and_hessian(direction_args,{"result":source},np)
                base.atomic_npy(out/"mode1-direction.npy",q);values={}
                for sign,label in ((1,"plus"),(-1,"minus")):
                    mol=original.mol.copy();mol.set_geom_(original.mol.atom_coords(unit="Bohr")+sign*SPINOSIN_STEP*q,unit="Bohr",inplace=True)
                    value,_=point(label,addons.project_dm_nr2nr(zero_mf.mol,zero_dm,mol),mol,original,binding,
                        {"mode_index_1based":1,"step_bohr":SPINOSIN_STEP,"sign":sign})
                    values[label]=value["energy_hartree"]
                state["central_energy_analysis"]=old.central_energy_curvature(zero["energy_hartree"],values["plus"],values["minus"],SPINOSIN_STEP)
                state["central_energy_analysis"].update(archived_analytic_directional_curvature_REFERENCE_ONLY=analytic,
                    reference_scope="CROSS_DEFINITION_AND_CROSS_QUADRATURE_NOT_SAME_METHOD_HESSIAN_ERROR",source_arrays=refs)
            require(base.array_fingerprint(zero_dm)==zero_fp,"Both signs must use same fresh grid-specific zero density")
            state["status"]=SPINOSIN_SUCCESS
    except BaseException as exc:
        state["status"]="PARTIAL_RESOURCE_STOP" if isinstance(exc,base.BudgetStop) else "FAILED_REBUILT_GRID_DIAGNOSTIC"
        state["error"]={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
        emit("worker_stopped",{"status":state["status"],"error":state["error"]})
    finally:
        state["stable_MF_state_unchanged"]=True
        for original,fingerprint,payload in originals:
            try:old.ensure_original_unchanged(original,fingerprint,payload,w,base)
            except ValueError as exc:state.update(status="FAILED_ORIGINAL_SOURCE_MUTATED",stable_MF_state_unchanged=False,source_mutation_error=str(exc))
        native.restore();state["native_direct_driver_control"]=native.summary()
        if not state["native_direct_driver_control"]["original_driver_restored"] or state["native_direct_driver_control"]["restoration_failures"]:
            state["status"]="FAILED_NATIVE_DRIVER_RESTORATION"
        for sig,handler in handlers.items():signal.signal(sig,handler)
        restorelog.close();gradientlog.close();state.update(ended_at_utc=utc(),worker_CPU_seconds=time.process_time()-started,
            worker_wall_seconds=time.perf_counter()-wall,energies=dict(state["energies_hartree"]))
        save(out/"result.json",state)
    return state


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute",action="store_true",required=True)
    p.add_argument("--stage",choices=("controls","spinosin"),required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--threads",type=int,default=2);p.add_argument("--max-cycles",type=int,default=80)
    p.add_argument("--grid-level",type=int,default=1);p.add_argument("--with-displacements",action="store_true")
    for label in ("source-zero","water-control","ethanol-control","grid-control"):
        p.add_argument("--"+label+"-result",type=Path);p.add_argument("--"+label+"-result-sha256")
    p.add_argument("--mode-array-sha256")
    p.add_argument("--main-pipeline-state",type=Path,default=ROOT/"reports/pyscf-spinosin-96h-continuation-20261105-v8/global-assemble-recovery-002/state.json")
    p.add_argument("--main-pipeline-state-sha256",default="f2974458d23abeb320db736ce6c4cdb3f0b4d509f707a4dc67fd7d5ceaaa2578")
    return p


def main():
    args=parser().parse_args();value=execute(args)
    print(json.dumps({"schema":value["schema"],"status":value["status"],"output":str(args.output)}))
    return 0 if value["status"] in (CONTROL_SUCCESS,SPINOSIN_SUCCESS) else 2


if __name__=="__main__":raise SystemExit(main())
