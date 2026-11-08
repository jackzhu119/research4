"""Compare two completed first-gradient points, never evaluate quantum chemistry.

The containing optimization may still be active. Only immutable completed point
files and an explicitly released, completed events prefix are read; checkpoints
are never read. Tolerances are fixed here before the native4 result is inspected.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BOHR = .52917721092
TOLERANCES = {"coordinates_angstrom": 5e-12, "energy_hartree": 1e-8,
              "gradient_components_hartree_per_bohr": 1e-8,
              "atom_gradient_norms_hartree_per_bohr": 1e-8}
TIMING_OVERHEAD = {"absolute_seconds": 1.0, "fraction_of_outer_delta": .01}
PHYSICS = {"xc": "PBE", "basis": "def2-svp", "grid_level": 1,
    "dispersion": "NONE_EXPLICIT_PBE_PILOT_NOT_D3", "density_fitting": False,
    "auxbasis": None, "density_fit_backend": "SCREENED_DIRECT_RKS_V4",
    "scf_conv_tol_hartree": 1e-8, "scf_max_cycle": 80, "max_memory_mb": 8000,
    "direct_scf": True, "direct_scf_tol": 1e-13,
    "global_four_center_ERI_cache_allowed": False, "charge": 0,
    "spin_2S": 0, "multiplicity": 1}
INCHIKEY = "VGGSULWDCMWZPO-ODEMIOGVSA-N"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (75, 3) or not np.isfinite(values).all():
        raise ValueError("One complete finite75x3 nuclear gradient is required")
    norms = np.linalg.norm(values, axis=1)
    return {"component_rms_hartree_per_bohr": float(np.sqrt(np.mean(values**2))),
        "component_max_hartree_per_bohr": float(np.max(np.abs(values))),
        "atom_rms_hartree_per_bohr": float(np.sqrt(np.mean(norms**2))),
        "atom_max_hartree_per_bohr": float(np.max(norms)),
        "gradient_norm_hartree_per_bohr": float(np.linalg.norm(values))}


def load_point(folder, expected_threads):
    folder = Path(folder).resolve()
    source_hashes = {}
    def record(path):
        source_hashes[str(path)] = sha(path)
        return path
    metadata_path = folder/"geometry_steps/step_0001.json"
    metadata = json.loads(record(metadata_path).read_text())
    source = json.loads(record(folder/"input.json").read_text())
    if metadata.get("cycle") != 1 or metadata.get("scf_converged") is not True:
        raise ValueError("Only completeSCF-converged first points may be compared")
    if source.get("phase") != "spinosin-optimize" or source.get("identity") != "spinosin":
        raise ValueError("Expected the original Spinosin optimization task")
    settings = source.get("settings", {})
    if (any(settings.get(key) != value for key, value in PHYSICS.items())
            or settings.get("threads") != expected_threads
            or settings.get("native_direct_driver_threads", settings.get("threads")) != expected_threads):
        raise ValueError("Physical method or prescribed1vs4thread allocation mismatch")
    provenance = source.get("input_provenance", {})
    if (provenance.get("identity_verified") is not True or provenance.get("stereochemistry_verified") is not True
            or provenance.get("source_inchikey") != INCHIKEY or not provenance.get("source_url")
            or provenance.get("electron_count") != 320):
        raise ValueError("Verified Spinosin identity/stereochemistry provenance required")
    arrays = {}
    for key, suffix in (("coordinates", "coordinates_angstrom"), ("coordinates_bohr", "coordinates_bohr"),
                        ("gradient", "gradient")):
        path = folder/"geometry_steps"/f"step_0001.{suffix}.npy"
        expected_sha = metadata.get(suffix+"_npy_sha256")
        if not expected_sha or sha(record(path)) != expected_sha:
            raise ValueError("Incomplete or mismatched completed-point NPY SHA: " + suffix)
        values = np.load(path, allow_pickle=False)
        if values.shape != (75,3) or values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError("Completed point array must be finite float64(75,3)")
        arrays[key] = values
    if np.max(np.abs(arrays["coordinates"]-arrays["coordinates_bohr"]*BOHR)) > TOLERANCES["coordinates_angstrom"]:
        raise ValueError("Bohr/Angstrom coordinate files disagree")
    precise_path = record(folder/"input_coordinates_angstrom.npy")
    precise = np.load(precise_path, allow_pickle=False)
    if precise.shape != (75,3) or not np.isfinite(precise).all():
        raise ValueError("Finite full-precision source coordinates required")
    expected_sha = source.get("input_coordinates_angstrom_npy_sha256")
    if not expected_sha or sha(precise_path) != expected_sha:
        raise ValueError("Full-precision phase input does not match recorded source SHA")
    if np.max(np.abs(arrays["coordinates"]-precise)) > TOLERANCES["coordinates_angstrom"]:
        raise ValueError("First point differs from actual source input coordinates")
    xyz_path = record(folder/"input.xyz")
    if sha(xyz_path) != source.get("input_xyz_sha256"):
        raise ValueError("Phase input XYZ differs from source record")
    lines = xyz_path.read_text().splitlines(); rows = [line.split() for line in lines[2:] if line.strip()]
    if (not lines or int(lines[0]) != 75 or len(rows) != 75 or any(len(row) != 4 for row in rows)
            or Counter(row[0] for row in rows) != Counter({"C":28,"H":32,"O":15})):
        raise ValueError("Source input atom count/composition mismatch")
    xyz_coordinates = np.asarray([row[1:] for row in rows],float)
    if not np.isfinite(xyz_coordinates).all() or np.max(np.abs(xyz_coordinates-precise)) > TOLERANCES["coordinates_angstrom"]:
        raise ValueError("Source XYZ/fullprecision NPY coordinates disagree")
    gradient_stats = statistics(arrays["gradient"])
    if any(not np.isclose(value, metadata.get(key, np.nan), rtol=1e-11, atol=1e-13) for key,value in gradient_stats.items()):
        raise ValueError("Actual completed gradient disagrees with paired metadata")
    energy = float(metadata.get("energy_hartree", np.nan))
    if not np.isfinite(energy): raise ValueError("Finite pairedSCF energy required")
    return {"folder":folder, "metadata_path":metadata_path, "metadata":metadata, "input":source,
            "symbols":[row[0] for row in rows], "source_coordinates":precise,
            "arrays":arrays, "statistics":gradient_stats, "energy_hartree":energy,
            "source_sha256":source_hashes}


def completed_gradient_prefix(events_path, point):
    """Pair exactly the first started/completed gradient and its cycle1 record."""
    started = completed = None
    lines = []
    # No eager read of the still-active file or of any later point's suffix.
    with Path(events_path).open("rb") as handle:
        for line_number, line in enumerate(handle,1):
            if not line.endswith(b"\n"):
                break  # An append in progress is not a complete event.
            lines.append(line)
            event = json.loads(line)
            kind = event.get("event")
            if kind == "direct_gradient_started":
                if started is not None: raise ValueError("Ambiguous or unpaired gradient before cycle1")
                if event.get("scf_converged") is not True: raise ValueError("Gradient did not start from convergedSCF")
                started = event
            elif kind == "direct_gradient_kernel_complete":
                if started is None or completed is not None: raise ValueError("Unpaired gradient completion")
                completed = event
            elif kind == "geometry_cycle":
                if event.get("cycle") != 1 or started is None or completed is None:
                    raise ValueError("Complete first gradient must precede its cycle1 record")
                if any(event.get(key) != point["metadata"].get(key) for key in
                       ("cycle","scf_converged","coordinates_angstrom_npy_sha256","coordinates_bohr_npy_sha256","gradient_npy_sha256")):
                    raise ValueError("Cycle1 event/step NPY hashes disagree")
                for value in (started.get("SCF_energy_hartree"),event.get("energy_hartree")):
                    if not np.isfinite(value) or abs(value-point["energy_hartree"]) > 1e-12:
                        raise ValueError("Gradient-start/pointSCF energy mismatch")
                for value in (started,completed):
                    if value.get("direct_integral_policy",{}).get("density_fitting") is not False:
                        raise ValueError("Gradient event does not verify the same direct method")
                clocks = [datetime.fromisoformat(value["utc"]) for value in (started,completed,event)]
                if any(value.tzinfo is None for value in clocks) or not clocks[0] <= clocks[1] <= clocks[2]:
                    raise ValueError("Gradient event chronology is invalid")
                cpu, wall = float(completed["gradient_cpu_seconds"]), float(completed["gradient_wall_seconds"])
                delta_cpu = completed["process_cpu_seconds"]-started["process_cpu_seconds"]
                delta_wall = completed["wall_seconds"]-started["wall_seconds"]
                if (not all(np.isfinite(x) and x > 0 for x in (cpu,wall,delta_cpu,delta_wall))
                        or cpu > delta_cpu+1e-6 or wall > delta_wall+1e-6):
                    raise ValueError("Invalid completed gradient timing")
                cpu_overhead, wall_overhead = float(delta_cpu-cpu), float(delta_wall-wall)
                if any(overhead > max(TIMING_OVERHEAD["absolute_seconds"],TIMING_OVERHEAD["fraction_of_outer_delta"]*outer)
                       for overhead,outer in ((cpu_overhead,delta_cpu),(wall_overhead,delta_wall))):
                    raise ValueError("Timing overhead is inconsistent; no valid benchmark may be reported")
                prefix=b"".join(lines)
                return {"prefix":prefix,"timing":{"gradient_cpu_seconds":cpu,"gradient_wall_seconds":wall,
                    "outer_event_CPU_delta_seconds":float(delta_cpu),"outer_event_wall_delta_seconds":float(delta_wall),
                    "outer_minus_inner_CPU_seconds":cpu_overhead,"outer_minus_inner_wall_seconds":wall_overhead,
                    "gradient_started_at_utc":started["utc"],"gradient_completed_at_utc":completed["utc"],
                    "cycle1_completed_at_utc":event["utc"],"completed_prefix_line_count":line_number,
                    "completed_events_prefix_sha256":hashlib.sha256(prefix).hexdigest(),
                    "completed_events_prefix_bytes":len(prefix),"events_after_completed_cycle1_not_read":True}}
    raise ValueError("Complete started/kernel-complete/cycle1 events prefix not available")


def compare_points(old, new):
    if old["symbols"] != new["symbols"]: raise ValueError("Atom order differs between runs")
    source_difference=float(np.max(np.abs(old["source_coordinates"]-new["source_coordinates"])))
    coordinate_difference=float(np.max(np.abs(old["arrays"]["coordinates"]-new["arrays"]["coordinates"])))
    difference=new["arrays"]["gradient"]-old["arrays"]["gradient"]
    old_norm=np.linalg.norm(old["arrays"]["gradient"],axis=1)
    new_norm=np.linalg.norm(new["arrays"]["gradient"],axis=1)
    metrics={"source_coordinate_max_difference_angstrom":source_difference,
        "point_coordinate_max_difference_angstrom":coordinate_difference,
        "energy_difference_hartree":abs(new["energy_hartree"]-old["energy_hartree"]),
        "gradient_component_max_difference_hartree_per_bohr":float(np.max(np.abs(difference))),
        "gradient_component_RMS_difference_hartree_per_bohr":float(np.sqrt(np.mean(difference**2))),
        "atom_gradient_norm_max_difference_hartree_per_bohr":float(np.max(np.abs(new_norm-old_norm)))}
    checks={"same_source_coordinates":source_difference<=TOLERANCES["coordinates_angstrom"],
        "same_point_coordinates":coordinate_difference<=TOLERANCES["coordinates_angstrom"],
        "same_energy":metrics["energy_difference_hartree"]<=TOLERANCES["energy_hartree"],
        "all225_gradient_components":metrics["gradient_component_max_difference_hartree_per_bohr"]<=TOLERANCES["gradient_components_hartree_per_bohr"],
        "all75_atom_gradient_norms":metrics["atom_gradient_norm_max_difference_hartree_per_bohr"]<=TOLERANCES["atom_gradient_norms_hartree_per_bohr"]}
    return {"metrics":metrics,"checks":checks,"numerical_equivalence_passed":bool(all(checks.values()))}


def run(args):
    out=Path(args.output).resolve()
    if out.exists(): raise FileExistsError("Use a fresh comparison output; preserve previous audits")
    release_path=Path(args.release).resolve(); release=json.loads(release_path.read_text())
    if release.get("allow_first_gradient_comparison") is not True or release.get("comparator_sha256") != sha(__file__):
        raise ValueError("Explicit dated release must bind this frozen comparator source")
    released=datetime.fromisoformat(release.get("created_at_utc",""))
    if released.tzinfo is None or released > datetime.now(timezone.utc): raise ValueError("Valid completed-point release time required")
    old=load_point(args.old_output,1); new=load_point(args.new_output,4)
    for name,point in (("old",old),("new",new)):
        if release.get(name+"_step_json_sha256") != sha(point["metadata_path"]):
            raise ValueError("Released completed-point JSON SHA does not match: "+name)
    pairs={name:completed_gradient_prefix(point["folder"]/"events.jsonl",point) for name,point in (("old",old),("new",new))}
    if any(released < datetime.fromisoformat(pair["timing"]["cycle1_completed_at_utc"]) for pair in pairs.values()):
        raise ValueError("Release precedes a completed comparison point")
    comparison=compare_points(old,new)
    for point in (old,new):
        for path,expected in point["source_sha256"].items():
            if sha(path)!=expected: raise ValueError("Completed source point/input changed during read")
    for name,point in (("old",old),("new",new)):
        with (point["folder"]/"events.jsonl").open("rb") as handle:
            unchanged = handle.read(len(pairs[name]["prefix"])) == pairs[name]["prefix"]
        if not unchanged:
            raise ValueError("Completed events prefix changed during read")
    out.mkdir(parents=True)
    old_cpu,old_wall=pairs["old"]["timing"]["gradient_cpu_seconds"],pairs["old"]["timing"]["gradient_wall_seconds"]
    new_cpu,new_wall=pairs["new"]["timing"]["gradient_cpu_seconds"],pairs["new"]["timing"]["gradient_wall_seconds"]
    timing={"old_native1":pairs["old"]["timing"],"new_native4":pairs["new"]["timing"],
        "observed_wall_ratio_native1_over_native4":old_wall/new_wall,
        "observed_CPU_ratio_native1_over_native4":old_cpu/new_cpu,
        "observed_wall_time_reduction_percent":100*(1-new_wall/old_wall),
        "not_a_machine_independent_benchmark":True,"scheduler_and_concurrent_tests_can_affect_observed_times":True,
        "predeclared_maximum_timer_overhead_policy":TIMING_OVERHEAD,
        "complete_gradient_kernel_includes_XC_and_J_not_J_only":True,
        "timing_scope":"Only each completed native gradient kernel; excludesSCF and full optimization"}
    for name,pair in pairs.items(): (out/(name+"_completed_events_prefix.jsonl")).write_bytes(pair["prefix"])
    with (out/"all75_atom_gradient_comparison.csv").open("w",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["atom_index1","element","old_gx","old_gy","old_gz","new_gx","new_gy","new_gz","delta_gx","delta_gy","delta_gz","old_atom_norm","new_atom_norm","delta_atom_norm"])
        for i,(symbol,a,b) in enumerate(zip(old["symbols"],old["arrays"]["gradient"],new["arrays"]["gradient"]),1):
            writer.writerow([i,symbol,*a,*b,*(b-a),np.linalg.norm(a),np.linalg.norm(b),np.linalg.norm(b)-np.linalg.norm(a)])
    result={"status":"PASS" if comparison["numerical_equivalence_passed"] else "NOT_EQUIVALENT",
        "created_at_utc":utc(),"comparator_sha256":sha(__file__),"release_sha256":sha(release_path),
        "tolerances_fixed_in_source_before_native4_result":TOLERANCES,**comparison,"timing":timing,
        "same_input_geometry_not_new_conformers":bool(comparison["checks"]["same_source_coordinates"] and comparison["checks"]["same_point_coordinates"]),
        "no_conformer_generation_performed_by_comparator":True,"completed_first_points_only_not_full_optimization_claim":True,
        "SCF_or_gradient_or_integral_or_Hessian_evaluations":0,"checkpoint_reads":0,
        "old":{"folder":str(old["folder"]),"energy_hartree":old["energy_hartree"],"gradient_statistics":old["statistics"],"source_sha256":old["source_sha256"]},
        "new":{"folder":str(new["folder"]),"energy_hartree":new["energy_hartree"],"gradient_statistics":new["statistics"],"source_sha256":new["source_sha256"]}}
    (out/"comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    (out/"README.md").write_text("# First-gradient native1/native4 audit\n\nStatus: "+result["status"]+". Fixed thresholds and all225 gradient components/75 atom norms are recorded in comparison.json and CSV.\n\nOnly completed first points are compared. Treat them as matching the same geometry only when both coordinate checks pass; no new conformers are generated here. These checks do not establish completed optimization/frequency results. Observed full-gradient kernel wall ratios include XC and J, are affected by scheduling and concurrent tests, and are not a hardware-independent benchmark or a forecast of Hessian speed. No quantum computation or checkpoint read occurs in this audit.\n")
    (out/"files_sha256.json").write_text(json.dumps({"created_at_utc":utc(),"files":[{"path":str(path.relative_to(out)),"sha256":sha(path),"bytes":path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file()]},indent=2)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("old-output","new-output","release","output"): parser.add_argument("--"+name,type=Path,required=True)
    result=run(parser.parse_args());print(json.dumps({key:result[key] for key in ("status","metrics","checks","timing")},indent=2))
    sys.exit(0 if result["numerical_equivalence_passed"] else 2)


if __name__=="__main__":main()
