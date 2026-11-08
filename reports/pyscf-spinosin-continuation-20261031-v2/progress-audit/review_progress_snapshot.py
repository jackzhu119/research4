"""One read-only snapshot of completed geometry evidence; no quantum calls."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "spinosin-extended-optimization-001"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    result_path = OUT / "progress_snapshot_review.json"
    if result_path.exists():
        raise FileExistsError("Keep this one-time audit snapshot")
    started = datetime.now(timezone.utc).isoformat()
    spec = importlib.util.spec_from_file_location("reviewed_old_geometry_verifier", ROOT / "scripts/prepare_pyscf_pilot_resume_20261031.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source_path = ROOT / "reports/pyscf-spinosin-pilot-20261031/structures/spinosin_verified_2d.json"
    source_record = json.loads(source_path.read_text())
    files = sorted((SOURCE / "geometry_steps").glob("step_*.json"))
    if not files:
        raise ValueError("No completed metadata: cannot audit a future geometry")
    captured = []
    snapshot_dir = OUT / "completed_geometry_snapshots"
    snapshot_dir.mkdir()
    for metadata in files:
        for suffix in (".json", ".xyz", ".gradient.npy", ".coordinates_angstrom.npy", ".coordinates_bohr.npy"):
            original = metadata.with_suffix(suffix)
            saved = snapshot_dir / original.name
            saved.write_bytes(original.read_bytes())
            captured.append({"source": str(original.relative_to(ROOT)), "snapshot": str(saved.relative_to(OUT)),
                             "sha256": sha(saved), "bytes": saved.stat().st_size})
    # Read each active append-only text stream exactly once; the saved prefix
    # is evidence of this timestamp, not a claim about the final future stream.
    for name in ("pyscf.log", "events.jsonl", "progress.json"):
        (OUT / ("active_" + name + ".snapshot")).write_bytes((SOURCE / name).read_bytes())
    log = (OUT / "active_pyscf.log.snapshot").read_text()
    event_lines = (OUT / "active_events.jsonl.snapshot").read_text().splitlines()
    events = []
    ignored_tail = False
    for index, line in enumerate(event_lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(event_lines) - 1:
                raise
            ignored_tail = True
    markers = list(re.finditer(r"Geometry optimization cycle\s+(\d+)", log))
    chunks = {int(marker.group(1)): log[marker.end():markers[i+1].start() if i+1 < len(markers) else len(log)]
              for i, marker in enumerate(markers)}
    rows = []
    for metadata in files:
        saved = snapshot_dir / metadata.name
        record = json.loads(saved.read_text())
        gradient = np.load(saved.with_suffix(".gradient.npy"), allow_pickle=False)
        coords = np.load(saved.with_suffix(".coordinates_angstrom.npy"), allow_pickle=False)
        bohr = np.load(saved.with_suffix(".coordinates_bohr.npy"), allow_pickle=False)
        symbols, rounded = helper.read_xyz(saved.with_suffix(".xyz"))
        if gradient.shape != (75, 3) or coords.shape != (75, 3) or not np.isfinite(gradient).all() or not np.isfinite(coords).all():
            raise ValueError("Invalid completed coordinate or gradient array")
        assert bohr.shape == (75, 3) and np.isfinite(bohr).all()
        assert np.allclose(coords, bohr * helper.BOHR_TO_ANGSTROM, atol=1e-12, rtol=0)
        assert np.allclose(coords, rounded, atol=5e-12, rtol=0)
        assert sha(saved.with_suffix(".gradient.npy")) == record["gradient_npy_sha256"]
        assert sha(saved.with_suffix(".coordinates_angstrom.npy")) == record["coordinates_angstrom_npy_sha256"]
        assert sha(saved.with_suffix(".coordinates_bohr.npy")) == record["coordinates_bohr_npy_sha256"]
        norms = np.linalg.norm(gradient, axis=1)
        values = {"component_rms_hartree_per_bohr": float(np.sqrt(np.mean(gradient**2))),
                  "component_max_hartree_per_bohr": float(np.abs(gradient).max()),
                  "atom_rms_hartree_per_bohr": float(np.sqrt(np.mean(norms**2))),
                  "atom_max_hartree_per_bohr": float(norms.max()),
                  "gradient_norm_hartree_per_bohr": float(np.linalg.norm(gradient))}
        for key, value in values.items():
            assert math.isclose(value, record[key], abs_tol=1e-13, rel_tol=1e-12), key
        cycle = int(record["cycle"])
        energies = [float(v) for v in re.findall(r"converged SCF energy\s*=\s*([-+0-9.eE]+)", chunks[cycle])]
        assert record["scf_converged"] is True and energies
        assert math.isclose(energies[-1], record["energy_hartree"], abs_tol=5e-9, rel_tol=0)
        identity = helper.verify_coordinates(source_record, symbols, coords)
        rows.append({"cycle": cycle, "energy_hartree": record["energy_hartree"],
                     "paired_log_converged_SCF_energy_hartree": energies[-1],
                     "log_SCF_energy_matches": True, "fullprecision_coordinate_and_gradient_sha_match": True,
                     "recomputed_gradient_statistics": values, "geometry_identity": identity,
                     "stationary_gradient_thresholds_passed": bool(values["atom_rms_hartree_per_bohr"] <= 3e-4 and values["atom_max_hartree_per_bohr"] <= 4.5e-4)})
    releases = [event for event in events if event["event"] == "gradient_started_after_cache_release"]
    timings = [event for event in events if event["event"] == "gradient_kernel_complete"]
    for release in releases:
        assert release["cache_released"] and release["auxmol_identity_preserved"] and not release["with_df_reset_called"]
        assert release["SCF_cache_before_release"]["nbytes"] == 5718709416
        assert release["current_rss_mb_before_release"] > release["current_rss_mb_after_release"]
    for timing in timings:
        assert timing["cache_after_gradient"]["storage"] == "not_built"
        assert timing["gradient_cpu_seconds"] > 0 and timing["gradient_wall_seconds"] > 0
    result = {"status": "PASS_ONE_TIME_PROGRESS_SNAPSHOT_NOT_FINAL_COMPLETION", "started_at_utc": started,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
              "source_phase_directory": str(SOURCE.relative_to(ROOT)), "new_quantum_jobs": 0,
              "active_checkpoint_read": False, "loop_monitoring": False,
              "completed_paired_geometry_points_audited": len(rows), "audited_geometries": rows,
              "cache_release_events_in_snapshot": [{"utc": e["utc"], "SCF_cache_bytes": e["SCF_cache_before_release"]["nbytes"],
                   "rss_mb_before": e["current_rss_mb_before_release"], "rss_mb_after": e["current_rss_mb_after_release"],
                   "rss_mb_drop": e["current_rss_mb_before_release"]-e["current_rss_mb_after_release"]} for e in releases],
              "completed_gradient_timings_in_snapshot": [{"utc": e["utc"], "cpu_seconds": e["gradient_cpu_seconds"],
                   "wall_seconds": e["gradient_wall_seconds"]} for e in timings],
              "incomplete_last_event_line_ignored": ignored_tail,
              "source_records_sha256": {str(source_path.relative_to(ROOT)): sha(source_path),
                                         "scripts/prepare_pyscf_pilot_resume_20261031.py": sha(ROOT / "scripts/prepare_pyscf_pilot_resume_20261031.py")},
              "captured_completed_files": captured,
              "interpretation": ["Every saved geometry audited here is a completed SCF/gradient pair, not a stationary minimum unless thresholds pass.",
                  "The source phase was active; future appended logs and future geometry points are outside this one-time snapshot.",
                  "The checkpoint was never opened; its current trial geometry is not certified by completed step evidence.",
                  "Memory drop establishes that the cache release occurred. Thread count and cache handling were changed together; this is not an isolated causal speed-up ablation.",
                  "No Spinosin frequency or chemical attribution is established by this progress audit."]}
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "points": len(rows), "last_energy_hartree": rows[-1]["energy_hartree"],
                      "last_atom_rms": rows[-1]["recomputed_gradient_statistics"]["atom_rms_hartree_per_bohr"],
                      "cache_release_events": len(releases), "quantum_jobs": 0, "active_checkpoint_read": False}))


if __name__ == "__main__":
    run()
