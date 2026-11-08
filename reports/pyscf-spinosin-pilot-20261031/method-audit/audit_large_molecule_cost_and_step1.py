"""Additive basis/storage sizing and saved step-1 review; no quantum jobs.

Only basis-library metadata and immutable completed geometry-step arrays are
read. No SCF, integral construction, optimization, gradient or Hessian is run.
The recorded live progress snapshot is evidence of state at that instant only.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".runtime/quantum-upgrade-site"))
from pyscf import __version__ as pyscf_version, gto  # noqa: E402

HERE = Path(__file__).parent
PILOT = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def count_spherical_functions(basis_name, element):
    shells = gto.basis.load(basis_name, element)
    counts = []
    for shell in shells:
        # These all-electron segmented bases use [l, primitive, ...], and each
        # primitive is [exponent, coefficient_1, ... coefficient_ncontracted].
        assert isinstance(shell[1], (list, tuple))
        primitive_widths = {len(primitive) for primitive in shell[1:]}
        assert len(primitive_widths) == 1
        contracted_shells = len(shell[1]) - 1
        counts.append((2 * shell[0] + 1) * contracted_shells)
    return sum(counts)


def main():
    preflight = HERE / "files_sha256.preflight.json"
    assert sha(preflight) == "4daeb6ba936205af607ff6eeca16bfc92b996df3bd7eefd9758dd81735df3d58"
    cost_path = HERE / "large_molecule_cached_df_cost_supplement.json"
    step_path = HERE / "spinosin_completed_step1_review.json"
    snapshot_path = HERE / "spinosin_progress_snapshot_cost_supplement.json"
    index_path = HERE / "files_sha256.cost-supplement.json"
    assert not any(path.exists() for path in (cost_path, step_path, snapshot_path, index_path))

    orbital = {element: count_spherical_functions("def2-svp", element)
               for element in ("C", "H", "O")}
    auxiliary = {element: count_spherical_functions("def2-universal-jfit", element)
                 for element in ("C", "H", "O")}
    assert orbital == {"C": 14, "H": 5, "O": 14}
    assert auxiliary == {"C": 49, "H": 11, "O": 49}
    compositions = {
        "Spinosin": {"C": 28, "H": 32, "O": 15},
        "Jujuboside A requested-formula sizing only": {"C": 58, "H": 94, "O": 26},
        "Jujuboside B requested-formula sizing only": {"C": 52, "H": 84, "O": 21},
    }
    sizes = {}
    for name, composition in compositions.items():
        natoms = sum(composition.values())
        nao = sum(orbital[element] * count for element, count in composition.items())
        naux = sum(auxiliary[element] * count for element, count in composition.items())
        pairs = nao * (nao + 1) // 2
        tensor_bytes = naux * pairs * np.dtype("float64").itemsize
        metric_bytes = naux * naux * np.dtype("float64").itemsize
        sizes[name] = {
            "composition_used_for_arithmetic": composition,
            "atoms": natoms,
            "expected_nonlinear_vibrations_not_computed": 3 * natoms - 6,
            "spherical_orbital_AO": nao,
            "spherical_auxiliary_AO": naux,
            "packed_orbital_pairs": pairs,
            "full_rank_float64_packed_cached_DF_tensor_bytes": tensor_bytes,
            "same_decimal_GB": tensor_bytes / 10**9,
            "same_binary_GiB": tensor_bytes / 1024**3,
            "auxiliary_metric_float64_bytes": metric_bytes,
            "auxiliary_metric_binary_GiB": metric_bytes / 1024**3,
            "cached_tensor_alone_exceeds_16GiB_physical_memory": tensor_bytes > 16 * 1024**3,
            "cached_tensor_alone_exceeds_initial_9_6GB_free_disk": tensor_bytes > 9.6 * 10**9,
            "identity_statement": (
                "Source and generated conformer verified separately; this is a sizing check."
                if name == "Spinosin" else
                "Formula-based sizing only; does not resolve source-entry, identity or stereochemistry ambiguities."
            ),
        }

    live_progress = PILOT / "spinosin-cached-optimization/progress.json"
    # Byte-copy one atomic-written progress record. Its SHA refers to this
    # snapshot, since the worker may legitimately update the original later.
    snapshot_bytes = live_progress.read_bytes()
    progress = json.loads(snapshot_bytes)
    with snapshot_path.open("xb") as handle:
        handle.write(snapshot_bytes)
    actual_cache = progress["density_fit_cache"]
    spinosin = sizes["Spinosin"]
    assert actual_cache["shape"] == [spinosin["spherical_auxiliary_AO"],
                                     spinosin["packed_orbital_pairs"]]
    assert actual_cache["nbytes"] == spinosin["full_rank_float64_packed_cached_DF_tensor_bytes"]
    cost = {
        "status": "DONE_READ_ONLY_METADATA_SIZING_NOT_A_QUANTUM_CALCULATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "quantum_and_integral_jobs_started_by_this_audit": 0,
        "pyscf_version": pyscf_version,
        "script_sha256": sha(__file__),
        "orbital_basis": "def2-SVP spherical all-electron",
        "orbital_AO_per_element_actual_library": orbital,
        "auxiliary_basis": "def2-universal-jfit spherical",
        "auxiliary_AO_per_element_actual_library": auxiliary,
        "molecules": sizes,
        "measured_Spinosin_worker_cache_metadata_snapshot": {
            "path": str(snapshot_path.relative_to(ROOT)),
            "sha256": sha(snapshot_path),
            "record_utc": progress.get("utc"),
            "worker_status_at_snapshot": progress.get("status"),
            "last_completed_geometry_cycle_at_snapshot": progress.get("last_geometry_cycle"),
            "density_fit_cache": actual_cache,
            "matches_metadata_size_arithmetic": True,
        },
        "interpretation": [
            "Exact dimension arithmetic for a full-rank float64 packed cached three-center DF tensor, not a full-method minimum memory requirement.",
            "Transient integral, eigensolver, gradient, Hessian and Python process memory are additional; storage sizes are not peak RAM estimates.",
            "The audited direct-J path avoids the global tensor but was slower in the actual Spinosin pilot; it is not declared impossible by these sizes.",
            "Requested-formula Jujuboside A/B cached tensors alone exceed this cloud's memory and initial free disk; their full DFT frequency calculations remain NOT_RUN.",
            "Source-entry ambiguity for Jujuboside A is not resolved by matching a formula or dimension.",
            "Runtime cannot be inferred reliably from AO counts, cached storage or the water smoke test.",
        ],
    }
    write_new(cost_path, cost)

    phase = PILOT / "spinosin-cached-optimization"
    step_files = {"json": phase / "geometry_steps/step_0001.json",
                  "xyz": phase / "geometry_steps/step_0001.xyz",
                  "gradient": phase / "geometry_steps/step_0001.gradient.npy",
                  "input": phase / "input.json",
                  "worker": ROOT / "scripts/pyscf_pilot_cached_df_worker_20261031.py"}
    record = json.loads(step_files["json"].read_text())
    inp = json.loads(step_files["input"].read_text())
    gradient = np.load(step_files["gradient"], allow_pickle=False)
    xyz_lines = step_files["xyz"].read_text().splitlines()
    composition = Counter(line.split()[0] for line in xyz_lines[2:] if line.strip())
    assert int(xyz_lines[0]) == 75 and composition == Counter({"C": 28, "H": 32, "O": 15})
    assert gradient.shape == (75, 3) and np.all(np.isfinite(gradient))
    atom_norms = np.linalg.norm(gradient, axis=1)
    recomputed = {
        "component_rms_hartree_per_bohr": float(np.sqrt(np.mean(gradient**2))),
        "component_max_hartree_per_bohr": float(np.max(np.abs(gradient))),
        "atom_rms_hartree_per_bohr": float(np.sqrt(np.mean(atom_norms**2))),
        "atom_max_hartree_per_bohr": float(atom_norms.max()),
        "gradient_norm_hartree_per_bohr": float(np.linalg.norm(gradient)),
    }
    for key, value in recomputed.items():
        assert np.isclose(value, record[key], rtol=1e-13, atol=1e-15), key
    criteria = inp["settings"]["optimizer_thresholds"]
    passes = (recomputed["atom_rms_hartree_per_bohr"] <= criteria["convergence_grms"] and
              recomputed["atom_max_hartree_per_bohr"] <= criteria["convergence_gmax"])
    assert record["scf_converged"] and not passes and not record["final_gradient_thresholds_passed"]
    assert sha(step_files["worker"]) == inp["script_sha256"]
    review = {
        "status": "DONE_INDEPENDENT_RECALCULATION_OF_COMPLETED_STEP1_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "quantum_and_integral_jobs_started_by_this_audit": 0,
        "script_sha256": sha(__file__),
        "completed_geometry_step": 1,
        "saved_scf_converged": record["scf_converged"],
        "saved_energy_hartree": record["energy_hartree"],
        "energy_scope": "Copied from the completed step record, not independently recomputed by a new electronic calculation.",
        "atom_count": 75,
        "composition": dict(composition),
        "settings_from_worker_input": inp["settings"],
        "gradient_statistics_recomputed_from_saved_array": recomputed,
        "all_five_gradient_statistics_match_saved_record": True,
        "passes_final_gradient_thresholds": passes,
        "gRMS_over_threshold_factor": recomputed["atom_rms_hartree_per_bohr"] / criteria["convergence_grms"],
        "gmax_over_threshold_factor": recomputed["atom_max_hartree_per_bohr"] / criteria["convergence_gmax"],
        "scientific_interpretation": "A real converged SCF and a real gradient are evidenced at the initial completed geometry step. It is not an optimized stationary structure. This review does not assert the current or final optimization status.",
        "frequencies_at_this_geometry_step": "NOT_RUN",
        "IR_intensities": "NOT_COMPUTED",
        "input_and_completed_step_evidence": {
            key: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
            for key, path in step_files.items()},
    }
    write_new(step_path, review)
    index = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Additive cost and completed-step1 supplement only. Original20-file preflight index remains unchanged.",
        "prior_preflight_index_sha256": sha(preflight),
        "files_sha256": {path.name: sha(path) for path in (
            Path(__file__), cost_path, step_path, snapshot_path)},
    }
    write_new(index_path, index)
    print(json.dumps({"status": "PASS_NO_NEW_QUANTUM_JOBS", "molecule_dimensions": sizes,
                      "step1_gradient_recomputed": recomputed,
                      "step1_optimized": False,
                      "supplement_index_sha256": sha(index_path)}, indent=2))


if __name__ == "__main__":
    main()
