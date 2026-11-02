"""Read ONLY fixed, closed water/ethanol metadata and arrays; no PySCF import.

This is an evidence inspector, not a restore/SCF/gradient/Hessian implementation.
No current Spinosin cache, checkpoint or HDF5 object is discovered or opened.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
RUN = ROOT / "reports/pyscf-spinosin-continuation-20261031-v3"
FIXTURES = (
    ("water-native4-validation-001", "water", ["hessian_cache_manifest.json"],
     "water-native4-validation-cache-001"),
    ("validation-ethanol-native4-001", "validation_ethanol", [
        "response_block4_native4/hessian_cache_manifest.json",
        "response_block8_native4/hessian_cache_manifest.json"],
     "validation-ethanol-native4-cache-001"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "payload_bytes": int(array.nbytes),
            "payload_sha256_C_order": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}


def main():
    output = REPORT / "closed_fixture_observations.json"
    if output.exists():
        raise FileExistsError("Keep this evidence immutable")
    rows, source_hashes = [], {}
    for directory, identity, manifest_names, cache_name in FIXTURES:
        folder = RUN / directory
        result_path = folder/"result.json"
        result = json.loads(result_path.read_text())
        if result.get("status") != "DONE" or result.get("identity") != identity:
            raise ValueError("A fixed completed validation fixture is required")
        source_hashes[str(result_path)] = sha(result_path)
        checkpoint = folder/"calculation.chk"
        source_hashes[str(checkpoint)] = sha(checkpoint)
        datasets, checkpoint_arrays = {}, {}
        with h5py.File(checkpoint, "r") as handle:
            def visitor(name, item):
                if isinstance(item, h5py.Dataset):
                    datasets[name] = {"shape": list(item.shape), "dtype": str(item.dtype)}
            handle.visititems(visitor)
            for key in ("mo_coeff", "mo_energy", "mo_occ", "e_tot"):
                checkpoint_arrays[key] = np.asarray(handle["scf/"+key][()])
            mol_text = handle["mol"][()]
            serialized = json.loads(mol_text.decode() if isinstance(mol_text, bytes) else str(mol_text))
        # Same canonicalization as frozen V8, but on serialized data only.
        original_marker = serialized["_env"][17]
        serialized["_env"][17] = 0.0
        canonical = hashlib.sha256(json.dumps(serialized, sort_keys=True, allow_nan=False).encode()).hexdigest()
        atoms, environment = np.asarray(serialized["_atm"]), np.asarray(serialized["_env"])
        coordinates = np.asarray([environment[int(row[1]):int(row[1])+3] for row in atoms])
        manifests = []
        for relative in manifest_names:
            path = folder/relative
            manifest = json.loads(path.read_text())
            source_hashes[str(path)] = sha(path)
            if manifest["status"] != "DONE_INTERMEDIATE_CAPTURE_ONLY":
                raise ValueError("Only a fully closed fixture cache is inspected")
            allowed_cache = (ROOT/"outputs/pyscf-spinosin-continuation-20261031-v3"/cache_name).resolve()
            cache = Path(manifest["cache_root"]).resolve()
            if not cache.is_relative_to(allowed_cache):
                raise ValueError("Refuse non-whitelisted cache (especially active target cache)")
            binding = manifest["binding"]
            binding_hash = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
            if binding_hash != manifest["binding_sha256"]:
                raise ValueError("Closed fixture binding hash differs")
            records = [manifest["partial_hessian"]["array"]]
            records += [entry["array"] for entry in manifest["H1AO_atoms"]]
            for block in manifest["response_blocks"]:
                for atom in block["arrays"]:
                    records.extend((atom["MO1"], atom["MO_E1"]))
            for record in records:
                artifact = Path(record["path"]).resolve()
                if not artifact.is_relative_to(cache) or sha(artifact) != record["file_sha256"]:
                    raise ValueError("Closed fixture payload path/hash changed")
                value = np.load(artifact, allow_pickle=False)
                if not np.isfinite(value).all() or fingerprint(value) != {key: record[key] for key in fingerprint(value)}:
                    raise ValueError("Closed fixture array values or dimensions differ")
            mo_checks = {key: fingerprint(checkpoint_arrays[key]) == binding[name]
                         for key, name in (("mo_coeff", "MO_coeff"), ("mo_energy", "MO_energy"), ("mo_occ", "MO_occ"))}
            manifests.append({"path": str(path), "manifest_sha256": sha(path),
                "cache_root": str(cache), "verified_array_file_count": len(records),
                "verified_array_payload_bytes": sum(record["payload_bytes"] for record in records),
                "dimensions": manifest["dimensions"], "response_block_atoms": manifest["response_block_atoms"],
                "complete_response_atom_ids": manifest["response_complete_global_atom_ids"],
                "checkpoint_MO_bytes_match_binding": mo_checks,
                "checkpoint_energy_exactly_matches_binding": float(checkpoint_arrays["e_tot"]) == binding["SCF_energy_hartree"],
                "checkpoint_coordinates_bytes_match_binding": fingerprint(coordinates) == binding["coordinates_bohr"],
                "checkpoint_canonical_Mole_matches_binding": canonical == binding["molecule_identity"]["canonical_serialized_sha256"],
                "source_checkpoint_last_rinv_marker_before_readonly_normalization": original_marker,
                "grid_coords_and_weights_stored_as_payload": False,
                "grid_coords_fingerprint": binding["grid_coordinates"],
                "grid_weights_fingerprint": binding["grid_weights"],
                "producer_worker_sha256": binding["worker_script_sha256"],
                "binding_top_level_keys": sorted(binding),
                "source_status_cross_process_restore": manifest["cross_process_restore"],
                "response_residuals_are_not_measured": all(block["residual_norm"] == "NOT_COMPUTED" for block in manifest["response_blocks"])})
        rows.append({"identity": identity, "source_directory": str(folder), "status": result["status"],
            "actual_SCF_converged_evidence": result.get("scf_converged"),
            "actual_fresh_gradient_passed_evidence": result.get("final_gradient_thresholds_passed"),
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
            "checkpoint_dataset_structure": datasets,
            "checkpoint_has_standalone_convergence_flag": any("converg" in key.lower() for key in datasets),
            "checkpoint_has_grid_payload": any("grid" in key.lower() for key in datasets),
            "cache_manifests": manifests})
    sources = {
        "scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py": ["436-503", "505-735", "909-948", "988-1055"],
        ".runtime/quantum-upgrade-site/pyscf/scf/chkfile.py": ["25-42"],
        ".runtime/quantum-upgrade-site/pyscf/scf/hf.py": ["1923-1950"],
        ".runtime/quantum-upgrade-site/pyscf/dft/rks.py": ["330-343", "499-519"],
        ".runtime/quantum-upgrade-site/pyscf/dft/gen_grid.py": ["Grids.build", "prune_by_density_"],
        ".runtime/quantum-upgrade-site/pyscf/hessian/rhf.py": ["42-89", "286-360", "546-572"],
        ".runtime/quantum-upgrade-site/pyscf/hessian/rks.py": ["Hessian.hess_elec alias", "make_h1"],
    }
    result = {"status": "NOT_IMPLEMENTED_NOT_VALIDATED", "created_at_UTC": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only feasibility; fixed closed software controls only",
        "new_SCF_gradient_Hessian_evaluations": 0, "resume_workers_or_commands_created": 0,
        "active_Spinosin_checkpoint_cache_or_HDF_reads": 0,
        "grid_reconstruction_or_Mole_construction_executed": False,
        "closed_validation_fixture_observations": rows,
        "input_hashes": source_hashes,
        "source_references": {path: {"sha256": sha(ROOT/path), "lines_or_symbols": lines} for path,lines in sources.items()}}
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "closed_fixture_count": len(rows),
        "manifests": sum(len(row["cache_manifests"]) for row in rows), "active_target_reads": 0,
        "new_quantum_evaluations": 0}))


if __name__ == "__main__":
    main()
