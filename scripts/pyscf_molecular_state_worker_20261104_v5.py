"""Supervised neutral CHO PBE state preparation; exact, pickle-free MF snapshots.

This entry does not compute a Hessian. A snapshot is eligible for a later
Hessian only after its *parent* V5 phase has exited cleanly and all source,
array, molecule and actual quadrature fingerprints have been checked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py"
SUPERVISOR_PATH = ROOT / "scripts/run_pyscf_actual_cpu_budget_20261104_v5.py"
BASE_SHA = "acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d"
SUPERVISOR_SHA = "33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee"
INPUT_SCHEMA = "research4-molecular-input-v1"
SNAPSHOT_SCHEMA = "research4-exact-pbe-mf-snapshot-v1"
RESULT_SCHEMA = "research4-molecular-state-result-v1"
ATOMIC_NUMBERS = {"H": 1, "C": 6, "O": 8}
GRID_ATTRIBUTES = ("level", "atom_grid", "alignment", "cutoff", "radi_cutoff",
                   "radi_method", "becke_scheme", "prune", "radii_adjust", "atomic_radii")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def document_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save_json(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive and path.exists():
        raise FileExistsError("Immutable artifact exists: " + str(path))
    temporary = path.with_name(path.name + ".writing" + ("." + str(os.getpid()) if exclusive else ""))
    with temporary.open("x" if exclusive else "w") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if exclusive:
        try:
            os.link(temporary, path)  # atomic no-replace; never overwrite a winner
        finally:
            temporary.unlink(missing_ok=True)
    else:
        os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError("Unsupported non-JSON molecular state: " + type(value).__name__)


def save_immutable_npy(path, value):
    """Close/fsync bytes, publish with atomic no-replace, then fsync directory."""
    path = Path(path)
    temporary = path.with_name(path.name + ".writing." + str(os.getpid()))
    try:
        with temporary.open("xb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def load_frozen(path, expected, name):
    if sha(path) != expected:
        raise ValueError(name + " source SHA differs from reviewed source")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_base():
    return load_frozen(BASE_PATH, BASE_SHA, "research4_frozen_v8_state_dependency")


def install_sites():
    for relative in (".runtime/quantum-pilot-site", ".runtime/quantum-upgrade-site"):
        location = str(ROOT / relative)
        if location not in sys.path:
            sys.path.insert(0, location)


def resolve_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def artifact_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def checked_artifact(record, *, max_bytes=64 * 1024**2):
    if not isinstance(record, dict) or set(record) - {"path", "sha256", "bytes"}:
        raise ValueError("Explicit artifact path/hash/size record required")
    path = resolve_path(record.get("path", ""))
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("Artifact missing or oversized: " + str(path))
    if record.get("bytes") is not None and record["bytes"] != path.stat().st_size:
        raise ValueError("Artifact byte count changed")
    if record.get("sha256") != sha(path):
        raise ValueError("Artifact SHA changed: " + str(path))
    return path


def validate_input(manifest_path, expected_sha):
    manifest_path = Path(manifest_path).resolve()
    if not expected_sha or sha(manifest_path) != expected_sha:
        raise ValueError("Explicit input manifest SHA required")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValueError("Unsupported molecular input schema")
    symbols = manifest.get("atom_symbols")
    if not isinstance(symbols, list) or not symbols or any(s not in ATOMIC_NUMBERS for s in symbols):
        raise ValueError("Only fully explicit CHO input atoms are supported")
    if manifest.get("composition") != dict(Counter(symbols)):
        raise ValueError("Declared composition differs from exact atom order")
    counts = Counter(symbols)
    order = [key for key in ("C", "H", "O") if key in counts] if "C" in counts else sorted(counts)
    formula = "".join(symbol + (str(counts[symbol]) if counts[symbol] != 1 else "") for symbol in order)
    if manifest.get("formula") != formula:
        raise ValueError("Molecular formula differs from explicit atom composition")
    if not isinstance(manifest.get("molecule_id"), str) or not manifest["molecule_id"].strip():
        raise ValueError("Explicit molecule ID required")
    if (manifest.get("charge") != 0 or manifest.get("spin_2S") != 0
            or manifest.get("multiplicity") != 1 or manifest.get("radical_electrons") != 0):
        raise ValueError("This pilot supports neutral nonradical singlets only")
    electrons = sum(ATOMIC_NUMBERS[s] for s in symbols)
    if manifest.get("electron_count") != electrons or electrons % 2:
        raise ValueError("Electron count inconsistent with composition")
    if manifest.get("connectivity_verified") is not True or manifest.get("stereochemistry_checked") is not True:
        raise ValueError("Connectivity and stereochemistry checks required")
    xyz = checked_artifact(manifest["input_xyz"])
    npy = checked_artifact(manifest["coordinates_angstrom_npy"])
    base = frozen_base()
    read_symbols, displayed = base.read_xyz(xyz, Counter(symbols))
    if read_symbols != symbols:
        raise ValueError("XYZ atom order differs; chemical identity does not imply AO identity")
    coordinates = np.load(npy, allow_pickle=False)
    if coordinates.dtype.kind != "f" or coordinates.shape != (len(symbols), 3) or not np.isfinite(coordinates).all():
        raise ValueError("Invalid exact coordinate array")
    if not np.allclose(coordinates, displayed, atol=5e-12, rtol=0):
        raise ValueError("Exact coordinates disagree with bound XYZ")
    for source in manifest.get("source_artifacts", []):
        checked_artifact(source)
    initial_checkpoint = manifest.get("initial_checkpoint")
    if initial_checkpoint is not None:
        checked_artifact(initial_checkpoint, max_bytes=32 * 1024**2)
        order_binding = manifest.get("initial_checkpoint_atom_order_sha256")
        if order_binding != document_hash(symbols):
            raise ValueError("Warmstart checkpoint lacks exact atom-order binding")
        if manifest.get("initial_checkpoint_geometry_npy_sha256") != manifest["coordinates_angstrom_npy"]["sha256"]:
            raise ValueError("Warmstart checkpoint lacks matching full-precision geometry")
    return manifest, symbols, coordinates


def method_settings(threads=4, maxsteps=35):
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError("One through four supervised threads required")
    if type(maxsteps) is not int or not 1 <= maxsteps <= 100:
        raise ValueError("Bounded optimizer steps required")
    return {"xc": "PBE", "basis": "def2-svp", "grid_level": 1,
            "dispersion": "NONE_EXPLICIT_PBE_PILOT_NOT_D3", "density_fitting": False,
            "auxbasis": None, "density_fit_backend": "SCREENED_DIRECT_RKS_V4",
            "direct_scf": True, "direct_scf_tol": 1e-13,
            "global_four_center_ERI_cache_allowed": False,
            "scf_conv_tol_hartree": 1e-8, "scf_max_cycle": 80,
            "hessian_conv_tol_cpscf": 1e-8, "max_memory_mb": 8000,
            "threads": threads, "native_direct_driver_threads": threads,
            "charge": 0, "spin_2S": 0, "multiplicity": 1,
            "optimizer_maxsteps": maxsteps, "optimizer_thresholds": dict(frozen_base().THRESHOLDS),
            "gradient_grid_response": False,
            "engineering_worker_version": "EXACT_MOLECULAR_STATE_V1_NO_HESSIAN"}


def validate_optimized_source(manifest, settings):
    path = checked_artifact(manifest.get("optimized_source_result", {}))
    old = json.loads(path.read_text())
    for key in ("geometry_optimized", "optimizer_converged", "scf_converged", "final_gradient_thresholds_passed"):
        if old.get(key) is not True:
            raise ValueError("Prepared geometry lacks original optimized evidence: " + key)
    if old.get("status") != "DONE":
        raise ValueError("Original optimization not DONE")
    if old.get("optimized_coordinates_angstrom_npy_sha256") != manifest["coordinates_angstrom_npy"]["sha256"]:
        raise ValueError("Original optimized full precision coordinates changed")
    if manifest.get("initial_checkpoint") and manifest["initial_checkpoint"]["sha256"] != old.get("checkpoint_sha256"):
        raise ValueError("Warmstart is not the checkpoint bound by the original optimization")
    for key in ("xc", "basis", "grid_level", "dispersion", "density_fitting", "direct_scf", "direct_scf_tol",
                "scf_conv_tol_hartree", "charge", "spin_2S", "multiplicity", "optimizer_thresholds"):
        if old.get("settings", {}).get(key) != settings[key]:
            raise ValueError("Prepared method differs from original optimization: " + key)
    return {"result": artifact_record(path), "original_optimizer_converged": True,
            "new_optimizer_run": False}


def callable_record(function):
    if function is None:
        return None
    if not callable(function) or not getattr(function, "__module__", "").startswith("pyscf."):
        raise ValueError("Only importable PySCF grid callables are supported")
    name = function.__qualname__
    if "<" in name:
        raise ValueError("Non-importable grid function")
    source = inspect.getsourcefile(function)
    return {"module": function.__module__, "qualname": name,
            "source_sha256": sha(source) if source else None}


def resolve_callable(record):
    if record is None:
        return None
    if not record.get("module", "").startswith("pyscf.") or any(part.startswith("_") for part in record.get("qualname", "").split(".")):
        raise ValueError("Unreviewed grid callable")
    value = importlib.import_module(record["module"])
    for part in record["qualname"].split("."):
        value = getattr(value, part)
    if callable_record(value) != record:
        raise ValueError("Grid callable source changed")
    return value


def grid_settings(grids):
    result = {}
    for name in GRID_ATTRIBUTES:
        if not hasattr(grids, name):
            continue
        value = getattr(grids, name)
        result[name] = {"callable": callable_record(value)} if callable(value) else {"value": json_safe(value)}
    return result


def environment_identity():
    install_sites()
    import pyscf
    import scipy
    from scipy.linalg import _fblas
    from pyscf import gto
    from pyscf.gto import moleintor
    from pyscf.dft import gen_grid, numint, rks, libxc
    from pyscf.grad import rks as grad_rks
    from pyscf.scf import _vhf, hf
    sources = {module.__name__: sha(inspect.getsourcefile(module))
               for module in (gto.mole, gen_grid, numint, rks, grad_rks, hf, _vhf)}
    for name, binary in (("libcvhf", _vhf.libcvhf._name), ("libcgto", moleintor.libcgto._name),
                         ("libdft", numint.libdft._name), ("libxc_itrf", libxc._itrf._name),
                         ("numpy_multiarray", np._core._multiarray_umath.__file__),
                         ("scipy_fblas", _fblas.__file__)):
        sources[name + "_binary"] = sha(binary)
    dependency_dir = Path(pyscf.__file__).parent / "lib/deps/lib"
    for pattern in ("libcint.so*", "libxc.so*"):
        for binary in sorted(dependency_dir.glob(pattern)):
            sources[binary.name] = sha(binary)
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "pyscf": pyscf.__version__, "scipy": scipy.__version__, "geometric": importlib.metadata.version("geometric"),
            "sources": sources}


def molecule_document(mol):
    """Use JSON native state; unlike gto.loads, restoration does not use eval."""
    document = json.loads(mol.dumps())
    for name in ("atom", "basis", "ecp", "pseudo"):
        document[name] = json_safe(getattr(mol, name))
    if mol.symmetry or mol.has_ecp() or mol.cart:
        raise ValueError("Snapshot limited to nonsymmetric all-electron spherical molecules")
    document.pop("output", None)
    document.pop("stdout", None)
    return document


def molecule_physical_fingerprint(mol):
    base = frozen_base()
    env = np.array(mol._env, copy=True)
    if len(env) <= 17 or not np.isfinite(env[17]) or int(env[17]) != env[17] or not 0 <= env[17] < mol.natm:
        raise ValueError("Unknown PySCF last-rinv atom bookkeeping marker")
    env[17] = 0
    return {"atm": base.array_fingerprint(mol._atm), "bas": base.array_fingerprint(mol._bas),
            "env_except_transient17": base.array_fingerprint(env),
            "coordinates_bohr": base.array_fingerprint(mol.atom_coords(unit="Bohr")),
            "charges": base.array_fingerprint(mol.atom_charges()), "natm": int(mol.natm),
            "nao": int(mol.nao_nr()), "nbas": int(mol.nbas), "charge": int(mol.charge), "spin": int(mol.spin),
            "nelectron": int(mol.nelectron), "cart": bool(mol.cart),
            "excluded_transient_env_indices": [17]}


def validate_arrays(arrays, natm, nao, electrons, nbas=None):
    required = {"MO_coeff", "MO_energy", "MO_occ", "grid_coordinates", "grid_weights", "coordinates_angstrom", "gradient"}
    if not required.issubset(arrays):
        raise ValueError("Missing complete snapshot arrays")
    if set(arrays) - required - {"grid_non0tab", "grid_screen_index", "grid_atm_idx", "grid_quadrature_weights"}:
        raise ValueError("Unexpected snapshot array")
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("Invalid real numeric array: " + name)
    coeff, energy, occ = (arrays[key] for key in ("MO_coeff", "MO_energy", "MO_occ"))
    if coeff.shape != (nao, nao) or energy.shape != (nao,) or occ.shape != (nao,):
        raise ValueError("Complete square AO/MO dimensions required")
    if not np.isin(occ, [0, 2]).all() or float(occ.sum()) != electrons:
        raise ValueError("Invalid restricted closed-shell occupations")
    coords, weights = arrays["grid_coordinates"], arrays["grid_weights"]
    if coords.ndim != 2 or coords.shape[1] != 3 or weights.shape != (coords.shape[0],) or not len(weights):
        raise ValueError("Actual full quadrature coordinates/weights required")
    if arrays["coordinates_angstrom"].shape != (natm, 3) or arrays["gradient"].shape != (natm, 3):
        raise ValueError("Invalid geometry/gradient dimensions")
    for name in ("grid_non0tab", "grid_screen_index"):
        if name in arrays:
            a = arrays[name]
            if a.dtype != np.uint8 or a.ndim != 2 or not a.size:
                raise ValueError("Invalid actual grid screen table")
            if nbas is not None and a.shape != ((len(weights) + 55) // 56, nbas):
                raise ValueError("Grid screen table dimensions disagree with BLKSIZE56/full basis")
    if "grid_non0tab" in arrays and "grid_screen_index" in arrays:
        if not np.array_equal(arrays["grid_non0tab"], arrays["grid_screen_index"]):
            raise ValueError("PySCF non0tab/screen_index aliases disagree")
    if "grid_atm_idx" in arrays:
        value = arrays["grid_atm_idx"]
        if value.shape != weights.shape or value.dtype.kind not in "iu" or np.any((value < -1) | (value >= natm)):
            raise ValueError("Invalid grid atom attribution")
    if "grid_quadrature_weights" in arrays and arrays["grid_quadrature_weights"].shape != weights.shape:
        raise ValueError("Grid quadrature-weight dimensions disagree")


def snapshot_arrays(mf, gradient):
    arrays = {"MO_coeff": np.asarray(mf.mo_coeff), "MO_energy": np.asarray(mf.mo_energy),
              "MO_occ": np.asarray(mf.mo_occ), "grid_coordinates": np.asarray(mf.grids.coords),
              "grid_weights": np.asarray(mf.grids.weights),
              "coordinates_angstrom": mf.mol.atom_coords(unit="Angstrom"), "gradient": np.asarray(gradient)}
    for attribute, name in (("non0tab", "grid_non0tab"), ("screen_index", "grid_screen_index"),
                            ("atm_idx", "grid_atm_idx"), ("quadrature_weights", "grid_quadrature_weights")):
        value = getattr(mf.grids, attribute, None)
        if value is not None:
            arrays[name] = np.asarray(value)
    validate_arrays(arrays, mf.mol.natm, mf.mol.nao_nr(), mf.mol.nelectron, mf.mol.nbas)
    return arrays


def save_snapshot(directory, mf, gradient, state):
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError("Exact snapshot directories are immutable")
    if (state.get("scf_converged") is not True or state.get("geometry_optimized") is not True
            or not mf.converged or not np.isfinite(mf.e_tot)):
        raise ValueError("SCF/optimized geometry not verified")
    if not frozen_base().stationary_checks(gradient)["final_gradient_thresholds_passed"]:
        raise ValueError("Fresh same-method gradient not stationary")
    frozen_base().direct_integral_policy(mf)
    directory.mkdir()
    arrays = snapshot_arrays(mf, gradient)
    files = {}
    for name, array in arrays.items():
        target = directory / (name + ".npy")
        save_immutable_npy(target, array)
        files[name] = {"file": target.name, "sha256": sha(target), **frozen_base().array_fingerprint(array)}
    doc = molecule_document(mf.mol)
    save_json(directory / "molecule.json", doc, exclusive=True)
    manifest = {"schema": SNAPSHOT_SCHEMA, "created_at_utc": utc(), "status": "EXACT_SCF_GRADIENT_STATE_SAVED",
                "source_worker": artifact_record(__file__), "frozen_V8_sha256": BASE_SHA,
                "supervisor_sha256": SUPERVISOR_SHA,
                "environment": state["versions"], "molecule": {"file": "molecule.json", "sha256": sha(directory / "molecule.json")},
                "molecule_fingerprint": molecule_physical_fingerprint(mf.mol),
                "arrays": files, "grid_settings": grid_settings(mf.grids),
                "energy_hartree": float(mf.e_tot), "settings": state["settings"],
                "MF_small_rho_cutoff": float(mf.small_rho_cutoff),
                "MF_grid_initialized": True, "MF_nlc": mf.nlc, "MF_disp": getattr(mf, "disp", None),
                "numint_cutoff": float(mf._numint.cutoff),
                "numint_omega": json_safe(mf._numint.omega),
                "producer": {"result_path": str(directory.parent / "result.json"),
                             "resource_supervision": state["resource_supervision"],
                             "input_manifest": state["input_manifest"],
                             "identity": state["identity"], "fresh_gradient_checks": frozen_base().stationary_checks(gradient),
                             "geometry_optimized": True, "scf_converged": True},
                "frequencies_status": "NOT_RUN", "IR_intensities_status": "NOT_COMPUTED"}
    save_json(directory / "manifest.json", manifest, exclusive=True)
    return artifact_record(directory / "manifest.json")


def validate_closed_producer(manifest, manifest_path):
    producer = manifest["producer"]
    supervision = producer["resource_supervision"]
    proof_path = resolve_path(supervision["proof_path"])
    if sha(proof_path) != supervision["proof_sha256"]:
        raise ValueError("Producer supervision proof changed")
    proof = json.loads(proof_path.read_text())
    if proof != supervision["proof"] or proof.get("schema") != "research4-pyscf-actual-cpu-supervisor-proof-v5":
        raise ValueError("Producer proof mismatch")
    if proof.get("supervisor_script_sha256") != SUPERVISOR_SHA:
        raise ValueError("Producer supervisor dependency changed")
    ledger = json.loads(resolve_path(proof["ledger_path"]).read_text())
    if ledger.get("schema") != "research4-pyscf-actual-cpu-budget-v5" or ledger.get("runner_sha256") != SUPERVISOR_SHA:
        raise ValueError("Producer ledger schema/source mismatch")
    rows = [row for row in ledger.get("phases", []) if row.get("phase") == proof.get("phase")]
    if len(rows) != 1:
        raise ValueError("Producer phase is not unique")
    row = rows[0]
    if (row.get("status") != "EXITED" or row.get("exit_code") != 0 or not row.get("ended_at_utc")
            or any(row.get(key) is not False for key in ("wall_terminated", "memory_terminated", "cpu_terminated", "monitor_failed", "hard_killed"))
            or row.get("residual_processes_require_review_after_main_payload_exit")):
        raise ValueError("Producer parent has not closed cleanly")
    if row.get("supervisor_proof_sha256") != sha(proof_path) or resolve_path(row["supervisor_proof_path"]) != proof_path:
        raise ValueError("Closed producer row does not bind immutable proof")
    if not math.isfinite(row.get("actual_cpu_seconds", float("nan"))) or row["actual_cpu_seconds"] < 0:
        raise ValueError("Producer CPU cost not measured")
    log = resolve_path(row["log"])
    if row.get("log_sha256") != sha(log):
        raise ValueError("Closed producer log changed")
    result_path = resolve_path(producer["result_path"])
    result = json.loads(result_path.read_text())
    if (result.get("schema") != RESULT_SCHEMA or result.get("status") != "DONE"
            or result.get("snapshot_manifest", {}).get("sha256") != sha(manifest_path)
            or resolve_path(result.get("snapshot_manifest", {}).get("path", "")) != Path(manifest_path).resolve()
            or result.get("resource_supervision") != supervision
            or result.get("source_worker") != manifest["source_worker"]
            or result.get("settings") != manifest["settings"]
            or result.get("versions") != manifest["environment"]
            or result.get("input_manifest") != producer["input_manifest"]
            or result.get("energy_hartree") != manifest["energy_hartree"]):
        raise ValueError("Snapshot lacks matching DONE producer result")
    for name in ("geometry_optimized", "scf_converged", "final_gradient_thresholds_passed"):
        if result.get(name) is not True:
            raise ValueError("Producer convergence evidence incomplete")
    if result.get("native_direct_driver_control", {}).get("original_driver_restored") is not True:
        raise ValueError("Producer native guard not restored")
    if result.get("native_direct_driver_control", {}).get("restoration_failures") != 0:
        raise ValueError("Producer native guard restoration failed")
    source = resolve_path(manifest["source_worker"]["path"])
    command = row.get("command", [])
    if str(source) not in [str(resolve_path(token)) for token in command if isinstance(token, str) and token.endswith(".py")]:
        raise ValueError("Producer command does not invoke the bound state worker")
    return {"phase": row["phase"], "closed_result_sha256": sha(result_path),
            "proof_sha256": sha(proof_path), "log_sha256": sha(log),
            "actual_cpu_seconds": row["actual_cpu_seconds"],
            "closed_row_sha256": document_hash(row)}


def load_snapshot_payload(directory, *, expected_manifest_sha256=None, require_closed_producer=True):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    if expected_manifest_sha256 and sha(manifest_path) != expected_manifest_sha256:
        raise ValueError("Snapshot manifest SHA differs from expected frozen state")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SNAPSHOT_SCHEMA or manifest.get("status") != "EXACT_SCF_GRADIENT_STATE_SAVED":
        raise ValueError("Unsupported/incomplete snapshot manifest")
    source = checked_artifact(manifest["source_worker"])
    if source != Path(__file__).resolve() or manifest["frozen_V8_sha256"] != BASE_SHA or manifest["supervisor_sha256"] != SUPERVISOR_SHA:
        raise ValueError("Snapshot source/dependencies differ")
    frozen_base()
    load_frozen(SUPERVISOR_PATH, SUPERVISOR_SHA, "research4_v5_snapshot_verifier_dependency")
    producer_evidence = validate_closed_producer(manifest, manifest_path) if require_closed_producer else None
    mol_record = manifest["molecule"]
    if mol_record.get("file") != "molecule.json" or mol_record.get("sha256") != sha(directory / "molecule.json"):
        raise ValueError("Molecule document SHA changed")
    molecule = json.loads((directory / "molecule.json").read_text())
    arrays = {}
    for name, record in manifest["arrays"].items():
        if record.get("file") != name + ".npy" or not name.replace("_", "").isalnum():
            raise ValueError("Unexpected array filename")
        path = directory / record["file"]
        if path.stat().st_size > 64 * 1024**2 or sha(path) != record["sha256"]:
            raise ValueError("Snapshot array byte SHA changed")
        array = np.load(path, allow_pickle=False)
        if frozen_base().array_fingerprint(array) != {key: record[key] for key in ("shape", "dtype", "payload_bytes", "payload_sha256_C_order")}:
            raise ValueError("Snapshot payload fingerprint changed")
        arrays[name] = array
    physical = manifest["molecule_fingerprint"]
    validate_arrays(arrays, physical["natm"], physical["nao"], physical["nelectron"], physical.get("nbas"))
    if not frozen_base().stationary_checks(arrays["gradient"])["final_gradient_thresholds_passed"]:
        raise ValueError("Stored gradient is not stationary")
    binding = {"schema": SNAPSHOT_SCHEMA, "snapshot_manifest_sha256": sha(manifest_path),
               "snapshot_directory": str(directory), "producer": manifest["producer"],
               "molecule_fingerprint": physical, "arrays": manifest["arrays"],
               "environment": manifest["environment"], "method_settings": manifest["settings"],
               "source_worker": manifest["source_worker"], "closed_producer": producer_evidence}
    return manifest, molecule, arrays, binding


def restore_snapshot(directory, *, expected_manifest_sha256=None, expected_source_sha256=None,
                     require_closed_producer=True, stdout=None):
    """Restore original converged MOs and grids, without a new SCF or grid build.

    Scientific consumers must use the default closed-producer gate. The optional
    false value is for pure fixture inspection only and cannot restore a live MF.
    """
    if not require_closed_producer:
        raise ValueError("Scientific MF restoration always requires a closed producer")
    manifest, molecule, arrays, binding = load_snapshot_payload(
        directory, expected_manifest_sha256=expected_manifest_sha256, require_closed_producer=True)
    if expected_source_sha256 and expected_source_sha256 != manifest["source_worker"]["sha256"]:
        raise ValueError("Unexpected snapshot producer source")
    if environment_identity() != manifest["environment"]:
        raise ValueError("Snapshot PySCF/dependency environment changed")
    from pyscf import dft, gto
    from pyscf.scf.hf import SCF
    mol = gto.Mole()
    mol.__dict__.update(molecule)
    for key, dtype in (("_atm", np.int32), ("_bas", np.int32), ("_env", np.float64), ("_ecpbas", np.int32)):
        setattr(mol, key, np.asarray(getattr(mol, key), dtype=dtype))
    if not mol._ecpbas.size:
        mol._ecpbas = mol._ecpbas.reshape(0, 8)
    mol.stdout = stdout if stdout is not None else sys.stdout
    mol.output = None
    mol.incore_anyway = False
    if molecule_physical_fingerprint(mol) != manifest["molecule_fingerprint"]:
        raise ValueError("Restored molecule differs from source physical state")
    if not np.array_equal(mol.atom_coords(unit="Angstrom"), arrays["coordinates_angstrom"]):
        raise ValueError("Molecule and snapshot geometry differ")
    mf = make_scf(mol, manifest["settings"], None)
    for name, record in manifest["grid_settings"].items():
        if name not in GRID_ATTRIBUTES:
            raise ValueError("Unknown grid setting")
        value = resolve_callable(record["callable"]) if "callable" in record else record["value"]
        if name == "atomic_radii":
            value = np.asarray(value, float)
        setattr(mf.grids, name, value)
    mf.grids.coords = arrays["grid_coordinates"]
    mf.grids.weights = arrays["grid_weights"]
    for name, key in (("non0tab", "grid_non0tab"), ("screen_index", "grid_screen_index"),
                      ("atm_idx", "grid_atm_idx"), ("quadrature_weights", "grid_quadrature_weights")):
        if key in arrays:
            setattr(mf.grids, name, arrays[key])
    mf.mo_coeff, mf.mo_energy, mf.mo_occ = (arrays[key] for key in ("MO_coeff", "MO_energy", "MO_occ"))
    mf.e_tot = manifest["energy_hartree"]
    mf.converged = True
    mf.small_rho_cutoff = manifest["MF_small_rho_cutoff"]
    mf.nlc = manifest["MF_nlc"]
    mf.disp = manifest["MF_disp"]
    mf._numint.cutoff = manifest["numint_cutoff"]
    mf._numint.omega = manifest["numint_omega"]
    mf.chkfile = None
    if grid_settings(mf.grids) != manifest["grid_settings"]:
        raise ValueError("Restored grid settings differ from source")
    for key, value in snapshot_arrays(mf, arrays["gradient"]).items():
        if frozen_base().array_fingerprint(value) != frozen_base().array_fingerprint(arrays[key]):
            raise ValueError("Restored actual quadrature/MO payload changed")
    mf._research4_restoration_status = "RESTORED_FROM_VERIFIED_SOURCE_SCF"
    mf._research4_snapshot_binding = binding
    return mf, binding


def make_scf(mol, settings, checkpoint):
    from pyscf import dft
    from pyscf.scf.hf import SCF
    base = frozen_base()
    cls = base.screened_direct_rks_type(dft.rks.RKS, SCF.get_jk)
    mf = cls(mol)
    mf.xc = settings["xc"]
    mf.disp = None
    mf.max_memory = settings["max_memory_mb"]
    mf.direct_scf = True
    mf.direct_scf_tol = settings["direct_scf_tol"]
    mf.conv_tol_cpscf = settings["hessian_conv_tol_cpscf"]
    mf.conv_tol = settings["scf_conv_tol_hartree"]
    mf.max_cycle = settings["scf_max_cycle"]
    mf.grids.level = settings["grid_level"]
    mf._eri = None
    mol.incore_anyway = False
    mf.chkfile = str(checkpoint) if checkpoint else None
    if checkpoint and Path(checkpoint).exists():
        mf.init_guess = "chkfile"
    base.direct_integral_policy(mf)
    return mf


def execute(args):
    supervisor = load_frozen(SUPERVISOR_PATH, SUPERVISOR_SHA, "research4_v5_molecular_state_supervisor")
    supervision = supervisor.verify_resource_supervision(expected_threads=args.threads)
    manifest, symbols, coordinates = validate_input(args.input_manifest, args.input_manifest_sha256)
    settings = method_settings(args.threads, args.maxsteps)
    old_optimization = validate_optimized_source(manifest, settings) if args.phase == "prepare-state" else None
    install_sites()
    from pyscf import gto, lib
    from pyscf.scf import _vhf
    from pyscf.grad.rks import Gradients
    from pyscf.geomopt import geometric_solver
    lib.num_threads(args.threads)
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Use a fresh molecular state output directory")
    out.mkdir(parents=True)
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    state = {"schema": RESULT_SCHEMA, "status": "PARTIAL", "phase": args.phase,
             "identity": manifest["molecule_id"], "started_at_utc": utc(),
             "settings": settings, "input_manifest": artifact_record(args.input_manifest),
             "source_worker": artifact_record(__file__), "resource_supervision": supervision,
             "versions": environment_identity(), "original_optimization": old_optimization,
             "scf_converged": False, "geometry_optimized": False, "optimizer_converged": False,
             "frequencies_status": "NOT_RUN", "IR_intensities_status": "NOT_COMPUTED"}
    save_json(out / "input.json", {**state, "molecular_input": manifest}, exclusive=True)
    shutil.copy2(checked_artifact(manifest["input_xyz"]), out / "input.xyz")
    frozen_base().atomic_npy(out / "input_coordinates_angstrom.npy", coordinates)
    checkpoint = out / "calculation.chk"
    if manifest.get("initial_checkpoint"):
        shutil.copy2(checked_artifact(manifest["initial_checkpoint"]), checkpoint)
    log = (out / "pyscf.log").open("x", buffering=1)
    def emit(event, details=None, **kwargs):
        record = {"utc": utc(), "event": event, "process_cpu_seconds": time.process_time() - started_cpu,
                  "wall_seconds": time.perf_counter() - started_wall, **(details or {}), **kwargs}
        with (out / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        save_json(out / "progress.json", {**state, **record})
    gradient_cls = frozen_base().direct_gradient_type(Gradients, emit, memory_mb=lambda: lib.current_memory()[0])
    def gradient_method(mf):
        gradient = gradient_cls(mf)
        gradient.grid_response = False
        return gradient
    def stop(sig, _frame):
        raise frozen_base().BudgetStop("Received supervised signal " + str(sig))
    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU)}
    guard = frozen_base().NativeDirectDriverThreadGuard(_vhf, lib.with_omp_threads, lib.num_threads,
            native_threads=args.threads, emit=lambda details: emit("native_driver_signature", details))
    try:
        guard.install()
        mol = gto.Mole()
        mol.stdout = log
        mol.build(atom=list(zip(symbols, coordinates)), unit="Angstrom", basis=settings["basis"],
                  charge=0, spin=0, symmetry=False, max_memory=settings["max_memory_mb"], verbose=4)
        mol.incore_anyway = False
        if mol.nelectron != manifest["electron_count"]:
            raise ValueError("Built molecule electron count differs")
        mf = make_scf(mol, settings, checkpoint)
        mf.callback = lambda env: emit("scf_cycle", cycle=int(env.get("cycle", -1)), energy_hartree=float(env["e_tot"]))
        if args.phase == "optimize-state":
            (out / "geometry_steps").mkdir()
            def geometry_callback(env):
                cycle = int(env["self"].cycle)
                step = out / "geometry_steps" / f"step_{cycle:04d}"
                coords = env["mol"].atom_coords(unit="Angstrom")
                gradient = np.asarray(env["gradients"])
                step.with_suffix(".xyz").write_text(frozen_base().xyz_text(symbols, coords, "Optimization intermediate, not a minimum certificate"))
                frozen_base().atomic_npy(step.with_suffix(".coordinates_angstrom.npy"), coords)
                frozen_base().atomic_npy(step.with_suffix(".gradient.npy"), gradient)
                info = {"cycle": cycle, "energy_hartree": float(env["energy"]),
                        "scf_converged": bool(env["g_scanner"].converged), **frozen_base().stationary_checks(gradient)}
                save_json(step.with_suffix(".json"), info, exclusive=True)
                emit("geometry_cycle", info)
            emit("geometry_optimization_started")
            converged, mol = geometric_solver.kernel(gradient_method(mf), maxsteps=args.maxsteps,
                    assert_convergence=True, callback=geometry_callback, **frozen_base().THRESHOLDS)
            state["optimizer_converged"] = bool(converged)
            if not converged:
                raise RuntimeError("Optimizer returned without convergence")
        else:
            state["optimizer_converged"] = True
            state["optimizer_convergence_basis"] = "VERIFIED_ORIGINAL_OPTIMIZATION_NO_NEW_OPT_RUN"
        coords = mol.atom_coords(unit="Angstrom")
        (out / "optimized.xyz").write_text(frozen_base().xyz_text(symbols, coords, "Inspect result.json for fresh gradient and optimization evidence"))
        frozen_base().atomic_npy(out / "optimized_coordinates_angstrom.npy", coords)
        state["optimized_xyz_sha256"] = sha(out / "optimized.xyz")
        state["optimized_coordinates_angstrom_npy_sha256"] = sha(out / "optimized_coordinates_angstrom.npy")
        # Fresh SCF at the final geometry; no optimizer gradient is reused.
        mf = make_scf(mol, settings, checkpoint)
        mf.callback = lambda env: emit("final_scf_cycle", cycle=int(env.get("cycle", -1)), energy_hartree=float(env["e_tot"]))
        state["energy_hartree"] = float(mf.kernel())
        state["scf_converged"] = bool(mf.converged)
        if not mf.converged or not np.isfinite(mf.e_tot):
            raise RuntimeError("Final same-method SCF not converged")
        gradient = np.asarray(gradient_method(mf).kernel())
        frozen_base().atomic_npy(out / "gradient_hartree_per_bohr.npy", gradient)
        state.update(frozen_base().stationary_checks(gradient))
        state["geometry_optimized"] = bool(state["optimizer_converged"] and state["final_gradient_thresholds_passed"])
        state.update(atoms=mol.natm, electrons=mol.nelectron, basis_functions=mol.nao_nr())
        if not state["geometry_optimized"]:
            raise RuntimeError("Fresh gradient did not pass stationarity thresholds")
        state["snapshot_manifest"] = save_snapshot(out / "exact-mf-snapshot", mf, gradient, state)
        state["status"] = "DONE"
        emit("exact_snapshot_published", snapshot_manifest=state["snapshot_manifest"])
    except BaseException as exc:
        state.update(status="PARTIAL", stop_reason=str(exc), exception_type=type(exc).__name__)
        (out / "exception_traceback.txt").write_text(traceback.format_exc())
    finally:
        guard.restore()
        state["native_direct_driver_control"] = guard.summary()
        if state["native_direct_driver_control"]["restoration_failures"]:
            state.update(status="PARTIAL", stop_reason="Native driver context restoration failed")
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        state.update(completed_at_utc=utc(), process_cpu_seconds=time.process_time()-started_cpu,
                     wall_seconds=time.perf_counter()-started_wall,
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     checkpoint=artifact_record(checkpoint) if checkpoint.exists() else None)
        log.flush()
        log.close()
        save_json(out / "result.json", state, exclusive=True)
    print(json.dumps({key: state.get(key) for key in ("status", "identity", "phase", "energy_hartree", "geometry_optimized", "stop_reason")}))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare-state", "optimize-state"), required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--maxsteps", type=int, default=35)
    args = parser.parse_args()
    state = execute(args)
    raise SystemExit(0 if state["status"] == "DONE" else 2)


if __name__ == "__main__":
    main()
