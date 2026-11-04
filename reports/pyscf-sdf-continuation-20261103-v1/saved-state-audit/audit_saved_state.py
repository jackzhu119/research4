"""Read only the explicitly authorized abandoned checkpoint and saved arrays.

No Mole deserialization, repr evaluation, SCF, integral, gradient, or Hessian
evaluation is used. This verifies saved components, not the abandoned phase.
"""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import csv
import hashlib
import json
import os

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "reports/pyscf-spinosin-pure-j-20261101-v4"
PHASE = OLD / "spinosin-pure-j-frequency-001"
FORENSIC = ROOT / "reports/pyscf-sdf-continuation-20261103-v1/interruption-audit"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, data):
    with Path(path).open("x") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def fingerprint(array):
    value = np.ascontiguousarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "payload_bytes": value.nbytes,
            "payload_sha256_C_order": hashlib.sha256(memoryview(value).cast("B")).hexdigest()}


def numeric_source_literals(path, names):
    """Read only numeric AST constants; never evaluate source or JSON repr."""
    result = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    value = node.value.value
                    if type(value) in (int, float):
                        result[target.id] = value
    if set(result) != set(names):
        raise ValueError("Required source numeric constants are not plain literals")
    return result


def main():
    if (OUT / "audit.json").exists():
        raise FileExistsError("Preserve existing saved-state audit")
    if any(Path(f"/proc/{pid}").exists() for pid in (18564, 18565)):
        raise RuntimeError("Original PID absence must remain true before abandoned reads")
    now = datetime.now(timezone.utc)
    chk = PHASE / "calculation.chk"
    files = [chk, PHASE / "gradient_hartree_per_bohr.npy",
             PHASE / "normal_mode_geometry_angstrom.npy", PHASE / "normal_mode_geometry_bohr.npy",
             PHASE / "input.xyz", PHASE / "progress.json", PHASE / "hessian_cache_manifest.json",
             PHASE / "rhf_helper_cache_manifest.json", OLD / "cpu_budget.json", OLD / "execution_lock.json",
             FORENSIC / "audit.json", FORENSIC / "forensic_release.json",
             ROOT / "scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py",
             ROOT / ".runtime/quantum-upgrade-site/pyscf/data/nist.py",
             ROOT / ".runtime/quantum-upgrade-site/pyscf/gto/mole.py"]
    before = {rel(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in files}
    forensic = read(FORENSIC / "audit.json")
    prior_inventory = {x["path"]: x for x in forensic["binary_byte_inventory_no_HDF_NPY_parse"]}
    checks = {"checkpoint_matches_pre_parse_forensic_byte_hash": sha(chk) == prior_inventory[rel(chk)]["sha256"]}
    progress = read(PHASE / "progress.json")
    manifest = read(PHASE / "hessian_cache_manifest.json")
    binding = manifest["binding"]
    rhf = read(PHASE / "rhf_helper_cache_manifest.json")
    ledger = read(OLD / "cpu_budget.json")
    target = next(x for x in ledger["phases"] if x["phase"] == "spinosin-pure-j-frequency-001")
    lock = read(OLD / "execution_lock.json")
    report = {
        "schema": "research4-abandoned-scf-component-readonly-audit-v1",
        "observed_at_utc": now.isoformat(),
        "observed_at_client_time": now.astimezone(ZoneInfo("America/Los_Angeles")).isoformat(),
        "scope": "Saved SCF identity and saved gradient verification only; no new electronic-structure evaluation",
        "authorization": "Root follow-up explicitly allowed read-only h5py/NumPy parsing of this abandoned state",
        "forensic_release_sha256": sha(FORENSIC / "forensic_release.json"),
        "new_quantum_evaluations": 0, "gto_loads_or_repr_evaluation_used": False,
        "SCF_energy_or_convergence_residual_independently_evaluated": False,
        "old_phase_status_not_changed": target["status"],
        "old_phase_final_wait4_CPU_seconds": target.get("actual_cpu_seconds"),
        "old_phase_exit_code": target.get("exit_code"), "old_phase_ended_at_utc": target.get("ended_at_utc"),
        "whole_phase_marked_DONE": False, "Hessian_restored": False,
        "IR_intensities_status": "NOT_COMPUTED", "source_files_before": before,
        "software_versions": {"numpy": np.__version__, "h5py": h5py.__version__},
    }
    errors = []
    atom_rows = []
    try:
        nist = ROOT / ".runtime/quantum-upgrade-site/pyscf/data/nist.py"
        mole_source = ROOT / ".runtime/quantum-upgrade-site/pyscf/gto/mole.py"
        bohr = numeric_source_literals(nist, ["BOHR"])["BOHR"]
        layout = numeric_source_literals(mole_source, ["CHARGE_OF", "PTR_COORD", "ANG_OF", "NCTR_OF"])
        with h5py.File(chk, "r") as handle:
            for key in ["mol", "scf", "scf/e_tot", "scf/mo_coeff", "scf/mo_energy", "scf/mo_occ"]:
                if not isinstance(handle.get(key, getlink=True), h5py.HardLink):
                    raise ValueError("Only local hard-linked checkpoint datasets/groups are allowed")
            raw_mol = handle["mol"][()]
            text = raw_mol.decode("utf-8") if isinstance(raw_mol, bytes) else str(raw_mol)
            molecule = json.loads(text)
            energy = np.asarray(handle["scf/e_tot"][()])
            arrays = {"MO_coeff": np.asarray(handle["scf/mo_coeff"][()]),
                      "MO_energy": np.asarray(handle["scf/mo_energy"][()]),
                      "MO_occ": np.asarray(handle["scf/mo_occ"][()])}
        atm = np.asarray(molecule["_atm"], dtype=np.int64)
        bas = np.asarray(molecule["_bas"], dtype=np.int64)
        env = np.asarray(molecule["_env"], dtype=np.float64)
        if atm.shape != (75, 6) or bas.ndim != 2 or bas.shape[1] != 8 or not np.isfinite(env).all():
            raise ValueError("Unexpected finite serialized atom/basis/environment tables")
        ptr = atm[:, layout["PTR_COORD"]]
        if np.any(ptr < 0) or np.any(ptr + 3 > len(env)):
            raise ValueError("Invalid serialized coordinate pointers")
        coordinates = np.asarray([env[i:i+3] for i in ptr], dtype=np.float64)
        arrays["coordinates_bohr"] = coordinates
        comparisons = {}
        for name, value in arrays.items():
            actual = fingerprint(value)
            expected = binding[name]
            passed = actual == expected and np.isfinite(value).all()
            comparisons[name] = {"actual": actual, "published": expected, "passed": bool(passed)}
            checks["binding_" + name] = bool(passed)
        canonical = json.loads(text)
        if np.asarray(canonical.get("_ecpbas", [])).size or canonical.get("_ecp"):
            raise ValueError("ECP normalization is outside this C/H/O saved-state audit")
        marker = canonical["_env"][17]
        if not np.isfinite(marker) or int(marker) != marker or not 0 <= marker < 75:
            raise ValueError("Unexpected nonphysical rinv bookkeeping marker")
        canonical["_env"][17] = 0.0
        canonical_sha = hashlib.sha256(json.dumps(canonical, sort_keys=True, allow_nan=False).encode()).hexdigest()
        checks["canonical_molecule_matches_binding"] = canonical_sha == binding["molecule_identity"]["canonical_serialized_sha256"]
        checks["exact_recorded_SCF_energy"] = energy.shape == () and str(energy.dtype) == "float64" and float(energy) == binding["SCF_energy_hartree"] == progress["energy_hartree"]
        z = atm[:, layout["CHARGE_OF"]]
        elements = [entry[0] for entry in molecule["_atom"]]
        expected_z = {"H": 1, "C": 6, "O": 8}
        checks["atom_composition_C28_H32_O15"] = Counter(elements) == Counter({"C": 28, "H": 32, "O": 15})
        checks["element_labels_match_atom_table"] = np.array_equal(z, np.asarray([expected_z[x] for x in elements]))
        charge = molecule.get("charge")
        occ = arrays["MO_occ"]
        spherical_nao = int(np.sum((2 * bas[:, layout["ANG_OF"]] + 1) * bas[:, layout["NCTR_OF"]]))
        checks["charge_electrons_AO_and_restricted_occupations_consistent"] = bool(
            charge == 0 and int(np.sum(z)) - charge == 320 and np.sum(occ) == 320
            and set(np.unique(occ)) == {0.0, 2.0} and np.count_nonzero(occ == 2) == 160
            and spherical_nao == 762 and progress["electrons"] == 320 and progress["basis_functions"] == 762)
        checks["recorded_SCF_and_gradient_flags_successful"] = progress.get("scf_converged") is True and progress.get("final_gradient_thresholds_passed") is True
        checks["recorded_spin_singlet_consistent_with_restricted_saved_state"] = progress["settings"].get("spin_2S") == 0 and progress["settings"].get("multiplicity") == 1
        geometry_ang = np.load(PHASE / "normal_mode_geometry_angstrom.npy", allow_pickle=False)
        geometry_bohr = np.load(PHASE / "normal_mode_geometry_bohr.npy", allow_pickle=False)
        gradient = np.load(PHASE / "gradient_hartree_per_bohr.npy", allow_pickle=False)
        for name, value in [("geometry_angstrom", geometry_ang), ("geometry_bohr", geometry_bohr), ("gradient", gradient)]:
            checks[name + "_full_finite_75x3_float64"] = value.shape == (75, 3) and value.dtype == np.dtype("float64") and bool(np.isfinite(value).all())
            if not checks[name + "_full_finite_75x3_float64"]:
                raise ValueError("Invalid saved 75x3 array: " + name)
        checks["HDF_coordinates_equal_saved_bohr_array"] = np.array_equal(coordinates, geometry_bohr)
        checks["HDF_coordinates_equal_saved_angstrom_array"] = np.array_equal(coordinates * bohr, geometry_ang)
        checks["serialized_atom_coordinates_equal_environment_coordinates"] = np.array_equal(np.asarray([x[1] for x in molecule["_atom"]]), coordinates)
        inp = (PHASE / "input.xyz").read_text().splitlines()
        xyz = [line.split() for line in inp[2:] if line.strip()]
        xyz_symbols = [x[0] for x in xyz]
        xyz_coordinates = np.asarray([[float(v) for v in x[1:4]] for x in xyz])
        source_path = rel(Path(progress["input_xyz_source"]))
        checks["old_input_xyz_matches_prequantum_lock"] = sha(PHASE / "input.xyz") == progress["input_xyz_sha256"] == lock["input_sha256"][source_path]
        checks["old_atom_order_and_positions_preserved"] = bool(len(xyz) == 75 and xyz_symbols == elements and np.isfinite(xyz_coordinates).all() and np.max(np.abs(xyz_coordinates - geometry_ang)) <= 1e-10)
        n = np.linalg.norm(gradient, axis=1)
        statistics = {"component_rms_hartree_per_bohr": float(np.sqrt(np.mean(gradient**2))),
                      "component_max_hartree_per_bohr": float(np.max(np.abs(gradient))),
                      "atom_rms_hartree_per_bohr": float(np.sqrt(np.mean(n*n))),
                      "atom_max_hartree_per_bohr": float(np.max(n)),
                      "gradient_norm_hartree_per_bohr": float(np.linalg.norm(gradient))}
        gradient_comparisons = {}
        for name, value in statistics.items():
            expected = progress[name]
            passed = bool(np.isclose(value, expected, rtol=1e-12, atol=1e-14))
            gradient_comparisons[name] = {"recomputed": value, "recorded": expected, "passed": passed}
            checks["saved_gradient_statistic_" + name] = passed
        for filename, expected in [("gradient_hartree_per_bohr.npy", binding["input_evidence"]["gradient_npy_sha256"]),
                                   ("normal_mode_geometry_angstrom.npy", progress["normal_mode_geometry_angstrom_npy_sha256"])]:
            checks["file_hash_" + filename] = sha(PHASE / filename) == expected
        checks["saved_gradient_passes_original_stationarity_thresholds"] = statistics["atom_rms_hartree_per_bohr"] < 0.0003 and statistics["atom_max_hartree_per_bohr"] < 0.00045
        report.update(
            array_bindings=comparisons,
            molecule_identity={"canonical_sha256": canonical_sha, "published_canonical_sha256": binding["molecule_identity"]["canonical_serialized_sha256"],
                               "normalization_scope": "Only serialized copy _env[17]=0.0, same frozen V8 rule; original molecule never modified",
                               "saved_rinv_marker": marker, "serialized_charge": charge,
                               "spin_field_present_in_mol_JSON": "spin" in molecule,
                               "serialized_spin_value": molecule.get("spin"),
                               "recorded_spin_2S": progress["settings"]["spin_2S"],
                               "spin_scope": "No new spin/state computation; recorded singlet input and 0/2 restricted occupations are consistent",
                               "formula": "C28H32O15", "atoms": len(elements), "electrons_from_atoms_and_charge": int(np.sum(z)) - charge,
                               "electrons_from_occupations": float(np.sum(occ)), "doubly_occupied_orbitals": int(np.count_nonzero(occ == 2)),
                               "spherical_AO_count_from_saved_basis_table": spherical_nao,
                               "basis_repr_kept_as_data_not_interpreted": molecule.get("basis")},
            saved_energy_hartree=float(energy), recorded_energy_hartree=progress["energy_hartree"],
            saved_gradient_element_count=int(gradient.size), saved_gradient_statistics=gradient_comparisons,
            geometry_comparison={"maximum_bohr_difference": float(np.max(np.abs(coordinates - geometry_bohr))),
                                 "maximum_angstrom_difference": float(np.max(np.abs(coordinates * bohr - geometry_ang))),
                                 "maximum_XYZ_print_roundoff_angstrom": float(np.max(np.abs(xyz_coordinates - geometry_ang))),
                                 "bohr_to_angstrom_from_actual_installed_source": bohr},
            checkpoint_dataset_inventory={"mol": "UTF-8 JSON parsed with json.loads only", "scf/e_tot": "float64 scalar",
                                          "scf/mo_coeff": "762x762 float64", "scf/mo_energy": "762 float64 orbital energies, not vibrational frequencies",
                                          "scf/mo_occ": "762 float64 restricted orbital occupations"},
            unavailable_checks={"SCF_equation_residual_or_independent_electronic_energy": "NOT_RUN",
                                "overlap_integral_MO_orthogonality": "NOT_RUN_NO_INTEGRALS",
                                "reconstructed_actual_grid_payload": "NOT_RUN_GRID_ARRAYS_NOT_STORED_IN_CHECKPOINT",
                                "Hessian_or_frequency_restoration": "NOT_AVAILABLE_NO_COMPLETE_CACHES"},
        )
        for i, symbol in enumerate(elements):
            atom_rows.append({"atom_index_zero_based": i, "old_XYZ_symbol": xyz_symbols[i], "checkpoint_symbol": symbol,
                              "nuclear_charge": int(z[i]), "x_bohr": coordinates[i,0], "y_bohr": coordinates[i,1], "z_bohr": coordinates[i,2],
                              "x_angstrom": geometry_ang[i,0], "y_angstrom": geometry_ang[i,1], "z_angstrom": geometry_ang[i,2]})
    except Exception as exc:
        errors.append({"exception_type": type(exc).__name__, "message": str(exc), "original_file_repaired": False})
    after = {rel(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in files}
    checks["all_read_source_bytes_unchanged"] = after == before
    checks["HDF_checkpoint_SHA_unchanged_before_after"] = after[rel(chk)]["sha256"] == before[rel(chk)]["sha256"]
    checks["original_processes_still_absent"] = all(not Path(f"/proc/{pid}").exists() for pid in (18564,18565))
    checks["no_available_complete_RHF_or_Hessian_cache"] = not rhf.get("arrays") and not manifest.get("partial_hessian") and not manifest.get("H1AO_atoms") and not manifest.get("response_blocks")
    checks = {name: bool(value) for name, value in checks.items()}
    verified = not errors and bool(checks) and all(checks.values())
    report.update(status="SCF_COMPONENT_VERIFIED" if verified else "UNVERIFIED_SAVED_STATE_COMPONENT",
                  all_checks_passed=verified, checks=checks, errors=errors,
                  source_files_after=after,
                  restart_recommendation={"this_checkpoint_as_fresh_SCF_initial_guess": "VERIFIED_COMPONENT_CANDIDATE" if verified else "NOT_RECOMMENDED_WITHOUT_FURTHER_CHECK",
                                          "checkpoint_sha256": before[rel(chk)]["sha256"],
                                          "old_closed_native_optimum_checkpoint_backup": "aae612aca345048028f85bce27f647c01f0834da1f9d06ec6ca986c0a2fffe8c",
                                          "new_round_requires_fresh_SCF_and_gradient": True,
                                          "new_healthy_snapshot_required": True,
                                          "old_abandoned_phase_must_not_be_marked_DONE": True,
                                          "old_Hessian_resume_available": False},
                  new_SDF_loaded=False)
    write_new(OUT / "audit.json", report)
    if atom_rows:
        with (OUT / "old_atom_order_and_geometry.csv").open("x",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(atom_rows[0]));writer.writeheader();writer.writerows(atom_rows)
    (OUT / "README.md").write_text(
        "# 旧SCF保存状态组件核验\n\n"
        f"观察时间（用户时区）：{report['observed_at_client_time']}。状态：**{report['status']}**。\n\n"
        "本次按照root明确授权，只读打开已不存在原进程的V4检查点，以h5py读取mol纯JSON及scf/e_tot、mo_coeff、mo_energy、mo_occ。没有调用gto.loads或解释atom/basis的repr字符串，没有SCF、电子积分、梯度或Hessian新求值。\n\n"
        "核验覆盖：762×762系数和762个轨道能／占据数的完整float64 payload SHA、75×3 Bohr坐标、分子JSON规范化身份、保存总能与旧binding及progress、C28H32O15/75原子/320电子、160个双占据轨道及762球谐AO。只在序列化副本中按旧V8规则规范化_env[17]非物理记账标志，不修改任何原数据。HDF mol JSON没有显式spin字段；旧记录指定spin=0，限制性0/2占据与该输入一致，未编造JSON自带的spin或独立自旋计算。\n\n"
        "另只读加载旧gradient、Bohr/Angstrom normal_mode_geometry NPY（allow_pickle=False），实际均75×3即225个元素；完整SHA、坐标及梯度统计与原记录比对。旧input.xyz还匹配量子计算前锁定的字节SHA；原子顺序与检查点坐标保持一致，未载入新SDF替换它。normal_mode_geometry只是几何，不是已得到的正常模式。\n\n"
        "**SCF_COMPONENT_VERIFIED不等于整个旧阶段DONE。** 它证明保存的数据组件与原已记录SCF状态一致，不是新做SCF收敛或电子能／残差复算。网格实际数组未存于检查点，未重建或验证网格payload；未计算重叠积分验证MO正交性。旧最终wait4、退出码、停止时刻仍未知，旧ledger保持原样。完整RHF／PBE Hessian及振型／频率仍不可恢复，IR强度NOT_COMPUTED。\n\n"
        "如组件核验通过，此checkpoint可作为本轮fresh SCF的初猜；仍要求新fresh SCF／梯度通过并创建健康新snapshot，不能为了省几分钟而放宽clean-parent恢复门槛。原V3已闭合native优化checkpoint（AA哈希）继续作为备用。所有原始文件在读取前后的字节SHA必须相同；损坏时只标UNVERIFIED，不修原件。\n"
    )
    entries={}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "files_sha256.json":
            entries[path.name]={"bytes":path.stat().st_size,"sha256":sha(path)}
    write_new(OUT / "files_sha256.json",{"scope":"Paths relative to saved-state-audit; self excluded", "files":entries})
    print(json.dumps({"status":report["status"],"check_count":len(checks),"errors":errors,
                      "checkpoint_sha256":before[rel(chk)]["sha256"],"index_sha256":sha(OUT / "files_sha256.json"),
                      "new_quantum_evaluations":0},ensure_ascii=False))


if __name__ == "__main__":
    main()
