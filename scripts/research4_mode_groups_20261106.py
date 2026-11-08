"""Describe saved C001 normal-mode displacement participation; no quantum calls.

This is an additive analysis of the actual non-minimum 219-mode result.  Graph
labels describe molecular regions, not chemical assignments of herbal peaks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "reports/external-evidence-20261104-v1/mode-interpretation-preparation"
SCIENCE = ROOT / "reports/pyscf-spinosin-96h-continuation-20261105-v8/global-assemble-recovery-002/science/global-assemble"
CHECKS = SCIENCE.parent.parent / "independent-modes/checks.json"
RESULT_SHA = "253c10a92c083f54aaeb3f1e407e095790649737f5f876bd3cecfd35bfb4f9b7"
CHECKS_SHA = "b5edbd486f8e248d83efa556859f01cde080c849fddabd047649f7f5551601d0"
MASS = {"H": 1.008, "C": 12.011, "O": 15.999}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size}


def save(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0]); atoms = [line.split() for line in lines[2:2+n]]
    require(len(atoms) == n and all(len(a) == 4 for a in atoms), "Malformed XYZ")
    return [a[0] for a in atoms], [[float(v) for v in a[1:]] for a in atoms]


def csvwrite(path, rows, fields):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def no_running_quantum():
    ledger = read(ROOT / "reports/pyscf-spinosin-96h-continuation-20261105-v8/cpu_budget.json")
    require(not any(row.get("status") == "RUNNING" for row in ledger["phases"]),
            "This lightweight analysis is scheduled only outside an active quantum phase")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    start = time.time(); start_cpu = time.process_time()
    no_running_quantum()
    output = args.output.resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Use a fresh repository output directory")
    require(sha(SCIENCE / "result.json") == RESULT_SHA and sha(CHECKS) == CHECKS_SHA,
            "Sealed actual result or independent checks changed")
    seal = read(PREP / "files_sha256.json")
    for row in seal["files"]:
        require(sha(PREP / row["path"]) == row["sha256"], "Static graph metadata changed")
    audit = read(PREP / "atom_order_audit.json")
    lineage = read(PREP / "code_lineage_review.json")
    for row in audit["inputs"].values():
        require(sha(ROOT / row["path"]) == row["sha256"], "Static C001 lineage source changed")
    require(lineage["status"] == "PASS" and audit["source_and_C001_indexed_graph_signatures_equal"] is True,
            "C001 graph/order lineage is not qualified")
    result = read(SCIENCE / "result.json"); checked = read(CHECKS)
    require(result["normal_mode_count"] == 219 and result["imaginary_frequency_count"] == 25,
            "This analysis is bound to the actual complete non-minimum attempt")
    require(checked["qualified_for_mechanism"] is False and checked["new_quantum_evaluations"] == 0,
            "Do not promote independent mathematical replay to chemical qualification")
    require(checked["independent_numpy_replay"]["masses_amu"] == MASS,
            "Mass convention differs from independently replayed normal modes")
    require(result["stationary_minimum_verified"] is False, "Minimum qualification unexpectedly changed")
    groups = read(PREP / "atom_groups_C001.json")["groups"]
    with (PREP / "C001_atom_map.csv").open(newline="") as handle:
        atom_map = list(csv.DictReader(handle))
    n = 75
    require(len(atom_map) == n, "Expected 75 atom mappings")
    for i, row in enumerate(atom_map):
        require(int(row["C001_SDF_index_zero_based"]) == i
                and int(row["current_PySCF_index_zero_based"]) == i
                and int(row["C001_SDF_atom_number_one_based"]) == i+1
                and int(row["current_XYZ_atom_number_one_based"]) == i+1, "Index mapping changed")
    symbols, geometry = xyz(SCIENCE / "normal_mode_geometry.xyz")
    old_symbols, old_geometry = xyz(ROOT / audit["inputs"]["v5_prepared_xyz"]["path"])
    require(symbols == old_symbols == [row["element"] for row in atom_map], "Atom element/order mismatch")
    require(Counter(symbols) == {"C": 28, "H": 32, "O": 15}, "Wrong formula")
    sdf = (ROOT / audit["inputs"]["C001_source_sdf"]["path"]).read_text().splitlines()
    require(int(sdf[3][:3]) == n and [line[31:34].strip() for line in sdf[4:4+n]] == symbols,
            "Original graph atom order differs")
    partition = [g for g in groups if g["family"] == "DISJOINT_PARTITION"]
    require(len(partition) == 4, "Four disjoint molecular regions expected")
    coverage = []
    for group in groups:
        indices = group["indices_zero_based"]
        require(len(set(indices)) == len(indices) and all(0 <= i < n for i in indices), "Bad group index")
        require(group["indices_one_based"] == [i+1 for i in indices], "One-based index mismatch")
        if group["family"] == "DISJOINT_PARTITION":
            coverage.extend(indices)
            require(all(atom_map[i]["disjoint_partition"] == group["group_id"] for i in indices),
                    "Partition differs from sealed atom map")
    require(sorted(coverage) == list(range(n)), "Partition must cover all 75 atoms exactly once")

    # Import NumPy only after small immutable provenance gates have passed.
    import numpy as np
    arrays = {}
    array_refs = []
    for name, shape in [("normal_modes.npy", (219,n,3)), ("reduced_mass_amu.npy", (219,)),
                        ("normal_mode_geometry_angstrom.npy", (n,3))]:
        path = SCIENCE / name
        require(path.stat().st_size < 500000, "This task must not read a giant checkpoint")
        bound = checked["input_hashes"][str(path)]
        require(sha(path) == bound["sha256"], "Array differs from independently replayed input")
        arr = np.load(path, allow_pickle=False)
        require(arr.shape == shape and np.isfinite(arr).all(), "Invalid saved mode/geometry array")
        arrays[name] = arr; array_refs.append(reference(path))
    geo = arrays["normal_mode_geometry_angstrom.npy"]
    delta = float(np.max(np.abs(geo - np.asarray(old_geometry))))
    require(delta <= 1e-10 and np.allclose(geo, geometry, atol=1e-10, rtol=0),
            "v8 geometry does not match graph-order anchor")
    q = arrays["normal_modes.npy"]
    squared = np.sum(q*q, axis=2)
    cart_total = np.sum(squared, axis=1)
    require(np.allclose(arrays["reduced_mass_amu.npy"], 1/cart_total, atol=1e-10, rtol=1e-10),
            "Cartesian mode convention disagrees with independently checked reduced mass")
    masses = np.asarray([MASS[s] for s in symbols])
    weighted = squared * masses[None,:]
    mass_total = np.sum(weighted, axis=1)
    require(np.all(cart_total > 0) and np.all(mass_total > 0), "Zero saved mode")
    cart_fraction = squared/cart_total[:,None]
    mass_fraction = weighted/mass_total[:,None]
    cm = np.asarray(result["frequencies_cm1"]); thz = np.asarray(result["frequencies_thz"])
    require(cm.shape == (219,) and int(np.sum(cm < 0)) == 25 and int(np.sum(cm > 0)) == 194,
            "Signed frequency counts changed")
    group_rows = []; mode_rows = []; descriptions = []
    partition_sums = []
    for mode in range(219):
        group_values = []
        for group in groups:
            indices = group["indices_zero_based"]
            c = float(cart_fraction[mode,indices].sum())
            m = float(mass_fraction[mode,indices].sum())
            row = {"mode": mode+1, "frequency_cm1": float(cm[mode]), "frequency_THz": float(thz[mode]),
                   "imaginary": bool(cm[mode] < 0), "group_id": group["group_id"],
                   "family": group["family"], "atom_count": len(indices),
                   "Cartesian_displacement_squared_fraction": c,
                   "mass_weighted_displacement_squared_fraction": m,
                   "qualified_for_mechanism": False, "IR_intensity_status": "NOT_COMPUTED"}
            group_rows.append(row)
            if group["family"] == "DISJOINT_PARTITION": group_values.append((c,m,group))
        partition_sums.append([sum(v[0] for v in group_values),sum(v[1] for v in group_values)])
        top_cart = max(group_values, key=lambda v:v[0]); top_mass = max(group_values,key=lambda v:v[1])
        a = int(np.argmax(cart_fraction[mode])); ma = int(np.argmax(mass_fraction[mode]))
        mode_rows.append({"mode":mode+1,"frequency_cm1":float(cm[mode]),"frequency_THz":float(thz[mode]),
                          "imaginary":bool(cm[mode]<0),"dominant_Cartesian_region":top_cart[2]["group_id"],
                          "dominant_Cartesian_region_fraction":top_cart[0],
                          "dominant_mass_weighted_region":top_mass[2]["group_id"],
                          "dominant_mass_weighted_region_fraction":top_mass[1],
                          "largest_Cartesian_atom_zero_based":a,"largest_Cartesian_atom_element":symbols[a],
                          "largest_Cartesian_atom_fraction":float(cart_fraction[mode,a]),
                          "largest_mass_weighted_atom_zero_based":ma,
                          "largest_mass_weighted_atom_fraction":float(mass_fraction[mode,ma]),
                          "qualified_for_mechanism":False,"IR_intensity_status":"NOT_COMPUTED"})
        if cm[mode]<0 or 0.1<=thz[mode]<=1.5:
            label = "虚频诊断模式" if cm[mode]<0 else "未通过最低点资格构象的正频模式"
            descriptions.append(f"| {mode+1} | {cm[mode]:.8f} | {thz[mode]:.8f} | {label} | "
                                f"{top_cart[2]['description_zh']}，{top_cart[0]*100:.2f}% | "
                                f"{top_mass[2]['description_zh']}，{top_mass[1]*100:.2f}% | {symbols[a]}{a} |")
    partition_error = float(np.max(np.abs(np.asarray(partition_sums)-1)))
    require(partition_error < 1e-12, "Disjoint participation must sum to 1 in each convention")
    output.mkdir(parents=True)
    csvwrite(output/"all_mode_group_participation.csv",group_rows,list(group_rows[0]))
    csvwrite(output/"all_mode_dominant_regions.csv",mode_rows,list(mode_rows[0]))
    files=[reference(Path(__file__)),reference(SCIENCE/"result.json"),reference(CHECKS),
           reference(PREP/"files_sha256.json"),reference(PREP/"atom_groups_C001.json"),
           reference(PREP/"C001_atom_map.csv"),reference(PREP/"atom_order_audit.json"),
           reference(PREP/"code_lineage_review.json"),reference(SCIENCE/"normal_mode_geometry.xyz"),*array_refs]
    summary={"schema":"research4-saved-C001-mode-group-description-v1",
             "status":"DONE_DESCRIPTIVE_DISPLACEMENT_PARTICIPATION_NONMINIMUM_NOT_CHEMICAL_ASSIGNMENT",
             "created_at_utc":datetime.now(timezone.utc).isoformat(),"new_quantum_evaluations":0,
             "new_model_fits":0,"normal_modes":219,"positive_modes":194,"imaginary_modes":25,
             "stationary_minimum_verified":False,"qualified_for_mechanism":False,
             "chemical_attribution_verified":False,"IR_intensities_status":"NOT_COMPUTED",
             "theory_enters_classifier":False,"atom_order_basis":"C001_LOCAL_GRAPH_AND_ORDER_PRESERVING_CODE_LINEAGE",
             "unique_mapping_to_later_uploaded_SDF_claimed":False,"atom_count":75,
             "geometry_max_abs_delta_from_v5_anchor_angstrom":delta,
             "group_count":len(groups),"disjoint_regions":len(partition),
             "overlapping_groups_must_not_be_summed":True,
             "group_rows":len(group_rows),"partition_sum_max_abs_error":partition_error,
             "mass_convention_amu":MASS,
             "Cartesian_fraction_definition":"sum_group,a,xyz q_a,xyz^2 / sum_all,a,xyz q_a,xyz^2",
             "mass_weighted_fraction_definition":"sum_group,a,xyz m_a q_a,xyz^2 / sum_all,a,xyz m_a q_a,xyz^2",
             "fractions_are_not_IR_intensity_energy_or_experimental_chemical_amount":True,
             "mode_sign_and_plot_amplitude_arbitrary":True,
             "saved_animation_role":"Display diagnostic displacements only, not trajectories or evidence of stable oscillations",
             "selected_mode_001":mode_rows[0],"selected_mode_026":mode_rows[25],
             "selected_descriptive_modes":len(descriptions),"source_artifacts":files,
             "small_array_hash_basis":"Matches hashes recorded by actual independent Hessian/mode replay; original mode producer did not publish a mode NPY hash",
             "process_cpu_seconds":time.process_time()-start_cpu,"wall_seconds":time.time()-start,
             "all_source_and_historical_results_unchanged":True}
    save(output/"summary.json",summary)
    readme="""# C001完整模式的分子区域位移描述

本次实际读取已保存的219个正常模式，按已核验的旧C001图索引计算位移参与比例。全部25个虚频保留。本分析未运行量子计算或模型训练；稳定极小点与机制资格仍为false，IR强度仍NOT_COMPUTED。

## 定义与边界

- `Cartesian_displacement_squared_fraction` 为组内Cartesian位移平方和除以全分子位移平方和。
- `mass_weighted_displacement_squared_fraction` 在上述平方和中逐原子乘其质量；使用H=1.008、C=12.011、O=15.999 amu。
- 四个DISJOINT_PARTITION区域恰好覆盖75原子一次，两套比例分别合计1。其余OVERLAPPING_GRAPH_SUBGROUP会重叠，不能跨组求和为100%。比例没有按组原子数平均，因此大区域本身可能占更大份额。
- 这些值是指定向量/区域的位移描述，不是IR强度、振动能量占比、成分含量、实验峰归属或分类特征重要性。
- 组名来自原始C001共价连接图。它们不证明药材的低频峰属于该化合物；孤立分子也不表示晶格、分子间网络或真实药材。
- 相同元素序列本身不足以证明原子顺序。本次同时校验已封存的C001图、顺序保持的代码谱系、静态输入哈希及当前坐标与v5锚点逐项一致。没有强行给后上传SDF建立唯一索引映射。
- 保存的模式动画是人为显示幅度下的位移帧，不是实际分子动力学；虚频模式不代表可稳定振荡的吸收峰。模式符号任意，位移平方比例不依赖向量整体正负号。

## 产物

- `all_mode_group_participation.csv`：全部219模式、全部图区域的两套比例，保持频率符号。
- `all_mode_dominant_regions.csv`：每模式的最大区域/原子，零基文件索引。
- `summary.json`：实际哈希、几何核对、计算时间和资格限制。
- 下表仅描述25个虚频和0.1–1.5 THz内正频模式的区域位移比例；没有新增运动类型或化学归属判定。

| 模式 | cm⁻¹ | THz | 结果范围 | 最大Cartesian区域 | 最大质量加权区域 | 最大Cartesian原子（零基） |
|---|---:|---:|---|---|---|---|
"""+"\n".join(descriptions)+"\n\n重现命令（需要新的空输出目录）：\n\n```bash\nOMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\\nPYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site \\\n.runtime/bin/python -B scripts/research4_mode_groups_20261106.py --output reports/cloud-followup-20261106-v1/mode-groups-replay\n```\n"
    (output/"README.md").write_text(readme)
    save(output/"artifact_index.json",{"files":[reference(p) for p in sorted(output.iterdir()) if p.is_file()],
                                      "does_not_self_hash":True})
    print(json.dumps({k:summary[k] for k in ["status","normal_modes","imaginary_modes","group_count","group_rows","partition_sum_max_abs_error","process_cpu_seconds","wall_seconds"]}))


if __name__ == "__main__":
    main()
