# PySCF 虚频修复：实时计算结果

状态：PARTIAL；完成 1/4 个分子。

所有新计算为孤立分子 PBE/def2-SVP、明确无 D3、PySCF2.13.0、原生直接 SCF、grid5、完整网格响应梯度。
完整双步长数值 Hessian 与偶极导数来自逐坐标实际位移，保留有符号频率。完成状态与论文资格分别记录。

旧结果、模型结构来源及已知误差见 [虚频诊断](PYSCF_IMAGINARY_FREQUENCY_DIAGNOSIS_20261111.md)。

| 分子 | 细步长模式数 | 两步长虚频数 | 最低频率 cm⁻¹ | 最低频率 THz | 本模型谐振数据资格 |
|---|---:|---|---:|---:|---|
| water | 3 | [0, 0] | 1608.573533 | 48.22382133 | True |

## water

新鲜核梯度最大分量：2.155732e-09 Eh/Bohr；RMS：9.0276159e-10。
轨道稳定性实际检查通过：True。

机器可读完整核验：[该分子结果](../reports/pyscf_repair_20261111/jobs/water/result.json)；[全部有符号频率与 IR 强度 CSV](../reports/pyscf_repair_20261111/jobs/water/modes/all_modes.csv)。

数值资格（原样保留）：
```json
{
  "center_electronic_quality": {
    "native_converged": true,
    "fresh_orbital_gradient_norm": 5.7493578335781825e-09,
    "energy_recheck_difference_hartree": 0.0,
    "orbital_residual_passed": true,
    "energy_recheck_passed": true,
    "upstream_qualification_declared": true,
    "upstream_qualified": true,
    "upstream_qualification_passed": true,
    "electron_count_error_declared": true,
    "electron_count_error": 1.7763568394002505e-15,
    "electron_count_check_passed": true,
    "qualified": true,
    "nuclear_gradient_max": 2.1557320373233324e-09,
    "nuclear_gradient_rms": 9.027615911041408e-10,
    "nuclear_stationarity_passed": true,
    "orbital_stability_passed": true
  },
  "point_electronic_quality": {
    "metadata_present": true,
    "qualification_evidence": "INDIVIDUAL_ELECTRONIC_RECHECKS",
    "point_count": 36,
    "qualified_point_count": 36,
    "native_unconverged_point_count": 0,
    "all_points_qualified": true,
    "all_native_converged": true,
    "max_fresh_orbital_gradient_norm": 7.552491169271853e-09,
    "expected_central_difference_point_count": 36,
    "complete_point_quality_ledger": true
  },
  "dual_step_comparison_available": true,
  "raw_antisymmetry_passed": true,
  "projected_step_agreement_passed": true,
  "empirical_positive_spectral_margin_passed": true,
  "all_vibrational_eigenvalues_strictly_positive_at_all_steps": true,
  "stationary_geometry_verified": true,
  "nuclear_minimum_numerically_supported": true,
  "orbital_stability_passed": true,
  "minimum_verified": true,
  "paper_ready_molecular_harmonic_data": true,
  "thresholds_are_qualification_not_execution_stops": true,
  "scope": "Local isolated-molecule harmonic calculation at the stated method; no global-conformer minimum or solid-state chemical peak assignment is established."
}
```

## 论文使用边界

只有通过新鲜电子收敛、核驻点、轨道稳定性、全核空间正曲率及双步长一致性核验的结果，才标为本计算模型下的合格最低点。未通过项及负模式保留。
IR 强度为真实偶极导数计算的孤立分子谐振预测；频率不截负、不取绝对值。没有纯品或晶体模型验证，不能据此确认药材固体 THz 吸收峰，也不作为逐样品化学标签。
