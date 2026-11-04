# Closed-result analyzer source contract

Status: READ_ONLY_SOURCE_CONTRACT_REVIEW_CLOSED_FIXTURES_ONLY. This note reads the frozen V9/V8 source and explicitly closed water/ethanol JSONs only. It does not read any active Spinosin result, array, HDF, checkpoint, progress file or process; it does not run PySCF, NumPy matrix replay or quantum work. Expected Spinosin fields below come from source, not a claim about its current progress.

## Source identities and backend scope

- New entry: `scripts/pyscf_spinosin_pure_j_worker_20261101.py`, SHA `e3215e563b4429212084cfccbcec98e6a9f482bd8c3352d53d3064402649b450`.
- Frozen V8 scaffold: `scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py`, SHA `acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d`.
- In a Spinosin frequency result, `script_sha256` and cache `binding.worker_script_sha256` still identify the genuine imported V8 file. This is deliberate, not a wrong entry SHA. The new executable is identified by `actual_entrypoint.path/sha256`, `settings.actual_entrypoint`, `settings.engineering_worker_version` and `settings.partial_hessian_engine` (`PURE_PBE_J_PARTIAL_V9_SAVE_ONLY`). `original_script_sha256_field_scope` explicitly distinguishes the scaffold.
- V8 global-capture `engineering_version` remains `SUPERVISED_PARALLEL_DIRECT_V8_SAVE_ONLY`, because that class is unchanged. Do not require every legacy backend tag to say V9.
- Top-level `engineering_changes` records removed unused K scripts and retained J/core/fullspace. Native guard `density_or_operator_arguments_modified=False` applies only to that thread wrapper. `SCREENED_DIRECT_RKS_V4` is the SCF/CPHF policy, not evidence of derivative VHFOpt. The candidate retains the exact-build bare-symbol None path, does not enable suffix optimizers, and does not claim all libcint internal screening is absent.
- Small-molecule `settings` are intentionally the same V8 physical settings for both cases; identify the actual variant using case `pure_j_helper_enabled` and the RHF helper manifest. The top-level `engineering_changes_scope` labels candidate-only changes.

## Closed small-molecule layout and measured JSON facts

Root directories are `water-pure-j-equivalence-001/` and `ethanol-pure-j-equivalence-001/`. Each has `result.json`, `comparison.json`, `derived_input_provenance.json`, `same_MF_fingerprint.json`, and two case directories: `stock_partial_B8_native4/` and `pure_j_partial_B8_native4/`. Both supervisor-completion JSONs record exit0. Their independent reviews are `validation-independent-audit/{water,ethanol}/matrix_replay.json`.

| JSON quantity | Water | Ethanol |
| --- | ---: | ---: |
| Top-level status／validation_passed | DONE／true | DONE／true |
| All modes compared | 3 | 21 |
| Maximum raw-Hessian difference | 1.2434497875801753e-14 | 2.842170943040401e-14 |
| Maximum paired frequency difference, cm⁻¹ | 1.318767317570746e-11 | 2.6545876607997343e-11 |
| Stock ek nonzero entries | 81 | 729 |
| Stock maximum absolute ek | 0.9347190198091084 | 0.7754391219431511 |
| Pure-J ek nonzero entries／maximum | 0／0 | 0／0 |

Both cases have positive-frequency minima and restored stock helper identities according to the closed result JSONs; the independent NumPy reviews have status `PASS_INDEPENDENT_CLOSED_VALIDATION_NUMPY_REPLAY`. This note verifies their JSON schema and provenance fields only; it does not repeat that matrix replay. These are software controls, not plant samples or Spinosin modes.

Top-level validation result fields: `validation_only`, `validation_passed`, `numerical_equivalence_passed`, `stationary_minimum_verified`, `final_gradient_thresholds_passed`, `same_MF_geometry_MO_grid_retained`, `stock_ek_nonzero_and_pure_ek_exact_zero_verified`, `tolerances_fixed_before_results`, `comparison`, `resource_supervision` and `native_driver_controls`. `geometry_optimized=True` refers to the prior closed geometry; `new_geometry_optimization_performed=False` and `geometry_optimized_claim_scope` prevent counting a new Opt.

`comparison.differences` keys are `energy_hartree`, `gradient`, `e1`, `ej`, `raw_hessian`, `symmetrized_hessian`, `frequencies_cm1`, each with measured `max_abs_difference`, `fixed_tolerance` and `passed`. Full-mode/coupling checks are `all_actual_modes_compared`, `all_atom_pair_cross_couplings_compared`, `positive_frequency_minima` and `both_minima_positive`. Do not use only a success Boolean.

Each case result contains `raw_hessian_sha256`, `analysis_hessian_sha256`, `normal_modes_sha256`, `raw_e1_ej_ek_cache_manifest_sha256`, `RHF_ek_nonzero_entries`, `RHF_ek_max_abs_hartree_per_bohr2`, `stock_partial_function_object_and_source_restored`, actual modes/frequencies, `CPU_seconds` and `wall_seconds`.

## Three different completion layers

1. **RHF helper only:** `rhf_helper_cache_manifest.json` has `status=DONE_RHF_HELPER_ONLY_BEFORE_RKS_XC`, `scope=ONLY_RHF_PARTIAL_BEFORE_RKS_XC`, `complete_RKS_partial_or_full_Hessian=False`. Its `arrays.e1/ej/ek` each have path, file SHA, payload SHA, dtype, shape and byte counts. `complete_RHF_helper_returned_before_RKS_XC`, `actual_helper_MO_arguments_verified_equal_bound_MF`, ek counts and `binding_sha256` add evidence. This is not a complete RKS partial Hessian or frequencies.
2. **RKS/global intermediate capture:** `hessian_cache_manifest.json` has `partial_hessian` (RKS partial including XC), `H1AO_atoms`, `H1AO_complete`, `response_blocks`, `response_complete_global_atom_ids` and `native_global_kernel_returned_normally`. Successful capture status is `DONE_INTERMEDIATE_CAPTURE_ONLY`; that name does not itself certify vibrational frequencies. Validate all atom IDs and nested file hashes, not merely nonempty lists.
3. **Scientific full Hessian/modes:** actual raw/symmetrized matrices, mode vectors and frequency outputs must independently agree with the result and complete atom/mode counts. A completed RHF tuple or global intermediate manifest cannot substitute for these files. All caches retain `cross_process_restore=NOT_IMPLEMENTED_SAVE_ONLY`, with no resume claim.

Each binding includes `molecule_identity`, coordinate payload fingerprints, `MO_coeff`, `MO_energy`, `MO_occ`, SCF energy, actual grid coordinates/weights, CPHF tolerance, actual MF/Hessian memory settings, method/settings, versions, input evidence and scaffold SHA. Recompute each binding's own canonical SHA. **Stock and pure-J binding hashes need not equal each other**: their `input_evidence.pure_j_enabled` differs. Compare physical coordinate/MO/grid/state subfields for equality, while retaining the intentional variant metadata difference.

For every array record, file SHA hashes the `.npy` bytes; payload SHA hashes contiguous array C-order numeric bytes. These are different hashes and must not be compared to one another.

## Expected Spinosin closure schema, derived from source only

The Spinosin report root should contain:

| File／field | Meaning and acceptance boundary |
| --- | --- |
| `result.json: phase=spinosin-frequency` | No new optimization in that phase; its `optimizer_converged/geometry_optimized` originate from a verified closed parent, followed by fresh same-method SCF/G |
| `optimization_result_sha256` | Closed native non-DF parent result binding, not DF-preoptimization equivalence |
| `gradient_hartree_per_bohr.npy` | New fresh gradient; hash also appears in capture `binding.input_evidence.gradient_npy_sha256` |
| `normal_mode_geometry_angstrom.npy`, `normal_mode_geometry_bohr.npy`, `normal_mode_geometry.xyz` | Actual current analysis geometry; Angstrom NPY file SHA is `normal_mode_geometry_angstrom_npy_sha256` |
| `hessian_hartree_per_bohr2.npy` | Complete raw shape `(75,75,3,3)`; file SHA is `raw_hessian_sha256` |
| `hessian_used_for_analysis.npy` | Symmetrized raw after maximum raw transpose asymmetry ≤1e-5; file SHA is `analysis_hessian_sha256` |
| `normal_modes.npy` | Actual `(219,75,3)` real finite nonzero modes; this V8 output does not include a dedicated mode-file SHA field, so analyzer must compute and record it rather than invent an expected hash |
| `reduced_mass_amu.npy` | Actual harmonic reduced masses; compute its new analyzer SHA as needed |
| `frequencies.csv` and result `frequencies_cm1/frequencies_thz` | All219 signed values, no discarded negatives; THz factor0.0299792458 only |
| `normal_mode_count`, `expected_normal_mode_count`, `normal_modes_complete` | Exactly219 modes expected/observed, only after complete matrix/mode checks |
| `imaginary_frequency_count`, `all_frequencies_real_positive`, `stationary_minimum_verified`, `frequency_validation_passed` | Preserve distinct computed-frequency and minimum-validation outcomes |
| `IR_intensities_status` | AlwaysNOT_COMPUTED, not a theoretical absorption-intensity curve |

The source sets `frequencies_status=DONE` when complete modes were computed. If any nonpositive mode prevents a verified minimum, the final overall `status` becomes PARTIAL even though full signed frequencies exist. An analyzer must preserve this scientifically useful partial outcome; it must not delete frequencies, turn them positive, or conflate incomplete arrays with a completed nonminimum. Mechanism matching requires the root protocol's qualified complete positive-frequency gate.

`hessian_cache_manifest_sha256` is present in the Spinosin result. The RHF early manifest exists separately; this adapter does not add its SHA to the Spinosin top-level result, so verify that manifest's own binding, source identities and referenced files directly. `native_direct_source_sha256` records `_vhf`, `nr_direct.c` and `libcvhf_binary`; new entry and original scaffold identities remain separate.

## V3 parent compatibility is physical, not whole-dict equality

Frozen V8 `require_optimized_result` requires DONE, geometry/optimizer/SCF convergence, gradient gate, exact optimized XYZ SHA and optimized-coordinate NPY SHA. It compares the following method keys: `xc`, `basis`, `grid_level`, `dispersion`, `density_fitting`, `auxbasis`, `density_fit_backend`, `direct_scf`, `direct_scf_tol`, `scf_conv_tol_hartree`, `optimizer_thresholds`, `global_four_center_ERI_cache_allowed`, `charge`, `spin_2S`, `multiplicity`.

The closed V3 non-DF direct optimization is eligible. A DF result cannot silently substitute: its backend and approximation differ. V9 engineering identities, completed-equivalence evidence, V4 actual-CPU supervision and resource epochs are intentionally new and are not expected to equal a V3 whole settings/proof dictionary. A checkpoint is only a fresh-SCF warm start; no old Hessian restoration is implemented.

## Closed parent validation/resource review

For water/ethanol, the entry's `verify_completed_equivalence_records` also checks real parent rows in the V4 ledger. Required evidence includes unique phase, EXITED/exit0/endtime, correct current runner/proof source SHA, child PID and actual-CPU contract agreement, no CPU/memory/wall/monitor/hardkill/containment/residual stop, proof-file and log-file SHA matches, and command ownership (`pure-j-equivalence`, correct identity, new entry and output directory). Use only those closed rows; a separate active Spinosin row is not a reason to run the whole-ledger validator or inspect its arrays early.

Independent review JSON fields are `parent_phase`, `parent_exit_code`, `parent_actual_wait4_CPU_seconds`, `all_raw_and_sym_Hessian_components_compared`, `all_atom_pair_couplings_compared`, `max_abs_differences`, `fixed_tolerances`, `same_MF_geometry_MO_grid_bindings_equal`, `stock_ek_nonzero_pure_ek_exact_zero`, `masses_amu`, `frequency_factor_cm1` and `input_files`. Its per-case `stock/pure_j` records contain mode counts, all frequencies positive, independent replay differences, mass-weighted orthogonality/projection/eigen-equation residuals, ek observations and cache-SHA checks. These closed reviews add independent evidence but do not predict Spinosin runtime or guarantee its success.
