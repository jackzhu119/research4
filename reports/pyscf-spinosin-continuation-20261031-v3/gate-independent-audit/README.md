# Independent audit of the completed V3 numerical gates

**PASS — closed water and ethanol engineering checks only. No new Spinosin vibrational results.**

The audit used NumPy and Python standard-library code to read already computed arrays. It imported no PySCF module, evaluated no SCF/gradient/integral/Hessian derivative, and read no active checkpoint. `checksum_crosscheck.json` includes hashing of the closed water checkpoint without loading it as a molecule or restart state. Gate sources and earlier results were preserved.

`matrix_replay.json` independently recomputes Cartesian external-motion projectors, mass weighting, projected eigenvalues and frequencies, mode orthogonality, eigen-equation residuals, reduced masses, raw-Hessian symmetry and all entries of the case comparisons. Its replay uses Cartesian translations/cross-product rotations rather than calling the producer's normal-mode function. Masses and physical constants match the versioned PySCF source; their source hashes are recorded.

| Ethanol case versus the same-MF native1 reference | Maximum raw-Hessian difference, Eh/Bohr² | Maximum frequency difference, cm⁻¹ | Maximum gradient difference, Eh/Bohr |
|---|---:|---:|---:|
| Full native1 reference | 0 | 0 | 0 |
| B4, native4 | 3.4722569708378614e-8 | 1.9401959605147567e-4 | 3.552713678800501e-15 |
| B8, native4 | 2.972314416815891e-8 | 8.415629520186485e-5 | 3.552713678800501e-15 |

All 81 atom pairs and all 729 Cartesian raw-Hessian entries were compared for each ethanol case. Energy differences are zero at the stored precision. The predeclared tolerances remain 1e-6 Eh/Bohr² for raw Hessians, 0.01 cm⁻¹ for frequencies and 1e-8 Eh/Bohr for gradients. All three ethanol cases have 21 strictly positive modes, approximately 225.04–3719.43 cm⁻¹. Independent replay differs from the saved frequencies by at most 3.87e-12 cm⁻¹. Mode orthogonality and projected eigen-equation residuals are at floating-point roundoff.

Water has three positive modes, approximately 1605.63, 3689.96 and 3789.04 cm⁻¹. Against the earlier, separately optimized V2 native1 control, maximum raw-Hessian and frequency differences are 7.362581598280293e-7 Eh/Bohr² and 0.0055188645392263425 cm⁻¹, inside the original tolerances. **That water comparison includes separate optimization runs; the ethanol comparison fixes the same geometry and MO state to isolate engineering changes.**

Fresh-gradient atom-RMS / atom-maximum values are:

- Water: 1.3637558592029502e-4 / 2.3043787845722363e-4 Eh/Bohr.
- Ethanol: 7.639664373555101e-5 / 1.5358569894336033e-4 Eh/Bohr.

Both satisfy the unchanged 3e-4 / 4.5e-4 criteria. `checksum_crosscheck.json` verifies 82 result/cache file bindings, finite shapes/dtypes and exact array payload hashes. It also confirms complete global atom coverage and actual 3000 MB MF/Hessian resource attributes in the captured Hessian manifests.

The independent provenance/log audit under `provenance-check/` verifies closed supervisor phases, proofs, source hashes, saved gradients and printed SCF/gradient records. Both phases exited successfully without resource kills: actual supervisor-accounted CPU is 5.378027 s and 182.105723 s, respectively. These include startup/cleanup, so they appropriately differ from worker-local process CPU. A `progress.json` can retain its historical PARTIAL snapshot; final result JSON, completion evidence and the closed ledger determine the completed state.

IR intensities remain **NOT_COMPUTED**. Passing these small-molecule gates supports trying the same implementation on Spinosin; it does not prove that a 75-atom Hessian will fit memory, finish within budget, have only positive frequencies, or identify experimental plant-spectrum peaks.
