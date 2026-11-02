# ethanol: independent closed-fixture validation

PASS: NumPy replay compared all 729 raw and symmetrized Hessian entries, all 21 modes and every atom-pair coupling. Maximum raw difference 2.8421709430404e-14 Eh/Bohr², frequency difference 2.65458766079973e-11 cm⁻¹. Both full spectra are positive; independent projected-matrix frequencies, mode normalization and eigen-equation residuals pass. Stock ek has 729 nonzero entries; pure-J ek is exactly zero. e1/ej, fresh/post gradients, same-MF geometry/orbital/grid bindings and saved cache file/payload hashes were checked.

Parent is closed EXITED0 without resource-stop flags; actual wait4 CPU 107.86994 s. This is a software/numerical fixture, not a medicinal-plant or Spinosin result. IR NOT_COMPUTED. No SCF, integral, gradient/Hessian evaluation, PySCF import or checkpoint read occurred in this audit. Existing data and frozen source were unchanged. Matrix replay uses independent Cartesian translation/rotation QR projection and NumPy eigendecomposition. Source-file hashes and all numeric checks are in matrix_replay.json; frequency rows are in frequencies_replayed.csv. Physical accuracy on Spinosin and general speedups are not established by these small controls.

## Single paired timing observation

| Case | CPU seconds | Wall seconds |
|---|---:|---:|
| Stock | 49.905362365000 | 24.358710232002 |
| Pure J | 50.807067304000 | 24.158746960995 |

These are single, sequential paired measurements (stock first), not a general benchmark. Ethanol's observed CPU can increase while wall time falls slightly; neither this nor the water observation proves large-molecule acceleration. A future Spinosin attempt is justified by numerical equivalence, earlier RHF checkpoints, stricter independently monitored actual CPU and a separately finite budget. V3 did not finish a full Spinosin Hessian, so its failed duration cannot be used as a completed baseline to calculate speedup.
