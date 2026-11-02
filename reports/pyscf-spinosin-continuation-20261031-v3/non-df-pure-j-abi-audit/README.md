# Non-DF partial-Hessian symbol guard audit

This is an ABI-only lookup on the installed PySCF 2.14 library. It imports no
PySCF module and creates no molecule, optimizer, density, integral or Hessian.
`ctypes.CDLL` can execute ordinary shared-library initializers; no exported
numerical function is invoked. Only symbol resolution via `hasattr` is used.

For all three exact unsuffixed names requested by `rhf._make_vhfopt`, the installed
library handle reports `False`; their `_sph` and `_cart` variants report `True`.
Thus the current frozen helper returns `None` at rhf.py:182-183, before creating
the screening object, for ipip1/ip1ip2/ipvip1. The suffix distinction is real,
not inferred solely from nm on one library that might miss dependencies.

This removes the `vhfopt.set_dm` concern for these three current calls only.
It does not authorize repairing a different build's screening or removing
Coulomb terms. A pure-PBE candidate may omit unused exchange outputs only in
new code, after full water/ethanol Hessian and frequency equivalence tests.
No such candidate is implemented or numerically validated here. Shared integral
generation means removing outputs does not guarantee proportional time or
scratch-memory savings.

All library/source hashes and individual booleans are in `symbols.json`; the
index seals those results and this probe. No active calculation file was read.
The per-probe OMP/BLAS environment is recorded and does not change a running job.
