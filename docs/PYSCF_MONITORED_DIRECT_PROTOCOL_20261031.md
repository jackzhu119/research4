# Direct PySCF continuation with monitored physical memory

This is an additive engineering protocol for the user's explicit instruction to
continue long cloud calculations. It does not alter the completed DF optimization,
the earlier twelve-GiB-address-space batches, any spectra, or any classifier.
The new run root is `reports/pyscf-spinosin-continuation-20261031-v3/`.

## Reason for a separately recorded resource policy

The PySCF 2.14 native direct-integral driver reserves approximately 3.2 GB of
virtual address space per OpenMP thread. The C allocation is not a memset of
that entire reservation: the output AO tiles determine the pages actually
touched. A twelve-GiB `RLIMIT_AS` can therefore prevent a four-thread calculation
even when resident memory is small. Increasing the address-space allowance is
not adding RAM or assuming the calculation will fit.

The intended new supervisor has a twenty-four-GiB address-space ceiling and
monitors the complete payload process group's RSS at 0.1-second intervals. At
RSS above ten GiB it requests termination, then kills after at most one second;
at cgroup memory above twelve GiB it kills immediately. This is explicitly a
**BEST_EFFORT_RSS_GUARD**, not a kernel-enforced ten-GiB resident-memory limit.
The existing sixteen-GiB cgroup remains the ultimate kernel boundary. Rapid
native allocations can occur between observations. No writable child cgroup is
available, so a stronger per-job physical-memory guarantee is not claimed.

Only one quantum worker runs at a time. The new supervisor must hold the prior
batch lock, verify that all prior phases are closed, and bind the closed ledger's
hash and actual CPU total. Old CPU accounting is never reset or rewritten. The
new finite batch may authorize at most forty-eight CPU hours and twenty-four
wall hours, with at most four cores; individual phases also have finite limits.
No GPU or paid service is used. Actual authorization, limits, process monitoring,
termination and `wait4` CPU consumption belong in the new ledger, not estimates.

## Numerical qualification before the large molecule

The electronic method remains neutral singlet PBE/def2-SVP, grid level 1, without
dispersion. This is explicitly not PBE-D3 or B3LYP-D3. The direct RKS backend,
integral screening, SCF tolerance and physical derivative formulas remain fixed.
SCF nuclear gradients must converge; Hessian work uses the reviewed 3000-MB MF
and Hessian grid-block memory context. Changing grid block size does not change
the quadrature grid itself.

Before a large Hessian, execute complete water Opt/Freq and a nine-atom ethanol
benchmark. The latter compares the standard full-system native-one-thread
Hessian with native-four-thread response blocks of four and eight atoms on the
same geometry, SCF solution and complete electron response. Its fixed numerical
tolerances are energy/gradient `1e-8`, maximum raw-Hessian difference `1e-6`, and
frequency difference `0.01 cm^-1`. The full matrix, including inter-block
couplings, is compared. Numerical equivalence and a positive-frequency minimum
are separate checks. These validation molecules are not acid-jujube samples or
additional plant spectra.

## Spinosin acceptance and preservation

Use only the independently audited source structure, its complete stereochemical
graph and its exact full-precision coordinates. A prior checkpoint is an initial
density guess, not a substitute for a fresh SCF or a gradient. Preserve all
completed and interrupted attempts, with the real termination reason. In-flight
HDF5 checkpoints are not read by analysis.

Frequency analysis requires the matching method's completed optimization and a
fresh passing nuclear gradient. Save the real raw Hessian and its explicitly
symmetrized analysis copy, all signed frequencies and all normal-mode vectors.
There should be 219 vibrational modes for nonlinear 75-atom Spinosin. A completed
frequency calculation with a negative mode must not be called a verified
minimum. IR intensities remain `NOT_COMPUTED`.

Only actual qualified modes may enter the 0.1-1.5 THz comparison table. Frequency
proximity does not establish a constituent's identity in solid medicinal plant
material. The isolated-molecule calculation does not include a crystal lattice,
collective intermolecular motion or the complete matrix. No theoretical peak is
used to modify the existing 96.0234% classifier in this mechanism pilot.

Hessian intermediates are saved locally with hashes. Current checkpointing is
save-only: cross-process loading/recovery must not be advertised until separately
implemented and validated. Preparation, passing software tests, a running job,
and completed molecular calculations have distinct statuses.
