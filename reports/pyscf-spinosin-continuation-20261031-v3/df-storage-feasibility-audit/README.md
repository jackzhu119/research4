# DF Hessian additional-storage feasibility

**DONE: source and local filesystem audit only. DF frequency execution remains
BLOCKED on local scratch capacity.** No quantum task, large allocation, mount,
external service or paid resource was started; no active checkpoint was read.

The current overlay has32 GB total capacity and about9.3 GB free. `/tmp` and
`/dev/shm` are separate tmpfs mounts with nominal8.8 GiB ceilings each; they both
consume the same existing16 GiB cgroup memory. They are not extra physical disk,
and their advertised capacities cannot be added as independent RAM. Using them
for tens of GiB would violate the memory envelope and the new cgroup guard.

The previously frozen PySCF2.14 source audit establishes an actual numeric
payload lower bound of46,734,874,272 bytes (43.525 GiB) for the DF H1 temporary
file. The earlier partial-Hessian `rho_ip1` stage needs34,267,286,304 bytes
(31.914 GiB). These file lifetimes are sequential, so they must not be summed
into a claimed simultaneous peak. Both independently exceed current scratch.
Deleting historical research files cannot make a32 GB overlay hold a43.5 GiB
H1 temporary; no historical data should be removed.

An existing external scratch volume with **at least64 GiB free** would be a
reasonable capacity target, allowing the43.5 GiB H1 data, a possible5.326 GiB SCF
DF cache, metadata and checkpoint/headroom. This is a recommendation, not a
provisioned or authorized paid service. More scratch would not resolve every RAM
and CPU issue; the existing16 GiB cgroup, numerical gates and resource supervisor
would still apply. Pointing `TMPDIR` at the present RAM-backed mounts does not
provide this capacity.

The reviewed built-in DF Hessian implementation has no configuration switch for
streaming away these global intermediates. Its H1 code writes and later reads
the all-AO `rho0_Pij` and `wj_ip1_pij` groups for real pure-PBE contributions.
The unused `vj1_buf` disk copy could save only1,045,159,200 bytes; the remaining
45,689,715,072-byte payload is still far too large. Exchange-only empty groups
are not the real payload in pure PBE. Reducing `max_memory` changes some block
sizes but not the global data dimensions. Disabling auxiliary-basis response
changes the derivative approximation and is not an equivalent resource fix.

Lossless HDF5 compression is mathematically compatible with the same values, but
there is no verified compression ratio for these dense floating-point tensors.
It cannot be assumed to compress enough to fit the available9.3 GB, and the
default source does not request such compression. A custom implementation which
recomputes or contracts blocks on demand may reduce storage, but it changes the
derivative implementation and requires independent small-molecule raw-Hessian
validation. No such streaming implementation or safe restore command was
created in this audit. Numerical precision reduction or dropping atoms/tensor
terms is not an equivalent shortcut.

The present actionable local path is therefore the separately gated direct
backend with explicit AS/RSS monitoring and real native-thread controls. The
current direct-gradient timings justify testing that engineering path; they do
not prove a future DF or direct Hessian completion time. A genuine64+ GiB scratch
volume would make the DF storage path worth a separate measured feasibility run.
