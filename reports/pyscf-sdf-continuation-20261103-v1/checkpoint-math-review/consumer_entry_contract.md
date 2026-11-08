# Consumer entry qualification required by this review

The static/pure PASS is limited to the reviewed exact-state factory and the
ordinary nonrelativistic RKS/stock Hessian workflow. It is not a safety claim
for arbitrary user-decorated mean-field objects.

The incremental module binds Mole/MO/grid/source arrays and numerical settings,
but does not independently fingerprint every possible core-Hamiltonian method
override. In particular, the same MO values manually attached to an MF with
X2C, solvent, QM/MM, pseudopotential, external-field, or custom `get_hcore` /
`hcore_generator` behavior must not consume old cached E1 rows.

Before actual quantum use, the root consumer must qualify and record its
construction path: the frozen `restore_snapshot` factory, its reviewed plain
screened-direct RKS implementation, and the original PySCF Hessian. Reject
`with_x2c`, `with_solvent`, `mm_mol`, nonempty pseudopotentials, or other
decorators. Check the relevant core method origins/identities (including the
gradient core generator reached by stock Hessian), or enforce and bind an
equivalently strict reviewed factory with no instance monkeypatching. Record
that entry qualification alongside the actual source and consumer proofs.

This is an external entry gate, not a change to the frozen incremental algebra.
The consumer must additionally verify all unit producer supervision evidence;
the store only records that evidence. Real same-state and separate-process
water/ethanol full-Hessian and all-mode gates remain mandatory before Spinosin.

No quantum calculations were run by this independent reviewer. The earlier
review receipts and first file index remain unchanged. This addendum narrows
the interpretation of the engineering PASS and is indexed by
`files_sha256_v2.json`.
