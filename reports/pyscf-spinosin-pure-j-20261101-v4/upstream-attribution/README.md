# PySCF upstream attribution for the V4 pure-J implementation

This directory preserves attribution for the PySCF-derived portion of
`scripts/pyscf_spinosin_pure_j_worker_20261101.py`. It does not relicense the
research4 repository, teacher-provided materials, datasets, or other code.

## Source and retained notices

The installed distribution is **PySCF 2.14.0**. Its `METADATA` identifies
`Apache-2.0` and names both `LICENSE` and `NOTICE` as license files. The local
installed `pyscf/hessian/rhf.py` has the copyright statement
**Copyright 2014-2019 The PySCF Developers. All Rights Reserved.**, the Apache
License 2.0 header, and an author credit to Qiming Sun. The distribution NOTICE
has its own copyright range and contributor list; it is retained verbatim.

- `PySCF-2.14.0-LICENSE.txt`: byte-for-byte copy of the installed LICENSE.
- `PySCF-2.14.0-NOTICE.txt`: byte-for-byte copy of the installed NOTICE.
- `source_header.txt`: exact initial copyright/license/author comment prefix
  of the installed `pyscf/hessian/rhf.py`, including its shebang and blank lines.
- `sources_sha256.json`: original local paths, byte lengths, SHA-256 hashes,
  installed license metadata, and the extraction rule.
- `output_index.json`: byte lengths and SHA-256 hashes of the attribution
  payload files; the index and its readback audit are deliberately excluded
  from their own payload hash list to avoid a circular hash dependency.
- `readback_verification.json`: independent file re-opening, copy equality,
  hash checks, and static source preservation checks at this snapshot.

The source function is **`pyscf.hessian.rhf._partial_hess_ejk`**, defined at
line 99 of the pinned installed `pyscf/hessian/rhf.py`. The local adapted
function is **`pure_j_partial_body`**, defined at line 123 of the V9 worker.
The upstream file SHA-256 is
`2d986bfc01aac98447034a207df639b3d6897f67ce1ab2a013fe2d0853c85de5`.
The local V9 worker SHA-256 is
`e3215e563b4429212084cfccbcec98e6a9f482bd8c3352d53d3064402649b450`.

## Local modification scope

The local body retains the upstream one-electron/core and Coulomb (J) algebra
for the reviewed pure-PBE path and omits exact-exchange (K) contractions whose
results are not consumed by that path. The wrapper delegates `with_k=True`
requests to the original implementation. The reviewed derivative path uses the
None-optimizer ABI; this snapshot does not establish support for an unreviewed
derivative-optimizer suffix path, hybrid functional, or arbitrary backend.
The frozen V8 assembly/scaffolding is reused separately. Original installed
PySCF and frozen V8 source files are not edited by this attribution operation.

This is a restricted implementation and resource-engineering adaptation of
existing PySCF routines, **not a new DFT functional or a new DFT method**.
Water/ethanol implementation-equivalence evidence belongs to the separate V4
validation artifacts. This static attribution snapshot does not itself run a
quantum calculation, establish Spinosin Hessian/frequency completion, provide
IR intensities, or confirm any chemical peak assignment.

## Official references and snapshot authority

- PySCF project: <https://github.com/pyscf/pyscf>
- Upstream license reference: <https://github.com/pyscf/pyscf/blob/master/LICENSE>
- Apache License 2.0: <https://www.apache.org/licenses/LICENSE-2.0>

These are reference links, not a claim that the moving upstream master branch
matches this installed source. No network retrieval was performed for this
snapshot. The installed distribution/source paths and SHA-256 values in
`sources_sha256.json` identify the exact preserved material.
