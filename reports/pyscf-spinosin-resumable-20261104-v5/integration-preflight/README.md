# V5 adapter integration — pure engineering evidence

The resource supervisor was versioned separately to fix its process-exit race.
This adapter task did not perform a budget migration or launch any SCF,
gradient, Hessian or frequency calculation.

The three new adapters preserve the previous scientific code through explicit
mechanical substitutions. They use the V5 supervisor proof/ledger schemas and
new dependency paths/SHAs. The consumer's save-only and default RHF cache paths
use `outputs/pyscf-spinosin-resumable-20261104-v5/`. Scientific input/snapshot/
result/execution/gate/archive schemas remain v1. The exact PBE calculation,
grids, basis, MO space, gradient conditions, RHF rows, RKS XC, H1, CPHF and normal
mode analysis have no changed scientific logic.

| Source | SHA-256 |
|---|---|
| V5 resource supervisor | `33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee` |
| `scripts/pyscf_molecular_state_worker_20261104_v5.py` | `be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8` |
| `scripts/pyscf_resumable_hessian_worker_20261104_v5.py` | `b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190` |
| `scripts/run_pyscf_serial_segments_20261104_v5.py` | `a01b6a79f457bca4b0ab90e63b45a3bb524b62fc586fab26905d6b43a2aba1d7` |
| Unmodified incremental RHF core | `ce96055f24b39833337844c178f94597a9ac4ba7721ba019816d03628f971f55` |

`mechanical-copy-manifest.json` records the initial exact replacements, their
occurrence counts, old hashes and new hashes. For all three production scripts,
reversing the explicitly allowed substitutions restores the old complete text
and AST exactly. An independent mathematical reviewer actually repeated this
check rather than accepting the manifest's claims: 21 mappings, 31 occurrences,
76 function definitions covered by the full-file comparison. Old V4 state,
consumer and controller files, the frozen V8 and the RHF core are untouched.

Two intentionally retained V4 labels are not resource protocol claims:

- `SCREENED_DIRECT_RKS_V4` names the frozen V8 scientific SCF backend.
- `.runtime/pyscf-v4-global.lock` is the shared V4/V5 global exclusion lock.

The three copied pure test suites received four additional protocol-isolation
tests. Together with the V5 supervisor suite, all 93 actual pure tests passed:
24 supervisor, 31 state, 21 consumer and 17 controller. The final source/test
hashes, actual timings, command, exit code and full log are recorded in
`unittest-v5-integration-receipt.json` and `unittest-v5-integration.log`.
The initial mechanical-copy receipt precedes those four test additions; final
test hashes in the actual run receipt are authoritative for the augmented tests.

V5 adapters reject old V4 proof/ledger schemas and old state-producer identity.
They must receive newly computed V5 source SCF/gradient snapshots and current
boot gradient receipts. Old V4 outputs remain historical scientific evidence;
they are not relabelled as healthy V5 producers. New source checks do not relax
the existing stock-Hamiltonian/operator gate.

Root owns the separately reviewed ledger migration: preserve the existing
consumed CPU, original cap and original wall deadline. This task neither resets
nor expands them. Actual V5 validation and molecular calculations remain
**NOT_RUN by this adapter integration task**. Root may launch them only after
the complete guard/adapter review release, with new output directories.
