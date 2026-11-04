# Independent V5 worker integration review

**PASS**, 2026-11-04 UTC. Final production sources:

| Component | SHA-256 |
|---|---|
| V5 supervisor | `33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee` |
| V5 exact-state worker | `be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8` |
| V5 Hessian consumer | `b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190` |
| V5 serial controller | `a01b6a79f457bca4b0ab90e63b45a3bb524b62fc586fab26905d6b43a2aba1d7` |
| Unchanged incremental core | `ce96055f24b39833337844c178f94597a9ac4ba7721ba019816d03628f971f55` |

The three production adapters independently pass exact forward literal-copy,
inverse full-text, and inverse AST comparison against their frozen V4 sources.
Only the reviewed source paths/SHA values, proof/ledger version identifiers,
engineering text, and new ignored output namespace change. The numerical
model, wavefunction/grid restoration, hcore gates, row algebra, full response,
signed modes, cost limits and scientific completion logic are unchanged.
`SCREENED_DIRECT_RKS_V4` remains the physical implementation label, and
`.runtime/pyscf-v4-global.lock` remains the shared V4/V5 lock path.

All **93 pure tests passed independently**:

```bash
.runtime/bin/python -m unittest discover -s tests -p '*v5.py' -v
```

The actual V5 state factory / stock Hessian operator gate also passed a small
basis-metadata check with integral, SCF and gradient entrypoints mocked to
raise. This imports PySCF and constructs metadata; it does not perform an
electronic-structure calculation.

The initial mechanical-copy manifest predates four added tests rejecting V4
proofs, ledger schema and snapshot sources. Its original test-copy hashes are
historical. Production hashes match that manifest exactly; the final current
test hashes and independently executed log are in `receipt.json`. No historical
manifest was overwritten to hide this staged development.

The reviewer performed no QM, actual migration, Git operation, real supervisor
launch, process signal, or active molecular HDF/NPY read. Root's actual
migration and repeated water/ethanol/source calculations are separate records.
