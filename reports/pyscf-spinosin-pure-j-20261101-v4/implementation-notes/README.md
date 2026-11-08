# V9 pure-PBE partial Hessian implementation

Status: IMPLEMENTED_PENDING_ACTUAL_SMALL_MOLECULE_EQUIVALENCE. This note is not a quantum result. Only root launches real workers after the source/test review and through the separate finite V4 actual-CPU supervisor.

The new entry is `scripts/pyscf_spinosin_pure_j_worker_20261101.py`. It imports the real frozen V8 scaffold, checks its SHA, and retains its full-system Hessian assembly, H1, CPHF, B8 response batching, grid/memory strategy and native4 guard. It does not impersonate the V8 `__file__` or use a renamed V3 supervision proof. Actual entrypoint and frozen-scaffold hashes have distinct metadata fields; the actual V4 verifier is imported from `run_pyscf_actual_cpu_budget_20261101.py`.

The only scientific-engineering change is removal of unused K-shaped scripts and their density bindings when `with_k` is exactly False for non-DF plain PBE with zero hybrid/range-separated coefficients. Original J scripts, density slices, coefficients, core/overlap terms, atom ordering and symmetric copies are retained. The True path delegates the original function with its unchanged arguments and returns the original object. Other functionals and an ABI exposing any of the three bare derivative optimizer symbols are rejected for the False path. No suffixed derivative optimizer is enabled.

The exact source/build hashes are pinned for `rhf.py`, `rks.py`, `_vhf.py`, `libcvhf`, `libcint` and `libcgto`; loaded module/binary identities and the unpatched stock helper identity are checked. The `SCREENED_DIRECT_RKS_V4` tag remains a statement about SCF/CPHF direct execution, not a claim that derivative VHFOpt exists or that all libcint internal screening is absent.

## Completed RHF helper capture

`RHFPartsCapture` saves e1/ej/ek immediately after the complete RHF helper returns and before subsequent RKS XC work. Candidate ek is a finite float64 zero array with the original shape. Its separate manifest is named `rhf_helper_cache_manifest.json`, has scope `ONLY_RHF_PARTIAL_BEFORE_RKS_XC`, and never claims a completed RKS partial Hessian or full Hessian. It binds the same molecule, actual MO arrays and dtype, grid, method/settings, input evidence and source identities; explicit helper MO overrides must equal the bound MF. Earlier files are never overwritten.

If later XC/response/global assembly fails, a completed RHF tuple remains explicitly usable only as saved evidence. A helper that never returned remains incomplete. **Restoration is NOT_IMPLEMENTED_SAVE_ONLY**; no resume command is supplied, and no old V3 unfinished Hessian is loaded.

The native thread wrapper's `density_or_operator_arguments_modified=False` describes only that wrapper. The top-level `engineering_changes` separately declares that the candidate helper removes unused K scripts; stock reference cases explicitly record removal as False.

## Actual validation interface, not yet a result

Use `--phase pure-j-equivalence --identity water|validation_ethanol` with a closed successful optimization result, its optimized XYZ and complete-precision NPY, and explicit matching SHA values. No geometry optimization runs in this entry. Water binds its prior result's optimized-coordinate SHA. The legacy ethanol result additionally requires `--closed-geometry-record` pointing to its closed B8 case result, which already binds the optimized coordinate SHA. Its original verified graph provenance is retained, but the initial MMFF XYZ hash is never used as the optimized XYZ hash. New derived input provenance records the actual closed-optimization/coordinate lineage.

The actual calculation sequence is a fresh SCF and stationary gradient, followed by two complete Hessians on the same MF, coordinates, MOs and actual grid:

1. Stock PySCF partial helper plus the previously validated V8 B8/native4 full-system assembly.
2. Pure-J partial helper plus exactly the same assembly and resource settings.

Both tuples, complete raw/symmetrized Hessians, actual modes/frequencies, post-Hessian gradients and timings are saved. Every atom-pair entry and all three water or 21 ethanol modes are compared, including cross-block terms and signed negative frequencies. Fixed gates are energy/gradient 1e-8, e1/ej/raw Hessian 1e-6 and paired frequencies 0.01 cm⁻¹. The actual stock case must have nonzero unused ek, and the pure-J case must have exact zero ek; both early helper manifests record observed counts, maximum magnitudes and file hashes. This prevents two accidentally patched cases from being treated as the required control. Numerical equivalence and positive-frequency minima have distinct fields; both must pass for the overall validation gate. IR intensities remain NOT_COMPUTED. Small-molecule timings are not extrapolated into a Spinosin speedup.

The stock function object and its source identity must be restored after each case, including exceptional exits and rejected nested contexts. No third Hessian is needed to check function identity. MF/MO/grid drift blocks comparison.

`--phase spinosin-frequency` additionally requires `--validation-results <water-result> <ethanol-result>`. It checks exact new-entry and supervisor hashes, fixed tolerances, measured full-matrix/all-mode gates, successful closed EXITED0 parent ledger rows, actual-CPU contracts, matching proof/log hashes and output-command ownership. A self-reported DONE JSON or a failed/unclosed parent cannot permit the Spinosin phase. The frozen V8 parent-geometry and fresh SCF/gradient checks remain in force.

The entry accepts only PBE/def2-SVP/grid1/noD3, four outer/native threads and B8 in this round. The independent actual-CPU supervisor supplies finite resources; this code does not silently extend a budget or launch another quantum job.

## Verification scope

`tests/test_pyscf_spinosin_pure_j_worker.py` uses the frozen stock helper AST as an independent arithmetic reference with synthetic NumPy matrices and fake contraction callbacks. It does not import PySCF or compute integrals. Tests cover retained algebra, branch delegation, ABI restrictions, nested restoration, provenance, unchanged MF control flow, SAVE_ONLY capture and numerical/resource launch gates. They do not establish real numerical equivalence; actual water/ethanol results must remain separate.

The root protocol fixes the physical calculation and acceptance conditions. Original PySCF, V8, old budgets, raw data and classification results are unchanged. No DFT accuracy improvement or new DFT functional is claimed.
