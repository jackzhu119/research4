# Closed V4 pure-J analyzer preflight

Status: FROZEN_ANALYZER_PREFLIGHT_ONLY. No final analysis export, Spinosin array/checkpoint read, new quantum evaluation, or results document was performed by this preflight. The large-molecule frequency phase is owned by root and remained outside these fixtures. Only root may issue the final release after all three frozen stages have actually ended.

The new `scripts/analyze_pyscf_pure_j_results_20261101.py` is frozen at SHA-256 `5d9544a11f17a2e45d8b4ffed221da97e99c1aa157c8c7aa59ee1d809f13aa3d`. Its tests are frozen at `a64d4ecc65d9950f2ee5b69cb0668ea898f6e5d7eae3baad8b0f3b28ef2b3f76`. Independent source review passed. All old workers, supervisors, ledgers, spectra, models and reports remain unchanged.

## Actually executed verification

The 22 lightweight tests passed. They use explicitly synthetic CSV/JSON and tiny mode arrays, never a new electronic-structure calculation. Coverage includes successful and rejected closed-release gates before any `np.load`, exact fixed stage order and commands, finite signed frequencies, missing CSV as empty evidence rather than zero absorption, complete positive 219-mode qualification, real physical-method flags, no-RHF-early-cache promotion, global full-response metadata, false canonical binding hashes, normalized mode signs/degenerate subspaces, all20×all-qualified-mode pairing and separate actual CPU pools with disclosed overruns.

`replay_closed_controls_only.py` was actually executed on the final frozen analyzer. Its inputs are exclusively the two closed water/ethanol rows in `gate-prefix-snapshots/both-validations-closed-ledger.json` and their saved software-control files. The helper verified historical parent proof/log hashes, exact geometry and fresh-gradient statistics, saved raw/symmetric Hessians and mode subspaces. It builds a Mole object and re-diagonalizes saved matrices; it does not evaluate SCF, gradients, integrals or a new Hessian. Both water3 and ethanol21 modes passed. Raw stock/pure-J differences remain 1.2434497875801753e-14 and 2.842170943040401e-14 Hartree/Bohr². Original case timings remain untouched, including the ethanol pure-J CPU increase.

`preflight_result.json` records actual commands, timestamps, exit codes and wait4-accounted process-tree cost: tests0.187173 CPU seconds and closed-control replay0.603366 CPU seconds, total0.790539. `closed_control_saved_matrix_replay.json` includes actual final source SHA and all read control hashes. An earlier exploratory helper execution used draft SHA c096358... and was reported separately in tool output; it is not relabeled as this final-source execution.

## Final root-release and export interface

The sibling `root-analysis-release.template.json` is deliberately non-executable: its allow flag is false and its date/final ledger hash are unset. Root must fill an actual timezone-aware date, explicitly allow export and bind the final unchanged ledger SHA after all water, ethanol and Spinosin stages close. A prefix ledger is not a final release. The release additionally binds the pre-quantum execution lock, actual pure-J entry, actual-CPU supervisor and final analyzer sources. Unknown/RUNNING/unended stages, missing stages, changed commands, helper source drift and unreviewed budget violations are rejected before any molecular array read.

```sh
.runtime/bin/python scripts/analyze_pyscf_pure_j_results_20261101.py \
  --run-root reports/pyscf-spinosin-pure-j-20261101-v4 \
  --output reports/pyscf-spinosin-pure-j-20261101-v4/final-analysis \
  --release ROOT_FINAL_ANALYSIS_RELEASE.json \
  --report docs/PYSCF_SPINOSIN_PURE_J_RESULTS_20261101.md
```

The main API is `run(root, output, release_path, report_path=None, budget_validator=None)`. It does not acquire flock locks; a root-owned finisher may hold the official V4/prior-V3 locks across the subprocess. All actual inputs are hash-checked again before export. Existing output/report files are never overwritten. Only the requested new results document is accepted by `--report`.

## Scientific and cost acceptance

Actual signed CSV frequencies survive separately even if source minimum, parent, mode or resource checks fail. Qualified Spinosin modes require a healthy actual V4 parent, the same-method completed old V3 optimization, a current real fresh SCF/gradient, a complete global PBE Hessian, full H1/75-atom B8 response metadata, raw/symmetric matrix replay, complete219 real positive modes and matching real mode subspaces. V8 scaffold provenance is preserved alongside the genuine V9 entry; no old script field is relabeled. An RHF e1/ej/zero-ek tuple is an incomplete pre-XC intermediate, never a full PBE Hessian. Cache restoration stays NOT_IMPLEMENTED_SAVE_ONLY. No uncomputed intensity is invented.

Only qualified Spinosin modes in0.1–1.5 THz enter the full Cartesian comparison with all20 immutable candidates from the fixed V3 final-analysis CSV. No fitted frequency scaling or nearest-peak selection is performed. With no qualified modes the pair table has zero actual pairs; the independent experiment table still has all20 rows. Empty/uncomputed theory is not zero absorption. Animations only show actual qualified modes with arbitrary visual amplitude. IR remains NOT_COMPUTED and chemical attribution remains unconfirmed.

All original small-molecule timing cases and actual V4 CPU/guard/configured limits are retained. Closed historical V3 actual111023.488913 seconds and V2 actual14846.81297 seconds are disclosed separately and summed only with an explicit V2+V3+V4 scope excluding the original pilot and nonquantum tests. V4 is an independent36-CPU-hour pool, not a claim that the old48-hour allocation still applies. Actual overruns and resource failures are not rounded away. No75-atom speedup is inferred from control timings or the previous failed calculation.

The isolated, single-conformer, PBE/def2-SVP/grid1/no-D3 model cannot represent the full solid medicinal matrix, lattice or collective interactions. Pure-compound and chemical measurements are still needed. The96.0234% classifier is unchanged.
