# V5 normal-mode postprocessor: final preflight and actual controls

**Spinosin target status: NOT_RUN.** No current Spinosin molecular array or
frequency was read by this task. The empty target preflight contains no fake
frequency table. Source/model/data from previous experiments are unchanged.

Final analyzer:
`scripts/analyze_spinosin_v5_modes_20261104.py`

SHA-256: `a2e3588c6dea121be58e4a69afb137bc6e5092199ca4180937fcdb34f6f57154`.

Nineteen meaningful pure tests passed. The independent reviewer also passed
these tests and checked the final actual control exports. The preceding
`modes-analysis-preflight-20261104-v1/` is retained development history; this v2
package is the final-source control validation.

| Actual previously computed input | Audited modes | Largest NumPy replay difference | Peak postprocessor RSS |
|---|---:|---:|---:|
| Closed V5 water segment 003 | 3 | 1.819e-12 cm^-1 | 86512 KiB |
| Closed V5 ethanol segment 004 | 21 | 2.387e-12 cm^-1 | 87708 KiB |

These are actual saved water/ethanol calculations processed again; no new QM
was performed. Both outputs are explicitly `DONE_CONTROL_VALIDATION`, with
`Spinosin_target_status=NOT_RUN_CONTROL_ONLY` and
`qualified_for_mechanism=false`. They cannot be presented as Spinosin results.
The CLI ran with one CPU affinity, BLAS threads=1 and a 1 GiB virtual-address
limit. Dependencies were reused without upgrading the quantum environment.

The script validates healthy V5 producer closure/source/proof/log/result and
the original same-state gradient parent before the first snapshot NPY read.
It then verifies source snapshot/MF/geometry, raw and prescribed symmetric
Hessian hashes, full global RKS/H1/response metadata, complete 75/219 dimensions
for the default target, all CSV/JSON signs and units, reduced masses, and
independent NumPy mass weighting/TR projection/eigenvalues/eigenvectors.
Current mode/CSV hashes that lack a producer file hash are recorded honestly
and checked against the anchored Hessian/JSON; no retrospective hash is forged.

After a real complete healthy Spinosin result exists, the default CLI is:

```text
.runtime/bin/python scripts/analyze_spinosin_v5_modes_20261104.py --result ACTUAL_CLOSED_RESULT_JSON --result-sha256 ACTUAL_RESULT_SHA256 --output NEW_ANALYSIS_DIRECTORY
```

Use the actual path/SHA from the completed job, and a new output directory.
The script refuses checkpoint-only, active or incomplete frequency states.
All 219 signed frequencies are preserved. Any imaginary/zero mode blocks
mechanism pairing rather than being filtered or converted to an absolute value.
Only a positive harmonic minimum with the original fresh-gradient evidence
enables a descriptive full Cartesian comparison of all low-frequency modes
with all 20 overlapping-fold candidate records. Those records are not 20
independent chemical peaks; no nearest-peak chemical assignment is made.

Figures are standalone PNG/PDF mode-position plots with equal display heights,
not IR-intensity or absorption curves. IR activity is unknown, and a normal
mode can be IR inactive. XYZ files use the actual source atom order, an
arbitrary 0.15 Angstrom maximum display amplitude and no new-SDF bond mapping;
they are not molecular dynamics. Experimental absolute alpha units remain
unknown. Isolated-molecule frequencies do not establish solid medicinal-material
peak assignments or a global conformer minimum. No theory/peak result enters
the classifier or changes its reported accuracy.

`execution_records.json` records every final command, exit code and source SHA.
Each control directory contains CSVs, checks, PNG/PDF, a display-only XYZ and its
own artifact index. The independent review is at
`../modes-postprocessor-independent-review-20261104-v1/`.
