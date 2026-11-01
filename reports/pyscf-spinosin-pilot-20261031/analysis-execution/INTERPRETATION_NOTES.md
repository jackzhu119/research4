# Actual export interpretation and display supplement

This directory records read-only postprocessing after the parent released the final, closed computation ledger. No new SCF, gradient or Hessian job was started here. The immutable export and its own hash index remain in `../analysis/`.

## Computational status

- **DONE:** the water Opt/Freq pipeline, independently checked with direct and cached density-fitting backends. Its six CSV rows describe the same water molecule's three physical normal modes measured by two numerical backends. They do not represent six molecular modes, two independent chemical samples, or Spinosin vibrations.
- **PARTIAL:** the Spinosin optimization produced four complete, paired SCF/geometry/gradient evaluations. The final paired energy is −2210.435960490465 hartree. The fifth SCF had no completed corresponding gradient and is not a fifth trajectory point.
- The raw phase result's SCF flag reports a completed SCF value, not proof that optimization converged. Use `../analysis/actual_optimization_steps.csv` to interpret matched energy and gradient values.
- Final paired atom-gradient RMS = 0.007035132810331266 hartree/bohr and maximum = 0.03296678304086501 hartree/bohr, versus thresholds of 0.0003 and 0.00045. A stationary minimum has not been demonstrated.
- **NOT_RUN:** Spinosin Hessian and harmonic frequencies. Zero exported Spinosin frequency rows mean no such calculation was performed, not a computed absence of modes in the 0.1–1.5 THz band.
- **BLOCKED:** experimental/theoretical chemical interpretation. All 20 existing experimental candidate rows are preserved, with theoretical frequencies and frequency differences blank. No peak is chemically attributed.
- **NOT_COMPUTED:** IR intensities. The water frequency-position markers have equal display heights and are not an absorption-intensity spectrum.

## Geometry display

`spinosin_model_geometry_uncropped.png` and its PDF show the same coordinate-derived computational geometry as the immutable export, with an uncropped z-axis unit label. They are a display supplement only; no geometry optimization, invented bond assignment or atom modification was performed. The coordinate file's SHA-256 is recorded in `validation_replay.json` and the original `../analysis/displayed_geometry_provenance.json`. The displayed geometry is the last completed evaluated geometry, not a verified stationary minimum or an experimental structure.

The original sealed geometry PNG remains unchanged and has a partly clipped z-axis label; use this supplemental PNG/PDF for presentation. The original optimization progress, water frequency positions and mechanism-comparison status figures remain valid.

## Resource and validation scope

The ledger reports 2666.234148 CPU seconds of quantum work, with 633.765852 seconds remaining in its configured quantum budget. Stopping with unused budget does not authorize a Hessian at an unconverged geometry. Postprocessing uses zero additional quantum jobs.

`validation.json` records the original full numerical audit. `validate_actual_export.py` and `validation_replay.json` independently reread every exported CSV, all 19 immutable export hashes and sizes, four geometry/gradient pairs, mode counts, units, the missing-theory state, actual CPU sum and geometry atom composition. `commands.jsonl` records the real export and replay commands; `export.log` and `validation_replay.log` preserve their outputs. The final hash index in this directory excludes only itself.
