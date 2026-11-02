# First-gradient comparator frozen preflight

11 synthetic pure tests passed. No actual V3 point/events or checkpoint has been read by this comparator preparation. Independent source review passed. Actual comparison remains NOT_RUN pending a dated root release.

Fixed numerical tolerances: coordinates5e-12 Angstrom, energy1e-8 Hartree, all225 gradient components and all75 atomic gradient norms1e-8 Hartree/Bohr. Native1/native4 are prescribed execution settings, not observed simultaneously active thread counts. Complete gradient timing includes XC and J, excludes SCF, and is not a hardware-independent benchmark or a Hessian performance prediction. Timing overhead is separately checked at max(1second,1% of outer event delta).

The comparator streams only through the completed cycle1 events prefix and rereads exactly that prefix length. It never reads any checkpoint or later active event suffix. It creates one new audit directory and does not overwrite existing results.

CLI: python scripts/compare_pyscf_spinosin_first_gradient_20261031.py --old-output OLD --new-output NEW --release RELEASE.json --output FRESH_AUDIT. Release fields: allow_first_gradient_comparison=true, created_at_utc, comparator_sha256, old_step_json_sha256, new_step_json_sha256.
