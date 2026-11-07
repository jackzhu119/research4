# Actual Spinosin computation

219 signed modes were computed and independently replayed. There are 25 imaginary modes; the positive-minimum and chemical-mechanism gates are false. IR NOT_COMPUTED.

The four residual audits measure selected equations, without a new convergence threshold. Spinosin coverage is only atoms 0 and 36: 6/225 nuclear RHS.

Source-hashed summary generation only, no DFT is run by this script:

```bash
PYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site .runtime/bin/python -B scripts/summarize_spinosin_completion_20261106.py --output reports/spinosin-summary-replay-FRESH
```

Use the ledger from this Git revision for the same resource snapshot. Future appended phases change costs, not the immutable frequency result. Large H1/MO1 caches stay local with their original manifests and hashes; they are not required to rebuild this small summary.
