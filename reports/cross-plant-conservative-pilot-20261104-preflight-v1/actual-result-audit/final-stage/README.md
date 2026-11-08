# Independent final-stage replay

`audit.json` contains the actual numerical/provenance replay. No model fitting, quantum kernels or Torch inference was executed by this auditor. `checks.json.gz` preserves every technical assertion exactly; these are data checks, not independent scientific replications. The full plaintext copy remains locally under the ignored path recorded in `verbose-checks-storage.json`. Re-run the included Python auditor with the original inputs to reproduce the numerical audit.
