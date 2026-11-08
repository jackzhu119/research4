# Frozen R2 OOF engineering robustness diagnostics

All prespecified conditions reuse the same 855 historically exposed development records and five exact historical R2 models. The unchanged baseline replayed every original prediction before corruption. No fitting or selection was allowed.

See metric_summary.csv for all cases and paired changes from baseline, per_class_metrics.csv for class denominators, confusions_and_metrics.json for every count matrix, and noise_level_curves.csv for every stochastic seed level. Seed minimum/maximum values are descriptive and are not confidence intervals.

- All 855 records were exposed historical development data; this is not independent confirmation.
- Each ZY record is user-declared independent; original collection ancestry was not independently verified.
- One source lot per anonymous origin group; origin and source-lot effects cannot be separated.
- Synthetic corruptions are engineering spectral-shape tests, not validated instrument noise, batch transfer, or chemical effects.
- Repeated corruption seeds and cases reuse the same 855 records; they do not increase specimen count.
- ZY02 has five records; per-class counts and denominators must accompany macro metrics.
- Frozen SVC.predict vote rules are preserved; OVR scores are not probabilities or an argmax prediction replacement.
- No p-values, biological confidence intervals, adaptation, refitting, parameter selection, or model ranking from corruption scores.
- Unclipped original alpha values, including high-frequency negatives, are retained.
- Preparation parses the already-labelled historical prediction CSV to project a truth-free reference; run-phase scoring access is logically separated, not OS-sandboxed or never-exposed truth.
