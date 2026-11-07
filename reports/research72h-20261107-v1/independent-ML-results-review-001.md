# Independent review of the two completed ML studies

PASS: 22566 read-only checks. Largest metric arithmetic difference 3.42e-14.

The frozen robustness study retained 855 original records in each of 18 cases (15,390 reused-record predictions). All 855 baseline predictions replayed exactly, all decision scores differed by 0, and baseline correctness remained 821/855. Every case accuracy, macro-F1, balanced accuracy, per-class denominator and confusion count was independently recomputed with Counter arithmetic and checked against sklearn. The exact 18 prediction files plus four audit files remain locked; five original model hashes and all frozen data/code hashes match.

The saved-prediction study retains 11 original same-ID/same-fold methods and 11 prespecified pairs. All 33 method estimates, 33 paired point differences, 77 per-class changes, 165 rare-record deletion rows and 165 fold-omission rows match independent arithmetic. The 2000-replicate conditional interval rows are finite and ordered; their bootstrap draws were not regenerated.

Guardian closed costs are 0.349637 CPU seconds and 3.381265 CPU seconds (3.730902 combined); both exited 0 with owned processes closed and matching immutable supervisor proof/log hashes. The historical ledger with SHA beginning 55b27cad remains byte-identical, with 187 original phases.

These are retrospective diagnostics on exposed within-dataset OOF predictions. Every anonymous origin has one source lot; this adds no independent-batch, chemical-causal, or population-generalization evidence. ZY02 still has only five records. No training, new inference, quantum work or Git operation occurred in this review. Original files, source encodings and CSV line endings were not rewritten.

See independent-ML-results-review-001.json for exact evidence hashes, every case metric, scope and check counts.
