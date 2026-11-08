# Current evidence-led execution: V4, same90,000 CPU-second grant

This is the corrected continuation of V3, not an additional authorization.
The original dataset/labels/roles and experimental matrix remain unchanged.
V3's first real fold reached51 healthy core/LDA stage models but failed in
scikit-learn1.8 string-label MLP early stopping. It was cleanly paused; all
failed logs, model stages and costs remain in the V3 directory.

V4 uses a distinct scientific source that internally encodes MLP training
labels and maps predictions back to the original source IDs. Other classifier
and selection logic is unchanged. The source has a real five-string-class
MLP regression test. A new output directory avoids replacing failed evidence.

`prior_execution_carry.json` pins the closed V3 outer/inner states and config.
The authoritative V4 outer ledger includes V3's22.199928 inner CPU seconds,
0.328275 controller CPU seconds and cumulative preparation, preserving the
same88,000+1,000+1,000 allocation and original deadline. The V4 inner ledger
is local to this continuation; consult the outer ledger for grant totals.

Scientific limits and all330 declared jobs are documented in
`docs/EVIDENCE_DRIVEN_VALIDATION_20261108.md`; branch audit evidence is in
`docs/CROSS_BRANCH_EVIDENCE_AUDIT_20261108.md`.

Run states are in `resource-control/state.json` and `control/state.json`.
Actual launch identities and root receipts are separate from preparation
configs. Admission waiting is not completed research. Only closed full
five-fold×three-repeat/seed groups support development recommendations.

This independent Git branch is
`research/evidence-driven-validation-90000-20261108`. It does not update main
or the old queue's Git index. The old species-relaunch queue continues under
its separate grant; the new admission layer observes it and can stop only
new nonce-verified children. No timed chat report is scheduled.

Large raw arrays and some trained preprocessing caches remain local with
hashes; code/data can reproduce them. Closed compact reports and bounded model
artifacts receive priority in automatic Git backups. Cloud host replacement
cannot be prevented by a detached process.
