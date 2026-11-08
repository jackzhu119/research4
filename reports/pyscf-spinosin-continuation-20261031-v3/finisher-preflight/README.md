# Finite conditional pipeline finalizer preflight

**DONE: implementation, independent source review and24 pure guard tests.**
All analysis and Git invocations in the tests are mocks. This agent did not run
the finalizer, export actual V3 results, start quantum work or mutate Git.

New source: `scripts/finalize_pyscf_monitored_pipeline_20261031.py`.
Tests: `tests/test_pyscf_monitored_finisher.py`.
Template: `docs/templates/PYSCF_MONITORED_FINISHER_PERMIT_20261031.template.json`.
The template's ROOT-GO flag is false and its finalizer SHA is a placeholder.
It cannot authorize execution.

## Root launch contract

Root must first review and commit the finalizer, tests, template and all bound
analysis/controller/supervisor source bytes on
`research/pyscf-spinosin-mechanism-20261031`. The eventual automatic Git delivery
only selects new V3 artifacts and the designated new scientific document;
uncommitted source files outside that scope are intentionally not swept in.

Root then copies and completes the template in the actual V3 report root:

- Final reviewed source SHA map, including frozen analyzer
  `92ef86dedcbef4a8879fd53a15fa2a209ea19c761c203b1a91177d60e2908a08`.
- Actual existing authorization SHA, immutable pipeline-plan SHA and unchanged
  V3 deadline.
- `allow_conditional_finalization=true`, actual timezone-aware creation time.
- `allow_git_push=true` only if root elects to launch with explicit `--push`.

Launch form, only after root GO and exact permit hash:

```bash
.runtime/bin/python scripts/finalize_pyscf_monitored_pipeline_20261031.py \
  --permit reports/pyscf-spinosin-continuation-20261031-v3/conditional_finisher_permit.json \
  --permit-sha256 ACTUAL_COMPLETED_ROOT_PERMIT_SHA256 \
  --delivery-dir outputs/pyscf-spinosin-continuation-20261031-v3/final-delivery-001 \
  --push
```

Omit `--push` for analysis/document generation only. The delivery directory must
be fresh and under ignored `outputs/`. No real permit or final release was
created by this implementation agent.

## Closure and bounded execution

Polling is15 seconds, with finite iterations and total wall time bounded by the
original V3 deadline. There is no extension, new budget, quantum launch or retry.
An empty ledger between stages is insufficient: the fixed pipeline state must
be DONE/PARTIAL with `ended_at_utc`, no ledger phase may be RUNNING, and the
controller's final ledger SHA must match actual bytes. Plan/source/authorization
bindings are rechecked after obtaining locks.

The finalizer holds V3 ledger → V3 global → prior V2 ledger locks in the same
order as the actual supervisor throughout export and optional Git delivery.
It emits a single root-conditional release binding permit SHA, final analyzer
and finalizer SHA, final ledger SHA and closed controller-state SHA. It invokes
the frozen analyzer once. Failure preserves partial output and ends without a
retry. If the deadline leaves insufficient export time, it records that limit
and does not invent a grace period.

The new `docs/PYSCF_SPINOSIN_MONITORED_RESULTS_20261031.md` is written only from
the actual analyzer summary and exact release hashes. It separates computed
Spinosin frequencies including negative/unvalidated values from validated full
positive-mode results, retains missing-worker/failure counts, excludes ethanol
and water from Spinosin evidence, leaves IR intensity NOT_COMPUTED and claims no
chemical attribution or classification improvement. Existing reports are never
overwritten.

## Optional Git delivery

The original shared index must be empty. Every stage checks the fixed research
branch, HEAD and actual closed bytes; branch/index changes stop Git without
resetting or unstaging anything. Candidates are restricted to the actual new V3
run and one new document, reject symlinks/deletions, and enforce25 MiB per file
and150 MiB total before staging. `git commit --only` names exactly those files.
Commit paths and each committed blob are verified against the audited work/index
content before push. Push uses the fixed audited commit SHA to the research
branch, without force, merge or main changes, closing a HEAD-resolution race.

If an unrelated staged path arrives concurrently it is preserved and push is
blocked. Actual commit SHA and push status are recorded only after each actual
operation in ignored delivery JSON/events; scientific documents contain no
premature publication claim or self-referential commit hash. A failed push
retains the real created commit SHA in delivery evidence.

The methods agent independently reviewed closure/release/scientific scope and
the final Git-race protections. Pure tests cover stage gaps, stale final hashes,
ROOT-GO/source changes, finite polling, bounded file selection, shared-index and
branch changes, committed-blob mismatch, fixed-SHA push, one failed analysis
attempt, ignored delivery records and misleading chemical-scope rejection.
