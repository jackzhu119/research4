# Research4: cache-aware continuation of the remaining scientific matrix

This operational continuation replaces repeated cache-based pauses and giant
legacy reporting jobs. It does **not** change scientific code, data, labels,
model parameters, seeds, source/query assignments or the original total grant.

The original corrected-species matrix has1,305 scientific jobs. Its825 closed
scientific jobs (225 classical+600 neural) remain unchanged. All480 unfinished
neural descriptors are copied **exactly**, including commands, inputs, outputs
and retry rules, into `remaining_jobs.json`. The controller verifies equality
and exhaustive coverage before launch. Completed models are not retrained.

The prior authoritative17,708.25995 CPU seconds are carried into the new outer
ledger. New controller and descendant CPU are added to that total, under the
same90,000 CPU-second cap and original deadline. The local inner ledger starts
at0 because it covers only this continuation; it is **not** the grant total.
No prior failure cost or discrepancy is refunded. The original inner/outer
ledgers and the empty unsealed backup00825 directory are preserved in evidence.

## Fewer artificial interruptions

- File-cache occupancy does not stop computation. Clean pages of closed files
  and immutable Git objects receive rate-limited read-only POSIX cache advice.
  No file is deleted or rewritten, and no global cache-drop operation occurs.
- The original controller's100-second suballocation stop is removed. The
  authoritative overall CPU cap remains unchanged.
- Old worktree Git packing is single-threaded with delta windows disabled and
  bounded caches. These worktree-only settings do not alter scientific values,
  Git object identities or main.
- Repeated500–800MB legacy summary JSONs are replaced by compact reports. Raw
  predictions and calibration records remain in their original output paths.
- Necessary protections remain for total authorized CPU/deadline, genuinely
  non-reclaimable memory exhaustion, disk exhaustion, provenance and ownership.
  Only this controller's nonce-verified children may be stopped or recovered.

## Current state and results

After actual launch, read `work/state.json` for the authoritative cumulative
grant CPU and process identities. `work/control/state.json` records local
remaining-job progress. The original resource-control PAUSED state is retained
as historical evidence and must not be mistaken for this continuation's state.
`CURRENT_RUNTIME_CONTROLLER.json` in the original run points here.

`reports/` contains bounded source-only pooled OOF reports; initial extraction
streams the actual compressed prediction records and then reuses sealed small
per-job caches. Complete five-fold coverage and all three model seeds are
required. Calibration results are preserved raw but are not summarized by this
base-only readout. Fold repetitions do not create independent specimens.

`publications/` records bounded, serial Git result publication to
`research/cache-aware-continuation-20261108`; main and the old Git index are not
used. Remote commit verification, not merely a completed publication attempt,
is required to claim a successful backup. Large artifacts may remain local
with hashes and are never silently represented as fully uploaded.

The separate evidence-validation V4 matrix already completed330/330 scientific
jobs and its final report was backed up at71a303e. It is not rerun here.

## Reproduce this runtime

The original scientific worktree is `/workspace/research4-relaunch`, and this
standalone runtime lives at `/workspace/research4-runtime-20261108`. Installed
CPU Python is pinned by the approved config. Configuration and sources are
hash checked. Default invocation is read-only:

```bash
python cache_aware_controller.py --config config_ROOT_APPROVED.json --config-sha256 CONFIG_SHA256
```

Root launches the explicit`--execute` invocation in a detached session only
after preflight and independent checks. A clean pause can use`--resume` with
the same frozen config and cumulative costs; unknown ownership blocks recovery.
No hourly chat automation, paid service, GPU, main merge or duplicate training
queue is created. The finite matrix ends when completed or its authorization
is exhausted; unused CPU is not deliberately burned.
