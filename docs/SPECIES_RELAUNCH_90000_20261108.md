# Corrected-species replication under a new finite grant

The user explicitly authorized a new 90,000 CPU core-second deployment on
2026-11-08. The previous host's live queue and uncommitted costs are unavailable.
This run neither resets that queue nor claims that its remaining grant was
recovered. Work is isolated on `research/species-relaunch-90000-20261108` in
`/workspace/research4-relaunch`; the original worktree and main are untouched.

## Scientific scope

Use only the active 18-workbook registry: 500 酸枣仁, 364 枣仁, 403 黄芪 and
405 天麻 records. Preserve all 1,672 IDs, labels, raw workbook hashes and the
999 TRAIN / 334 CAL / 339 PRED assignment. Rebuilt features exactly match the
historical feature-cache SHA. Workers receive only 999 TRAIN rows.

错误混种分类成绩已按2026-11-12用户决定撤回；论文使用main的paper/current白名单。
it is not pure-酸枣仁 origin performance. This run first completes a pure-acid
five-fold development comparison of original R0, optimized R2 MC-UVE/SVM,
full-spectrum SVM and PLS-DA. Subsequent fixed groups examine n/alpha fusion,
other-herb tasks, matched target-only/extra-training/cross-herb-pretraining
neural arms, and conservative support adaptation. No master CAL/PRED tuning,
new DFT work, laboratory measurements or external confirmation is claimed.
Retain negative transfer, harmful calibration and incomplete groups.

The predeclared maximum is 1,305 scientific jobs plus 55 summaries and 55
filtered backup jobs. Stop earlier when the finite resource allocation ends;
the whole matrix is not guaranteed to fit. Selection needs complete five-fold
groups and all prescribed seeds. Details: `SPECIES_RELAUNCH_SCIENTIFIC_REVIEW_20261108.md`.

## Execution and resource evidence

Run root: `data/research/species_relaunch_20261108_v2/`.

- `user_authorization_NEW_90000.json`: new grant and prior-runtime limitations.
- `preparation_cpu.json`: measured preparation phases; frozen before launch.
- `jobs.json`, `protocol.json`, `input_lock.json`: frozen queue and TRAIN allowlist.
- `resource-protocol/config_ROOT_APPROVED.json`: source/config/manifest pins.
- `resource-control/launch-*.json`, `state.json`: actual wrapper identity and
  outer resource ledger, created only by execution.
- `control/state.json`, `receipts/`, `logs/`: actual supervisor cost, job
  states, immutable completed-job receipts and attempt logs.
- `runs/`, `cache/`: real experiments and exact-identity, stage-boundary caches.
- `snapshots/`, `backups/`, `results/`: immutable readouts and verified remote
  publication receipts. A configuration or receipt marked NOT_RUN is not a fit.

Allocation: 88,000 inner CPU seconds (including all fit, summary and backup
children), 1,900 reserved for preparation, 100 for the outer monitor. Unused
reserves are not moved or deliberately consumed. One numerical thread/job,
owned-process RSS stop 10 GiB, cgroup stop 12 GiB, disk floor 1 GiB, aggregate
wall cap 36 h, and existing hard deadline 2026-11-10T02:50:37.289380Z. Small
cleanup reserves may stop computation before the nominal ceiling.

Standard-library resource/receipt tests and existing leakage tests precede
launch. The neural self-test changes excluded holdouts and verifies unchanged
source data, five-fold coverage, support/query isolation, nested support draws
and paired initialization without fitting scientific models.

## Monitoring and recovery

Read the two small `state.json` files and compare actual `/proc` PID start
ticks, UID, boot ID and CPU changes. Do not equate a missing PID, a copied
receipt or a prepared queue with successful completion. Do not start a second
supervisor while any verified owned job remains active.

Read-only status command, from this worktree:

```bash
/workspace/.venvs/research4-species-relaunch-20261108/bin/python -B scripts/species_background_budget.py status --state-dir data/research/species_relaunch_20261108_v2/control
```

The root launches the reviewed resource wrapper in a separate session with
closed stdin and an exclusive console file. It is independent of chat-client
closure while this cloud host remains alive. Host replacement/reclamation
cannot be prevented by a daemon. Clean resource pauses may resume under the
same frozen config and cumulative costs after identity/budget verification;
unclean unknown states block automatic takeover and never refund unknown CPU.

Filtered Git backups run serially inside the budget after immutable summaries,
beginning after five experiments. They publish only closed, hash-checked small
TRAIN-only evidence to the research branch and independently check the remote
commit. Raw workbooks, feature NPZ files, live ledgers and unsealed/oversized
caches are excluded. Network failures remain explicit NOT_BACKED_UP receipts,
without restarting scientific jobs. No scheduled chat notification is created.
