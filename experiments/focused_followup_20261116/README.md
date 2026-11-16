# Focused follow-up, 2026-11-16

This branch adds experimental evidence only. It preserves the paper, current
allowlist, raw workbooks, historical outputs and unrelated computations.

The latest user authorization replaces the pasted 36,000 CPU-second budget
with **990,000 cumulative CPU seconds** for this fixed task queue. It does not
restart or inherit the old 1,360,000 CPU-second study. MLP extensions, fixed-epoch
MLP and new-batch training remain disabled. No score, significance, target
accuracy or retrospective stability threshold controls continuation.

`protocol.json` defines the scientific scope, source commits, candidates,
comparisons and seeds. `queue.json` contains the complete ordered task list.
`code_lock.json` seals the executable files before launch.
`protected_baseline.json` seals the original tracked file bytes. Actual archived
sample IDs and source input hashes must verify before any dependent fit starts.

The old V4 three-repeat outer5/inner2 bridge and the later outer5/inner5x2
study remain separate. The six primary contrasts are explanatory comparisons
locked after existing results were visible. Specimen bootstrap intervals are
conditional on the archived predictions and do not establish external-batch
generalization. Repeated splits never increase the physical specimen count.

Run on the dedicated branch:

```bash
python -m pip install -r experiments/focused_followup_20261116/requirements.txt
python experiments/focused_followup_20261116/orchestrate.py --publish
```

The workflow `.github/workflows/focused-followup.yml` starts on the initial
experiment-code push. It checks out the latest experiment branch, restores the
source cache, executes pending atomic tasks, and pushes task receipts, raw
experimental outputs, logs and the cumulative CPU ledger. Before a hosted
runner reaches its platform lifetime, the same frozen queue continues on a
fresh runner. Source caches are saved before continuation is requested.

Read `reports/focused_followup_20261116/runtime/state.json` and the Actions run
logs for actual progress. Successful execution requires a real running hosted
job and a verified automatic result/checkpoint push. A detached process in the
chat workspace is not evidence of durable execution. Hosting, permissions,
quota or service failures can still interrupt a job; report those failures
and retain the same task identity and CPU ledger when recovering.

The 12 GB multipart historical archive is streamed with shard and final member
hash checks. Only explicitly selected corrected evidence is retained in the
runtime cache. The archive and old training queue are not rerun. Cached source
evidence is never itself counted as a new model fit.

Every negative result and numerical failure remains in the evidence. Missing
raw signals/export definitions, unrecoverable prediction artifacts and absent
new physical batches receive explicit blocked statuses. The new-batch protocol
and empty metadata templates are delivered without treating historical
resplitting as external validation.
