# Finite completion observer preflight

Source: `scripts/observe_spinosin_v5_completion_20261104.py`, frozen SHA-256
`af784f43efa39c90c6c3b2596dae3099a627014b166846c78ca2f71262eb908e`.
23 pure metadata tests and independent review passed. The actual launch-template
dryrun returned NOT_RUN and the finite deadline
`2026-11-05T01:30:50.391742+00:00`. Preflight performs no Git command, target-array
read, analyzer execution or quantum calculation.

Root registers this manifest/source and explicitly starts the observer. This
preflight itself does not start it. The example manifest is `launch-template.json`,
SHA-256 `daf51386bc2021a745fb5474a8958046524a91aab28d3bf1bee6d9f12a5d46ae`.
The command from the repository root is:

```sh
PYTHONPATH=.runtime/quantum-pilot-site:.runtime/quantum-upgrade-site \
  .runtime/bin/python scripts/observe_spinosin_v5_completion_20261104.py \
  --manifest reports/pyscf-spinosin-resumable-20261104-v5/completion-observer-preflight-20261104-v1/launch-template.json \
  --manifest-sha256 daf51386bc2021a745fb5474a8958046524a91aab28d3bf1bee6d9f12a5d46ae \
  --execute
```

Omit `--execute` for the safe dryrun. Copying/changing the registered launch
manifest requires its actual new SHA. `observer_output` must be a fresh
`completion-observer*` subtree within the V5 report root. It cannot occupy the
quantum `auto-segments` tree. No code or test is committed while the root job is
running: its evolving HEAD remains exclusively owned by the original backup job.

The observer polls JSON and a bounded console tail, at most until the earlier of
the registered 72-hour research deadline and the original quantum deadline plus
600 seconds. A monotonic deadline also bounds the wait. It does not extend the
quantum budget or start/retry quantum workers. PARTIAL, missing terminal evidence
or timeout saves a blocked status/recovery context locally and launches neither
analyzer nor Git publication. Cloud/session persistence is not guaranteed.

After the exact `DONE_PUBLISHED` JSON console marker, it requires the matching
local/remote research commit, a clean index, complete controller150/219 metadata,
the frozen source snapshot/Greceipt/gate, healthy source SCF and gradient, and
all150 healthy RHF unit producers. No forensic unit is automatically approved.
Only then does it spawn the separately frozen analyzer SHA
`a2e3588c6dea121be58e4a69afb137bc6e5092199ca4180937fcdb34f6f57154`.
The analyzer operates with one CPU, one BLAS thread, its existing1-GiB virtual
memory cap and at most900 seconds, additionally bounded by remaining observation
time and a cleanup reserve. The observer never opens target NPY/HDF itself.

Output is the new `final-mode-analysis/` directory and
`docs/PYSCF_SPINOSIN_FINAL_STATUS_20261104.md`. Analyzer exit0, source/input hashes,
its output index and resource profile must all validate. Complete signed modes
can still contain negative/zero frequencies; these are retained and mechanism
qualification is reported separately. No IR intensities, solid-state assignment,
pure-compound confirmation or classifier gain is invented.

Publication reuses the root job's frozen `publish` function, including global
quantum lock, exact research-branch CAS, narrow staging/tree checks and non-force
push. The publisher runs in a separate bounded session and repeats the terminal,
closure, analysis and scope checks. It commits only registered code/tests,
preflights, actual analysis, final report, observer evidence and launch manifest.
The committed observer status truthfully says ready for publication; the exact
successful publication commit is emitted afterward as `OBSERVER_DONE_PUBLISHED`
in stdout, avoiding a self-referencing commit-hash loop. Failed publication is
blocked for manual review, without changing main or force pushing.
