# Independent V5 supervisor and migration review

**PASS**, source SHA-256
`33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee`.
The original V4 `354ff51b...` is unchanged. This review ran no electronic
structure calculations, launched no actual resource supervisor, and sent no
signals to real processes.

V4 had three real exit-boundary monitor failures despite payload exit code 0.
Those histories remain failures with their original measured costs. V5 does
not relabel or consume their snapshots as healthy new sources. The root is
repeating source/gate calculations under the new supervisor separately.

The reviewed change permits a terminal-to-wait4 transition for the original
owned process only, for at most 0.5 seconds. A direct identity observation that
is live, unreadable through permission/I/O failure, malformed, changed, or
accompanied by unexpected process-group members still fails closed. During
the terminal grace, cgroup bounds, the last CPU high-water mark, the phase wall
limit and the original overall deadline remain active. Missing RSS is not
reinterpreted as zero. An observed cgroup violation is retained even if wait4
becomes ready in that poll. Bounded final cleanup cannot wait forever; unknown
final CPU remains unknown and blocks further budget use.

Twelve original process-identity, signal, CPU/RSS observation, runtime-proof,
prior-lock and phase-limit functions have exactly equal ASTs between V4 and V5.
The new proof/schema/source versions are explicit, and the V4/V5 global-lock
path is shared. No unsafe derivative VHF optimizer or quantum method is changed.

The new runner forbids fresh initialization. Migration verifies and locks the
closed original V4 source, preserves its complete phase prefix and original
created time/deadline/caps/costs, records both source versions, and checks the
same historical prefix and bounds again before further phases. It does not
create another 36-hour CPU pool or restart the wall clock.

Actual independent checks:

- 24 pure tests passed, including a fake-Popen monitor-loop terminal race and
  retained actual-cost accounting, live/unreadable fail-closed behavior,
  bounded cleanup, and migration invariants. No real payload was launched.
- A separate private complete-migration/validation fixture passed eight
  checks: preservation, rejected cap/deadline/cost reset, rejected historical
  failure relabel/omission, additive new cost, and changed-source rejection.
  Its authorization/prior verifiers are mocked and its authorization file is
  deliberately invalid; it is not an actual research budget.
- Read-only real JSON checks confirmed all 13 V4 phase rows remain identical
  in the actual migrated V5 prefix. Original CPU `3605.273693 s`, CPU cap
  `129600 s`, wall cap `64800 s`, and deadline
  `2026-11-05T01:20:50.391742+00:00` are preserved. Original V4 SHA is
  `4533c152abb267c685a451a2690c9e6be8e94e2fe7e88e9194c8f2dda7384e35`.
  The mutable V5 observation is explicitly not final whole-pool accounting.

`replay_guard_review.py` reproduces the pure/static review in a fresh output
directory. `verify_actual_migration_json.py` reads JSON only and refuses to
overwrite its first receipt. No active molecular HDF/NPY was read. New V5
water/ethanol/Spinosin scientific results must be assessed from root's actual
execution records, not these reviewer fixtures.
