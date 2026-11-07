# V4 passive finisher preflight

Status: **PURE_TESTS_PASS; independent source review PASS**. This directory contains software-control evidence only. It does not contain a new quantum calculation or a new Spinosin frequency result.

The completed targeted invocation ran 68 tests: 46 finisher tests and 22 frozen closed-only analyzer tests. The captured log reports `Ran 68 tests in 0.456s` and `OK`. The outer invocation completed in 0.6643 seconds; the recorded UTC start/end and exact command are in `preflight.json`. The tests use temporary fake repositories, JSON fixtures, mocked subprocesses and temporary file locks. No real analyzer, passive observer, quantum job or Git mutation was started by this preflight. No active Spinosin array, HDF or checkpoint was read.

The tests cover the exact target and all three stage closure gates; root wrapper completion chronology; partial JSON marker polling; one analysis attempt; V4/global/prior lock ownership without analyzer re-locking; the fixed runtime interpreter and locked environment; unchanged scientific failure status; shared-index preservation; global Git scope; source/blob/parent/remote checks; fixed author and committer identities; and the exact closed target SCF checkpoint size/hash binding. The checkpoint is only a possible SCF warm-start artifact, not a full Hessian restart.

The finisher remains **NOT_RUN** at this preflight. There is no real finalization permit, analysis release or frequency export created by these tests. Root must separately bind the frozen source/test hashes, exact already-started target identity and execution lock in an explicit future permit before launching it. The analyzer does not take a flock; the finisher owns V4, global and prior ledger locks across its single analyzer call. Root retains responsibility for the real active calculation and its completion receipt.

Final frozen finisher source SHA-256: `147e1111b2d9f94bccbe2ce3bbed4c026b4767bb56296a9cfba27c2a8d303de5`.

Final finisher test source SHA-256: `b8011933ed19ff38d34c50fde71442eb866ae062a7e74c6ffbbabe6acd47025f`.

Independent reviewer `/root/pyscf_continuation_methods` checked the final source, tests, all seven source hashes, the test log and the final boundary changes, and returned final freeze review PASS before this directory was created. `files_sha256.json` indexes only the three explicitly named static evidence files and the seven named source files. No run-root recursion or active-phase scan was performed.
