# Water / ethanol V6b A–B–C control commands — NOT_RUN

This file prepares commands only. No quantum payload, numerical import, NPY/HDF/CHK/XYZ parsing, fitting, Git action, or ledger/auth modification was performed. Root's updated schedule allows these small controls in the idle interval while NNv2 is still source preparation only; all Q controls must close before NNv2 launches. If NN training or its numerical audit has started, wait until both close. Root confirms strict serial execution before each command. Do not register another ledger or reset the existing budget.

Frozen guard SHA-256: `b985d6d6e20f5e4a48387094461f19b133dae761338664e88836c62051f993a5`.
Current staged worker source SHA-256: `f13701964cd8a5fbda25a19ca27ec91b4a414f6b5cb49c14abacf6c82eaa448d`.
Actual ledger supplied by root: `/workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/cpu_budget.json`.
Closed V5 source ledger metadata was independently read: SHA `678fb05a2795651971f2dcffc1bc36d6c85d5b31eef64c6ec132e8cc13580ee9`, actual cumulative CPU `129473.109659` s. Root reports the existing V6b registration explicitly adds 86,400 CPU s and 79,200 deadline s; this preparation did not re-register or modify that allocation.

Both parent and child use **`--threads 1`** in every new control stage. Old snapshots, source MOs/grids, atom order and PBE/def2-SVP/grid1/noDF/noD3 method stay fixed. A/B/C each launch a fresh process; no fresh SCF or geometry optimization is requested.

## Exact verified source identities and immutable references

Names below are taken from the verified old input manifests and their geometry/source-result lineage, not inferred from molecular formulas. `--control-validation` is only the scoped software-control switch; it does not establish chemical identity by itself. Full reference hashes and every 6/18-unit parent phase are in `control_metadata_references.json`.

### water

- Verified old identity: `validation_water_verified_old_optimized_geometry`.
- Input manifest: `/workspace/research4/reports/pyscf-sdf-continuation-20261103-v1/inputs/validation_water_prepare_input.json`; SHA `865877bc001ebca8a5a9fde48dbb16cd9a21504a14b435c23a276bfb6f5e9ded`.
- Prepare result: `/workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-water-prepare-001/result.json`; SHA `e6d54f0e5f6ef1d77506b92e9bf42f019ecd46ab19b5bf1b0a41a4c49ad037cf`.
- Snapshot manifest: `/workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-water-prepare-001/exact-mf-snapshot/manifest.json`; SHA `123e3e760111f349315703fd213d94aac26eda87e02d03caf98dc9ebbd0cfaf4`.
- Restored gradient result: `/workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-water-restore-gradient-001/result.json`; SHA `517de6a69fc64ff96abac6d6ca5b1482c6e429ade82cf4328d36b596548f6397`.
- Full stock reference result: `/workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-water-stock-hessian-001/result.json`; SHA `986be9aa60ff6810693169055e078d202dc7135f8a6ddf61c47f0a921018a930`.
- RHF header: `/workspace/research4/outputs/pyscf-spinosin-resumable-20261104-v5/water-restart-validation/header.json`; SHA `97724c31488cc239029cf08e082eb5d4edfbb0093a0ed8588e21e6e6b7932e1a`.
- Original geometry XYZ (path/hash from verified input manifest only): `/workspace/research4/reports/pyscf-spinosin-continuation-20261031-v3/water-native4-validation-001/optimized.xyz`; SHA `21f50784ccb6b67f072c14e87d1bd3adede278b6a74371f5cb2cc85a10abf916`.
- Healthy RHF cache: `/workspace/research4/outputs/pyscf-spinosin-resumable-20261104-v5/water-restart-validation`; 6/6 complete units; header binding `7c8f9b1f33f7ecca00e1049d6ccef72994f6a43ff81761bde8d9ed1b593a9e25`.
- Old full stock reference: 3 modes; native threads=4. All selected old parent rows are EXITED/0 with CPU/memory/wall/monitor/hardkill flags false. This is a metadata closure recheck; the actual worker must still validate source/proof/log/array hashes before use.

### ethanol

- Verified old identity: `validation_ethanol_verified_old_optimized_geometry`.
- Input manifest: `/workspace/research4/reports/pyscf-sdf-continuation-20261103-v1/inputs/validation_ethanol_prepare_input_v2.json`; SHA `7c3ddf300ab6cb806a475a8e08c3a11944c3cc1e744332dab65ffbd52b638e07`.
- Prepare result: `/workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-ethanol-prepare-001/result.json`; SHA `043d881cfcf11c0815acd762d686ca2535b169d1dcc6b2fa44579a0883461070`.
- Snapshot manifest: `/workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-ethanol-prepare-001/exact-mf-snapshot/manifest.json`; SHA `9966fbec6184b744e236471d3b38e0cac2d40cde7cb16b45cd444b1da1e35cba`.
- Restored gradient result: `/workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-ethanol-restore-gradient-001/result.json`; SHA `205fa6d04c591bbac3a0397f184a5cf64673b6198cce7500592ba2cce29fe8bc`.
- Full stock reference result: `/workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-ethanol-stock-hessian-001/result.json`; SHA `6c35b214b22e927185ca2298674fb8e20e1a6fede06bbecc33be22e7910d1e61`.
- RHF header: `/workspace/research4/outputs/pyscf-spinosin-resumable-20261104-v5/ethanol-restart-validation/header.json`; SHA `5a64a22acb5da8fc2eea9ec5407911a730e164bf96e81be2c816e031be893a79`.
- Original geometry XYZ (path/hash from verified input manifest only): `/workspace/research4/reports/pyscf-spinosin-continuation-20261031-v3/validation-ethanol-native4-001/optimized.xyz`; SHA `4d38510a503b2a1f957df5590e18947c044ccccd2ebc6d190b7630739c0941f3`.
- Healthy RHF cache: `/workspace/research4/outputs/pyscf-spinosin-resumable-20261104-v5/ethanol-restart-validation`; 18/18 complete units; header binding `4bcd75441aef3cab501ec16a00b0e25ec31695ace5219e1ca6fd253fc9b27e6b`.
- Old full stock reference: 21 modes; native threads=4. All selected old parent rows are EXITED/0 with CPU/memory/wall/monitor/hardkill flags false. This is a metadata closure recheck; the actual worker must still validate source/proof/log/array hashes before use.

## Commands, one stage at a time

Before B or C, root must first observe the preceding supervisor return EXITED/0 with all failure/resource flags false and the expected STAGE_COMPLETE result. **Directory or result-file existence alone is insufficient.** The small SHA commands below read only the newly closed result JSON; do not substitute a failed or active result. B/C then independently requalify producer closure inside the frozen worker.

Proposed short per-phase limits below are bounded requests within the already registered ledger, not new allocations. Actual wait4 CPU is debited. Stop on any nonzero exit or unexpected status; do not automatically retry or increase limits.

### water — rks-partial

```bash
/workspace/research4/.runtime/bin/python \
  /workspace/research4/scripts/run_pyscf_continuation_budget_20261105_v6b.py \
  run \
  --ledger \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/cpu_budget.json \
  --phase \
  v6b-control-water-A-001 \
  --cpu-seconds \
  1200 \
  --wall-seconds \
  900 \
  --threads \
  1 \
  --memory-gib \
  24 \
  --log \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/execution/v6b-control-water-A-001.log \
  -- \
  /workspace/research4/.runtime/bin/python \
  /workspace/research4/scripts/pyscf_staged_hessian_worker_20261105_v6b.py \
  --execute \
  --control-validation \
  --stage \
  rks-partial \
  --threads \
  1 \
  --snapshot \
  /workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-water-prepare-001/exact-mf-snapshot \
  --snapshot-manifest-sha256 \
  123e3e760111f349315703fd213d94aac26eda87e02d03caf98dc9ebbd0cfaf4 \
  --gradient-check-result \
  /workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-water-restore-gradient-001/result.json \
  --gradient-check-result-sha256 \
  517de6a69fc64ff96abac6d6ca5b1482c6e429ade82cf4328d36b596548f6397 \
  --output \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-water-rks-partial-001 \
  --rhf-cache \
  /workspace/research4/outputs/pyscf-spinosin-resumable-20261104-v5/water-restart-validation
```

### water — h1ao

```bash
R4_CONTROL_WATER_A_SHA="$(/workspace/research4/.runtime/bin/python -S -c 'import hashlib,pathlib,sys;print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-water-rks-partial-001/result.json)"
```

```bash
/workspace/research4/.runtime/bin/python \
  /workspace/research4/scripts/run_pyscf_continuation_budget_20261105_v6b.py \
  run \
  --ledger \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/cpu_budget.json \
  --phase \
  v6b-control-water-B-001 \
  --cpu-seconds \
  1200 \
  --wall-seconds \
  900 \
  --threads \
  1 \
  --memory-gib \
  24 \
  --log \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/execution/v6b-control-water-B-001.log \
  -- \
  /workspace/research4/.runtime/bin/python \
  /workspace/research4/scripts/pyscf_staged_hessian_worker_20261105_v6b.py \
  --execute \
  --control-validation \
  --stage \
  h1ao \
  --threads \
  1 \
  --snapshot \
  /workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-water-prepare-001/exact-mf-snapshot \
  --snapshot-manifest-sha256 \
  123e3e760111f349315703fd213d94aac26eda87e02d03caf98dc9ebbd0cfaf4 \
  --gradient-check-result \
  /workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-water-restore-gradient-001/result.json \
  --gradient-check-result-sha256 \
  517de6a69fc64ff96abac6d6ca5b1482c6e429ade82cf4328d36b596548f6397 \
  --output \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-water-h1ao-001 \
  --partial-result \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-water-rks-partial-001/result.json \
  --partial-result-sha256 \
  "$R4_CONTROL_WATER_A_SHA"
```

### water — global

```bash
R4_CONTROL_WATER_A_SHA="$(/workspace/research4/.runtime/bin/python -S -c 'import hashlib,pathlib,sys;print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-water-rks-partial-001/result.json)"
```

```bash
R4_CONTROL_WATER_B_SHA="$(/workspace/research4/.runtime/bin/python -S -c 'import hashlib,pathlib,sys;print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-water-h1ao-001/result.json)"
```

```bash
/workspace/research4/.runtime/bin/python \
  /workspace/research4/scripts/run_pyscf_continuation_budget_20261105_v6b.py \
  run \
  --ledger \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/cpu_budget.json \
  --phase \
  v6b-control-water-C-001 \
  --cpu-seconds \
  1800 \
  --wall-seconds \
  1200 \
  --threads \
  1 \
  --memory-gib \
  24 \
  --log \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/execution/v6b-control-water-C-001.log \
  -- \
  /workspace/research4/.runtime/bin/python \
  /workspace/research4/scripts/pyscf_staged_hessian_worker_20261105_v6b.py \
  --execute \
  --control-validation \
  --stage \
  global \
  --threads \
  1 \
  --snapshot \
  /workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-water-prepare-001/exact-mf-snapshot \
  --snapshot-manifest-sha256 \
  123e3e760111f349315703fd213d94aac26eda87e02d03caf98dc9ebbd0cfaf4 \
  --gradient-check-result \
  /workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-water-restore-gradient-001/result.json \
  --gradient-check-result-sha256 \
  517de6a69fc64ff96abac6d6ca5b1482c6e429ade82cf4328d36b596548f6397 \
  --output \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-water-global-001 \
  --partial-result \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-water-rks-partial-001/result.json \
  --partial-result-sha256 \
  "$R4_CONTROL_WATER_A_SHA" \
  --h1-result \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-water-h1ao-001/result.json \
  --h1-result-sha256 \
  "$R4_CONTROL_WATER_B_SHA"
```

### ethanol — rks-partial

```bash
/workspace/research4/.runtime/bin/python \
  /workspace/research4/scripts/run_pyscf_continuation_budget_20261105_v6b.py \
  run \
  --ledger \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/cpu_budget.json \
  --phase \
  v6b-control-ethanol-A-001 \
  --cpu-seconds \
  1800 \
  --wall-seconds \
  1200 \
  --threads \
  1 \
  --memory-gib \
  24 \
  --log \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/execution/v6b-control-ethanol-A-001.log \
  -- \
  /workspace/research4/.runtime/bin/python \
  /workspace/research4/scripts/pyscf_staged_hessian_worker_20261105_v6b.py \
  --execute \
  --control-validation \
  --stage \
  rks-partial \
  --threads \
  1 \
  --snapshot \
  /workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-ethanol-prepare-001/exact-mf-snapshot \
  --snapshot-manifest-sha256 \
  9966fbec6184b744e236471d3b38e0cac2d40cde7cb16b45cd444b1da1e35cba \
  --gradient-check-result \
  /workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-ethanol-restore-gradient-001/result.json \
  --gradient-check-result-sha256 \
  205fa6d04c591bbac3a0397f184a5cf64673b6198cce7500592ba2cce29fe8bc \
  --output \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-ethanol-rks-partial-001 \
  --rhf-cache \
  /workspace/research4/outputs/pyscf-spinosin-resumable-20261104-v5/ethanol-restart-validation
```

### ethanol — h1ao

```bash
R4_CONTROL_ETHANOL_A_SHA="$(/workspace/research4/.runtime/bin/python -S -c 'import hashlib,pathlib,sys;print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-ethanol-rks-partial-001/result.json)"
```

```bash
/workspace/research4/.runtime/bin/python \
  /workspace/research4/scripts/run_pyscf_continuation_budget_20261105_v6b.py \
  run \
  --ledger \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/cpu_budget.json \
  --phase \
  v6b-control-ethanol-B-001 \
  --cpu-seconds \
  1200 \
  --wall-seconds \
  900 \
  --threads \
  1 \
  --memory-gib \
  24 \
  --log \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/execution/v6b-control-ethanol-B-001.log \
  -- \
  /workspace/research4/.runtime/bin/python \
  /workspace/research4/scripts/pyscf_staged_hessian_worker_20261105_v6b.py \
  --execute \
  --control-validation \
  --stage \
  h1ao \
  --threads \
  1 \
  --snapshot \
  /workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-ethanol-prepare-001/exact-mf-snapshot \
  --snapshot-manifest-sha256 \
  9966fbec6184b744e236471d3b38e0cac2d40cde7cb16b45cd444b1da1e35cba \
  --gradient-check-result \
  /workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-ethanol-restore-gradient-001/result.json \
  --gradient-check-result-sha256 \
  205fa6d04c591bbac3a0397f184a5cf64673b6198cce7500592ba2cce29fe8bc \
  --output \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-ethanol-h1ao-001 \
  --partial-result \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-ethanol-rks-partial-001/result.json \
  --partial-result-sha256 \
  "$R4_CONTROL_ETHANOL_A_SHA"
```

### ethanol — global

```bash
R4_CONTROL_ETHANOL_A_SHA="$(/workspace/research4/.runtime/bin/python -S -c 'import hashlib,pathlib,sys;print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-ethanol-rks-partial-001/result.json)"
```

```bash
R4_CONTROL_ETHANOL_B_SHA="$(/workspace/research4/.runtime/bin/python -S -c 'import hashlib,pathlib,sys;print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-ethanol-h1ao-001/result.json)"
```

```bash
/workspace/research4/.runtime/bin/python \
  /workspace/research4/scripts/run_pyscf_continuation_budget_20261105_v6b.py \
  run \
  --ledger \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/cpu_budget.json \
  --phase \
  v6b-control-ethanol-C-001 \
  --cpu-seconds \
  1800 \
  --wall-seconds \
  1200 \
  --threads \
  1 \
  --memory-gib \
  24 \
  --log \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/execution/v6b-control-ethanol-C-001.log \
  -- \
  /workspace/research4/.runtime/bin/python \
  /workspace/research4/scripts/pyscf_staged_hessian_worker_20261105_v6b.py \
  --execute \
  --control-validation \
  --stage \
  global \
  --threads \
  1 \
  --snapshot \
  /workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-ethanol-prepare-001/exact-mf-snapshot \
  --snapshot-manifest-sha256 \
  9966fbec6184b744e236471d3b38e0cac2d40cde7cb16b45cd444b1da1e35cba \
  --gradient-check-result \
  /workspace/research4/reports/pyscf-spinosin-resumable-20261104-v5/v5-validation-ethanol-restore-gradient-001/result.json \
  --gradient-check-result-sha256 \
  205fa6d04c591bbac3a0397f184a5cf64673b6198cce7500592ba2cce29fe8bc \
  --output \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-ethanol-global-001 \
  --partial-result \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-ethanol-rks-partial-001/result.json \
  --partial-result-sha256 \
  "$R4_CONTROL_ETHANOL_A_SHA" \
  --h1-result \
  /workspace/research4/reports/pyscf-spinosin-staged-20261105-v6b/control-ethanol-h1ao-001/result.json \
  --h1-result-sha256 \
  "$R4_CONTROL_ETHANOL_B_SHA"
```

## Required final control gate, still NOT_RUN

A: status STAGE_COMPLETE, complete full RKS partial and full 6/18 RHF units; frequencies NOT_RUN. B: status STAGE_COMPLETE, complete H1AO for every 3/9 atom; frequencies NOT_RUN. C: FREQUENCIES_COMPUTED, complete all-atom global Hessian and exactly 3 water / 21 ethanol signed normal modes. All six scientific results must have healthy closed V6b producer rows and unchanged physical MF binding. IR intensities remain NOT_COMPUTED.

Old stock references were computed with 4 native threads; the new A/B/C chain uses 1. Core RHF runtime binding does not encode native thread count, but the new staged physical binding explicitly does; keep **all three new stages at1**. Numerical equivalence to the old stock reference must be measured after C closes, not assumed from source reuse.

Previously fixed independent control thresholds (from V5 `real-restart-validation-independent-audit/{water,ethanol}/audit.json`) are raw/sym Hessian maximum absolute difference1e-6 Eh/Bohr², frequency0.01 cm⁻¹, independent frequency replay1e-7 cm⁻¹ and restored-gradient1e-8 Eh/Bohr. Do not loosen these after seeing differences. Compare every Hessian element and all expected frequencies/modes; do not compare only diagonals.

The old independent audit validates V5 proof/result schemas. A new actual V6b control audit must understand V6b stage/parent proofs; it cannot reuse the old PASS label as if it covered the new execution. No new numerical comparison was performed in this preparation.

Only after both control chains and their independent actual audits pass should root decide whether to start a separately bounded Spinosin stage. Failed V5 partial/H1 output remains excluded. Do not overlap Q with NN, array audits, Git, package installation or other large I/O.
