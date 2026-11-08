"""One-shot authorized healthy149 relocation; stdlib/byte hashes only, no Q/Git.

Frozen V5 metadata gate functions are extracted unchanged from pinned source
AST, with only their stdlib helper surface. No numerical module is imported.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = Path(__file__).resolve().parent
ARCHIVE = ROOT / "reports/pyscf-spinosin-resumable-20261104-v5/clean-149-checkpoint-archive"
SEAL = ARCHIVE / "seals/v5-spinosin-clean-prefix-149-001.json"
SEAL_SHA = "be974fbb6718ed2e52200b2cce8896196e756ebb87a825891d889cde2b5d499f"
TARGET = ROOT / "outputs/pyscf-spinosin-staged-20261105-v6b/spinosin-C001-healthy149"
RECEIPT = PROTOCOL / "healthy149-seed-copy-20261105.json"
SNAPSHOT_SHA = "3089d6014ba2d2283540eb13cc7dbcdbbf284721042e7250b624197fc15c3311"
V5_LEDGER = ROOT / "reports/pyscf-spinosin-resumable-20261104-v5/cpu_budget.json"
V5_LEDGER_SHA = "678fb05a2795651971f2dcffc1bc36d6c85d5b31eef64c6ec132e8cc13580ee9"
V5 = ROOT / "scripts/pyscf_resumable_hessian_worker_20261104_v5.py"
V5_SHA = "b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190"
STATE = ROOT / "scripts/pyscf_molecular_state_worker_20261104_v5.py"
STATE_SHA = "be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8"
CONTROLLER = ROOT / "scripts/run_pyscf_serial_segments_20261104_v5.py"
CONTROLLER_SHA = "a01b6a79f457bca4b0ab90e63b45a3bb524b62fc586fab26905d6b43a2aba1d7"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def extract(path, expected, names, namespace):
    require(sha(path) == expected, "Frozen gate source changed")
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({n.name for n in nodes} == set(names), "Missing frozen metadata gate function")
    require(not any(isinstance(n, (ast.Import, ast.ImportFrom)) for node in nodes for n in ast.walk(node)), "Nonstdlib gate dependency")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest() for n in nodes}


def main():
    require(not TARGET.exists() and not RECEIPT.exists(), "Fresh target/receipt required; never overwrite")
    require(sha(SEAL) == SEAL_SHA and sha(V5_LEDGER) == V5_LEDGER_SHA, "Exact committed seal/closed V5 ledger required")
    require(not ARCHIVE.is_symlink() and not SEAL.is_symlink(), "Symlink source rejected")
    seal = json.loads(SEAL.read_text())
    require(seal["schema"] == "research4-closed-rhf-checkpoint-archive-v1" and seal["completed_units"] == 149
            and seal["total_units"] == 150 and Path(seal["archive_directory"]).resolve() == ARCHIVE, "Unexpected healthy149 seal")
    dependencies = []
    for item in seal["dependencies"]:
        path = Path(item["path"])
        require(path.is_file() and not path.is_symlink() and sha(path) == item["sha256"], "Archive dependency changed")
        dependencies.append(record(path))
    std = {"ROOT": ROOT, "Path": Path, "json": json, "hashlib": hashlib, "math": math, "os": os}
    state = dict(std, __file__=str(STATE))
    state_nodes = extract(STATE, STATE_SHA, ("sha", "resolve_path", "checked_artifact"), state)
    worker = dict(std, __file__=str(V5), SCHEMA="research4-resumable-full-rks-hessian-result-v1",
                  EXECUTION_SCHEMA="research4-rhf-unit-producer-v1", SUCCESS={"DONE", "CHECKPOINT_SEGMENT_COMPLETE", "FREQUENCIES_COMPUTED"},
                  SUPERVISOR_SHA="33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee",
                  state_module=lambda: SimpleNamespace(checked_artifact=state["checked_artifact"], resolve_path=state["resolve_path"]))
    worker_nodes = extract(V5, V5_SHA, ("sha", "canonical", "healthy_closed_row", "verify_closed_execution"), worker)
    controller = dict(std, __file__=str(CONTROLLER))
    controller_nodes = extract(CONTROLLER, CONTROLLER_SHA, ("sha", "read", "checkpoint_records", "_copy_immutable"), controller)
    header = json.loads((ARCHIVE / "header.json").read_text())
    require(header["binding"]["stable_snapshot"]["snapshot_manifest_sha256"] == SNAPSHOT_SHA
            and hashlib.sha256(worker["canonical"](header["binding"]).encode()).hexdigest() == header["binding_sha256"], "Exact source MF/cache binding required")
    entries = controller["checkpoint_records"](ARCHIVE)
    require(len(entries) == 149 and [x[0] for x in entries] == list(range(149)), "Continuous healthy0..148 only")
    expected_files, producers, records = {"header.json": sha(ARCHIVE / "header.json")}, {}, []
    for index, path, unit in entries:
        require(unit["schema"] == header["schema"] and unit["scope"] == "ONE_COMPLETE_RHF_UNIT_NOT_A_COMPLETE_HESSIAN", "Only complete RHF units")
        evidence = unit["execution_evidence"]
        require(evidence["schema"] == worker["EXECUTION_SCHEMA"] and evidence["phase"] == "hessian-segment"
                and evidence["snapshot_manifest_sha256"] == SNAPSHOT_SHA and evidence["source_worker"]["sha256"] == V5_SHA,
                "Genuine unchanged V5 producer/source required")
        key = worker["canonical"](evidence)
        if key not in producers:
            producers[key] = worker["verify_closed_execution"](evidence, expected_snapshot_sha=SNAPSHOT_SHA)
        expected_files[str(path.relative_to(ARCHIVE))] = sha(path)
        shapes = {"jdiag": [3, 3]} if index < 75 else {"e1": [index-74, 3, 3], "ej": [index-74, 3, 3]}
        require(set(unit["arrays"]) == set(shapes), "Complete array coverage required")
        for name, shape in shapes.items():
            item = unit["arrays"][name]
            require(item["file"] == name+".npy" and item["payload"]["shape"] == shape
                    and item["payload"]["dtype"] == "float64" and item["payload"]["bytes"] == math.prod(shape)*8,
                    "Complete unit metadata shape/dtype differs")
            expected_files[str((path.parent / item["file"]).relative_to(ARCHIVE))] = item["file_sha256"]
        records.append({"index": index, "source_record": record(path), "producer": producers[key],
                        "qualification": "CLOSED_HEALTHY_V5_PRODUCER"})
    require(expected_files == seal["files"] and len(expected_files) == 373, "Copy precisely seal payload; exclude seal/unknown files")
    for relative, expected in expected_files.items():
        path = ARCHIVE / relative
        require(not path.is_symlink() and path.is_file() and sha(path) == expected, "Source archive payload changed")
    source_bytes = sum((ARCHIVE / name).stat().st_size for name in expected_files)
    input_refs = {"seal": record(SEAL), "closed_V5_ledger": record(V5_LEDGER), "header": record(ARCHIVE / "header.json"),
                  "V5_worker": record(V5), "V5_state_worker": record(STATE), "V5_controller": record(CONTROLLER)}
    start = time.monotonic(); lock_start = None
    global_path = ROOT / ".runtime/pyscf-v4-global.lock"
    with global_path.open("a+b") as global_handle:
        fcntl.flock(global_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_start = time.monotonic()
        require(not TARGET.exists() and sha(SEAL) == SEAL_SHA and sha(V5_LEDGER) == V5_LEDGER_SHA, "Target/source changed before exclusive copy")
        TARGET.mkdir(parents=True, exist_ok=False)
        with (TARGET / "cache.lock").open("xb") as cache_handle:
            fcntl.flock(cache_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for name in sorted(expected_files, key=lambda n: n.endswith("complete.json")):
                controller["_copy_immutable"](ARCHIVE / name, TARGET / name, expected_files[name])
            copied = controller["checkpoint_records"](TARGET)
            require(len(copied) == 149 and [x[0] for x in copied] == list(range(149)), "Copied prefix mismatch")
            require(not (TARGET / "units/000149-row-0074").exists(), "Never include failed final unit149")
            for relative, expected in expected_files.items():
                require(sha(TARGET / relative) == expected, "Target byte SHA differs")
            actual = {str(p.relative_to(TARGET)) for p in TARGET.rglob("*") if p.is_file()}
            require(actual == set(expected_files) | {"cache.lock"}, "Unknown/nonpayload target file")
            locked_seconds = time.monotonic()-lock_start
    for value in input_refs.values():
        require(sha(value["path"]) == value["sha256"], "Immutable source reference changed")
    require(not any(n.split(".")[0] in {"numpy", "scipy", "pyscf", "h5py"} for n in sys.modules), "Stdlib only")
    receipt = {"schema": "research4-healthy149-seed-relocation-v6b-v1", "status": "COPIED_VERIFIED_HEALTHY149_RHF_ONLY_NO_QUANTUM",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_commit_root_reported": "91909fe",
        "source_archive": str(ARCHIVE), "target_cache": str(TARGET), "input_references": input_refs,
        "archive_dependencies": dependencies, "source_snapshot_manifest_sha256": SNAPSHOT_SHA,
        "header_sha256": expected_files["header.json"], "binding_sha256": header["binding_sha256"],
        "copied_files": len(expected_files), "copied_payload_bytes": source_bytes, "completed_units": 149,
        "indices_included": list(range(149)), "unit149_included": False, "NPY_files_byte_hashed_not_parsed": 223,
        "complete_record_files": 149, "unique_healthy_V5_producers": len(producers), "unit_producers": records,
        "global_lock": {"path": str(global_path), "exclusive_nonblocking": True, "held_seconds": locked_seconds,
                        "target_cache_lock_held_inside": True, "released_before_receipt": True},
        "frozen_gate_execution": {"mode": "Unmodified pinned source function AST with stdlib-only frozen helper dependency surface",
            "V5_gate_AST_sha256": worker_nodes, "state_helper_AST_sha256": state_nodes, "controller_helper_AST_sha256": controller_nodes},
        "copied_file_sha256": expected_files, "seal_json_copied_into_cache": False,
        "forensic_release_used": False, "old_failed_cache_or_old_files_modified": False,
        "ledger_authorization_budget_modified": False, "new_Q_payload_launched": False, "numerical_libraries_imported": False,
        "NPY_values_parsed_or_new_finiteness_test": False, "Git_operations": 0,
        "full_RKS_partial_H1_response_frequencies": "NOT_RUN", "script": record(__file__),
        "elapsed_seconds_copy_and_final_checks": time.monotonic()-start}
    with RECEIPT.open("x") as handle:
        handle.write(json.dumps(receipt, indent=2, allow_nan=False)+"\n");handle.flush();os.fsync(handle.fileno())
    print(json.dumps({"status": receipt["status"], "files": len(expected_files), "bytes": source_bytes,
                      "producers": len(producers), "lock_seconds": locked_seconds, "receipt": record(RECEIPT)}))


if __name__ == "__main__":
    main()
