"""Bounded, stdlib-only axis-member audit. No spectra, labels, imports or fits."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import struct
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_AXIS = "fbf09eefe1c1970452b8b33c4dd91400768e3dc1d0065a9ac6e394824fb95887"


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")


def axis_member(path):
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("axis.npy")
        if info.file_size >= 100000: raise ValueError("Axis member exceeds diagnostic limit")
        blob = archive.read("axis.npy")
    if blob[:6] != b"\x93NUMPY" or blob[6] not in (1, 2): raise ValueError("Unexpected NPY header")
    start = 10 if blob[6] == 1 else 12
    length = struct.unpack("<H" if start == 10 else "<I", blob[8:start])[0]
    offset = start + length
    header = ast.literal_eval(blob[start:offset].decode("latin1"))
    if header != {"descr": "<f8", "fortran_order": False, "shape": (6750,)}: raise ValueError("Expected axis-only float64 vector")
    values = list(struct.unpack("<6750d", blob[offset:]))
    payload_sha = hashlib.sha256(blob[offset:]).hexdigest()
    if payload_sha != EXPECTED_AXIS: raise ValueError("Unexpected registered native axis")
    return values, blob, payload_sha


def profile(axis):
    delta = [b - a for a, b in zip(axis, axis[1:])]
    if not all(math.isfinite(v) for v in axis) or min(delta) <= 0: raise ValueError("Invalid increasing time coordinates")
    step = (axis[-1] - axis[0]) / (len(axis) - 1)
    uniform = [axis[0] + i * step for i in range(len(axis))]
    uniform[-1] = axis[-1]
    coordinate_error = max(abs(a - b) for a, b in zip(axis, uniform))
    step_error = max(abs(d - step) for d in delta)
    f32_error = max(abs(v - struct.unpack("<f", struct.pack("<f", v))[0]) for v in axis)
    return {"points": len(axis), "start": axis[0], "end": axis[-1], "first10": axis[:10], "last10": axis[-10:],
        "endpoint_uniform_step": step, "uniform_axis_sha256_raw_float64": hashlib.sha256(struct.pack("<6750d", *uniform)).hexdigest(),
        "delta_min": min(delta), "delta_max": max(delta), "delta_mean": statistics.fmean(delta),
        "delta_median": statistics.median(delta), "delta_population_SD": statistics.pstdev(delta),
        "max_native_uniform_coordinate_deviation": coordinate_error, "max_coordinate_deviation_in_steps": coordinate_error / step,
        "max_native_uniform_step_deviation": step_error, "max_relative_step_deviation": step_error / step,
        "max_distance_to_float32_coordinate": f32_error,
        "float32_quantization_interpretation": "COORDINATES_CONSISTENT_WITH_FLOAT32_EXPORT_ROUNDING_NOT_PROOF_OF_PHYSICAL_CLOCK_UNIFORMITY",
        "rounded10_delta_counts": dict(Counter(format(v, ".10g") for v in delta)),
        "repair": "FIXED_WITHIN_SPAN_LINEAR_INTERPOLATION_TO_6750_ENDPOINT_UNIFORM_POINTS_BEFORE_UNCHANGED_POLYPHASE",
        "time_unit": "NOT_EXPLICITLY_DECLARED"}


def worker(output):
    resource.setrlimit(resource.RLIMIT_CPU, (20, 21))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2,) * 2)
    signal.alarm(45)
    started = utc()
    registry_path = ROOT / "data/supplemental/teacher_20261104_v2/merged_registry.json"
    registry = json.loads(registry_path.read_text())
    paths = [ROOT / "data/multiplant_database/zaoren/time.npz"] + [ROOT / d["artifact"] for d in registry["datasets"] if d["modality"] == "time"]
    if len(paths) != 4: raise ValueError("Exactly four original time-axis families required")
    records = []
    for path in paths:
        axis, blob, payload_sha = axis_member(path)
        records.append({"path": str(path.relative_to(ROOT)), "axis_npy_sha256": hashlib.sha256(blob).hexdigest(),
            "native_axis_sha256_raw_float64": payload_sha, "axis_npy_bytes": len(blob), "profile": profile(axis)})
    save(output / "axis_diagnostic.json", {"status": "AXIS_ONLY_DIAGNOSTIC_DONE", "started_at_utc": started,
        "ended_at_utc": utc(), "versions": {"python": sys.version, "stdlib_only": True}, "script_sha256": sha(__file__),
        "registry_path": str(registry_path.relative_to(ROOT)), "registry_sha256": sha(registry_path), "records": records,
        "NPZ_members_read": ["axis.npy"], "spectra_or_record_ids_or_query_labels_read": False, "model_fits": 0,
        "whole_NPZ_hash_verified_in_this_diagnostic": False,
        "numerical_packages_imported": any(k.split(".")[0] in {"numpy", "scipy", "torch", "pyscf"} for k in sys.modules)})


def supervise(output):
    if output.exists(): raise ValueError("Fresh audit directory required")
    output.mkdir(parents=True)
    started, wall0, cpu0 = utc(), time.monotonic(), time.process_time()
    if hasattr(os, "sched_getaffinity"): os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[key] = "1"
    lock = Path("/workspace/research4/.runtime/pyscf-v4-global.lock")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output)]
        with (output / "worker.log").open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            killed = False
            while True:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid: break
                if time.monotonic() - wall0 > 60:
                    os.killpg(process.pid, signal.SIGKILL); killed = True
                    pid, status, usage = os.wait4(process.pid, 0); break
                time.sleep(.02)
        code = os.waitstatus_to_exitcode(status); process.returncode = code
    child_cpu = usage.ru_utime + usage.ru_stime
    supervisor_cpu = time.process_time() - cpu0
    result = {"status": "DONE_AXIS_AUDIT_NO_FITS" if code == 0 and not killed else "FAILED_CLOSED_KEEP_ARTIFACTS",
        "started_at_utc": started, "ended_at_utc": utc(), "child_PID": process.pid, "exit_code": code,
        "wall_terminated": killed, "actual_wait4_child_CPU_seconds": child_cpu,
        "observed_supervisor_process_CPU_seconds": supervisor_cpu,
        "CPU_charge_seconds": child_cpu + supervisor_cpu,
        "CPU_accounting_scope": "exact_wait4_child_plus_observed_supervisor_through_receipt_before_final_write_return",
        "wall_seconds": time.monotonic() - wall0, "limits": {"child_CPU_seconds": 20, "wall_seconds": 60,
            "threads": 1, "AS_bytes": 256 * 1024**2}, "peak_child_RSS_kib": usage.ru_maxrss,
        "command": command, "script_sha256": sha(__file__), "worker_log_sha256": sha(output / "worker.log"),
        "model_fits": 0, "numerical_packages_imported_by_supervisor": False}
    if code == 0: result["axis_diagnostic_sha256"] = sha(output / "axis_diagnostic.json")
    save(output / "execution_receipt.json", result)
    print(json.dumps(result))
    return 0 if result["status"] == "DONE_AXIS_AUDIT_NO_FITS" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(); output = args.output.resolve()
    if not output.is_relative_to(ROOT / "reports"): raise ValueError("Report output required")
    if args.worker: worker(output); return 0
    return supervise(output)


if __name__ == "__main__": raise SystemExit(main())
