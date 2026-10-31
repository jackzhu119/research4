"""Inspect research runtime capabilities without printing credentials or environment values."""
import argparse
import ctypes.util
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo


def _run_json(command, marker, timeout=60):
    try:
        cp = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout)
        matches = re.findall(re.escape(marker) + r"([^\n]+)", cp.stdout)
        if cp.returncode == 0 and matches:
            return json.loads(matches[-1])
        return {"available": False, "error": "probe_failed", "returncode": cp.returncode}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {"available": False, "error": type(exc).__name__}


def _candidate(runtime, name, alternatives=()):
    local = runtime / "bin" / name
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    for name in alternatives:
        found = shutil.which(name)
        if found:
            return found
    return None


def inspect_environment(runtime, spec, matlab=None, probe_matlab=False):
    runtime = Path(runtime).resolve()
    python = _candidate(runtime, "python") or sys.executable
    packages = {**spec["python_packages"], **spec["torch_packages"]}
    code = """import importlib,importlib.metadata as md,json,sys,ctypes,ctypes.util
packages=PACKAGES
names=IMPORTS
result={'available':True,'version':sys.version.split()[0],'packages':{},'imports':{}}
for name in packages:
 try:result['packages'][name]=md.version(name)
 except md.PackageNotFoundError:result['packages'][name]=None
for name,module in names.items():
 try:importlib.import_module(module);result['imports'][name]=True
 except Exception as exc:result['imports'][name]=type(exc).__name__
try:
 import torch
 result['torch_cpu']=not torch.cuda.is_available()
except Exception:result['torch_cpu']=None
try:
 path=ctypes.util.find_library('archive');ctypes.CDLL(path) if path else (_ for _ in ()).throw(OSError())
 result['libarchive']=True
except Exception:result['libarchive']=False
print('DOCTOR_JSON='+json.dumps(result))
""".replace("PACKAGES", repr(packages)).replace("IMPORTS", repr({**spec["python_imports"], "torch": "torch"}))
    py = _run_json([python, "-c", code], "DOCTOR_JSON=")
    py["executable"] = python
    py["locked_versions_match"] = all(py.get("packages", {}).get(k) == v for k, v in packages.items())
    py["core_ready"] = all(py.get("packages", {}).get(k) == v and py.get("imports", {}).get(k) is True
                           for k, v in spec["python_packages"].items())
    py["cnn_ready"] = py.get("packages", {}).get("torch") == spec["torch_packages"]["torch"] and py.get("imports", {}).get("torch") is True

    octave = _candidate(runtime, "octave", ("octave-cli", "octave"))
    if octave:
        expression = "warning('off','Octave:shadowed-function'); names={'statistics','signal','io','image','optim'}; loaded=struct(); for i=1:numel(names);try;pkg('load',names{i});loaded.(names{i})=true;catch;loaded.(names{i})=false;end;end; p=pkg('list');versions=struct();for i=1:numel(p);versions.(p{i}.name)=p{i}.version;end;apis={'pca','kmeans','plsregress','xlsread','svmtrain','svmpredict','imfilter','templateSVM','fitcecoc','newff','network','categorical','trainNetwork','trainlssvm','my_pca'};available=struct();for i=1:numel(apis);available.(apis{i})=exist(apis{i},'file')>0;end;fprintf('DOCTOR_JSON=%s\\n',jsonencode(struct('available',true,'version',version,'loaded',loaded,'packages',versions,'apis',available)));"
        oct = _run_json([octave, "--eval", expression], "DOCTOR_JSON=")
        oct["executable"] = octave
    else:
        oct = {"available": False, "error": "executable_not_found", "executable": None}
    oct["ready"] = bool(oct.get("available") and oct.get("version") == spec["octave_version"]
                        and all(oct.get("packages", {}).get(k) == v and oct.get("loaded", {}).get(k) is True
                                for k, v in spec["octave_packages"].items()))

    matlab = str(Path(matlab).resolve()) if matlab else _candidate(runtime, "matlab", ("matlab",))
    mat = {"executable_found": bool(matlab), "executable": matlab, "native_execution_verified": False,
           "license_status": "not_configured_or_not_verified", "installed_by_this_bootstrap": False}
    if matlab and probe_matlab:
        expression = "r=struct('available',true,'version',version,'statistics',logical(license('test','Statistics_Toolbox')),'deep_learning',logical(license('test','Neural_Network_Toolbox')));fprintf('DOCTOR_JSON=%s\\n',jsonencode(r));"
        probe = _run_json([matlab, "-batch", expression], "DOCTOR_JSON=", timeout=90)
        mat["probe"] = probe
        mat["native_execution_verified"] = probe.get("available") is True
        mat["license_status"] = "probe_completed" if mat["native_execution_verified"] else "probe_failed"
    return {"schema_version": 1, "checked_at_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "runtime_dir": str(runtime), "python": py, "octave": oct, "matlab": mat,
            "open_source_ready": bool(py["core_ready"] and py["cnn_ready"] and py["locked_versions_match"]
                                      and py.get("libarchive") and oct["ready"]),
            "all_original_code_ready": False,
            "known_external_or_code_requirements": [
                "Native MATLAB execution needs a valid user/institution license and required toolboxes",
                "Missing trainlssvm/my_pca and incomplete stored algorithms are not supplied by installing Octave",
                "Compatibility adapters and logic patches remain distinct from unchanged original scripts"
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path(".runtime"))
    parser.add_argument("--spec", type=Path, default=Path(__file__).resolve().parents[1] / "environment/runtime-spec.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--matlab", type=Path)
    parser.add_argument("--probe-matlab", action="store_true", help="Explicitly start an existing MATLAB to test its licensed runtime")
    args = parser.parse_args()
    result = inspect_environment(args.runtime_dir, json.loads(args.spec.read_text()), args.matlab, args.probe_matlab)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["open_source_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
