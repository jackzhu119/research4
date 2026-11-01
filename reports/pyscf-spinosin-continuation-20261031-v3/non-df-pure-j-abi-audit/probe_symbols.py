#!/usr/bin/env python3
"""Resolve symbols only; do not import PySCF or call integral functions."""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
SITE = ROOT / ".runtime/quantum-upgrade-site/pyscf"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for name in ("symbols.json", "README.md", "index.json"):
        if (OUTPUT / name).exists():
            raise FileExistsError("Existing evidence refused: " + name)
    expected = {
        "lib/libcvhf.so": "5d7461e3976fd40fce7c9ba571e45b01830dd018d75927e389c32a6f4d09df0c",
        "lib/deps/lib/libcint.so": "950bb0a297507369dce95b9013ef2c5b64b0fa36a0018c303c98e95efce7db7e",
        "hessian/rhf.py": "2d986bfc01aac98447034a207df639b3d6897f67ce1ab2a013fe2d0853c85de5",
        "hessian/rks.py": "d61dbd1727b286845737853b92e1812a3e6a966dca29a9a8a79ff5e9a7725c03",
        "scf/_vhf.py": "ef0a04f43338780a7f57d9268f9634c1079ae989ba3030d6154c8612798f82ba",
    }
    evidence = {}
    for relative, digest in expected.items():
        path = SITE / relative
        if path.is_symlink() or sha(path) != digest:
            raise ValueError("Fixed source/library differs: " + relative)
        evidence[str(path.relative_to(ROOT))] = {"sha256": digest, "bytes": path.stat().st_size}
    cgto = SITE / "lib/libcgto.so"
    evidence[str(cgto.relative_to(ROOT))] = {"sha256": sha(cgto), "bytes": cgto.stat().st_size}
    # Loading resolves dependency symbols; no returned function is called.
    library = ctypes.CDLL(str(SITE / "lib/libcvhf.so"))
    names = ("int2e_ipip1ipip2", "int2e_ip1ip2", "int2e_ipvip1ipvip2")
    observed = [{"requested_guard_symbol": name,
        "unsuffixed_hasattr": hasattr(library, name),
        "sph_hasattr": hasattr(library, name + "_sph"),
        "cart_hasattr": hasattr(library, name + "_cart")} for name in names]
    imported = sorted(name for name in sys.modules if name == "pyscf" or name.startswith("pyscf."))
    if imported:
        raise ValueError("Unexpected PySCF import: " + repr(imported))
    all_none = all(not row["unsuffixed_hasattr"] for row in observed)
    if not all_none or not all(row["sph_hasattr"] and row["cart_hasattr"] for row in observed):
        raise ValueError("Observed ABI differs; do not write the fixed-current-build disclosure")
    payload = {"schema": "research4-libcvhf-symbol-resolution-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DONE_ABI_SYMBOL_LOOKUP_ONLY",
        "python": sys.version, "platform": platform.platform(),
        "loader": "ctypes.CDLL on exact installed libcvhf.so; hasattr only, including its dependency scope",
        "symbols": observed,
        "current_make_vhfopt_guards_all_return_None_before_construction": all_none,
        "scope_of_conclusion": "Current exact library/source hashes only; no claim about other PySCF builds.",
        "source_control_flow": "hessian/rhf.py:180-183 returns None when hasattr(libcvhf, vhf_intor) is False; its dms argument is not used before this return.",
        "current_partial_hessian_dm_cond_branch": "NOT_ENTERED_FOR_THESE_THREE_HELPERS" if all_none else "REQUIRES_ADDITIONAL_SCREENING_REVIEW",
        "candidate_implication": "Dropping unused exchange contractions still requires separate new-code whole-Hessian validation. This ABI evidence is not a speed, memory or numerical-equivalence result.",
        "screening_limits": "Do not silently enable screening, change thresholds, or generalize the None result to a build exposing bare symbols.",
        "direct_driver_cost_limit": "J/K outputs share integral generation; native thread scratch reservation and AO traversal need not fall in proportion to removed outputs.",
        "explicit_library_numerical_function_calls": 0, "integral_evaluations": 0,
        "new_SCF_gradient_Hessian_evaluations": 0, "PySCF_imported_modules": imported,
        "Mole_instances_created": 0, "make_vhfopt_calls": 0,
        "shared_library_initializers_may_run_on_dlopen": True,
        "active_calculation_files_read": 0, "active_process_signals": 0,
        "files_or_limits_modified_outside_this_new_evidence_directory": False,
        "isolated_probe_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "source_evidence": evidence, "probe_source_sha256": sha(Path(__file__))}
    text = """# Non-DF partial-Hessian symbol guard audit

This is an ABI-only lookup on the installed PySCF 2.14 library. It imports no
PySCF module and creates no molecule, optimizer, density, integral or Hessian.
`ctypes.CDLL` can execute ordinary shared-library initializers; no exported
numerical function is invoked. Only symbol resolution via `hasattr` is used.

For all three exact unsuffixed names requested by `rhf._make_vhfopt`, the installed
library handle reports `False`; their `_sph` and `_cart` variants report `True`.
Thus the current frozen helper returns `None` at rhf.py:182-183, before creating
the screening object, for ipip1/ip1ip2/ipvip1. The suffix distinction is real,
not inferred solely from nm on one library that might miss dependencies.

This removes the `vhfopt.set_dm` concern for these three current calls only.
It does not authorize repairing a different build's screening or removing
Coulomb terms. A pure-PBE candidate may omit unused exchange outputs only in
new code, after full water/ethanol Hessian and frequency equivalence tests.
No such candidate is implemented or numerically validated here. Shared integral
generation means removing outputs does not guarantee proportional time or
scratch-memory savings.

All library/source hashes and individual booleans are in `symbols.json`; the
index seals those results and this probe. No active calculation file was read.
The per-probe OMP/BLAS environment is recorded and does not change a running job.
"""
    for relative, item in evidence.items():
        if sha(ROOT / relative) != item["sha256"]:
            raise ValueError("Source changed during symbol lookup")
    with (OUTPUT / "symbols.json").open("x") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (OUTPUT / "README.md").open("x") as handle:
        handle.write(text)
    files = [OUTPUT / "README.md", Path(__file__).resolve(), OUTPUT / "symbols.json"]
    index = {"status": "FROZEN_ABI_ONLY_NO_NUMERICAL_FUNCTION_CALLS", "files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(files)]}
    with (OUTPUT / "index.json").open("x") as handle:
        json.dump(index, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"observed": observed, "all_three_guard_returns_None": all_none, "index_sha256": sha(OUTPUT / "index.json")}))


if __name__ == "__main__":
    main()
