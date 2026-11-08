#!/usr/bin/env python3
"""Qualify the final license-header-only revision and repeat pure tests.

No quantum calculations are performed. Earlier receipts stay unchanged.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "scripts/pyscf_incremental_rhf_checkpoint_20261103.py"
OLD_SHA = "88a9f150ae1b5ba889c5b4a9ea177205466fd9a5873d401ca97d68bf65e25c2e"
FINAL_SHA = "ce96055f24b39833337844c178f94597a9ac4ba7721ba019816d03628f971f55"


def sha(data):
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    source = SOURCE.read_bytes()
    assert sha(source) == FINAL_SHA
    original_start = source.index(b'"""Restartable pure-PBE RHF partial Hessian')
    assert sha(source[original_start:]) == OLD_SHA, "Revision is not just a prepended notice"
    output = HERE / "licensed-final"
    output.mkdir()
    spec = importlib.util.spec_from_file_location("final_licensed_checkpoint_review",
                                                HERE / "review_implementation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = output
    module.parent()
    assert sha(SOURCE.read_bytes()) == FINAL_SHA
    receipt = {
        "status": "FINAL_LICENSED_SOURCE_STATIC_AND_PURE_REVIEW_PASS_REAL_QM_PENDING",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": FINAL_SHA, "unchanged_original_body_sha256": OLD_SHA,
        "only_prepend_license_header_verified": True,
        "wrapper_script_sha256": sha(Path(__file__).read_bytes()),
        "new_quantum_evaluations": 0,
        "water_ethanol_real_quantum_gate": "NOT_RUN",
    }
    (output / "licensed_source_qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
