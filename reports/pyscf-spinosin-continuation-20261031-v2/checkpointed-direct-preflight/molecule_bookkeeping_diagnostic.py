"""Mole-only negative control: raw serialization changes without physical change."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
for site in (".runtime/quantum-upgrade-site", ".runtime/quantum-pilot-site"):
    sys.path.insert(0,str(ROOT/site))
from pyscf import gto
from pyscf.gto import mole

spec=importlib.util.spec_from_file_location("V5_identity",ROOT/"scripts/pyscf_spinosin_checkpointed_direct_worker_20261031.py")
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
mol=gto.M(atom="O 0 0 .2;H 0 .7 1;H 0 -.7 1",basis="sto-3g",verbose=0)
before,env,coords=mol.dumps(),mol._env.copy(),mol.atom_coords().copy()
identity=worker.molecule_cache_identity(mol)
with mol.with_rinv_at_nucleus(1):
    pass
after=mol.dumps();changed=np.where(env!=mol._env)[0].tolist()
result={"status":"DONE_MOLE_ONLY_BINDING_NEGATIVE_CONTROL", "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "integral_SCF_gradient_Hessian_calls":0, "wavefunctions_constructed":0,
        "molecule_objects_constructed":1,"source_mole_py_sha256":hashlib.sha256(Path(mole.__file__).read_bytes()).hexdigest(),
        "naive_raw_serialization_invariance_passed":before==after,
        "original_added_unittest_assumption_failed_before_canonicalization":True,
        "changed_environment_indices":changed,"before_changed_values":env[changed].tolist(),
        "after_changed_values":mol._env[changed].tolist(),
        "physical_coordinates_unchanged":bool(np.array_equal(coords,mol.atom_coords())),
        "canonical_identity_preserved":identity==worker.molecule_cache_identity(mol),
        "canonicalization_does_not_modify_Mole_marker":bool(mol._env[17]==1),
        "raw_before_sha256":hashlib.sha256(before.encode()).hexdigest(),
        "raw_after_sha256":hashlib.sha256(after.encode()).hexdigest()}
assert changed==[17] and before!=after and result["canonical_identity_preserved"]
p=Path(__file__).with_name("molecule_bookkeeping_diagnostic.json")
if p.exists():raise FileExistsError("Preserve this actual metadata diagnostic")
p.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
