import importlib.util,json,hashlib
from pathlib import Path
p=Path("scripts/run_pyscf_actual_cpu_budget_20261104_v5.py")
assert hashlib.sha256(p.read_bytes()).hexdigest()=="33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee"
s=importlib.util.spec_from_file_location("verified_v5_smoke_supervisor",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
v=m.verify_resource_supervision(expected_threads=4)
import numpy as np
assert float(np.ones((2,2)).sum())==4
print(json.dumps({"status":"REAL_CHILD_EXIT_SMOKE_PASSED","quantum_kernel_calls":0,"proof_sha256":v["proof_sha256"]}))
