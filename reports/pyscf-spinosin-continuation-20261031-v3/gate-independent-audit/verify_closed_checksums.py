"""Verify closed gate result/cache hashes only; no quantum module imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/"reports/pyscf-spinosin-continuation-20261031-v3"
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    evidence=[]
    water=BASE/"water-native4-validation-001"
    ethanol=BASE/"validation-ethanol-native4-001"
    folders=[water]+[ethanol/label for label in ("standard_full_native1","response_block4_native4","response_block8_native4")]
    bindings={"raw_hessian_sha256":"hessian_hartree_per_bohr2.npy",
              "analysis_hessian_sha256":"hessian_used_for_analysis.npy",
              "normal_modes_file_sha256":"normal_modes.npy",
              "normal_mode_geometry_angstrom_npy_sha256":"normal_mode_geometry_angstrom.npy",
              "optimized_coordinates_angstrom_npy_sha256":"optimized_coordinates_angstrom.npy",
              "optimized_xyz_sha256":"optimized.xyz", "checkpoint_sha256":"calculation.chk",
              "hessian_cache_manifest_sha256":"hessian_cache_manifest.json"}
    for folder in folders:
        result=json.loads((folder/"result.json").read_text())
        for key,name in bindings.items():
            if key in result:
                path=folder/name
                actual=sha(path)
                assert actual==result[key], str(path)
                evidence.append({"type":"result_SHA_binding","path":str(path.relative_to(ROOT)),"sha256":actual,"result_key":key})
        manifest_path=folder/"hessian_cache_manifest.json"
        if not manifest_path.exists():continue
        manifest=json.loads(manifest_path.read_text())
        assert manifest["status"]=="DONE_INTERMEDIATE_CAPTURE_ONLY"
        assert manifest["response_complete_global_atom_ids"]==list(range(result.get("atoms",3 if folder==water else 9)))
        binding=manifest["binding"]
        assert binding["actual_MF_max_memory_during_Hessian_mb"]==3000 and binding["actual_Hessian_max_memory_mb"]==3000
        assert binding["settings"]["density_fitting"] is False
        def visit(value):
            if isinstance(value,dict):
                if "file_sha256" in value:
                    path=Path(value["path"]);actual=sha(path);assert actual==value["file_sha256"]
                    array=np.load(path,allow_pickle=False)
                    assert np.isfinite(array).all() and str(array.dtype)==value["dtype"]
                    assert list(array.shape)==value["shape"]
                    payload=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                    assert payload==value["payload_sha256_C_order"]
                    evidence.append({"type":"closed_cache_SHA_shape_finite_payload","path":str(path.relative_to(ROOT)),"sha256":actual,"shape":list(array.shape)})
                for child in value.values():visit(child)
            elif isinstance(value,list):
                for child in value:visit(child)
        visit(manifest)
    target=OUT/"checksum_crosscheck.json";assert not target.exists()
    target.write_text(json.dumps({"status":"PASS","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "quantum_computations":0,"active_checkpoint_reads":0,"closed_checkpoint_hashing_only":True,
        "source_script_sha256":sha(__file__),"validated_file_count":len(evidence),"checks":evidence},indent=2)+"\n")
    print(json.dumps({"status":"PASS","validated_file_count":len(evidence),"audit_sha256":sha(target)}))


if __name__=="__main__":main()
