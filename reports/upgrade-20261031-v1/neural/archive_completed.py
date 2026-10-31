"""Archive the completed guarded neural run without fitting or reselection."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import gzip
import hashlib
import io
import json
import shutil
import tarfile


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


def archive(root,source,dest):
    if not (source/"scoring_manifest.json").exists():raise RuntimeError("Independent scoring must complete before final archive")
    dest.mkdir(parents=True,exist_ok=True);records=[]
    def copy(p,relative=None):
        target=dest/(p.relative_to(source) if relative is None else relative)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=p.read_bytes():raise FileExistsError(str(target))
        shutil.copyfile(p,target)
        assert sha(p)==sha(target)
        records.append({"original":str(p.relative_to(root)),"archived":str(target.relative_to(root)),"sha256":sha(p),"bytes":p.stat().st_size,"compression":"none"})
    for p in source.iterdir():
        if p.is_file() and p.name not in {"fit_scope_events.jsonl"}:
            copy(p)
    # Source and inference matrices are deterministic derivatives of the public
    # repository DB and saved IDs; do not mirror their roughly200MB again.
    prep=json.loads((source/"preparation.json").read_text())
    omitted=[]
    for context in prep["contexts"]:
        for key in ("source_pack","inference_pack"):
            p=root/context[key];assert sha(p)==context[key+"_sha256"]
            omitted.append({"path":context[key],"sha256":sha(p),"bytes":p.stat().st_size,"contains_source_labels":key=="source_pack","split_id":context["split_id"]})
        copy(root/context["scoring_truth"])
    write(dest/"omitted_context_matrices.json",{"reason":"Rebuild with prepare from source DB and fixed split; these derivatives are not new samples","files":omitted})
    for folder in ("pre_source_schema_v1","pre_source_architecture_v2"):
        for p in (source/folder).rglob("*"):
            if p.is_file():copy(p)
    inner=source/"selection"
    for outer in range(5):
        paths=sorted(p for p in inner.rglob("*") if p.is_file() and p.parts[len(inner.parts)].startswith(f"outer{outer:02d}."))
        output=dest/f"inner_outer_{outer:02d}_evidence.tar.gz"
        if output.exists():raise FileExistsError(str(output))
        with tarfile.open(output,"w:gz") as tar:
            for p in paths:tar.add(p,arcname=str(p.relative_to(source)))
        with tarfile.open(output,"r:gz") as tar:
            for p in paths:
                relative=str(p.relative_to(source));data=tar.extractfile(relative).read()
                assert hashlib.sha256(data).hexdigest()==sha(p)
                records.append({"original":str(p.relative_to(root)),"archive":str(output.relative_to(root)),"member":relative,"sha256":sha(p),"bytes":p.stat().st_size,"compression":"tar.gz"})
        assert output.stat().st_size<100*1024**2
    for p in (source/"evaluation").rglob("*"):
        if p.is_file():copy(p)
    source_log=source/"fit_scope_events.jsonl";compressed=dest/"fit_scope_events.jsonl.gz"
    compressed.write_bytes(gzip.compress(source_log.read_bytes(),mtime=0))
    assert hashlib.sha256(gzip.decompress(compressed.read_bytes())).hexdigest()==sha(source_log)
    records.append({"original":str(source_log.relative_to(root)),"archived":str(compressed.relative_to(root)),"original_sha256":sha(source_log),"archived_sha256":sha(compressed),"compression":"gzip"})
    allowed={c["split_id"]:set(c["source_ids"]) for c in prep["contexts"]};counts=Counter();components=Counter();by_scope=defaultdict(Counter)
    for line in source_log.read_text().splitlines():
        entry=json.loads(line);counts[entry["event"]]+=1;components[entry.get("component","")]+=1
        assert set(entry.get("fit_sample_ids",[]))<=allowed[entry["split_id"]]
        by_scope[entry["split_id"]][entry["event"]]+=1
    assert counts["fit_start"]==counts["fit_completed"] and not counts["fit_failed"] and not counts["fit_rejected"]
    write(dest/"fit_scope_manifest.json",{"status":"INSTRUMENTED_SCOPE_CHECK_PASS","study_status":"EXPLORATORY","events":dict(counts),"component_event_counts":dict(components),"source_scopes":[{"split_id":c["split_id"],"allowed_ids":c["source_ids"],"event_counts":dict(by_scope[c["split_id"]])} for c in prep["contexts"]],"raw_log_sha256":sha(source_log),"raw_log_gzip_sha256":sha(compressed),"all_actual_logged_fit_ids_within_allowed_source":True,"uninstrumented_external_code_certified_absent":False})
    manifest=json.loads((source/"run_manifest.json").read_text())
    for relative,expected in manifest["source_sha256"].items():
        p=root/relative;assert sha(p)==expected;target=dest/"source_snapshot"/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    copy(root/"data/research/upgrade_neural_protocol_20261031.json","protocol_snapshot.json")
    copy(root/"outputs/upgrade-neural-boundary-tests.log","boundary-tests.log")
    write(dest/"archive_validation.json",{"status":"BYTE_COPIES_VERIFIED","records":records,"raw_context_matrices_not_mirrored":len(omitted),"inner_weight_archives":5,"outer_complete_pipeline_files":len(list((dest/"evaluation").rglob("pipeline.joblib"))),"new_model_fits":0})
    print(json.dumps({"status":"archived","original_files_verified":len(records),"outer_models":120,"fit_calls_completed":counts["fit_completed"],"omitted_context_matrices":len(omitted)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--source",type=Path,required=True);parser.add_argument("--dest",type=Path,required=True)
    args=parser.parse_args();archive(Path.cwd(),args.source.resolve(),args.dest.resolve())
