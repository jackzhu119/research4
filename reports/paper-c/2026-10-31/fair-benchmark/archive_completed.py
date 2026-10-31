"""Copy the finished benchmark byte-for-byte; gzip only large raw JSON."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    dest = Path(__file__).resolve().parent
    status = json.loads((run/"status.json").read_text())
    assert status["status"] == "completed"
    copied = []
    for path in run.rglob("*"):
        if not path.is_file() or "selection" in path.relative_to(run).parts:
            continue
        relative = path.relative_to(run)
        out = dest/relative
        out.parent.mkdir(parents=True, exist_ok=True)
        if relative.as_posix() == "all_inner_metrics.json":
            out = out.with_name(out.name+".gz")
            raw = path.read_bytes()
            out.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
            assert gzip.decompress(out.read_bytes()) == raw
        else:
            shutil.copyfile(path, out)
            assert sha(out) == sha(path)
        copied.append({"original":str(path), "archived":str(out), "original_sha256":sha(path),
                       "archived_sha256":sha(out), "compressed":out.suffix == ".gz" and path.suffix != ".gz"})
    combined = json.loads((run/"all_inner_metrics.json").read_text())
    reconstructable = []
    for fold in range(5):
        path = run/"selection"/f"outer_{fold:02d}.json"
        rows = [r for r in combined if r["outer_fold"] == fold]
        raw = (json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()
        assert raw == path.read_bytes()
        reconstructable.append({"original":str(path), "sha256":sha(path), "reconstruction":"filter all_inner_metrics by outer_fold; json.dumps ensure_ascii=False indent=2 allow_nan=False plus newline"})
    result = {"status":"pass", "copied_files":copied, "per_fold_records_reconstructable_exactly":reconstructable,
              "new_model_fits":0, "models_copied":sum(r["archived"].endswith(".joblib") for r in copied)}
    (dest/"archive_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status":"pass", "files":len(copied), "models":result["models_copied"], "new_model_fits":0}))


if __name__ == "__main__":
    main()
