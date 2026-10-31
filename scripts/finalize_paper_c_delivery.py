"""Check the completed paper-C delivery and build a small reading bundle.

This validates stored artifacts only. It never trains or selects a model.
"""
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
import zipfile


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/paper-c/2026-10-31"
DOCS = [ROOT / "docs" / name for name in (
    "RESEARCH4_PAPER_C_MASTER_PLAN.md", "RESEARCH4_EXPERIMENT_AUDIT.md", "RESEARCH4_PAPER_OUTLINE.md")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files(root):
    return sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


def main():
    index_checks = 0
    for name in ("external-data", "calibration", "strong-mlp", "paper-assets"):
        directory = REPORT / name
        index = json.loads((directory / "files_sha256.json").read_text())
        for relative, expected in index.items():
            assert sha(directory / relative) == expected, str(directory / relative)
            index_checks += 1
    best = REPORT / "best-candidate"
    value = json.loads((best / "validation.json").read_text())
    for key, directory in (("package_files_sha256", best), ("delivery_source_sha256", ROOT)):
        for relative, expected in value[key].items():
            assert sha(directory / relative) == expected, str(directory / relative)
            index_checks += 1
    fair = json.loads((REPORT / "fair-benchmark/archive_validation.json").read_text())
    for entry in fair["copied_files"]:
        p = Path(entry["archived"])
        assert sha(p) == entry["archived_sha256"], str(p)
        if entry["compressed"]:
            assert hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest() == entry["original_sha256"]
        else:
            assert entry["original_sha256"] == entry["archived_sha256"]
        index_checks += 1

    source_checks = 0
    protocol_hashes = {}
    for name, manifest_relative in (
        ("fair-benchmark", "run_manifest.json"),
        ("strong-mlp", "run_manifest.json"),
        ("calibration", "run/run_manifest.json"),
    ):
        manifest = json.loads((REPORT / name / manifest_relative).read_text())
        for relative, expected in manifest.get("source_sha256", manifest.get("source_code_sha256", {})).items():
            assert sha(ROOT / relative) == expected, relative
            source_checks += 1
        protocol_hashes[name] = manifest["protocol_sha256"]
    release = json.loads((REPORT / "evaluation_release.json").read_text())
    for item in release["experiments"]:
        assert sha(ROOT / item["protocol"]) == item["protocol_sha256"] == protocol_hashes[item["name"]]
        assert not item["new_evaluation_artifact_existed_at_release"]

    workbooks = [p for p in files(REPORT) if p.suffix == ".xlsx"]
    for p in workbooks:
        with zipfile.ZipFile(p) as archive:
            assert archive.testzip() is None, str(p)
    for p in [*ROOT.glob("src/herb_spectra/paper_c_*.py"), *ROOT.glob("scripts/*paper_c*.py"),
              *ROOT.glob("tests/test_paper_c_*.py")]:
        compile(p.read_text(), str(p), "exec")
    test_log = (REPORT / "unit-tests.log").read_text()
    assert "Ran 292 tests" in test_log and test_log.rstrip().endswith("OK")
    command_rows = [json.loads(line) for line in (REPORT / "command_invocations.jsonl").read_text().splitlines()]
    completed = [r for r in command_rows if r["event"] == "command_completed"]
    assert len(completed) == 4 and all(row["returncode"] == 0 for row in completed)

    markdown = [ROOT / "README.md", *DOCS, *REPORT.rglob("*.md")]
    links = 0
    for p in markdown:
        for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", p.read_text()):
            target = raw.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            path = Path(unquote(target))
            if not path.is_absolute():
                path = p.parent / path
            assert path.exists(), (str(p), raw)
            links += 1
    original_paths = set(subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "b89c328"], cwd=ROOT, text=True).splitlines())
    changed = [p for p in subprocess.check_output(
        ["git", "diff", "--name-only", "b89c328"], cwd=ROOT, text=True).splitlines() if p in original_paths]
    assert set(changed) <= {".gitattributes", "AGENTS.md", "README.md"}, changed
    all_files = files(REPORT)
    assert max(p.stat().st_size for p in all_files) < 100 * 1024**2
    validation = {"status":"pass", "checked_at_utc":datetime.now(timezone.utc).isoformat(),
                  "owner_archive_entries_checked":index_checks,"frozen_source_hash_checks":source_checks,
                  "local_markdown_links_checked":links,"xlsx_crc_checked":len(workbooks),
                  "completed_commands_exit0":len(completed),"project_tests_passed":292,
                  "preexisting_tracked_files_changed":changed,"new_model_fits":0,
                  "protocol_hashes":protocol_hashes,"document_sha256":{str(p.relative_to(ROOT)):sha(p) for p in DOCS},
                  "delivery_script_sha256":sha(Path(__file__))}
    (REPORT / "delivery_validation.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2)+"\n")
    index = {str(p.relative_to(REPORT)):sha(p) for p in files(REPORT) if p.name != "files_sha256.json" or p.parent != REPORT}
    (REPORT / "files_sha256.json").write_text(json.dumps(index,ensure_ascii=False,indent=2)+"\n")

    bundle = ROOT / "outputs/paper-c-20261031/方案C_报告与论文图表.zip"
    bundle.parent.mkdir(parents=True,exist_ok=True)
    selections = [*DOCS, REPORT / "README.md", *files(REPORT / "paper-assets"),
                  REPORT / "history-audit/main_results.csv", REPORT / "fair-benchmark/model_comparison.csv"]
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for p in selections:
            archive.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
    print(json.dumps({**validation,"root_index_entries":len(index),"reading_bundle":str(bundle.relative_to(ROOT)),
                      "reading_bundle_bytes":bundle.stat().st_size},ensure_ascii=False))


if __name__ == "__main__":
    main()
