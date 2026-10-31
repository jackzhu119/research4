#!/usr/bin/env python3
"""Download author-published public plant THz sources without executing source code.

Unknown-license originals stay locally ignored; manifests are suitable for Git.
The inherited network policy and proxy are intentionally left unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/public-thz-20261031-v1/acquisition"
LOCAL = ROOT / "data/external/public_thz_20261031_v1/acquired"
FRUIT_COMMIT = "16943a2c499ce7918afb1e20a223267d94ad7a93"
LEAF_REPO = "hasantahir/Characterization-and-Water-Content-Estimation-Method-of-Living-Plant-Leaves-Using-Terahertz-Waves"
LEAF_COMMIT = "866b9c229e07ce7bec05861d94f9797ddf5da654"


def fetch(record: dict, max_bytes: int = 200 * 1024 * 1024) -> dict:
    rec = dict(record)
    rec["attempted_utc"] = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    dest = LOCAL / record["dataset_id"] / record["filename"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    # No redistribution license has been established for these author repositories.
    (dest.parent / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    rec["local_path"] = str(dest.relative_to(ROOT))
    try:
        if dest.exists():
            rec["http_status"] = "CACHED_EXISTING"
        else:
            req = urllib.request.Request(rec["url"], headers={"User-Agent": "research4-public-dataset-audit/1.0"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                rec["http_status"] = resp.status
                rec["resolved_url"] = resp.url
                rec["content_type"] = resp.headers.get("Content-Type")
                if int(resp.headers.get("Content-Length", 0)) > max_bytes:
                    raise ValueError("source larger than explicit size limit")
                tmp = dest.with_suffix(dest.suffix + ".part")
                n = 0
                with tmp.open("wb") as stream:
                    while chunk := resp.read(1024 * 1024):
                        n += len(chunk)
                        if n > max_bytes:
                            raise ValueError("download exceeds explicit size limit")
                        stream.write(chunk)
                tmp.replace(dest)
        rec["size_bytes"] = dest.stat().st_size
        with dest.open("rb") as stream:
            rec["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        rec["download_status"] = "DOWNLOADED"
    except Exception as exc:
        rec["download_status"] = "DOWNLOAD_BLOCKED_OR_FAILED"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, urllib.error.HTTPError):
            rec["http_status"] = exc.code
        dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
    rec["elapsed_s"] = round(time.monotonic() - started, 3)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fruit", action="store_true")
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    sources = []
    if not args.skip_fruit:
        sources.append({
            "dataset_id": "princeton_metasticker_fruit",
            "filename": "author_repository.zip",
            "url": f"https://codeload.github.com/PrincetonUniversity/-Sub-Terahertz-Fruit-Sensing/zip/{FRUIT_COMMIT}",
            "repository": "https://github.com/PrincetonUniversity/-Sub-Terahertz-Fruit-Sensing",
            "commit": FRUIT_COMMIT,
            "paper_doi": "10.1145/3625687.3625815",
            "related_later_paper_doi_not_identical_cohort": "10.1038/s43016-024-01083-x",
            "license": "NOT_FOUND_IN_AUTHOR_REPOSITORY_DO_NOT_REDISTRIBUTE",
            "purpose": "external sub-THz fruit reflection feasibility; repeated fruit IDs require grouped evaluation",
        })
    for species in ("baby", "basil", "coffee", "coriander", "lettuce", "mint", "parsley", "peashoot", "spinach"):
        for quantity in ("real_n_idx", "real_eps"):
            sources.append({
            "dataset_id": "living_leaves_author_plot_data",
            "filename": f"{species}_leaf_{quantity}_all_days.tikz",
            "url": f"https://raw.githubusercontent.com/{LEAF_REPO}/{LEAF_COMMIT}/mdpi_template/figures/Used/{species}_leaf_{quantity}_all_days.tikz",
            "repository": f"https://github.com/{LEAF_REPO}",
            "commit": LEAF_COMMIT,
            "paper_doi": "10.3390/app9142781",
            "license": "NOT_FOUND_IN_AUTHOR_REPOSITORY_DO_NOT_REDISTRIBUTE",
            "purpose": "original author numeric plot coordinates; not a labeled independent specimen cohort",
        })
    sources.append({
        "dataset_id": "living_leaves_author_plot_data",
        "filename": "submission.tex",
        "url": f"https://raw.githubusercontent.com/{LEAF_REPO}/{LEAF_COMMIT}/MDPI_template/submission.tex",
        "repository": f"https://github.com/{LEAF_REPO}",
        "commit": LEAF_COMMIT,
        "license": "NOT_FOUND_IN_AUTHOR_REPOSITORY_DO_NOT_REDISTRIBUTE",
        "purpose": "read author methods to establish experimental unit",
    })
    results = []
    if args.skip_fruit and (REPORT / "download_manifest.json").exists():
        results = [r for r in json.loads((REPORT / "download_manifest.json").read_text()) if r["dataset_id"] == "princeton_metasticker_fruit"]
        for rec in results:
            rec["paper_doi"] = "10.1145/3625687.3625815"
            rec["related_later_paper_doi_not_identical_cohort"] = "10.1038/s43016-024-01083-x"
    for rec in sources:
        result = fetch(rec)
        results.append(result)
        print(json.dumps({k: result.get(k) for k in ("dataset_id", "filename", "download_status", "size_bytes", "error")}), flush=True)
        (REPORT / "download_manifest.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
