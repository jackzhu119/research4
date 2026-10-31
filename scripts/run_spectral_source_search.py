"""Evaluate fixed source-only spectral pipelines and lock Stage-A choices."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from herb_spectra.spectral_source_search import run_source_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    invocation = args.output.parent / "source_run_invocations.jsonl"
    with invocation.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "start", "utc": datetime.now(timezone.utc).isoformat(),
                                 "argv": sys.argv}, ensure_ascii=False)+"\n")
    result = run_source_search(args.input, args.split_plan, args.protocol, args.output, resume=args.resume)
    with invocation.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "complete", "utc": datetime.now(timezone.utc).isoformat()})+"\n")
    print(json.dumps({"status": "completed", "selected": result["selected"]["id"]}), flush=True)


if __name__ == "__main__":
    main()
