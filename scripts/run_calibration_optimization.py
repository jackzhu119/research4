"""Run the finite, checkpointed source-only calibration exploration protocol."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from herb_spectra.calibration_exploration import run_exploration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    invocation_log = args.output.parent / (args.output.name + "_invocations.jsonl")
    with invocation_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "start", "utc": datetime.now(timezone.utc).isoformat(),
                                 "argv": sys.argv}, ensure_ascii=False) + "\n")
    result = run_exploration(args.input, args.split_plan, args.protocol, args.output, resume=args.resume)
    with invocation_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "complete", "utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "completed", "summary_rows": len(result), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
