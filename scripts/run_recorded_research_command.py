"""Run a research command while recording its actual UTC start and completion."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a command after --")
    args.events.parent.mkdir(parents=True, exist_ok=True)

    def record(event, **details):
        value = {"event": event, "label": args.label,
                 "utc": datetime.now(timezone.utc).isoformat(), "argv": command,
                 "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **details}
        with args.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            handle.flush()

    record("command_start")
    try:
        completed = subprocess.run(command, check=False)
    except BaseException as exc:
        record("command_error", error_type=type(exc).__name__)
        raise
    record("command_completed", returncode=completed.returncode)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
