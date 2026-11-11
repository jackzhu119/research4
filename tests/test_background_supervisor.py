from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import background_supervisor
from background_supervisor import atomic_json


SCRIPT = Path(background_supervisor.__file__).resolve()


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.state = self.root / "state"
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self.temp.cleanup()

    def launch(self, code: str, state: Path | None = None):
        process = subprocess.Popen([
            sys.executable, str(SCRIPT), "--cwd", str(self.root),
            "--state-dir", str(state or self.state),
            "--heartbeat-seconds", "0.03", "--restart-delay", "0.03",
            "--max-restart-delay", "0.1", "--", sys.executable, "-u", "-c", code,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
           start_new_session=True)
        self.processes.append(process)
        return process

    def await_condition(self, predicate, seconds=5):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            try:
                if predicate():
                    return
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.02)
        self.fail("condition did not become true")

    def status(self):
        return json.loads((self.state / "status.json").read_text())

    def test_atomic_json_replaces_complete_state(self):
        path = self.root / "state.json"
        atomic_json(path, {"first": "中文"})
        atomic_json(path, {"second": 2})
        self.assertEqual(json.loads(path.read_text()), {"second": 2})
        self.assertEqual(list(self.root.glob(".state.json.*")), [])

    def test_restarts_on_success_and_failure_and_stops_explicitly(self):
        worker = "from pathlib import Path; import sys; p=Path('count'); n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n)); sys.exit(n%2)"
        process = self.launch(worker)
        self.await_condition(lambda: int((self.root / "count").read_text()) >= 4)
        (self.state / "STOP").touch()
        process.wait(timeout=5)
        self.assertEqual(process.returncode, 0)
        status = self.status()
        self.assertEqual(status["phase"], "stopped")
        self.assertEqual(status["stop_reason"], "STOP_file")
        events = [json.loads(line) for line in (self.state / "events.jsonl").read_text().splitlines()]
        exits = [e["returncode"] for e in events if e["event"] == "worker_exited"]
        self.assertIn(0, exits)
        self.assertIn(1, exits)
        self.assertGreaterEqual(len(list((self.state / "logs").glob("worker-*"))), 4)

    def test_single_instance_and_sigterm_forwarding(self):
        worker = "import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM, lambda *_: (Path('terminated').touch(), exit(0))); Path('ready').touch(); time.sleep(30)"
        first = self.launch(worker)
        self.await_condition(lambda: (self.root / "ready").exists())
        original = self.status()
        duplicate = self.launch(worker)
        duplicate.wait(timeout=5)
        self.assertEqual(duplicate.returncode, 2)
        self.assertEqual(self.status()["supervisor_pid"], original["supervisor_pid"])
        first.send_signal(signal.SIGTERM)
        first.wait(timeout=5)
        self.assertTrue((self.root / "terminated").exists())
        self.assertEqual(self.status()["stop_reason"], "signal:SIGTERM")

    def test_initial_stop_does_not_start_worker(self):
        self.state.mkdir()
        (self.state / "STOP").touch()
        process = self.launch("from pathlib import Path; Path('should_not_exist').touch()")
        process.wait(timeout=5)
        self.assertFalse((self.root / "should_not_exist").exists())
        self.assertEqual(self.status()["attempt"], 0)

    def test_surviving_worker_keeps_lock_after_supervisor_kill(self):
        first = self.launch("import time; from pathlib import Path; Path('ready').touch(); time.sleep(30)")
        self.await_condition(lambda: (self.root / "ready").exists())
        child_pid = self.status()["worker_pid"]
        first.kill()
        first.wait(timeout=3)
        try:
            duplicate = self.launch("pass")
            duplicate.wait(timeout=3)
            self.assertEqual(duplicate.returncode, 2)
        finally:
            try:
                os.killpg(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    unittest.main()
