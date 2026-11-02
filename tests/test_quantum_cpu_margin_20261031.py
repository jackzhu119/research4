import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/watch_quantum_cpu_margin_20261031.py"
spec = importlib.util.spec_from_file_location("quantum_margin_fixture", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MarginTests(unittest.TestCase):
    def test_cpu_parser_with_spaces_parentheses_and_thread_cpu(self):
        fields = ["S"] + ["0"] * 49
        fields[11], fields[12] = "1200", "300"
        self.assertEqual(module.proc_cpu_seconds("42 (worker (fixture)) " + " ".join(fields), 100), 15)

    def test_finished_phase_is_not_signal_target(self):
        self.assertIsNone(module.active_phase({"phases": [{"phase": "p", "status": "EXITED", "pid": 1}]}, "p"))

    def test_only_exact_running_phase_can_be_selected(self):
        row = {"phase": "p", "status": "RUNNING", "pid": 2}
        self.assertEqual(module.active_phase({"phases": [row]}, "p"), row)
        self.assertIsNone(module.active_phase({"phases": [row]}, "other"))


if __name__ == "__main__":
    unittest.main()
