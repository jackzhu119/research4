"""Pure restart preparation guards with synthetic ledgers/checkpoints; no DFT."""
from contextlib import contextmanager
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pyscf_resume_prepare", ROOT / "scripts/prepare_pyscf_pilot_resume_20261031.py")
resume = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(resume)
RUN = ROOT / "reports/pyscf-spinosin-pilot-20261031"


def ledger(consumed=1000.25):
    return {"quantum_cpu_cap_seconds": 3300, "user_total_cpu_cap_seconds": 3600,
            "non_quantum_reserve_seconds": 300, "consumed_cpu_seconds": consumed,
            "remaining_quantum_cpu_seconds": 3300 - consumed,
            "phases": [{"phase": "original-opt", "status": "PARTIAL_OR_FAILED", "actual_cpu_seconds": consumed}]}


@contextmanager
def fixture(consumed=1000.25):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "run"; root.mkdir()
        source = root / "structures"; source.mkdir()
        shutil.copyfile(RUN / "structures/spinosin_verified_2d.json", source / "spinosin_verified_2d.json")
        phase = root / "calculations/original-opt"; phase.mkdir(parents=True)
        data = ledger(consumed)
        data["phases"][0]["command"] = ["python", "worker.py", "--phase", "spinosin-optimize", "--output", str(phase)]
        settings = {"xc": "PBE", "basis": "def2-svp", "grid_level": 1, "charge": 0, "spin_2S": 0,
                    "dispersion": "NONE_EXPLICIT_PBE_PILOT_NOT_D3"}
        resume.write(root / "cpu_budget.json", data)
        resume.write(phase / "input.json", {"settings": settings, "synthetic_fixture": True})
        resume.write(phase / "result.json", {"settings": settings, "status": "PARTIAL", "geometry_optimized": False, "scf_converged": False, "synthetic_fixture": True})
        (phase / "calculation.chk").write_bytes(b"SYNTHETIC_CHECKPOINT_NOT_QUANTUM_DATA")
        release = root / "root_release.json"
        resume.write(release, {"allow_resume_preparation": True, "created_at_utc": "2026-10-31T00:00:00Z",
            "cpu_budget_sha256": resume.sha(root / "cpu_budget.json"), "source_phase": "original-opt",
            "source_result_sha256": resume.sha(phase / "result.json")})
        symbols, coordinates = resume.read_xyz(RUN / "conformers/C001_mmff94s.xyz")
        with mock.patch.object(resume, "load_checkpoint_geometry", return_value=(symbols, coordinates, coordinates / resume.BOHR_TO_ANGSTROM)) as loader:
            yield root, phase, release, symbols, coordinates, loader


class ResumeBudgetGuards(unittest.TestCase):
    def test_grant_is_capped_by_remaining_original_ledger(self):
        result = resume.budget_decision(ledger(), 3300, "resume-001")
        self.assertEqual(result["bounded_cpu_grant_seconds"], 2297)
        self.assertEqual(result["status"], "READY")
        self.assertFalse(result["ledger_reinitialized"])
        self.assertFalse(result["automatic_budget_enlargement"])

    def test_insufficient_budget_and_zero_request_block(self):
        self.assertEqual(resume.budget_decision(ledger(3296.5), 900, "resume-001")["status"], "BLOCKED_INSUFFICIENT_ORIGINAL_CPU_BUDGET")
        self.assertEqual(resume.budget_decision(ledger(), 0, "resume-001")["bounded_cpu_grant_seconds"], 0)

    def test_running_or_inconsistent_ledger_rejected(self):
        data = ledger(); data["phases"][0]["status"] = "RUNNING"
        with self.assertRaisesRegex(ValueError, "running"):
            resume.validate_ledger(data)
        data = ledger(); data["consumed_cpu_seconds"] += 10
        with self.assertRaisesRegex(ValueError, "totals"):
            resume.validate_ledger(data)

    def test_caps_cannot_be_silently_raised(self):
        data = ledger(); data["quantum_cpu_cap_seconds"] = 3600
        with self.assertRaises(ValueError):
            resume.validate_ledger(data)
        with self.assertRaises(ValueError):
            resume.budget_decision(ledger(), 3301, "resume-001")

    def test_phase_reuse_or_path_traversal_rejected(self):
        for phase in ("original-opt", "../escape", "/absolute", "name with spaces"):
            with self.assertRaises(ValueError):
                resume.budget_decision(ledger(), 900, phase)


class ResumePreparationGuards(unittest.TestCase):
    def test_actual_pyscf_checkpoint_molecule_serialization_is_fullprecision_without_scf(self):
        resume.isolated_sites()
        from pyscf import gto
        from pyscf.lib.chkfile import save_mol
        symbols, coordinates = resume.read_xyz(RUN / "conformers/C001_mmff94s.xyz")
        coordinates[0, 0] += 1e-12
        molecule = gto.M(atom=list(zip(symbols, coordinates)), basis="sto-3g", unit="Angstrom", charge=0, spin=0, verbose=0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "synthetic-molecule-only-no-scf.chk"
            save_mol(molecule, str(checkpoint))
            loaded_symbols, angstrom, bohr = resume.load_checkpoint_geometry(checkpoint)
            self.assertEqual(loaded_symbols, symbols)
            np.testing.assert_array_equal(bohr, molecule.atom_coords(unit="Bohr"))
            np.testing.assert_array_equal(angstrom, molecule.atom_coords(unit="Angstrom"))

    def test_fullprecision_recovery_and_command_keep_same_budget_no_execution(self):
        with fixture() as (root, phase, release, symbols, coordinates, loader):
            precise = coordinates.copy(); precise[0, 0] += 1e-13
            loader.return_value = symbols, precise, precise / resume.BOHR_TO_ANGSTROM
            original = (root / "cpu_budget.json").read_bytes()
            out = root / "continuation"
            manifest = resume.prepare(root, "original-opt", out, release, "resume-001", 3300)
            np.testing.assert_array_equal(np.load(out / "resume_coordinates_angstrom.npy"), precise)
            self.assertEqual((root / "cpu_budget.json").read_bytes(), original)
            self.assertEqual(manifest["new_quantum_jobs"], 0)
            self.assertFalse(manifest["command_executed"])
            command = manifest["worker_command"]
            self.assertEqual(command[command.index("--ledger") + 1], str(root / "cpu_budget.json"))
            self.assertEqual(command[command.index("--cpu-seconds") + 1], "2297")
            self.assertNotIn("--initialize", command)
            self.assertNotIn("--total-cpu-seconds", command)
            self.assertFalse((root / "calculations/resume-001").exists())
            provenance = resume.read(out / "resume_geometry_provenance.json")
            self.assertTrue(provenance["fresh_SCF_and_gradient_required"])
            self.assertFalse(provenance["geometry_stationarity_verified_by_this_utility"])
            self.assertIn("trial molecule", provenance["checkpoint_warning"])
            self.assertEqual(resume.verify(out)["new_quantum_jobs"], 0)

    def test_blocked_budget_still_recovers_evidence_without_launch_command(self):
        with fixture(3296.5) as (root, phase, release, *_):
            out = root / "continuation"
            manifest = resume.prepare(root, "original-opt", out, release, "resume-001")
            self.assertIn("BLOCKED", manifest["status"])
            self.assertIsNone(manifest["worker_command"])
            self.assertTrue((out / "resume_geometry.xyz").exists())
            self.assertFalse((out / "resume_same_budget.sh").exists())
            self.assertEqual(resume.verify(out)["new_quantum_jobs"], 0)

    def test_running_budget_lock_refuses_before_checkpoint_read(self):
        with fixture() as (root, phase, release, *rest):
            loader = rest[-1]
            with (root / "cpu_budget.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, "execution lock"):
                    resume.prepare(root, "original-opt", root / "continuation", release, "resume-001")
            loader.assert_not_called()

    def test_release_ledger_hash_and_parent_settings_must_match(self):
        with fixture() as (root, phase, release, *_):
            data = resume.read(release); data["cpu_budget_sha256"] = "0" * 64; resume.write(release, data)
            with self.assertRaisesRegex(ValueError, "root release"):
                resume.prepare(root, "original-opt", root / "continuation", release, "resume-001")
        with self.assertRaises(ValueError):
            resume.validate_parent_settings({"xc": "B3LYP"})

    def test_no_silent_fallback_from_existing_corrupt_checkpoint(self):
        with fixture() as (root, phase, release, *rest):
            rest[-1].side_effect = ValueError("synthetic corrupted checkpoint")
            with self.assertRaisesRegex(ValueError, "no silent"):
                resume.prepare(root, "original-opt", root / "continuation", release, "resume-001")
            self.assertFalse((root / "continuation").exists())

    def test_missing_checkpoint_uses_complete_gradient_step_and_no_MO_guess(self):
        with fixture() as (root, phase, release, symbols, coordinates, loader):
            (phase / "calculation.chk").unlink()
            steps = phase / "geometry_steps"; steps.mkdir()
            xyz = "75\nSynthetic completed geometry fixture, not a quantum calculation\n" + "\n".join(f"{s} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}" for s, p in zip(symbols, coordinates)) + "\n"
            (steps / "step_0001.xyz").write_text(xyz)
            np.save(steps / "step_0001.gradient.npy", np.zeros((75, 3)))
            resume.write(steps / "step_0001.json", {"cycle": 1, "scf_converged": True, "energy_hartree": -1, "synthetic_fixture": True})
            resume.write(steps / "step_0002.json", {"cycle": 2, "scf_converged": True, "energy_hartree": -2, "synthetic_fixture": True})
            out = root / "continuation"
            result = resume.prepare(root, "original-opt", out, release, "resume-001")
            self.assertNotIn("--checkpoint", result["worker_command"])
            self.assertIn("rounded", resume.read(out / "resume_geometry_provenance.json")["coordinate_origin"])
            loader.assert_not_called()

    def test_changed_ledger_geometry_or_shell_command_rejected_before_resume(self):
        for name in ("ledger", "geometry", "shell"):
            with self.subTest(name=name), fixture() as (root, phase, release, *_):
                out = root / "continuation"
                resume.prepare(root, "original-opt", out, release, "resume-001")
                path = {"ledger": root / "cpu_budget.json", "geometry": out / "resume_geometry.xyz", "shell": out / "resume_same_budget.sh"}[name]
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaises(ValueError):
                    resume.verify(out)

    def test_wrong_stereo_or_atom_order_is_rejected(self):
        with fixture() as (_, _, _, symbols, coordinates, _):
            source = resume.read(RUN / "structures/spinosin_verified_2d.json")
            with self.assertRaisesRegex(ValueError, "stereochemistry"):
                resume.verify_coordinates(source, symbols, coordinates * np.asarray([-1, 1, 1]))
            changed = list(symbols); index = changed.index("O"); changed[0], changed[index] = changed[index], changed[0]
            with self.assertRaisesRegex(ValueError, "Atom order"):
                resume.verify_coordinates(source, changed, coordinates)

    def test_existing_preparation_directory_is_preserved(self):
        with fixture() as (root, phase, release, *_):
            out = root / "continuation"; out.mkdir(); marker = out / "keep"; marker.write_text("original")
            with self.assertRaises(FileExistsError):
                resume.prepare(root, "original-opt", out, release, "resume-001")
            self.assertEqual(marker.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
