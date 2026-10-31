import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/archive_cross_plant_transfer_evidence_20261031.py"
SPEC = importlib.util.spec_from_file_location("cross_plant_transfer_archive", SCRIPT)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class TransferEvidenceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        for entry in archive.ENTRIES:
            folder = self.source / entry
            folder.mkdir(parents=True)
            for n in range(10):
                (folder / f"fixture{n:02d}.joblib").write_bytes(bytes(range(n, n + 80)))
        (self.source / archive.ENTRIES[0] / "fit_scope.jsonl").write_bytes(b"scope-fixture\n" * 400)
        self.out = self.root / "archives"
        self.target = self.root / "restored"

    def tearDown(self):
        self.temp.cleanup()

    def create(self):
        return archive.create_evidence_archive(self.source, self.out, chunk_bytes=1200, max_archive_bytes=8192)

    def save(self, manifest):
        (self.out / "archive_manifest.json").write_text(json.dumps(manifest))

    def test_bulk_roundtrip_bounded_handles_idempotence_and_original_bytes(self):
        original = {p.relative_to(self.source).as_posix(): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        manifest = self.create()
        nzip = sum(v["format"] == "zip" for v in manifest["archives"].values())
        with patch.object(archive.zipfile, "ZipFile", wraps=archive.zipfile.ZipFile) as calls:
            result = archive.restore_evidence_archive(self.out, self.target)
            # One validation open and one restore open per ZIP, independent of member count.
            self.assertEqual(calls.call_count, 2 * nzip)
        self.assertEqual(result["zip_handles_opened_for_restore"], nzip)
        self.assertEqual(result["restored"], 31)
        second = archive.restore_evidence_archive(self.out, self.target)
        self.assertEqual(second["already_matching"], 31)
        self.assertEqual(second["zip_handles_opened_for_restore"], 0)
        for name, content in original.items():
            self.assertEqual((self.source / name).read_bytes(), content)
            self.assertEqual((self.target / name).read_bytes(), content)

    def test_missing_completed_directory_refused_without_archive(self):
        empty = self.root / "not-yet-trained"
        empty.mkdir()
        with self.assertRaisesRegex(archive.ArchiveError, "completed"):
            archive.create_evidence_archive(empty, self.out)
        self.assertFalse(self.out.exists())

    def test_foreign_source_scope_refused_before_restore(self):
        manifest = self.create()
        manifest["source_entries"][0] = "unknown-license-fruit"
        self.save(manifest)
        with self.assertRaisesRegex(archive.ArchiveError, "eligibility"):
            archive.restore_evidence_archive(self.out, self.target)
        self.assertFalse(self.target.exists())

    def test_existing_parent_or_file_mismatch_never_overwritten(self):
        self.create()
        path = self.target / "neural/source_models"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"pre-existing-conflict")
        with self.assertRaisesRegex(archive.ArchiveError, "parent is not a directory"):
            archive.restore_evidence_archive(self.out, self.target)
        self.assertEqual(path.read_bytes(), b"pre-existing-conflict")
        self.assertFalse((self.target / "empirical").exists())

    def test_decoded_hash_failure_does_not_publish_the_bad_file(self):
        manifest = self.create()
        manifest["files"][0]["sha256"] = "0" * 64
        bad = manifest["files"][0]["path"]
        self.save(manifest)
        with self.assertRaisesRegex(archive.ArchiveError, "Restored bytes differ"):
            archive.restore_evidence_archive(self.out, self.target)
        self.assertFalse((self.target / bad).exists())
        self.assertFalse(any(p.name.startswith(".transfer-restore-") for p in self.target.rglob("*")))

    def test_prior_helper_hash_is_checked_without_mutating_old_source(self):
        before = archive.old.digest(archive.OLD_HELPER)
        with patch.object(archive, "OLD_HELPER_SHA256", "0" * 64):
            with self.assertRaisesRegex(archive.ArchiveError, "dependency bytes changed"):
                self.create()
        self.assertEqual(archive.old.digest(archive.OLD_HELPER), before)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
