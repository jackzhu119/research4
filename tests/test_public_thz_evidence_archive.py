import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/archive_public_thz_evidence_20261031.py"
SPEC = importlib.util.spec_from_file_location("public_thz_evidence_archive", SCRIPT)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class EvidenceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "selection_models").mkdir(parents=True)
        (self.source / "transform_cache").mkdir()
        (self.source / "selection_models/a.joblib").write_bytes(bytes(range(200)))
        (self.source / "selection_models/b.json").write_bytes(b'{"model":"fixture"}\n')
        (self.source / "transform_cache/c.joblib").write_bytes(b"matrix-fixture" * 5)
        (self.source / "fit_scope_events.jsonl").write_bytes(b'{"scope":"source"}\n' * 100)
        self.output = self.root / "archives"
        self.target = self.root / "restored"

    def tearDown(self):
        self.temp.cleanup()

    def create(self):
        return archive.create_archive(self.source, self.output, chunk_bytes=600, max_archive_bytes=4096)

    def save_manifest(self, manifest):
        (self.output / "archive_manifest.json").write_text(json.dumps(manifest))

    def test_roundtrip_split_gzip_multiple_zip_idempotence_originals_unchanged(self):
        before = {p.relative_to(self.source).as_posix(): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        m = self.create()
        self.assertEqual(m["source_files"], 4)
        self.assertGreater(sum(v["format"] == "zip" for v in m["archives"].values()), 1)
        self.assertGreater(sum(v["format"] == "gzip" for v in m["archives"].values()), 1)
        self.assertTrue(all(v["bytes"] <= 4096 for v in m["archives"].values()))
        self.assertEqual(archive.restore_archive(self.output, self.target)["restored"], 4)
        self.assertEqual(archive.restore_archive(self.output, self.target)["already_matching"], 4)
        for name, content in before.items():
            self.assertEqual((self.source / name).read_bytes(), content)
            self.assertEqual((self.target / name).read_bytes(), content)
        with self.assertRaises(FileExistsError):
            self.create()

    def test_archive_tamper_refused_before_any_target_write(self):
        m = self.create()
        path = self.output / next(iter(m["archives"]))
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(archive.ArchiveError, "bytes changed"):
            archive.restore_archive(self.output, self.target)
        self.assertFalse(self.target.exists())

    def test_existing_nonmatching_target_refuses_every_restore(self):
        self.create()
        p = self.target / "transform_cache/c.joblib"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"do not overwrite")
        with self.assertRaisesRegex(archive.ArchiveError, "Existing target differs"):
            archive.restore_archive(self.output, self.target)
        self.assertEqual(p.read_bytes(), b"do not overwrite")
        self.assertFalse((self.target / "selection_models").exists())

    def test_parent_file_conflict_refused_before_any_restore(self):
        self.create()
        self.target.mkdir()
        (self.target / "transform_cache").write_bytes(b"parent is a file")
        with self.assertRaisesRegex(archive.ArchiveError, "parent is not a directory"):
            archive.restore_archive(self.output, self.target)
        self.assertFalse((self.target / "selection_models").exists())
        self.assertEqual((self.target / "transform_cache").read_bytes(), b"parent is a file")

    def test_traversal_and_noncanonical_manifest_paths_rejected(self):
        m = self.create()
        for value in ["../escape", "/absolute", "x/../../escape", "C:/windows", "x\\escape", "x//escape", "./escape"]:
            m["files"][0]["path"] = value
            self.save_manifest(m)
            with self.subTest(path=value), self.assertRaises(archive.ArchiveError):
                archive.restore_archive(self.output, self.target)
        self.assertFalse(self.target.exists())

    def test_source_and_target_symlinks_rejected(self):
        (self.source / "selection_models/linked").symlink_to(self.source / "transform_cache/c.joblib")
        with self.assertRaisesRegex(archive.ArchiveError, "symlink"):
            self.create()
        (self.source / "selection_models/linked").unlink()
        self.create()
        self.target.mkdir()
        (self.target / "transform_cache").symlink_to(self.source / "transform_cache", target_is_directory=True)
        with self.assertRaisesRegex(archive.ArchiveError, "Symlink"):
            archive.restore_archive(self.output, self.target)
        self.assertFalse((self.target / "selection_models").exists())

    def test_extra_zip_member_rejected_even_with_new_archive_digest(self):
        m = self.create()
        name = next(n for n, v in m["archives"].items() if v["format"] == "zip")
        p = self.output / name
        with zipfile.ZipFile(p, "a") as z:
            z.writestr("../escape", b"bad")
        m["archives"][name].update(bytes=p.stat().st_size, sha256=archive.digest(p))
        self.save_manifest(m)
        with self.assertRaisesRegex(archive.ArchiveError, "Unexpected ZIP member"):
            archive.restore_archive(self.output, self.target)
        self.assertFalse(self.target.exists())

    def test_decoded_content_hash_checked_and_not_published(self):
        m = self.create()
        for rec in m["files"]:
            rec["sha256"] = "0" * 64
        self.save_manifest(m)
        with self.assertRaisesRegex(archive.ArchiveError, "Restored bytes differ"):
            archive.restore_archive(self.output, self.target)
        self.assertFalse(any(p.is_file() for p in self.target.rglob("*")))


if __name__ == "__main__":
    unittest.main()
