"""Meaningful integrity and extraction tests using tiny synthetic archives."""
import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import restore_backfill as m


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, members=None, inventory=None, mutate_gzip=None):
        members = [("data/result.json", b'{"real":true}\n', None)] if members is None else members
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name, content, kind in members:
                entry = tarfile.TarInfo(name)
                entry.mode = 0o600
                entry.mtime = 0
                if kind is not None:
                    entry.type = kind
                    entry.linkname = "outside"
                    archive.addfile(entry)
                else:
                    entry.size = len(content)
                    archive.addfile(entry, io.BytesIO(content))
        payload = gzip.compress(stream.getvalue(), mtime=0)
        if mutate_gzip:
            payload = mutate_gzip(payload)
        part = self.root / "archives/closed-results.tar.gz.part001"
        part.parent.mkdir(exist_ok=True)
        part.write_bytes(payload)
        if inventory is None:
            inventory = [{"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                         for name, content, kind in members if kind is None]
        data = {"schema": m.SCHEMA, "files": inventory,
                "parts": [{"path": str(part.relative_to(self.root)), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}
        self.manifest.write_text(json.dumps(data), encoding="utf-8")
        return part, data

    def test_verify_stream_has_no_extraction_side_effect(self):
        self.fixture()
        result = m.verify(self.manifest)
        self.assertEqual((result["status"], result["mode"], result["files"]), ("PASS", "VERIFY_ONLY", 1))
        self.assertFalse((self.root / "data").exists())

    def test_restore_exact_bytes_and_private_mode(self):
        self.fixture()
        out = self.root / "restored"
        result = m.verify(self.manifest, output=out)
        self.assertEqual(result["mode"], "RESTORED")
        self.assertEqual((out / "data/result.json").read_bytes(), b'{"real":true}\n')
        self.assertEqual((out / "data/result.json").stat().st_mode & 0o777, 0o600)

    def test_transport_tamper_blocks_before_creating_output(self):
        part, _ = self.fixture()
        part.write_bytes(part.read_bytes() + b"tampered")
        out = self.root / "not_created"
        with self.assertRaisesRegex(m.VerificationError, "Part checksum"):
            m.verify(self.manifest, output=out)
        self.assertFalse(out.exists())

    def test_inner_file_hash_tamper_is_detected(self):
        _, data = self.fixture()
        data["files"][0]["sha256"] = "0" * 64
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(m.VerificationError, "Member checksum"):
            m.verify(self.manifest)

    def test_crc_failure_even_if_transport_hash_is_updated(self):
        def change_crc(payload):
            value = bytearray(payload)
            value[-8] ^= 1
            return bytes(value)
        self.fixture(mutate_gzip=change_crc)
        with self.assertRaises((gzip.BadGzipFile, EOFError)):
            m.verify(self.manifest)

    def test_missing_member_detected(self):
        _, data = self.fixture()
        data["files"].append({"path": "data/missing.txt", "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()})
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(m.VerificationError, "missing manifest"):
            m.verify(self.manifest)

    def test_unknown_member_rejected(self):
        self.fixture(members=[("data/one", b"one", None), ("data/two", b"two", None)],
                     inventory=[{"path": "data/one", "bytes": 3, "sha256": hashlib.sha256(b"one").hexdigest()}])
        with self.assertRaisesRegex(m.VerificationError, "Unknown/duplicate"):
            m.verify(self.manifest)

    def test_duplicate_member_rejected(self):
        self.fixture(members=[("data/one", b"one", None), ("data/one", b"one", None)],
                     inventory=[{"path": "data/one", "bytes": 3, "sha256": hashlib.sha256(b"one").hexdigest()}])
        with self.assertRaisesRegex(m.VerificationError, "Unknown/duplicate"):
            m.verify(self.manifest)

    def test_member_after_first_tar_eof_cannot_be_hidden(self):
        part, data = self.fixture()
        original = part.read_bytes()
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            entry = tarfile.TarInfo("hidden/unexpected.txt")
            entry.mode = 0o600
            entry.mtime = 0
            entry.size = 6
            archive.addfile(entry, io.BytesIO(b"hidden"))
        payload = original + gzip.compress(stream.getvalue(), mtime=0)
        part.write_bytes(payload)
        data["parts"][0].update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(m.VerificationError, "Unknown/duplicate"):
            m.verify(self.manifest)

    def test_symlink_member_rejected(self):
        self.fixture(members=[("data/one", b"one", None), ("data/link", b"", tarfile.SYMTYPE)])
        with self.assertRaisesRegex(m.VerificationError, "Nonregular"):
            m.verify(self.manifest)

    def test_traversal_rejected(self):
        self.fixture()
        _, data = self.fixture()
        data["files"][0]["path"] = "../outside"
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(m.VerificationError, "Unsafe"):
            m.verify(self.manifest)

    def test_nonempty_output_is_never_overwritten(self):
        self.fixture()
        out = self.root / "existing"
        out.mkdir()
        (out / "keep").write_text("safe")
        with self.assertRaisesRegex(m.VerificationError, "must be empty"):
            m.verify(self.manifest, output=out)
        self.assertEqual((out / "keep").read_text(), "safe")

    def test_report_is_exclusive_and_separate(self):
        self.fixture()
        report = self.root / "verification.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(["--manifest", str(self.manifest), "--verify-only", "--report", str(report)]), 0)
            self.assertEqual(m.main(["--manifest", str(self.manifest), "--verify-only", "--report", str(report)]), 1)
        self.assertEqual(json.loads(report.read_text())["status"], "PASS")

    def test_continuous_reader_crosses_part_boundaries(self):
        paths = []
        for i, data in enumerate([b"ab", b"", b"cdef", b"g"]):
            p = self.root / ("part%d" % i)
            p.write_bytes(data)
            paths.append(p)
        with m.PartsReader(paths) as raw, io.BufferedReader(raw, buffer_size=3) as buffered:
            self.assertEqual(buffered.read(2), b"ab")
            self.assertEqual(buffered.read(4), b"cdef")
            self.assertEqual(buffered.read(4), b"g")


if __name__ == "__main__":
    unittest.main()
