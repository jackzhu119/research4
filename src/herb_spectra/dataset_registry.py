"""Resolve current workbooks explicitly; never fall back to archived inputs."""
from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import warnings


REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WorkbookSource:
    path: Path
    relative_path: str
    dataset_version: str
    selection_mode: str
    sha256: str | None = None
    dataset: str | None = None
    record_count: int | None = None
    origin_group_id: str | None = None


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def select_workbooks(*, repo=None, manifest=None, data_root=None):
    """Validate the active registry, or perform an explicitly requested override.

    Registry paths are relative to ``repo``. An explicit data_root retains the
    older directory-inspection behavior, is labelled as an override, and does
    not establish registry metadata, scientific labels or physical identities.
    """
    repo = Path(repo or REPO).resolve()
    if manifest is not None and data_root is not None:
        raise ValueError("--manifest and --data-root are mutually exclusive")
    if data_root is not None:
        root = Path(data_root)
        root = root.resolve() if root.is_absolute() else (repo / root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Explicit --data-root directory is missing: {root}")
        warnings.warn("Explicit --data-root override: active-workbook registry is bypassed", UserWarning)
        result = []
        for path in sorted(root.rglob("*.xlsx")):
            try:
                relative = path.relative_to(repo).as_posix()
            except ValueError:
                relative = path.as_posix()
            result.append(WorkbookSource(path, relative, "explicit-data-root-override",
                                         "explicit_data_root_override"))
        if not result:
            raise ValueError("Explicit --data-root contains no XLSX workbooks")
        return result

    registry = Path(manifest) if manifest is not None else Path("data/active_workbooks.json")
    registry = registry if registry.is_absolute() else repo / registry
    if not registry.is_file():
        raise FileNotFoundError(f"Active workbook registry is missing: {registry}; no raw-directory fallback")
    payload = json.loads(registry.read_text(encoding="utf-8-sig"))
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Active registry requires a nonempty version")
    try:
        date.fromisoformat(payload["effective_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Active registry requires an ISO effective_date") from exc
    records = payload.get("workbooks")
    if not isinstance(records, list) or not records:
        raise ValueError("Active registry requires a nonempty workbooks array")
    result, seen_paths, seen_hashes, seen_groups = [], set(), set(), set()
    for record in records:
        if not isinstance(record, dict) or record.get("status") != "active":
            raise ValueError("Every active-registry workbook must have status='active'")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError("Workbook path must be a relative POSIX path")
        rel = PurePosixPath(relative)
        if rel.is_absolute() or ".." in rel.parts or rel.suffix.lower() != ".xlsx":
            raise ValueError("Workbook path must be a relative XLSX path without parent traversal")
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo) or not path.is_file():
            raise FileNotFoundError(f"Active workbook is missing or outside the repository: {relative}")
        expected = record.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError(f"Active workbook requires a SHA-256: {relative}")
        expected = expected.lower()
        if path in seen_paths or expected in seen_hashes:
            raise ValueError("Active registry contains a duplicate workbook path or content hash")
        if file_sha256(path) != expected:
            raise ValueError(f"Active workbook SHA-256 mismatch: {relative}")
        dataset, count = record.get("dataset"), record.get("record_count")
        if dataset not in {"huangqi", "zaoren"}:
            raise ValueError(f"Unknown active dataset: {dataset}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("record_count must be a positive integer")
        group = record.get("origin_group_id")
        if dataset == "zaoren":
            if not isinstance(group, str) or not re.fullmatch(r"ZY[0-9]{2}", group):
                raise ValueError("Zaoren workbooks require a ZY01-style origin_group_id")
            if group in seen_groups:
                raise ValueError("An origin group occurs in more than one active workbook")
            seen_groups.add(group)
        seen_paths.add(path)
        seen_hashes.add(expected)
        result.append(WorkbookSource(path, relative, version, "active_registry", expected,
                                     dataset, count, group))
    return result
