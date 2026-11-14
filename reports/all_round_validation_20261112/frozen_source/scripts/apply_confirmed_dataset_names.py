"""Apply the 2026-11-08 user-confirmed labels after exact-content intake checks.

Old numbered XLSX bytes and record identifiers are preserved. New raw files
retain their uploaded bytes; the complete Gansu export replaces its incomplete
predecessor. Run only against the 2026-10-31 registry with the audited uploads.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from zipfile import ZipFile

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

VERSION = "2026-11-08-confirmed-origin-kind-v1"
DATE = "2026-11-08"
BASIS = "user_confirmed_origin_and_kind_20261108"
PROCESSED = "data/processed/zaoren_20261031"
ARCHIVE = "data/archive/pre_confirmation_20261108"
LABELS = {
    "ZY01": ("陕西延安枣仁.xlsx", "陕西延安", "枣仁"),
    "ZY02": ("云南枣仁.xlsx", "云南", "枣仁"),
    "ZY03": ("内蒙古酸枣仁.xlsx", "内蒙古", "酸枣仁"),
    "ZY04": ("山西灵石酸枣仁.xlsx", "山西灵石", "酸枣仁"),
    "ZY05": ("安徽亳州酸枣仁.xlsx", "安徽亳州", "酸枣仁"),
    "ZY06": ("河北石家庄酸枣仁.xlsx", "河北石家庄", "酸枣仁"),
    "ZY07": ("山西酸枣仁.xlsx", "山西", "酸枣仁"),
    "HQ01": ("黄芪曲靖.xlsx", "曲靖", "黄芪"),
    "HQ02": ("黄芪内蒙古固阳.xlsx", "内蒙古固阳", "黄芪"),
    "HQ03": ("黄芪甘肃.xlsx", "甘肃", "黄芪"),
    "HQ04": ("黄芪山西大同.xlsx", "山西大同", "黄芪"),
    "ZR01": ("陕西榆林枣仁.xlsx", "陕西榆林", "枣仁"),
    "ZR02": ("辽宁枣仁.xlsx", "辽宁", "枣仁"),
    "ZR03": ("内蒙古枣仁.xlsx", "内蒙古", "枣仁"),
    "TM01": ("天麻贵州雷山.xlsx", "贵州雷山", "天麻"),
    "TM02": ("天麻吉林长白山.xlsx", "吉林长白山", "天麻"),
    "TM03": ("天麻安徽毫州.xlsx", "安徽毫州", "天麻"),
    "TM04": ("天麻云南.xlsx", "云南", "天麻"),
}
HQ_OLD = {
    "HQ01": "data/raw/huangqi/黄芪曲靖202605310559.xlsx",
    "HQ02": "data/raw/huangqi/黄芪内蒙古固阳202605310518.xlsx",
    "HQ03": "data/raw/huangqi/黄芪甘肃202605310347.xlsx",
    "HQ04": "data/raw/huangqi/黄芪山西大同202605310205.xlsx",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def safe_audit(row):
    result = {k: v for k, v in row.items() if k not in {"path", "sheets"}}
    if "sheets" in row:
        result["sheets"] = []
        for sheet in row["sheets"]:
            summary = {k: v for k, v in sheet.items() if k != "headers"}
            headers = sheet.get("headers", [])
            summary["header_preview"] = headers if len(headers) <= 5 else headers[:3] + headers[-2:]
            summary["header_preview_note"] = "First three and last two cells only; inspect the original XLSX for all headers. Counts/checks cover the complete worksheet."
            result["sheets"].append(summary)
    return result


def apply(args):
    root = args.repository.resolve()
    registry_path = root / "data/active_workbooks.json"
    old = read_json(registry_path)
    if old["version"] == VERSION:
        raise ValueError("This migration is already applied; use the validation script.")
    assert old["version"] == "2026-10-31-zaoren-numbered-v1"
    base_commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    audited = {r["name"]: r for r in read_json(args.upload_audit)}
    recovery = read_json(args.recovery_json)
    yanan_sha = sha(args.recovered_yanan)
    old_originals, _ = read_csv(root / "data/archive/zaoren_superseded_20261031/manifest.csv")
    original_by_group = {r["origin_group_id"]: r for r in old_originals}
    assert yanan_sha == original_by_group["ZY01"]["sha256"]
    audited["陕西延安枣仁.xlsx"] = {
        "name": "陕西延安枣仁.xlsx", "path": str(args.recovered_yanan),
        "sha256": yanan_sha, "size": args.recovered_yanan.stat().st_size,
        "zip_crc_status": "passed", "recovered_from_volumes": True,
    }
    assert len(audited) == 18
    old_by_group = {r.get("origin_group_id"): r for r in old["workbooks"] if r.get("origin_group_id")}
    old_by_path = {r["path"]: r for r in old["workbooks"]}
    plans = []
    for group, (name, origin, species) in LABELS.items():
        incoming = audited[name]
        assert sha(incoming["path"]) == incoming["sha256"], name
        with ZipFile(incoming["path"]) as z:
            assert z.testzip() is None, name
        if group.startswith("ZY"):
            previous = old_by_group[group]
            original = original_by_group[group]
            assert incoming["sha256"] == original["sha256"], name
            old_path = previous["path"]
            new_path = f"{PROCESSED}/{name}"
            count = previous["record_count"]
            kind = "exact_original_sha256; numbered_workbook_filename_only"
            source_original = original["archived_path"]
        elif group.startswith("HQ"):
            old_path = HQ_OLD[group]
            previous = old_by_path[old_path]
            new_path = f"data/raw/huangqi/{name}"
            count = previous["record_count"]
            kind = "gansu_complete_reexport_numeric_comparison" if group == "HQ03" else "exact_existing_sha256"
            if group != "HQ03":
                assert previous["sha256"] == incoming["sha256"], name
            source_original = new_path
        else:
            old_path = ""
            previous = None
            folder = "data/raw/tianma" if species == "天麻" else "data/raw/zaoren_20261108"
            new_path = f"{folder}/{name}"
            counts = {r["sample_columns"] for r in incoming["sheets"]}
            assert len(counts) == 1 and len(incoming["sheets"]) == 4
            count = counts.pop()
            kind = "new_user_confirmed_upload"
            source_original = new_path
        if old_path:
            assert sha(root / old_path) == previous["sha256"], old_path
        assert not (root / new_path).exists(), new_path
        plans.append({
            "group_id": group, "old_path": old_path, "new_path": new_path,
            "uploaded_filename": name, "origin": origin, "species": species,
            "record_count": count, "match_basis": kind,
            "source_original_path": source_original,
            "source_sha256": incoming["sha256"],
            "previous_active_sha256": previous["sha256"] if previous else "",
            "output_sha256": previous["sha256"] if group.startswith("ZY") else incoming["sha256"],
            "label_basis": BASIS,
        })
    assert sum(p["record_count"] for p in plans) == 1672
    # All preconditions are checked before mutations; retain the old registry and annotations.
    archive = root / ARCHIVE
    archive.mkdir(parents=True, exist_ok=False)
    for name in ["active_workbooks.json", "manifest.csv", "metadata_template.csv"]:
        shutil.copy2(root / "data" / name, archive / name)
    (archive / "README.md").write_text(
        "# 2026-11-08更名前快照\n\n"
        "保存2026-10-31版本的有效清单、校验清单和1258行元数据，供历史实验溯源。"
        "快照中的旧路径是当时位置，当前路径见../../confirmed_names_20261108.csv。\n\n"
        "黄芪甘肃202605310347.xlsx是被本轮完整导出替代的旧版，折射率工作表为空；"
        "该原文件字节保持不变，不再列为当前输入。\n", encoding="utf-8")
    for p in plans:
        destination = root / p["new_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if p["group_id"] == "HQ03":
            (root / p["old_path"]).rename(archive / Path(p["old_path"]).name)
            shutil.copy2(audited[p["uploaded_filename"]]["path"], destination)
        elif p["old_path"]:
            (root / p["old_path"]).rename(destination)
        else:
            shutil.copy2(audited[p["uploaded_filename"]]["path"], destination)
        assert sha(destination) == p["output_sha256"]
    records = []
    for p in plans:
        entry = {
            "path": p["new_path"], "sha256": p["output_sha256"],
            "dataset": "huangqi" if p["species"] == "黄芪" else "tianma" if p["species"] == "天麻" else "zaoren",
            "record_count": p["record_count"], "status": "active", "origin_group_id": p["group_id"],
            "origin": p["origin"], "species": p["species"],
            "origin_verified": True, "species_verified": True,
            "label_verification_level": "user_confirmation", "annotation_basis": BASIS,
            "uploaded_filename": p["uploaded_filename"], "source_sha256": p["source_sha256"],
            "content_match_basis": p["match_basis"],
            "record_id_scope": "workbook spectrum records; physical independence only where separately declared",
        }
        if p["old_path"]:
            entry["previous_active_path"] = p["old_path"]
        if p["group_id"] == "HQ03":
            entry["superseded_original_path"] = f"{ARCHIVE}/黄芪甘肃202605310347.xlsx"
            entry["replacement_note"] = "Complete refractive-index sheet; 8 pre-existing numeric cells differ only in floating-point export precision, max_abs_diff=1.4210854715202004e-14."
        if p["group_id"].startswith("ZY"):
            entry["numbered_headers_preserved"] = True
            entry["independence_basis"] = "user_declared_independent_20261031"
            entry["source_batches_per_origin_user_declared"] = 1
        else:
            entry["physical_independence_status"] = "not_provided"
            entry["batch_count_status"] = "not_provided"
        records.append(entry)
    registry = {
        "version": VERSION, "effective_date": DATE,
        "authority": "User: 去比对替换一下仓库名字，现在我确定所有表格的产地和种类了; followed by complete Yanan RAR volumes.",
        "previous_version": old["version"],
        "migration_base_commit": base_commit,
        "zaoren_workbooks": 10, "zaoren_records": 864,
        "other_herb_workbooks": 8, "other_herb_records": 808,
        "huangqi_workbooks": 4, "huangqi_records": 403,
        "tianma_workbooks": 4, "tianma_records": 405,
        "historical_zy_records": 855, "newly_added_records": 414,
        "total_active_workbooks": 18, "total_active_records": 1672,
        "kind_counts": {"枣仁": 364, "酸枣仁": 500, "黄芪": 403, "天麻": 405},
        "label_verification_level": "User-confirmed origin and category labels; not independent laboratory species authentication.",
        "scope_note": "ZY01-ZY07 are stable historical identifiers spanning 枣仁 and 酸枣仁; their old results are not evidence for a pure single-species geographical task. New uploads are not declared independent specimens or batches.",
        "confirmation_record": "data/annotations/origin_kind_confirmation_20261108.json",
        "rename_map": "data/confirmed_names_20261108.csv",
        "previous_registry_snapshot": f"{ARCHIVE}/active_workbooks.json",
        "superseded_manifest": "data/archive/zaoren_superseded_20261031/manifest.csv",
        "workbooks": records,
    }
    write_json(registry_path, registry)
    write_csv(root / "data/manifest.csv", [
        {"path": p["new_path"], "size_bytes": (root / p["new_path"]).stat().st_size, "sha256": p["output_sha256"]}
        for p in plans])
    write_csv(root / "data/confirmed_names_20261108.csv", plans)
    metadata, fields = read_csv(archive / "metadata_template.csv")
    extra_fields = ["species", "origin_basis", "species_basis", "label_verification_level", "uploaded_filename"]
    fields = fields + [f for f in extra_fields if f not in fields]
    old_plan = {p["old_path"]: p for p in plans if p["old_path"]}
    for row in metadata:
        p = old_plan[row["source_workbook"]]
        row["source_workbook"] = p["new_path"]
        if not row["record_id"]:
            index = int(re.fullmatch(r"样品([0-9]+)", row["scan_header"])[1])
            row["record_id"] = f"{p['group_id']}-{index:03d}"
            row["origin_group_id"] = p["group_id"]
        row["notes"] = ("2026-10-31用户声明每个ZY编号独立、每来源1批；" if p["group_id"].startswith("ZY") else "独立物理样品与批次信息未提供；") + "2026-11-08用户确认产地和种类。未提供独立实验鉴定记录或实际批次编号。"
        row.update({"origin": p["origin"], "species": p["species"], "origin_verified": "true",
                    "species_verified": "true", "origin_group_verified": "true", "dataset_version": VERSION,
                    "origin_basis": BASIS, "species_basis": BASIS,
                    "label_verification_level": "user_confirmation", "uploaded_filename": p["uploaded_filename"]})
    for p in plans:
        if p["old_path"]:
            continue
        incoming = audited[p["uploaded_filename"]]
        headers = next(s for s in incoming["sheets"] if s["sheet"] == "吸收系数")["headers"]
        for header in headers[1:]:
            old_header = header["value"]
            index = int(re.fullmatch(r"样品([0-9]+)", old_header)[1])
            column = re.match(r"[A-Z]+", header["cell"])[0]
            identifier = f"{p['group_id']}-{index:03d}"
            row = dict.fromkeys(fields, "")
            row.update({"scan_id": hashlib.sha256(f"{p['source_sha256']}|吸收系数|{column}|{old_header}".encode()).hexdigest()[:20],
                        "source_workbook": p["new_path"], "sheet": "吸收系数", "excel_column": column,
                        "scan_header": old_header, "record_id": identifier, "origin_group_id": p["group_id"],
                        "origin": p["origin"], "species": p["species"], "origin_verified": "true",
                        "species_verified": "true", "origin_group_verified": "true", "dataset_version": VERSION,
                        "origin_basis": BASIS, "species_basis": BASIS, "label_verification_level": "user_confirmation",
                        "uploaded_filename": p["uploaded_filename"],
                        "notes": "用户确认产地/种类；record_id仅为元数据中的谱记录标识，未改原表头，未认定物理独立样品或批次数。"})
            metadata.append(row)
    assert len(metadata) == 1672
    assert len({r["record_id"] for r in metadata}) == 1672
    assert len({r["scan_id"] for r in metadata}) == 1672
    write_csv(root / "data/metadata_template.csv", metadata, fields)
    by_group = {p["group_id"]: p for p in plans}
    # Historical names, hashes and check results remain original evidence.
    for name in ["导入记录.json", "编号核验.json", "独立核验记录.json"]:
        path = root / PROCESSED / name
        value = read_json(path)
        value["current_dataset_version"] = VERSION
        value["current_rename_map"] = "data/confirmed_names_20261108.csv"
        value["historical_evidence_note"] = "2026-10-31 names/hashes/counts/checks remain historical; only current pointers added on 2026-11-08."
        for item in value.get("imports", value.get("workbooks", [])):
            group = item.get("group_id")
            if group in by_group:
                p = by_group[group]
                item["current_workbook_path"] = p["new_path"]
                item["current_confirmed_origin"] = p["origin"]
                item["current_confirmed_species"] = p["species"]
                item["current_annotation_basis"] = BASIS
        write_json(path, value)
    for item in old_originals:
        p = by_group[item["origin_group_id"]]
        item["current_replacement_path"] = p["new_path"]
        item["confirmed_origin_20261108"] = p["origin"]
        item["confirmed_species_20261108"] = p["species"]
    write_csv(root / "data/archive/zaoren_superseded_20261031/manifest.csv", old_originals)
    annotation = {
        "declaration_id": "origin-kind-confirmation-20261108", "date": DATE,
        "user_statement": "去比对替换一下仓库名字，现在我确定所有表格的产地和种类了",
        "followup_statement": "给你分卷上传陕西延安枣仁",
        "basis": BASIS, "label_verification_level": "user_confirmation",
        "migration_base_commit": base_commit,
        "filename_policy": "Use exact user-confirmed basenames; retain 天麻安徽毫州.xlsx spelling and province-only granularity for 山西酸枣仁/内蒙古酸枣仁.",
        "supersedes": "Earlier filename-based hints and exploratory spectral origin guesses, including old 河北 wording for ZY03.",
        "scope": {"workbooks": 18, "spectrum_records": 1672, "dataset_version": VERSION},
        "limitations": ["Category labels are user-confirmed, not independent botanical or laboratory species authentication.",
                       "The 2026-10-31 independence/single-batch declaration still applies only to ZY01-ZY07 (855 records).",
                       "No batch identifiers, new independent cohorts or new model scores were created.",
                       "Historical anonymous-group experiments now have mixed 枣仁/酸枣仁 labels; do not relabel their accuracy as pure within-species geographical identification."],
        "workbooks": plans,
        "yanan_recovery": {"restored_filename": "陕西延安枣仁.xlsx", "sha256": yanan_sha,
                            "matches_historical_original": original_by_group["ZY01"]["archived_path"],
                            "verification_record": "data/validation/confirmed_names_20261108/yanan_recovery.json"},
    }
    write_json(root / "data/annotations/origin_kind_confirmation_20261108.json", annotation)
    validation = root / "data/validation/confirmed_names_20261108"
    validation.mkdir(parents=True, exist_ok=True)
    write_json(validation / "upload_audit.json", [safe_audit(r) for r in audited.values()])
    write_json(validation / "gansu_numeric_compare.json", read_json(args.gansu_comparison))
    # Recovery paths are execution-local; persist only portable evidence.
    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, str) and value.startswith("/workspace/"):
            return Path(value).name
        return value
    write_json(validation / "yanan_recovery.json", portable(recovery))
    wb = Workbook()
    ws = wb.active
    ws.title = "当前18份工作簿"
    columns = ["group_id", "uploaded_filename", "origin", "species", "record_count", "old_path", "new_path", "source_original_path", "match_basis", "source_sha256", "output_sha256"]
    ws.append(columns)
    for p in plans:
        ws.append([p[k] for k in columns])
        ws.cell(ws.max_row, 7).hyperlink = str(Path(p["new_path"]).relative_to("data"))
    notes = wb.create_sheet("说明")
    for row in [("生效日期", DATE), ("数据版本", VERSION), ("当前数量", "18份工作簿，1672条光谱"),
                ("种类", "枣仁364；酸枣仁500；黄芪403；天麻405"),
                ("依据", "用户确认产地与类别；不是独立实验鉴定"),
                ("旧编号", "ZY01—ZY07及原scan_id/record_id/physical_sample_id保持不变"),
                ("新编号", "HQ/ZR/TM为元数据记录标识；未重写新增原表头，未认定独立物理样品"),
                ("甘肃替换", "补齐折射率；已有表8格仅浮点末位变化，详见核验JSON"),
                ("历史", "旧原件、旧编号对照、历史声明及实验结果保留当时事实")]:
        notes.append(row)
    for sheet in wb:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="24465A")
        for col in sheet.columns:
            width = min(65, max(15, max(len(str(c.value or "")) for c in col) * 1.25))
            sheet.column_dimensions[col[0].column_letter].width = width
            for cell in col:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for index in range(2, sheet.max_row + 1):
            sheet.row_dimensions[index].height = 45
    wb.save(root / "data/confirmed_workbooks_20261108.xlsx")
    print(json.dumps({"version": VERSION, "workbooks": len(plans), "records": len(metadata), "kind_counts": registry["kind_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--upload-audit", type=Path, required=True)
    parser.add_argument("--recovered-yanan", type=Path, required=True)
    parser.add_argument("--recovery-json", type=Path, required=True)
    parser.add_argument("--gansu-comparison", type=Path, required=True)
    apply(parser.parse_args())
