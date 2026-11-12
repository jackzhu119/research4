"""Generate deterministic methods tables from the already-frozen specimen roles.

No output is written by this module. The paper builder owns publication and its
manifest. Workbook dimensions are reconstructed from real row tags and header
columns because these source XLSX files report an incorrect OOXML dimension A1.
"""

from collections import Counter
from datetime import datetime
import csv
import hashlib
import io
import json
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


ROLES = ("train", "calibration", "prediction")
ROLE_ZH = {"train": "训练", "calibration": "校准", "prediction": "预测"}
SPECIES = ("枣仁", "酸枣仁", "黄芪", "天麻")
SHEETS = {"时域": "time", "频域": "frequency", "折射率": "refractive_index", "吸收系数": "absorption"}
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS = {"m": MAIN_NS}
REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
FIXED_TIME = datetime(2026, 11, 12, 0, 0, 0)
EXPORTED_PATHS = (
    "methods/dataset_by_species.csv",
    "methods/dataset_by_workbook.csv",
    "methods/record_role_assignment.csv",
    "methods/task_role_counts.csv",
    "methods/nominal_thickness_by_workbook.csv",
    "methods/dataset_split_audit.json",
    "methods/data_splits.xlsx",
)
ROW_REF = re.compile(rb"<row\b[^>]*\br=[\"']([0-9]+)[\"']")
THICKNESS = re.compile(r"^(.*?)_(\d+(?:\.\d+)?)\s*mm$", re.IGNORECASE)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _csv_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def _csv_bytes(rows):
    if not rows:
        raise ValueError("An exported table cannot be empty")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def _column_number(column):
    number = 0
    for char in column:
        if char < "A" or char > "Z":
            raise ValueError("Invalid Excel column")
        number = number * 26 + ord(char) - ord("A") + 1
    return number


def _worksheet_header(raw, strings):
    reported_dimension = None
    header = None
    for _, element in ET.iterparse(io.BytesIO(raw), events=("end",)):
        if element.tag == "{" + MAIN_NS + "}dimension":
            reported_dimension = element.attrib.get("ref")
        if element.tag == "{" + MAIN_NS + "}row":
            if element.attrib.get("r") != "1":
                raise ValueError("Expected a first-row workbook header")
            header = {}
            for cell in element:
                address = cell.attrib.get("r", "")
                match = re.fullmatch(r"([A-Z]+)1", address)
                if not match:
                    raise ValueError("Unexpected header cell reference")
                value = cell.find("m:v", NS)
                if cell.attrib.get("t") == "s":
                    text = strings[int(value.text)]
                elif cell.attrib.get("t") == "inlineStr":
                    inline = cell.find("m:is", NS)
                    text = "".join(inline.itertext()) if inline is not None else ""
                else:
                    text = value.text if value is not None else ""
                header[match.group(1)] = text
            break
    if not header:
        raise ValueError("Missing worksheet header")
    row_numbers = [int(match.group(1)) for match in ROW_REF.finditer(raw)]
    if not row_numbers or row_numbers[0] != 1 or len(row_numbers) != len(set(row_numbers)):
        raise ValueError("Missing or duplicate worksheet row references")
    columns = max(_column_number(c) for c in header)
    if {_column_number(c) for c in header} != set(range(1, columns + 1)):
        raise ValueError("Non-contiguous workbook header columns")
    return {
        "header": header,
        "reported_dimension": reported_dimension,
        "rows_including_header": max(row_numbers),
        "row_tags": len(row_numbers),
        "columns_including_coordinates": columns,
        "reconstructed_dimension": "A1:" + get_column_letter(columns) + str(max(row_numbers)),
    }


def _workbook_headers(raw):
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = list(workbook.find("m:sheets", NS))
        if [sheet.attrib["name"] for sheet in sheets] != list(SHEETS):
            raise ValueError("Unexpected workbook views")
        relationships = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(item.itertext()) for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        for sheet in sheets:
            target = relationships[sheet.attrib[REL_ID]]
            target = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            result[sheet.attrib["name"]] = _worksheet_header(archive.read(target), strings)
    return result


HEADERS = {
    "species": "药材种类", "workbooks": "工作簿数", "records": "样品数（条）",
    "train": "训练集", "calibration": "校准集", "prediction": "预测集",
    "origin_group_id": "来源组编号", "origin": "用户确认产地", "filename": "当前文件名",
    "source_workbook": "当前工作簿路径", "sha256": "原谱SHA-256",
    "record_id": "记录编号", "physical_sample_id": "实物样品编号",
    "physical_sample_id_basis": "实物独立性声明依据", "absorption_sheet": "吸收页原名",
    "absorption_excel_column": "吸收页Excel列", "scan_header": "吸收页原列头",
    "role": "固定角色代码", "role_zh": "固定角色", "time_excel_column": "时域页Excel列",
    "time_scan_header": "时域页原列头", "nominal_thickness_mm": "列头标称厚度（mm）",
    "thickness_basis": "厚度依据", "task": "任务代码", "task_name": "任务范围",
    "class_count": "类别数", "classes": "当前类集", "training_cv": "已保存的训练内选择方式",
    "thickness_1_5mm": "标称1.5 mm数量", "thickness_1mm": "标称1 mm数量",
    "other_or_missing_thickness": "其他或缺失厚度数量", "note": "说明",
    "source": "来源", "path": "路径", "sha256_source": "SHA-256", "scope": "适用范围",
    "instrument_name": "仪器名称", "instrument_model": "型号", "manufacturer": "设备厂商",
    "parameter": "测量项目", "value": "参数／内容",
}
for _name, _prefix in SHEETS.items():
    HEADERS.update({
        _prefix + "_sheet_name": _name + "页原名",
        _prefix + "_rows": _name + "行数（含表头）",
        _prefix + "_columns": _name + "列数（含坐标/参考）",
        _prefix + "_sample_columns": _name + "样品列数",
        _prefix + "_reported_dimension": _name + "原声明尺寸",
        _prefix + "_reconstructed_dimension": _name + "重建尺寸",
    })


def _styled_sheet(book, name, rows):
    sheet = book.create_sheet(name)
    keys = list(rows[0])
    sheet.append([HEADERS.get(key, key) for key in keys])
    for row in rows:
        sheet.append([row[key] for key in keys])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.row_dimensions[1].height = 34
    sheet.sheet_view.showGridLines = False
    sheet.print_title_rows = "1:1"
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.paperSize = sheet.PAPERSIZE_A3
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    header_fill = PatternFill("solid", fgColor="17365D")
    stripe_fill = PatternFill("solid", fgColor="EFF4FA")
    bottom = Border(bottom=Side(style="thin", color="D6DFEA"))
    role_colors = {"训练": "E5F2E5", "校准": "FFF1CC", "预测": "E6EDFA"}
    for cell in sheet[1]:
        cell.font = Font(name="微软雅黑", bold=True, color="FFFFFF", size=10)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for cells in sheet.iter_rows(min_row=2):
        for cell in cells:
            cell.font = Font(name="微软雅黑", size=10)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = bottom
            if cell.row % 2 == 0:
                cell.fill = stripe_fill
        if "role_zh" in keys:
            role_cell = cells[keys.index("role_zh")]
            role_cell.fill = PatternFill("solid", fgColor=role_colors[role_cell.value])
    for index, key in enumerate(keys, 1):
        widths = [sum(2 if ord(char) > 127 else 1 for char in str(sheet.cell(row, index).value or "")) for row in range(1, min(sheet.max_row, 100) + 1)]
        width = min(46, max(13, max(widths) + 2))
        if key.endswith("sha256") or key in ("sha256_source", "source_workbook", "path"):
            width = 43
        sheet.column_dimensions[get_column_letter(index)].width = width
    return sheet


def _xlsx_bytes(tables, source_rows, equipment_rows, measurement_rows):
    book = Workbook()
    book.remove(book.active)
    book.properties.creator = "Research4"
    book.properties.lastModifiedBy = "Research4"
    book.properties.title = "修正种类与产地后的数据量及冻结角色划分"
    book.properties.created = FIXED_TIME
    book.properties.modified = FIXED_TIME
    for name, rows in tables:
        _styled_sheet(book, name, rows)
    _styled_sheet(book, "主要仪器", equipment_rows)
    _styled_sheet(book, "测量条件", measurement_rows)
    _styled_sheet(book, "来源说明", source_rows)
    stream = io.BytesIO()
    book.save(stream)
    normalized = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as source, zipfile.ZipFile(normalized, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        for name in sorted(source.namelist()):
            raw = source.read(name)
            if name == "docProps/core.xml":
                raw = re.sub(rb"(<dcterms:(?:created|modified)\b[^>]*>).*?(</dcterms:(?:created|modified)>)", rb"\g<1>2026-11-12T00:00:00Z\g<2>", raw)
            info = zipfile.ZipInfo(name, (2026, 11, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            target.writestr(info, raw)
    return normalized.getvalue()


def generate(root: Path) -> dict[str, bytes]:
    """Return paper/current-relative artifacts without changing any source file."""
    root = Path(root).resolve()
    current = root / "paper/current"
    paths = ["inputs/active_workbooks.json", "inputs/metadata.csv", "inputs/split_manifest.csv", "inputs/split_protocol.json", "protocols/fixed_classification.json"]
    sources = {"paper/current/" + path: (current / path).read_bytes() for path in paths}
    declaration_path = "data/annotations/acquisition_conditions_user_declaration_20261112.json"
    sources[declaration_path] = (root / declaration_path).read_bytes()
    registry = json.loads(sources["paper/current/inputs/active_workbooks.json"])
    protocol = json.loads(sources["paper/current/inputs/split_protocol.json"])
    declaration = json.loads(sources[declaration_path])
    configuration = json.loads(sources["paper/current/protocols/fixed_classification.json"])
    rows = _csv_rows(sources["paper/current/inputs/split_manifest.csv"])
    metadata = _csv_rows(sources["paper/current/inputs/metadata.csv"])
    expected_hashes = {"inputs/active_workbooks.json": "active_registry_sha256", "inputs/metadata.csv": "metadata_sha256", "inputs/split_manifest.csv": "manifest_sha256"}
    for relative, key in expected_hashes.items():
        if _sha(sources["paper/current/" + relative]) != protocol[key]:
            raise ValueError("Frozen input checksum differs: " + relative)
    if configuration["split_sha256"] != protocol["manifest_sha256"]:
        raise ValueError("Classification configuration does not use the frozen master roles")
    by_id = {row["record_id"]: row for row in rows}
    meta_by_id = {row["record_id"]: row for row in metadata}
    if len(rows) != 1672 or len(by_id) != 1672 or len(metadata) != 1672 or set(by_id) != set(meta_by_id):
        raise ValueError("Specimen IDs are missing, duplicated or not one-to-one")
    if any(not row["record_id"] or not row["physical_sample_id"] for row in rows) or len({row["physical_sample_id"] for row in rows}) != 1672:
        raise ValueError("Physical specimen IDs are not unique")
    for row in rows:
        if row["role"] not in ROLES:
            raise ValueError("Unknown frozen role")
        for key in ("physical_sample_id", "physical_sample_id_basis", "origin_group_id", "species", "origin", "source_workbook", "excel_column", "scan_header"):
            if row[key] != meta_by_id[row["record_id"]][key]:
                raise ValueError("Metadata and frozen role map disagree: " + row["record_id"] + "/" + key)
    if set(declaration["applies_to_species"]) != set(SPECIES):
        raise ValueError("Acquisition declaration does not cover the four herb categories")

    by_workbook, assignments, thickness_rows, workbook_audit = [], [], [], []
    covered_ids = set()
    for workbook in registry["workbooks"]:
        path = workbook["path"]
        raw = (root / path).read_bytes()
        actual_sha = _sha(raw)
        if actual_sha != workbook["sha256"] or protocol["workbook_hashes"][path] != actual_sha:
            raise ValueError("Original spectral workbook changed: " + path)
        group = sorted((row for row in rows if row["source_workbook"] == path), key=lambda row: _column_number(row["excel_column"]))
        if len(group) != workbook["record_count"] or not group:
            raise ValueError("Workbook sample count differs")
        if len({row["excel_column"] for row in group}) != len(group):
            raise ValueError("Two IDs map to the same source sample column")
        sheet_info = _workbook_headers(raw)
        record = {"origin_group_id": workbook["origin_group_id"], "species": workbook["species"], "origin": workbook["origin"], "filename": Path(path).name, "source_workbook": path, "records": len(group)}
        counts = Counter(row["role"] for row in group)
        record.update({role: counts[role] for role in ROLES})
        thick = Counter()
        for name, prefix in SHEETS.items():
            info = sheet_info[name]
            offset = 2 if name in ("时域", "频域") else 1
            sample_columns = info["columns_including_coordinates"] - offset
            if sample_columns != len(group):
                raise ValueError("Workbook view sample count differs: " + path + "/" + name)
            record.update({prefix + "_sheet_name": name, prefix + "_rows": info["rows_including_header"], prefix + "_columns": info["columns_including_coordinates"], prefix + "_sample_columns": sample_columns, prefix + "_reported_dimension": info["reported_dimension"], prefix + "_reconstructed_dimension": info["reconstructed_dimension"]})
        for ordinal, row in enumerate(group, 1):
            if row["species"] != workbook["species"] or row["origin"] != workbook["origin"] or row["origin_group_id"] != workbook["origin_group_id"]:
                raise ValueError("Workbook kind/origin label disagrees with its sample map")
            absorption_column = get_column_letter(ordinal + 1)
            time_column = get_column_letter(ordinal + 2)
            if row["excel_column"] != absorption_column:
                raise ValueError("Frozen source column order changed")
            for name in ("吸收系数", "折射率"):
                if sheet_info[name]["header"][absorption_column] != row["scan_header"]:
                    raise ValueError("Source optical-view header mismatch")
            if sheet_info["频域"]["header"][time_column] != row["scan_header"]:
                raise ValueError("Frequency-view header mismatch")
            time_header = sheet_info["时域"]["header"][time_column]
            match = THICKNESS.fullmatch(time_header)
            if not match or match.group(1) != row["scan_header"]:
                raise ValueError("Time-view header does not map to the optical sample")
            nominal = float(match.group(2))
            thick[nominal] += 1
            covered_ids.add(row["record_id"])
            assignments.append({"record_id": row["record_id"], "physical_sample_id": row["physical_sample_id"], "physical_sample_id_basis": row["physical_sample_id_basis"], "origin_group_id": row["origin_group_id"], "species": row["species"], "origin": row["origin"], "source_workbook": path, "absorption_sheet": "吸收系数", "absorption_excel_column": absorption_column, "scan_header": row["scan_header"], "role": row["role"], "role_zh": ROLE_ZH[row["role"]], "time_excel_column": time_column, "time_scan_header": time_header, "nominal_thickness_mm": int(nominal) if nominal.is_integer() else nominal, "thickness_basis": "时域列头标称值；非逐片独立测厚"})
        by_workbook.append(record)
        thickness_rows.append({"origin_group_id": workbook["origin_group_id"], "species": workbook["species"], "origin": workbook["origin"], "source_workbook": path, "records": len(group), "thickness_1_5mm": thick[1.5], "thickness_1mm": thick[1.0], "other_or_missing_thickness": len(group) - thick[1.5] - thick[1.0], "thickness_basis": "原时域样品列头后缀，未提供逐片测厚记录"})
        workbook_audit.append({"path": path, "sha256": actual_sha, "records": len(group), "role_counts": {role: counts[role] for role in ROLES}, "nominal_thickness_counts_mm": {str(key): thick[key] for key in sorted(thick)}, "sheets": {name: {key: value for key, value in info.items() if key != "header"} for name, info in sheet_info.items()}})
    if covered_ids != set(by_id) or len(by_workbook) != 18:
        raise ValueError("Registry coverage differs from the frozen sample map")
    totals = Counter(row["role"] for row in rows)
    if totals != {"train": 999, "calibration": 334, "prediction": 339}:
        raise ValueError("Frozen master role counts differ")
    nominal_counts = Counter(row["nominal_thickness_mm"] for row in assignments)
    one_mm_ids = [row["record_id"] for row in assignments if row["nominal_thickness_mm"] == 1]
    if nominal_counts != {1.5: 1670, 1: 2} or one_mm_ids != ["ZY02-001", "ZY02-002"]:
        raise ValueError("Nominal thickness source labels changed")
    by_species = []
    expected_species_counts = {"枣仁": 364, "酸枣仁": 500, "黄芪": 403, "天麻": 405}
    for species in SPECIES:
        group = [row for row in rows if row["species"] == species]
        counts = Counter(row["role"] for row in group)
        if len(group) != expected_species_counts[species]:
            raise ValueError("Corrected species count differs")
        by_species.append({"species": species, "workbooks": sum(w["species"] == species for w in by_workbook), "records": len(group), **{role: counts[role] for role in ROLES}})
    class_counts = [("binary_kinds", "枣仁／酸枣仁二分类", ("枣仁", "酸枣仁"), 2), ("acid_origins", "纯酸枣仁五来源", ("酸枣仁",), 5), ("four_kinds", "四种类分类", SPECIES, 4), ("huangqi_origins", "黄芪四来源", ("黄芪",), 4), ("tianma_origins", "天麻四来源", ("天麻",), 4), ("jujube_origins", "纯枣仁五来源", ("枣仁",), 5)]
    task_rows = []
    for task, name, kinds, count in class_counts:
        if task not in configuration["tasks"]:
            raise ValueError("Task not present in frozen classification configuration")
        group = [row for row in rows if row["species"] in kinds]
        roles = Counter(row["role"] for row in group)
        classes = sorted({row["species"] if task in ("binary_kinds", "four_kinds") else row["origin_group_id"] for row in group})
        if len(classes) != count:
            raise ValueError("Corrected task classes differ")
        cv = "TRAIN内" + str(configuration["inner_cv"]) + "折源侧选择" if task in configuration["main_tasks"] else "固定既有参数，不新调参"
        task_rows.append({"task": task, "task_name": name, "class_count": count, "classes": " / ".join(classes), "records": len(group), **{role: roles[role] for role in ROLES}, "training_cv": cv})

    source_rows = [{"source": "冻结输入或用户声明", "path": path, "sha256_source": _sha(raw), "scope": "原字节读取；未修改角色或原谱"} for path, raw in sources.items()]
    source_rows.extend({"source": "当前原始光谱工作簿", "path": item["path"], "sha256_source": item["sha256"], "scope": "四页共享同一组样品列，未四倍计数"} for item in workbook_audit)
    source_rows.extend([
        {"source": "角色说明", "path": "TRAIN / CAL / PRED", "sha256_source": "", "scope": "训练内选参；校准仅预设适配；预测仅评分。预设全量重训对照可用TRAIN+CAL，但不以PRED选模型。"},
        {"source": "角色与内部CV", "path": "主划分999/334/339；V4使用TRAIN内酸枣仁300", "sha256_source": "", "scope": "V4的240/60外折及B/meta内部Source/Support/Query不改变主角色。"},
        {"source": "表头尺寸", "path": "OOXML dimension / 重建尺寸", "sha256_source": "", "scope": "原声明A1不可信；行数由真实row标签重建，列数从已核对的首行列头确定。行数含表头，不是样品数。"},
        {"source": "标称厚度", "path": "1.5 mm×1670；1 mm×2", "sha256_source": "", "scope": "1 mm仅ZY02-001和ZY02-002。来自时域列头，不是独立逐片测厚。"},
        {"source": "小样本与批次", "path": "ZR01—ZR03各1/1/1", "sha256_source": "", "scope": "新枣仁来源每角色仅1份；真实新批次验证未运行，不能由角色互斥推断跨批次有效。"},
    ])
    equipment_rows = [{"instrument_name": item["name"], "instrument_model": item["model"], "manufacturer": item["manufacturer"]} for item in declaration["equipment"]]
    measurement = declaration["measurement"]
    measurement_rows = [
        {"parameter": "适用药材", "value": " / ".join(declaration["applies_to_species"]), "note": "用户2026-11-12统一确认"},
        {"parameter": "实验室", "value": measurement["laboratory"], "note": measurement["technique"]},
        {"parameter": "温度", "value": str(measurement["temperature_k"]) + " K", "note": "用户提供的测量条件"},
        {"parameter": "光学参数", "value": str(measurement["optical_center_wavelength_nm"]) + " nm / " + str(measurement["optical_power_mw"]) + " mW", "note": "光学激发/探测相关参数，不是太赫兹中心波长/发射功率"},
        {"parameter": "环境", "value": "氮气充罩；湿度<" + str(measurement["enclosure_humidity_percent_upper_exclusive"]) + "%", "note": "未补写湿度计定义与运行日志"},
        {"parameter": "参考", "value": declaration["reference"]["procedure_zh"], "note": "用户确认已作聚乙烯参考响应校正；设置仍待补"},
        {"parameter": "采购", "value": declaration["procurement"]["description_zh"], "note": "标称产地与用户声明，不是独立采购溯源审计"},
    ]
    audit = {"schema": "research4-methods-split-table-audit-v1", "dataset_version": registry["version"], "frozen_sources_sha256": {path: _sha(raw) for path, raw in sources.items()}, "split_seed": protocol["split_seed"], "model_seed": protocol["model_seed"], "master_counts": {"workbooks": 18, "records": 1672, **{role: totals[role] for role in ROLES}}, "by_species": by_species, "tasks": task_rows, "nominal_thickness_counts_mm": {"1.5": 1670, "1": 2}, "nominal_1mm_record_ids": one_mm_ids, "thickness_is_header_nominal_not_independent_measurement": True, "workbooks": workbook_audit, "checks": {"unique_record_and_physical_ids": True, "metadata_split_one_to_one": True, "all_frozen_roles_unchanged": True, "all_eighteen_original_sha256_equal": True, "all_four_view_sample_headers_mapped": True, "all_four_views_counted_once": True, "no_random_assignment_or_model_fit": True, "deterministic_output": True}, "dimension_note": "Source OOXML dimension often A1. Rows reconstructed from real row references; columns are mapped first-row columns including coordinate/reference. These are table dimensions, not additional specimens.", "no_cross_batch_claim": True}
    output = {
        "methods/dataset_by_species.csv": _csv_bytes(by_species),
        "methods/dataset_by_workbook.csv": _csv_bytes(by_workbook),
        "methods/record_role_assignment.csv": _csv_bytes(assignments),
        "methods/task_role_counts.csv": _csv_bytes(task_rows),
        "methods/nominal_thickness_by_workbook.csv": _csv_bytes(thickness_rows),
        "methods/dataset_split_audit.json": (json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    }
    output["methods/data_splits.xlsx"] = _xlsx_bytes([("各药材汇总", by_species), ("各工作簿划分", by_workbook), ("逐样品角色", assignments), ("六项任务", task_rows), ("标称厚度", thickness_rows)], source_rows, equipment_rows, measurement_rows)
    if set(output) != set(EXPORTED_PATHS):
        raise ValueError("Methods-table output set differs from its declared paths")
    return output
