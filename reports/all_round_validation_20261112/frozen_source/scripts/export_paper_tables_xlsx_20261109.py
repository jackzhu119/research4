#!/usr/bin/env python3
"""Convenience workbook from frozen CSVs. Requires openpyxl; never fits models."""
import csv
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


def main():
    root = Path(__file__).resolve().parents[1] / "reports/paper-evidence-20261109"
    book = Workbook()
    intro = book.active
    intro.title = "READ_ME"
    for row in [
        ["Research4论文结果工作簿", "2026-11-09保存证据汇编；无新增训练"],
        ["主模型", "纯酸枣仁300TRAIN，3×5外折/2内折；MC20 Accuracy93.4444%, Macro-F1 .9344009122"],
        ["独立样品", "900条重复预测来自300份物理样品，不是900份新样品"],
        ["协议", "不同sheet/行的任务、角色与搜索不同，不能按分数混合排名"],
        ["历史96.0234%", "855份混合枣仁/酸枣仁七来源任务，不是纯酸五来源"],
        ["校准", "纯酸五来源无稳定改善；B二分类来源适配有协议内收益，任务分开"],
        ["量子与未运行", "查docs/paper-writing-20261109/HISTORY_QUANTUM_AND_NOT_RUN.md"],
        ["数值依据", "工作簿保留CSV原文本；CSV/JSON及其SHA是权威证据"],
        ["全部原始结果", "archives/README.md提供1635任务的小结果、预测覆盖与恢复"],
    ]:
        intro.append(row)
    tables = [
        ("主模型同协议", "current_classical_relative_to_raw.csv"),
        ("全部最终设置", "all_final_model_settings.csv"),
        ("特征与参数", "feature_and_parameter_selection.csv"),
        ("逐类指标", "current_per_class_by_repeat.csv"),
        ("历史实验登记", "historical_quantum_experiment_register.csv"),
        ("模型家族协议", "model_protocol_register.csv"),
        ("B_V2神经对照", "parallel_b_v2_neural_complete_blocks.csv"),
        ("B_V2任务", "parallel_b_v2_experiment_register.csv"),
        ("B_V3任务", "parallel_b_experiment_register.csv"),
        ("科学任务1635", "scientific_job_register.csv"),
        ("来源分支", "branch_inventory.csv"),
    ]
    for title, name in tables:
        sheet = book.create_sheet(title)
        with (root / "tables" / name).open(encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                if any(len(cell) > 32767 for cell in row):
                    raise ValueError(f"Excel cell limit exceeded: {name}")
                sheet.append(row)
        sheet.auto_filter.ref = sheet.dimensions
    for sheet in book:
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="24476A")
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for i in range(1, sheet.max_column + 1):
            sheet.column_dimensions[get_column_letter(i)].width = 25 if i < 4 else 20
    intro.column_dimensions["B"].width = 105
    book.save(root / "RESEARCH4_PAPER_RESULTS_20261109.xlsx")


if __name__ == "__main__":
    main()
