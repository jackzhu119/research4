"""Export real final calibration predictions, without fitting or selection."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill

HERE=Path(__file__).resolve().parent
metrics=json.loads((HERE/"run/final_metrics.json").read_text())
summary=json.loads((HERE/"run/summary.json").read_text())
per_class=[]
for row in metrics:
    cm=np.asarray(row["confusion_matrix"],dtype=int)
    support,found,tp=cm.sum(1),cm.sum(0),cm.diagonal()
    precision=np.divide(tp,found,out=np.zeros(7),where=found>0)
    recall=np.divide(tp,support,out=np.zeros(7),where=support>0)
    f1=np.divide(2*tp,support+found,out=np.zeros(7),where=support+found>0)
    assert abs(float(f1.mean())-row["macro_f1"])<1e-12
    assert abs(float(recall.mean())-row["balanced_accuracy"])<1e-12
    for i in range(7):
        per_class.append({"selector":row["selector"],"method":row["method"],"repeat":row["repeat"],"class":f"ZY{i+1:02d}",
                          "precision":float(precision[i]),"recall":float(recall[i]),"f1":float(f1[i]),
                          "true_support":int(support[i]),"predicted_count":int(found[i]),"true_positives":int(tp[i])})
grouped=defaultdict(list)
for row in per_class:
    grouped[(row["selector"],row["method"],row["class"])].append(row)
means=[]
for key,rows in grouped.items():
    row=dict(zip(("selector","method","class"),key));row["support_draws"]=len(rows)
    for name in ("precision","recall","f1","true_support","predicted_count","true_positives"):
        values=np.array([r[name] for r in rows])
        row[name+"_mean"]=float(values.mean());row[name+"_sd"]=float(values.std())
    means.append(row)
confusions=[]
for selector,method in sorted({(r["selector"],r["method"]) for r in metrics}):
    cases=[r for r in metrics if r["selector"]==selector and r["method"]==method]
    cm=np.mean([np.array(r["confusion_matrix"]) for r in cases],axis=0)
    for i in range(7):
        confusions.append({"selector":selector,"method":method,"true_class":f"ZY{i+1:02d}",
                           **{f"pred_ZY{j+1:02d}":float(cm[i,j]) for j in range(7)}})
flat=[]
for row in metrics:
    flat.append({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})
tables={"抽样汇总":summary,"逐抽样指标":flat,"逐类均值":means,"逐抽样逐类":per_class,"平均混淆矩阵":confusions}
book=Workbook();book.remove(book.active)
for name,rows in tables.items():
    with (HERE/f"{name}.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
    sheet=book.create_sheet(name);sheet.append(list(rows[0]));sheet.freeze_panes="C2"
    for row in rows:sheet.append(list(row.values()))
    sheet.auto_filter.ref=sheet.dimensions
    for c in sheet[1]:c.font=Font(bold=True,color="FFFFFF");c.fill=PatternFill("solid",fgColor="244F70")
    for column in sheet.columns:
        sheet.column_dimensions[column[0].column_letter].width=25
    sheet.column_dimensions["B"].width=45
    for row in sheet.iter_rows(min_row=2):
        for c in row:
            if isinstance(c.value,float):c.number_format="0.000000"
output=HERE/"少样本保守校准_完整指标.xlsx"
book.save(output)
verified=0;maxdiff=0.
readback=load_workbook(output,read_only=True,data_only=True)
for name,expected in tables.items():
    actual=list(readback[name].values)
    assert list(actual[0])==list(expected[0])
    for av,ev in zip(actual[1:],expected,strict=True):
        for a,b in zip(av,ev.values(),strict=True):
            if isinstance(b,float):
                difference=abs(a-b);maxdiff=max(maxdiff,difference);assert difference<1e-10
            else:assert a==b
            verified+=1
readback.close()
(HERE/"table_validation.json").write_text(json.dumps({"status":"pass","final_evaluations":len(metrics),
    "per_class_draw_rows":len(per_class),"per_class_mean_rows":len(means),"confusion_rows":len(confusions),
    "excel_cells_checked":verified,"maximum_numeric_difference":maxdiff,"new_model_fits":0,
    "input_sha256":{"run/final_metrics.json":hashlib.sha256((HERE/"run/final_metrics.json").read_bytes()).hexdigest()},
    "workbook_sha256":hashlib.sha256(output.read_bytes()).hexdigest()},indent=2)+"\n")
print(json.dumps({"status":"pass","excel_cells_checked":verified,"maximum_numeric_difference":maxdiff}))
