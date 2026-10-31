"""Render already-scored neural evidence; no fitting, selection or score changes."""
from pathlib import Path
import csv
import hashlib
import json


NAMES={"cnn_alpha":"轻量CNN：α", "resnet_alpha":"ResNet1D：α", "se_alpha":"SE-ResNet1D：α",
    "dual_alpha_mcuve":"完整α＋MC-UVE点MLP双支", "fusion_alpha_n":"α＋n双支", "cnn_n":"轻量CNN：n",
    "fusion_time_frequency":"时域＋频域双支", "fusion_all_four":"四表示CNN融合", "inner_selected_neural":"内层选择的主神经策略"}


def main(root):
    primary=json.loads((root/"primary_seed42_metrics.json").read_text());pooled=json.loads((root/"pooled_metrics_by_seed.json").read_text());sensitivity=json.loads((root/"seed_sensitivity.json").read_text())
    comparison=[];all_seeds=[];classes=[]
    for family,stats in primary.items():
        mean=sensitivity[family]
        row={"family":family,"display_name":NAMES[family],"protocol":"upgrade-20261031-v1-neural-small-budget/prefit-v3","real_samples":855,"primary_seed":42,
            "correct_seed42":stats["correct"],"accuracy_seed42":stats["accuracy"],"macro_f1_seed42":stats["macro_f1"],"balanced_accuracy_seed42":stats["balanced_accuracy"],
            "accuracy_mean_3seeds":mean["accuracy"]["mean"],"accuracy_sample_sd_3seeds":mean["accuracy"]["sample_sd"],
            "macro_f1_mean_3seeds":mean["macro_f1"]["mean"],"macro_f1_sample_sd_3seeds":mean["macro_f1"]["sample_sd"],
            "ZY02_recall_seed42":stats["per_class"]["ZY02"]["recall"],"observed_gain_pp_vs_historical_R2":100*(stats["accuracy"]-821/855),
            "scope":"EXPLORATORY exposed single-batch; no independent confirmation"}
        comparison.append(row)
        for seed,value in pooled[family].items():
            all_seeds.append({"family":family,"seed":int(seed),"correct":value["correct"],"total":value["total"],"accuracy":value["accuracy"],"macro_f1":value["macro_f1"],"balanced_accuracy":value["balanced_accuracy"]})
            for label,item in value["per_class"].items():classes.append({"family":family,"seed":int(seed),"label":label,**item})
    for name,rows in [("model_comparison.csv",comparison),("all_seeds.csv",all_seeds),("class_metrics.csv",classes)]:
        with (root/name).open("w",encoding="utf-8-sig",newline="") as h:
            writer=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    lines=["# 新多表示神经最小流程：实际结果与隔离边界","","2026-10-31。保留历史MC-UVE＋SVM的821／855＝96.0234%、Macro-F1 0.9197作为强传统参照。本轮没有改写旧数据、权重或分数。","",
        "本次是固定小预算的新路线验证，不是穷举神经最优：同855份、原五外折／二内折，每家族两个预定耦合profile，全部seed42／43／44。当前历史样品已经多轮曝光且每产地仅一批，科学状态为EXPLORATORY，不是新批次确认。","",
        "## 全部模型结果","","| 模型 | seed42正确／855 | Accuracy | Macro-F1 | Balanced Accuracy | 三seed准确率均值±样本SD（百分点） |","| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in comparison:lines.append(f"| {r['display_name']} | {r['correct_seed42']} | {100*r['accuracy_seed42']:.4f}% | {r['macro_f1_seed42']:.4f} | {100*r['balanced_accuracy_seed42']:.4f}% | {100*r['accuracy_mean_3seeds']:.4f}% ± {100*r['accuracy_sample_sd_3seeds']:.4f} |")
    lines += ["","固定主策略由每外折源内部的8家族×2profile×2内折×3seed均值选择，Macro-F1优先，随后Balanced Accuracy、Accuracy及固定家族／profile字典序。五折均选四表示regularized24，因此主策略与四表示族预测精确相同；这是预设策略别名，不增加模型或真实样品数。没有按外层最高分回填家族。","",
        "本轮神经结果没有超过96.02%强传统基线；四表示较本轮α单支高，但不证明普遍融合有效或神经模型达到最优。两profile同时改变宽度、深度、dropout、学习率、类别权重与训练轮数；这是小预算路线选择，不是逐因素因果消融。SE没有稳定胜出，MC双支的总体准确率与少数类指标存在取舍。不能从注意力权重推断化学成分含量。","",
        "## 实际执行与防护","",
        "- 480个真实内层模型／164160条内层预测，全部锁定后才获root外层放行。120个真实外层管线／15个主策略角色别名；23085条最终预测分为9角色×3seed，每个角色每seed恰855份。折指标和pooled指标分文件保存，seed重复不增加样品数。",
        "- source包仅含当前源标签；推理包严格只含record_ids和四个数值X字段，未知真值伪装字段／文件名特征拒绝。准备、选择、外层推理、独立评分是实际分开的CLI进程；外层先锁类别与七列logits，再连接真值评分。训练有源监督标签，metadata中的target_labels_received_by_training=false仅指禁止的外层query真值。",
        "- StandardScaler、MC-UVE及其每一次内部PLS拟合、torch训练均进入实际FitScope。MC调用原MCUVESelector，没有重写其数学；内层当前342份拟合，外层当前684份拟合，绝不使用历史UVE点并集或全855排序。非连续MC点只进入MLP，完整α才卷积。",
        "- 每内折／外折／seed全新模型、优化器；固定epoch12／24候选在内层选择，无外层早停。BatchNorm在推理使用源running统计，eval/no_grad。每外层模型全171份做单独／不同batch／逆序及混排检查，参数和buffer保持不变；内层每模型检查前7份，未把它称全内层样本覆盖。",
        "- 原四表以原编号和逐行源数值绑定，不按列号盲配。α/n为612点；时域6750点以同样品连续块均值降为512，频域1001点降为256；块边界只由原轴长度固定，未用标签、跨样品均值或查询统计，未增加分辨率。α逐谱SNV，再源StandardScaler；其它表示仅源StandardScaler。",
        "- 原始reference列不作为样品或模型特征；其仪器采集／反演来源未由本轮重新认证。时域／频域与α／n相关，同一测量链的四表示不是四批样品或四台独立传感器。",
        "- 代码／输入／训练ID／split／参数／seed绑定的learned-cache保留实际fit来源并拒绝跨折；本轮所有训练都是fresh fit，无resume。prefit schema及整体架构策略两次修订均在0真实source fit之前封存，没有在外层后改候选。",
        "",
        "文件/API及运行时hook隔离已经实测；没有OS级强制阻止同用户进程读取整个仓库，日志和SHA不是外部独立执行认证。技术检查不证明绝对零泄漏，也不替代未知祖先／批次溯源；ZY02仅5份。完整血缘及原数据限制见数据库和总防泄漏报告。","",
        "## 档案与重建","",
        "[逐seed比较](all_seeds.csv)、[主表](model_comparison.csv)、[逐类指标](class_metrics.csv)、[分折指标](fold_metrics.json)、[全部pooled指标](pooled_metrics_by_seed.json)与[选择锁](selection_lock.json)均来自实际保存预测。保存的score是原logits，分类交叉熵内部使用softmax；未做概率校准，不称已校准概率。","",
        "五个inner_outer_XX_evidence.tar.gz逐字节保存全部480个内层完整pipeline及训练/推理记录；outer120完整pipeline在evaluation/直接展开。fit_scope原JSONL采用无损gzip，原SHA见fit_scope_manifest.json；archive_validation.json记录每份原件／归档SHA和恢复路径。模型加载用原源码及load_fitted目录接口，跨折需指定expected_split_id和expected_source_ids，并先核对run_manifest全部源码版本。","",
        "prepared source/inference矩阵为约149MiB压缩缓存（未压缩矩阵约200MiB），不重复镜像进Git：原数据库、固定分区、原数值绑定与每pack SHA均保留，重建命令如下，必须使用全新输出目录，不能重用旧授权文件：","",
        "```bash",".runtime/bin/python scripts/run_upgrade_neural.py prepare --protocol data/research/upgrade_neural_protocol_20261031.json --output outputs/upgrade-neural-reproduction-new",
        "# 新source授权绑定当前protocol/preparation/全部源码SHA；选择完成后另核新锁并生成outer授权。",
        ".runtime/bin/python scripts/run_upgrade_neural.py selection --protocol data/research/upgrade_neural_protocol_20261031.json --output outputs/upgrade-neural-reproduction-new --release <new-source-gate.json>",
        ".runtime/bin/python scripts/run_upgrade_neural.py evaluation --protocol data/research/upgrade_neural_protocol_20261031.json --output outputs/upgrade-neural-reproduction-new --release <new-outer-release.json>",
        ".runtime/bin/python scripts/run_upgrade_neural.py score --protocol data/research/upgrade_neural_protocol_20261031.json --output outputs/upgrade-neural-reproduction-new","```","",
        "MC-only MLP／双支＋SE仅边界单测，真实研究NOT_RUN；CBAM／频率位置attention未实现。黄芪／天麻独立祖先与单位证据未通过，跨植物预训练、迁移、真实跨批次校准及DFT先验分类均NOT_RUN／BLOCKED，不填虚构分数。有限12／24轮预算的负结果不能概括所有深度学习配置。"]
    (root/"README.md").write_text("\n".join(lines)+"\n")
    index={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.name!="files_sha256.json"}
    (root/"files_sha256.json").write_text(json.dumps(index,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"role_seed_rows":len(all_seeds),"class_rows":len(classes),"files_indexed":len(index)}))


if __name__=="__main__":main(Path(__file__).resolve().parent)
