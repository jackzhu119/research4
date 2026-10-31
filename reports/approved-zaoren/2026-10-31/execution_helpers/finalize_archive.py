"""Finalize descriptive evidence metadata and hashes; no model execution."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv,hashlib,json

ARCHIVE=Path(__file__).resolve().parents[1]
ROOT=ARCHIVE.parents[2]
SHA=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (ARCHIVE/'independent-audit/pooled_metrics.csv').open(encoding='utf-8-sig',newline='') as f:metrics=list(csv.DictReader(f))
with (ARCHIVE/'independent-audit/comparison_deltas.csv').open(encoding='utf-8-sig',newline='') as f:deltas=list(csv.DictReader(f))
quality=json.loads((ARCHIVE/'data-preparation/quality_report.json').read_text())
generation=json.loads((ARCHIVE/'datasets/generation.json').read_text())
protocol=json.loads((ARCHIVE/'experiments/frozen_protocol.json').read_text())
audit=json.loads((ARCHIVE/'independent-audit/independent_recalculation.json').read_text())
synthetic=json.loads((ARCHIVE/'synthetic-transfer-results/status.json').read_text())
unit=json.loads((ARCHIVE/'tests_summary.json').read_text())
assert len(metrics)==13 and unit['tests']==100 and synthetic['evaluations']==21
source_hashes=[{'path':r['path'],'sha256':r['sha256'],'records':r['registered_records'],'origin_group_id':r['origin_group_id']} for r in quality['sources']]
assert all(SHA(ROOT/entry['path'])==entry['sha256'] for entry in source_hashes)
for name,value in protocol['source_code_sha256'].items():assert SHA(ARCHIVE/'experiments/execution_code'/name)==value
negative_absorption=sum(r['alpha']['negative_cells'] for r in quality['sources'])
negative_candidate=sum(r['alpha']['candidate_band_negative_cells'] for r in quality['sources'])
post_run={'reported_by':'User in the research chat','statement':'每个产地目前只有一批','known_after_main_matrix_execution':True,
          'source_batches_per_origin_group':1,'applies_to_groups':list(quality['group_counts']),
          'measurement_batch_dates_or_instrument_mapping':'not supplied',
          'interpretation':'Each origin group currently has only one source batch. Origin-group and source-batch effects cannot be separated. These OOF results describe internal recognition among this collection, not cross-batch geographic-origin performance.',
          'evidence_preservation':'Original frozen protocol, QA readiness and audit wording are retained as recorded at execution; this post-run declaration updates interpretation without changing predictions.'}
summary={'date':'2026-10-31','archived_at_shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
         'scope':'Single-source-batch anonymous origin-group internal recognition on user-declared independent acid jujube samples; origin and source batch are confounded.',
         'gaussian_included':False,'real_data':{'dataset_version':quality['dataset_version'],'origin_groups':7,'samples_user_declared_independent':855,'group_counts':quality['group_counts'],'geographic_names_known':False,'source_batches_per_group':1,'cross_batch_experiment_performed':False},
         'real_classification_counts':{'model_paths':13,'outer_fold_models':65,'oof_predictions':11115,'outer_folds':5,'inner_folds':2,'cars_internal_cv':2},
         'engineering_checks':unit,
         'synthetic_transfer':{'scope':'synthetic_engineering_fixture','real_herb_samples':0,'evaluations':21,'sample_predictions':synthetic['sample_predictions'],'report_directory':'synthetic-transfer-results','note':'Tests exercise calibration protocol/code; synthetic results do not establish real-herb calibration gains.'},
         'post_run_batch_declaration':post_run,
         'data_and_code_hashes':{'input_workbooks':source_hashes,'metadata_at_preparation_sha256':quality['metadata_sha256'],'full_prepared_npz_sha256':quality['dataset_sha256'],'candidate_npz_sha256':generation['output_sha256'],'frozen_execution_code_sha256':protocol['source_code_sha256'],'shared_split_sha256':next(iter(audit['all_job_split_sha256'].values()))},
         'quality':{'numeric_nonfinite_cells':sum(r['alpha']['nonfinite_cells']+r['n']['nonfinite_cells'] for r in quality['sources']),'full_alpha_negative_cells':negative_absorption,'candidate_alpha_negative_cells':negative_candidate,'negative_values_modified':False,'records_removed':0,'seven_numbered_workbooks_bytes_preserved':True},
         'validation':{'independent_metric_max_absolute_difference':audit['maximum_saved_metric_absolute_difference'],'all_cases_share_splits':True,'each_sample_tested_once_per_path':True,'physical_train_test_overlap':0,'ZY02_test_samples_per_fold':1,'SVC_predict_vs_argmax_disagreements':audit['total_svc_predict_argmax_disagreements']},
         'metric_table':'independent-audit/pooled_metrics.csv','class_recalls':'independent-audit/class_recalls.csv','all_positive_and_negative_comparisons':'independent-audit/comparison_deltas.csv',
         'descriptive_findings':{'candidate_MC20_vs_full_SVM':next(r for r in deltas if r['comparison']=='MC20_vs_full_SVM'),'candidate_stable_vs_MC20':next(r for r in deltas if r['comparison']=='stable_vs_MC20'),'candidate_concat_vs_alpha_SVM':next(r for r in deltas if r['comparison']=='concat_vs_alpha_SVM'),'full_range_alpha_PLSDA_accuracy':1.0,'full_range_interpretation':'Sensitivity result only; negative high-frequency absorption values and single-batch confounding prevent treating 100% as proof of physical validity, chemical causality or cross-batch origin identification.'},
         'plot_files':['figures/metric_comparison.png','figures/metric_comparison.pdf','figures/confusion_alpha_svm.png','figures/confusion_alpha_svm.pdf','figures/confusion_concat_svm.png','figures/confusion_concat_svm.pdf'],
         'inferential_significance_or_confidence_intervals_computed':False,
         'archive_policy':{'caches_included':False,'joblib_or_pickles_included':False,'derived_npz_inputs_included':False,'original_execution_text_rewritten':False,'absolute_paths_in_execution_helpers':'Retained as original execution evidence; use repository entrypoints for portable reruns.'}}
write(ARCHIVE/'stage_summary.json',summary)
labels={'alpha_primary/alpha_dummy':'候选区α／多数类','alpha_primary/alpha_svc':'候选区α／SVM','alpha_primary/alpha_plsda':'候选区α／PLS-DA','alpha_primary/alpha_mcuve20_svc':'候选区α／MC20＋SVM','alpha_primary/alpha_cars_svc':'候选区α／CARS-style＋SVM','modality_controls/n_svc':'候选区n／SVM','modality_controls/n_plsda':'候选区n／PLS-DA','modality_controls/concat_svc':'候选区n＋α／SVM','modality_controls/concat_plsda':'候选区n＋α／PLS-DA','stability_controls/alpha_mcuve100_svc':'候选区α／MC100＋SVM','stability_controls/alpha_stable_svc':'候选区α／重复MC＋SVM','full_export_sensitivity/alpha_svc':'全导出α／SVM（敏感性）','full_export_sensitivity/alpha_plsda':'全导出α／PLS-DA（敏感性）'}
lines=['# 获批酸枣仁实验执行证据（2026-10-31）','',
'**执行后用户补充确认：每个产地目前只有一个来源批次。以下结果是用户声明独立样品上的匿名产地组单批次内部识别，产地与来源批次不可分离；没有真实跨批次验证。** 原执行文本保留当时状态，最新解释以[stage_summary.json](stage_summary.json)的`post_run_batch_declaration`为准。','',
'本轮排除Gaussian。只使用[当前7份编号版](../../../data/CURRENT_DATASET.md)，共855份用户声明独立样品；真实地名、制样／测量日期及仪器映射仍未提供。','',
'## 实际执行范围','',
'- 13条真实分类路径、65个外层折模型、每路径855份OOF预测，共11,115份预测；所有方法共用5外折／2内折，CARS-style内部CV显式为2。','- 主比较范围固定为0.1—1.5 THz工程候选区（612点）；全导出范围（2578点）另列敏感性，不根据测试分数改称最优有效频段。','- ZY01=350、ZY02=5、ZY03—ZY07各100；ZY02每个外折测试仅1份，不能用复制或重复扫描增加独立样本量。','- 100项单元／集成工程检查通过；合成分类迁移另有21组评价、630份样品预测，真实药材样品数为0，不能作为真实校准收益。','',
'## 全部真实路径结果','',
'| 输入／方法 | 准确率（%） | 平衡准确率（%） | 宏F1 |','| --- | ---: | ---: | ---: |']
for row in metrics:lines.append(f"| {labels[row['case']]} | {100*float(row['accuracy']):.2f} | {100*float(row['balanced_accuracy']):.2f} | {float(row['macro_f1']):.4f} |")
lines += ['',
'候选区MC20相对全谱SVM数值改善；重复稳定筛选未超过MC20，n＋α拼接也未超过α单独输入。全部正负结果均保留，这些差值未作显著性或置信区间推断。全导出α＋PLS-DA的100%来自实际保存预测，但不证明跨批次鉴别、化学因果或频段物理有效性。','',
f'完整α数据中有{negative_absorption:,}个负值单元格，工程候选区负α单元格为{negative_candidate}；原值保留，没有按结果删除样品或修正数值。ZY02还包含2份1.0 mm与3份1.5 mm压片，其他组为1.5 mm，应结合制样、仪器和来源批次记录解释。','',
'## 图和可检查证据','',
'- [13路径指标比较PNG](figures/metric_comparison.png)／[PDF](figures/metric_comparison.pdf)：主宏F1、辅平衡准确率，坐标固定0—1，橙色仅表示全导出敏感性。','- [预定α-SVM混淆矩阵PNG](figures/confusion_alpha_svm.png)／[PDF](figures/confusion_alpha_svm.pdf)。','- [预定n＋α-SVM混淆矩阵PNG](figures/confusion_concat_svm.png)／[PDF](figures/confusion_concat_svm.pdf)。两矩阵按预定模型展示，单元格为原计数及行百分比，明确ZY02分母5。','- [独立复算](independent-audit/independent_recalculation.json)、[总体CSV](independent-audit/pooled_metrics.csv)、[逐类召回CSV](independent-audit/class_recalls.csv)、[全部差值](independent-audit/comparison_deltas.csv)。复算与保存指标最大绝对差为0。','- [SVC平票策略诊断](independent-audit/tie_policy_diagnostics.json)：8条SVC路径的`predict`与OVR分数argmax共31处不同；主结果保留冻结的`predict`，未据此改模型重跑。','- [数值／配对质控](data-preparation/quality_report.json)、[逐记录清单](data-preparation/record_manifest.csv)、[候选区生成记录](datasets/generation.json)。','- [冻结协议](experiments/frozen_protocol.json)、[实际运行命令及耗时](experiments/runs_manifest.json)、源码快照`experiments/execution_code/`及原始日志`experiments/logs/`。','- [100项检查摘要](tests_summary.json)和[完整日志](all-tests.log)。','- `synthetic-transfer-results/`与[输入范围说明](synthetic-transfer-inputs/input_scope.json)仅为合成工程验证。原`status.json`中的protocol scope不表示使用了真实药材。','',
'## 归档与复现边界','',
'本目录没有归档`.cache`、joblib／pickle模型缓存或派生NPZ输入；约2.2 GB缓存留在被Git忽略的outputs中。派生输入hash、原始7表hash及元数据准备时hash均记录，原始7份编号版字节未改。`execution_helpers/`保留实际执行辅助脚本，其中绝对路径是原运行证据；可移植入口仍以仓库scripts中的正式命令为准。','',
'`independent-audit/outer_fold_selection_frequency.csv`是5个重叠外层训练集合之间的选点频率，不能冒充重复筛选内部10次重采样频率、p值、FDR控制或独立批次稳定性。','',
'本地冻结协议不属于外部预注册。后续需新增各产地独立批次及可核验采样／制样记录，才能检验跨批次泛化与小样本校准。','']
(ARCHIVE/'README.md').write_text('\n'.join(lines),encoding='utf-8')
assert not any(p.name in {'.cache','joblib','__pycache__'} or p.suffix.lower() in {'.npz','.joblib','.pkl','.pyc'} for p in ARCHIVE.rglob('*'))
files=[p for p in ARCHIVE.rglob('*') if p.is_file() and p.name!='files_sha256.json']
manifest={'scope':'All archival files except this self-referential hash index','file_count':len(files),'files':[{'path':str(p.relative_to(ARCHIVE)),'size_bytes':p.stat().st_size,'sha256':SHA(p)} for p in sorted(files)]}
write(ARCHIVE/'files_sha256.json',manifest)
assert all(SHA(ARCHIVE/item['path'])==item['sha256'] for item in manifest['files'])
print(json.dumps({'archive':str(ARCHIVE),'files_hashed':len(files),'total_bytes':sum(p.stat().st_size for p in ARCHIVE.rglob('*') if p.is_file()),'figures':6,'model_paths':13,'outer_models':65,'oof_predictions':11115,'tests':100,'synthetic_evaluations':21,'negative_alpha_cells_preserved':negative_absorption},ensure_ascii=False))
