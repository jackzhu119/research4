# Journal and claims audit — 2026-11-16

Read-only audit of manuscript source and corrected evidence boundary. No paper source, raw spectra, scientific models or declarations changed. Read repository AGENTS.md, paper/current/README.md and manifest, current manuscript_content.py, supplied revision instructions and doubao review, archived journal instructions, JOURNAL_COMPLIANCE.json, and user acquisition/independence declarations.

## Decisions

1. **The ≤200-word abstract limit is not established as an Applied Spectroscopy rule.** The archived official journal instructions (authoring_sources/Applied_Spectroscopy_Guidelines_Archived.txt, line 145) require an unstructured abstract stating purpose, major findings and conclusions, but give no abstract word ceiling. Keep ≤200 as the requested editorial constraint, not as an attributed journal mandate. Current compliance JSON reports 197 words but does not itself prove a journal maximum. A fresh urllib HTTP GET to https://journals.sagepub.com/author-instructions/ASP returned HTTP 403 on this audit; no live policy confirmation is claimed. The archived requirements remain the checkable basis. Preserve this distinction in regenerated JOURNAL_COMPLIANCE.json.

2. **The requested stability-based title is scientifically defensible with an explicit distinction.** The implemented ranking uses class-wise Monte Carlo coefficient consistency, so “Stability-based multiclass spectral selection for complementary terahertz discrimination of medicinal materials” accurately identifies the ranking principle. Nogueira Φ = 0.8360 is a descriptive saved-refit outcome, not the optimization objective or proof of chemical specificity. Do not imply Φ drives feature selection or that high Φ establishes universally superior prediction. Use the requested Chinese title, or more precisely “面向中药材太赫兹辅助判别的基于系数稳定性的多类别光谱特征选择”. This is an emphasis change, not a new algorithmic result.

3. **Reject the proposed mechanistic kernel-degradation claim.** The 10.78 pp comparison establishes an evaluated pipeline benefit over its same-partition full-spectrum RBF-SVM reference. Feature selection also changes dimensionality, training-dependent numerical gamma, budget search and representation; these data do not isolate redundant coordinates as the cause of kernel-distance degradation. “Removing degradation observed in the nonlinear RBF-SVM kernel” turns a performance contrast into an unmeasured mechanism. A coordinate can be traced to its spectral frequency without being chemically assigned. Dense LDA versus selected-input SVM is a valid practical contrast, but does not establish easier deployment, faster runtime or superiority.

4. **Avoid “statistically comparable”.** An interval including zero does not establish equivalence or non-inferiority, especially for these conditional saved-fit bootstrap intervals. The point difference is about 1.67 pp, with MC20-minus-LDA interval −1.11 to 4.44 pp. State that no clear accuracy advantage over LDA was established under this comparison. Retain LDA as a strong comparator and do not treat SVM benefit as cross-classifier benefit.

5. **“Separately procured ZY07” exceeds located provenance.** The 2026-10-31 declaration says every numbered record is an independent specimen and each workbook represents one source group; one lot per group was user-declared, without actual lot identifiers or sampling records. The 2026-11-12 declaration supports online purchase generally, but has no suppliers, orders or purchase dates, and no ZY07-specific separate procurement statement. Separate workbook records and 100 specimens per Suanzaoren group are supported; independently purchased lots, independently verified origins and separate procurement occasions are not established. The supplied edit's “separately procured and archived” is a request for wording, not an explicit new documentary confirmation.

6. **Include the fixed-PRED positive and negative contrasts, but weaken the feature-budget mechanism.** The reported fixed MC20/R0 and full-SVM contrasts can describe task-dependent performance in the existing comparisons. They do not show that narrow within-material tasks generally benefit from aggressive pruning or that four-material tasks causally require broader spectral coverage. Task, selected features, fitting choices and budget vary together; R0 reselection is another pipeline and must remain separately named. Fixed PRED is not a new confirmation cohort and is not comparable numerically to nested CV or LOWO.

## Suggested safe EN/ZH amendments

### Main SVM/LDA paragraph

EN: “Full-spectrum shrinkage LDA remained a strong comparator at 91.78%; the MC20-minus-LDA paired interval of −1.11 to 4.44 percentage points included zero, so this comparison did not establish an accuracy advantage over LDA. MC20–SVM achieved 93.44% with an average of 220.6 selected, traceable spectral coordinates, compared with 82.67% for the matched 612-coordinate full-spectrum SVM. This supports the evaluated selection workflow for SVM while retaining a smaller, explicit predictor set; it does not identify the mechanism of the full-spectrum performance gap or establish a universal classifier ranking.” Retain existing raw/MC100 results and the ranking-stage-only draw-count qualification.

ZH：“全谱收缩LDA仍是较强对照（91.78%）；MC20减LDA的配对区间为−1.11至4.44个百分点，包含零，因此本比较未确立相对LDA的准确率优势。MC20–SVM采用平均220.6个可追溯光谱坐标时达到93.44%，同划分的612坐标全谱SVM为82.67%。这些结果支持该SVM特征选择流程及其明确缩减的预测变量集合，但未识别全谱性能差距的具体机制，也未建立普遍的分类器排名。”

### Source labels

EN: “The five Suanzaoren source groups each contain 100 user-declared independent specimens recorded in separate workbooks: Inner Mongolia (ZY03), Lingshi in Shanxi (ZY04), Bozhou in Anhui (ZY05), Shijiazhuang in Hebei (ZY06), and a separately recorded Shanxi group (ZY07). The two Shanxi workbooks are retained as distinct recorded source groups; source discrimination refers to these five recorded labels.”

ZH：“五个酸枣仁来源组各含100份由研究者声明为独立实物的样品，并分别记录于独立工作簿：内蒙古（ZY03）、山西灵石（ZY04）、安徽亳州（ZY05）、河北石家庄（ZY06）及另一单独记录的山西组（ZY07）。两份山西工作簿保留为不同的记录来源组；来源判别针对上述五个记录标签。” Existing provenance caveats should remain.

### Fixed-PRED table interpretation

Retain the supplied numerical sentences after table verification. Replace the causal final sentence with:

EN: “These fixed-role comparisons show that the evaluated selector's accuracy benefit did not carry uniformly across tasks. They motivate task-specific validation of feature budgets, without establishing a general causal relationship between task breadth and required spectral coverage.”

ZH：“这些固定角色比较表明，所评价选择器的准确率收益并未一致延续至不同任务。这提示应按任务验证特征预算，但尚不足以确立任务范围与所需光谱覆盖之间的普遍因果关系。”

### Contribution and protocol linkage

EN: “The contribution is a training-contained multiclass ranking-and-selection workflow for complementary medicinal-material discrimination, with a measured improvement against the matched SVM reference and explicit spectral-coordinate reduction. Its ranking interface is reusable with other estimators, whose accuracy gains require direct evaluation. Separate workbook-held-out and support-label analyses define grouping limits and identify representation, candidate-pool and supervision effects within the existing collection.”

ZH：“本研究的贡献是服务于中药材辅助判别、限定于训练过程的多类别排名与特征选择流程，实测改善同划分SVM参照的表现，并明确缩减光谱坐标。排名接口可供其他估计器复用，其准确率收益仍需直接评价。独立报告的工作簿留出与支持标签分析界定分组验证边界，并揭示现有样本集合中的表征、候选池和监督效应。”

Use “separate protocol” rather than “independent protocol” in Chinese/English if “independent” could suggest independently acquired specimens. “Decision-function updating” is better supported than a physical or general causal mechanism.

## Guardrails for implementation

- Preserve baseline author names/order, emails, ORCIDs, institutions, author contributions, acknowledgements including AI disclosure, declarations and Data provenance verbatim. Do not rewrite locked provenance to incorporate the above wording; place supported scope language in editable scientific narrative only.
- No scientific training, PLS ranking reruns, quantum work or raw-data changes. Derivations from allowlisted saved outputs are permissible only within delegated audit scope.
- Do not import doubao's bg/330-repeat, permutation, random/uniform or PLS-DA numbers unless independently authorized by the corrected manifest and protocol audit. The requested text polish does not authorize restoring excluded gamma-cache results. “At chance under permutation” would not itself rule out every leakage mode.
- Keep positive and negative results, protocol populations and denominators explicit. Do not add B-control gains to the primary accuracy or describe label-control changes as selection gains.
