# Supplementary information

Tracing absorption information in terahertz discrimination of medicinal materials

Ziyuan Zhu, Jiayi Ren, Shuyu Zhou, Mengting Zhu, Yin Shen*

College of Artificial Intelligence Medicine, Chongqing Medical University, Chongqing, China

*Corresponding author: Yin Shen; shenyin285819016@126.com

ORCID: https://orcid.org/0000-0002-3221-7875

This supplement provides the complete computational specification, specimen allocation, representation controls, fixed role classification and adaptation results, mechanism experiments, and qualified molecular reference calculations. Numerical classifier results use the corrected material categories and the current paper evidence manifest. The machine readable tables preserve all included positive and negative settings.

## S1 Materials acquisition and specimen allocation

The measured dataset was provided by the corresponding author, Yin Shen, for the analyses reported here. It forms part of a shared resource used in other supervised studies. The collection contains 1672 investigator declared independent specimens represented in 18 workbooks: 364 Zaoren, 500 Suanzaoren, 403 Huangqi and 405 Tianma. Material names follow the Chinese commodity labels recorded in the dataset. The declarations for the original 855 and the additional 817 specimens are retained separately. Each workbook contains time domain, frequency domain, refractive index and absorption coefficient views of the same specimens. Reference columns and copies under historical filenames are not additional specimens.

All four materials were measured at the terahertz laboratory of the National Engineering Research Center for Information Technology in Agriculture using a Menlo Systems TERA K15 THz time domain spectrometer. The optical wavelength was 1550 nm and the optical power was 33 mW. The measurement temperature was 292 K. Nitrogen was supplied to the enclosure to keep humidity below 4%. A polyethylene blank pellet provided the reference response. Polyethylene reference response correction was performed before the exported optical parameters were calculated.

Preparation equipment comprised an XQM-0.4L planetary ball mill, a DHG101-0B drying oven, an FA2104N electronic balance and a Specac Atlas TM 15T manual press. Acquisition conditions and equipment were confirmed in the investigator declaration of 12 November 2026. Original time domain column headers record nominal thicknesses of 1.5 mm for 1670 specimens and 1 mm for two Yunnan Zaoren specimens, ZY02 001 and ZY02 002. These annotations are retained as recorded. The present analysis uses the supplied absorption and refractive index arrays, together with their workbook coordinates, specimen labels and frozen role assignments.

Table S1. Source groups and frozen specimen roles.

|Group|Material|Declared source|Total|TRAIN|CAL|PRED|
|---|---|---|---|---|---|---|
|ZY01|Zaoren|Yan'an Shaanxi|350|210|70|70|
|ZY02|Zaoren|Yunnan|5|3|1|1|
|ZY03|Suanzaoren|Inner Mongolia|100|60|20|20|
|ZY04|Suanzaoren|Lingshi Shanxi|100|60|20|20|
|ZY05|Suanzaoren|Bozhou Anhui|100|60|20|20|
|ZY06|Suanzaoren|Shijiazhuang Hebei|100|60|20|20|
|ZY07|Suanzaoren|Shanxi|100|60|20|20|
|HQ01|Huangqi|Qujing|100|60|20|20|
|HQ02|Huangqi|Guyang Inner Mongolia|100|60|20|20|
|HQ03|Huangqi|Gansu|100|60|20|20|
|HQ04|Huangqi|Datong Shanxi|103|61|20|22|
|ZR01|Zaoren|Yulin Shaanxi|3|1|1|1|
|ZR02|Zaoren|Liaoning|3|1|1|1|
|ZR03|Zaoren|Inner Mongolia|3|1|1|1|
|TM01|Tianma|Leishan Guizhou|100|60|20|20|
|TM02|Tianma|Changbai Mountain Jilin|102|61|20|21|
|TM03|Tianma|Anhui|103|61|20|22|
|TM04|Tianma|Yunnan|100|60|20|20|
|Total|||1672|999|334|339|

The TM03 workbook retains the uploaded spelling 天麻安徽毫州.xlsx. The Shanxi and Lingshi source groups retain their separate recorded labels; administrative nonoverlap is not inferred from the labels. A workbook is not automatically an independently verified procurement batch. Each original ZY source group had one investigator declared source batch. No additional actual batch identifiers are inferred.

The specimen roles of all original 855 records were inherited without reassignment. Additional records were sorted by record identifier within group and assigned using independent NumPy PCG64 streams. A stream seed was obtained from the first eight bytes of SHA 256 applied to the string containing split seed 20261101 and the group identifier. The allocation used max(1, floor(0.60N)) TRAIN records, max(1, floor(0.20N)) CAL records and the remainder as PRED, reserving at least one PRED specimen. Thus groups of three have 1/1/1 roles, groups of 102 have 61/20/21 roles and groups of 103 have 61/20/22 roles. Assignment did not use spectra or prediction scores. Exact record identifiers, original column headers, labels and roles are provided in record_role_assignment.csv and the frozen split_manifest.csv.

## S2 Spectral representation and optical parameters

Absorption coefficient and refractive index exports are denoted α and n. The common analysis retains the 612 original shared coordinates satisfying 0.1 ≤ f ≤ 1.5 on the workbook THz axis. The actual endpoints are 0.100708011000165 and 1.49917607284337. No interpolation is applied to these primary representations. This interval is the model analysis interval, not the claimed hardware bandwidth. Descriptive means and dispersion plots use measured TRAIN spectra. Absorption values are displayed on the original workbook export scale.

For the optical slab model, T denotes the complex transmission response after polyethylene reference response correction to an equivalent air reference. It must be distinguished from the uncorrected sample to polyethylene signal ratio. Under the homogeneous parallel slab approximation, near normal incidence, and with multiple reflections neglected or otherwise treated, the transfer response contains the Fresnel amplitude 4n/(n+1)², the attenuation exp(−αd/2) and phase exp[−i(n−1)ωd/c]. The unwrapped reference minus sample phase delay Φ gives n = 1 + cΦ/(ωd). Absorption is α = −(2/d) ln{|T|(n+1)²/(4n)}. Here ω = 2πf and d is specimen thickness. SI evaluation requires f in Hz, d in m and c = 299792458 m s⁻¹, giving α in m⁻¹ for the slab equations. The present analysis retained the existing workbook arrays on their recorded numerical scales without recalculating the optical parameters.

Standard normal variate transformation subtracts each row's mean and divides by its sample standard deviation with ddof = 1. For joint representations it is applied to each signal block separately. Feature wise StandardScaler parameters are fitted on the current fitting subset. Validation and query spectra receive those fitted parameters. The candidate representations are raw α, SNV α, raw n, SNV n, joint SNV n and SNV α, and joint raw n and SNV α. Single signal candidate matrices contain 612 variables; joint matrices contain 1224, with n at global indices 0–611 and α at 612–1223.

## S3 Variable ranking classifiers and nested selection

The implemented Monte Carlo selector is a PLS coefficient stability adaptation related to MC-UVE. It uses an explicit retained variable count rather than an artificial noise threshold. In each draw, nominally 80% of specimens are sampled without replacement separately within each class, rounding the class count upward. An outer fitting set has 48 specimens per class and therefore draws 39 per class, or 195 of 240. An inner fitting set has 24 per class and draws 20 per class, or 100 of 120.

Two component one versus rest PLS fits provide class specific coefficients. Predictors and response are explicitly standardized within each Monte Carlo fitting subset, followed by PLSRegression with scale = False. For each variable and class, stability is the absolute coefficient mean divided by its between draw sample standard deviation. The denominator floor is max(1, max(sd)) × 10⁻¹². Importance is the Euclidean norm of the class specific stability values. The first K variables in the resulting ranking are retained. MC20 uses 20 draws and seed 42. MC100 uses 100 draws. The rank aggregation control averages normalized Borda ranks from five MC20 streams with seeds 42 + 1009i for i = 0,…,4 and resolves ties by mean importance and then index.

SVM comparisons use class balanced radial basis function classifiers. In the matched V4 study the candidate C values are 0.1, 1, 10 and 100; gamma is the training scale rule, 0.1 times that value or 10 times that value. Eligible total feature budgets are 31, 62, 123, 184 and 245. For a global joint selector, K is the total retained across both blocks. The block allocation control ranks each block separately and retains floor(K/2) n variables plus the remaining α variables. It consequently uses two MC20 ranking procedures. Weighted concatenation considers n block weights of 1, 0.5 and 0.25 after training fitted scaling and selection, with no further normalization that would cancel the weight.

Full absorption reference models are shrinkage LDA and an MLP. LDA uses solver = lsqr, shrinkage = auto and equal class priors. The MLP considers 32 or 64 hidden units, alpha = 0.01, batch size 32, maximum 300 iterations, initial learning rate 0.001, random state 42, early stopping on 20% of the current fitting subset and n_iter_no_change = 15. Raw versus SNV preprocessing is selected within the inner folds for these full absorption references. All full absorption references retain 612 predictors.

The principal matched analysis uses the 300 Suanzaoren TRAIN specimens. Three stratified partition repeats use seeds 20261031, 20261108 and 42, each with five outer folds and two inner folds. Each outer Source contains 240 specimens, or 48 per source group; each Query contains 60, or 12 per source group. Inner mean macro F1 selects eligible K and settings, with balanced accuracy, accuracy and deterministic simplicity rules resolving ties. All fitted transformations, variable selection and hyperparameter selection occur within the relevant fitting subset. Outer predictions are pooled over 300 specimens within each repeat, and reported estimates average the three pooled metrics. Repeated predictions do not increase the 300 independent specimen denominator.

Table S2. Complete internally selected primary comparison.

|Representation and method|Mean retained K|Accuracy %|Macro F1|
|---|---|---|---|
|SNV α MC20 SVM|220.6|93.44|0.9344|
|SNV α MC100 SVM|216.5|93.44|0.9345|
|Raw α MC20 SVM|204.3|93.22|0.9325|
|SNV α five MC20 streams SVM|216.5|92.78|0.9280|
|Full α shrinkage LDA|612.0|91.78|0.9176|
|Joint SNV n and SNV α MC20 SVM|216.5|91.67|0.9169|
|Weighted raw n and SNV α MC20 SVM|232.8|89.22|0.8923|
|Equal block raw n and SNV α SVM|236.9|88.78|0.8875|
|Joint raw n and SNV α MC20 SVM|228.7|87.11|0.8711|
|Raw n MC20 SVM|149.3|83.67|0.8348|
|Full α MLP|612.0|73.22|0.7286|
|SNV n MC20 SVM|224.7|62.56|0.6220|

Balanced accuracy equals accuracy in the balanced pooled source task. The primary SNV α MC20 selector retains 184 variables in six outer refits and 245 in nine, giving a mean of 220.6 and a 63.95% reduction relative to 612. Compression for a fixed K of 62 must not be attributed to the primary inner selected estimate. Equal predictor width does not equalize initial candidate width or computation across selector families. Saved candidate stage timings do not measure complete end to end cost.

## S4 Feature budget sensitivity and predictor provenance

Table S3. Matched fixed budget sensitivity.

|Total K|SNV α accuracy %|Joint SNV accuracy %|Joint minus α percentage points|Retained n occurrences in joint fits|
|---|---|---|---|---|
|31|79.6667|83.7778|4.1111|0/465|
|62|81.0000|87.1111|6.1111|0/930|
|123|89.6667|91.0000|1.3333|0/1845|
|184|92.6667|91.0000|−1.6667|14/2760|
|245|93.7778|92.4444|−1.3333|119/3675|

All five budgets were specified in the original sensitivity protocol; classifier settings remain inner selected. No fixed budget result replaces the primary inner selected K result. At K = 62, both pipelines retain absorption predictors exclusively. Their mean intersection is 1.3333 of 62 coordinates per outer fit, ranging from zero to five, with mean Jaccard overlap 0.011112. C differs in two of the 15 paired refits and the gamma specification differs in eight. Ranking, the selected absorption subset and classifier settings can therefore change together. The findings identify the final signal path without assigning its predictive difference to a single isolated cause.

The n occurrence proportions at K = 184 and 245 are 0.5072% and 3.2381%. Denominators count selected variable occurrences across 15 refits, not distinct frequencies or additional specimens. The modality audit refers to final outer refits. Unrecorded frequency coordinates are not assigned artificial zero selection rates in exported selection frequency plots.

A separate reanalysis of the source-matched saved V4 predictions used 5000 specimen-cluster bootstrap draws with seed 20261109, retaining all three paired predictions for each sampled identifier and the same sampled indices across all five budgets. The joint-SNV minus SNV-absorption difference intervals in percentage points are 0.11 to 8.33 at K31, 2.33 to 10.00 at K62, −1.00 to 3.78 at K123, −3.78 to 0.44 at K184 and −3.11 to 0.44 at K245. These are exploratory percentile intervals without multiplicity adjustment. Raw absorption at K62 achieves 88.7778%, so the joint-pool advantage concerns the matched SNV comparison. The 62 saved V4 source files used for this replay were byte-matched to the current allowlisted source commit.

The archived primary comparison includes a 500 draw physical specimen cluster bootstrap with seed 20261108. Each sampled identifier retains all three paired repeat predictions. For example, the LDA minus SNV α MC20 accuracy interval is −4.33 to 1.33 percentage points. These conditional exploratory intervals characterize the archived fits and specimen population. They are separate from any later fixed budget bootstrap calculation and do not incorporate independently sampled batch variation or a complete model refitting bootstrap.

## S5 Fixed role tasks calibration and complete refitting

Table S4. Task definitions and denominators.

|Task|Classes|TRAIN|CAL|PRED|Correct PRED with common fixed SVM|
|---|---|---|---|---|---|
|Zaoren versus Suanzaoren|2|516|174|174|174/174|
|Four materials|4|999|334|339|331/339|
|Suanzaoren source groups|5|300|100|100|85/100|
|Zaoren source groups|5|216|74|74|72/74|
|Huangqi source groups|4|241|80|82|69/82|
|Tianma source groups|4|242|80|83|81/83|

The common fixed baseline uses all 612 absorption coordinates, row SNV, TRAIN fitted StandardScaler and a class balanced RBF SVM with C = 10, gamma = scale and seed 42. It performs no variable selection or task specific hyperparameter search and excludes CAL. Other eligible source configurations in the binary and Suanzaoren tasks use TRAIN only twofold selection; the remaining source tasks use prespecified fixed parameters where tiny groups preclude stratified tuning. Task populations overlap and their scores are not consecutive improvements on one benchmark.

The four material result has macro F1 0.9751 and balanced accuracy 97.50%. Zaoren source accuracy is 97.30%, macro F1 is 0.5972 and balanced accuracy is 60.00%. The largest source contributes 70 of its 74 PRED specimens, and a majority reference obtains 70/74 correct. The three additional Zaoren groups each have one specimen in each frozen role. Full source class recalls and confusion counts are retained in the attached tables.

Frozen score adaptation and complete CAL refitting answer separate questions. Two prespecified score adapters use regularization 1, or regularization 1000 with blend 0.05. Support budgets are one, three, five and ten specimens per class with 20 support draws where feasible. Source SVC.predict voting is compared with its own source control. Score argmax adaptation uses a separate zero support argmax control with fixed score column order; changes between SVC voting and score argmax are not treated as learned adaptation. Complete refitting uses locked source hyperparameters and fits every transformation, selector and classifier anew on TRAIN plus CAL.

For the corrected Suanzaoren PRED task, source only full spectrum, fixed MC20, source reselected configuration 0 and source reselected configuration 2 obtain 85%, 87%, 94% and 87%, respectively. Complete CAL refitting gives 84%, 82%, 96% and 93%. Frozen configuration 2 with regularization 1 gives mean accuracies of 75.90%, 81.50%, 84.05% and 86.30% over the four support budgets; the source only accuracy is 87%. The strongly regularized blend retains 87% at all four budgets. The evidence therefore preserves the unadapted source output as a defined operational option. Additional support draws are repeated evaluations, not newly acquired physical specimens.

## S6 Neural auxiliary material comparisons

The corrected fixed role neural comparison uses 300 Suanzaoren TRAIN, 100 CAL and 100 PRED specimens. The target representation is 128 linearly interpolated absorption coordinates from 0.21 to 1.49 THz without extrapolation, followed by row SNV and a task specific TRAIN fitted StandardScaler. The model has width 12, two residual blocks, dropout 0.35, a 16 dimensional embedding and a fresh five output target head. Target training uses Adam, learning rate 0.0005, weight decay 0.001, batch size 64 and 96 epochs. Pretraining uses batch size 32 and 120 steps. Batch normalization statistics are reset before target training. No PRED statistics or prediction dependent epoch choice is used.

Arms are target training alone, 120 extra target pretraining steps followed by a fresh target head, and alternating Huangqi and Tianma pretraining with separate auxiliary heads followed by a fresh target head. Seeds are 42, 43 and 44. Mean fixed PRED accuracies are 81.00%, 80.33% and 79.00%, respectively. The separate calibration settings and support draws are preserved in fixed_transfer.csv and fixed_transfer_paired.csv.

The matched TRAIN only matrix additionally controls representation and update budget. With 128 absorption coordinates, 240 source specimens, 192 target epochs and 480 pretraining steps, mean accuracies are 75.44% for target only, 77.11% for extra target training, 76.00% with Huangqi, 71.11% with Tianma and 79.78% with both auxiliary materials. With 512 coordinates and the same source count and epoch setting, corresponding estimates are 67.22%, 67.22%, 67.33%, 59.11% and 63.44%. All settings are retained. Effects depend on representation and training budget. A different number of spectral coordinates, training samples or update steps is not an architecture only comparison against the classical pipelines.

## S7 Internal mechanism experiments

The five mechanism rounds contain 1380 distinct task bundles: B1 105, B2 345, B3 375, B4 375 and B5 180. They use corrected material labels and roles formed inside master TRAIN. Source, Support candidate pool and Query are mutually exclusive within a task. The standard binary layout is 240/20/40, with Source Zaoren/Suanzaoren counts 180/60, 120/120 or 60/180 and Query 20 per class. The five source Suanzaoren layout is 190/50/60, or 38/10/12 per class. These Support and Query roles are distinct from master CAL and PRED.

Usual support K values of 1, 3, 5 and 10 refer to specimens per class: binary totals 2/6/10/20 and five class totals 5/15/25/50. Fixed total support acquisition or composition interventions use their stated total budgets instead. The 240 endpoint in B2 F04 already contains Source 190 and Support 50 and is not counted as 240 plus 50.

B1 tests source coverage at a fixed sample count, source class proportions, frozen source adaptation and the balanced five source task. B2 varies adapter flexibility and score units, separates full refitting contributions from scaler and gamma changes, tests class weighting against source coverage, and follows five source label budgets. B3 examines loss normalization, intercept penalties, label versus unlabeled statistics, controlled support label corruption, and the effective C constraints associated with class weighting. B4 controls total loss mass and added supervision, tests C by class weight interactions, checks analytic decision activation under weak calibration, and varies class composition at a fixed support count. B5 evaluates Source internal strategy selection, label blind support acquisition, output preserving OVO calibration and additional effective sample size shrinkage.

In B5 I01, binary internal Source/Support/validation roles are 180/20/40 and five class roles are 100/50/40. Selection remains inside the outer Source; outer Support and Query do not select the strategy. Each mechanism round supplies metrics, paired transitions, mechanistic descriptors and uncertainty tables. Full family settings and estimates are retained in the accompanying B1 through B5 evidence files. Quantitative mechanism findings are supplied in the following complete evidence tables rather than combining incompatible tasks into a single improvement ranking.

Table S5. B1 five source support learning curves in percent accuracy.

|Method|K 1|K 3|K 5|K 10|
|---|---|---|---|---|
|Source without adaptation|80.0000|80.0000|80.0000|80.0000|
|Anchor regularization 1|69.8889|75.1111|78.2222|78.7778|
|Bias only|77.3333|78.5556|78.8889|79.7778|
|Regularization 1000 blend 0.05|80.0000|80.0000|80.0000|80.1111|
|Complete pipeline refitting|80.0000|80.6667|82.1111|83.8889|

At K = 10 the full refit corrects 49 and changes 14 correct outcomes to errors among 900 related repeat predictions, a net 35/900 or 3.8889 percentage points. The 900 outcomes arise from 15 contexts of 60 Query predictions and do not represent 900 independent specimens. At K = 1 the strong anchor corrects 22 and damages 113 outcomes. Binary tasks average 98.4444% for the source, 98.3333% for anchor or bias adaptation and 98.4444% for weak adaptation. All 1800 recorded binary affine threshold checks have positive slopes and pass their analytic equivalence check. For a score difference d transformed as ad + b with a > 0, rank order is preserved and classification changes correspond to the threshold −b/a. Such calibration does not learn a new spectral representation.

Table S6. Controlled five source results at K = 10.

|Round and intervention|Accuracy %|Difference from source in percentage points|Descriptive 95% interval for difference|
|---|---|---|---|
|B2 true labels refit with frozen scaler and gamma|83.7778|3.7778|2.1111 to 5.6694|
|B2 frozen scaler updated gamma|83.7778|3.7778|2.1111 to 5.7778|
|B2 updated scaler frozen gamma|83.8889|3.8889|2.2194 to 6.0000|
|B2 updated scaler and gamma|83.8889|3.8889|2.2194 to 6.0000|
|B3 true label complete refit|83.8889|3.8889|2.2194 to 6.0000|
|B3 unlabeled support statistics update|80.1111|0.1111|−0.2222 to 0.4444|
|B3 cyclically incorrect support labels|73.3333|−6.6667|−9.2222 to −4.1083|
|B5 Source internal selected strategy|83.6667|3.6667|2.0000 to 5.5556|

Intervals cluster repeated outcomes by physical specimen identifier and describe the current collection and archived fits. They do not represent new batch uncertainty or account for all historical development and selection. Updated scaling contributes 0.1111 percentage points relative to frozen scaler and gamma, with interval −0.2222 to 0.5556. True labels contribute 3.7778 points relative to the unlabeled statistics control, with interval 2.1111 to 5.8889. At K = 10, B5 Source internal selection differs from the fixed frozen scaler refit by −0.1111 points, with interval −0.5556 to 0.2222. At K = 1, that selected strategy obtains 79.2222% compared with source 80.0000%, with difference interval −1.7778 to 0.4444.

The B2 F04 label budget curve uses 50, 100, 150, 190 and 240 fitting specimens and obtains 68.7778%, 74.4444%, 78.5556%, 80.0000% and 83.8889%, respectively. In B2 F01, K = 1 anchor accuracy is 71.3535% with the original scores, 74.8990% after score multiplication by 0.1, 58.4040% after multiplication by 10, and 74.7172% after Source RMS normalization. At K = 10, diagonal and Source RMS adapters obtain 77.4444% and 79.2222%, respectively. Randomized support labels give 24.7071% in that diagnostic. K = 1 F01 has 11 support draws per context, whereas K = 10 has one; the paired and clustered calculations preserve that endpoint structure.

B3 G01 compares sum and mean fitting losses and intercept penalties. K = 10 penalized sum loss gives 78.7778%; mean loss gives 80.0000%. Removing the intercept penalty gives 78.8889% for sum loss and 80.0000% for mean loss. B3 G03 introduces 0%, 10%, 20%, 40% and 100% cyclic corruption of support labels. Corresponding complete refit accuracies are 83.8889%, 82.6444%, 81.9333%, 80.0889% and 73.3333%; strong anchor accuracies are 78.7778%, 78.8889%, 77.5111%, 66.2667% and 6.4444%. Weak adaptation remains close to 80%, reflecting small decision changes. The binary class weight control finds identical predictions, scores, supports and dual coefficients for balanced versus unweighted fits in 240 contexts at the examined C and margins, with no support vectors near the upper bound. This is a condition specific constraint activation result.

B4 H01 separates additional labels from total loss mass. Increasing source effective loss mass alone retains 80.0000%, whereas adding support and refitting at fixed C gives 83.7778%. Normalizing the refit C to preserve the original total C × N also gives 83.7778%: Source N = 190, C = 10 yields total 1900; refit N = 240 uses C = 7.9166667 for the same total, compared with 2400 under fixed C = 10. Thus the observed gain is retained when this loss mass factor is controlled.

B4 H03 checks weak adapter decision motion. In binary F01, all scores change in 13200 related predictions at each of K = 1, 3 and 5 and in 1200 at K = 10, while predicted classes remain unchanged. All have strict margin certificates with no violations. In the five source K = 10 task, all 900 score vectors change, two predicted classes change, and 893 have strict margin certificates, producing 80.1111% versus source 80.0000%. In B4 H04, a balanced support composition gives anchor 77.6667%. Source deficient unweighted compositions give 65.8889%–70.1481%, and class balanced residual weighting gives 72.1111%–76.0370%. Output preservation and effective adaptation are therefore reported separately.

Table S7. B5 label blind support acquisition with complete refitting.

|Total acquired labels|Uncertainty acquisition accuracy %|Range across three random acquisition seeds %|Diversity acquisition accuracy %|
|---|---|---|---|
|5|81.6667|80.0000 to 80.2222|80.3333|
|15|82.6667|80.5556 to 81.5556|80.7778|
|25|83.1111|81.8889 to 82.1111|81.8889|
|50|83.7778|83.7778|83.7778|

At total budget five, uncertainty acquisition exceeds random seeds 601, 602 and 603 by 1.4444, 1.6667 and 1.4444 points, with respective descriptive intervals 0.2222–2.8889, 0.5556–2.8917 and 0.4444–2.4444. These acquisition budgets do not guarantee equal labels per class. With a guarded mean anchor instead of complete refitting, uncertainty acquisition at total budget 15 gives 78.1111%. Acquisition effects therefore depend on the subsequent learner. B5 I03 gives K = 10 estimates of 80.4444% for OVO trace normalized calibration, 80.3333% for OVO PCA trace calibration, and 80.0000% for OVR trace or raw mean anchor calibration. B5 I04 effective sample size shrinkage gives approximately 78.56%–79.89% in source deficient compositions compared with approximately 72.11%–76.04% without that shrinkage. The full settings, seed specific results and support composition records remain in the machine readable tables.

## S8 Molecular reference calculations

Protocatechuic acid provides a qualified isolated molecule reference calculated at PBE/def2-SVP using PySCF 2.13.0. The neutral closed shell model uses grid level 5 and no D3 correction. Two independently accepted numerical implementations are retained: direct Coulomb evaluation and RI-J with the def2-universal-jkfit auxiliary basis and the recorded grid and auxiliary basis responses. Each has passed its stationary point, full mode and independently evaluated three step IR checks.

The 17 atom C7H6O4 model has 3N − 6 = 45 vibrational modes. Grid level 5 uses the recorded PySCF default NWChem pruning and grid_response = True. The RI-J model additionally uses auxbasis_response = True. Direct Hessians use displacement magnitudes of 0.001 and 0.0005 Bohr; RI-J uses 0.002 and 0.001 Bohr. Each model has two complete sets of 102 signed Cartesian displacements, for 204 Hessian displacement calculations. The independently evaluated Cartesian IR step series is 0.001, 0.0005 and 0.00025 Bohr for direct evaluation, and 0.002, 0.001 and 0.0005 Bohr for RI-J. Normal coordinate displacement series are 0.002, 0.001 and 0.0005 Bohr√amu for direct evaluation and 0.004, 0.002 and 0.001 Bohr√amu for RI-J. Each independent mode derivative check uses 45 modes × 2 signs × 3 steps = 270 displaced dipole calculations.

The accepted direct center has energy −570.2261138802517 Eh, RMS nuclear gradient 3.4088839779 × 10⁻⁷ Eh Bohr⁻¹ and maximum component 9.0119264406 × 10⁻⁷ Eh Bohr⁻¹. RI-J has energy −570.2262808707744 Eh, RMS gradient 8.0528701431 × 10⁻⁸ Eh Bohr⁻¹ and maximum component 2.2248563658 × 10⁻⁷ Eh Bohr⁻¹. Absolute energies from the two different integral treatments are not used to rank conformer stability. These calculations establish qualified local models rather than a global conformer search or a basis convergence result.

Each model has 45 vibrational modes and no imaginary modes. The lowest direct model mode is 74.0311896756 cm⁻¹, or 2.21939923215 THz, with IR intensity 2.14307101839 km mol⁻¹. The RI-J mode is 74.0061664997 cm⁻¹, or 2.21864905621 THz, with intensity 2.13888370090 km mol⁻¹. Comparing sorted full spectra gives a maximum absolute frequency difference of 0.4048637553 cm⁻¹ and an RMS difference of 0.1106920003 cm⁻¹. The maximum absolute IR intensity difference is 0.4653021293 km mol⁻¹. The complete per mode coordinates, frequencies and intensities accompany the numerical sources.

Both models place their lowest isolated molecule fundamental above the 0.1–1.5 THz classifier interval. This qualified reference sets a spectral scale for interpretation. It does not assign the discriminative herbal coordinates to protocatechuic acid or establish the absence of collective, packing or hydration contributions in the measured materials. Isolated molecule harmonic predictions and bulk herbal absorption address distinct physical systems. Molecular calculations were not used as priors or features in the classifier.

The accepted numerical source is fixed at commit 83df7a9ffa82c7045a22722382aeb78aed55db6c under reports/pyscf_ir_20261111/jobs. Complete local copies preserve mode tables, original results, numerical arrays and verification receipts. Each target molecule and numerical implementation has its own acceptance status. Completion of older spinosin or calycosin numerical output, or acceptance of water controls, is not acceptance of a target vibrational minimum or its IR spectrum.

Table S8. Molecular result status at the inspected fixed snapshots.

|Molecule or control|Available result|Use in this paper|
|---|---|---|
|Protocatechuic acid direct grid 5|Qualified stationary point and full 45 positive mode IR spectrum|Numerical isolated molecule reference|
|Protocatechuic acid RI-J grid 5|Qualified stationary point and full 45 positive mode IR spectrum|Independent integral treatment comparison|
|Calycosin latest continuation|263 of 396 full double step derivative calculations at the 15 November snapshot; IR precursor only|Progress record without a qualified final spectrum|
|Spinosin C003 and C008|26 and 27 gradient evaluations at the 15 November snapshot; optimization continuing; no production Hessian or qualified IR|Progress record without mode or chemical assignments|
|Spinosin C002 and C004|Ten qualified energy or gradient records representing eight distinct geometries; no converged minimum or final Hessian|Separate optimization record|
|Earlier spinosin and calycosin|Nonminimum numerical diagnostics including imaginary modes|Diagnostic motivation for the qualified repair workflow|
|Water controls|Passed direct and RI-J numerical frequency and IR controls|Implementation control, separate from target acceptance|

Spinosin gradient evaluation counts are not accepted optimization step counts, numbers of conformers or numbers of vibrational modes. Fixed snapshots describe archived progress rather than a live monitor of remote jobs. Older nonstationary protocatechuic acid frequency estimates are superseded by the two qualified models above.

Figure S1. Qualified isolated molecule IR predictions for protocatechuic acid at PBE/def2-SVP without D3, using direct electron repulsion integrals and RI-J fitting. Panel a compares the low frequency predictions with the shaded experimental analysis interval; panel b shows all 45 positive modes. Stick positions and heights are the archived frequencies and dipole derived intensities, with no broadening or fitted scaling. Both profiles have their lowest fundamental above 2.218 THz. The figure describes the stated local isolated molecule models rather than a solid state assignment.

## S9 Reproducibility and evidence files

The corrected background matrix comprises 1305 tasks, including 225 classical and 1080 neural tasks. Its saved intermediate backup and continuation are parts of that same matrix. V4 adds 330 tasks, including 15 classical and 315 neural tasks. The resulting 1635 tasks and the separate B1–B5 total of 1380 task bundles are computational accounting units, not independent validation cohorts or additional specimens. The current protocol register contains the corrected comparisons and complete settings, with source commits and paths. Existing names containing legacy or OLD in those corrected sources identify the original corrected matrix and do not change its label definition.

Source files are provided through the frozen paper manifest, including registry and feature audit, specimen roles, classifier settings and summaries, selected feature counts and parameters, paired effect tables, class specific confusion counts and all B round summaries. The principal V4 source is commit b051733989161517a68c4a60cce75766880c95f5. The common fixed role classification source is commit 0156e2189d13456d1faf05ebfd53d86eeeb4f5e7. The current main paper input view includes the acquisition declaration of 12 November 2026 and its verified materials and methods tables. Molecular source qualification is documented independently of classifier result verification.

Specimen level prediction replay, immutable source objects, current workbook hashes and role counts have been checked in the archived acceptance receipts. Rechecking saved predictions does not refit the original models. Classification validation remains internal to the represented collection, and the B analyses remain exploratory TRAIN only comparisons on previously exposed specimens. A genuinely new batch support/query experiment has not been run because independently annotated new batches are not available. The incomplete continuous meta learning work is not reported as a completed result.

AI assistance used Codex for repository inspection, evidence aggregation and classification, manuscript language editing and checking, and code design and review. The supplementary evidence files identify the underlying measured data and archived numerical outputs. All authors reviewed and approved the manuscript, supplementary material and submission and accept responsibility for the accuracy and integrity of the work.
