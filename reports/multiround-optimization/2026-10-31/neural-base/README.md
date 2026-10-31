# Supervised dense source-model search evidence

This is a retrospective same-batch source-representation search on real uploaded
spectra with anonymous origin groups. It uses no calibration labels. The locked
b006 model is SNV, source-only feature scaling, no PCA, hidden64-32, ReLU,200epochs,
unweighted source cross-entropy. It is independently implemented in PyTorch and
is not an exact MATLAB newff reproduction.

24candidates x6training-only tasks x3seeds =432development source fits,44496query
predictions. A source-only4/2 selection/gate locks the configuration before the
three final513source models predict171fixed specimens each. All four research
rounds were development-locked before the common final gate was released.

The final mean accuracy89.0838% is close to SVM88.8889%, but mean macroF1 .7729
is lower than SVM .8616. Neither this source improvement nor replacing a neural
backbone is a few-shot calibration gain. All seeds, imbalanced-class failures,
and all candidate development results remain inspectable.

`training_checkpoints.jsonl.gz` stores each original UTF-8training.json as text
plus original path and SHA; encoding its text field to UTF-8 recovers exact bytes.
`prediction_files.jsonl.gz` preserves each original compressed CSV file as base64
plus its path and SHA, allowing even original gzip bytes to be recovered.
`all_predictions.csv.gz` provides the same45009rows as an ordinary unified CSV.
`original_files_sha256.json` covers original source weights,preprocessors,training
metadata,predictions,metrics and root manifests. All435metadata and435compressed
prediction files were byte-compared after restoration. The435development/final
cache contexts are retained locally; routine reproduction regenerates the search.

Only18locked development sources plus3final sources are exported separately
as21PyTorchstate_dicts and21source-fittedpreprocessors for downstream adaptation.
`development_arithmetic_audit.json` checks44496prediction rows and source-only
selection using independent sklearn formulas; `final_arithmetic_and_resume_audit.json`
checks513final rows,inference replay,pointwise predictions and no-fit resume.
Their scripts assume the original execution output layout,not this compact folder.

All855specimens have previous research exposure. Each origin currently has one
batch; these splits and repeated tasks are not independently acquired domains.
The anonymous labels have not independently verified real geographic names.
