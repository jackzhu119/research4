"""Read-only auxiliary source review; no estimator or adapter is fitted."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "outputs/upgrade-20261031-v1/auxiliary/v2"
DEST = Path(__file__).resolve().parent
PROTOCOL = ROOT / "data/research/auxiliary_multiplant_protocol_20261031.json"
checks = 0
cells = 0


def require(condition, detail):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(detail)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def values_equal(actual, expected, detail):
    global cells
    require(actual.shape == expected.shape, detail + " shape")
    require(np.array_equal(actual, expected), detail + " values")
    cells += actual.size


def main():
    target = DEST / "source_review.json"
    if target.exists():
        raise FileExistsError("Preserve completed review")
    protocol, prep, candidate = load(PROTOCOL), load(RUN / "preparation.json"), load(RUN / "candidate_manifest.json")
    require(candidate["protocol_sha256"] == sha(PROTOCOL), "Candidate protocol binding")
    require(candidate["preparation_sha256"] == sha(RUN / "preparation.json"), "Candidate input preparation binding")
    for relative, expected in candidate["source_sha256"].items():
        require(sha(ROOT / relative) == expected, "Frozen source changed: " + relative)
    core_manifest = load(ROOT/"outputs/upgrade-20261031-v1/neural/run_manifest.json")
    for relative in ("src/herb_spectra/upgrade_neural.py","src/herb_spectra/leakage_guard.py","src/herb_spectra/selection.py","src/herb_spectra/classification.py"):
        require(candidate["source_sha256"][relative] == core_manifest["source_sha256"][relative], "Frozen core dependency rewritten")
    require(sha(ROOT/"reports/upgrade-20261031-v1/neural-independent-audit/final_review.json") == "5de5b01bb52753f0d3dd75761e74f9a6142f852652a54816d64cfb0b439d3af0", "Completed core audit changed")
    require(protocol == load(RUN / "protocol_snapshot.json"), "Protocol snapshot changed")
    old_protocol = load(RUN.parent/"protocol_snapshot.json")
    require(all(protocol[k] == v for k,v in old_protocol.items() if k not in {"protocol_id","score_protocol"}), "Numerical protocol changed during boundary repair")
    require(protocol["datasets"] == ["huangqi", "tianma"], "Separate two-plant tasks")
    require(protocol["outer_folds"] == 5 and protocol["inner_folds"] == 2 and protocol["split_seed"] == 20261031, "Fixed split protocol")
    require(protocol["seeds"] == [42,43,44] and protocol["primary_seed"] == 42, "Seed protocol")
    require(protocol["sample_counts"] == {"huangqi":403,"tianma":405}, "Population changed")
    require(protocol["batch_ids_known"] is False and protocol["independent_confirmation"] is False and protocol["named_geography_externally_verified"] is False, "Scientific overstatement")
    require(protocol["fusions"] == {"huangqi":["time","frequency"],"tianma":["alpha","n"]}, "Incomplete-cohort path substituted")
    require(protocol["fixed_alpha_n_band"] == [.1,1.5], "Prior candidate frequency window changed")
    require(len(protocol["candidates"]["full_svm"]) == 4 and len(protocol["candidates"]["mcuve_svm"]) == 8, "Traditional model budget")
    require(all(len(protocol["candidates"][f]) == 2 for f in ("cnn","resnet","se","fusion")), "Neural model budget")
    base = [{"C":C,"gamma_rule":g} for C in (1.,10.) for g in ("scale","scale_x0.1")]
    expected_full = [{"id":f"full{i:02d}","kind":"full_svm",**v} for i,v in enumerate(base)]
    expected_mc = [{"id":f"mc{i:02d}","kind":"mcuve_svm",**v,"mc_runs":r,"mc_components":lv,"mc_ratio":ratio}
        for i,(r,lv,ratio,v) in enumerate((r,lv,ratio,v) for r,lv,ratio in [(10,1,.1),(20,2,.2)] for v in base)]
    require(protocol["candidates"]["full_svm"] == expected_full and protocol["candidates"]["mcuve_svm"] == expected_mc, "SVM/MC parameter grid changed")
    core_protocol = load(ROOT/"data/research/upgrade_neural_protocol_20261031.json")
    for family in ("cnn","resnet","se","fusion"):
        require(protocol["candidates"][family] == [{**v,"kind":family} for v in core_protocol["profiles"]], "Inherited fixed neural profiles changed")
    require(sum(len(v) for v in protocol["candidates"].values()) * 20 * 3 == 1200, "Actual inner fit budget")
    require(not (RUN / "selection_lock.json").exists() and not list(RUN.glob("selection/**/pipeline.joblib")) and not (RUN / "outer_predictions.csv.gz").exists(), "Pre-source review must precede actual source fits")
    boundary = load(RUN / "boundary_test_provenance.json")
    require(boundary["tested_implementation_sha256"] == candidate["source_sha256"]["src/herb_spectra/auxiliary_multiplant.py"], "Boundary suite binds another module")
    require(sha(ROOT / boundary["test_log"]) == boundary["test_log_sha256"], "Boundary test log changed")
    require(sha(ROOT / "tests/test_auxiliary_multiplant.py") == boundary["test_module_sha256"], "Boundary test module changed")
    require(boundary["tests"] >= 8 and (ROOT / boundary["test_log"]).read_text().rstrip().endswith("OK"), "Boundary suite incomplete")

    overlay = load(ROOT / protocol["overlay_manifest"])
    for key in ("overlay_manifest","overlay_lineage","annotation"):
        require(sha(ROOT / protocol[key]) == protocol[key+"_sha256"], "Overlay provenance changed: " + key)
    require(overlay["base_lineage_sha256"] == "f9b8b2529c17a9b3b8b14c04c8a3bdd0560a5e8a2a943bb6213923cfeb588ddb", "Historical lineage identity changed")
    require(sha(ROOT / overlay["base_lineage"]) == overlay["base_lineage_sha256"], "Historical lineage was rewritten")
    require(sha(ROOT / "data/multiplant_database/metadata/sample_lineage.csv") == overlay["base_lineage_sha256"], "Database historical lineage changed")
    base = [r for r in read_csv(ROOT / overlay["base_lineage"]) if r["dataset_id"] in protocol["datasets"]]
    over = read_csv(ROOT / protocol["overlay_lineage"])
    key = lambda row:(row["dataset_id"],row["modality"],row["record_id"])
    bybase, byover = {key(r):r for r in base}, {key(r):r for r in over}
    require(len(over) == len(byover) == len(base) == len(bybase) == 2929, "Overlay has added or removed measurements")
    require(set(bybase) == set(byover), "Overlay identity set changed")
    permitted = {"physical_sample_id","independence_evidence","quality_status"}
    for k, row in byover.items():
        original = bybase[k]
        require(all(row[f] == value for f,value in original.items() if f not in permitted), "Overlay changed original metadata " + str(k))
        require(row["physical_sample_id"] == row["record_id"], "Incorrect independent specimen mapping")
        require(row["independence_evidence"] == "user_declared_no_repeated_scans_or_repressing_20261031", "Unsupported independence evidence")
        require(all(row[f] == "" for f in ("parent_sample_id","preparation_id","batch_id","measurement_id")), "Unknown ancestry/batch was fabricated")
        require(row["quality_status"] == "EXPLORATORY_USER_DECLARED_INDEPENDENCE_BATCH_UNKNOWN", "Overlay confirmation overstatement")
    annotation = load(ROOT / protocol["annotation"])
    require(annotation["verbatim_reply"] == "不存在重复扫描和重新压片，是单独的", "User declaration altered")
    require(annotation["independent_confirmation_established"] is False and annotation["original_database_and_lineage_rewritten"] is False, "Declaration overstatement")

    secondary_manifest_path = ROOT / "data/multiplant_database/data_quality/secondary_read_evidence/manifest.json"
    secondary = load(secondary_manifest_path)
    build = load(ROOT / "data/multiplant_database/metadata/build_manifest.json")
    require(sha(secondary_manifest_path) == build["secondary_raw_cache_manifest_sha256"], "Independent OOXML reader manifest changed")
    require(sha(ROOT / "data/multiplant_database/data_quality/secondary_read_evidence/read_secondary.py") == secondary["reader_script_sha256"], "Independent reader implementation changed")
    evidence = {(r["dataset"],r["source_id"]):r for r in secondary["source_workbooks"]}
    raw, transformed, labels_by_plant, index_by_plant = {}, {}, {}, {}
    sheet_bindings, raw_workbooks = 0, set()
    for dataset in protocol["datasets"]:
        classes = protocol["class_orders"][dataset]
        require(classes == [("HQ" if dataset == "huangqi" else "TM")+f"{i:02d}" for i in range(1,5)], "Class spaces changed")
        raw[dataset], transformed[dataset] = {}, {}
        for mode, entry in protocol["inputs"][dataset].items():
            require(sha(ROOT / entry["path"]) == entry["sha256"], "Original DB modality changed")
            p = npz(ROOT / entry["path"]); raw[dataset][mode] = p
            ids = p["record_ids"].astype(str).tolist(); groups = p["filename_groups"].astype(str).tolist()
            require(len(ids) == len(set(ids)) == protocol["sample_counts"][dataset], "Raw cohort denominator")
            require(set(groups) == set(classes) and all(not v for v in p["labels"]), "Raw filename groups must not masquerade as authenticated origins")
            require(p["X"].shape == (len(ids),len(p["axis"])) and np.isfinite(p["X"]).all(), "Raw numeric shape/finiteness")
            require(np.isfinite(p["axis"]).all() and (np.diff(p["axis"])>0).all(), "Raw coordinate axis")
            if mode == "alpha":
                labels_by_plant[dataset] = dict(zip(ids,groups,strict=True)); index_by_plant[dataset] = {v:i for i,v in enumerate(ids)}
            require(dict(zip(ids,groups,strict=True)) == labels_by_plant[dataset], "Modality pairing/group mismatch")
            for group in classes:
                proof = evidence[(dataset,group)]
                require(sha(ROOT/proof["source_file"]) == proof["sha256"] == proof["expected_sha256"], "Original workbook changed")
                raw_workbooks.add(proof["source_file"])
                rec = next(r for r in proof["sheets"] if r["modality"] == mode)
                require(rec["status"] == "READ_COMPLETE" and rec["crc"] == "pass", "Blocked sheet was used")
                require(sha(ROOT/rec["artifact"]) == rec["artifact_sha256"], "Independent original-sheet values changed")
                sheet = npz(ROOT/rec["artifact"])
                values_equal(p["axis"],sheet["axis"],"Independent original axis")
                source_rows = [r for r in over if r["dataset_id"] == dataset and r["modality"] == mode and r["filename_group"] == group]
                require(len(source_rows) == len(sheet["X"]), "Sheet record coverage")
                dbindex = {v:i for i,v in enumerate(ids)}
                for row in source_rows:
                    require(row["source_file"] == proof["source_file"] and row["source_hash"] == proof["sha256"], "Canonical ID source workbook mismatch")
                    position = sheet["column_numbers"].tolist().index(int(row["excel_column"]))
                    require(str(sheet["headers"][position]) == row["original_header"], "Canonical ID original header mismatch")
                    values_equal(p["X"][dbindex[row["record_id"]]],sheet["X"][position],"Canonical ID numeric row mismatch")
                    sheet_bindings += 1
            if mode in ("alpha","n"):
                mask = (p["axis"] >= .1) & (p["axis"] <= 1.5)
                X, axis = p["X"][:,mask],p["axis"][mask]
                require(X.shape[1] == 612, "Candidate axis width")
            else:
                blocks = np.array_split(np.arange(len(p["axis"])),512 if mode=="time" else 256)
                X = np.stack([p["X"][:,b].mean(1) for b in blocks],axis=1)
                axis = np.asarray([p["axis"][b].mean() for b in blocks])
                require(prep["transforms"][dataset][mode]["operation"]["fixed_single_spectrum_blocks"] == [[int(b[0]),int(b[-1])+1] for b in blocks], "Fixed per-spectrum block boundaries")
            transformed[dataset][mode] = (ids,X)
            values_equal(np.asarray(prep["transforms"][dataset][mode]["axis"]),axis,"Prepared spectral coordinate")
        require(set(transformed[dataset]) == {"alpha",*protocol["fusions"][dataset]}, "Forbidden/missing fusion mode")
    require(len(raw_workbooks) == 8, "Source workbook set changed")

    require(len(prep["contexts"]) == len({c["split_id"] for c in prep["contexts"]}) == 30, "Expected thirty split roles")
    old_contexts = {c["split_id"]:c for c in load(RUN.parent/"preparation.json")["contexts"]}
    for c in prep["contexts"]:
        old = old_contexts[c["split_id"]]
        require(all(c[k] == old[k] for k in ("source_ids","inference_ids","source_pack_sha256","inference_pack_sha256")), "Boundary repair changed prepared records or spectra")
        require(sha(ROOT/c["scoring_truth"]) == sha(ROOT/old["scoring_truth"]), "Boundary repair changed truth labels")
    splits = load(RUN/"split_manifest.json")
    require(splits["no_seed_search"] is True and splits["independent_confirmation"] is False, "Split scientific status")
    context_by_role = {(c["dataset_id"],c["outer_fold"],c["stage"],c["inner_fold"]):c for c in prep["contexts"]}
    outer_denominators = {}
    for dataset in protocol["datasets"]:
        ids = raw[dataset]["alpha"]["record_ids"].astype(str); labels = raw[dataset]["alpha"]["filename_groups"].astype(str)
        expected_outer, outer_seen = [], []
        for fold,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261031).split(ids,labels)):
            inner = list(StratifiedKFold(2,shuffle=True,random_state=20261031+fold).split(ids[train],labels[train]))
            expected_outer.append({"outer_fold":fold,"train_ids":ids[train].tolist(),"test_ids":ids[test].tolist(),
                "inner_folds":[{"inner_fold":j,"train_ids":ids[train[it]].tolist(),"validation_ids":ids[train[iv]].tolist()} for j,(it,iv) in enumerate(inner)]})
            roles = [("inner",j,train[it],train[iv]) for j,(it,iv) in enumerate(inner)] + [("outer",None,train,test)]
            outer_seen.extend(ids[test].tolist())
            for stage,j,source,inference in roles:
                c = context_by_role[(dataset,fold,stage,j)]
                require(c["source_ids"] == ids[source].tolist() and c["inference_ids"] == ids[inference].tolist(), "Fixed source/inference role changed")
                require(not set(c["source_ids"]) & set(c["inference_ids"]), "Specimen leaks across source/inference")
                if stage == "inner":require(not set(ids[test]) & (set(c["source_ids"]) | set(c["inference_ids"])), "Outer specimen enters inner tuning")
                require(c["lineage_status"] == "EXPLORATORY", "Prepared provenance overstatement")
                for kind, positions in (("source",source),("inference",inference)):
                    require(sha(ROOT/c[kind+"_pack"]) == c[kind+"_pack_sha256"], "Pack bytes changed")
                    p = npz(ROOT/c[kind+"_pack"])
                    expected_keys = {"record_ids",*("X_"+m for m in transformed[dataset])} | ({"source_labels"} if kind=="source" else set())
                    require(set(p) == expected_keys, "Feature/label pack schema")
                    require(p["record_ids"].tolist() == ids[positions].tolist(), "Pack canonical ID order changed")
                    for mode,(original_ids,X) in transformed[dataset].items():
                        index = {v:i for i,v in enumerate(original_ids)}
                        values_equal(p["X_"+mode],X[[index[v] for v in ids[positions]]],"Pack numeric binding")
                    if kind == "source":require(p["source_labels"].tolist() == labels[positions].tolist(), "Training labels changed")
                truth_rows = read_csv(ROOT/c["scoring_truth"])
                require(sha(ROOT/c["scoring_truth"]) == c["scoring_truth_sha256"], "Independent scoring truth changed")
                require(truth_rows == [{"record_id":str(ids[i]),"label":str(labels[i])} for i in inference], "Scoring truth binding")
                require(len({r["label"] for r in truth_rows}) == 4, "Class silently removed")
        require(expected_outer == splits["datasets"][dataset], "Persisted split differs from fixed generated folds")
        require(len(outer_seen) == len(set(outer_seen)) == len(ids), "OOF specimen denominator")
        outer_denominators[dataset] = len(outer_seen)
    result = {"status":"TECHNICAL_AUXILIARY_SOURCE_REVIEW_PASS_EXPLORATORY_DATA","allow_source_training":True,
        "completed_at_utc":datetime.now(timezone.utc).isoformat(),"checks":checks,"numeric_cells_compared":cells,
        "original_source_workbooks":len(raw_workbooks),"canonical_original_sheet_row_bindings":sheet_bindings,
        "split_contexts":30,"fixed_outer_folds":5,"fixed_inner_folds":2,"outer_specimens_by_plant":outer_denominators,
        "inner_candidate_fits_planned":1200,"new_estimator_fits":0,"outer_predictions_scored":False,
        "protocol_sha256":sha(PROTOCOL),"candidate_manifest_sha256":sha(RUN/"candidate_manifest.json"),
        "preparation_sha256":sha(RUN/"preparation.json"),"boundary_tests":boundary["tests"],
        "historical_lineage_sha256":overlay["base_lineage_sha256"],"overlay_manifest_sha256":sha(ROOT/protocol["overlay_manifest"]),
        "review_script_sha256":sha(Path(__file__)),
        "scope":["Source code and run-manifest bindings","User declaration overlay preserves historical measurements and unknown ancestors","Original-sheet cached values bound by workbook SHA/header/column","Every prepared source/inference/truth value and canonical ID","Fixed five-outer/two-inner folds and all classes retained","Training-only scalers/MC-PLS/SVC/neural code reviewed; final scope verified after actual fits","Separate label spaces and no cross-plant or theory prior input"],
        "limits":["User declaration supports exploratory independence; upstream material and batches remain unknown","Named locations and species not externally authenticated","Units and instrument compatibility unknown; cross-plant transfer blocked","API/file separation is not OS access control","Scientific claim is source-file-group classification, not independent origin confirmation"]}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(dict,list))},ensure_ascii=False))


if __name__ == "__main__":
    main()
