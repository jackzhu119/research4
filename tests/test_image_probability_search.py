import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from herb_spectra.image_probability import candidate_grid,PROVIDERS
from herb_spectra.image_probability_search import select_methods,get_score_contexts,run_search
from herb_spectra.neural_base_search import write_json,object_digest,digest


POLICY={"minimum_macro_f1_gain":.002,"maximum_accuracy_loss":.005,"maximum_balanced_accuracy_loss":.005,
        "minimum_nll_gain":.001,"maximum_brier_loss":.002}


def rows_fixture():
    rows=[]
    for c in candidate_grid():
        for stage in ("selection","gate"):
            rows.append({"candidate_id":c["id"],"provider":c["provider"],"stage":stage,
                "accuracy":.8,"balanced_accuracy":.8,"macro_f1":.7,
                "nll":1. if c["kind"]=="raw" else .9,"brier":.3,"ece10":.1})
    return rows


class ProbabilitySearchTests(unittest.TestCase):
    def test_candidate_grid_and_separate_nll_selection(self):
        self.assertEqual(len(candidate_grid()),21);choice=select_methods(rows_fixture(),POLICY)
        for provider in PROVIDERS:
            self.assertEqual(choice["choices"][provider]["selected"]["kind"],"raw")
            self.assertEqual(choice["choices"][provider]["probability_selected"]["kind"],"temperature")

    def test_macro_gain_must_pass_gate_and_accuracy_guard(self):
        rows=rows_fixture();candidate=next(c for c in candidate_grid() if c["kind"]=="prior_bias")
        for row in rows:
            if row["candidate_id"]==candidate["id"]:
                row["macro_f1"]+=.1
                if row["stage"]=="gate":row["accuracy"]-=.02
        choice=select_methods(rows,POLICY)["choices"][candidate["provider"]]
        self.assertEqual(choice["eligible_classification_candidates"],1)
        self.assertEqual(choice["selected"]["kind"],"raw")

    def test_final_context_api_rejects_prediction_truth_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);input_path=root/"input";plan=root/"plan"
            input_path.write_bytes(b"input");plan.write_bytes(b"plan")
            write_json(root/"protocol_snapshot.json",{})
            write_json(root/"run_manifest.json",{"input_sha256":digest(input_path),"plan_sha256":digest(plan),
                "protocol_sha256":digest(root/"protocol_snapshot.json")})
            write_json(root/"final_contexts.json",["fake"])
            with patch("herb_spectra.image_probability_search._load_context",return_value={"prediction":{"y":np.array([0])}}):
                with self.assertRaises(AssertionError):get_score_contexts(input_path,plan,root,"final")

    def test_completed_development_resume_has_no_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);protocol=root/"protocol.json";output=root/"output";output.mkdir()
            config={"candidate_grid_sha256":object_digest(candidate_grid()),"selection":POLICY}
            write_json(protocol,config);write_json(output/"run_manifest.json",{})
            write_json(output/"development_completed.json",{})
            rows=rows_fixture();write_json(output/"all_development_metrics.json",rows)
            write_json(output/"selection_lock.json",select_methods(rows,POLICY))
            with patch("herb_spectra.image_probability_search.manifest",return_value={}), \
                 patch("herb_spectra.image_probability_search.get_score_contexts",return_value=([{"lineage":{"directory":"fake"}}],{})), \
                 patch("herb_spectra.image_probability_search._read_evaluation",return_value=rows), \
                 patch("herb_spectra.image_probability_search.build_context",side_effect=AssertionError("unexpected fit")):
                result=run_search(root/"input",root/"plan",protocol,output,resume=True)
            self.assertEqual(result["status"],"development_already_completed")


if __name__=="__main__":unittest.main()
