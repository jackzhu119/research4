import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from herb_spectra.image_feature_models import candidate_grid
from herb_spectra.image_feature_search import _cached_fit, choose_candidate, digest, reference_candidate, verify_release


class ImageFeatureSearchTests(unittest.TestCase):
    def rows(self):
        return [{"candidate_id":c["id"],"stage":stage,"status":"ok","accuracy":.8,
            "balanced_accuracy":.7,"macro_f1":.6,"selected_features":40}
            for c in candidate_grid() for stage in ("selection","gate")]

    def policy(self):
        return {"minimum_macro_f1_gain":.002,"maximum_accuracy_or_balanced_loss":.005,"macro_f1_tie_tolerance":.002}

    def test_macro_first_selection_accuracy_guard_and_gate(self):
        grid=candidate_grid();rows=self.rows();good=grid[3];bad=grid[4]
        for row in rows:
            if row["candidate_id"]==good["id"]:row["macro_f1"]=.65;row["accuracy"]=.798
            if row["candidate_id"]==bad["id"]:row["macro_f1"]=.8;row["accuracy"]=.78
        choice=choose_candidate(rows,grid,self.policy())
        self.assertEqual(choice["selected"],good)
        for row in rows:
            if row["candidate_id"]==good["id"] and row["stage"]=="gate":row["macro_f1"]=.6
        self.assertEqual(choose_candidate(rows,grid,self.policy())["selected"],reference_candidate())

    def test_identity_fallback_without_macro_gain(self):
        choice=choose_candidate(self.rows(),candidate_grid(),self.policy())
        self.assertTrue(choice["fallback_to_reference"])
        self.assertEqual(choice["eligible_selection_candidates"],0)

    def test_global_release_requires_exact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/"features";run.mkdir();protocol=root/"p.json";protocol.write_text("{}")
            for name in ["selection_lock.json","run_manifest.json"]:(run/name).write_text("{}")
            with self.assertRaisesRegex(ValueError,"release is required"):verify_release(run,protocol)
            record={"selection_lock":str(run/"selection_lock.json"),
                "selection_lock_sha256":digest(run/"selection_lock.json"),"protocol_sha256":digest(protocol),
                "run_manifest_sha256":digest(run/"run_manifest.json")}
            release=root/"all_development_locks.json";release.write_text(json.dumps({"rounds":[record]}))
            self.assertEqual(verify_release(run,protocol),release)
            protocol.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,"hashes changed"):verify_release(run,protocol)

    def test_all_twelve_pca_variants_roundtrip_and_resume_without_refit(self):
        rng=np.random.default_rng(52);X=rng.normal(size=(35,25));y=np.asarray(["A"]*17+["B"]*17+["C"])
        source={"X":X,"y":y,"record_ids":np.asarray([f"S{i:03d}" for i in range(len(X))])}
        axis=np.linspace(.1,1.5,25)
        candidates=[c for c in candidate_grid() if c["family"]=="pca_lda"]
        self.assertEqual(len(candidates),12)
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            for candidate in candidates:
                model,metadata=_cached_fit(directory,candidate,source,axis,42,cache={},resume=False)
                self.assertEqual(joblib.hash(model),metadata["model_fingerprint"])
                self.assertTrue(metadata["source_dump_load_labels_equal"])
                self.assertLessEqual(metadata["source_dump_load_max_score_difference"],1e-9)
                with patch("herb_spectra.image_feature_search.fit_candidate",side_effect=AssertionError("unexpected fit")):
                    restored,cached=_cached_fit(directory,candidate,source,axis,42,cache={},resume=True)
                self.assertEqual(metadata,cached)
                np.testing.assert_allclose(model.predict_scores(X),restored.predict_scores(X),rtol=0,atol=1e-12)


if __name__=="__main__":unittest.main()
