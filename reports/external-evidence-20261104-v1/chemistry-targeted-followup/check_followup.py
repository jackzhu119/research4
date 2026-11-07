"""Read-only source/axis checks; does not import scientific engines or fit models."""
from pathlib import Path
import csv
import hashlib
import json
import math


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    queries = json.loads((here / "search_log.json").read_text())
    assert queries["query_count"] == len(queries["queries"]) == 8
    assert len({x["query"] for x in queries["queries"]}) == 8
    assert queries["downloaded_original_spectral_files"] == 0
    evidence = json.loads((here / "evidence_register.json").read_text())
    for name in ("new_quantum_calculations", "new_classifier_fits",
                 "new_independent_spectral_samples", "new_raw_dataset_files"):
        assert evidence[name] == 0
    assert all(not x["direct_855_training_eligible"] for x in evidence["records"])
    units = json.loads((here / "scope_and_units.json").read_text())
    factor = units["speed_of_light_cm_per_s"] / 1e12
    assert factor == units["cm_minus_1_to_THz_exact"]
    with (here / "literature_peak_references.csv").open(newline="") as handle:
        peaks = list(csv.DictReader(handle))
    assert len(peaks) == 2
    assert {float(x["wavenumber_cm_minus_1"]) for x in peaks} == {1076, 1614}
    for item in peaks:
        frequency = float(item["equivalent_frequency_THz"])
        assert math.isclose(frequency, float(item["wavenumber_cm_minus_1"]) * factor,
                            rel_tol=0, abs_tol=1e-10)
        assert not (0.1 <= frequency <= 1.5)
        for key in ("in_target_0p1_1p5_THz", "our_DFT_result",
                    "raw_spectral_vector", "direct_855_training_eligible"):
            assert item[key] == "False"
        assert item["transcription_kind"] == "LITERATURE_PROSE_TRANSCRIPTION_NOT_RAW_CSV"
    receipt = json.loads((here / "sealed_prior_receipt.json").read_text())
    previous = repo / "reports/external-evidence-20261104-v1/chemistry"
    index = previous / "files_sha256.json"
    doc = repo / "docs/RESEARCH4_CHEMICAL_EVIDENCE_20261104.md"
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(index) == receipt["expected_prior_index_sha256"]
    assert sha(doc) == receipt["expected_prior_doc_sha256"]
    for row in json.loads(index.read_text())["files"]:
        assert sha(previous / row["path"]) == row["sha256"]
    followup_index = here / "files_sha256.json"
    if followup_index.exists():
        for row in json.loads(followup_index.read_text())["files"]:
            assert sha(here / row["path"]) == row["sha256"]
    print(json.dumps({"status": "PASS", "query_count": len(queries["queries"]),
                      "literature_prose_reference_rows": len(peaks),
                      "target_low_THz_reference_rows": 0,
                      "prior_sealed_payloads_unchanged": True,
                      "quantum_or_training_executed": False}))


if __name__ == "__main__":
    main()
