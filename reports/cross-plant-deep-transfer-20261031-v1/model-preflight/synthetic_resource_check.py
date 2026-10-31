"""Measure the frozen encoder using invented fixture numbers, not research data."""
from pathlib import Path
import json
import time
import numpy as np
from herb_spectra import cross_plant_transfer as trial

def pack(prefix, n, classes, seed):
    rng = np.random.default_rng(seed)
    return {"X": rng.normal(size=(n, 128)), "axis": trial.AXIS.copy(),
            "record_ids": np.asarray([prefix + f"-{i:04d}" for i in range(n)]),
            "source_labels": np.asarray([classes[i % len(classes)] for i in range(n)])}

out = Path("outputs/cross-plant-deep-transfer-synthetic-resource-20261031-v1")
out.mkdir(exist_ok=False)
source = pack("SYNTH_ZY", 342, trial.CLASSES, 1)
aux = {"huangqi": pack("SYNTH_HQ", 403, ["HQ01", "HQ02", "HQ03", "HQ04"], 2),
       "tianma": pack("SYNTH_TM", 405, ["TM01", "TM02", "TM03", "TM04"], 3),
       "ginseng": pack("SYNTH_GS", 42, ["0", "1"], 4)}
profile = dict(trial.protocol()["profiles"][0], id="synthetic_resource2", epochs=2, pretrain_steps=4)
records = []
begin = time.perf_counter()
for arm in trial.ARMS:
    fitted = trial.fit_pipeline(source, aux, arm, profile, 42, "synthetic_resource_only", out / arm)
    metadata = fitted.metadata
    records.append({k: metadata[k] for k in ("arm", "elapsed_seconds", "target_training_seconds", "pretrain_seconds",
                                            "pretrain_steps", "target_steps", "target_model_parameters")})
result = {"completed_at_utc": trial.utc(), "purpose": "SYNTHETIC_RESOURCE_CHECK_NOT_RESEARCH_RESULT",
          "real_research_fits": 0, "synthetic_fits": 4, "profile": profile,
          "records": records, "wall_seconds": time.perf_counter() - begin,
          "source_versions": trial.source_versions(), "main_source_pipeline_budget": 240,
          "main_outer_pipeline_budget": 60, "main_source_target_steps": 103680,
          "main_pretrain_steps": 27000, "main_outer_target_steps_range": [31680, 63360]}
trial.write(Path("reports/cross-plant-deep-transfer-20261031-v1/model-preflight/synthetic_resource_check.json"), result)
print(json.dumps(result, ensure_ascii=False))
