"""Explain unchanged argmax using source-development score corrections only."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
RUN=Path(__file__).resolve().parent/"run"
protocol=json.loads((RUN/"protocol_snapshot.json").read_text())
budget=json.loads((RUN/"development_budget.json").read_text())
stats=defaultdict(lambda:{"necessary_bound_pairs":0,"top_two_equal_clipped_shift_pairs":0,"actual_flips":0})
for selector in protocol["selectors"]:
    for episode in range(30):
        with np.load(RUN/"development"/selector/f"episode_{episode:02d}_scores.npz",allow_pickle=False) as f:
            z={k:f[k] for k in f.files}
        ss,sq,sp=z["source_scores"],z["query_scores"],z["pool_scores"]
        ordered=np.argsort(sq,axis=1)
        top,second=ordered[:,-1],ordered[:,-2]
        gaps=sq[np.arange(len(sq)),top]-sq[np.arange(len(sq)),second]
        sg=np.sort(ss,axis=1);sg=sg[:,-1]-sg[:,-2]
        meta=next(r for r in budget if r["selector"]==selector and r["episode"]==episode)
        for draw in meta["draws"]:
            indices=np.array([list(z["pool_ids"]).index(v) for v in draw["ids"]])
            support=sp[indices];target=z["pool_y"][indices]
            errors=np.count_nonzero(z["classes"][np.argmax(support,axis=1)]!=target)
            design=np.column_stack([support,np.ones(len(support))])
            truth=np.eye(7)[[list(z["classes"]).index(v) for v in target]]
            weights=np.linalg.solve(design.T@design+np.eye(8),design.T@(truth-support))
            raw=np.column_stack([sq,np.ones(len(sq))])@weights
            for c in protocol["candidates"]:
                if errors<c["minimum_support_errors"]:
                    continue
                cap=c["cap_fraction"]*np.median(sg)
                gate=np.quantile(sg,c["source_gate_quantile"])
                possible=(gaps<=2*cap*c["blend"])&(gaps<=gate)
                change=np.clip(raw,-cap,cap)*c["blend"]
                change[gaps>gate]=0.
                difference=change[np.arange(len(sq)),top]-change[np.arange(len(sq)),second]
                row=stats[(selector,c["id"])]
                row["necessary_bound_pairs"]+=int(possible.sum())
                row["top_two_equal_clipped_shift_pairs"]+=int(np.count_nonzero(possible&(np.abs(difference)<1e-12)))
                row["actual_flips"]+=int(np.count_nonzero(np.argmax(sq+change,axis=1)!=top))
rows=[{"selector":k[0],"candidate_id":k[1],**v} for k,v in stats.items()]
result={"scope":"source-development only, no final query data", "explanation":"For pairs passing the necessary gap bound, equal clipping of the original top two scores preserves their ordering. This diagnoses the present score controller, not usefulness of all calibration approaches.",
        "necessary_bound_pairs":sum(r["necessary_bound_pairs"] for r in rows),
        "top_two_equal_clipped_shift_pairs":sum(r["top_two_equal_clipped_shift_pairs"] for r in rows),
        "actual_flips":sum(r["actual_flips"] for r in rows),"rows":rows,"new_model_fits":0}
(RUN.parent/"clipping_toptwo_diagnostic.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:result[k] for k in ("necessary_bound_pairs","top_two_equal_clipped_shift_pairs","actual_flips")}))
