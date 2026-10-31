"""Source-development-only margin bound and clipping saturation diagnosis."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists())
RUN=Path(__file__).resolve().parent/"run"
P=json.loads((RUN/"protocol_snapshot.json").read_text())
rows=[]
for selector in P["selectors"]:
    directory=RUN/"development"/selector
    with gzip.open(directory/"metrics.csv.gz","rt",encoding="utf-8") as f:
        metrics=list(csv.DictReader(f))
    grouped=defaultdict(list)
    for m in metrics:
        grouped[(int(m["episode"]),m["candidate_id"])].append(m)
    for episode in range(30):
        with np.load(directory/f"episode_{episode:02d}_scores.npz",allow_pickle=False) as f:
            source,query=f["source_scores"],f["query_scores"]
        sq=np.sort(source,axis=1); source_gap=sq[:,-1]-sq[:,-2]
        qq=np.sort(query,axis=1); query_gap=qq[:,-1]-qq[:,-2]
        baseline=np.argmax(query,axis=1)
        predicted=defaultdict(list)
        with gzip.open(directory/f"episode_{episode:02d}_predictions.csv.gz","rt",encoding="utf-8") as f:
            for m in csv.DictReader(f):
                predicted[m["candidate_id"]].append(int(m["y_predicted"][-2:])-1)
        for candidate in P["candidates"]:
            m=grouped[(episode,candidate["id"])]
            cap=candidate["cap_fraction"]*np.median(source_gap)
            bound=2*cap*candidate["blend"]
            gate=float(np.quantile(source_gap,candidate["source_gate_quantile"]))
            possible=(query_gap<=bound)&(query_gap<=gate)
            accepted=sum(not json.loads(r["support_rejected"].lower()) for r in m)
            allpred=np.array(predicted[candidate["id"]]).reshape(20,len(query))
            rows.append({"selector":selector,"episode":episode,"stage":"selection" if episode<24 else "gate", "candidate_id":candidate["id"],
                         "source_median_margin":float(np.median(source_gap)),"query_min_margin":float(query_gap.min()),
                         "query_median_margin":float(np.median(query_gap)),"maximum_score_gap_correction":float(bound),
                         "source_gate_margin":gate,"query_samples":len(query),"support_draws":20,
                         "theoretically_flippable_query_samples_ignoring_support_evidence":int(possible.sum()),
                         "theoretically_flippable_draw_sample_pairs_with_support_evidence":int(possible.sum())*accepted,
                         "actual_flip_draw_sample_pairs":int(np.count_nonzero(allpred!=baseline[None,:])),
                         "support_evidence_rejected_draws":20-accepted})
with (RUN.parent/"cap_structural_diagnostic.csv").open("w",encoding="utf-8-sig",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
summary=[]
for selector in P["selectors"]:
    for candidate in P["candidates"]:
        cases=[r for r in rows if r["selector"]==selector and r["candidate_id"]==candidate["id"]]
        keys=["query_samples","support_draws","theoretically_flippable_query_samples_ignoring_support_evidence",
              "theoretically_flippable_draw_sample_pairs_with_support_evidence","actual_flip_draw_sample_pairs","support_evidence_rejected_draws"]
        out={"selector":selector,"candidate_id":candidate["id"],**{k:sum(r[k] for r in cases) for k in keys}}
        out["support_evidence_rejection_rate"]=out["support_evidence_rejected_draws"]/out["support_draws"]
        summary.append(out)
result={"status":"pass","scope":"source-development only, no final query labels or scores", "source_candidate_cases":len(rows),
        "explanation":"A necessary condition to change argmax is source query top-two gap <= 2*cap*blend, plus margin/evidence gates. Not sufficient: clipped corrections can be common offsets. Counts repeat overlapping samples and are not independent sample counts.",
        "all_candidates":summary}
(RUN.parent/"cap_structural_diagnostic.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"source_candidate_cases":len(rows),"all_theoretical_draw_sample_pairs":sum(r["theoretically_flippable_draw_sample_pairs_with_support_evidence"] for r in rows),
                  "all_actual_flips":sum(r["actual_flip_draw_sample_pairs"] for r in rows),
                  "mean_support_rejection_rate":float(np.mean([r["support_evidence_rejection_rate"] for r in summary]))}))
