"""Source-only probability calibration and multiclass decision adjustments.

SVM decision scores are not probabilities. Their softmax is an explicitly
uncalibrated normalized-score reference. Positive scalar temperature preserves
argmax; class decision biases are not a posterior-calibration claim.
"""
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
import torch
from torch.nn import functional as F

from .calibration_exploration import fast_metrics
from .paper_fewshot_classification import sample_meta_episodes, state_fingerprint

PROVIDERS = ("svm", "mlp", "mlp_sup100")


def _scores(values):
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or len(result) < 1 or result.shape[1] < 2 or not np.isfinite(result).all():
        raise ValueError("Finite nonempty samples-by-class scores required")
    return result


def _labels(values, n, classes):
    result = np.asarray(values)
    if result.shape != (n,) or result.dtype.kind not in "iu" or np.any(result < 0) or np.any(result >= classes):
        raise ValueError("Known integer class required per specimen")
    return result.astype(int)


def softmax_probabilities(scores, temperature=1.):
    scores = _scores(scores)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    return softmax(scores/temperature, axis=1)


def probability_metrics(probabilities, labels):
    p = _scores(probabilities); y = _labels(labels, len(p), p.shape[1])
    if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1., atol=1e-10):
        raise ValueError("Class probabilities must be nonnegative and sum to one")
    confidence = p.max(1); correct = p.argmax(1) == y
    ece = 0.
    # [0,.1), ... [.9,1], so confidence=1 is included exactly once.
    bins = np.minimum((confidence*10).astype(int), 9)
    for bucket in range(10):
        rows = bins == bucket
        if np.any(rows):
            ece += np.mean(rows)*abs(float(correct[rows].mean())-float(confidence[rows].mean()))
    onehot = np.eye(p.shape[1])[y]
    return {"nll": float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()),
            "brier": float(np.square(p-onehot).sum(axis=1).mean()), "ece10": float(ece)}


def fit_temperature(scores, labels, regularization=.01):
    scores = _scores(scores); labels = _labels(labels,len(scores),scores.shape[1])
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("Nonnegative temperature regularization required")
    def objective(log_temperature):
        probability = softmax_probabilities(scores,float(np.exp(log_temperature)))
        return probability_metrics(probability, labels)["nll"] + regularization*float(log_temperature)**2
    result = minimize_scalar(objective,bounds=(-3.,3.),method="bounded",options={"xatol":1e-8})
    if not result.success or not np.isfinite(result.fun):
        raise FloatingPointError("Source OOF temperature fit failed")
    temperature = float(np.exp(result.x))
    return temperature, {"temperature":temperature,"log_temperature":float(result.x),
        "source_oof_rows":len(labels),"regularization":regularization,
        "bounded_log_temperature":[-3.,3.],"objective":float(result.fun),
        "raw_oof":probability_metrics(softmax_probabilities(scores),labels),
        "temperature_oof":probability_metrics(softmax_probabilities(scores,temperature),labels),
        "argmax_unchanged":bool(np.array_equal(scores.argmax(1),softmax_probabilities(scores,temperature).argmax(1)))}


def singleton_anchor_folds(labels, identifiers, folds=3, seed=20261111):
    """Hold singleton classes in fit; never fabricate their OOF labels."""
    y = np.asarray(labels); ids = np.asarray(identifiers)
    if y.ndim != 1 or ids.shape != y.shape or len(set(ids.tolist())) != len(ids) or folds < 2:
        raise ValueError("Unique source physical identifiers and a feasible fold count required")
    classes, counts = np.unique(y,return_counts=True)
    anchors = np.flatnonzero(np.isin(y,classes[counts < folds]))
    eligible = np.flatnonzero(~np.isin(y,classes[counts < folds]))
    if not len(eligible):
        raise ValueError("No class can supply genuine source OOF validation rows")
    rng = np.random.default_rng(seed); validation = [[] for _ in range(folds)]
    for label in classes[counts >= folds]:
        rows = np.flatnonzero(y == label)
        for fold, part in enumerate(np.array_split(rng.permutation(rows),folds)):
            validation[fold].extend(part.tolist())
    result = []
    all_rows = np.arange(len(y))
    for fold, rows in enumerate(validation):
        valid = np.asarray(sorted(rows),dtype=int); train = np.setdiff1d(all_rows,valid)
        if not len(valid) or set(train) & set(valid) or not set(anchors).issubset(set(train)):
            raise AssertionError("Invalid singleton-anchor source OOF partition")
        result.append({"fold":fold,"train":train,"validation":valid})
    if sorted(np.concatenate([r["validation"] for r in result]).tolist()) != eligible.tolist():
        raise AssertionError("Every eligible source specimen needs exactly one OOF prediction")
    return result, {"folds":folds,"seed":seed,"excluded_oof_indices":anchors.tolist(),
        "excluded_oof_ids":ids[anchors].tolist(),"excluded_oof_classes":classes[counts < folds].tolist(),
        "source_rows":len(y),"oof_rows":len(eligible),"singleton_anchor_in_all_training_folds":True}


def supervised_continuation(initial, features, labels, identifiers, seed, updates=100):
    """Mirror the pre-existing paired supervised arm, without computing FO."""
    x = torch.as_tensor(np.ascontiguousarray(features,dtype=np.float32)); y = torch.as_tensor(labels,dtype=torch.long)
    if x.ndim != 2 or y.shape != (len(x),) or not torch.isfinite(x).all():
        raise ValueError("Finite source features and integer categories required")
    before = state_fingerprint(initial); model = deepcopy(initial)
    tasks,policy = sample_meta_episodes(np.asarray(labels),np.asarray(identifiers),episodes=updates,
                                       seed=seed,way=3,support=1,query=1)
    optimizer = torch.optim.Adam(model.parameters(),lr=.0001); losses=[]; manifests=[]
    for task in tasks:
        model.train();optimizer.zero_grad(set_to_none=True)
        episodic = F.cross_entropy(model(x[task["query"]]),y[task["query"]])
        anchor = F.cross_entropy(model(x),y)
        (episodic+anchor).backward();optimizer.step()
        losses.append({"update":task["episode"]+1,"source_query_ce":float(episodic.detach()),"source_anchor_ce":float(anchor.detach())})
        manifests.append({"episode":task["episode"],"classes":task["classes"],
            "support_ids":np.asarray(identifiers)[task["support"]].tolist(),
            "query_ids":np.asarray(identifiers)[task["query"]].tolist()})
    model.eval()
    if state_fingerprint(initial) != before:
        raise AssertionError("Source initializer changed")
    return model,{"source_only":True,"seed":seed,"updates":updates,"policy":policy,
        "source_tasks":manifests,"losses":losses,"fingerprint":state_fingerprint(model)}


def candidate_grid():
    result=[]
    for provider in PROVIDERS:
        variants=[{"kind":"raw"},{"kind":"temperature"}]
        variants += [{"kind":"prior_bias","beta":b} for b in (.25,.5,1.)]
        variants += [{"kind":"coordinate_bias","penalty":p} for p in (.001,.01)]
        for v in variants:
            result.append({"id":f"p{len(result):03d}_{provider}_{v['kind']}","provider":provider,**v})
    return result


def fit_coordinate_bias(probabilities, labels, source_labels, penalty=.001):
    p=_scores(probabilities);y=_labels(labels,len(p),p.shape[1]);source=np.asarray(source_labels)
    counts=np.bincount(source,minlength=p.shape[1]);fixed=sorted(set([0]+np.flatnonzero(counts < 3).tolist()))
    bias=np.zeros(p.shape[1]);trace=[];logp=np.log(np.maximum(p,1e-15))
    def objective(b):
        return fast_metrics(y,np.argmax(logp+b,axis=1),p.shape[1])["macro_f1"]-penalty*float(np.mean(b*b))
    for sweep in range(2):
        for c in range(p.shape[1]):
            if c in fixed:continue
            options=[]
            for value in (-1.,-.5,-.25,0.,.25,.5,1.):
                proposed=bias.copy();proposed[c]=value
                options.append((objective(proposed),-abs(value),-value,value))
            best=max(options);bias[c]=best[3]
            trace.append({"sweep":sweep,"class_index":c,"bias":float(bias[c]),"penalized_macro_f1":float(best[0])})
    return bias,{"bias":bias.tolist(),"penalty":penalty,"fixed_bias_classes":fixed,
        "source_oof_fit_rows":len(y),"objective":"7-class MacroF1 minus penalty*mean(bias^2)","trace":trace}


def fit_decision(candidate, oof_scores, oof_y, source_y, temperature):
    kind=candidate["kind"];scores=_scores(oof_scores)
    if kind not in {"raw","temperature","prior_bias","coordinate_bias"}:raise ValueError("Unknown decision kind")
    t=1. if kind=="raw" else float(temperature);bias=np.zeros(scores.shape[1]);record={}
    if kind=="prior_bias":
        counts=np.bincount(np.asarray(source_y,dtype=int),minlength=scores.shape[1])
        if np.any(counts==0):raise ValueError("Source class vocabulary incomplete")
        bias=-candidate["beta"]*np.log(counts/counts.sum());bias-=bias[0]
    elif kind=="coordinate_bias":
        bias,record=fit_coordinate_bias(softmax_probabilities(scores,t),oof_y,source_y,candidate["penalty"])
    return {"candidate":candidate,"temperature":t,"bias":bias.tolist(),"fit_record":record,
        "decision_bias_is_not_posterior_calibration":kind.endswith("bias")}


def apply_decision(fitted,scores):
    probability=softmax_probabilities(scores,fitted["temperature"])
    predicted=np.argmax(np.log(np.maximum(probability,1e-15))+np.asarray(fitted["bias"]),axis=1)
    return predicted,probability
