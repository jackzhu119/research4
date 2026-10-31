"""Paper-inspired classification pilot; not the authors' regression reproduction.

The compact common ResNet, GroupNorm, categorical losses and seven-class task
are explicit adaptations. Meta episodes use only eligible source specimens;
the rare class remains in supervised initialization and final classification.
Final query truth is absent from every adaptation function.
"""
from collections import defaultdict
from copy import deepcopy
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

from .calibration_exploration import development_episodes, encoded, fast_metrics, nested_draws
from .within_batch_holdout import prepare_holdout


STRATEGIES = ("head_ft", "full_ft", "prototype", "fomaml_ft")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def candidate_grid():
    result = []
    for strategy in ("head_ft", "full_ft", "fomaml_ft"):
        for learning_rate in (.0001, .001, .01):
            for steps in (1, 5, 10):
                result.append({"id": f"n{len(result):03d}_{strategy}", "strategy": strategy,
                               "learning_rate": learning_rate, "steps": steps})
    for blend in (.1, .25, .5, 1.):
        result.append({"id": f"n{len(result):03d}_prototype", "strategy": "prototype", "blend": blend})
    return result


def snv_tensor(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 1 or values.shape[1] < 16 or not np.isfinite(values).all():
        raise ValueError("Expected finite specimens-by-frequency spectra with at least16points")
    scale = values.std(axis=1, ddof=1, keepdims=True)
    if np.any(scale <= np.finfo(float).eps):
        raise ValueError("SNV cannot normalize a constant spectrum")
    normalized = (values-values.mean(axis=1, keepdims=True))/scale
    return torch.from_numpy(normalized.astype(np.float32))[:, None, :]


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(4, output_channels)
        self.conv2 = nn.Conv1d(output_channels, output_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(4, output_channels)
        self.skip = (nn.Identity() if input_channels == output_channels and stride == 1
                     else nn.Conv1d(input_channels, output_channels, 1, stride=stride, bias=False))

    def forward(self, values):
        hidden = F.relu(self.norm1(self.conv1(values)))
        hidden = self.norm2(self.conv2(hidden))
        return F.relu(hidden+self.skip(values))


class SmallResNet1D(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1,16,7,stride=2,padding=3,bias=False),
                                  nn.GroupNorm(4,16), nn.ReLU())
        self.blocks = nn.Sequential(ResidualBlock(16,16), ResidualBlock(16,32,stride=2))
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.projection = nn.Linear(32*8,32)
        self.head = nn.Linear(32,n_classes)

    def encode(self, values):
        return F.relu(self.projection(self.pool(self.blocks(self.stem(values))).flatten(1)))

    def forward(self, values):
        return self.head(self.encode(values))


def state_fingerprint(model):
    digest_ = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest_.update(name.encode()); digest_.update(str(tensor.dtype).encode())
        digest_.update(np.asarray(tensor.shape,dtype="<i8").tobytes())
        digest_.update(tensor.numpy().tobytes())
    return digest_.hexdigest()


def _checked_labels(labels, n, n_classes):
    values = np.asarray(labels)
    if (values.shape != (n,) or values.dtype.kind not in "iu" or np.any(values < 0)
            or np.any(values >= n_classes)):
        raise ValueError("One known integer category is required per specimen")
    return torch.as_tensor(values, dtype=torch.long)


def supervised_fit(features, labels, *, seed=42, n_classes=7, epochs=40,
                   batch_size=64, learning_rate=.001, weight_decay=.001):
    """Only the caller's source arrays enter supervised optimization."""
    torch.manual_seed(seed)
    y = _checked_labels(labels, len(features), n_classes)
    if set(y.tolist()) != set(range(n_classes)):
        raise ValueError("All known classes must occur in supervised source")
    model = SmallResNet1D(n_classes)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    counts = torch.bincount(y,minlength=n_classes).float()
    class_weights = len(y)/(n_classes*counts)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(y),generator=generator)
        total, samples = 0., 0
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features[indices]),y[indices],weight=class_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite supervised loss")
            loss.backward(); optimizer.step()
            total += float(loss.detach())*len(indices); samples += len(indices)
        losses.append({"epoch":epoch,"mean_weighted_cross_entropy":total/samples})
    model.eval()
    return model, losses


def sample_meta_episodes(labels, identifiers, *, episodes=100, seed=42,
                         way=3, support=1, query=3):
    labels = np.asarray(labels); identifiers = np.asarray(identifiers)
    if labels.shape != identifiers.shape or len(set(identifiers.tolist())) != len(labels):
        raise ValueError("Meta episodes require unique physical specimen IDs")
    pools = {int(c):np.flatnonzero(labels == c) for c in np.unique(labels)}
    eligible = sorted(c for c, rows in pools.items() if len(rows) >= support+query)
    excluded = sorted(set(pools)-set(eligible))
    if way < 2 or len(eligible) < way or support < 1 or query < 1:
        raise ValueError("Insufficient source classes for disjoint meta support/query")
    rng = np.random.default_rng(seed); result = []
    for episode in range(episodes):
        chosen = rng.choice(eligible,way,replace=False)
        support_indices, query_indices = [], []
        for c in chosen:
            order = rng.permutation(pools[int(c)])
            support_indices.extend(order[:support]); query_indices.extend(order[support:support+query])
        result.append({"episode":episode,"classes":chosen.tolist(),
            "support":np.asarray(support_indices,dtype=int),"query":np.asarray(query_indices,dtype=int)})
    return result, {"eligible_classes":eligible,"excluded_insufficient_classes":excluded,
        "support_per_class":support,"query_per_class":query,"way":way,
        "source_only":True,"tasks_are_real_batches":False}


def prototype_scores(model, source_x, source_y, query_x, *, support_x=None,
                     support_y=None, blend=0.):
    """Source centroids at blend0 are the matched zero-support classifier."""
    if not np.isfinite(blend) or not 0 <= blend <= 1:
        raise ValueError("Prototype blend must be in[0,1]")
    classes = np.arange(model.head.out_features)
    source_labels = np.asarray(source_y)
    if set(source_labels.tolist()) != set(classes.tolist()):
        raise ValueError("Source prototype vocabulary is incomplete")
    model.eval()
    with torch.no_grad():
        source_z = F.normalize(model.encode(source_x),dim=1)
        query_z = F.normalize(model.encode(query_x),dim=1)
        centroids = torch.stack([source_z[source_labels == c].mean(0) for c in classes])
        centroids = F.normalize(centroids,dim=1)
        if blend > 0:
            if support_x is None or support_y is None or set(np.asarray(support_y).tolist()) != set(classes.tolist()):
                raise ValueError("Every known class requires labelled support")
            support_z = F.normalize(model.encode(support_x),dim=1)
            target = torch.stack([support_z[np.asarray(support_y) == c].mean(0) for c in classes])
            centroids = F.normalize((1-blend)*centroids+blend*F.normalize(target,dim=1),dim=1)
        return (-torch.cdist(query_z,centroids).square()).cpu().numpy()


def meta_fit(initial, features, labels, identifiers, *, strategy="prototype", seed=42,
             episodes=100, learning_rate=.0001, way=3, support=1, query=3,
             inner_learning_rate=.01):
    """Actual episodic training; meta query belongs exclusively to source."""
    if strategy not in {"supervised","prototype","fomaml"}:
        raise ValueError("Unknown meta-training strategy")
    model = deepcopy(initial)
    before_initial = state_fingerprint(initial)
    y = _checked_labels(labels,len(features),model.head.out_features)
    tasks, policy = sample_meta_episodes(labels,identifiers,episodes=episodes,seed=seed,
                                        way=way,support=support,query=query)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    losses, manifests = [], []
    model.train()
    for task in tasks:
        si, qi = task["support"],task["query"]
        optimizer.zero_grad(set_to_none=True)
        if strategy == "supervised":
            both = np.concatenate([si,qi])
            loss = F.cross_entropy(model(features[both]),y[both])
            loss.backward()
        elif strategy == "prototype":
            support_z = F.normalize(model.encode(features[si]),dim=1)
            query_z = F.normalize(model.encode(features[qi]),dim=1)
            classes = task["classes"]
            centroids = F.normalize(torch.stack([support_z[y[si] == c].mean(0) for c in classes]),dim=1)
            local_truth = torch.as_tensor([classes.index(int(v)) for v in y[qi]],dtype=torch.long)
            loss = F.cross_entropy(-torch.cdist(query_z,centroids).square(),local_truth)
            loss.backward()
        else:
            parameters = dict(model.named_parameters())
            support_loss = F.cross_entropy(functional_call(model,parameters,(features[si],)),y[si])
            gradients = torch.autograd.grad(support_loss,tuple(parameters.values()))
            # First-order MAML deliberately omits derivatives through this step.
            adapted = {name:(value-inner_learning_rate*gradient).detach().requires_grad_(True)
                       for (name,value),gradient in zip(parameters.items(),gradients)}
            loss = F.cross_entropy(functional_call(model,adapted,(features[qi],)),y[qi])
            query_gradients = torch.autograd.grad(loss,tuple(adapted.values()))
            for value,gradient in zip(parameters.values(),query_gradients):
                value.grad = gradient.detach().clone()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite episodic loss")
        optimizer.step()
        losses.append({"episode":task["episode"],"source_meta_query_cross_entropy":float(loss.detach())})
        manifests.append({"episode":task["episode"],"classes":task["classes"],
            "support_ids":np.asarray(identifiers)[si].tolist(),"query_ids":np.asarray(identifiers)[qi].tolist()})
    model.eval()
    if state_fingerprint(initial) != before_initial:
        raise AssertionError("Meta training changed the shared supervised initializer")
    return model, {"strategy":strategy,"first_order":strategy == "fomaml",
                   "policy":policy,"episodes":manifests,"losses":losses}


def model_scores(model, features):
    model.eval()
    with torch.no_grad():
        return model(features).cpu().numpy()


def prototype_context(model, source_x, source_y, query_x):
    """Cache source centroids and pointwise query embeddings, never query stats."""
    model.eval()
    with torch.no_grad():
        source_z = F.normalize(model.encode(source_x),dim=1)
        query_z = F.normalize(model.encode(query_x),dim=1)
        centroids = F.normalize(torch.stack([source_z[np.asarray(source_y) == c].mean(0)
                              for c in range(model.head.out_features)]),dim=1)
    return {"source_centroids":centroids,"query_embeddings":query_z}


def cached_prototype_predictions(model, context, support_x, support_y, blend):
    centroids = context["source_centroids"]
    if blend:
        with torch.no_grad():
            support_z = F.normalize(model.encode(support_x),dim=1)
            target = F.normalize(torch.stack([support_z[np.asarray(support_y) == c].mean(0)
                                  for c in range(model.head.out_features)]),dim=1)
            centroids = F.normalize((1-blend)*centroids+blend*target,dim=1)
    return torch.argmin(torch.cdist(context["query_embeddings"],centroids).square(),dim=1).cpu().numpy()


def finetune_predictions(initial, support_x, support_y, query_x, candidates):
    """Return all specified SGD path checkpoints without any query labels."""
    if not candidates:
        return {}
    n_classes = initial.head.out_features
    labels = _checked_labels(support_y,len(support_x),n_classes)
    if set(labels.tolist()) != set(range(n_classes)):
        raise ValueError("Fine-tuning needs every known class in support")
    before = state_fingerprint(initial)
    outputs = {}
    grouped = defaultdict(list)
    for candidate in candidates:
        if candidate["strategy"] not in {"head_ft","full_ft","fomaml_ft"}:
            raise ValueError("Expected a gradient adaptation strategy")
        if candidate["steps"] < 1 or candidate["learning_rate"] <= 0:
            raise ValueError("Fine-tuning needs positive steps and learning rate")
        grouped[(candidate["strategy"],candidate["learning_rate"])].append(candidate)
    for (strategy, rate), configs in grouped.items():
        fitted = deepcopy(initial)
        if strategy == "head_ft":
            for name,value in fitted.named_parameters():
                value.requires_grad_(name.startswith("head."))
        parameters = [p for p in fitted.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters,lr=rate)
        selected_steps = {c["steps"]:c for c in configs}
        fitted.train()
        for step in range(1,max(selected_steps)+1):
            fitted.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(fitted(support_x),labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite target support loss")
            loss.backward(); optimizer.step()
            if step in selected_steps:
                outputs[selected_steps[step]["id"]] = np.argmax(model_scores(fitted,query_x),axis=1)
    if state_fingerprint(initial) != before:
        raise AssertionError("Fine-tuning mutated the frozen source initializer")
    return outputs


def all_strategy_predictions(models, source_x, source_y, support_x, support_y, query_x, candidates,
                             *, proto_context=None, zero_predictions=None):
    """Predict from numerical tensors; neither IDs nor query truth are inputs."""
    result = dict(zero_predictions) if zero_predictions is not None else {"zero_head_ft":np.argmax(model_scores(models["supervised"],query_x),axis=1),
              "zero_full_ft":np.argmax(model_scores(models["supervised"],query_x),axis=1),
              "zero_fomaml_ft":np.argmax(model_scores(models["fomaml"],query_x),axis=1),
              "zero_prototype":np.argmax(prototype_scores(models["prototype"],source_x,source_y,query_x),axis=1)}
    for strategy in ("head_ft","full_ft","fomaml_ft"):
        model = models["fomaml" if strategy == "fomaml_ft" else "supervised"]
        selected = [c for c in candidates if c["strategy"] == strategy]
        result.update(finetune_predictions(model,support_x,support_y,query_x,selected))
    for candidate in (c for c in candidates if c["strategy"] == "prototype"):
        if proto_context is not None:
            result[candidate["id"]] = cached_prototype_predictions(models["prototype"],proto_context,support_x,support_y,candidate["blend"])
        else:
            scores = prototype_scores(models["prototype"],source_x,source_y,query_x,
                                      support_x=support_x,support_y=support_y,blend=candidate["blend"])
            result[candidate["id"]] = np.argmax(scores,axis=1)
    return result


def train_models(source, classes, protocol, seed, directory, *, resume=False):
    """Persist only source-trained CPU state dictionaries and traceable losses."""
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    metadata_path = directory/"training.json"
    x = snv_tensor(source["X_alpha"]); y = encoded(source["y"],classes)
    if metadata_path.exists() and resume:
        metadata = json.loads(metadata_path.read_text())
        if metadata["source_ids"] != source["record_ids"].tolist():
            raise ValueError("Cached neural source cohort changed")
        models = {}
        for name in ("supervised","prototype","fomaml"):
            path = directory/(name+".pt")
            if digest(path) != metadata["artifacts"][name]["sha256"]:
                raise ValueError("Cached neural weights changed")
            model = SmallResNet1D(len(classes))
            model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)); model.eval()
            if state_fingerprint(model) != metadata["artifacts"][name]["fingerprint"]:
                raise ValueError("Cached neural state fingerprint differs")
            models[name] = model
        return models, x, y, metadata
    config = protocol["training"]
    supervised, losses = supervised_fit(x,y,seed=seed,n_classes=len(classes),
        epochs=config["epochs"],batch_size=config["batch_size"],learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"])
    initializer_fingerprint = state_fingerprint(supervised)
    models = {}; meta_records = {}
    for name in ("supervised","prototype","fomaml"):
        model, record = meta_fit(supervised,x,y,source["record_ids"],strategy=name,seed=seed+100,
            episodes=config["meta_episodes"],learning_rate=config["meta_learning_rate"],
            way=config["meta_way"],support=config["meta_support"],query=config["meta_query"],
            inner_learning_rate=config["maml_inner_lr"])
        models[name] = model; meta_records[name] = record
    if any(meta_records[name]["episodes"] != meta_records["supervised"]["episodes"]
           for name in ("prototype","fomaml")):
        raise AssertionError("Strategies did not share the same source episodic records")
    artifacts = {}
    for name,model in models.items():
        path = directory/(name+".pt")
        torch.save(model.state_dict(),path)
        artifacts[name] = {"path":str(path),"sha256":digest(path),"fingerprint":state_fingerprint(model)}
    metadata = {"source_ids":source["record_ids"].tolist(),"seed":seed,"classes":classes.tolist(),
        "supervised_loss":losses,"meta_training":meta_records,"artifacts":artifacts,
        "parameter_count":sum(p.numel() for p in supervised.parameters()),
        "supervised_initializer_fingerprint":initializer_fingerprint,
        "source_optimization_budget":{"shared_initializer_epochs":config["epochs"],
            "extra_source_updates_per_strategy":config["meta_episodes"],
            "same_source_task_ids":True,"equal_FLOPs":False},
        "scope":"paper-inspired classification, source-only pseudo episodes, not real acquisition batches",
        "preprocessing":"per-spectrum SNV ddof1; no fitted query statistics",
        "architecture":"compact residual1D16/32,GroupNorm,adaptive8,embedding32; not exact author backbone"}
    write_json(metadata_path,metadata)
    return models, x, y, metadata


def _means(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["strategy"],row["candidate_id"],row["stage"])].append(row)
    result = {}
    for key,cases in grouped.items():
        result[key] = {name:float(np.mean([r[name] for r in cases])) for name in METRICS}
        result[key].update({"matched_"+name:float(np.mean([r["matched_"+name] for r in cases])) for name in METRICS})
        result[key]["evaluations"] = len(cases)
    return result


def choose_strategies(rows, candidates, selection):
    means = _means(rows); output = {}
    for strategy in STRATEGIES:
        configs = [c for c in candidates if c["strategy"] == strategy]
        ranked = sorted(configs,key=lambda c:(-means[(strategy,c["id"],"selection")]["accuracy"],
            c.get("blend",1.),c.get("steps",0),c.get("learning_rate",0.),c["id"]))
        proposed = ranked[0]
        def passes(config,stage):
            values = means[(strategy,config["id"],stage)]
            return (values["accuracy"]-values["matched_accuracy"] > selection["minimum_accuracy_gain"]
                and values["macro_f1"]-values["matched_macro_f1"] >= -selection["max_macro_f1_loss"]
                and values["balanced_accuracy"]-values["matched_balanced_accuracy"] >= -selection["max_balanced_accuracy_loss"])
        eligible = [c for c in ranked if passes(c,"selection")]
        proposed = eligible[0] if eligible else proposed
        selected = proposed if eligible and passes(proposed,"gate") else None
        output[strategy] = {"selected":selected,"diagnostic":proposed,
            "fallback_to_zero":selected is None,
            "selection":means[(strategy,proposed["id"],"selection")],
            "gate":means[(strategy,proposed["id"],"gate")],
            "selected_using_final_query":False}
    return output


def _subset(training, indices):
    return {"X_alpha":training["X_alpha"][indices],"y":training["y"][indices],
            "record_ids":training["record_ids"][indices]}


def _provenance(input_path, plan_path, protocol_path):
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__),root/"scripts/run_paper_fewshot_pilot.py",
             Path(__file__).with_name("calibration_exploration.py"),Path(__file__).with_name("within_batch_holdout.py")]
    return {"input_sha256":digest(input_path),"plan_sha256":digest(plan_path),
        "protocol_sha256":digest(protocol_path),"grid_sha256":object_digest(candidate_grid()),
        "source_sha256":{str(p.relative_to(root)):digest(p) for p in files},
        "python":platform.python_version(),"versions":{n:importlib.metadata.version(n) for n in ["torch","numpy","scikit-learn"]}}


def validate_protocol(protocol):
    expected_architecture = {"stem_channels":16,"stem_kernel":7,"stem_stride":2,
        "residual_channels":[16,32],"residual_kernel":3,"adaptive_pool_length":8,
        "embedding_dim":32,"classes":7,"dropout":0}
    if any(protocol["architecture"].get(k) != v for k,v in expected_architecture.items()):
        raise ValueError("Protocol architecture differs from implemented compact backbone")
    if protocol["preprocessing"]["mode"] != "per-row SNV ddof1":
        raise ValueError("Protocol preprocessing differs from implemented SNV")
    for key in ("head_finetune","full_finetune","fomaml_finetune"):
        if protocol["adaptation"][key] != {"learning_rates":[.0001,.001,.01],"steps":[1,5,10]}:
            raise ValueError("Protocol fine-tuning grid differs from implementation")
    if protocol["adaptation"]["classification_prototypes"]["source_support_blend"] != [.1,.25,.5,1]:
        raise ValueError("Protocol prototype grid differs from implementation")
    if not 0 < protocol["development"]["selection_episodes"] < protocol["development"]["episodes"]:
        raise ValueError("Pilot needs separate source-only selection and gate tasks")
    if not protocol["training"]["supervised_episode_continuation"]:
        raise ValueError("Matched extra source updates are required")
    if protocol.get("candidate_grid_sha256") != object_digest(candidate_grid()):
        raise ValueError("Protocol must lock the exact31candidate grid")


def run_pilot(input_path,plan_path,protocol_path,output,*,resume=False):
    output = Path(output); protocol = json.loads(Path(protocol_path).read_text())
    validate_protocol(protocol)
    candidates = candidate_grid(); provenance = _provenance(input_path,plan_path,protocol_path)
    expected_grid = protocol.get("candidate_grid_sha256")
    if expected_grid is not None and expected_grid != provenance["grid_sha256"]:
        raise ValueError("Neural candidate grid disagrees with recorded protocol")
    if output.exists() and any(output.iterdir()):
        if not resume or not (output/"run_manifest.json").is_file():
            raise FileExistsError("Use a fresh pilot output or verified --resume")
        if json.loads((output/"run_manifest.json").read_text()) != provenance:
            raise ValueError("Neural resume refused: input/code/protocol/runtime changed")
        if (output/"status.json").exists() and json.loads((output/"status.json").read_text())["status"] == "completed":
            return json.loads((output/"summary.json").read_text())
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"run_manifest.json",provenance); write_json(output/"candidate_grid.json",candidates)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    with np.load(input_path,allow_pickle=False) as d:arrays = {k:d[k] for k in d.files}
    plan = json.loads(Path(plan_path).read_text())
    roles,truth,preparation = prepare_holdout(arrays,plan,
        frequency_min=protocol["frequency"]["lower_inclusive"],frequency_max=protocol["frequency"]["upper_inclusive"])
    if (str(np.asarray(arrays["dataset_version"])) != protocol["dataset_version"]
            or preparation["feature_count"] != protocol["frequency"]["expected_points"]):
        raise ValueError("Neural input version/frequency differs from protocol")
    expected_counts = {"training":protocol["final"]["training_samples"],
                       "calibration":protocol["final"]["calibration_pool_samples"],
                       "prediction":protocol["final"]["query_samples"]}
    if {k:len(v["record_ids"]) for k,v in roles.items()} != expected_counts:
        raise ValueError("Neural role counts differ from recorded protocol")
    write_json(output/"preparation.json",preparation)
    training = roles["training"]; classes = np.unique(training["y"])
    query_ids = roles["prediction"]["record_ids"].tolist()
    rng = np.random.default_rng(protocol["final"]["random_query_seed"])
    random_ids = rng.choice(np.asarray(sorted(query_ids)),protocol["final"]["random_query_examples"],replace=False).tolist()
    write_json(output/"random_query_ids.json",random_ids)
    dev = protocol["development"]
    partitions = development_episodes(training,dev["episodes"],dev["split_seed"])
    all_rows, development_records = [], []
    started = time.monotonic()
    for episode,partition in enumerate(partitions):
        source = _subset(training,partition["source"])
        pool = _subset(training,partition["support_pool"])
        query = _subset(training,partition["query"])
        pool_x,query_x = snv_tensor(pool["X_alpha"]),snv_tensor(query["X_alpha"])
        pool_y,query_y = encoded(pool["y"],classes),encoded(query["y"],classes)
        draws = nested_draws(pool["y"],classes,[1],dev["support_draws"],dev["support_seed"]+episode)
        for seed in protocol["seeds"]:
            directory = output/"development"/f"episode_{episode:02d}"/f"seed_{seed}"
            metrics_path = directory/"metrics.json"
            if resume and metrics_path.exists():
                record = json.loads((directory/"record.json").read_text())
                if record["metrics_sha256"] != digest(metrics_path):
                    raise ValueError("Cached neural development metrics changed")
                all_rows.extend(json.loads(metrics_path.read_text())); development_records.append(record)
                continue
            models,source_x,source_y,model_meta = train_models(source,classes,protocol,seed,directory,resume=resume)
            hashes = {name:state_fingerprint(model) for name,model in models.items()}
            pc = prototype_context(models["prototype"],source_x,source_y,query_x)
            zero_predictions = all_strategy_predictions(models,source_x,source_y,pool_x[:0],pool_y[:0],query_x,[])
            rows = []
            for draw in draws:
                ids = draw["indices"]
                predictions = all_strategy_predictions(models,source_x,source_y,pool_x[ids],pool_y[ids],query_x,candidates,
                                                         proto_context=pc,zero_predictions=zero_predictions)
                zeros = {strategy:fast_metrics(query_y,predictions["zero_"+strategy],len(classes)) for strategy in STRATEGIES}
                for candidate in candidates:
                    strategy = candidate["strategy"]
                    metrics = fast_metrics(query_y,predictions[candidate["id"]],len(classes))
                    rows.append({"episode":episode,"seed":seed,"repeat":draw["repeat"],
                        "stage":"selection" if episode < dev["selection_episodes"] else "gate",
                        "strategy":strategy,"candidate_id":candidate["id"],**metrics,
                        **{"matched_"+name:zeros[strategy][name] for name in METRICS}})
            if hashes != {name:state_fingerprint(model) for name,model in models.items()}:
                raise AssertionError("Target development adaptation changed source weights")
            record = {"episode":episode,"seed":seed,
                "partition":{name:training["record_ids"][indices].tolist() for name,indices in partition.items()},
                "support_draws":[{"repeat":d["repeat"],"ids":pool["record_ids"][d["indices"]].tolist()} for d in draws],
                "source_fingerprints_unchanged":hashes,"evaluations":len(rows)}
            write_json(metrics_path,rows); record["metrics_sha256"] = digest(metrics_path)
            write_json(directory/"record.json",record)
            all_rows.extend(rows); development_records.append(record)
            print(json.dumps({"stage":"development","episode":episode+1,"seed":seed,"evaluations":len(rows)}),flush=True)
    choices = choose_strategies(all_rows,candidates,protocol["selection"])
    locked = {"choices":choices,"source_only_development":True,
              "provenance_sha256":object_digest(provenance),"development_evaluations":len(all_rows)}
    lock_path = output/"selection_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != locked:
        raise ValueError("Cached neural selection lock disagrees")
    write_json(output/"all_development_metrics.json",all_rows); write_json(lock_path,locked)
    lock_sha = digest(lock_path)
    print(json.dumps({"stage":"choices_locked","sha256":lock_sha}),flush=True)
    final_rows,random_rows = [], []
    cal_x = snv_tensor(roles["calibration"]["X_alpha"])
    cal_y = encoded(roles["calibration"]["y"],classes)
    query_x = snv_tensor(roles["prediction"]["X_alpha"])
    draws = nested_draws(roles["calibration"]["y"],classes,[1],protocol["final"]["support_draws"],protocol["final"]["support_seed"])
    # Choices are immutable before final query truth is joined for scoring.
    query_y = encoded([truth[v] for v in query_ids],classes)
    final_meta = []
    for seed in protocol["seeds"]:
        directory = output/"final"/f"seed_{seed}"
        models,source_x,source_y,model_meta = train_models(training,classes,protocol,seed,directory,resume=resume)
        hashes = {name:state_fingerprint(model) for name,model in models.items()}
        pc = prototype_context(models["prototype"],source_x,source_y,query_x)
        zero_predictions = all_strategy_predictions(models,source_x,source_y,cal_x[:0],cal_y[:0],query_x,[])
        selected_configs = {c["id"]:c for value in choices.values() for c in [value["selected"],value["diagnostic"]] if c is not None}
        predictions_path = directory/"predictions.csv.gz"
        with gzip.open(predictions_path,"wt",encoding="utf-8",newline="") as handle:
            fields = ["seed","repeat","strategy","method","candidate_id","record_id","y_true","y_predicted","is_correct"]
            writer = csv.DictWriter(handle,fieldnames=fields,lineterminator="\n"); writer.writeheader()
            for draw in draws:
                if digest(lock_path) != lock_sha:raise ValueError("Neural lock changed during final evaluation")
                ids = draw["indices"]
                predictions = all_strategy_predictions(models,source_x,source_y,cal_x[ids],cal_y[ids],query_x,list(selected_configs.values()),
                                                         proto_context=pc,zero_predictions=zero_predictions)
                for strategy in STRATEGIES:
                    zero = predictions["zero_"+strategy]
                    zero_metrics = fast_metrics(query_y,zero,len(classes))
                    for method in ("zero","selected","diagnostic"):
                        config = None if method == "zero" else choices[strategy][method]
                        key = "zero_"+strategy if config is None else config["id"]
                        predicted = predictions[key]
                        metrics = fast_metrics(query_y,predicted,len(classes),detailed=True)
                        final_rows.append({"seed":seed,"repeat":draw["repeat"],"strategy":strategy,
                            "method":method,"candidate_id":key,**metrics,
                            **{name+"_gain":metrics[name]-zero_metrics[name] for name in METRICS},
                            "corrected_errors":int(np.count_nonzero((zero != query_y)&(predicted == query_y))),
                            "new_errors":int(np.count_nonzero((zero == query_y)&(predicted != query_y)))})
                        for identifier,actual,prediction in zip(query_ids,query_y,predicted):
                            row = {"seed":seed,"repeat":draw["repeat"],"strategy":strategy,"method":method,
                                "candidate_id":key,"record_id":identifier,"y_true":str(classes[actual]),
                                "y_predicted":str(classes[prediction]),"is_correct":int(actual == prediction)}
                            writer.writerow(row)
                            if identifier in random_ids and draw["repeat"] == 0:random_rows.append(row)
        after = {name:state_fingerprint(model) for name,model in models.items()}
        if hashes != after:raise AssertionError("Final adaptation changed source neural states")
        final_meta.append({"seed":seed,"source_ids":training["record_ids"].tolist(),
            "query_ids":query_ids,"fingerprints_before":hashes,"fingerprints_after":after,
            "prediction_sha256":digest(predictions_path),
            "support_draws":[{"repeat":d["repeat"],"ids":roles["calibration"]["record_ids"][d["indices"]].tolist()} for d in draws]})
        print(json.dumps({"stage":"final_seed_completed","seed":seed}),flush=True)
    summaries = []
    for strategy in STRATEGIES:
        for method in ("zero","selected","diagnostic"):
            rows = [r for r in final_rows if r["strategy"] == strategy and r["method"] == method]
            summary = {"strategy":strategy,"method":method,"evaluations":len(rows),"seeds":protocol["seeds"],
                       "candidate_id":rows[0]["candidate_id"]}
            for metric in METRICS:
                values = [r[metric] for r in rows]
                summary.update({metric+"_"+name:float(operation(values)) for name,operation in
                                [("mean",np.mean),("sd",np.std),("min",np.min),("max",np.max)]})
                summary[metric+"_gain_mean"] = float(np.mean([r[metric+"_gain"] for r in rows]))
            summary.update(positive_draws=sum(r["accuracy_gain"]>0 for r in rows),
                           unchanged_draws=sum(r["accuracy_gain"]==0 for r in rows),
                           negative_draws=sum(r["accuracy_gain"]<0 for r in rows))
            summaries.append(summary)
    write_json(output/"final_metrics.json",final_rows); write_json(output/"summary.json",summaries)
    write_json(output/"final_metadata.json",final_meta)
    with (output/"random_predictions.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(random_rows[0]),lineterminator="\n")
        writer.writeheader(); writer.writerows(random_rows)
    write_json(output/"status.json",{"status":"completed","seconds":time.monotonic()-started,
        "development_evaluations":len(all_rows),"final_evaluations":len(final_rows),
        "prediction_rows":sum(r["total"] for r in final_rows),"selection_lock_sha256":lock_sha,
        "source_training_models":(dev["episodes"]+1)*len(protocol["seeds"]),
        "supervised_continuation_models":(dev["episodes"]+1)*len(protocol["seeds"]),
        "meta_training_models":2*(dev["episodes"]+1)*len(protocol["seeds"]),
        "saved_source_model_artifacts":3*(dev["episodes"]+1)*len(protocol["seeds"]),
        "paper_exact_reproduction":False,"cross_batch_validation":False,
        "query_labels_used_to_fit":False,"query_statistics_fitted":False})
    return summaries
