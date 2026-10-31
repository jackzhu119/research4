"""Executable provenance boundaries for the new, explicitly guarded workflows.

These guards detect named violations within instrumented entry points. They do
not establish biological independence or intercept arbitrary uninstrumented code.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import threading

import joblib
import numpy as np


class LeakageError(ValueError):
    """A declared data-access, lineage, cache or inference boundary was violated."""


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else str(v))


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _ids(values):
    result = tuple(str(v) for v in values)
    if not result or any(not v.strip() for v in result) or len(set(result)) != len(result):
        raise LeakageError("Sample IDs must be nonempty, unique and explicitly supplied")
    return result


def array_fingerprint(values):
    value = np.asarray(values)
    if value.dtype.kind == "O":
        payload = _json(value.tolist()).encode()
    else:
        payload = np.ascontiguousarray(value).tobytes()
    h = hashlib.sha256(_json({"shape":value.shape, "dtype":str(value.dtype)}).encode())
    h.update(payload)
    return h.hexdigest()


def source_fingerprint(X, y=None):
    """Fingerprint actual ordered training values; supervision is included."""
    return _digest({"X":array_fingerprint(X), "y":None if y is None else array_fingerprint(y)})


@dataclass(frozen=True)
class LineageRecord:
    record_id: str
    physical_sample_id: str | None = None
    parent_sample_id: str | None = None
    independence_evidence: str = "unknown"
    dataset_id: str = ""
    modality: str = ""
    label: str | None = None
    preparation_id: str | None = None
    batch_id: str | None = None
    measurement_id: str | None = None
    source_file: str = ""
    source_hash: str = ""


def validate_lineage_partitions(records, partitions, *, allow_user_declared=True,
                                allow_unknown_exploratory=False):
    """Check ancestors across saved roles; unknown identity cannot receive PASS.

    A record may appear once per modality in the metadata. Its identity and label
    must agree. All IDs, including parent/preparation IDs, must be globally scoped.
    Different batches are deliberately not treated as different physical parents.
    """
    rows = [asdict(r) if is_dataclass(r) else dict(r) for r in records]
    if not rows or not partitions:
        raise LeakageError("Explicit lineage and nonempty role mapping required")
    by_record = {}
    seen_modality = set()
    for row in rows:
        rid = str(row.get("record_id", ""))
        if not rid.strip():
            raise LeakageError("Missing record ID")
        key = (rid, str(row.get("modality", "")))
        if key in seen_modality:
            raise LeakageError("Duplicate record/modality metadata")
        seen_modality.add(key)
        if rid in by_record:
            for field in ("physical_sample_id", "parent_sample_id", "dataset_id", "label"):
                if (row.get(field) or None) != (by_record[rid].get(field) or None):
                    raise LeakageError("Modalities disagree on physical identity or label")
        else:
            by_record[rid] = row
    # Multiple records that claim the same entity cannot silently change label.
    labels_by_entity = {}
    for row in by_record.values():
        entity, label = row.get("physical_sample_id"), row.get("label")
        if entity and label not in (None, ""):
            if entity in labels_by_entity and labels_by_entity[entity] != label:
                raise LeakageError("Conflicting labels for the same physical entity")
            labels_by_entity[entity] = label

    def ancestors(rid, trail=()):
        if rid in trail:
            raise LeakageError("Cyclic parent lineage")
        row = by_record[rid]
        result = {"record:"+rid}
        for field, prefix in (("physical_sample_id", "entity:"), ("parent_sample_id", "entity:"),
                              ("preparation_id", "preparation:"), ("measurement_id", "measurement:")):
            if row.get(field):
                result.add(prefix+str(row[field]))
        parent = row.get("parent_sample_id")
        if parent and str(parent) in by_record:
            result |= ancestors(str(parent), (*trail, rid))
        # Parent may name a physical ID rather than a record ID.
        if parent:
            for other, candidate in by_record.items():
                if other != rid and candidate.get("physical_sample_id") == parent:
                    result |= ancestors(other, (*trail, rid))
        return result

    role_ids, tokens = {}, {}
    for role, ids in partitions.items():
        role_ids[role] = _ids(ids)
        if set(role_ids[role]) - set(by_record):
            raise LeakageError("Partition contains records absent from lineage")
        tokens[role] = set().union(*(ancestors(rid) for rid in role_ids[role]))
    roles = list(role_ids)
    for i, left in enumerate(roles):
        for right in roles[i+1:]:
            shared = tokens[left] & tokens[right]
            if shared:
                raise LeakageError(f"Shared physical ancestor across {left}/{right}: {sorted(shared)[:5]}")
    used = set().union(*(set(v) for v in role_ids.values()))
    unknown, declared = [], []
    for rid in sorted(used):
        row = by_record[rid]
        evidence = str(row.get("independence_evidence") or "unknown").lower()
        if not row.get("physical_sample_id") or evidence in {"unknown", "", "not_provided", "filename_only"}:
            unknown.append(rid)
        elif "user_declared" in evidence or "user declaration" in evidence:
            declared.append(rid)
    blocked = bool(unknown and not allow_unknown_exploratory or declared and not allow_user_declared)
    return {"status":"BLOCKED" if blocked else "EXPLORATORY" if unknown or declared else "TECHNICAL_CHECK_PASS",
            "role_counts":{r:len(v) for r,v in role_ids.items()},
            "unknown_independence_record_ids":unknown, "user_declared_record_ids":declared,
            "ancestor_intersections_empty":True, "biological_independence_proved_by_this_check":False,
            "independent_confirmation_established":False}


def validate_fixed_partitions(partitions, *, expected_digest=None, expected_ids=None):
    normalized = {str(role):list(_ids(ids)) for role, ids in partitions.items()}
    flat = [rid for ids in normalized.values() for rid in ids]
    if len(flat) != len(set(flat)):
        raise LeakageError("One saved record occurs in multiple roles")
    if expected_ids is not None and set(flat) != set(_ids(expected_ids)):
        raise LeakageError("Saved partition coverage differs from declared population")
    digest = _digest(normalized)
    if expected_digest is not None and digest != expected_digest:
        raise LeakageError("Saved partition identity/order changed")
    return digest


def assert_modalities_aligned(expected_ids, modalities):
    expected = _ids(expected_ids)
    if not modalities:
        raise LeakageError("No modalities supplied")
    for name, identifiers in modalities.items():
        if _ids(identifiers) != expected:
            raise LeakageError("Missing, duplicated or misordered modality pairing: "+str(name))
    return True


class FitScope:
    """Wrap actual fit entry points; scope is never inferred from a model name."""
    def __init__(self, allowed_ids, split_id, dataset_version, log_path=None, *, code_version="unrecorded"):
        self.allowed_ids = _ids(allowed_ids)
        if not str(split_id).strip() or not str(dataset_version).strip():
            raise LeakageError("Explicit dataset version and split ID required")
        self.split_id, self.dataset_version = str(split_id), str(dataset_version)
        self.code_version = str(code_version)
        self.log_path = None if log_path is None else Path(log_path)
        self.entries = []

    def _record(self, event, **value):
        item = {"event":event, "utc":datetime.now(timezone.utc).isoformat(), "split_id":self.split_id,
                "dataset_version":self.dataset_version, "code_version":self.code_version, **value}
        self.entries.append(item)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            with self.log_path.open("a",encoding="utf-8") as handle:
                handle.write(_json(item)+"\n")
        return item

    @contextmanager
    def operation(self, component, sample_ids, X=None, y=None):
        ids = _ids(sample_ids)
        if set(ids)-set(self.allowed_ids):
            self._record("fit_rejected",component=str(component),fit_sample_ids=list(ids),reason="forbidden_sample_id")
            raise LeakageError("Fit contains samples outside current allowed training scope")
        if X is not None:
            if isinstance(X, dict):
                if not X or any(len(v) != len(ids) for v in X.values()):
                    raise LeakageError("Each source modality must match fit_sample_ids")
                input_fp = _digest({str(k):array_fingerprint(v) for k,v in sorted(X.items())})
                data_fp = _digest({"X":input_fp,"y":None if y is None else array_fingerprint(y)})
            else:
                matrix = np.asarray(X)
                if matrix.ndim < 2 or len(matrix) != len(ids) or matrix.dtype.kind not in "bifuc":
                    raise LeakageError("Numerical source rows must match fit_sample_ids; IDs are not features")
                data_fp = source_fingerprint(X,y)
        else:
            data_fp = None
        if y is not None and len(y) != len(ids):
            raise LeakageError("Source labels do not match explicit fit IDs")
        proof = {"fit_sample_ids":list(ids),"train_id_digest":_digest(list(ids)),
                 "dataset_version":self.dataset_version,"split_id":self.split_id,
                 "code_version":self.code_version,"data_fingerprint":data_fp}
        self._record("fit_start",component=str(component),**proof)
        try:
            yield proof
        except BaseException as exc:
            self._record("fit_failed",component=str(component),error_type=type(exc).__name__,**proof)
            raise
        self._record("fit_completed",component=str(component),**proof)

    def fit(self, estimator, X, y=None, *, sample_ids, component, **kwargs):
        with self.operation(component, sample_ids, X=X, y=y) as proof:
            if y is None:
                estimator.fit(X, **kwargs)
            else:
                estimator.fit(X,y,**kwargs)
            estimator._leakage_fit_provenance = dict(proof)
        return estimator


_PLS_INSTRUMENTATION_LOCK = threading.RLock()
_PLS_INSTRUMENTATION_ACTIVE = False


@contextmanager
def instrument_mcuve_pls(scope, source_X, source_ids):
    """Observe each legacy MC-UVE PLS fit without changing its mathematics.

    This bounded, single-process instrumentation covers the existing selection
    module's full-width MC-UVE coefficient route. It temporarily wraps functions,
    restores them even on failure, and fails on ambiguous numerical row binding.
    It is not a process-wide security sandbox or a CARS/aggregated-group adapter.
    Run independent fits in separate processes, not concurrent threads here.
    """
    from sklearn.cross_decomposition import PLSRegression
    from . import selection
    global _PLS_INSTRUMENTATION_ACTIVE
    X = np.asarray(source_X,dtype=float)
    ids = _ids(source_ids)
    if X.ndim != 2 or len(X)!=len(ids) or not np.isfinite(X).all():
        raise LeakageError("MC tracing requires finite source rows and matching IDs")
    if set(ids)-set(scope.allowed_ids):
        raise LeakageError("MC source outside current fit scope")
    lookup = {}
    for rid,row in zip(ids,X,strict=True):
        digest = array_fingerprint(row)
        if digest in lookup:
            raise LeakageError("Numerically duplicate MC rows need explicit unambiguous group binding")
        lookup[digest] = rid
    with _PLS_INSTRUMENTATION_LOCK:
        if _PLS_INSTRUMENTATION_ACTIVE:
            raise LeakageError("Nested MC instrumentation is not supported")
        _PLS_INSTRUMENTATION_ACTIVE = True
        original_coefficients = selection._coefficients
        original_fit = PLSRegression.fit
        active = {"ids":None,"Z":None}

        def coefficients(values,targets,components):
            current = np.asarray(values,dtype=float)
            if current.ndim!=2 or current.shape[1]!=X.shape[1]:
                raise LeakageError("MC trace only accepts full-width source coefficient rows")
            try:
                fitted_ids = [lookup[array_fingerprint(row)] for row in current]
            except KeyError as exc:
                raise LeakageError("Coefficient fit includes rows absent from its current training source") from exc
            _ids(fitted_ids)
            scale = current.std(axis=0)
            scale[scale<=np.finfo(float).eps] = 1
            previous = dict(active)
            active.update(ids=fitted_ids,Z=(current-current.mean(axis=0))/scale)
            try:
                with scope.operation("MC-UVE.coefficient_standardization",fitted_ids,X=current,y=targets):
                    return original_coefficients(values,targets,components)
            finally:
                active.update(previous)

        def traced_fit(estimator,values,targets,*args,**kwargs):
            if active["ids"] is None or not np.array_equal(np.asarray(values),active["Z"]):
                raise LeakageError("PLS fit cannot be bound to the active MC training subset")
            with scope.operation("MC-UVE.PLSRegression.fit",active["ids"],X=values,y=targets) as proof:
                result = original_fit(estimator,values,targets,*args,**kwargs)
                estimator._leakage_fit_provenance = dict(proof)
                return result
        selection._coefficients = coefficients
        PLSRegression.fit = traced_fit
        try:
            yield
        finally:
            selection._coefficients = original_coefficients
            PLSRegression.fit = original_fit
            _PLS_INSTRUMENTATION_ACTIVE = False


class CacheKey(str):
    def __new__(cls, identity):
        value = str.__new__(cls,_digest(identity))
        value.identity = dict(identity)
        return value


def guarded_cache_key(*, dataset_version, train_ids, split_id, preprocessing,
                      selector, code_version, seed, data_fingerprint=None, data_sha256=None):
    ids = _ids(train_ids)
    fingerprint = data_fingerprint or data_sha256
    if not fingerprint or not dataset_version or not split_id or not code_version:
        raise LeakageError("Cache identity requires data, dataset, split and code fingerprints")
    return CacheKey({"schema":"learned-cache-v1","dataset_version":str(dataset_version),
                     "train_id_digest":_digest(list(ids)),"split_id":str(split_id),
                     "preprocessing":preprocessing,"selector":selector,"code_version":str(code_version),
                     "seed":int(seed),"data_fingerprint":str(fingerprint)})


def _verify_proof(proof, key):
    if not isinstance(key,CacheKey):
        raise LeakageError("Reconstruct the full cache identity rather than trusting a filename")
    if not isinstance(proof,dict):
        raise LeakageError("Learned cache requires actual fit provenance")
    for field in ("train_id_digest","dataset_version","split_id","code_version","data_fingerprint"):
        if proof.get(field) != key.identity[field]:
            raise LeakageError("Cached fitted object belongs to another source/context: "+field)


def save_guarded_cache(path, payload, key, *, fit_provenance=None):
    proof = fit_provenance or getattr(payload,"_leakage_fit_provenance",None)
    _verify_proof(proof,key)
    own_proof = getattr(payload,"_leakage_fit_provenance",None)
    if own_proof is not None:
        _verify_proof(own_proof,key)
    path = Path(path)
    if path.exists():
        raise FileExistsError("Learned caches are immutable; use a new key/path")
    path.parent.mkdir(parents=True,exist_ok=True)
    buffer = io.BytesIO(); joblib.dump(payload,buffer,compress=3); blob = buffer.getvalue()
    joblib.dump({"schema":"guarded-learned-cache-v1","key":str(key),"identity":key.identity,
                 "fit_provenance":proof,"payload_sha256":hashlib.sha256(blob).hexdigest(),
                 "payload_bytes":blob},path,compress=0)


def load_guarded_cache(path, expected_key):
    value = joblib.load(path)
    if (not isinstance(expected_key,CacheKey) or value.get("schema") != "guarded-learned-cache-v1"
            or value.get("key") != str(expected_key) or value.get("identity") != expected_key.identity):
        raise LeakageError("Cross-fold or changed-parameter cache rejected")
    _verify_proof(value.get("fit_provenance"),expected_key)
    blob = value["payload_bytes"]
    if hashlib.sha256(blob).hexdigest() != value["payload_sha256"]:
        raise LeakageError("Learned cache payload bytes changed")
    payload = joblib.load(io.BytesIO(blob))
    own_proof = getattr(payload,"_leakage_fit_provenance",None)
    if own_proof is not None:
        _verify_proof(own_proof,expected_key)
        if own_proof != value["fit_provenance"]:
            raise LeakageError("Cached weight provenance disagrees with envelope")
    return payload


def audit_pointwise_inference(model, inputs, *, forward=None, batch_sizes=(1,3), atol=1e-5, rtol=1e-5):
    """Check induction/state invariance for Tensor or paired modality dictionaries.

    The inference callable accepts only features. Current input identity/labels
    never enter this API. A custom forward receives its batched feature object.
    """
    import torch
    model.eval()
    modules = list(model.modules())
    if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and not m.track_running_stats for m in modules):
        raise LeakageError("BatchNorm without frozen running statistics uses query batch statistics")
    state_before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    device = next(model.parameters(),torch.empty(0)).device
    def convert(v):
        return torch.as_tensor(v,dtype=torch.float32,device=device)
    if isinstance(inputs,dict) and not inputs:
        raise LeakageError("Inference needs at least one paired modality")
    values = {k:convert(v) for k,v in inputs.items()} if isinstance(inputs,dict) else convert(inputs)
    n = len(next(iter(values.values()))) if isinstance(values,dict) else len(values)
    if n < 1 or isinstance(values,dict) and (not values or any(len(v)!=n for v in values.values())):
        raise LeakageError("Inference needs paired nonempty feature rows")
    def batch(indices):
        return {k:v[indices] for k,v in values.items()} if isinstance(values,dict) else values[indices]
    fn = model if forward is None else forward
    def infer(indices):
        with torch.no_grad():
            result = fn(batch(indices))
        if not torch.is_tensor(result) or result.ndim != 2 or len(result)!=len(indices):
            raise LeakageError("Inference must return an aligned samples-by-classes tensor")
        output = result.detach().cpu().numpy()
        if not np.isfinite(output).all():
            raise LeakageError("Nonfinite inference output")
        return output
    reference = infer(np.arange(n))
    maximum, conditions = 0., 0
    for order in (np.arange(n),np.arange(n)[::-1].copy(),np.random.default_rng(0).permutation(n)):
        for size in batch_sizes:
            if int(size) < 1:
                raise LeakageError("Batch size must be positive")
            found = np.concatenate([infer(order[start:start+int(size)]) for start in range(0,n,int(size))])
            reordered = np.empty_like(found);reordered[order] = found
            maximum = max(maximum,float(np.max(np.abs(reordered-reference))))
            if not np.allclose(reordered,reference,atol=atol,rtol=rtol):
                raise LeakageError("Prediction depends on query order or batch composition")
            conditions += 1
    current = model.state_dict()
    if set(current) != set(state_before) or any(not torch.equal(v.detach().cpu(),state_before[k]) for k,v in current.items()):
        raise LeakageError("Inference mutated parameters or running statistics")
    if model.training or any(m.training for m in model.modules()):
        raise LeakageError("Inference left a module in training state")
    return {"status":"TECHNICAL_CHECK_PASS","samples_checked":n,"conditions":conditions,
            "maximum_numeric_difference":maximum,"atol":atol,"rtol":rtol,"eval_mode":True,
            "gradients_enabled_during_forward":False,"parameters_and_buffers_unchanged":True,
            "absolute_zero_leakage_proved":False}
