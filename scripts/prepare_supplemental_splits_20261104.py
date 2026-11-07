"""Prepare immutable supplemental role packs and exact-number XLSX derivatives.

No model fitting occurs here. Legacy jujube memberships are preserved; new
three-record groups have one source/support/query specimen and no new-class CV.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import zipfile
from xml.sax.saxutils import escape

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from herb_spectra.leakage_guard import array_fingerprint

ROLES = ("source", "support", "query")
LEGACY_ROLES = {"training": "source", "calibration": "support", "prediction": "query"}
MODALITIES = (("time", "时域"), ("frequency", "频域"), ("n", "折射率"), ("alpha", "吸收系数"))
QUANTITY = "absorption_coefficient_alpha;physical_length_unit_UNVERIFIED"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(repo, ref):
    path = (Path(repo) / ref["path"]).resolve()
    require(path.is_relative_to(Path(repo).resolve()) and path.is_file(), "Missing/outside source path")
    require(sha(path) == ref["sha256"], "Source artifact SHA changed: " + str(path))
    return path


def reference(path, repo=ROOT):
    path = Path(path)
    return {"path": str(path.resolve().relative_to(Path(repo).resolve())), "sha256": sha(path), "bytes": path.stat().st_size}


def jwrite(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def csvwrite(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def truthy(value):
    return str(value).lower() == "true"


def identity_fingerprint(axis, row):
    axis, row = np.asarray(axis, dtype="<f8").copy(), np.asarray(row, dtype="<f8").copy()
    require(axis.ndim == 1 and row.shape == axis.shape and np.isfinite(axis).all() and np.isfinite(row).all(),
            "Finite, aligned alpha identity input required")
    axis[axis == 0] = 0.
    row[row == 0] = 0.
    h = hashlib.sha256(b"axis-value-float64-v1\0")
    h.update(len(axis).to_bytes(8, "little"))
    h.update(axis.tobytes())
    h.update(row.tobytes())
    return h.hexdigest()


def identity_aliases(record):
    """One namespace union: a legacy ID can equal another row's record ID."""
    aliases = {str(record[k]) for k in ("record_id", "entity_id") if record.get(k)}
    legacy = record.get("legacy_match_ids", [])
    if isinstance(legacy, str):
        legacy = json.loads(legacy)
    require(isinstance(legacy, list), "Legacy aliases must be an explicit JSON list")
    aliases.update(str(v) for v in legacy if str(v))
    for key in ("parent_sample_id", "preparation_id"):
        if record.get(key):
            aliases.add(str(record[key]))
    return aliases


def assert_partition_identity(records, assignments):
    require(len(records) == len({r["record_id"] for r in records}), "Duplicate record IDs")
    require(set(assignments) == {r["record_id"] for r in records}, "Roles must cover every record exactly once")
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        parent[find(a)] = find(b)
    for row in records:
        keys = sorted(identity_aliases(row))
        require(keys, "Missing physical identity")
        for key in keys[1:]:
            union(keys[0], key)
    roles_by_component, labels_by_component = defaultdict(set), defaultdict(set)
    fingerprint_roles = defaultdict(set)
    for row in records:
        role = assignments[row["record_id"]]
        require(role in ROLES, "Unknown role")
        component = find(next(iter(identity_aliases(row))))
        roles_by_component[component].add(role)
        labels_by_component[component].add(row["label"])
        if row.get("alpha_fingerprint"):
            fingerprint_roles[row["alpha_fingerprint"]].add(role)
    require(all(len(v) == 1 for v in roles_by_component.values()), "Physical/legacy alias component spans roles")
    require(all(len(v) == 1 for v in labels_by_component.values()), "One identity has conflicting anonymous labels")
    require(all(len(v) == 1 for v in fingerprint_roles.values()), "Identical alpha inputs span roles; provenance review required")
    return {"status": "TECHNICAL_ROLE_IDENTITY_CHECK_PASS", "record_count": len(records),
            "entity_components": len(roles_by_component), "independence_basis": "USER_DECLARATION_NOT_LAB_AUDIT",
            "independent_confirmation_established": False}


def allocate_roles(records, seed):
    groups = defaultdict(list)
    for row in records:
        groups[row["label"]].append(row)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    result = {}
    for label in sorted(groups):
        ordered = sorted(groups[label], key=lambda r: (r["entity_id"], r["record_id"]))
        require(len({r["entity_id"] for r in ordered}) == len(ordered), "Repeated entities require canonical intake resolution")
        require(len(ordered) >= 3, "BLOCKED: fewer than three entities cannot fill three roles")
        shuffled = [ordered[i] for i in rng.permutation(len(ordered))]
        n_source, n_support = int(.6 * len(ordered)), max(1, int(.2 * len(ordered)))
        require(n_source >= 1 and len(ordered) - n_source - n_support >= 1, "Unsupported three-role allocation")
        for role, values in zip(ROLES, (shuffled[:n_source], shuffled[n_source:n_source+n_support], shuffled[n_source+n_support:])):
            result.update({r["record_id"]: role for r in values})
    assert_partition_identity(records, result)
    return result


def make_folds(records, assignments, seed, requested=3):
    source = [r for r in records if assignments[r["record_id"]] == "source"]
    by_class = defaultdict(list)
    for row in source:
        by_class[row["label"]].append(row["record_id"])
    count = min(int(requested), min(map(len, by_class.values())))
    require(count >= 2, "BLOCKED: source singleton cannot enter ordinary source CV")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    valid = [[] for _ in range(count)]
    for label in sorted(by_class):
        values = np.asarray(sorted(by_class[label]))
        pieces = np.array_split(values[rng.permutation(len(values))], count)
        for k, piece in enumerate(pieces):
            valid[k].extend(piece.tolist())
    all_ids = sorted(r["record_id"] for r in source)
    return [{"train_ids": [i for i in all_ids if i not in set(v)], "valid_ids": sorted(v)} for v in valid]


def uniform_shots(records, assignments, requested=(1, 3, 5)):
    by_class = defaultdict(list)
    for row in records:
        if assignments[row["record_id"]] == "support":
            by_class[row["label"]].append(row["record_id"])
    classes = sorted({r["label"] for r in records})
    available = [k for k in requested if all(len(by_class[c]) >= k for c in classes)]
    return available, {str(k): "BLOCKED_INSUFFICIENT_DISTINCT_SUPPORT" for k in requested if k not in available}


def make_draws(records, assignments, seed, repeats=10):
    available, blocked = uniform_shots(records, assignments)
    pools = defaultdict(list)
    for row in records:
        if assignments[row["record_id"]] == "support":
            pools[row["label"]].append(row["record_id"])
    draws = []
    for number in range(repeats):
        rng = np.random.Generator(np.random.PCG64(int(seed) + number))
        shuffled = {c: rng.permutation(sorted(pools[c])).tolist() for c in sorted(pools)}
        draws.append({"draw_id": number, "seed": int(seed)+number,
                      "shots": {str(k): sum((shuffled[c][:k] for c in sorted(shuffled)), []) for k in available}})
    return draws, available, blocked


def make_episodes(records, assignments, seed_base, available_shots):
    source = [r for r in records if assignments[r["record_id"]] == "source"]
    by_class = defaultdict(list)
    for row in source:
        by_class[row["label"]].append(row["record_id"])
    feasible = [k for k in available_shots if all(len(v) >= k + max(1, int(.2*len(v))) + 1 for v in by_class.values())]
    if not feasible:
        return []
    k_max = max(feasible)
    episodes = []
    for number in range(6):
        rng = np.random.Generator(np.random.PCG64(int(seed_base)+number))
        base, support, query, nested = [], [], [], {str(k): [] for k in feasible}
        for label in sorted(by_class):
            values = sorted(by_class[label])
            if len(values) == 3 and k_max == 1:
                permutation = list(itertools.permutations(values))[number]
                b, s, q = [permutation[0]], [permutation[1]], [permutation[2]]
            else:
                values = rng.permutation(values).tolist()
                n_query = max(1, int(.2 * len(values)))
                s, q, b = values[:k_max], values[k_max:k_max+n_query], values[k_max+n_query:]
            require(b and s and q, "Each pseudo role must contain each source class")
            base += b; support += s; query += q
            for k in feasible:
                nested[str(k)] += s[:k]
        require(not set(base)&set(support) and not set(base)&set(query) and not set(support)&set(query), "Pseudoroles overlap")
        require(set(base+support+query) == {r["record_id"] for r in source}, "Pseudo episode does not cover source")
        pseudo_assign = {i: role for role, ids in (("source", base), ("support", support), ("query", query)) for i in ids}
        assert_partition_identity(source, pseudo_assign)
        episodes.append({"episode_id": number, "seed": int(seed_base)+number, "model_seed": 42,
                         "stage": "selection" if number < 3 else "gate", "base_ids": sorted(base),
                         "support_ids": sorted(support), "query_ids": sorted(query), "shots": nested,
                         "scope": "SOURCE_ONLY_RESAMPLING_NOT_INDEPENDENT_GUARD_VALIDATION"})
    return episodes


def build_dataset(repo, species, incoming_metadata, incoming_registry, include_new=False):
    repo = Path(repo)
    old_path = repo / "data/multiplant_database" / species / "alpha.npz"
    old = load_npz(old_path)
    axis = np.asarray(old["axis"], dtype=np.float64)
    require(axis.ndim == 1 and np.isfinite(axis).all() and np.all(np.diff(axis) > 0), "Invalid canonical frequency axis")
    mask = (axis >= .1) & (axis <= 1.5)
    require(int(mask.sum()) == 612, "This run requires the registered 612-point engineering window")
    X = np.asarray(old["X"], dtype=np.float64)[:, mask]
    ids = old["record_ids"].astype(str).tolist()
    labels = (old["labels"] if species == "zaoren" else old["filename_groups"]).astype(str).tolist()
    require(X.shape == (len(ids), 612) and np.isfinite(X).all() and all(labels), "Invalid canonical alpha records")
    records = []
    for identifier, label, row in zip(ids, labels, X):
        fp = identity_fingerprint(axis[mask], row)
        records.append({"record_id": identifier, "entity_id": "MEAS-"+fp[:24], "label": label,
                        "legacy_match_ids": [identifier], "batch_id": "LEGACY_FILE_BATCH_"+label,
                        "source_id": "LEGACY_"+label, "species": species, "stratum": "legacy_exposed",
                        "alpha_fingerprint": fp, "input_source": str(old_path.relative_to(repo))})
    refs = [reference(old_path, repo)]
    if include_new:
        entries = [v for v in incoming_registry["datasets"] if v["species"] == species and v["modality"] == "alpha"]
        require(len(entries) == 1, "A unique incoming alpha axis family is required")
        entry = entries[0]
        path = checked(repo, {"path": entry["artifact"], "sha256": entry["sha256"]})
        incoming = load_npz(path)
        require(np.array_equal(incoming["axis"], axis), "New/legacy axes differ; no implicit resampling")
        positions = {str(v): k for k, v in enumerate(incoming["record_ids"])}
        additions = sorted([r for r in incoming_metadata if r["species"] == species and truthy(r["is_new_unmatched"]) and truthy(r["canonical_for_model"])], key=lambda r:r["sample_id"])
        require(len(additions) == 9 and sorted(Counter(r["task_group_id"] for r in additions).values()) == [3,3,3],
                "New receipt metadata changed; revise the protocol before adding observations")
        extra = []
        for row in additions:
            require(row["legacy_match_ids"] == "[]", "Novel row has historical ancestry")
            values = np.asarray(incoming["X"][positions[row["sample_id"]]], dtype=np.float64)[mask]
            fp = identity_fingerprint(axis[mask], values)
            require(row["entity_id"] == "MEAS-"+fp[:24], "Incoming identity differs from alpha payload")
            records.append({"record_id": row["sample_id"], "entity_id": row["entity_id"], "label": row["task_group_id"],
                            "legacy_match_ids": [], "batch_id": row["batch_id"], "source_id": row["source_id"],
                            "species": species, "stratum": "new_three_record_group", "alpha_fingerprint": fp,
                            "input_source": str(path.relative_to(repo))})
            extra.append(values)
        X = np.vstack([X, np.asarray(extra)])
        refs.append(reference(path, repo))
    require(len({r["record_id"] for r in records}) == len(records), "Canonical IDs are not unique")
    require(len({r["entity_id"] for r in records}) == len(records), "Duplicate entity in modeling task requires intake review")
    return {"X": X, "axis": axis[mask], "records": records, "input_refs": refs}


def write_task(repo, out, task, data, assignments, recipe):
    records, X, axis = data["records"], data["X"], data["axis"]
    integrity = assert_partition_identity(records, assignments)
    directory = out / task["id"]
    directory.mkdir()
    task = dict(task)
    task.update(quantity=QUANTITY, axis_sha256=array_fingerprint(axis), classes=sorted({r["label"] for r in records}),
                input_refs=data["input_refs"], quantity_axis_compatibility="ENGINEERING_ALPHA_SHAPE_ONLY_UNITS_UNVERIFIED",
                independence_evidence="USER_DECLARED_DISTINCT_SPECIMENS_FILE_BATCHES_NOT_LAB_VERIFIED")
    counts = {}
    truth_rows, manifest_rows = [], []
    for role in ROLES:
        indices = sorted([i for i, r in enumerate(records) if assignments[r["record_id"]] == role], key=lambda i:records[i]["record_id"])
        chosen = [records[i] for i in indices]
        counts[role] = len(chosen)
        payload = {"X": X[indices], "axis": axis, "record_ids": np.asarray([r["record_id"] for r in chosen]),
                   "entity_ids": np.asarray([r["entity_id"] for r in chosen]), "batch_ids": np.asarray([r["batch_id"] for r in chosen]),
                   "legacy_match_ids": np.asarray([json.dumps(r["legacy_match_ids"], separators=(",", ":")) for r in chosen]),
                   "strata": np.asarray([r["stratum"] for r in chosen]), "source_ids": np.asarray([r["source_id"] for r in chosen])}
        if role != "query":
            payload["y"] = np.asarray([r["label"] for r in chosen])
        path = directory / (role+".npz")
        np.savez_compressed(path, **payload)
        task[role+"_pack"] = reference(path, repo)
        for row in chosen:
            manifest_rows.append({"task_id":task["id"],"record_id":row["record_id"],"entity_id":row["entity_id"],
                                  "species":row["species"],"label":row["label"],"role":role,"stratum":row["stratum"],
                                  "legacy_match_ids":json.dumps(row["legacy_match_ids"],separators=(",",":")),
                                  "input_source":row["input_source"],"alpha_fingerprint":row["alpha_fingerprint"]})
            if role == "query":
                truth_rows.append({"record_id":row["record_id"],"y":row["label"],"entity_id":row["entity_id"],"stratum":row["stratum"]})
    if task.get("expected_roles"):
        require(counts == task["expected_roles"], "Expected role counts changed")
    require(len(task["classes"]) == task["classes_expected"], "Class vocabulary changed")
    task["counts"] = counts
    task["strata"] = {s: [r["record_id"] for r in records if assignments[r["record_id"]] == "query" and r["stratum"] == s]
                      for s in sorted({r["stratum"] for r in records})}
    truth = directory / "evaluation" / "truth.csv"
    csvwrite(truth, truth_rows, ["record_id", "y", "entity_id", "stratum"])
    task["truth"] = reference(truth, repo)
    task["inner_folds"] = [] if task.get("inherit_selection_from") else make_folds(records, assignments, recipe["split_seed"])
    draws, shots, unavailable = make_draws(records, assignments, recipe["calibration"]["support_draw_seed"], recipe["calibration"]["support_draws"])
    task.update(calibration_draws=draws, shots_available=shots, shots_blocked=unavailable)
    if task.get("inherit_selection_from"):
        task["source_calibration_episodes"] = []
        task["calibration"] = {"conservative_policy":"global_no_op", "reason":task["conservative_fallback_reason"]}
    else:
        task["source_calibration_episodes"] = make_episodes(records, assignments, recipe["calibration"]["conservative"]["source_episode_seed_base"], shots)
        task["calibration"] = {"conservative_policy":"source_selection_then_gate" if task["source_calibration_episodes"] else "global_no_op",
                               "reason":"" if task["source_calibration_episodes"] else "INSUFFICIENT_SOURCE_META_ROLE_COUNTS"}
    csvwrite(directory/"split_manifest.csv", manifest_rows, list(manifest_rows[0]))
    jwrite(directory/"identity_integrity.json", integrity)
    return task, manifest_rows


def prepare(repo, recipe_path, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    require(output.is_relative_to(repo/"data/splits") and not output.exists(), "Use a fresh role output directory")
    recipe = json.loads(Path(recipe_path).read_text())
    metadata_path = repo/recipe["intake_metadata"]
    registry_path = repo/recipe["intake_manifest"]
    registry = json.loads(registry_path.read_text())
    require(registry["status"] == "COMPLETE" and sha(metadata_path) == registry["sample_metadata_sha256"], "Intake v2 is not frozen/complete")
    metadata = list(csv.DictReader(metadata_path.open(encoding="utf-8-sig")))
    output.mkdir(parents=True)
    old_manifest_path = repo/recipe["legacy_zaoren_split"]
    old_roles = {r["record_id"]: LEGACY_ROLES[r["role"]] for r in csv.DictReader(old_manifest_path.open(encoding="utf-8-sig"))}
    require(Counter(old_roles.values()) == {"source":513,"support":171,"query":171}, "Historical roles changed")
    tasks, all_rows, role_lookup, ready_refs = [], [], {}, []
    for template in recipe["tasks"]:
        expanded = template["id"] == "zaoren_expanded10"
        data = build_dataset(repo, template["species"], metadata, registry, include_new=expanded)
        if template["species"] == "zaoren":
            assignments = {r["record_id"]:old_roles[r["record_id"]] for r in data["records"] if r["stratum"] == "legacy_exposed"}
            if expanded:
                added = [r for r in data["records"] if r["stratum"] != "legacy_exposed"]
                assignments.update(allocate_roles(added, recipe["split_seed"]))
        else:
            assignments = allocate_roles(data["records"], recipe["split_seed"])
        task, rows = write_task(repo, output, template, data, assignments, recipe)
        tasks.append(task);all_rows += rows
        for row in data["records"]:
            key = (row["species"], row["record_id"])
            if key in role_lookup:
                require(role_lookup[key] == assignments[row["record_id"]], "A/B legacy roles differ")
            role_lookup[key] = assignments[row["record_id"]]
        ready_refs += [task[k] for k in ("source_pack","support_pack","query_pack","truth")]
    fields = list(all_rows[0])
    csvwrite(output/"all_tasks_split_manifest.csv", all_rows, fields)
    incoming_roles = []
    for row in metadata:
        hits = json.loads(row["legacy_match_ids"])
        require(len(hits) <= 1, "Multiple historical parents require explicit resolution")
        canonical = hits[0] if hits else row["sample_id"]
        key = (row["species"], canonical)
        require(key in role_lookup, "Incoming record is absent from registered modeling tasks")
        incoming_roles.append({"sample_id":row["sample_id"],"entity_id":row["entity_id"],"species":row["species"],
                               "source_id":row["source_id"],"canonical_model_record_id":canonical,"role":role_lookup[key],
                               "legacy_match_basis":row["legacy_match_basis"],"new_independent_specimen_claim":row["is_new_unmatched"]})
    csvwrite(output/"incoming_role_map.csv", incoming_roles, list(incoming_roles[0]))
    preparation = {"status":"MODEL_PACKS_READY", "created_at_utc":datetime.now(timezone.utc).isoformat(),
                   "recipe":reference(recipe_path,repo), "intake_manifest":reference(registry_path,repo),
                   "intake_sources":reference(repo/recipe["intake_sources"],repo), "intake_metadata":reference(metadata_path,repo),
                   "legacy_split":reference(old_manifest_path,repo),"split_script":reference(__file__,repo),"tasks":tasks,
                   "incoming_unique_records":len(metadata),"new_unmatched_records":sum(truthy(r["is_new_unmatched"]) for r in metadata),
                   "unique_modeling_entities":len({(r["species"],r["entity_id"]) for r in all_rows}),
                   "task_rows_are_not_additional_independent_observations":True,
                   "no_model_fits":True,"independent_confirmation_established":False}
    jwrite(output/"task_drafts.json", preparation)
    jwrite(output/"MODEL_PACKS_READY.json", {"status":"MODEL_PACKS_READY", "task_drafts":reference(output/"task_drafts.json",repo),"artifacts":ready_refs})
    return preparation


def colname(number):
    value = ""
    while number:
        number, rem = divmod(number-1,26)
        value = chr(65+rem)+value
    return value


def write_exact_xlsx(path, sheets):
    """Streaming OOXML: Python repr decimals preserve float64 round-trip bytes."""
    path = Path(path)
    require(not path.exists(), "Never overwrite a workbook derivative")
    path.parent.mkdir(parents=True, exist_ok=True)
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    content = '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    content += ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,len(sheets)+1))+'</Types>'
    book = f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>'
    book += ''.join(f'<sheet name="{escape(s["name"])}" sheetId="{i}" r:id="rId{i}"/>' for i,s in enumerate(sheets,1))+'</sheets></workbook>'
    relations = '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    relations += ''.join(f'<Relationship Id="rId{i}" Type="{rel}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(sheets)+1))+'</Relationships>'
    root_rel = f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="{rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    def item(z,name,value):
        info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
        z.writestr(info,value)
    blanks=0
    with zipfile.ZipFile(path,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,value in (("[Content_Types].xml",content),("_rels/.rels",root_rel),("xl/workbook.xml",book),("xl/_rels/workbook.xml.rels",relations)):
            item(z,name,value)
        for n,sheet in enumerate(sheets,1):
            axis=np.asarray(sheet['axis']);values=np.asarray(sheet['X']);refs=np.asarray(sheet['reference_values'])
            headers=[sheet['axis_header'],*sheet['reference_headers'],*sheet['record_ids']]
            require(values.shape==(len(sheet['record_ids']),len(axis)) and refs.shape==(len(sheet['reference_headers']),len(axis)),"Workbook sheet dimensions differ")
            info=zipfile.ZipInfo(f'xl/worksheets/sheet{n}.xml',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            with z.open(info,'w') as h:
                h.write(f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{ns}"><dimension ref="A1:{colname(len(headers))}{len(axis)+1}"/><sheetData>'.encode())
                header='<row r="1">'+''.join(f'<c r="{colname(k)}1" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>' for k,v in enumerate(headers,1))+'</row>'
                h.write(header.encode())
                for row_no in range(len(axis)):
                    numbers=[axis[row_no],*refs[:,row_no],*values[:,row_no]]
                    cells=[]
                    for k,value in enumerate(numbers,1):
                        if np.isfinite(value):cells.append(f'<c r="{colname(k)}{row_no+2}"><v>{repr(float(value))}</v></c>')
                        else:blanks+=1
                    h.write(('<row r="'+str(row_no+2)+'">'+''.join(cells)+'</row>').encode())
                h.write(b'</sheetData></worksheet>')
    return {"nonfinite_cells_exported_as_blank":blanks,"float_storage":"repr_float64_roundtrip_OOXML_numbers","container_timestamp":"fixed1980_deterministic_not_measurement_time"}


def verify_xlsx(path, sheets):
    from openpyxl import load_workbook
    book=load_workbook(path,read_only=True,data_only=True)
    require(book.sheetnames==[s['name'] for s in sheets],"Workbook sheet order changed")
    cells=0
    try:
        for spec in sheets:
            rows=book[spec['name']].iter_rows(values_only=True)
            expected_header=tuple([spec['axis_header'],*spec['reference_headers'],*spec['record_ids']])
            require(tuple(next(rows))==expected_header,"Anonymous workbook headers changed")
            count=0
            for index,row in enumerate(rows):
                expected=np.r_[spec['axis'][index],spec['reference_values'][:,index],spec['X'][:,index]]
                observed=np.array([float(v) if v is not None else np.nan for v in row])
                require(observed.shape==expected.shape and np.array_equal(observed,expected,equal_nan=True),"Numeric XLSX roundtrip changed a source value")
                cells+=len(expected);count+=1
            require(count==len(spec['axis']),"Workbook axis row count changed")
    finally:book.close()
    return {"status":"PASS_INDEPENDENT_OPENPYXL_ALL_VALUES", "numeric_cells_checked":cells}


def export_workbooks(repo, output, recipe):
    repo,output=Path(repo).resolve(),Path(output).resolve()
    ready=json.loads((output/'task_drafts.json').read_text())
    checked(repo,ready['recipe']);checked(repo,ready['intake_sources'])
    sources=json.loads(checked(repo,ready['intake_sources']).read_text())['sources']
    role_rows=list(csv.DictReader((output/'incoming_role_map.csv').open(encoding='utf-8-sig')))
    lookup={r['sample_id']:r for r in role_rows}
    base=output/'workbooks';require(not base.exists(),"Preserve prior workbook exports");base.mkdir()
    receipts=[]
    limit=int(recipe['split']['role_export_max_sample_columns'])
    for source in sources:
        modalities={}
        for mode,name in MODALITIES:
            ref=source['modalities'][mode]
            require(ref['status']=='READ_COMPLETE',"Incomplete incoming modality cannot be silently exported")
            modalities[mode]=(load_npz(checked(repo,{'path':ref['artifact'],'sha256':ref['artifact_sha256']})),ref)
        alpha=modalities['alpha'][0]
        ids=alpha['record_ids'].astype(str).tolist()
        for role in ROLES:
            chosen=[i for i in ids if lookup[i]['role']==role]
            for chunk,start in enumerate(range(0,len(chosen),limit),1):
                subset=chosen[start:start+limit];sheets=[]
                for mode,name in MODALITIES:
                    pack,ref=modalities[mode];positions={str(v):k for k,v in enumerate(pack['record_ids'])}
                    require(set(subset)<=set(positions),"Cross-modality sample is missing")
                    indices=[positions[i] for i in subset]
                    headers=[]
                    for i,k in zip(subset,indices):
                        original=str(pack['original_headers'][k]);suffix=original[original.find('_'):] if '_' in original else ''
                        headers.append(i+suffix)
                    sheets.append({'name':name,'axis_header':ref['profile']['axis_header'],'axis':pack['axis'],
                                   'reference_headers':pack['reference_headers'].astype(str).tolist(),
                                   'reference_values':pack['reference_values'],'record_ids':headers,'X':pack['X'][indices]})
                path=base/source['species']/role/f'{source["source_id"]}_{role}_part{chunk:03d}.xlsx'
                export=write_exact_xlsx(path,sheets);verification=verify_xlsx(path,sheets)
                receipts.append({'source_id':source['source_id'],'species':source['species'],'role':role,'chunk':chunk,
                                 'sample_ids':subset,'canonical_model_ids':[lookup[i]['canonical_model_record_id'] for i in subset],
                                 'artifact':reference(path,repo),'source_hash':source['sha256'],**export,**verification,
                                 'numeric_basis':'incoming_raw_numeric_values; modeling rolepacks use frozen registered canonical legacy values for matched records'})
    jwrite(output/'workbook_export_index.json',{'status':'DONE','max_sample_columns_per_workbook':limit,'workbooks':receipts,
            'note':'Referencecolumnsareinstrumentreference,notextrastudysamples;modelinginputsarealphaNPZ. Formatting/stylesnotcopied;rawbytesretained.'})
    return receipts


def bind(repo, output):
    repo,output=Path(repo).resolve(),Path(output).resolve()
    prepared=json.loads((output/'task_drafts.json').read_text())
    recipe_path=checked(repo,prepared['recipe']);recipe=json.loads(recipe_path.read_text())
    for task in prepared['tasks']:
        for role in ROLES:checked(repo,task[role+'_pack'])
        checked(repo,task['truth'])
    checked(repo,prepared['intake_manifest']);checked(repo,prepared['intake_metadata']);checked(repo,prepared['legacy_split'])
    recipe.update(tasks=prepared['tasks'],intake_manifest=prepared['intake_manifest'],intake_sources=prepared['intake_sources'],
                  intake_metadata=prepared['intake_metadata'],recipe=prepared['recipe'],preparation=reference(output/'task_drafts.json',repo),
                  status='BOUND_INPUTS_READY_NO_MODEL_FITS')
    files=sorted((repo/'src/herb_spectra').glob('*.py'))+[repo/'scripts/run_supplemental_benchmark_20261104.py',Path(__file__)]
    require(all(p.is_file() for p in files),'Benchmark source must be complete before binding')
    recipe['code_files']={str(p.relative_to(repo)):sha(p) for p in files}
    jwrite(output/'bound_protocol.json',recipe)
    return reference(output/'bound_protocol.json',repo)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','export-workbooks','bind'))
    parser.add_argument('--recipe',type=Path,default=Path('data/research/supplemental_spectra_protocol_20261104.json'))
    parser.add_argument('--output',type=Path,default=Path('data/splits/supplemental_spectra_20261104_v1'))
    args=parser.parse_args();recipe=ROOT/args.recipe;out=ROOT/args.output
    if args.stage=='prepare':result=prepare(ROOT,recipe,out);print(json.dumps({'status':result['status'],'tasks':[{k:t[k] for k in ('id','counts','classes')} for t in result['tasks']]}))
    elif args.stage=='export-workbooks':result=export_workbooks(ROOT,out,json.loads(recipe.read_text()));print(json.dumps({'status':'DONE','workbooks':len(result)}))
    else:print(json.dumps(bind(ROOT,out)))


if __name__=='__main__':main()
