"""V7 PBE XC grid-chunk mechanics; top-level is stdlib only, NOT_RUN at prep.

Unmodified RKS XC helpers consume the full frozen grid through an owned
block_loop instance wrapper. The compiled native AST differs only by one
continue guard BEFORE AO evaluation. No sub-grid objects or mask slicing.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
PYSCF_ROOT = ROOT / ".runtime/quantum-upgrade-site/pyscf"
PINNED = {
    "hessian/rks.py": "d61dbd1727b286845737853b92e1812a3e6a966dca29a9a8a79ff5e9a7725c03",
    "dft/numint.py": "ff7384551c7a6b80ff8a5ffcd915eb8bfbf6ca4b8f987f912aa82a1d6edda7e8",
    "grad/rks.py": "cffc3e9efa10cf1b547d757e9d165c1000377e9159c794f87464a79c8c93ab46",
    "dft/gen_grid.py": "44aa665fb25efc544a42cb80fdad737767c787c20479d4df0ae4b1649028ff14"}
PLAN_SCHEMA = "research4-v7-fixed-full-grid-XC-plan-v1"
KINDS = {"xc-diag": 3, "xc-deriv2": 2}
SELECTOR_NAME = "_R4_V7_SELECTED_ORIGINAL_GRID_RANGES"
POLICY = {"xc": "PBE", "xctype": "GGA", "full_grid_values_unchanged": True,
    "original_global_masks_and_sparse_decision_retained": True,
    "native_RKS_XC_helper_math_unmodified": True, "NumInt_block_loop_AST_only_one_selection_continue_added": True,
    "contract_each_native_XC_subset_immediately": True, "NLC_hybrid_range_separation_supported": False,
    "finalized_ipip_in_native_deriv2_not_added_again": True,
    "floating_reduction_order_changes_require_actual_controls": True}


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()


def check_sources():
    actual = {name: sha(PYSCF_ROOT / name) for name in PINNED}
    require(actual == PINNED, "Unreviewed native RKS/NumInt/grid source")
    return actual


def validate_runtime_sources(ni, *, numint, rks, grad_rks, gen_grid):
    """Bind the modules actually EXECUTED, not another installed file tree."""
    check_sources()
    modules = {"dft/numint.py": numint, "hessian/rks.py": rks,
        "grad/rks.py": grad_rks, "dft/gen_grid.py": gen_grid}
    for name, module in modules.items():
        actual = Path(module.__file__).resolve()
        require(actual == PYSCF_ROOT / name and sha(actual) == PINNED[name], "Imported native module differs from pinned source: " + name)
    require(rks.numint is numint and rks.rks_grad is grad_rks and grad_rks.numint is numint,
        "Imported RKS/gradient NumInt module aliases differ")
    function = getattr(ni.block_loop, "__func__", None)
    require(function is numint.NumInt.block_loop and function.__globals__ is numint.__dict__
        and Path(inspect.getsourcefile(function)).resolve() == PYSCF_ROOT / "dft/numint.py",
        "Actual NumInt function/global namespace differs")
    require(numint.BLKSIZE == gen_grid.BLKSIZE == 56, "Imported native grid alignment changed")
    return {name: {"path": str(Path(module.__file__).resolve()), "sha256": PINNED[name]} for name, module in modules.items()}


def native_loop_ast():
    check_sources()
    tree = ast.parse((PYSCF_ROOT / "dft/numint.py").read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NumInt"]
    require(len(classes) == 1, "Unique original NumInt class required")
    functions = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "block_loop"]
    require(len(functions) == 1 and not functions[0].decorator_list, "Unique undecorated native block_loop required")
    return functions[0]


def filtered_loop_ast():
    """Pure source construction; does not compile/execute numerical code."""
    original = native_loop_ast(); modified = copy.deepcopy(original)
    loops = [n for n in ast.walk(modified) if isinstance(n, ast.For)]
    require(len(loops) == 1, "Native grid iterator layout changed")
    loop = loops[0]
    wanted_target = ast.parse("(ip0, ip1)", mode="eval").body
    wanted_target.ctx = ast.Store()
    for item in wanted_target.elts: item.ctx = ast.Store()
    require(ast.dump(loop.target) == ast.dump(wanted_target)
        and ast.unparse(loop.iter) == "lib.prange(0, ngrids, blksize)", "Unique native grid loop header changed")
    require(ast.unparse(loop.body[0]) == "coords = grids.coords[ip0:ip1]", "Selection must precede all coordinates/AO work")
    inserted = ast.parse(f"if (ip0, ip1) not in {SELECTOR_NAME}:\n    continue\n").body[0]
    loop.body.insert(0, inserted)
    # Exact AST equality after deleting our SINGLE added statement, including
    # signature/defaults, full-grid sparse/mask logic, AO calls and yield.
    stripped = copy.deepcopy(modified)
    stripped_loop = [n for n in ast.walk(stripped) if isinstance(n, ast.For)][0]
    stripped_loop.body.pop(0)
    require(ast.dump(stripped, include_attributes=False) == ast.dump(original, include_attributes=False),
            "Unexpected native mathematical/scheduling rewrite")
    return ast.fix_missing_locations(modified), {
        "original_function_AST_sha256": digest(ast.dump(original, include_attributes=False)),
        "filtered_function_AST_sha256": digest(ast.dump(modified, include_attributes=False)),
        "allowed_change": "ONE_IF_ORIGINAL_IP0_IP1_NOT_SELECTED_CONTINUE_BEFORE_COORDS_AO",
        "native_source_sha256": PINNED["dft/numint.py"], "all_other_AST_identical": True}


def make_plan(*, kind, nao, natm, ngrids, max_memory_mb, grid_binding, physical_binding_sha256, control=False):
    require(kind in KINDS and all(type(v) is int and v > 0 for v in (nao, natm, ngrids)), "Explicit positive dimensions/kind required")
    require(type(max_memory_mb) in (int, float) and math.isfinite(max_memory_mb) and max_memory_mb > 0, "Frozen finite memory policy required")
    deriv = KINDS[kind]; comp = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
    # Exact pinned NumInt formula; only controls explicitly force224 to ensure
    # more than one original grid block without changing full-grid semantics.
    native = max(4, min(int(max_memory_mb * 1e6 / ((comp + 1) * nao * 8 * 56)), ngrids // 56 + 1, 1200)) * 56
    blocksize = 224 if control else native
    require(blocksize % 56 == 0, "Native non0tab boundary must remain56-aligned")
    blocks = [{"index": i, "start": start, "stop": min(start + blocksize, ngrids)}
        for i, start in enumerate(range(0, ngrids, blocksize))]
    require(not control or len(blocks) >= 3, "Real software controls must exercise multiple224-point blocks")
    return {"schema": PLAN_SCHEMA, "kind": kind, "ao_deriv": deriv, "nao": nao, "natm": natm, "ngrids": ngrids,
        "max_memory_mb": max_memory_mb, "native_automatic_blocksize": native, "blocksize": blocksize,
        "blocksize_mode": "EXPLICIT_CONTROL224" if control else "FROZEN_ORIGINAL_AUTOMATIC_FORMULA",
        "BLKSIZE": 56, "full_global_grid_and_masks": grid_binding, "physical_binding_sha256": physical_binding_sha256,
        "blocks": blocks, "native_sources": dict(PINNED), "checkpoint_policy": dict(POLICY)}


def validate_plan(plan):
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("checkpoint_policy") == POLICY and plan.get("native_sources") == PINNED,
            "Unreviewed XC plan/policy/source")
    expected = make_plan(kind=plan["kind"], nao=plan["nao"], natm=plan["natm"], ngrids=plan["ngrids"],
        max_memory_mb=plan["max_memory_mb"], grid_binding=plan["full_global_grid_and_masks"],
        physical_binding_sha256=plan["physical_binding_sha256"], control=plan["blocksize_mode"] == "EXPLICIT_CONTROL224")
    require(plan == expected, "Frozen complete block plan changed")
    return plan


def selected_blocks(plan, start, count):
    validate_plan(plan)
    require(type(start) is int and type(count) is int and start >= 0 and count > 0
        and start + count <= len(plan["blocks"]), "Nonempty contiguous registered block range required")
    return plan["blocks"][start:start + count]


def coverage(plans, chunks):
    """Metadata-only exact per-kind coverage; duplicate or missing blocks fail."""
    require(set(plans) == set(KINDS), "Both native XC kinds required")
    common = None
    for kind, plan in plans.items():
        validate_plan(plan)
        require(plan["kind"] == kind, "Plan kind/key mismatch")
        identity = {k: plan[k] for k in ("physical_binding_sha256", "full_global_grid_and_masks", "nao", "natm", "ngrids", "max_memory_mb", "blocksize_mode")}
        require(common is None or identity == common, "XC kinds changed physical/full-grid identity or policy")
        common = identity
    result = {}
    for kind, plan in plans.items():
        validate_plan(plan); observed = []
        own = [r for r in chunks if r["kind"] == kind]
        for row in own:
            require(row["plan_binding_sha256"] == digest(plan), "Chunk belongs to another frozen plan")
            chosen = selected_blocks(plan, row["start_block"], row["block_count"])
            require(row["selected_blocks"] == chosen, "Chunk's actual original block ranges differ")
            observed.extend(b["index"] for b in chosen)
        require(sorted(observed) == list(range(len(plan["blocks"]))), "Missing/duplicate original XC grid block")
        result[kind] = {"native_blocks": len(observed), "grid_points": plan["ngrids"], "closed_chunks": len(own),
            "each_original_grid_point_once": True, "original_full_grid_fingerprint": plan["full_global_grid_and_masks"]}
    require(all(r["kind"] in KINDS for r in chunks), "Unknown XC contribution kind")
    return result


def grid_binding(hobj, np, fingerprint):
    grids = hobj.grids if hobj.grids is not None else hobj.base.grids
    require(grids.coords is not None and grids.weights is not None, "Only initialized frozen original grids supported")
    require(grids is hobj.base.grids and grids.mol is hobj.mol,
        "Only actual original MF grids with the original molecule identity supported")
    records = {"coordinates": fingerprint(grids.coords), "weights": fingerprint(grids.weights)}
    for name in ("non0tab", "screen_index", "atm_idx", "quadrature_weights"):
        value = getattr(grids, name, None)
        records[name] = None if value is None else fingerprint(value)
    return {"arrays": records, "grids_mol_is_actual_mol": grids.mol is hobj.mol,
        "grids_is_MF_grids": grids is hobj.base.grids, "cutoff": float(grids.cutoff),
        "NumInt_cutoff": float(hobj.base._numint.cutoff), "full_grid_points": int(grids.coords.shape[0])}


class GridSelectionContext:
    def __init__(self, ni, mol, grids, plan, selected, emit=None):
        self.ni, self.mol, self.grids, self.plan, self.selected = ni, mol, grids, validate_plan(plan), selected
        require(selected and all(b in plan["blocks"] for b in selected), "Only registered original grid blocks may be selected")
        require(len({b["index"] for b in selected}) == len(selected), "Duplicate selected block")
        self.emit = emit or (lambda *args, **kwargs: None); self.completed = []

    def __enter__(self):
        import numpy as np
        from pyscf.dft import numint
        check_sources()
        require(numint.BLKSIZE == 56, "Native mask alignment changed")
        require(getattr(self.ni.block_loop, "__func__", None) is numint.NumInt.block_loop
            and "block_loop" not in self.ni.__dict__
            and self.ni.block_loop.__func__.__globals__ is numint.__dict__
            and Path(inspect.getsourcefile(self.ni.block_loop.__func__)).resolve() == PYSCF_ROOT / "dft/numint.py",
            "Do not nest or overwrite another NumInt wrapper/source")
        node, self.ast_evidence = filtered_loop_ast()
        namespace = dict(self.ni.block_loop.__func__.__globals__)
        namespace[SELECTOR_NAME] = frozenset((b["start"], b["stop"]) for b in self.selected)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(Path(__file__).resolve()), "exec"), namespace)
        compiled = namespace["block_loop"]
        saved = self.ni.block_loop; selected = self.selected; outer = self
        def selected_loop(ni, mol, grids, nao=None, deriv=0, max_memory=2000, non0tab=None, blksize=None, buf=None):
            require(mol is outer.mol and grids is outer.grids and nao == outer.plan["nao"]
                and deriv == outer.plan["ao_deriv"] and max_memory == outer.plan["max_memory_mb"]
                and non0tab is None and buf is None and blksize is None, "Native XC helper block-loop call changed")
            generator = compiled(ni, mol, grids, nao, deriv, max_memory, non0tab, outer.plan["blocksize"], buf)
            for expected in selected:
                try: values = next(generator)
                except StopIteration as exc: raise ValueError("Selected original AO block missing") from exc
                ao, mask, weight, coords = values; start, stop = expected["start"], expected["stop"]
                require(np.array_equal(coords, grids.coords[start:stop]) and np.array_equal(weight, grids.weights[start:stop])
                    and ao.shape[-2] == stop - start, "Original coordinate/weight order or AO block changed")
                outer.emit("xc_block_started", **expected)
                yield values  # All original mask objects and AO buffers unchanged.
                # Executes only when native consumer requests the next block
                # after finishing this block. A thrown/closed generator does
                # not get labelled complete by an exception/finally path.
                outer.completed.append(expected["index"]); outer.emit("xc_block_completed", **expected)
            try: next(generator)
            except StopIteration: return
            raise ValueError("Native iterator produced an unselected/duplicate block")
        self.saved, self.saved_function = saved, saved.__func__
        self.ni.block_loop = types.MethodType(selected_loop, self.ni)
        return self

    def __exit__(self, *exc):
        del self.ni.__dict__["block_loop"]
        require(self.ni.block_loop.__func__ is self.saved_function, "Original NumInt block_loop was not restored")

    def require_complete(self):
        require(self.completed == [b["index"] for b in self.selected], "Native XC helper did not finish all selected original blocks")


def contract_xc(kind, value, dm0, aoslices, natm, np):
    """Exactly rks.partial_hess_elec106/109/112, per contribution kind."""
    nao = dm0.shape[0]
    wanted = (3, 3, nao, nao) if kind == "xc-diag" else (natm, 3, 3, nao, nao)
    require(kind in KINDS and value.shape == wanted and value.dtype == np.float64 and np.isfinite(value).all(), "Native complete XC array invalid")
    result = np.zeros((natm, natm, 3, 3), dtype=np.float64)
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        if kind == "xc-diag":
            result[ia, ia] += np.einsum('xypq,pq->xy', value[:, :, p0:p1], dm0[p0:p1]) * 2
        else:
            veff = value[ia]
            for ja in range(ia + 1):
                q0, q1 = aoslices[ja][2:]
                result[ia, ja] += np.einsum('xypq,pq->xy', veff[:, :, q0:q1], dm0[q0:q1]) * 2
            for ja in range(ia): result[ja, ia] = result[ia, ja].T
    return result


def native_xc_chunk(hobj, plan, start, count, *, emit=None, return_control_AO=False):
    import numpy as np
    from pyscf.hessian import rks
    from pyscf.dft import numint, gen_grid
    from pyscf.grad import rks as grad_rks
    mf, mol = hobj.base, hobj.mol; ni = mf._numint
    actual_sources = validate_runtime_sources(ni, numint=numint, rks=rks, grad_rks=grad_rks, gen_grid=gen_grid)
    require(str(mf.xc).strip().upper() == "PBE" and ni._xc_type(mf.xc) == "GGA"
        and not ni.libxc.is_hybrid_xc(mf.xc) and not mf.do_nlc()
        and all(float(v) == 0 for v in ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin))
        and getattr(mf, "with_df", None) is None and getattr(mf, "disp", None) is None,
        "Only original full-AO, non-DF, nonhybrid plain PBE without NLC/dispersion supported")
    require(not return_control_AO or mol.natm in (3, 9), "Never retain huge Spinosin AO derivative arrays")
    selected = selected_blocks(plan, start, count)
    grids = hobj.grids if hobj.grids is not None else mf.grids
    require(mol.natm == plan["natm"] and mol.nao_nr() == plan["nao"]
        and grids.coords.shape[0] == plan["ngrids"], "Actual full-system dimensions differ from frozen plan")
    helper = rks._get_vxc_diag if plan["kind"] == "xc-diag" else rks._get_vxc_deriv2
    require(Path(inspect.getsourcefile(helper)).resolve() == PYSCF_ROOT / "hessian/rks.py", "Only pinned native XC helper supported")
    with GridSelectionContext(ni, mol, grids, plan, selected, emit) as context:
        value = helper(hobj, mf.mo_coeff, mf.mo_occ, plan["max_memory_mb"])
        context.require_complete()
    dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    contracted = contract_xc(plan["kind"], value, dm0, mol.aoslice_by_atom(), mol.natm, np)
    metadata = {"kind": plan["kind"], "plan_binding_sha256": digest(plan), "start_block": start, "block_count": count,
        "selected_blocks": selected, "native_helper_returned_normally": True, "original_block_loop_restored": True,
        "all_selected_original_blocks_completed": context.completed, "block_loop_AST_evidence": context.ast_evidence,
        "actual_imported_native_sources": actual_sources,
        "ipip_finalized_inside_native_helper_only": plan["kind"] == "xc-deriv2",
        "no_extra_ipip_nuclear_RHF_or_CPHF_added": True, "full_grid_objects_unchanged": True}
    return contracted, metadata, value if return_control_AO else None


def sum_components(rhf, chunks, np):
    """Deterministic one-time sum, not symmetry averaging or mode clipping."""
    require(rhf.dtype == np.float64 and rhf.ndim == 4 and rhf.shape[0] == rhf.shape[1]
        and rhf.shape[2:] == (3, 3) and np.isfinite(rhf).all(), "Complete finite native RHF part required")
    result = rhf.copy()
    # Caller supplies validated sorted kind/start order and exact coverage.
    for value in chunks:
        require(value.dtype == np.float64 and value.shape == result.shape and np.isfinite(value).all(), "Complete XC contracted chunk required")
        result += value
    return result
