"""Pure-stdlib V7 core contracts; never import or execute numerical modules."""
from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts/pyscf_xc_chunk_checkpoint_20261105_v7.py"
SPEC = importlib.util.spec_from_file_location("xc_chunk_core_v7_under_test", CORE_PATH)
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


def plan(kind="xc-diag", **changes):
    arguments = dict(kind=kind, nao=8, natm=3, ngrids=1001,
        max_memory_mb=0.001,
        grid_binding={"full_grid_points": 1001, "arrays": {
            "coordinates": {"sha256": "1" * 64},
            "weights": {"sha256": "2" * 64}}},
        physical_binding_sha256="a" * 64)
    arguments.update(changes)
    return CORE.make_plan(**arguments)


def plans():
    return {kind: plan(kind) for kind in CORE.KINDS}


def chunks_for(all_plans):
    rows = []
    for kind, frozen in all_plans.items():
        for start in range(0, len(frozen["blocks"]), 2):
            count = min(2, len(frozen["blocks"]) - start)
            rows.append({"kind": kind, "plan_binding_sha256": CORE.digest(frozen),
                "start_block": start, "block_count": count,
                "selected_blocks": copy.deepcopy(frozen["blocks"][start:start + count])})
    return rows


class NativeASTContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These parse/hash only the small, pinned native source files.
        cls.original = CORE.native_loop_ast()
        cls.filtered, cls.evidence = CORE.filtered_loop_ast()

    def test_core_top_level_imports_only_stdlib(self):
        tree = ast.parse(CORE_PATH.read_text())
        allowed = {"__future__", "ast", "copy", "hashlib", "inspect", "json",
                   "math", "pathlib", "types"}
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                self.assertTrue(all(alias.name.split(".")[0] in allowed
                                    for alias in statement.names))
            elif isinstance(statement, ast.ImportFrom):
                self.assertIn(statement.module.split(".")[0], allowed)
        self.assertFalse(any(name == "numpy" or name == "pyscf"
                             or name.startswith("pyscf.") for name in sys.modules))

    def test_exactly_one_guard_before_coords_and_all_other_ast_equal(self):
        before = [node for node in ast.walk(self.original) if isinstance(node, ast.For)]
        after = [node for node in ast.walk(self.filtered) if isinstance(node, ast.For)]
        self.assertEqual(len(before), 1)
        self.assertEqual(len(after), 1)
        expected = ast.parse(
            f"if (ip0, ip1) not in {CORE.SELECTOR_NAME}:\n    continue\n").body[0]
        self.assertEqual(ast.dump(after[0].body[0]), ast.dump(expected))
        self.assertEqual(ast.unparse(after[0].body[1]), "coords = grids.coords[ip0:ip1]")
        self.assertEqual(sum(isinstance(node, ast.Continue)
                             for node in ast.walk(self.filtered)), 1)
        stripped = copy.deepcopy(self.filtered)
        [node for node in ast.walk(stripped) if isinstance(node, ast.For)][0].body.pop(0)
        self.assertEqual(ast.dump(stripped, include_attributes=False),
                         ast.dump(self.original, include_attributes=False))
        self.assertTrue(self.evidence["all_other_AST_identical"])

    def test_full_grid_sparse_and_mask_decisions_stay_native(self):
        text = ast.unparse(self.filtered)
        self.assertIn("ngrids = grids.coords.shape[0]", text)
        self.assertIn("allow_sparse = ngrids % ALIGNMENT_UNIT == 0 and nao > SWITCH_SIZE", text)
        self.assertIn("mask = screen_index[ip0 // BLKSIZE:]", text)
        self.assertIn("if not allow_sparse and (not _sparse_enough(mask)):", text)
        self.assertIn("non0tab=mask, cutoff=grids.cutoff, out=buf", text)
        self.assertIn("yield (ao, mask, weight, coords)", text)

    def test_default_blocksize_matches_actual_native_scalar_ast(self):
        comp = next(node for node in self.original.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "comp"
                            for target in node.targets))
        branch = next(node for node in self.original.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "blksize is None")
        scalar = ast.fix_missing_locations(ast.Module(
            body=[copy.deepcopy(comp), *copy.deepcopy(branch.body)], type_ignores=[]))
        compiled = compile(scalar, "<native NumInt scalar blocksize formula>", "exec")
        for kind, derivative in CORE.KINDS.items():
            for nao, ngrids, memory in ((1, 1, 0.001), (8, 1001, 0.001),
                    (240, 12345, 2000), (3, 100000, 1e8), (24, 225, 0.25)):
                with self.subTest(kind=kind, nao=nao, ngrids=ngrids, memory=memory):
                    namespace = dict(deriv=derivative, nao=nao, ngrids=ngrids,
                                     max_memory=memory, BLKSIZE=56)
                    exec(compiled, namespace)
                    frozen = plan(kind, nao=nao, ngrids=ngrids, max_memory_mb=memory)
                    self.assertEqual(frozen["blocksize"], namespace["blksize"])
                    self.assertEqual(frozen["native_automatic_blocksize"], namespace["blksize"])


class PlanContracts(unittest.TestCase):
    def test_control_224_and_nonempty_tail_blocks(self):
        frozen = plan(control=True)
        self.assertEqual(frozen["blocksize"], 224)
        self.assertEqual(frozen["blocksize_mode"], "EXPLICIT_CONTROL224")
        self.assertGreaterEqual(len(frozen["blocks"]), 3)
        self.assertEqual(frozen["blocks"][-1], {"index": 4, "start": 896, "stop": 1001})
        self.assertEqual(CORE.validate_plan(frozen), frozen)

    def test_control_requires_at_least_three_actual_blocks(self):
        for size in (1, 224, 448):
            with self.subTest(size=size), self.assertRaises(ValueError):
                plan(control=True, ngrids=size)
        self.assertEqual(len(plan(control=True, ngrids=449)["blocks"]), 3)

    def test_bad_dimensions_memory_and_kind_rejected(self):
        for arguments in ({"nao": True}, {"natm": 0}, {"ngrids": -1},
                          {"max_memory_mb": True}, {"max_memory_mb": 0},
                          {"max_memory_mb": float("nan")},
                          {"max_memory_mb": float("inf")}, {"kind": "unknown"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                plan(**arguments)

    def test_plan_mutation_or_extra_field_rejected(self):
        for mutation in ("blocks", "blocksize", "policy", "source", "extra"):
            frozen = plan()
            if mutation == "blocks": frozen["blocks"][0]["stop"] -= 1
            elif mutation == "blocksize": frozen["blocksize"] += 56
            elif mutation == "policy": frozen["checkpoint_policy"]["xc"] = "LDA"
            elif mutation == "source": frozen["native_sources"]["dft/numint.py"] = "0" * 64
            else: frozen["extra"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                CORE.validate_plan(frozen)

    def test_selection_requires_exact_integer_nonempty_in_bounds(self):
        frozen = plan()
        self.assertEqual(CORE.selected_blocks(frozen, 1, 2), frozen["blocks"][1:3])
        for start, count in ((False, 1), (0, True), (-1, 1), (0, 0), (4, 2), (5, 1)):
            with self.subTest(start=start, count=count), self.assertRaises(ValueError):
                CORE.selected_blocks(frozen, start, count)

    def test_context_rejects_empty_duplicate_or_unregistered_selection_cold(self):
        frozen = plan()
        for selected in ([], [frozen["blocks"][0], frozen["blocks"][0]],
                         [{"index": 99, "start": 0, "stop": 224}]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                CORE.GridSelectionContext(object(), object(), object(), frozen, selected)


class CoverageContracts(unittest.TestCase):
    def test_two_kinds_cover_each_original_block_once(self):
        frozen = plans()
        actual = CORE.coverage(frozen, list(reversed(chunks_for(frozen))))
        for kind in CORE.KINDS:
            self.assertEqual(actual[kind]["native_blocks"], len(frozen[kind]["blocks"]))
            self.assertEqual(actual[kind]["grid_points"], 1001)
            self.assertTrue(actual[kind]["each_original_grid_point_once"])

    def test_missing_duplicate_and_unknown_kind_rejected(self):
        frozen = plans()
        good = chunks_for(frozen)
        unknown = copy.deepcopy(good[0]); unknown["kind"] = "unknown"
        for rows in (good[1:], [*good, copy.deepcopy(good[0])], [*good, unknown]):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                CORE.coverage(frozen, rows)

    def test_wrong_plan_digest_or_actual_selection_rejected(self):
        frozen = plans()
        for mutation in ("digest", "selection"):
            rows = chunks_for(frozen)
            if mutation == "digest": rows[0]["plan_binding_sha256"] = "0" * 64
            else: rows[0]["selected_blocks"][0]["stop"] -= 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                CORE.coverage(frozen, rows)

    def test_both_kinds_required(self):
        frozen = plans()
        del frozen["xc-diag"]
        with self.assertRaises(ValueError): CORE.coverage(frozen, chunks_for(frozen))

    def test_plan_dictionary_key_must_equal_internal_kind(self):
        frozen = plans()
        swapped = {"xc-diag": frozen["xc-deriv2"], "xc-deriv2": frozen["xc-diag"]}
        with self.assertRaises(ValueError): CORE.coverage(swapped, chunks_for(swapped))

    def test_two_kinds_must_share_physical_grid_and_dimensions(self):
        for change in ({"physical_binding_sha256": "b" * 64},
                       {"grid_binding": {"full_grid_points": 1001, "different_grid": True}},
                       {"nao": 9}, {"natm": 4}, {"ngrids": 1002}):
            frozen = plans()
            frozen["xc-deriv2"] = plan("xc-deriv2", **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                CORE.coverage(frozen, chunks_for(frozen))


if __name__ == "__main__":
    unittest.main()
