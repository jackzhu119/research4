"""Inspect cached-DF memory/disk paths without constructing any integrals."""
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".runtime/quantum-upgrade-site"))
from pyscf import dft, gto  # noqa: E402
from pyscf.df import df as df_module, df_jk, incore, outcore  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path(__file__).parent / "cached_df_execution_audit.json"
    if out.exists():
        raise FileExistsError("Keep earlier audit")
    build_source = inspect.getsource(df_module.DF.build)
    incore_source = inspect.getsource(incore.cholesky_eri)
    block_source = inspect.getsource(outcore.cholesky_eri_b)
    compatible_source = inspect.getsource(outcore.cholesky_eri)
    assert "self._compatible_format" in build_source
    assert "outcore.cholesky_eri_b" in build_source
    assert "cderi = numpy.empty((naux, nao_pair))" in incore_source
    assert "max_words = max_memory*.98e6/8 - low.size - cderi.size" in incore_source
    assert "max_words/naoaux/comp/3" in incore_source
    assert "swapfile = lib.NamedTemporaryFile" not in block_source
    assert "swapfile = lib.NamedTemporaryFile" in compatible_source
    mol = gto.M(atom="O 0 0 0; H 0 .8 .6; H 0 -.8 .6", basis="def2-svp", verbose=0)
    mol.incore_anyway = True
    mf = dft.RKS(mol)
    mf.xc = "PBE"
    mf.max_memory = 8000
    mf = mf.density_fit(auxbasis="def2-universal-jfit", only_dfj=True)
    assert not mf._numint.libxc.is_hybrid_xc(mf.xc)
    assert mol.incore_anyway and mf.with_df._cderi is None
    assert mf.only_dfj and mf.with_df.max_memory == 8000
    assert mf.with_df._compatible_format is False
    naux, nao_pair = 2459, 762 * 763 // 2
    tensor_bytes, metric_bytes = naux * nao_pair * 8, naux * naux * 8
    max_memory = 8000
    ideal_max_words = max_memory * .98e6 / 8 - naux * naux - naux * nao_pair
    ideal_buffer_pairs = int(ideal_max_words / naux / 3)
    ideal_one_buffer_bytes = ideal_buffer_pairs * naux * 8
    paths = {"DF_build": inspect.getsourcefile(df_module.DF.build),
             "DF_dispatch": inspect.getsourcefile(df_jk.get_jk),
             "incore_cholesky": inspect.getsourcefile(incore.cholesky_eri),
             "outcore_blocked": inspect.getsourcefile(outcore.cholesky_eri_b)}
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CACHED_DF_EXECUTION_SOURCE_REVIEW_PASS_WITH_RESOURCE_GUARDS_REQUIRED",
        "quantum_or_integral_jobs": 0, "script_sha256": sha(__file__),
        "source_files": {k: {"path": v, "sha256": sha(v)} for k, v in paths.items()},
        "same_physical_method_if": ["PBE pure nonhybrid only", "same def2-SVP orbital and def2-universal-jfit auxiliary bases",
            "same integration grid, charge, multiplicity and SCF thresholds", "each geometry resets the DF cache"],
        "incore_dispatch": {"mol_incore_anyway": True, "only_dfj": True,
            "effect": "Suppresses direct-J shortcut; DF.loop builds cached tensor when _cderi is None, including after scanner resets",
            "must_not_call_undo_df_or_hybrid_HF_fallback": True,
            "reason": "incore_anyway on an ordinary/hybrid exact-exchange branch could request impossible full4-center ERIs"},
        "Spinosin_full_rank_sizes": {"orbital_AO": 762, "auxiliary_AO": naux,
            "packed_pairs": nao_pair, "cached_tensor_float64_bytes": tensor_bytes,
            "auxiliary_metric_float64_bytes": metric_bytes},
        "incore_allocation_source_arithmetic": {"configured_MB": max_memory,
            "ideal_buffer_pairs_before_shell_rounding": ideal_buffer_pairs,
            "ideal_single_buffer_bytes": ideal_one_buffer_bytes,
            "base_tensor_plus_two_buffers_and_metric_bytes": tensor_bytes + 2 * ideal_one_buffer_bytes + metric_bytes,
            "possible_third_buffer_temporary_bytes": ideal_one_buffer_bytes,
            "limitations": "Source arithmetic only, excludes process baseline, shell-rounding, libcint workspaces and transient copies. Not a measured memory guarantee. PySCF source also warns temporary use can be about twice returned tensor."},
        "outcore_alternative": {"compatible_format": False,
            "path": "outcore.cholesky_eri_b writes blocked transformed tensor directly to oneHDF5",
            "approximate_final_tensor_bytes": tensor_bytes,
            "compatible_format_True_unsafe_reason": "outcore.cholesky_eri creates a full blocked swap and a second compatible final tensor; totalabout11.44GB exceeds9.6GB initial free disk",
            "reset_file_caution": "HDF5 creation helper opens existing files in append mode and deletes oldgroup; explicitly truncate/replace closed oldcache between geometries rather than assuming filesize reuse"},
        "runtime_required_checks": ["CPU, address-space, RSS and disk limits enforced by parent supervisor",
            "Confirm actual cached tensor type/shape and measuredmemory after build",
            "Drop/reset previous cached mf and garbage collect BEFORE constructing a fresh final-SCF mf; avoid simultaneously retaining two5.72GB tensors",
            "Run the small-molecule cached-vs-direct numerical check before interpreting the changed execution policy",
            "Keep original direct-J phase/logs/checkpoints unchanged and declare any restarted phase separately"]}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cached_df_audit_sha256": sha(out)}))


if __name__ == "__main__":
    main()
