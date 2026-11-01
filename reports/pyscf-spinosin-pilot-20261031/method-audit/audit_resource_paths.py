"""Read-only density-fitting storage and optimizer-norm audit; no SCF calls."""
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".runtime/quantum-pilot-site"))
sys.path.insert(0, str(ROOT / ".runtime/quantum-upgrade-site"))
from pyscf import dft, df, gto  # noqa: E402
from pyscf.df import df_jk  # noqa: E402
from pyscf.df.grad import rhf as df_rhf_gradient  # noqa: E402
import geometric.optimize  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path(__file__).parent / "resource_path_audit.json"
    if out.exists():
        raise FileExistsError("Preserve previous audits")
    composition = {"C": 28, "H": 32, "O": 15}
    nao = 762
    tables = {}
    for basis in ("def2-universal-jfit", "def2-universal-jkfit", "def2-svp-jkfit"):
        atom_counts = {el: sum((2 * shell[0] + 1) * (len(shell[1]) - 1)
                              for shell in gto.basis.load(basis, el)) for el in composition}
        naux = sum(atom_counts[el] * number for el, number in composition.items())
        tensor_bytes = naux * nao * (nao + 1) // 2 * 8
        tables[basis] = {"spherical_AO_count_per_element": atom_counts,
                        "Spinosin_total_auxiliary_AO": naux,
                        "hypothetical_full_rank_packed_3center_float64_bytes": tensor_bytes,
                        "same_GiB": tensor_bytes / 1024**3,
                        "auxiliary_metric_float64_bytes": naux * naux * 8}
    direct_source = inspect.getsource(df_jk.get_jk)
    grad_source = inspect.getsource(df_rhf_gradient.get_j)
    norm_source = inspect.getsource(geometric.optimize.Optimizer.calcGradNorm)
    assert "not with_k" in direct_source and "dfobj._cderi is None" in direct_source
    assert "return get_j(dfobj, dm, hermi, direct_scf_tol), None" in direct_source
    assert "for shl0, shl1, nL in ao_ranges:" in grad_source
    assert "rms_gradient = np.sqrt(np.mean(atomgrad**2))" in norm_source
    assert "atomgrad = np.sqrt(np.sum((gradxc.reshape(-1,3))**2, axis=1))" in norm_source
    mol = gto.M(atom="O 0 0 0; H 0 .8 .6; H 0 -.8 .6", basis="def2-svp", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "PBE"
    mf = mf.density_fit(auxbasis="def2-universal-jfit", only_dfj=True)
    assert mf.with_df._cderi is None and not mol.incore_anyway and mf.only_dfj
    assert not mf._numint.libxc.is_hybrid_xc(mf.xc)
    # Size query is deliberately via separate auxiliary molecule constructor;
    # with_df.get_naoaux() may trigger a full integral tensor build.
    auxmol = df.addons.make_auxmol(mol, "def2-universal-jfit")
    assert mf.with_df._cderi is None
    fake_gradient = np.asarray([[3., 4., 0.], [0., 0., 0.]])
    atom_norms = np.linalg.norm(fake_gradient, axis=1)
    grms = float(np.sqrt(np.mean(atom_norms**2)))
    assert np.isclose(grms, np.sqrt(12.5)) and float(atom_norms.max()) == 5.
    source_objects = {"DF_get_jk_dispatch": df_jk.get_jk,
                      "direct_DF_J": df_jk.get_j,
                      "DF_J_gradient": df_rhf_gradient.get_j,
                      "geometric_calcGradNorm": geometric.optimize.Optimizer.calcGradNorm}
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READ_ONLY_RESOURCE_PATH_AND_GRADIENT_NORM_REVIEW_PASS",
        "quantum_jobs": 0, "script_sha256": sha(__file__),
        "source_evidence": {name: {"path": inspect.getsourcefile(obj),
                                    "sha256": sha(inspect.getsourcefile(obj))}
                            for name, obj in source_objects.items()},
        "sizing": tables,
        "recommended_plain_PBE_path": {"xc_set_before_density_fit": "PBE",
            "auxbasis": "def2-universal-jfit", "only_dfj": True,
            "mol_incore_anyway": False, "initial_cderi_is_None": True,
            "SCF_J_dispatch": "When with_k=False and _cderi is None, direct screened 3-center J; no global Cholesky tensor",
            "gradient_J_dispatch": "Direct shell-blocked 3-center derivatives plus auxiliary-basis response",
            "avoid_calls": ["with_df.build()", "with_df.get_naoaux()", "with_df.loop()", "materializing mf._cderi"],
            "runtime_assertions_needed": ["Check mf.with_df._cderi remains None after actual SCF and gradient", "Record actual memory, CPU and scratch-file usage"],
            "Hessian": "DF-RKS analytical implementation exists; auxbasis_response2 retained. This audit does not certify its cost or actual numerical correctness for Spinosin."},
        "optimizer_gradient_norm": {"source": "geomeTRIC1.1.1 Optimizer.calcGradNorm",
            "for_unconstrained_nonrigid_molecule": "gRMS=sqrt(mean_i(sum_xyz(G_i,xyz^2)))=norm(G)/sqrt(Natoms); gmax=max_i(norm(G_i))",
            "not_equal_to": "RMS over all3N components or max absolute component",
            "units": "Hartree/Bohr", "synthetic_formula_check_only": True,
            "atom_norm_RMS_formula_check": grms},
        "scope": "Source inspection, basis dimension arithmetic, molecule construction and synthetic scalar arithmetic only. No SCF/integrals/gradients/Hessians/optimization were computed by this audit."}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "Jfit_aux": tables["def2-universal-jfit"]["Spinosin_total_auxiliary_AO"],
                      "resource_path_audit_sha256": sha(out)}))


if __name__ == "__main__":
    main()
