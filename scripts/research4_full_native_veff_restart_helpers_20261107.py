"""Small full-space restart helpers; importing this module performs no chemistry.

The checkpoint is an initial guess only.  Historical cached Coulomb matrices
cannot be recovered from a checkpoint that did not store them.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math

FULL_BUILD_POLICY = "FULL_NATIVE_RKS_VEFF_EVERY_INVOCATION_DM_LAST_NONE_VHF_LAST_NONE"


def full_native_veff_type(parent_type, native_get_veff, *, emit=None):
    """Return a new class; never alter the restored source or parent class."""
    class FullNativeVEFF(parent_type):
        _research4_full_native_veff_policy = FULL_BUILD_POLICY

        def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
            if mol is not None and mol is not self.mol:
                raise ValueError("Explicit VEFF molecule must own the fixed grid")
            if (self.direct_scf is not True or self._eri is not None
                    or getattr(self, "with_df", None) is not None
                    or self.mol.incore_anyway):
                raise ValueError("Full native direct AO policy changed")
            if self.grids.mol is not self.mol or self.small_rho_cutoff != 0:
                raise ValueError("Fixed molecule-owned quadrature changed")
            count = getattr(self, "_research4_full_veff_calls", 0) + 1
            self._research4_full_veff_calls = count
            if emit is not None:
                emit("full_native_VEFF_started", {"invocation_1based": count,
                    "incoming_incremental_arguments_ignored": dm_last is not None or vhf_last is not None,
                    "policy": FULL_BUILD_POLICY})
            return native_get_veff(self, mol, dm, dm_last=None, vhf_last=None, hermi=hermi)

    FullNativeVEFF.__name__ = "Research4FullNativeVEFF" + parent_type.__name__
    return FullNativeVEFF


def validate_checkpoint_restart_payload(payload, source_molecule, *, np,
                                        atoms, basis_functions, electrons):
    """Pure array/JSON checks, with no overlap/integral/SCF evaluation."""
    molecule = payload["molecule"]
    if isinstance(molecule, (bytes, str)):
        molecule = json.loads(molecule)
    if isinstance(source_molecule, (bytes, str)):
        source_molecule = json.loads(source_molecule)
    for name in ("_atm", "_bas", "_env"):
        if not np.array_equal(np.asarray(molecule[name]), np.asarray(source_molecule[name])):
            raise ValueError("Checkpoint original molecular payload differs: " + name)
    def decoded(value):
        # Mole.dumps serializes selected attributes using repr, whereas the
        # authenticated snapshot molecule.json already contains plain values.
        if isinstance(value, str):
            try:return ast.literal_eval(value)
            except (ValueError,SyntaxError):pass
        return value
    if (molecule.get("charge", 0) != 0 or molecule.get("spin", 0) != 0
            or str(decoded(molecule.get("basis", ""))).lower() != "def2-svp"
            or decoded(molecule.get("_ecp")) or decoded(molecule.get("ecp"))):
        raise ValueError("Checkpoint molecular method/charge/spin/ECP changed")
    if len(molecule["_atm"]) != atoms:
        raise ValueError("Checkpoint atom count differs")
    if any(np.iscomplexobj(payload[name]) for name in ("mo_coeff", "mo_energy", "mo_occ")):
        raise ValueError("Complex checkpoint arrays are not the archived real RKS state")
    arrays = {name: np.array(payload[name], dtype=np.float64, copy=True)
              for name in ("mo_coeff", "mo_energy", "mo_occ")}
    if (arrays["mo_coeff"].shape != (basis_functions, basis_functions)
            or arrays["mo_energy"].shape != (basis_functions,)
            or arrays["mo_occ"].shape != (basis_functions,)
            or not all(np.isfinite(value).all() for value in arrays.values())
            or not np.isin(arrays["mo_occ"], [0., 2.]).all()
            or float(arrays["mo_occ"].sum()) != electrons
            or not math.isfinite(float(payload["energy_hartree"]))):
        raise ValueError("Checkpoint complete finite closed-shell MO space differs")
    dm = (arrays["mo_coeff"] * arrays["mo_occ"]) @ arrays["mo_coeff"].T
    if not np.isfinite(dm).all() or float(np.max(np.abs(dm-dm.T))) > 1e-12:
        raise ValueError("Checkpoint density is not finite symmetric")
    arrays["density"] = dm
    metadata = {"scope": "INITIAL_GUESS_ONLY_NOT_CONVERGENCE_CERTIFICATION",
        "geometry_basis_payload_exact": True, "atoms": atoms,
        "basis_functions": basis_functions, "electrons_from_occupations": electrons,
        "occupied_orbitals": int(np.count_nonzero(arrays["mo_occ"])),
        "native_checkpoint_energy_hartree_NOT_CERTIFIED": float(payload["energy_hartree"]),
        "historical_cached_J_VEFF_Fock_available": False,
        "density_symmetry_max_abs_error": float(np.max(np.abs(dm-dm.T))),
        "array_payload_sha256": {name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                                 for name, value in arrays.items()}}
    return arrays, metadata


def compare_same_density_veff(mf, dm, reference_dm, *, native_get_veff, np):
    """A NEW incremental reference, never a reconstruction of historical J.

    The previous full reference VEFF and present incremental VEFF are both
    actually computed in this invocation.  All AO matrices stay local.
    """
    hcore = mf.get_hcore(mf.mol)
    overlap = mf.get_ovlp(mf.mol)
    reference = native_get_veff(mf, mf.mol, reference_dm, dm_last=None, vhf_last=None)
    incremental = native_get_veff(mf, mf.mol, dm, dm_last=reference_dm, vhf_last=reference)
    full = native_get_veff(mf, mf.mol, dm, dm_last=None, vhf_last=None)
    full_fock = mf.get_fock(h1e=hcore, s1e=overlap, vhf=full, dm=dm,
                           cycle=-1, diis=None, level_shift_factor=0, damp_factor=0)
    incremental_fock = mf.get_fock(h1e=hcore, s1e=overlap, vhf=incremental, dm=dm,
                                  cycle=-1, diis=None, level_shift_factor=0, damp_factor=0)
    result = {"reference_origin": "NEW_IN_PROCESS_FULL_REFERENCE_NOT_HISTORICAL_CACHED_OPERATOR",
        "same_density_for_both_current_evaluations": True,
        "reference_density_difference_Frobenius": float(np.linalg.norm(dm-reference_dm)),
        "full_energy_hartree": float(mf.energy_tot(dm, hcore, full)),
        "incremental_energy_hartree": float(mf.energy_tot(dm, hcore, incremental)),
        "VEFF_max_abs_difference_hartree": float(np.max(np.abs(full-incremental))),
        "Fock_max_abs_difference_hartree": float(np.max(np.abs(full_fock-incremental_fock)))}
    result["energy_full_minus_incremental_hartree"] = result["full_energy_hartree"]-result["incremental_energy_hartree"]
    for label, value in (("full", full), ("incremental", incremental)):
        vj = getattr(value, "vj", None)
        if vj is not None:
            result[label+"_J_Frobenius_norm"] = float(np.linalg.norm(vj))
    if getattr(full, "vj", None) is not None and getattr(incremental, "vj", None) is not None:
        result["J_max_abs_difference_hartree"] = float(np.max(np.abs(full.vj-incremental.vj)))
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise ValueError("Nonfinite same-density operator comparison")
    return result
