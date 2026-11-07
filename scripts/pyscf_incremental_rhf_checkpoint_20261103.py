# RHF Hessian algebra derived from PySCF:
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Original RHF Hessian author: Qiming Sun <osirpt.sun@gmail.com>
# Research4 modification: atom-output partition and immutable partial restart.

"""Restartable pure-PBE RHF partial Hessian, with immutable atom-sized units.

No SCF, gradient, integrals, or quantum work starts on import.  The numerical
algebra derives from PySCF's Apache-2.0 RHF Hessian (Qiming Sun and contributors).
See reports/pyscf-spinosin-pure-j-20261101-v4/upstream-attribution/.  This changes
only the scheduling/persistence of J/core/overlap terms; RKS XC, H1, CPHF and the
full-system Hessian assembly remain the caller's stock implementations.

This module is an engineering facility, not evidence of a completed frequency
calculation.  Real cross-process full-Hessian validation is a separate gate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "research4-incremental-pure-pbe-rhf-v10"
PINNED = {
    "hessian/rhf.py": "2d986bfc01aac98447034a207df639b3d6897f67ce1ab2a013fe2d0853c85de5",
    "hessian/rks.py": "d61dbd1727b286845737853b92e1812a3e6a966dca29a9a8a79ff5e9a7725c03",
    "scf/_vhf.py": "ef0a04f43338780a7f57d9268f9634c1079ae989ba3030d6154c8612798f82ba",
    "lib/libcvhf.so": "5d7461e3976fd40fce7c9ba571e45b01830dd018d75927e389c32a6f4d09df0c",
    "lib/deps/lib/libcint.so": "950bb0a297507369dce95b9013ef2c5b64b0fa36a0018c303c98e95efce7db7e",
    "lib/libcgto.so": "337d956486a156f1aa7ca636ac5a91c6277058ffedb6db2efe714fee65f726e0",
}
DERIVATIVE_GUARDS = ("int2e_ipip1ipip2", "int2e_ip1ip2", "int2e_ipvip1ipvip2")


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    a = np.asarray(value)
    if a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError("Finite real numeric arrays required")
    return {"shape": list(a.shape), "dtype": str(a.dtype), "bytes": int(a.nbytes),
            "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


def _source_identity(rhf):
    directory = Path(rhf.__file__).resolve().parents[1]
    actual = {name: sha(directory / name) for name in PINNED}
    if actual != PINNED:
        raise ValueError("Unreviewed PySCF integral/Hessian source or library")
    return {"PySCF_dependencies": actual, "incremental_module_sha256": sha(__file__),
            "python_major_minor": list(sys.version_info[:2]), "numpy": np.__version__}


def _check_stock_callable(function, rhf):
    if (function.__name__ != "_partial_hess_ejk" or
            Path(inspect.getsourcefile(function)).resolve() != Path(rhf.__file__).resolve()):
        raise ValueError("The original unmodified stock RHF helper is required")


def _require_plain_pbe(hobj, rhf):
    mf, mol = hobj.base, hobj.mol
    if mf.mol is not mol or str(mf.xc).strip().upper() != "PBE" or getattr(mf, "with_df", None) is not None:
        raise ValueError("Only full-system non-DF plain PBE is supported")
    ni = mf._numint
    if (ni._xc_type(mf.xc) != "GGA" or ni.libxc.is_hybrid_xc(mf.xc) or
            any(float(x) != 0 for x in ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin))):
        raise ValueError("Zero hybrid and range-separation coefficients required")
    if any(hasattr(rhf._vhf.libcvhf, name) for name in DERIVATIVE_GUARDS):
        raise ValueError("Unreviewed derivative optimizer ABI; suffix screening is not enabled")
    if getattr(mf, "disp", None) is not None or getattr(mf, "nlc", "") not in ("", None, 0):
        raise ValueError("This scoped helper is restricted to the reviewed plain-PBE model")


def runtime_state_fingerprint(hobj):
    """Bind actual complete MOs, initialized grid and physical Mole state.

    Only env[17], PySCF's last-rinv-atom bookkeeping, is normalized on a copy.
    No geometry, basis, electron space, grid, or live object is modified.
    """
    mf, mol = hobj.base, hobj.mol
    if not bool(mf.converged) or not np.isfinite(mf.e_tot):
        raise ValueError("A verified, converged source SCF state is required")
    nao, natm = int(mol.nao_nr()), int(mol.natm)
    if mol.spin != 0 or mol.cart or mol.has_ecp() or getattr(mol, "symmetry", False):
        raise ValueError("Only nonsymmetric closed-shell all-electron spherical states are supported")
    if not set(map(int, mol.atom_charges())).issubset({1, 6, 8}):
        raise ValueError("This initial implementation is limited to reviewed C/H/O molecules")
    if (mf.max_memory != 3000 or hobj.max_memory != 3000 or not mf.direct_scf or
            mf.direct_scf_tol != 1e-13 or getattr(mf, "_eri", None) is not None or mol.incore_anyway):
        raise ValueError("Reviewed direct-SCF / 3000 MB Hessian memory context required")
    coeff, energy, occ = map(np.asarray, (mf.mo_coeff, mf.mo_energy, mf.mo_occ))
    if coeff.shape != (nao, nao) or energy.shape != (nao,) or occ.shape != (nao,):
        raise ValueError("Full square MO coefficient/energy/occupation arrays required")
    if not np.isin(occ, [0, 2]).all() or float(occ.sum()) != mol.nelectron:
        raise ValueError("Restricted occupations disagree with electron count")
    if any(a.dtype != np.float64 for a in (coeff, energy, occ)):
        raise ValueError("Reviewed real float64 MO arrays required")
    env = np.array(mol._env, copy=True)
    if (env.ndim != 1 or len(env) <= 17 or not np.isfinite(env[17]) or
            env[17] != int(env[17]) or not 0 <= env[17] < natm):
        raise ValueError("Unknown last-rinv-atom environment marker")
    env[17] = 0
    coords, weights = np.asarray(mf.grids.coords), np.asarray(mf.grids.weights)
    if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords) or weights.shape != (len(coords),):
        raise ValueError("Actual initialized full SCF quadrature required")
    if getattr(hobj, "grids", None) not in (None, mf.grids):
        raise ValueError("A separate Hessian quadrature is outside the reviewed contract")
    grid = {"coords": fingerprint(coords), "weights": fingerprint(weights)}
    for name in ("non0tab", "screen_index", "atm_idx", "quadrature_weights"):
        value = getattr(mf.grids, name, None)
        grid[name] = None if value is None else fingerprint(value)
    settings = {}
    for name in ("conv_tol", "conv_tol_cpscf", "max_cycle", "direct_scf_tol", "small_rho_cutoff", "max_memory"):
        value = getattr(mf, name)
        settings[name] = float(value) if isinstance(value, (np.floating, float)) else int(value)
    settings.update(xc=str(mf.xc), numint_cutoff=float(mf._numint.cutoff),
                    numint_omega=getattr(mf._numint, "omega", None),
                    grid_level=int(mf.grids.level), grid_cutoff=float(mf.grids.cutoff),
                    hessian_max_memory=float(hobj.max_memory))
    return {"molecule": {"atm": fingerprint(mol._atm), "bas": fingerprint(mol._bas),
               "env_except_17": fingerprint(env), "coords_bohr": fingerprint(mol.atom_coords(unit="Bohr")),
               "charges": fingerprint(mol.atom_charges()), "natm": natm, "nao": nao,
               "nbas": int(mol.nbas), "charge": int(mol.charge), "spin": int(mol.spin),
               "nelectron": int(mol.nelectron), "aoslices": np.asarray(mol.aoslice_by_atom(), int).tolist()},
            "MO_coeff": fingerprint(coeff), "MO_energy": fingerprint(energy), "MO_occ": fingerprint(occ),
            "energy_hartree": float(mf.e_tot), "grid": grid, "settings": settings}


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _immutable_write(path, content):
    """Publish by link(2): atomic and fails rather than replacing any old file."""
    path = Path(path)
    temporary = path.with_name("." + path.name + ".writing-" + uuid.uuid4().hex)
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()
    _fsync_directory(path.parent)


def _immutable_array(path, array):
    import io
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    _immutable_write(path, buffer.getvalue())


class CheckpointSegmentComplete(RuntimeError):
    """Normal finite segment ended; the complete RHF/Hessian is NOT done."""
    def __init__(self, progress):
        self.progress = progress
        super().__init__("Incremental RHF segment saved: " + json.dumps(progress, sort_keys=True))


class IncrementalPureJContext:
    """Scoped, locked replacement of the stock helper for a single exact state.

    ``state_binding`` must be the stable identity returned by the verified state
    loader, without consumer PID/proof data.  The caller owns SCF/gradient-source
    qualification, resource supervision and same-state gradient acceptance.
    ``execution_evidence`` is recorded per unit and never part of state identity.
    This low-level cache does not validate supervisor proof/ledger files.  The
    scientific caller must verify every producer/consumer execution record; a
    complete array unit alone never certifies a closed, successful QM phase.
    """
    def __init__(self, hobj, cache_dir, state_binding, max_new_units=10, *, rhf_module=None,
                 execution_evidence=None):
        if isinstance(max_new_units, bool) or not isinstance(max_new_units, int) or max_new_units < 0:
            raise ValueError("Finite nonnegative integer max_new_units required")
        if not isinstance(state_binding, dict) or not state_binding:
            raise ValueError("Nonempty verified immutable snapshot binding required")
        self.hobj, self.cache_dir = hobj, Path(cache_dir).resolve()
        self.state_binding = json.loads(canonical(state_binding))
        self.execution_evidence = json.loads(canonical(execution_evidence or {}))
        self.max_new_units, self.new_units = max_new_units, 0
        self.rhf, self.active, self.lock = rhf_module, False, None
        self.original = self.wrapper = None
        self.records = []

    def __enter__(self):
        if self.rhf is None:
            from pyscf.hessian import rhf
            self.rhf = rhf
        original = self.rhf._partial_hess_ejk
        if self.active or getattr(original, "_research4_incremental_pure_j", False) or getattr(original, "_research4_pure_j", False):
            raise RuntimeError("Do not nest pure-J helper contexts")
        _check_stock_callable(original, self.rhf)
        _require_plain_pbe(self.hobj, self.rhf)
        self.runtime = runtime_state_fingerprint(self.hobj)
        sources = _source_identity(self.rhf)
        self.natm, self.nao = int(self.hobj.mol.natm), int(self.hobj.mol.nao_nr())
        self.aoslices = np.asarray(self.hobj.mol.aoslice_by_atom(), int)
        if self.aoslices.shape != (self.natm, 4):
            raise ValueError("Invalid atom AO/shell layout")
        effective = {"stable_snapshot": self.state_binding, "actual_runtime": self.runtime, "sources": sources}
        self.binding_sha256 = hashlib.sha256(canonical(effective)).hexdigest()
        self.header = {"schema": SCHEMA, "binding": effective, "binding_sha256": self.binding_sha256,
                       "natm": self.natm, "nao": self.nao, "total_units": 2*self.natm,
                       "unit_order": "ALL_J_DIAGONALS_THEN_COMPLETE_LOWER_E1_EJ_ROWS",
                       "scope": "RHF_PARTIAL_ONLY_NOT_RKS_XC_OR_COMPLETE_HESSIAN"}
        try:
            self._open_store()
            self.original = original
            signature = inspect.signature(original)
            def wrapped(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                values = bound.arguments
                if values["with_k"] is not False:
                    return original(*args, **kwargs)
                self._validate_call(values)
                return self._compute(values)
            wrapped._research4_incremental_pure_j = True
            self.wrapper, self.active = wrapped, True
            self.rhf._partial_hess_ejk = wrapped
            return self
        except BaseException:
            self._close_lock()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        changed = self.rhf._partial_hess_ejk is not self.wrapper
        self.rhf._partial_hess_ejk = self.original
        self.active = False
        self._close_lock()
        if changed and exc_type is None:
            raise RuntimeError("Another writer replaced the scoped RHF helper")
        return False

    def _close_lock(self):
        if self.lock is not None:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None

    def _open_store(self):
        if not self.cache_dir.is_relative_to(ROOT / "outputs") or self.cache_dir == ROOT / "outputs":
            raise ValueError("Incremental payloads require a dedicated ignored outputs/ subdirectory")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock = (self.cache_dir / "cache.lock").open("a+b")
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        header = self.cache_dir / "header.json"
        if header.is_symlink():
            raise ValueError("Symlink cache header rejected")
        if header.exists():
            if json.loads(header.read_text()) != self.header:
                raise ValueError("Checkpoint physical/source/snapshot binding differs")
        else:
            if any(p.name != "cache.lock" for p in self.cache_dir.iterdir()):
                raise ValueError("Payloads exist without their complete immutable header")
            _immutable_write(header, canonical(self.header) + b"\n")
        self.header_sha256 = sha(header)
        self.units = self.cache_dir / "units"
        if self.units.is_symlink():
            raise ValueError("Symlink units directory rejected")
        self.units.mkdir(exist_ok=True)
        _fsync_directory(self.cache_dir)
        self._load_prefix()

    def _unit_name(self, index):
        kind, atom = ("diag", index) if index < self.natm else ("row", index-self.natm)
        return f"{index:06d}-{kind}-{atom:04d}"

    def _expected(self, index):
        if index < self.natm:
            return {"jdiag": (3, 3)}
        return {"e1": (index-self.natm+1, 3, 3), "ej": (index-self.natm+1, 3, 3)}

    def _read_record(self, attempt, index):
        path = attempt / "complete.json"
        if path.is_symlink() or path.stat().st_size > 131072:
            raise ValueError("Unsafe checkpoint completion record")
        record = json.loads(path.read_text())
        expected = self._expected(index)
        if (record.get("schema") != SCHEMA or record.get("header_sha256") != self.header_sha256 or
                record.get("binding_sha256") != self.binding_sha256 or record.get("index") != index or
                record.get("unit_name") != self._unit_name(index) or set(record.get("arrays", {})) != set(expected)):
            raise ValueError("Checkpoint completion identity/coverage differs")
        arrays = {}
        for name, shape in expected.items():
            entry = record["arrays"][name]
            filename = name + ".npy"
            payload = attempt / filename
            if entry.get("file") != filename or payload.is_symlink() or payload.stat().st_size > 2**20:
                raise ValueError("Unsafe checkpoint payload")
            if sha(payload) != entry["file_sha256"]:
                raise ValueError("Checkpoint payload bytes changed")
            value = np.load(payload, allow_pickle=False)
            if value.dtype != np.float64 or value.shape != shape or fingerprint(value) != entry["payload"]:
                raise ValueError("Checkpoint payload dtype/shape/fingerprint differs")
            value.flags.writeable = False
            arrays[name] = value
        return {"record": record, "arrays": arrays, "attempt": str(attempt)}

    def _load_prefix(self):
        names = {self._unit_name(i): i for i in range(2*self.natm)}
        found = {}
        for unit in self.units.iterdir():
            if unit.is_symlink() or not unit.is_dir() or unit.name not in names:
                raise ValueError("Unexpected checkpoint unit path")
            index = names[unit.name]
            completed = []
            for attempt in unit.iterdir():
                if (attempt.is_symlink() or not attempt.is_dir() or not attempt.name.startswith("attempt-") or
                        not attempt.name[8:].isdigit()):
                    raise ValueError("Unexpected checkpoint attempt path")
                if (attempt / "complete.json").exists():
                    completed.append(self._read_record(attempt, index))
            if len(completed) > 1:
                raise ValueError("Duplicate completed immutable checkpoint unit")
            if completed:
                found[index] = completed[0]
        if sorted(found) != list(range(len(found))):
            raise ValueError("Only a complete continuous checkpoint prefix may be restored")
        self.records = [found[i] for i in range(len(found))]

    def progress(self):
        count = len(self.records)
        return {"diag_completed": min(count, self.natm), "row_completed": max(0, count-self.natm),
                "total_units_completed": count, "total_units": 2*self.natm,
                "new_units_this_context": self.new_units, "complete_RHF_helper": count == 2*self.natm,
                "cache_dir": str(self.cache_dir), "binding_sha256": self.binding_sha256}

    def _budget_check(self):
        if len(self.records) < 2*self.natm and self.new_units >= self.max_new_units:
            raise CheckpointSegmentComplete(self.progress())

    def _publish(self, index, arrays):
        if index != len(self.records) or set(arrays) != set(self._expected(index)):
            raise ValueError("Can append only the next complete unit")
        unit = self.units / self._unit_name(index)
        unit.mkdir(exist_ok=True)
        _fsync_directory(self.units)
        serial = max([int(p.name[8:]) for p in unit.iterdir() if p.name.startswith("attempt-") and p.name[8:].isdigit()] or [0])+1
        attempt = unit / f"attempt-{serial:06d}"
        attempt.mkdir()
        _fsync_directory(unit)
        entries = {}
        for name, shape in self._expected(index).items():
            value = np.asarray(arrays[name])
            if value.dtype != np.float64 or value.shape != shape:
                raise ValueError("Bad newly computed checkpoint shape or dtype")
            payload_fingerprint = fingerprint(value)
            path = attempt / (name + ".npy")
            _immutable_array(path, value)
            entries[name] = {"file": path.name, "file_sha256": sha(path), "payload": payload_fingerprint}
        record = {"schema": SCHEMA, "header_sha256": self.header_sha256,
                  "binding_sha256": self.binding_sha256, "index": index, "unit_name": self._unit_name(index),
                  "arrays": entries, "completed_utc": utc(), "execution_evidence": self.execution_evidence,
                  "scope": "ONE_COMPLETE_RHF_UNIT_NOT_A_COMPLETE_HESSIAN"}
        _immutable_write(attempt / "complete.json", canonical(record) + b"\n")
        _fsync_directory(unit)
        self.records.append(self._read_record(attempt, index))
        self.new_units += 1
        self._budget_check()

    def _validate_call(self, values):
        if values["hessobj"] is not self.hobj or runtime_state_fingerprint(self.hobj) != self.runtime:
            raise ValueError("Actual MF/MO/grid/geometry changed during the context")
        for name in ("mo_energy", "mo_coeff", "mo_occ"):
            value = values[name]
            if value is not None and fingerprint(value) != fingerprint(getattr(self.hobj.base, name)):
                raise ValueError("Helper arguments differ from the complete source MO state")
        atoms = values["atmlst"]
        if atoms is not None and list(atoms) != list(range(self.natm)):
            raise ValueError("Complete original atom ordering required; fragments are not supported")
        _require_plain_pbe(self.hobj, self.rhf)

    def _compute(self, values):
        self._budget_check()
        if len(self.records) == 2*self.natm:
            return self._assemble()
        mol, mf, rhf = self.hobj.mol, self.hobj.base, self.rhf
        mocc = mf.mo_coeff[:, mf.mo_occ > 0]
        dm0 = np.dot(mocc, mocc.T)*2
        dme0 = np.einsum("pi,qi,i->pq", mocc, mocc, mf.mo_energy[mf.mo_occ > 0])*2
        # Partition only output AO i; density and all other AO/shell axes remain global.
        while len(self.records) < self.natm:
            self._budget_check()
            atom = len(self.records)
            sh0, sh1, p0, p1 = self.aoslices[atom]
            slice_ = (int(sh0), int(sh1)) + (0, int(mol.nbas))*3
            vj = rhf._get_jk(mol, "int2e_ipip1", 9, "s2kl", ["lk->s1ij", dm0],
                             shls_slice=slice_, vhfopt=None)[0]
            if vj.shape != (9, p1-p0, self.nao):
                raise ValueError("Unexpected sliced global ipip1 output AO contract")
            diagonal = np.einsum("xypq,pq->xy", vj.reshape(3, 3, p1-p0, self.nao), dm0[p0:p1])*2
            self._publish(atom, {"jdiag": diagonal})
        hcore_deriv = self.hobj.hcore_generator(mol)
        s1aa, s1ab, _ = rhf.get_ovlp(mol)
        while len(self.records) < 2*self.natm:
            self._budget_check()
            atom = len(self.records)-self.natm
            sh0, sh1, p0, p1 = self.aoslices[atom]
            slice_ = (int(sh0), int(sh1)) + (0, int(mol.nbas))*3
            vj1 = rhf._get_jk(mol, "int2e_ip1ip2", 9, "s1", ["ji->s1kl", dm0[:, p0:p1]],
                              shls_slice=slice_, vhfopt=None)[0]
            vj2 = rhf._get_jk(mol, "int2e_ipvip1", 9, "s2kl", ["lk->s1ij", dm0],
                              shls_slice=slice_, vhfopt=None)[0]
            if vj1.shape != (9, self.nao, self.nao) or vj2.shape != (9, p1-p0, self.nao):
                raise ValueError("Unexpected row integral output AO contract")
            vj1[:, :, p0:p1] += vj2.transpose(0, 2, 1)*.5
            vj1 = vj1.reshape(3, 3, self.nao, self.nao)
            e1, ej = np.zeros((atom+1, 3, 3)), np.zeros((atom+1, 3, 3))
            ej[atom] += self.records[atom]["arrays"]["jdiag"]
            e1[atom] -= np.einsum("xypq,pq->xy", s1aa[:, :, p0:p1], dme0[p0:p1])*2
            for other in range(atom+1):
                q0, q1 = self.aoslices[other][2:]
                ej[other] += np.einsum("xypq,pq->xy", vj1[:, :, q0:q1], dm0[q0:q1])*4
                e1[other] -= np.einsum("xypq,pq->xy", s1ab[:, :, p0:p1, q0:q1], dme0[p0:p1, q0:q1])*2
                e1[other] += np.einsum("xypq,pq->xy", hcore_deriv(atom, other), dm0)
            self._publish(self.natm+atom, {"e1": e1, "ej": ej})
        return self._assemble()

    def _assemble(self):
        if len(self.records) != 2*self.natm:
            raise ValueError("A full RHF helper cannot be assembled from incomplete units")
        e1 = np.zeros((self.natm, self.natm, 3, 3))
        ej, ek = np.zeros_like(e1), np.zeros_like(e1)
        for atom in range(self.natm):
            row = self.records[self.natm+atom]["arrays"]
            e1[atom, :atom+1] = row["e1"]
            ej[atom, :atom+1] = row["ej"]
            for other in range(atom):
                e1[other, atom] = e1[atom, other].T
                ej[other, atom] = ej[atom, other].T
        # Fresh buffers: the stock RKS caller may modify its e1 result in place.
        return e1, ej, ek
