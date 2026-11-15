# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded PySCF 2.13 PBE grid response using the original Becke formulas.

The Becke partition derivative below is copied from PySCF 2.13.0 grad/rks.py,
with its atom-grid loop split BEFORE allocating (natm,natm,3,ngrids) dpbecke.
The GGA AO/weight/coordinate response is summed over the same blocks with the
explicit original grid-owner atom. No grid, XC, basis, pruning, or response
term is removed. Installed library and vendor files are never changed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import hashlib
import inspect
from pathlib import Path

import numpy
import pyscf
from pyscf import gto
from pyscf.dft import gen_grid, numint, radi
from pyscf.grad import rks as native_rks

DESIGN = {
    'schema': 'research4-bounded-pbe-full-grid-response-v1',
    'default_grid_block_size': 2048,
    'AO_mask_alignment': 56,
    'effective_block_rule': 'Floor requested size to a multiple of native BLKSIZE when >=BLKSIZE; otherwise use requested tiny block',
    'supported_pyscf_version': '2.13.0',
    'supported_xc': 'PBE', 'supported_xc_type': 'GGA',
    'supported_partition': 'original_becke',
    'supported_density': 'one real closed-shell Hermitian RKS density',
    'quadrature': 'Original atomic grid/radial/pruning/Becke formulas, weights, nuclear grid-owner identity retained',
    'responses': ['XC nuclear AO derivative', 'Becke weight response', 'grid-coordinate response'],
    'unsupported_behavior': 'Explicit exception; never fall back to unbounded full-atom response',
    'partial_blocks_are_scientific_checkpoints': False,
}
_NATIVE_FUNCTION = native_rks.get_vxc_full_response
_NATIVE_PARTITION = native_rks.grids_response_becke
_INSTALLATION = None
_LAST_AUDIT = {}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    """Stable before/after install; safe during a preparation-only phase."""
    return {'design': copy.deepcopy(DESIGN), 'adapter_sha256': _sha(__file__),
        'pyscf_version': pyscf.__version__,
        'native_rks_file_sha256': _sha(native_rks.__file__),
        'native_partition_source_sha256': hashlib.sha256(inspect.getsource(_NATIVE_PARTITION).encode()).hexdigest(),
        'native_full_response_source_sha256': hashlib.sha256(inspect.getsource(_NATIVE_FUNCTION).encode()).hexdigest(),
        'native_AO_mask_BLKSIZE': int(numint.BLKSIZE)}


def effective_block_size(requested):
    if isinstance(requested, bool) or not isinstance(requested, (int, numpy.integer)) or requested < 1:
        raise ValueError('A positive integer maximum grid block is required')
    if numint.BLKSIZE != DESIGN['AO_mask_alignment']:
        raise RuntimeError('Untested PySCF AO mask block size')
    return int(requested // numint.BLKSIZE * numint.BLKSIZE) if requested >= numint.BLKSIZE else int(requested)


def memory_bound(atom_count, nao, block_size=2048):
    count = effective_block_size(block_size)
    dp = (int(atom_count), int(atom_count), 3, count)
    ao = (10, count, int(nao))
    return {'requested_block_size': int(block_size), 'effective_block_size': count,
        'partition_derivative_shape_bound': list(dp), 'partition_derivative_bytes_bound': int(numpy.prod(dp)) * 8,
        'AO_derivative_shape_bound': list(ao), 'AO_derivative_bytes_bound': int(numpy.prod(ao)) * 8,
        'partition_freed_before_AO_evaluation': True,
        'scope': 'These dominant adapter tensors only; not a bound on complete process RSS'}


def _validate_grids(grids):
    if pyscf.__version__ != DESIGN['supported_pyscf_version']:
        raise RuntimeError('This adapter is validated only for PySCF 2.13.0')
    if grids.becke_scheme is not gen_grid.original_becke:
        raise NotImplementedError('Only the original Becke partition is supported')
    if grids.radii_adjust not in (None, radi.treutler_atomic_radii_adjust, radi.becke_atomic_radii_adjust):
        raise NotImplementedError('Untested radii-adjustment scheme')


def get_last_audit():
    return copy.deepcopy(_LAST_AUDIT)


def _record_partition(owner, ngrid, shape):
    if _LAST_AUDIT.get('active'):
        _LAST_AUDIT['partition_blocks_generated'] += 1
        _LAST_AUDIT['largest_partition_derivative_bytes'] = max(
            _LAST_AUDIT['largest_partition_derivative_bytes'], int(numpy.prod(shape)) * 8)
        _LAST_AUDIT['largest_partition_derivative_shape'] = list(shape) if ngrid >= _LAST_AUDIT['largest_partition_grid_block'] else _LAST_AUDIT['largest_partition_derivative_shape']
        _LAST_AUDIT['largest_partition_grid_block'] = max(_LAST_AUDIT['largest_partition_grid_block'], int(ngrid))


def iter_grid_response_blocks(grids, block_size=2048):
    """Yield (original_owner_atom, coords, weight, weight_response) blocks."""
    _validate_grids(grids)
    yield from _iter_becke_blocks(grids, effective_block_size(block_size))


# The byte-preserved Becke formulas are appended below by the controlled source
# extraction used to create this module. The final module contains their full
# auditable implementation and performs no source generation at runtime.

def _iter_becke_blocks(grids, block_size):
    mol = grids.mol
    atom_grids_tab = grids.gen_atomic_grids(mol, grids.atom_grid,
                                            grids.radi_method,
                                            grids.level, grids.prune)
    atm_coords = numpy.asarray(mol.atom_coords() , order='C')
    atm_dist = gto.inter_distance(mol, atm_coords)

    def _radii_adjust(mol, atomic_radii):
        charges = mol.atom_charges()
        if grids.radii_adjust == radi.treutler_atomic_radii_adjust:
            rad = numpy.sqrt(atomic_radii[charges]) + 1e-200
        elif grids.radii_adjust == radi.becke_atomic_radii_adjust:
            rad = atomic_radii[charges] + 1e-200
        else:
            fadjust = lambda i, j, g: g
            gadjust = lambda *args: 1
            return fadjust, gadjust

        rr = rad.reshape(-1,1) * (1./rad)
        a = .25 * (rr.T - rr)
        a[a<-.5] = -.5
        a[a>0.5] = 0.5

        def fadjust(i, j, g):
            return g + a[i,j]*(1-g**2)

        #: d[g + a[i,j]*(1-g**2)] /dg = 1 - 2*a[i,j]*g
        def gadjust(i, j, g):
            return 1 - 2*a[i,j]*g
        return fadjust, gadjust

    fadjust, gadjust = _radii_adjust(mol, grids.atomic_radii)

    def get_smooth_function_derivative(mu):
        if grids.becke_scheme == gen_grid.original_becke:
            p0 = mu
            p1 = (3 - p0**2) * p0 * .5
            p2 = (3 - p1**2) * p1 * .5
            p3 = (3 - p2**2) * p2 * .5
            t_uab = 27./8. * (1-p2**2) * (1-p1**2) * (1-p0**2)
            return p3, t_uab
        elif grids.becke_scheme == gen_grid.stratmann:
            a = .64
            ma = mu/a
            ma2 = ma * ma
            g1 = (1/16.)*(ma*(35 + ma2*(-35 + ma2*(21 - 5 *ma2))))
            g1[mu<=-a] = -1
            g1[mu>= a] =  1

            t_uab = (35.0/16.0 / a) * (1.0 - ma2 * (3.0 - ma2 * (3.0 - ma2)))
            t_uab[mu<=-a] = 0
            t_uab[mu>= a] = 0
            return g1, t_uab
        else:
            raise NotImplementedError(f"Grid response for becke_scheme = {grids.becke_scheme} is not implemented")

    def gen_grid_partition(coords, atom_id):
        ngrids = coords.shape[0]
        grid_dist = []
        grid_norm_vec = []
        for ia in range(mol.natm):
            v = (atm_coords[ia] - coords).T
            normv = numpy.linalg.norm(v,axis=0) + 1e-200
            v /= normv
            grid_dist.append(normv)
            grid_norm_vec.append(v)

        def get_du(ia, ib):  # JCP 98, 5612 (1993); (B10)
            uab = atm_coords[ia] - atm_coords[ib]
            duab = 1./atm_dist[ia,ib] * grid_norm_vec[ia]
            duab-= uab[:,None]/atm_dist[ia,ib]**3 * (grid_dist[ia]-grid_dist[ib])
            return duab

        pbecke = numpy.ones((mol.natm,ngrids))
        dpbecke = numpy.zeros((mol.natm,mol.natm,3,ngrids))
        for ia in range(mol.natm):
            for ib in range(ia):
                g = 1/atm_dist[ia,ib] * (grid_dist[ia]-grid_dist[ib])
                g_radii_adjusted = fadjust(ia, ib, g)
                smoothed, t_uab = get_smooth_function_derivative(g_radii_adjusted)
                t_uab *= 0.5 * gadjust(ia, ib, g)

                s_uab = .5 * (1 - smoothed + 1e-200)
                s_uba = .5 * (1 + smoothed + 1e-200)
                pbecke[ia] *= s_uab
                pbecke[ib] *= s_uba
                pt_uab =-t_uab / s_uab
                pt_uba = t_uab / s_uba

# * When grid is on atom ia/ib, ua/ub == 0, d_uba/d_uab may have huge error
#   How to remove this error?
                duab = get_du(ia, ib)
                duba = get_du(ib, ia)
                if ia == atom_id:
                    dpbecke[ia,ia] += pt_uab * duba
                    dpbecke[ia,ib] += pt_uba * duba
                else:
                    dpbecke[ia,ia] += pt_uab * duab
                    dpbecke[ia,ib] += pt_uba * duab

                if ib == atom_id:
                    dpbecke[ib,ib] -= pt_uba * duab
                    dpbecke[ib,ia] -= pt_uab * duab
                else:
                    dpbecke[ib,ib] -= pt_uba * duba
                    dpbecke[ib,ia] -= pt_uab * duba

# * JCP 98, 5612 (1993); (B8) (B10) miss many terms
                if ia != atom_id and ib != atom_id:
                    ua_ub = grid_norm_vec[ia] - grid_norm_vec[ib]
                    ua_ub /= atm_dist[ia,ib]
                    dpbecke[atom_id,ia] -= pt_uab * ua_ub
                    dpbecke[atom_id,ib] -= pt_uba * ua_ub

        for ia in range(mol.natm):
            dpbecke[:,ia] *= pbecke[ia]

        return pbecke, dpbecke

    natm = mol.natm
    for ia in range(natm):
        base_coords, base_vol = atom_grids_tab[mol.atom_symbol(ia)]
        for start in range(0, len(base_coords), block_size):
            stop = min(start + block_size, len(base_coords))
            coords = base_coords[start:stop] + atm_coords[ia]
            vol = base_vol[start:stop]
            _record_partition(ia, len(coords), (natm, natm, 3, len(coords)))
            pbecke, dpbecke = gen_grid_partition(coords, ia)
            z = 1./pbecke.sum(axis=0)
            w1 = dpbecke[:,ia] * z
            w1 -= pbecke[ia] * z**2 * dpbecke.sum(axis=1)
            w1 *= vol
            w0 = vol * pbecke[ia] * z
            # Release the natm-squared tensor before AO evaluation and before
            # the next generator block allocates a new partition tensor.
            del dpbecke, pbecke, z
            yield ia, coords, w0, w1
            del coords, vol, w0, w1


def get_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                          max_memory=2000, verbose=None, *, block_size=2048, progress=None):
    """Original PBE GGA full nuclear grid response, summed in bounded blocks."""
    global _LAST_AUDIT
    _validate_grids(grids)
    if str(xc_code).strip().upper() != 'PBE' or ni._xc_type(xc_code) != 'GGA':
        raise NotImplementedError('Only the validated pure PBE GGA task is supported')
    if ni.libxc.is_hybrid_xc(xc_code):
        raise NotImplementedError('Hybrid functionals are not supported')
    density = numpy.asarray(dms)
    if (relativity != 0 or hermi != 1 or density.ndim != 2
            or density.shape != (mol.nao, mol.nao)
            or numpy.iscomplexobj(density) or not numpy.isfinite(density).all()):
        raise NotImplementedError('Only one finite, real Hermitian closed-shell RKS density is supported')
    effective = effective_block_size(block_size)
    coordinates = numpy.ascontiguousarray(mol.atom_coords(), dtype='<f8')
    _LAST_AUDIT = {
        'geometry_hash': hashlib.sha256(coordinates.tobytes()).hexdigest(),
        'geometry_hash_definition': "SHA256 of contiguous little-endian float64 (<f8) Bohr Cartesian bytes, atom order preserved",
        'coordinates_shape': list(coordinates.shape), 'atom_count': int(mol.natm), 'nao': int(mol.nao),
        'requested_block_size': int(block_size), 'effective_block_size': effective,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'active': True, 'completed': False, 'partition_blocks_generated': 0, 'blocks_done': 0,
        'grid_points_done': 0, 'largest_partition_grid_block': 0,
        'largest_partition_derivative_bytes': 0, 'largest_partition_derivative_shape': [],
        'largest_AO_derivative_bytes': 0, 'largest_AO_derivative_shape': [],
        'owner_atoms_seen': [], 'responses_retained': list(DESIGN['responses']),
        'max_memory_hint_MB': float(max_memory),
        'bound': memory_bound(mol.natm, mol.nao, int(block_size)),
    }
    try:
        # Retain native TaggedArray MO metadata and its native rho evaluator;
        # numpy.asarray above is used only for input validation/contractions.
        make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
        if nset != 1:
            raise NotImplementedError('Multiple density matrices are not supported')
        ao_loc = mol.ao_loc_nr()
        excsum = numpy.zeros((mol.natm, 3))
        vmat = numpy.zeros((3, nao, nao))
        vtmp = numpy.zeros_like(vmat)
        owners = set()
        for owner, coords, weight, weight1 in iter_grid_response_blocks(grids, int(block_size)):
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=2, non0tab=mask, cutoff=grids.cutoff)
            if ao.shape != (10, len(coords), nao) or len(coords) > effective:
                raise RuntimeError('Unexpected or unbounded AO derivative allocation')
            _LAST_AUDIT['largest_AO_derivative_bytes'] = max(_LAST_AUDIT['largest_AO_derivative_bytes'], int(ao.nbytes))
            if ao.nbytes >= _LAST_AUDIT['largest_AO_derivative_bytes']:
                _LAST_AUDIT['largest_AO_derivative_shape'] = list(ao.shape)
            rho = make_rho(0, ao[:4], mask, 'GGA')
            exc, vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype='GGA')[:2]
            wv = weight * vxc
            wv[0] *= .5
            vtmp.fill(0)
            native_rks._gga_grad_sum_(vtmp, mol, ao, wv, mask, ao_loc)
            vmat += vtmp
            # Both terms are unchanged from PySCF. owner is the original
            # atom, not the block enumeration index.
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho[0], weight1)
            excsum[owner] += numpy.einsum('xij,ji->x', vtmp, density) * 2
            owners.add(int(owner))
            _LAST_AUDIT.update(blocks_done=_LAST_AUDIT['blocks_done'] + 1,
                grid_points_done=_LAST_AUDIT['grid_points_done'] + len(coords),
                owner_atoms_seen=sorted(owners))
            if progress is not None:
                progress({'phase': 'grid_response_block', 'owner_atom': int(owner),
                          'block_grid_points': int(len(coords)),
                          'blocks_done': _LAST_AUDIT['blocks_done'],
                          'grid_points_done': _LAST_AUDIT['grid_points_done']})
            # In particular do not retain the preceding full AO buffer while
            # requesting a new Becke-partition block from the generator.
            del ao, rho, exc, vxc, wv, mask, coords, weight, weight1
        if owners != set(range(mol.natm)):
            raise RuntimeError('Incomplete original grid-owner atom coverage')
        if not numpy.isfinite(excsum).all() or not numpy.isfinite(vmat).all():
            raise FloatingPointError('Nonfinite full grid response')
        _LAST_AUDIT.update(completed=True, finished_at_utc=datetime.now(timezone.utc).isoformat())
        return excsum, -vmat
    except Exception as error:
        _LAST_AUDIT.update(completed=False, error_type=type(error).__name__, error=str(error),
                           failed_at_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        _LAST_AUDIT['active'] = False


def install(block_size=2048, progress=None):
    """Install the bounded call in this process only; never edit library files."""
    global _INSTALLATION
    effective = effective_block_size(block_size)
    if pyscf.__version__ != DESIGN['supported_pyscf_version']:
        raise RuntimeError('This adapter is validated only for PySCF 2.13.0')
    if _INSTALLATION is not None:
        if _INSTALLATION['requested'] != int(block_size):
            raise RuntimeError('Adapter is already installed with a different block size')
        return {'requested_block_size': int(block_size), 'effective_block_size': effective,
                'provenance': provenance()}
    if native_rks.get_vxc_full_response is not _NATIVE_FUNCTION:
        raise RuntimeError('Another grid-response replacement is already active')
    def bounded(ni, mol, grids, xc_code, dms, relativity=0, hermi=1, max_memory=2000, verbose=None):
        return get_vxc_full_response(ni, mol, grids, xc_code, dms, relativity, hermi,
            max_memory, verbose, block_size=int(block_size), progress=progress)
    _INSTALLATION = {'requested': int(block_size), 'function': bounded}
    native_rks.get_vxc_full_response = bounded
    return {'requested_block_size': int(block_size), 'effective_block_size': effective,
            'provenance': provenance()}


def uninstall():
    """Restore the exact original function if this adapter still owns it."""
    global _INSTALLATION
    if _INSTALLATION is None:
        return
    if native_rks.get_vxc_full_response is not _INSTALLATION['function']:
        raise RuntimeError('Active grid-response function changed outside this adapter')
    native_rks.get_vxc_full_response = _NATIVE_FUNCTION
    _INSTALLATION = None

