#!/usr/bin/env python3
"""Import immutable, authenticated IR inputs from the other PySCF branch.

This module never checks out, updates, or executes the upstream worktree. Git
objects are read by commit. A completed calculation is distinct from a qualified
minimum, and an upstream IR failure does not block independent IR repair.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import time

import numpy as np

SCHEMA = 'research4-pyscf-ir-source-v1'
BRANCHES = ('research/pyscf-publication-recovery-20261111',
            'research/pyscf-frequency-repair-20261111')
PUBLICATION_ROOT = 'reports/pyscf_publication_20261111'
FREQUENCY_ROOT = 'reports/pyscf_repair_20261111'
MOLECULES = {'water', 'acid', 'calycosin', 'spinosin'}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def _relative(path):
    """Map old absolute workspace paths to repository-relative report paths."""
    text = str(path)
    if text.startswith('/'):
        if '/reports/' not in text:
            raise ValueError('Source path outside reports: ' + text)
        text = 'reports/' + text.split('/reports/', 1)[1]
    pure = PurePosixPath(text)
    if not text.startswith('reports/') or any(x in ('.', '..') for x in pure.parts):
        raise ValueError('Unsafe report source path: ' + text)
    return str(pure)


class GitBlobs:
    """A batched object reader, so thousands of point arrays need one process."""

    def __init__(self, repo, commit, root=PUBLICATION_ROOT):
        self.repo, self.commit = Path(repo), commit
        entries = _git(repo, 'ls-tree', '-rz', commit, '--', root).split(b'\0')
        self.objects = {}
        for entry in entries:
            if not entry:
                continue
            info, path = entry.split(b'\t', 1)
            mode, kind, oid = info.split()
            if kind == b'blob' and mode in (b'100644', b'100755'):
                self.objects[path.decode()] = oid.decode()
        self.cache = {}

    def read_many(self, paths):
        paths = list(dict.fromkeys(paths))
        missing = [path for path in paths if path not in self.cache]
        if missing:
            for path in missing:
                if path not in self.objects:
                    raise FileNotFoundError('Source commit lacks tracked artifact: ' + path)
            request = ''.join(self.objects[path] + '\n' for path in missing).encode()
            output = subprocess.run(['git', '-C', str(self.repo), 'cat-file', '--batch'],
                input=request, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
            offset = 0
            for path in missing:
                end = output.index(b'\n', offset)
                oid, kind, length = output[offset:end].split()
                if kind != b'blob' or oid.decode() != self.objects[path]:
                    raise ValueError('Git object response mismatch')
                start, size = end+1, int(length)
                blob = output[start:start+size]
                if len(blob) != size or output[start+size:start+size+1] != b'\n':
                    raise ValueError('Truncated Git artifact')
                self.cache[path] = blob
                offset = start+size+1
            if offset != len(output):
                raise ValueError('Unexpected extra Git object bytes')
        return {path: self.cache[path] for path in paths}

    def read(self, path):
        return self.read_many([path])[path]


def _finite(value, shape):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('Incomplete/nonfinite source array; expected ' + str(shape))
    return array


def _electronic(point):
    audit = point.get('electronic_audit', {})
    if point.get('qualified') is not True or point.get('native_converged') is not True:
        return False
    if audit.get('native_converged') is not True or audit.get('qualified') is not True:
        return False
    for name, limit in [('fresh_orbital_gradient_norm', 1e-8),
                        ('fresh_energy_difference_hartree', 1e-10),
                        ('electron_count_error', 1e-7)]:
        value = audit.get(name)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or abs(value) > limit:
            return False
    return True


def authenticate_case(reader, case_path):
    """Validate seals, original point arrays and independent H/D reassembly.

    reader exposes read(path), read_many(paths), and objects (tracked paths).
    The returned private _evidence/_arrays values are consumed by import_case.
    """
    case_path = _relative(case_path)
    prefix = case_path + '/'
    result_path, seal_path = prefix+'result.json', prefix+'COMPLETE.json'
    result_bytes, seal_bytes = reader.read(result_path), reader.read(seal_path)
    case, seal = json.loads(result_bytes), json.loads(seal_bytes)
    if seal.get('result_sha256') != _sha(result_bytes):
        raise ValueError('Stationary-case COMPLETE seal mismatch')
    if case.get('molecule') not in MOLECULES:
        raise ValueError('Unapproved molecule in IR source')
    mode_path, gate_path = prefix+'modes/result.json', prefix+'modes/publication_quality.json'
    mode_bytes, gate_bytes = reader.read(mode_path), reader.read(gate_path)
    mode, gate = json.loads(mode_bytes), json.loads(gate_bytes)
    embedded_gate = case.get('publication_gate')
    # verify() legitimately rewrites this receipt with a new audit timestamp.
    # Scientific flags, arrays, hashes and thresholds must remain identical.
    if not isinstance(embedded_gate,dict) or _canonical({k:v for k,v in embedded_gate.items() if k != 'at_utc'}) != _canonical({k:v for k,v in gate.items() if k != 'at_utc'}):
        raise ValueError('Case publication gate differs from saved independent gate')
    if case.get('analysis', {}).get('input_fingerprint') != mode.get('input_fingerprint'):
        raise ValueError('Case analysis differs from saved mode inputs')
    provenance = mode['input_provenance']
    if _sha(_canonical(provenance)) != mode.get('input_fingerprint'):
        raise ValueError('Mode input fingerprint mismatch')
    manifest = gate.get('authenticated_artifact_sha256')
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('Authenticated original derivative ledger is absent')
    expected = {}
    for path, digest in manifest.items():
        relative = _relative(path)
        if not relative.startswith(prefix):
            raise ValueError('Artifact escaped its sealed stationary case')
        expected[relative] = digest
    for artifact in mode.get('artifacts', {}).values():
        relative = str(PurePosixPath(prefix+'modes') / artifact['path'])
        if not relative.startswith(prefix+'modes/') or '..' in PurePosixPath(relative).parts:
            raise ValueError('Unsafe analysis artifact path')
        if relative in expected and expected[relative] != artifact['sha256']:
            raise ValueError('Conflicting analysis artifact hashes')
        expected[relative] = artifact['sha256']
    blobs = reader.read_many(expected)
    for path, digest in expected.items():
        if _sha(blobs[path]) != digest:
            raise ValueError('Original source artifact hash mismatch: ' + path)
    if expected.get(mode_path) != _sha(mode_bytes):
        raise ValueError('Mode result is absent from authenticated source ledger')
    npz_path = prefix+'modes/analysis_arrays.npz'
    with np.load(io.BytesIO(blobs[npz_path]), allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    symbols = mode['symbols']
    natom, ncart = len(symbols), len(symbols)*3
    coords = _finite(arrays['coordinates_bohr'], (natom, 3))
    if not np.array_equal(_finite(case['coordinates_bohr'], (natom, 3)), coords):
        raise ValueError('Source case geometry and mode geometry differ')
    steps = provenance['steps_bohr']
    if len(steps) < 2 or len(set(steps)) != len(steps) or any(not np.isfinite(x) or x <= 0 for x in steps):
        raise ValueError('Distinct positive source derivative steps required')
    if mode.get('matrix_count') != len(steps):
        raise ValueError('Incomplete derivative step matrix count')
    center_path = _relative(case['center_point_path'])
    if not center_path.startswith(prefix):
        raise ValueError('Center point escaped its source case')
    center_bytes = reader.read(center_path)
    center = json.loads(center_bytes)
    if gate.get('center_point_sha256') != _sha(_canonical(center)):
        raise ValueError('Center point canonical hash mismatch')
    if center.get('symbols') != symbols or not np.array_equal(np.asarray(center['coordinates_bohr']), coords):
        raise ValueError('Center electronic point atom identity differs')
    if center.get('settings') != case.get('method'):
        raise ValueError('Center electronic model differs from sealed case')
    all_points_qualified, fingerprints = _electronic(center), set()
    original_count = 0
    derivatives = []
    for hi, step in enumerate(steps):
        fd_prefix = prefix+f'finite-differences/h{hi}/'
        expected_point_paths = {fd_prefix+f'c{j:03d}-{label}/result.json'
            for j in range(ncart) for label in ('plus', 'minus')}
        actual_point_paths = {path for path in reader.objects
            if path.startswith(fd_prefix) and path.endswith('/result.json')
            and len(path[len(fd_prefix):].split('/')) == 2}
        if actual_point_paths != expected_point_paths:
            raise ValueError('Missing or unexpected unique Cartesian displacement ledger')
        hs, ds = [], []
        for j in range(ncart):
            pair = []
            for sign, label in ((1, 'plus'), (-1, 'minus')):
                point_path = fd_prefix+f'c{j:03d}-{label}/result.json'
                if point_path not in blobs:
                    raise ValueError('Original electronic point lacks authenticated hash')
                point = json.loads(blobs[point_path])
                displaced = coords.copy()
                displaced.flat[j] += sign*step
                if point.get('symbols') != symbols or not np.array_equal(np.asarray(point['coordinates_bohr']), displaced):
                    raise ValueError('Original displacement geometry differs from labeled step')
                if point.get('settings') != center['settings']:
                    raise ValueError('Electronic model differs between displaced points')
                fingerprint = point.get('input_fingerprint')
                if not isinstance(fingerprint, str) or not fingerprint or fingerprint in fingerprints:
                    raise ValueError('Duplicate/missing original point fingerprint')
                fingerprints.add(fingerprint)
                for key, shape, filename in [('coordinates_bohr', (natom,3), 'coordinates_bohr.npy'),
                    ('gradient', (natom,3), 'gradient.npy'), ('dipole_au', (3,), 'dipole_au.npy')]:
                    path = point_path.rsplit('/',1)[0]+'/'+filename
                    if path not in blobs:
                        raise ValueError('Original point array lacks authenticated hash')
                    saved = np.load(io.BytesIO(blobs[path]), allow_pickle=False)
                    if not np.array_equal(_finite(point[key], shape), _finite(saved, shape)):
                        raise ValueError('Original point JSON/NPY array mismatch')
                all_points_qualified &= _electronic(point)
                pair.append(point)
                original_count += 1
            hs.append((np.asarray(pair[0]['gradient'])-np.asarray(pair[1]['gradient'])).ravel()/(2*step))
            ds.append((np.asarray(pair[0]['dipole_au'])-np.asarray(pair[1]['dipole_au']))/(2*step))
        hessian, derivative = np.stack(hs,axis=1), np.stack(ds,axis=1)
        if not np.array_equal(hessian, arrays[f'step_{hi}_raw_hessian']):
            raise ValueError('Hessian differs from original point reassembly')
        if not np.array_equal(derivative, arrays[f'step_{hi}_dipole_derivative_au']):
            raise ValueError('Dipole derivative differs from original point reassembly')
        derivatives.append(derivative)
    if original_count != gate.get('original_displacement_point_count'):
        raise ValueError('Original displacement point count differs from gate')
    fine = int(np.argmin(steps))
    masses = _finite(arrays[f'step_{fine}_masses_amu'], (natom,))
    if np.any(masses <= 0):
        raise ValueError('Nonpositive atomic mass')
    cartesian = np.asarray(arrays[f'step_{fine}_cartesian_mass_normalized_modes'])
    weighted = np.asarray(arrays[f'step_{fine}_mass_weighted_modes'])
    frequencies = np.asarray(arrays[f'step_{fine}_signed_frequencies_cm1'])
    nmode = len(frequencies)
    _finite(cartesian, (ncart,nmode)); _finite(weighted, (ncart,nmode)); _finite(frequencies,(nmode,))
    root_mass = np.sqrt(np.repeat(masses,3))
    if not np.allclose(cartesian*root_mass[:,None], weighted, atol=1e-12, rtol=1e-12):
        raise ValueError('Cartesian and mass-weighted mode definitions differ')
    if not np.allclose(weighted.T@weighted, np.eye(nmode), atol=1e-9,rtol=1e-9):
        raise ValueError('Imported modes are not mass normalized')
    minimum = gate.get('minimum_verified') is True and all_points_qualified
    ir_ready = gate.get('paper_ready_molecular_harmonic_data') is True and minimum
    # The ID excludes commit, paths, timestamps and ZIP container metadata.
    qualification_identity = {'minimum_verified':bool(minimum),'ir_qualified':bool(ir_ready),
                              'gates':gate.get('gates',{}),'thresholds':gate.get('thresholds',{})}
    identity = {'symbols':symbols, 'method':case['method'], 'steps_bohr':steps,
        'qualification':qualification_identity,
        'array_hashes':{key:_sha(np.ascontiguousarray(value,dtype='<f8').tobytes())
            for key,value in sorted(arrays.items())}}
    source_id = _sha(_canonical(identity))
    evidence = dict(blobs)
    evidence.update({result_path:result_bytes,seal_path:seal_bytes,gate_path:gate_bytes,
                     center_path:center_bytes})
    return {'schema':SCHEMA,'id':source_id,'molecule':case['molecule'],
        'profile':case['profile'],'symbols':symbols,'method':case['method'],
        'steps_bohr':steps,'fine_matrix_index':fine,
        'source_qualification_identity':qualification_identity,
        'source_minimum_qualified':bool(minimum),'source_ir_qualified':bool(ir_ready),
        'source_center_energy_hartree':float(center['energy_hartree']),
        'source_center_dipole_au':_finite(center['dipole_au'],(3,)).tolist(),
        'all_original_points_electronically_qualified':bool(all_points_qualified),
        'original_displacement_point_count':original_count,
        'source':{'case_path':case_path,'source_result_sha256':_sha(result_bytes),
                  'upstream_gate_failed_gates':gate.get('failed_gates',[]),
                  'qualification_scope':'Authenticated upstream local molecular minimum; IR qualification is separate.'},
        '_evidence':evidence,'_arrays':arrays}


def authenticate_frequency_case(reader, case_path):
    """Requalify older frequency cases from raw objects, not their old flags.

    Current publication arithmetic is replayed against temporary copies of all
    original point files. Generated consumer receipts are labeled separately;
    the original upstream result and its COMPLETE seal remain byte-exact.
    """
    import pyscf_publication_quality_20261111 as publication_quality
    case_path = _relative(case_path)
    prefix = case_path+'/'
    result_path,seal_path=prefix+'result.json',prefix+'COMPLETE.json'
    original_result,original_seal=reader.read(result_path),reader.read(seal_path)
    case=json.loads(original_result)
    if json.loads(original_seal).get('result_sha256') != _sha(original_result):
        raise ValueError('Frequency case COMPLETE seal mismatch')
    if case.get('molecule') not in MOLECULES:
        raise ValueError('Unapproved frequency-case molecule')
    mode_path=prefix+'modes/result.json'
    mode=json.loads(reader.read(mode_path))
    optimization_path=_relative(case['optimization_path'])
    if not optimization_path.startswith(prefix):
        raise ValueError('Frequency optimization escaped source case')
    optimization_bytes=reader.read(optimization_path)
    optimization=json.loads(optimization_bytes)
    embedded_center=optimization['final_point']
    center_path=_relative(str(PurePosixPath(embedded_center['gradient_path']).parent/'result.json'))
    if not center_path.startswith(prefix):
        raise ValueError('Frequency center escaped source case')
    center_bytes=reader.read(center_path)
    center=json.loads(center_bytes)
    if _canonical(center)!=_canonical(embedded_center):
        raise ValueError('Frequency optimization center differs from original center point')
    paths=[path for path in reader.objects if path.startswith(prefix+'modes/')
        and path.rsplit('/',1)[1] in ('result.json','analysis_arrays.npz','all_modes.csv')]
    paths += [path for path in reader.objects if path.startswith(prefix+'finite-differences/')
        and len(path[len(prefix+'finite-differences/'):].split('/'))==3
        and path.rsplit('/',1)[1] in ('result.json','coordinates_bohr.npy','gradient.npy','dipole_au.npy')]
    originals=reader.read_many(paths)
    # Imports the maintained current audit; no upstream flags are accepted.
    with tempfile.TemporaryDirectory(prefix='pyscf-ir-frequency-audit-') as temporary:
        staging=Path(temporary)
        for path,data in originals.items():
            local=staging/path[len(prefix):]
            local.parent.mkdir(parents=True,exist_ok=True)
            local.write_bytes(data)
        stability=case.get('electronic_stability',{})
        stable=(stability.get('status')=='DONE'
                and stability.get('tested_electronic_stability_passed') is True)
        gate=publication_quality.audit_publication(staging/'modes',staging/'finite-differences',
            center,electronic_stability_passed=stable)
        # Map fresh audit receipts back to the exact original Git paths.
        hashes={}
        for path,digest in gate['authenticated_artifact_sha256'].items():
            relative=Path(path).relative_to(staging).as_posix()
            hashes[prefix+relative]=digest
        gate['authenticated_artifact_sha256']=hashes
    method=center['settings']
    profile=('df' if method.get('density_fitting') else 'direct')+'-grid'+str(method['grid_level'])
    adapter={'status':'CONSUMER_REQUALIFIED','molecule':case['molecule'],'profile':profile,
        'method':method,'analysis':mode,'publication_gate':gate,
        'center_point_path':center_path,'coordinates_bohr':center['coordinates_bohr']}
    adapter_bytes=_canonical(adapter)
    overlay=dict(originals)
    overlay.update({result_path:adapter_bytes,
        seal_path:_canonical({'result_sha256':_sha(adapter_bytes)}),
        prefix+'modes/publication_quality.json':_canonical(gate),center_path:center_bytes})
    class View:
        objects={**reader.objects,**{key:'consumer-generated' for key in overlay}}
        def read(self,path):
            return overlay[path] if path in overlay else reader.read(path)
        def read_many(self,paths):
            return {path:self.read(path) for path in paths}
    record=authenticate_case(View(),case_path)
    evidence=record['_evidence']
    evidence[result_path],evidence[seal_path]=original_result,original_seal
    evidence[optimization_path]=optimization_bytes
    evidence.pop(prefix+'modes/publication_quality.json',None)
    evidence[prefix+'consumer-generated/publication_quality.json']=_canonical(gate)
    evidence[prefix+'consumer-generated/case-adapter.json']=adapter_bytes
    record['source'].update(source_result_sha256=_sha(original_result),
        consumer_requalification=True,upstream_historical_qualification_flags_used=False,
        consumer_quality_implementation_sha256=_sha(Path(publication_quality.__file__).read_bytes()),
        generated_evidence_paths=[prefix+'consumer-generated/publication_quality.json',
                                  prefix+'consumer-generated/case-adapter.json'])
    return record


def import_case(reader, case_path, output, branch, commit, *, frequency=False):
    record = authenticate_frequency_case(reader,case_path) if frequency else authenticate_case(reader, case_path)
    evidence, arrays = record.pop('_evidence'), record.pop('_arrays')
    destination = Path(output)/'sources'/record['id']
    record['source'].update(branch=branch,commit=commit)
    if (destination/'source.json').exists():
        descriptor = json.loads((destination/'source.json').read_text())
        for key,value in record.items():
            if key != 'source' and _canonical(descriptor.get(key)) != _canonical(value):
                raise ValueError('Existing source manifest differs from independently reauthenticated input: '+key)
        descriptor['snapshot_path'] = str(destination.resolve())
        descriptor['source_record_sha256'] = _sha((destination/'source.json').read_bytes())
        load_source(descriptor)
        return descriptor
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ir-import-',dir=destination.parent) as temporary:
        staging = Path(temporary)
        hashes = {}
        for relative,data in evidence.items():
            path = staging/'evidence'/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(data)
            hashes[str(path.relative_to(staging))] = _sha(data)
        np.savez_compressed(staging/'analysis_arrays.npz',**arrays)
        hashes['analysis_arrays.npz'] = _sha((staging/'analysis_arrays.npz').read_bytes())
        record['snapshot_hashes'] = hashes
        _write_json(staging/'source.json',record)
        # Move a stable subdirectory; TemporaryDirectory retains its wrapper.
        frozen = staging/'frozen'
        frozen.mkdir()
        for child in list(staging.iterdir()):
            if child != frozen:
                child.rename(frozen/child.name)
        frozen.rename(destination)
    descriptor = dict(record,snapshot_path=str(destination.resolve()))
    descriptor['source_record_sha256'] = _sha((destination/'source.json').read_bytes())
    load_source(descriptor)
    return descriptor


def load_source(descriptor):
    """Load the consumer arrays, authenticating the frozen snapshot each time."""
    folder = Path(descriptor['snapshot_path'])
    record_bytes = (folder/'source.json').read_bytes()
    if descriptor.get('source_record_sha256') != _sha(record_bytes):
        raise ValueError('IR source snapshot manifest changed')
    record = json.loads(record_bytes)
    if record.get('id') != descriptor['id']:
        raise ValueError('IR source snapshot identity mismatch')
    qualification=record.get('source_qualification_identity',{})
    if record.get('source_minimum_qualified') is not qualification.get('minimum_verified') or record.get('source_ir_qualified') is not qualification.get('ir_qualified'):
        raise ValueError('IR source qualification manifest fields disagree')
    for relative,digest in record['snapshot_hashes'].items():
        path = folder/relative
        if path.resolve().is_relative_to(folder.resolve()) is False or _sha(path.read_bytes()) != digest:
            raise ValueError('IR source snapshot artifact changed: '+relative)
    with np.load(folder/'analysis_arrays.npz',allow_pickle=False) as archive:
        arrays = {key:np.array(archive[key],copy=True) for key in archive.files}
    identity = {'symbols':record['symbols'],'method':record['method'],'steps_bohr':record['steps_bohr'],
        'qualification':record['source_qualification_identity'],
        'array_hashes':{key:_sha(np.ascontiguousarray(value,dtype='<f8').tobytes())
            for key,value in sorted(arrays.items())}}
    if _sha(_canonical(identity)) != record['id']:
        raise ValueError('IR source scientific content identity changed')
    fine = record['fine_matrix_index']
    return {**record,'snapshot_path':str(folder.resolve()),
        'coordinates_bohr':arrays['coordinates_bohr'],
        'masses_amu':arrays[f'step_{fine}_masses_amu'],
        'cartesian_modes':arrays[f'step_{fine}_cartesian_mass_normalized_modes'],
        'mass_weighted_modes':arrays[f'step_{fine}_mass_weighted_modes'],
        'frequencies_cm1':arrays[f'step_{fine}_signed_frequencies_cm1'],
        'cartesian_dipole_derivatives':np.stack([arrays[f'step_{hi}_dipole_derivative_au']
            for hi in range(len(record['steps_bohr']))])}


def sync_sources(repo, output, *, refresh=True, interval_seconds=600):
    """Refresh read-only remote refs and return imported JSON descriptors.

    Network failures and unauthenticated/incomplete cases are recorded for the
    next scan; they do not terminate the caller's queue. Older frequency cases
    receive a newly computed independent gate. No old completion or qualification
    flag is treated as an acceptance gate.
    """
    repo, output = Path(repo), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    state_path = output/'sources_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    now = time.time()
    branches = state.setdefault('branches',{})
    if refresh and now-state.get('last_refresh_attempt_epoch',0) >= interval_seconds:
        state['last_refresh_attempt_epoch'] = now
        for branch in BRANCHES:
            try:
                _git(repo,'fetch','--no-tags','origin',f'refs/heads/{branch}:refs/remotes/origin/{branch}')
                branches[branch] = {'fetch_status':'FETCHED','at_epoch':time.time()}
            except Exception as error:
                branches[branch] = {'fetch_status':'RETRY_PENDING','error':str(error),'at_epoch':time.time()}
        _write_json(state_path,state)
    descriptors = {}
    snapshot_manifest_hashes = state.setdefault('snapshot_manifest_sha256',{})
    for path in (output/'sources').glob('*/source.json'):
        try:
            descriptor = json.loads(path.read_text())
            descriptor['snapshot_path'] = str(path.parent.resolve())
            digest = snapshot_manifest_hashes.get(descriptor['id'])
            if digest is None:
                raise ValueError('Previously unregistered snapshot manifest requires source reauthentication')
            descriptor['source_record_sha256'] = digest
            load_source(descriptor)
            descriptors[descriptor['id']] = descriptor
        except Exception as error:
            state.setdefault('snapshot_errors',{})[str(path)] = str(error)
    for branch,root,frequency in ((BRANCHES[0],PUBLICATION_ROOT,False),(BRANCHES[1],FREQUENCY_ROOT,True)):
        try:
            commit = _git(repo,'rev-parse',f'refs/remotes/origin/{branch}').decode().strip()
            reader = GitBlobs(repo,commit,root)
            candidates = sorted({path.rsplit('/',1)[0] for path in reader.objects
                if path.endswith('/COMPLETE.json') and path.rsplit('/',1)[0]+'/modes/result.json' in reader.objects})
            entries = state.setdefault('cases',{})
            for case_path in candidates:
                try:
                    seal = json.loads(reader.read(case_path+'/COMPLETE.json'))
                    key = seal.get('result_sha256')
                    old = entries.get(case_path,{})
                    if old.get('result_sha256') == key and old.get('source_id') in descriptors:
                        continue
                    descriptor = import_case(reader,case_path,output,branch,commit,frequency=frequency)
                    descriptors[descriptor['id']] = descriptor
                    snapshot_manifest_hashes[descriptor['id']] = descriptor['source_record_sha256']
                    entries[case_path] = {'status':'IMPORTED','source_id':descriptor['id'],
                        'result_sha256':key,'commit':commit}
                except Exception as error:
                    entries[case_path] = {'status':'AUTHENTICATION_RETRY_PENDING','error':str(error),'commit':commit}
            branches.setdefault(branch,{})['consumer_status']='SCANNED'
            branches[branch]['commit']=commit
        except Exception as error:
            branches.setdefault(branch,{})['scan_error']=str(error)
    state['last_scan_epoch'] = time.time()
    state['imported_source_count'] = len(descriptors)
    _write_json(state_path,state)
    return sorted(descriptors.values(),key=lambda x:({'water':0,'acid':1,'calycosin':2,'spinosin':3}[x['molecule']],x['profile'],x['id']))


def authenticate_preview(reader,optimization_path,molecule):
    """Authenticate a stationary center while its full Hessian is still pending.

    This grants no minimum, spectral or paper qualification. It only permits
    saving dipole derivatives in parallel at the exact sealed center geometry.
    """
    if molecule not in MOLECULES-{'water'}:
        raise ValueError('Preview molecule is not an approved target')
    optimization_path=_relative(optimization_path)
    optimization_bytes=reader.read(optimization_path)
    optimization=json.loads(optimization_bytes)
    if optimization.get('status')!='DONE' or optimization.get('optimizer_converged') is not True or optimization.get('minimum_stationarity_passed') is not True:
        raise ValueError('Optimization is not a completed stationary precursor')
    embedded=optimization['final_point']
    point_path=_relative(str(PurePosixPath(embedded['gradient_path']).parent/'result.json'))
    optimization_folder=optimization_path.rsplit('/',1)[0]+'/'
    if not point_path.startswith(optimization_folder):
        raise ValueError('Preview center escaped its optimization checkpoint')
    point_bytes=reader.read(point_path)
    point=json.loads(point_bytes)
    if _canonical(point)!=_canonical(embedded):
        raise ValueError('Preview optimization and original center disagree')
    if not _electronic(point):
        raise ValueError('Preview center electronic audit is not qualified')
    symbols=point['symbols'];natom=len(symbols)
    coords=_finite(point['coordinates_bohr'],(natom,3))
    gradient=_finite(point['gradient'],(natom,3))
    dipole=_finite(point['dipole_au'],(3,))
    if not np.array_equal(coords,np.asarray(optimization['coordinates_bohr'])):
        raise ValueError('Preview optimized center coordinates disagree')
    if np.max(np.abs(gradient))>1e-5 or np.sqrt(np.mean(gradient**2))>3e-6:
        raise ValueError('Preview nuclear stationarity evidence is insufficient')
    evidence={optimization_path:optimization_bytes,point_path:point_bytes}
    for key,filename,shape in [('coordinates_bohr','coordinates_bohr.npy',(natom,3)),
        ('gradient','gradient.npy',(natom,3)),('dipole_au','dipole_au.npy',(3,))]:
        path=point_path.rsplit('/',1)[0]+'/'+filename
        raw=reader.read(path)
        ref=point.get('array_refs',{}).get(key,{})
        if ref.get('sha256')!=_sha(raw):
            raise ValueError('Preview original center array seal mismatch')
        if not np.array_equal(_finite(np.load(io.BytesIO(raw),allow_pickle=False),shape),np.asarray(point[key])):
            raise ValueError('Preview center JSON/NPY disagreement')
        evidence[path]=raw
    method=point['settings']
    profile=('df' if method.get('density_fitting') else 'direct')+'-grid'+str(method['grid_level'])
    identity={'symbols':symbols,'coordinates_bohr':coords.tolist(),'method':method}
    return {'schema':SCHEMA,'id':_sha(_canonical(identity)),'molecule':molecule,'profile':profile,
        'status':'PRECURSOR_ONLY','source_minimum_qualified':False,'source_ir_qualified':False,
        'symbols':symbols,'coordinates_bohr':coords.tolist(),'method':method,
        'source_center_energy_hartree':float(point['energy_hartree']),
        'source_center_dipole_au':dipole.tolist(),
        'source':{'optimization_path':optimization_path,'center_point_path':point_path,
            'optimization_sha256':_sha(optimization_bytes),'center_point_sha256':_sha(point_bytes),
            'qualification_scope':'Stationary electronic center only; full minimum, modes and IR qualification pending.'},
        '_evidence':evidence}


def preview_centers(repo,output):
    """Return authenticated stationary precursors from the two explicit jobs roots.

    Call sync_sources first to refresh refs. This scan does no network requests,
    geometry optimization, Hessian generation or qualification promotion.
    """
    repo,output=Path(repo),Path(output)
    state_path=output/'preview_sources_state.json'
    state={'branches':{},'errors':{}}
    records={}
    for branch,root in ((BRANCHES[0],PUBLICATION_ROOT),(BRANCHES[1],FREQUENCY_ROOT)):
        try:
            commit=_git(repo,'rev-parse',f'refs/remotes/origin/{branch}').decode().strip()
            reader=GitBlobs(repo,commit,root+'/jobs')
            state['branches'][branch]={'commit':commit}
            for path in sorted(reader.objects):
                if not path.endswith('/result.json'):
                    continue
                folder=path.rsplit('/',2)[-2]
                if folder not in ('optimize','grid5-optimize') and not folder.startswith('reoptimized-stable-state-'):
                    continue
                molecule=path[len(root+'/jobs/'):].split('/',1)[0]
                if molecule not in MOLECULES-{'water'}:
                    continue
                try:
                    record=authenticate_preview(reader,path,molecule)
                    evidence=record.pop('_evidence')
                    record['source'].update(branch=branch,commit=commit)
                    folder=output/'precursors'/record['id']
                    manifest_path=folder/'source.json'
                    if not manifest_path.exists():
                        for relative,data in evidence.items():
                            target=folder/'evidence'/relative
                            target.parent.mkdir(parents=True,exist_ok=True)
                            target.write_bytes(data)
                        record['snapshot_hashes']={'evidence/'+relative:_sha(data) for relative,data in evidence.items()}
                        _write_json(manifest_path,record)
                    else:
                        record=json.loads(manifest_path.read_text())
                        identity={'symbols':record['symbols'],'coordinates_bohr':record['coordinates_bohr'],'method':record['method']}
                        if _sha(_canonical(identity))!=record['id'] or record.get('source_minimum_qualified') is not False or record.get('source_ir_qualified') is not False:
                            raise ValueError('Frozen precursor identity or qualification changed')
                        for relative,digest in record['snapshot_hashes'].items():
                            if _sha((folder/relative).read_bytes())!=digest:
                                raise ValueError('Frozen precursor artifact changed')
                    record['snapshot_path']=str(folder.resolve())
                    record['source_record_sha256']=_sha(manifest_path.read_bytes())
                    records[record['id']]=record
                except Exception as error:
                    state['errors'][path]=str(error)
        except Exception as error:
            state['branches'][branch]={'error':str(error)}
    state['stationary_precursor_count']=len(records)
    _write_json(state_path,state)
    return sorted(records.values(),key=lambda x:({'acid':0,'calycosin':1,'spinosin':2}[x['molecule']],x['profile'],x['id']))
