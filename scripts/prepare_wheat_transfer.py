"""Prepare the pinned IDRC2016 wheat new-sample/new-instrument transfer data.

The author describes 248 original samples and another 150 new samples. IDs here
encode row positions in those documented populations; they are not published
specimen identifiers. Target A/B are the same 150 samples on different devices.
R serialization is decoded with rdata; no R interpreter or supplied R code runs.

Optional isolated parser install used for this run:
  python -m pip install --target outputs/wheat-parser rdata==1.1.0 xarray==2026.9.0
Pass that directory with --parser-path, or install rdata in your own environment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np


REPO = Path(__file__).resolve().parents[1]
COMMIT = 'dad77175e6148ce587d49f2f1979a7d6de3271c7'
SOURCE_URL = f'https://raw.githubusercontent.com/HMzhu/CTVPdtw/{COMMIT}/data/wheat.rda'
SOURCE_SHA256 = 'fdd0580cbd90ad55026e32c434fe9cdbe00853ba2f9907f9ccfb9536324c28a1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_hashes(X):
    X = np.asarray(X, dtype='<f8').copy()
    X[X == 0] = 0.0
    return [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in X]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-rda', type=Path, default=REPO/'outputs/external-transfer-sources/wheat.rda')
    parser.add_argument('--download', action='store_true', help='Fetch the pinned source when absent')
    parser.add_argument('--parser-path', type=Path, help='Optional isolated Python rdata/xarray installation')
    parser.add_argument('--output', type=Path, default=REPO/'outputs/research-stage2/wheat-new')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose an empty/new output directory')
    if not args.source_rda.is_file():
        if not args.download:parser.error('Source absent; supply --source-rda or --download')
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:content = response.read()
        if hashlib.sha256(content).hexdigest()!=SOURCE_SHA256:raise ValueError('Downloaded source hash mismatch')
        args.source_rda.parent.mkdir(parents=True, exist_ok=True)
        args.source_rda.write_bytes(content)
    if digest(args.source_rda)!=SOURCE_SHA256:raise ValueError('Cached wheat.rda hash mismatch')
    if args.parser_path is not None:sys.path.insert(0,str(args.parser_path.resolve()))
    try:
        import rdata
    except ImportError as exc:
        raise RuntimeError('Install rdata and its dependencies, or pass --parser-path; see module instructions') from exc
    objects = rdata.read_rda(args.source_rda)
    if set(objects)!={'wheat'} or not isinstance(objects['wheat'],dict):
        raise ValueError('Unexpected R data structure')
    original = objects['wheat']
    required = {'A1.CAL':(248,397),'Y.CAL':(248,1),'vliadA':(150,397),
                'vliadB':(150,397),'y.Valid':(150,1)}
    data = {}
    for key,shape in required.items():
        value = original[key]
        if not isinstance(value,np.ndarray) or value.dtype.kind not in 'fiu':
            raise ValueError('Expected numeric matrix: '+key)
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid matrix: '+key)
        data[key]=np.asarray(value,dtype=float)
    source_ids=np.asarray([f'source:{i:03d}' for i in range(248)])
    target_ids=np.asarray([f'new:{i:03d}' for i in range(150)])
    source_hashes=set(rows_hashes(data['A1.CAL']))
    audits={}
    for key,name in [('vliadA','transfer_input.npz'),('vliadB','transfer_B_input.npz')]:
        overlap=len(source_hashes & set(rows_hashes(data[key])))
        if overlap:raise ValueError('Exact source-target spectrum overlap')
        audits[name]={'X_source_shape':[248,397],'X_target_shape':[150,397],
                      'exact_source_target_spectrum_overlap':overlap,
                      'source_duplicate_spectra':248-len(source_hashes),
                      'target_duplicate_spectra':150-len(set(rows_hashes(data[key])))}
    args.output.mkdir(parents=True,exist_ok=True)
    for key,name in [('vliadA','transfer_input.npz'),('vliadB','transfer_B_input.npz')]:
        np.savez_compressed(args.output/name,X_source=data['A1.CAL'],
                            y_source=data['Y.CAL'].reshape(-1),source_ids=source_ids,
                            X_target=data[key],y_target=data['y.Valid'].reshape(-1),target_ids=target_ids)
    files=[{'path':name,'bytes':(args.output/name).stat().st_size,'sha256':digest(args.output/name)}
           for name in ['transfer_input.npz','transfer_B_input.npz']]
    registry={'id':'idrc2016_wheat_new_instrument_new_samples',
              'repository':'https://github.com/HMzhu/CTVPdtw','commit':COMMIT,
              'source_url':SOURCE_URL,'source_document':f'https://github.com/HMzhu/CTVPdtw/blob/{COMMIT}/man/wheat.Rd',
              'associated_paper_doi':'10.1039/C9AY01139K','source_archive_sha256':SOURCE_SHA256,
              'source_local_path':str(args.source_rda.resolve()),'task':'regression',
              'target_name':'reference protein (source documentation)',
              'target_unit':'source-units-unspecified',
              'source_domain':'A1.CAL, original instrument A1, 248 original samples',
              'target_domains':{'transfer_input.npz':'vliadA, new instrument A,150 new samples',
                                'transfer_B_input.npz':'vliadB,new instrument B,SAME150 new samples'},
              'source_identity_basis':'Author documentation states248 originals and another150 new samples; IDs encode row positions inside those populations, not published specimen IDs',
              'target_A_B_same150_samples':True,'source_sample_count':248,'target_sample_count':150,
              'feature_count_common':397,'preprocessing':'none;copy given numerical matrices without fit or query statistics',
              'parser':{'name':'rdata','version':importlib.metadata.version('rdata'),
                        'xarray_version':importlib.metadata.version('xarray'),'executed_R_code':False},
              'license':'Repository software LGPL2+; independent data-specific license not established; source arrays stay local ignored outputs',
              'files':files,
              'caveats':['Actual physical specimen IDs/instrument serials/wavelength vector not supplied in parsed matrices',
                         'A/B alignment uses common397-column source representation; exact physical axes not reconstructed',
                         'Tests joint new-device/new-population shift, not isolated instrument effect',
                         'A and B target files are the same150 samples, not independent external cohorts',
                         'Wheat protein transfer does not validate medicinal-herb origin classification']}
    audit={'status':'pass','prepared_at_utc':datetime.now(timezone.utc).isoformat(),
           'all_finite':True,'source_ids_unique':True,'target_ids_unique':True,
           'source_target_id_overlap':0,'documented_disjoint_sample_populations':True,
           'physical_specimen_identifiers_verified':False,'target_unit_verified':False,
           'target_A_B_independent_cohorts':False,'matrices':audits,'files':files,
           'preparation_code_sha256':digest(__file__)}
    write_json(args.output/'source_registry.json',registry)
    write_json(args.output/'audit.json',audit)
    print(json.dumps({'status':'pass','output':str(args.output),'rdata_version':registry['parser']['version'],
                      'source_rows':248,'target_rows':150,'features':397},indent=2))


if __name__=='__main__':main()
