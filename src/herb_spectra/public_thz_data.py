"""Source-backed numeric interfaces for public THz plant observations.

The first interface is the existing 42-tablet Ginseng/American-ginseng V2
archive. Export file groups are technical provenance, not independently known
biological plants or batches. No modelling, fitting, cleaning or label inference
occurs here. Historical source files are never modified.
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GINSENG_ORIGINAL = ROOT / 'data/external/ginseng_hvbp5mdtxx_v2/original'
GINSENG_INSPECTION = ROOT / 'data/external/ginseng_hvbp5mdtxx_v2/annotated/inspection.json'
GINSENG_INSPECTION_SHA256 = 'cfea731b774a69888f8380951881448cb312564e928698485e1b935632ec8a62'
GINSENG_RAR_SHA256 = '3b1bbcff42c0825361c372cab329abee2c21c29d3b4b30ac7c1366865e7c3fcc'
DATASET_ID = 'public-ginseng-hvbp5mdtxx-v2-existing-42-tablet-observations'
STATUS = 'EXPLORATORY_OBSERVATION_LEVEL_UNKNOWN_UPSTREAM'
SOURCE_URL = 'https://data.mendeley.com/datasets/hvbp5mdtxx/2'
NEAR_CORRELATION_THRESHOLD = .99999
FIELDS = ('X_alpha', 'X_n', 'axis', 'y', 'record_ids', 'export_groups')

class PublicTHzDataError(ValueError):
    pass

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def expected_ginseng_hashes():
    if sha256(GINSENG_INSPECTION) != GINSENG_INSPECTION_SHA256:
        raise PublicTHzDataError('Frozen historical inspection manifest changed')
    inspection = json.loads(GINSENG_INSPECTION.read_text())
    return {item['file']: item['sha256'] for item in inspection['member_manifest']}

def validate_ginseng_arrays(data):
    """Validate exact shape/identity contract, without claiming sample independence."""
    if set(data) != set(FIELDS):
        raise PublicTHzDataError('Expected exactly the declared paired spectrum and metadata fields')
    result = {}
    for name in ('X_alpha', 'X_n'):
        raw = np.asarray(data[name])
        if raw.dtype.kind not in 'iuf' or raw.shape != (42, 65) or not np.isfinite(raw).all():
            raise PublicTHzDataError(f'{name} requires42 finite numeric65-point observations')
        result[name] = np.asarray(raw, dtype=np.float64)
    axis = np.asarray(data['axis'])
    if axis.dtype.kind not in 'iuf' or axis.shape != (65,) or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
        raise PublicTHzDataError('Invalid frequency axis')
    if not np.allclose(axis, np.linspace(.2, 1.6, 65), atol=1e-12, rtol=0):
        raise PublicTHzDataError('Source V2 axis is0.2–1.6THz; no implicit interpolation')
    result['axis'] = axis.astype(np.float64)
    for name in ('y', 'record_ids', 'export_groups'):
        raw = np.asarray(data[name])
        if raw.dtype.kind not in 'US' or raw.shape != (42,) or any(not str(v).strip() for v in raw):
            raise PublicTHzDataError(f'{name} requires42 explicit strings')
        result[name] = raw.astype(str)
    if len(set(result['record_ids'])) != 42:
        raise PublicTHzDataError('Observation IDs must be unique')
    if set(result['y']) != {'0', '1'} or list(np.unique(result['y'], return_counts=True)[1]) != [22, 20]:
        raise PublicTHzDataError('Published species labels must remain22American/20Asian observations')
    expected_ids = np.array([f'G{g}_S{s:02}' for g in range(1, 7) for s in range(1, 8)])
    expected_groups = np.repeat([f'G{g}' for g in range(1, 7)], 7)
    if not np.array_equal(result['record_ids'], expected_ids) or not np.array_equal(result['export_groups'], expected_groups):
        raise PublicTHzDataError('Source export group and within-group observation pairing changed')
    for group in np.unique(expected_groups):
        if set(result['y'][result['export_groups'] == group]) != {'0', '1'}:
            raise PublicTHzDataError('Each held export group must retain both published classes')
    return result

def load_ginseng(input_dir=None, *, expected_files_sha256=None):
    """Return X_alpha/X_n[42,65],axis,y,record_ids,export_groups.

    Metadata is separate from numeric features. Labels come solely from paired
    label_j.csv and are strings0/1, with0=Panax quinquefolius and1=P.ginseng.
    Optional expected_files_sha256 permits explicit synthetic parser fixtures;
    production callers use the pinned historical member manifest by default.
    """
    folder = GINSENG_ORIGINAL if input_dir is None else Path(input_dir)
    expected = expected_ginseng_hashes() if expected_files_sha256 is None else dict(expected_files_sha256)
    required = {f'{prefix}{g}.csv' for g in range(1, 7) for prefix in ['data_a', 'data_n', 'label_']}
    if set(expected) != required:
        raise PublicTHzDataError('Exactly18 source CSV byte hashes are required')
    for name, value in expected.items():
        if sha256(folder/name) != value:
            raise PublicTHzDataError(f'Frozen CSV bytes changed: {name}')
    xs = {'alpha': [], 'n': []}; labels=[]; axis=None
    for group in range(1, 7):
        for modality, prefix in [('alpha', 'a'), ('n', 'n')]:
            path=folder/f'data_{prefix}{group}.csv'
            with path.open(encoding='utf-8-sig',newline='') as f:
                header=next(csv.reader(f),[])
            if len(header) != 8 or header[0].strip() != 'Frequency (THz)' or any(v.strip() for v in header[1:]):
                raise PublicTHzDataError('Source header must be a frequency and7 unnamed original spectrum columns')
            raw=np.loadtxt(path,delimiter=',',skiprows=1,encoding='utf-8-sig')
            if raw.shape != (65,8) or not np.isfinite(raw).all():
                raise PublicTHzDataError('Original source table shape/finiteness mismatch')
            if axis is None:axis=raw[:,0].copy()
            if not np.array_equal(axis,raw[:,0]):
                raise PublicTHzDataError('Paired frequency axes differ')
            xs[modality].append(raw[:,1:].T.copy())
        y=np.loadtxt(folder/f'label_{group}.csv',delimiter=',',ndmin=1,encoding='utf-8-sig')
        if y.shape != (7,) or not np.isfinite(y).all() or not set(y).issubset({0.,1.}):
            raise PublicTHzDataError('Published label file must contain7 binary values')
        labels.extend(y.astype(int).astype(str))
    data={'X_alpha':np.vstack(xs['alpha']),'X_n':np.vstack(xs['n']),'axis':axis,
          'y':np.asarray(labels,dtype=str),'record_ids':np.asarray([f'G{g}_S{s:02}' for g in range(1,7) for s in range(1,8)]),
          'export_groups':np.repeat([f'G{g}' for g in range(1,7)],7)}
    for name,value in expected.items():
        if sha256(folder/name)!=value:raise PublicTHzDataError('Source CSV changed during read')
    return validate_ginseng_arrays(data)

def duplicate_diagnostics(data):
    """QC only; never delete spectra or relabel a biological identity."""
    data=validate_ginseng_arrays(data);ids=data['record_ids'];groups=data['export_groups'];results={}
    for modality,x in [('alpha',data['X_alpha']),('n',data['X_n']),('paired',np.column_stack([data['X_alpha'],data['X_n']]))]:
        exact=[];near=[];largest=-1.;largest_pair=None
        for i in range(len(x)):
            for j in range(i+1,len(x)):
                pair={'record_a':str(ids[i]),'record_b':str(ids[j]),'different_export_groups':bool(groups[i]!=groups[j])}
                if np.array_equal(x[i],x[j]):exact.append(pair)
                a=x[i]-x[i].mean();b=x[j]-x[j].mean();denom=np.linalg.norm(a)*np.linalg.norm(b)
                cosine=float(np.dot(a,b)/denom) if denom else 0.
                if cosine>largest:largest=cosine;largest_pair=pair
                if cosine>=NEAR_CORRELATION_THRESHOLD:near.append({**pair,'centered_cosine':cosine})
        results[modality]={'pairs_checked':861,'exact_pairs':exact,'near_pairs':near,
            'largest_centered_cosine':largest,'largest_pair':largest_pair,'nonfinite_cells':int((~np.isfinite(x)).sum()),
            'negative_cells':int((x<0).sum()),'minimum':float(x.min()),'maximum':float(x.max())}
    return {'threshold_predefined':NEAR_CORRELATION_THRESHOLD,'no_observations_removed':True,
            'near_correlation_is_biological_duplicate_proof':False,'absence_of_duplicates_proves_independence':False,'modalities':results}

def prepare_ginseng(output_dir):
    """Write a new observation-level interface with provenance; no model fitting."""
    out=Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):raise FileExistsError('Fresh ginseng preparation directory required')
    data=load_ginseng();out.mkdir(parents=True,exist_ok=True)
    original_hashes=expected_ginseng_hashes()
    rar=GINSENG_ORIGINAL/'Dataset.rar'
    if sha256(rar)!=GINSENG_RAR_SHA256:raise PublicTHzDataError('Original uploaded archive changed')
    np.savez_compressed(out/'observations.npz',**data)
    with (out/'observations.npz').open('rb') as f:prepared_sha=hashlib.sha256(f.read()).hexdigest()
    rows=[]
    for i,rid in enumerate(data['record_ids']):
        group=int(data['export_groups'][i][1:]);column=i%7+2;label=str(data['y'][i])
        for modality,prefix,unit in [('alpha','a','cm^-1'),('n','n','dimensionless')]:
            filename=f'data_{prefix}{group}.csv';label_file=f'label_{group}.csv'
            rows.append({'dataset_id':DATASET_ID,'record_id':str(rid),'observation_unit':'publisher-described tablet observation',
                'physical_sample_id':'','parent_sample_id':'','plant_id':'','preparation_id':'','batch_id':'','measurement_id':'',
                'export_group':str(data['export_groups'][i]),'modality':modality,'label':label,
                'species_label':'Panax ginseng' if label=='1' else 'Panax quinquefolius','source_file':str((GINSENG_ORIGINAL/filename).relative_to(ROOT)),
                'source_hash':original_hashes[filename],'source_column_1based':column,'label_file':label_file,'label_file_sha256':original_hashes[label_file],
                'label_item_1based':i%7+1,'axis_unit':'THz','physical_quantity_unit':unit,
                'independence_evidence':'unknown upstream plant/material/batch mapping; publisher describes42tablets',
                'scientific_status':STATUS,'paired_by':'same export suffix and column position, publisher description'})
    with (out/'observation_lineage.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    _json(out/'duplicate_qc.json',duplicate_diagnostics(data))
    source={'observed_at_utc':datetime.now(timezone.utc).isoformat(),'dataset_url':SOURCE_URL,'doi':'10.17632/hvbp5mdtxx.2',
        'title':'Complex Refractive Indices of Ginseng/American Ginseng Tablets','creator':'Jamil Mai','publisher':'Mendeley Data',
        'version':2,'published':'2025-05-08','institution':'National Tsing Hua University, Hsinchu, Taiwan',
        'license':'CC BY4.0','license_url':'https://creativecommons.org/licenses/by/4.0/',
        'official_description_rechecked_this_task':True,'official_description_claims':['42tablets split6groups','18CSV files; first spectrum column is frequencyTHz, next7columns are observations','label1Asian ginseng,label0American ginseng','nRIU,alpha cm^-1'],
        'upstream_independence_statement_found':False,'source_csv_or_rar_contains_biological_sample_table':False,
        'existing_acquisition':'User-uploaded Dataset.rar previously archived; no new remote numeric download by this interface',
        'new_numeric_downloads_this_interface':0,'official_archive_checksum_independently_verified':False,
        'related_2026_dataset_url':'https://data.mendeley.com/datasets/k4pjjjmgnr/1',
        'related_2026_note':'Four schemes x6pre-exported train/test folds are representations/splits, not proof of a new cohort; no additional observations counted here',
        'primary_source_pages':['https://data.mendeley.com/datasets/hvbp5mdtxx/2','https://data.mendeley.com/datasets/compare/hvbp5mdtxx','https://data.mendeley.com/datasets/k4pjjjmgnr/1'],
        'paper_record_status':'2025 manuscript named by publisher; full primary methodological paper not retrieved by this interface',
        'later_paper_official_lab_index':'https://www.ee.nthu.edu.tw/shanghua/publications/index.html',
        'later_paper_doi':'10.1109/ACCESS.2026.3692252','paper_reported_accuracy_used_as_reproduced_result':False}
    _json(out/'source_provenance.json',source)
    manifest={'dataset_id':DATASET_ID,'created_at_utc':datetime.now(timezone.utc).isoformat(),'scientific_status':STATUS,
        'records':42,'source_described_tablets':42,'verified_independent_plants':None,'verified_independent_batches':None,
        'record_ids':data['record_ids'].tolist(),'export_groups':data['export_groups'].tolist(),'class_counts':{'0':22,'1':20},
        'class_meaning':{'0':'Panax quinquefolius (American ginseng)','1':'Panax ginseng (Asian ginseng)'},
        'axis_points':65,'frequency_range_thz':[.2,1.6],'modalities':{'alpha':'cm^-1','n':'dimensionless'},
        'observation_lineage_rows':84,'modalities_expand_record_count':False,
        'export_group_is_biological_batch':False,'geographic_origins_available':False,'chemical_content_targets_available':False,
        'pairing_identity_externally_verified':False,'prepared_arrays_sha256':prepared_sha,'prepared_array_fields':list(FIELDS),
        'raw_members_sha256':original_hashes,'original_rar_sha256':GINSENG_RAR_SHA256,
        'historical_inspection_sha256':GINSENG_INSPECTION_SHA256,'loader_sha256':sha256(__file__),
        'recommended_evaluation':'6outer leave-one-export-group-out; each inner loop leaves one remaining export group out; exploratory observation-level results only',
        'true_cross_batch_generalization_status':'BLOCKED_NO_VERIFIED_BATCH_ANCESTRY','cross_plant_transfer_status':'BLOCKED_PENDING_PHYSICAL_AND_UPSTREAM_COMPATIBILITY',
        'new_model_fits':0,'new_independent_samples_added':0,'old_data_modified':False}
    _json(out/'manifest.json',manifest)
    readback=np.load(out/'observations.npz',allow_pickle=False)
    with readback:
        assert set(readback.files)==set(FIELDS)
        for key in FIELDS:np.testing.assert_array_equal(readback[key],data[key])
    return manifest
