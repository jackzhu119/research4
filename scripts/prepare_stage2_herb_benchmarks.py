"""Prepare fixed, deduplicated herb-origin folds from a pinned data wheel.

The wheel is read as a ZIP archive only: no packaged Python/bytecode is imported,
installed, deserialized, or executed. Exact duplicate spectra are collapsed into
one record before splitting, and conflicting labels cause failure. Spectrum
hashes identify numeric rows, not independently verified physical specimens.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
from sklearn.model_selection import StratifiedKFold


REPO = Path(__file__).resolve().parents[1]
SEED = 20261030
WHEEL_NAME = 'qsi_tk-1.4.1-py3-none-any.whl'
WHEEL_SHA256 = 'c710fdb6f1a75158eed89743b13cd37ba6a66d482d8e1c58b709c0c2195067e8'
WHEEL_BYTES = 91457663
PYPI_JSON = 'https://pypi.org/pypi/qsi-tk/1.4.1/json'
PYPI_PAGE = 'https://pypi.org/project/qsi-tk/1.4.1/'
WHEEL_URL = 'https://files.pythonhosted.org/packages/72/04/8b68279b34c7d90b44597b151b470208a9c05e4f71ebecef2def364cca34/' + WHEEL_NAME
DATASETS = [
    {'id': 'qsi_huangqi_raman', 'filename': '7044X_RAMAN.csv',
     'description_file': '7044X_RAMAN desc.txt', 'expected_rows': 160, 'expected_features': 2090,
     'source_name': 'Radix astragali / Huang-qi / 黄芪', 'spectral_type': 'Raman',
     'feature_axis_unit': 'cm^-1', 'axis_unit_status': 'Explicit Raman-shift unit in packaged description',
     'class_names': ['Inner Mongolia', 'Sichuan', 'Shanxi', 'Gansu'],
     'class_mapping_status': 'Description explicitly maps integer labels0..3 to these names',
     'cohort_id': 'qsi_huangqi_2022_multimodal',
     'paper_doi': '10.1016/j.saa.2022.121348',
     'caveats': ['Raman and UV files describe the same160-sample Huangqi cohort; do not count them as two independent cohorts.',
                 'No physical specimen, batch, or replicate identifiers in CSV; cross-modal row pairing is not independently verified.']},
    {'id': 'qsi_huangqi_uv', 'filename': '7143X_UV.csv',
     'description_file': '7143X_UV desc.txt', 'expected_rows': 160, 'expected_features': 601,
     'source_name': 'Radix astragali / Huang-qi / 黄芪', 'spectral_type': 'UV-visible',
     'feature_axis_unit': 'nm', 'axis_unit_status': 'Explicit wavelength unit in packaged description',
     'class_names': ['Inner Mongolia', 'Sichuan', 'Shanxi', 'Gansu'],
     'class_mapping_status': 'Description explicitly maps integer labels0..3 to these names',
     'cohort_id': 'qsi_huangqi_2022_multimodal',
     'paper_doi': '10.1016/j.saa.2022.121348',
     'caveats': ['Packaged description says602 dimensions; CSV has601 spectral columns (200..800nm inclusive) plus one label column. Values and axis are preserved; no invented channel is added.',
                 'Raman and UV files describe the same160-sample Huangqi cohort; do not count them as two independent cohorts.',
                 'No physical specimen, batch, or replicate identifiers in CSV; cross-modal row pairing is not independently verified.']},
    {'id': 'qsi_raspberry_raman', 'filename': 'raspberry.csv',
     'description_file': 'raspberry desc.txt', 'expected_rows': 250, 'expected_features': 1682,
     'source_name': 'raspberry / Rubus idaeus L. (as labelled by the package)', 'spectral_type': 'Raman',
     'feature_axis_unit': 'not explicitly stated in CSV; Raman-shift context',
     'axis_unit_status': 'Numeric source axis retained; cm^-1 would require independent confirmation',
     'class_names': ['Anhui', 'Anji', "Chun'an", "Pan'an", 'Wuyi'],
     'class_mapping_status': 'Description explicitly maps integer labels0..4 to these place names',
     'cohort_id': 'qsi_raspberry_unverified_species', 'paper_doi': None,
     'caveats': ['The packaged description names Rubus idaeus L. while also referring to Chinese medicinal Fu-pen-zi; do not assume identity with pharmacopoeial Rubus chingii material.',
                 'The labels mix a province and localities; use the supplied categorical origin labels without inventing geographical hierarchy.',
                 'No independently confirmed taxonomy, physical specimen, batch, or replicate identifiers.']},
    {'id': 'qsi_shihu3_raman', 'filename': 'shihu3.CSV',
     'description_file': 'shihu3 desc.txt', 'expected_rows': 200, 'expected_features': 1681,
     'source_name': 'Dendrobium officinale dried stems (as labelled by the package)', 'spectral_type': 'Raman',
     'feature_axis_unit': 'not explicitly stated in CSV; Raman-shift context',
     'axis_unit_status': 'Description specifies instrument resolution in cm^-1 but not an explicit CSV-axis unit; numeric values retained',
     'class_names': ['label_0', 'label_1', 'label_2'],
     'class_mapping_status': 'Three places listed in description but no explicit mapping of all integer labels to locations; retain numeric label names',
     'candidate_places_unmapped': ["Pan'an", 'Wenzhou', 'Yunan (source spelling)'],
     'cohort_id': 'qsi_shihu3_2023', 'paper_doi': None,
     'paper_title': 'Origin identification of dendrobium officinale based on NNRW-stacking ensembles',
     'caveats': ['Description reports100/50/50 samples from three places but does not explicitly establish label1/2 location mapping; no mapping is guessed.',
                 'Collected June2023 per description; this is not a sample-level date or multiple-year domain label.',
                 'The instrument averages1000 scans per acquisition; CSV does not provide a replicate-to-physical-sample mapping.']},
]


def digest(content):return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def checked_wheel(path, download):
    if not path.is_file():
        if not download:raise FileNotFoundError(f'{path}: use --download for the pinned source')
        with urllib.request.urlopen(PYPI_JSON, timeout=30) as response:metadata=json.load(response)
        files=[r for r in metadata['urls'] if r['filename']==WHEEL_NAME]
        if len(files)!=1 or files[0]['digests']['sha256']!=WHEEL_SHA256 or files[0]['size']!=WHEEL_BYTES:
            raise ValueError('PyPI wheel metadata differs from the pinned source')
        if files[0]['url']!=WHEEL_URL:raise ValueError('PyPI wheel URL differs from the pinned source')
        with urllib.request.urlopen(WHEEL_URL, timeout=90) as response:content=response.read()
        if len(content)!=WHEEL_BYTES or digest(content)!=WHEEL_SHA256:raise ValueError('Downloaded wheel hash mismatch')
        path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(content)
    content=path.read_bytes()
    if len(content)!=WHEEL_BYTES or digest(content)!=WHEEL_SHA256:raise ValueError('Local wheel hash mismatch')


def load_member(archive, dataset):
    member='qsi/data/'+dataset['filename'];content=archive.read(member)
    rows=list(csv.reader(io.StringIO(content.decode('utf-8-sig'))))
    if rows[0][0].strip().lower()!='label':raise ValueError('CSV must have label as first column')
    if len(rows)!=dataset['expected_rows']+1 or any(len(row)!=dataset['expected_features']+1 for row in rows):
        raise ValueError('Source CSV dimensions differ from preregistered expectations')
    axis=np.asarray([float(v) for v in rows[0][1:]],dtype=np.float64)
    labels=np.asarray([float(row[0]) for row in rows[1:]],dtype=np.float64)
    X=np.asarray([[float(v) for v in row[1:]] for row in rows[1:]],dtype=np.float64)
    if not np.isfinite(axis).all() or not np.isfinite(labels).all() or not np.isfinite(X).all():
        raise ValueError('Nonfinite axis, label, or feature values')
    if not np.all(np.diff(axis)>0):raise ValueError('Source axis is not strictly increasing')
    if not np.array_equal(labels,np.round(labels)):raise ValueError('Class labels are not integers')
    labels=labels.astype(np.int64)
    if not np.array_equal(np.unique(labels),np.arange(len(dataset['class_names']))):
        raise ValueError('Class labels differ from documented source vocabulary')
    # Canonicalize signed zeros for identity only; spectral values are preserved.
    groups={};ordered=[]
    for row_number,(spectrum,label) in enumerate(zip(X,labels)):
        canonical=np.asarray(spectrum,dtype='<f8').copy();canonical[canonical==0]=0.
        identity=digest(canonical.tobytes())
        if identity not in groups:
            groups[identity]={'id':identity,'first_original_row_zero_based':row_number,
                              'original_rows_zero_based':[],'label':int(label)}
            ordered.append(identity)
        if groups[identity]['label']!=int(label):raise ValueError('Identical spectra have conflicting class labels')
        groups[identity]['original_rows_zero_based'].append(row_number)
    mapping=[groups[k] for k in ordered]
    first=np.asarray([r['first_original_row_zero_based'] for r in mapping])
    unique_X,unique_y,ids=X[first],labels[first],np.asarray(ordered)
    if min(Counter(unique_y).values())<5:raise ValueError('Deduplicated class too small for five folds')
    description_member='qsi/data/'+dataset['description_file']
    description=archive.read(description_member)
    info={**dataset,'csv_member':member,'csv_sha256':digest(content),
          'description_member':description_member,'description_sha256':digest(description),
          'original_rows':len(X),'deduplicated_rows':len(unique_X),'removed_exact_duplicate_rows':len(X)-len(unique_X),
          'features':X.shape[1],
          'original_class_counts':{str(k):int(v) for k,v in sorted(Counter(labels).items())},
          'deduplicated_class_counts':{str(k):int(v) for k,v in sorted(Counter(unique_y).items())},
          'axis_first':float(axis[0]),'axis_last':float(axis[-1]),'axis_spacing_median':float(np.median(np.diff(axis))),
          'all_values_finite':True,'conflicting_duplicate_labels':0,
          'constant_features':int(np.sum(np.std(X,axis=0)<=np.finfo(float).eps)),
          'constant_spectra':int(np.sum(np.std(X,axis=1)<=np.finfo(float).eps)),
          'source_feature_range':[float(X.min()),float(X.max())]}
    return unique_X,unique_y,ids,axis,mapping,info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel',type=Path,default=REPO/'outputs/research-stage2/herb_catalog_downloads'/WHEEL_NAME)
    parser.add_argument('--output',type=Path,default=REPO/'outputs/research-stage2/herb_inputs')
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):parser.error('Choose a new/empty output directory')
    checked_wheel(args.wheel,args.download)
    args.output.mkdir(parents=True,exist_ok=True)
    specs=[];datasets=[];audits=[]
    with zipfile.ZipFile(args.wheel) as archive:
        license_member='qsi_tk-1.4.1.dist-info/LICENCE'
        license_text=archive.read(license_member).decode('utf-8-sig')
        for dataset in DATASETS:
            X,y,ids,axis,mapping,info=load_member(archive,dataset)
            write_json(args.output/(dataset['id']+'_original_row_mapping.json'),mapping)
            counts=np.zeros(len(y),dtype=int);fold_audits=[]
            for number,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y),1):
                if set(ids[train])&set(ids[test]):raise ValueError('Exact spectrum identity crosses train/test')
                counts[test]+=1
                item_id=dataset['id']+f'_fold{number}'
                path=args.output/(item_id+'.npz')
                np.savez_compressed(path,X_train=X[train],y_train=y[train],X_test=X[test],y_test=y[test],
                                    train_ids=ids[train],test_ids=ids[test],feature_axis=axis,
                                    feature_axis_unit=np.asarray(dataset['feature_axis_unit']))
                spec={'id':item_id,'dataset':dataset['id'],'herb':dataset['source_name'],'fold':number,
                      'task':'classification','data_path':str(path.resolve()),'data_sha256':digest(path.read_bytes()),
                      'source_url':PYPI_PAGE,'source_wheel_sha256':WHEEL_SHA256,'source_member':info['csv_member'],
                      'source_member_sha256':info['csv_sha256'],'target_name':'source geographic-origin label',
                      'target_unit':'class','class_names':dataset['class_names'],
                      'class_mapping_status':dataset['class_mapping_status'],'cohort_id':dataset['cohort_id'],
                      'preprocess_modes':['raw','snv'],
                      'input_transform':'None; preserve packaged numeric spectra and full source axis',
                      'split_policy':f'No CSV official split; deduplicate exact spectra first, then StratifiedKFold5 shuffle=True seed={SEED}',
                      'identity_scope':'SHA256 of canonical numeric spectrum; not original physical specimen or batch IDs',
                      'n_train':len(train),'n_test':len(test),'n_features':X.shape[1],
                      'target_leakage_note':'No use of held-out scores in deduplication, fixed split selection, or preparation'}
                specs.append(spec)
                fold_audits.append({'fold':number,'train_ids':ids[train].tolist(),'test_ids':ids[test].tolist(),
                                    'train_class_counts':{str(k):int(v) for k,v in sorted(Counter(y[train]).items())},
                                    'test_class_counts':{str(k):int(v) for k,v in sorted(Counter(y[test]).items())},
                                    'exact_spectrum_overlap':0})
            if not np.all(counts==1):raise ValueError('Every deduplicated spectrum must be outer test once')
            datasets.append(info)
            audits.append({'dataset':dataset['id'],'folds':fold_audits,'every_unique_spectrum_tested_once':True,
                           'original_row_mapping_file':dataset['id']+'_original_row_mapping.json',
                           'physical_specimen_ids_available':False,'batch_ids_available':False})
    write_json(args.output/'spec.json',{'datasets':specs})
    write_json(args.output/'spec_firstfold.json',{'datasets':[s for s in specs if s['fold']==1]})
    write_json(args.output/'source_registry.json',{
        'package':'qsi-tk','version':'1.4.1','metadata_url':PYPI_JSON,'source_url':PYPI_PAGE,
        'wheel':{'filename':WHEEL_NAME,'url':WHEEL_URL,'sha256':WHEEL_SHA256,'bytes':WHEEL_BYTES},
        'wheel_read_policy':'Read ZIP CSV/text members only; package code/bytecode never imported, installed, deserialized, or executed.',
        'license':{'member':license_member,'sha256':digest(license_text.encode()),
                   'statement':license_text.strip(),'scope':'Statement explicitly licenses code LGPL3.0; a separate data-specific license is not established.',
                   'delivery':'Use source links/downloader and hashes; keep raw arrays and wheel under ignored outputs/.'},
        'cohort_counting':'Four modalities/tasks represent at most three described cohorts; the two Huangqi modalities share a160-sample source cohort, and independently verified physical specimen count remains unknown.',
        'datasets':datasets})
    write_json(args.output/'audit.json',{
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':digest(Path(__file__).read_bytes()),
        'wheel_hash_verified':True,'seed':SEED,'outer_folds':5,'fold_tasks':len(specs),
        'deduplication':'Exact numeric X hash; collapse to first representative and preserve all original row indices; conflicting labels rejected.',
        'preprocessing_fitted':False,'test_scores_used':False,'datasets':audits,
        'limitations':['Spectrum-row independence only; no physical specimen, acquisition batch, or replicate grouping provided.',
                       'Raman and UV Huangqi tasks are not independent cohorts.',
                       'Packaged upstream spectral processing is not independently audited; this adapter does not fit or scale spectra.',
                       'Raspberry species identity and Shihu integer-to-place mapping remain unverified as recorded in registry.']})
    print(json.dumps({'fold_tasks':len(specs),'source_wheel_sha256':WHEEL_SHA256,
                      'datasets':[{'id':d['id'],'original_rows':d['original_rows'],'unique_rows':d['deduplicated_rows'],
                                   'features':d['features'],'duplicates_removed':d['removed_exact_duplicate_rows']} for d in datasets]},indent=2))


if __name__=='__main__':main()
