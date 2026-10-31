"""Read-only Gastrodia upload profile. It never rewrites uploaded workbooks."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import hashlib, json, warnings, zipfile
import numpy as np
import openpyxl
from herb_spectra.io import read_spectra

ROOT=Path('/workspace/research4')
WORK=ROOT/'outputs/teacher-intake-20261031'
items=json.loads((WORK/'downloads.json').read_text())
with np.load(ROOT/'outputs/approved-zaoren-20261031/datasets/candidate_0p1_1p5.npz',allow_pickle=False) as a:
    primary_axis=a['axis'].copy()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
    return h.hexdigest()

def reference(h):
    return h is not None and ('参考' in str(h) or str(h).strip().lower() in {'reference','ref'})

def numeric(value):
    if value is None: return np.nan
    try: return float(value)
    except (TypeError,ValueError): return np.nan

profiles=[]
for item in items:
    if item['dataset']!='gastrodia':continue
    path=Path(item['download_path']); before=digest(path)
    result={'file_name':item['file_name'],'file_id':item['file_id'],'download_path':str(path),
            'size_bytes':path.stat().st_size,'sha256':before,'sheets':[],'zip_members':[]}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            record={'member':member.filename,'uncompressed_bytes':member.file_size,'crc_pass':False}
            try:
                with archive.open(member) as handle:
                    for part in iter(lambda:handle.read(1024*1024),b''):pass
                record['crc_pass']=True
            except Exception as exc:
                record.update(error_type=type(exc).__name__,error=str(exc))
            result['zip_members'].append(record)
    result['zip_crc_pass']=all(r['crc_pass'] for r in result['zip_members'])
    result['bad_zip_members']=[r['member'] for r in result['zip_members'] if not r['crc_pass']]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        workbook=openpyxl.load_workbook(path,read_only=True,data_only=True)
    try:
        for ws in workbook:
            record={'sheet':ws.title,'complete_read':False}
            try:
                record['declared_dimensions']=ws.calculate_dimension()
                ws.reset_dimensions()
                stream=iter(ws.iter_rows(values_only=True)); header=next(stream,())
                data=[]; width=len(header); nonempty_rows=0; max_nonempty_col=0
                for row in stream:
                    data.append(row);width=max(width,len(row))
                    if any(v is not None for v in row):nonempty_rows+=1
                    max_nonempty_col=max(max_nonempty_col,max((i+1 for i,v in enumerate(row) if v is not None),default=0))
                labels=list(header)+[None]*max(0,width-len(header))
                columns=[i for i,h in enumerate(labels) if i>0 and h is not None and str(h).strip() and not reference(h)]
                refs=[i for i,h in enumerate(labels) if i>0 and reference(h)]
                matrix=np.full((len(data),width),np.nan)
                non_numeric=0
                for i,row in enumerate(data):
                    for j,v in enumerate(row):
                        matrix[i,j]=numeric(v)
                        if v is not None and not np.isfinite(matrix[i,j]):
                            try:float(v)
                            except (TypeError,ValueError):non_numeric+=1
                axis=matrix[:,0] if width else np.array([]);finite=axis[np.isfinite(axis)]
                spectrum=matrix[:,columns].T
                duplicate_names=[h for h,count in __import__('collections').Counter(str(labels[c]) for c in columns).items() if count>1]
                hashgroups=defaultdict(list)
                for c in columns:hashgroups[hashlib.sha256(matrix[:,c].astype('<f8').tobytes()).hexdigest()].append(str(labels[c]))
                record.update(complete_read=True,actual_rows_including_header=len(data)+bool(header),actual_data_rows=len(data),
                              nonempty_data_rows=nonempty_rows,actual_max_tuple_columns=width,last_nonempty_data_column=max_nonempty_col,
                              coordinate_header=header[0] if header else None,reference_columns=[{'excel_column_number':c+1,'header':str(labels[c])} for c in refs],
                              named_spectrum_column_count=len(columns),sample_headers=[str(labels[c]) for c in columns],
                              sample_excel_columns=[c+1 for c in columns],duplicate_sample_headers=duplicate_names,
                              spectra_nonfinite_cells=int((~np.isfinite(spectrum)).sum()),non_numeric_cells=non_numeric,
                              axis_points=len(axis),axis_finite_points=len(finite),axis_min=float(finite.min()) if len(finite) else None,
                              axis_max=float(finite.max()) if len(finite) else None,
                              axis_strictly_increasing=bool(len(axis)>1 and np.isfinite(axis).all() and (np.diff(axis)>0).all()),
                              axis_sha256=hashlib.sha256(axis.astype('<f8').tobytes()).hexdigest(),
                              exact_duplicate_spectrum_header_groups=[v for v in hashgroups.values() if len(v)>1])
            except Exception as exc:
                record.update(error_type=type(exc).__name__,error=str(exc),note='Incomplete sheet read; no complete dimensions or spectral counts claimed')
            result['sheets'].append(record)
    finally:workbook.close()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            alpha=read_spectra(path,'吸收系数')
        mask=(alpha.axis>=.1)&(alpha.axis<=1.5)
        result['alpha']={'strict_reader_pass':True,'record_count':len(alpha.scan_names),'frequency_points':len(alpha.axis),
                         'scan_names':list(alpha.scan_names),'scan_excel_columns':list(alpha.column_numbers),
                         'nonfinite_cells':int((~np.isfinite(alpha.values)).sum()),'negative_cells_full_export':int((alpha.values<0).sum()),
                         'candidate_0p1_1p5_points':int(mask.sum()),'candidate_nonfinite_cells':int((~np.isfinite(alpha.values[:,mask])).sum()),
                         'candidate_negative_cells':int((alpha.values[:,mask]<0).sum()),
                         'candidate_axis_equal_zaoren_612':bool(np.array_equal(alpha.axis[mask],primary_axis)),
                         'candidate_axis_sha256':hashlib.sha256(alpha.axis[mask].astype('<f8').tobytes()).hexdigest(),
                         'physical_independence_verified':False,'true_origin_verified':False,'species_verified':False,
                         'batch_verified':False,'absorption_length_unit_verified':False,
                         'axis_numeric_equality_does_not_establish_physical_compatibility':True}
    except Exception as exc:result['alpha']={'strict_reader_pass':False,'error_type':type(exc).__name__,'error':str(exc)}
    assert before==digest(path)
    result['input_bytes_unchanged']=True
    result['use_status']='quarantine_container_integrity_failed' if not result['zip_crc_pass'] else 'archived_pending_scientific_annotations'
    profiles.append(result)
    print(json.dumps({'file':path.name,'crc_pass':result['zip_crc_pass'],'alpha':result['alpha'].get('record_count'),'alpha_points':result['alpha'].get('frequency_points'),'sheets':[(s['sheet'],s['complete_read'],s.get('actual_data_rows'),s.get('named_spectrum_column_count')) for s in result['sheets']]},ensure_ascii=False),flush=True)

value={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Read-only new Gastrodia file and sheet audit, not independent sample verification or model fitting',
       'source_script_sha256':digest(Path(__file__)),'profiles':profiles,'total_alpha_records_if_readable':sum(p['alpha'].get('record_count',0) for p in profiles),
       'new_model_fits':0,'input_workbooks_rewritten':0,'origin_names_are_filename_text_only':True}
(WORK/'gastrodia_profiles.json').write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print('saved',str(WORK/'gastrodia_profiles.json'))
