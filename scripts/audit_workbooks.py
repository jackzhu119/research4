import hashlib
import json
import warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import openpyxl
from herb_spectra.dataset_registry import REPO, select_workbooks

warnings.filterwarnings('ignore', message='Workbook contains no default style')
import argparse
parser=argparse.ArgumentParser(description='Profile actual workbook cells, ignoring dimension declarations')
selection=parser.add_mutually_exclusive_group()
selection.add_argument('--manifest',type=Path,help='Explicit active-workbook registry')
selection.add_argument('--data-root',type=Path,help='Explicit legacy directory override; bypasses active registry')
parser.add_argument('--output',type=Path,help='Full audit defaults to outputs/workbook_profiles.json; list-only prints unless provided')
parser.add_argument('--list-only',action='store_true',help='Validate registry/hash and list inputs without reading spectra')
args=parser.parse_args()
sources=select_workbooks(repo=REPO,manifest=args.manifest,data_root=args.data_root)
if args.list_only:
    listing={'selection_mode':sources[0].selection_mode,'dataset_version':sources[0].dataset_version,
             'workbook_count':len(sources),'declared_record_count':sum(s.record_count or 0 for s in sources),
             'workbooks':[{'path':s.relative_path,'sha256':s.sha256,'dataset':s.dataset,
                           'record_count':s.record_count,'origin_group_id':s.origin_group_id} for s in sources]}
    rendered=json.dumps(listing,ensure_ascii=False,indent=2)+'\n'
    print(rendered,end='')
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered,encoding='utf-8')
    raise SystemExit(0)
profiles=[]
for source in sources:
    path=source.path
    wb=openpyxl.load_workbook(path,read_only=True,data_only=True)
    profile={'file':source.relative_path,'dataset_version':source.dataset_version,
             'selection_mode':source.selection_mode,'sheets':[]}
    for ws in wb:
        declared=ws.calculate_dimension()
        ws.reset_dimensions()
        rows=iter(ws.iter_rows(values_only=True))
        header=next(rows,())
        data=list(rows)
        width=max([len(header)]+[len(r) for r in data])
        matrix=np.full((len(data),width),np.nan)
        non_numeric=0
        for i,row in enumerate(data):
            for j,v in enumerate(row):
                if v is None:continue
                try:matrix[i,j]=float(v)
                except (TypeError,ValueError):non_numeric+=1
        labels=list(header)+['']*max(0,width-len(header))
        sample_indices=[i for i,label in enumerate(labels) if i>0 and label is not None and '参考' not in str(label)]
        spectral=matrix[:,sample_indices].T if sample_indices else np.empty((0,len(data)))
        good_axis=matrix[:,0] if width and len(data) else np.array([])
        finite_axis=good_axis[np.isfinite(good_axis)]
        hashes=defaultdict(list)
        for j,col in enumerate(sample_indices):
            hashes[hashlib.sha256(spectral[j].astype('<f8').tobytes()).hexdigest()].append(str(labels[col]))
        duplicates=[g for g in hashes.values() if len(g)>1]
        sheet={'sheet':ws.title,'declared_dimensions':declared,'actual_data_rows':len(data),'actual_columns':width,
               'dimension_underreported':bool(width>1 and declared=='A1:A1'),
               'headers_first_8':list(header)[:8], 'last_header':header[-1] if header else None,
               'spectra_columns':len(sample_indices),'numeric_nonfinite_cells':int((~np.isfinite(spectral)).sum()),
               'non_numeric_cells':non_numeric,'axis_min':float(finite_axis.min()) if len(finite_axis) else None,
               'axis_max':float(finite_axis.max()) if len(finite_axis) else None,
               'axis_strictly_increasing':bool(len(good_axis)>1 and np.isfinite(good_axis).all() and (np.diff(good_axis)>0).all()),
               'duplicate_spectrum_column_groups':duplicates}
        if ws.title=='吸收系数' and source.record_count is not None and len(sample_indices)!=source.record_count:
            raise ValueError(f'Registered record_count differs from workbook: {source.relative_path}')
        if len(finite_axis)>0:
            sheet['axis_sha256']=hashlib.sha256(good_axis.astype('<f8').tobytes()).hexdigest()
        if len(spectral):
            mask=np.isfinite(good_axis)&(good_axis>=0.1)&(good_axis<=1.5)
            sheet['candidate_0p1_1p5_points']=int(mask.sum())
            sheet['candidate_band_nonfinite_cells']=int((~np.isfinite(spectral[:,mask])).sum())
        profile['sheets'].append(sheet)
        print(json.dumps({'file':path.name,**sheet},ensure_ascii=False),flush=True)
    wb.close();profiles.append(profile)
output=args.output or Path('outputs/workbook_profiles.json')
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(profiles,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
