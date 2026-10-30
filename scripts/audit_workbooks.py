import hashlib
import json
import warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import openpyxl

warnings.filterwarnings('ignore', message='Workbook contains no default style')
import argparse
parser=argparse.ArgumentParser(description='Profile actual workbook cells, ignoring dimension declarations')
parser.add_argument('--data-root',type=Path,default=Path('data/raw'))
parser.add_argument('--output',type=Path,default=Path('outputs/workbook_profiles.json'))
args=parser.parse_args()
repo=Path.cwd()
profiles=[]
for path in sorted(args.data_root.resolve().rglob('*.xlsx')):
    wb=openpyxl.load_workbook(path,read_only=True,data_only=True)
    profile={'file':path.relative_to(repo).as_posix(),'sheets':[]}
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
        if len(finite_axis)>0:
            sheet['axis_sha256']=hashlib.sha256(good_axis.astype('<f8').tobytes()).hexdigest()
        if len(spectral):
            mask=np.isfinite(good_axis)&(good_axis>=0.1)&(good_axis<=1.5)
            sheet['candidate_0p1_1p5_points']=int(mask.sum())
            sheet['candidate_band_nonfinite_cells']=int((~np.isfinite(spectral[:,mask])).sum())
        profile['sheets'].append(sheet)
        print(json.dumps({'file':path.name,**sheet},ensure_ascii=False),flush=True)
    wb.close();profiles.append(profile)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(profiles,ensure_ascii=False,indent=2)+'\n')
