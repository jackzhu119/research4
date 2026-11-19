"""Verify exported accepted quantum tables; no electronic-structure jobs are run."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
base=Path(__file__).resolve().parent
checks={}
for label in ['acid_d2f61d35','acid_f626f77f']:
 p=base/label; result=json.loads((p/'result.json').read_text());complete=json.loads((p/'COMPLETE.json').read_text());q=json.loads((p/'upstream_publication_quality.json').read_text());a=np.load(p/'ir_arrays.npz');tab=pd.read_csv(p/'ir_modes.csv');L=a['cartesian_modes'];mass=np.repeat(a['masses_amu'],3)
 item={'result_sha256_valid':hashlib.sha256((p/'result.json').read_bytes()).hexdigest()==complete['result_sha256'],'npz_sha256_valid':hashlib.sha256((p/'ir_arrays.npz').read_bytes()).hexdigest()==result['array_sha256'],'csv_sha256_valid':hashlib.sha256((p/'ir_modes.csv').read_bytes()).hexdigest()==result['csv_sha256'],'source_case_sha256_valid':hashlib.sha256((p/'upstream_case_result.json').read_bytes()).hexdigest()==result['source']['source_result_sha256'],'minimum_verified':q['minimum_verified'],'upstream_failed_gates':q['failed_gates'],'independent_ir_status':result['status'],'independent_ir_paper_ready':result['paper_ready'],'mass_normalization_max_error':float(np.max(abs(L.T@(mass[:,None]*L)-np.eye(45)))),'frequency_csv_npz_max_difference_cm1':float(np.max(abs(a['signed_frequencies_cm1']-tab.signed_frequency_cm1))),'intensity_reconstruction_max_difference_km_mol':float(np.max(abs(974.8801098*np.sum(a['projected_dipole_derivatives']**2,axis=1)-tab.IR_intensity_km_mol))),'mode_count':len(tab),'imaginary_count':int(tab.imaginary.sum()),'analysis_band_mode_count_0p1_to_1p5_THz':int(tab.signed_frequency_THz.between(.1,1.5).sum()),'lowest_cm1':float(tab.signed_frequency_cm1.min()),'lowest_THz':float(tab.signed_frequency_THz.min()),'lowest_IR_km_mol':float(tab.IR_intensity_km_mol.iloc[0])}
 assert all(item[k] for k in ['result_sha256_valid','npz_sha256_valid','csv_sha256_valid','source_case_sha256_valid','minimum_verified','independent_ir_paper_ready'])
 assert not item['upstream_failed_gates'] and item['imaginary_count']==0 and item['mode_count']==45 and item['mass_normalization_max_error']<1e-9 and item['frequency_csv_npz_max_difference_cm1']<1e-9 and item['intensity_reconstruction_max_difference_km_mol']<1e-7
 checks[label]=item
x=pd.read_csv(base/'acid_d2f61d35'/'ir_modes.csv');y=pd.read_csv(base/'acid_f626f77f'/'ir_modes.csv');d=x.signed_frequency_cm1-y.signed_frequency_cm1;e=x.IR_intensity_km_mol-y.IR_intensity_km_mol
checks['comparison']={'paired_by':'ascending frequency, all 45 modes; descriptive comparison of two model profiles','max_abs_frequency_difference_cm1':float(abs(d).max()),'rms_frequency_difference_cm1':float(np.sqrt(np.mean(d*d))),'max_abs_intensity_difference_km_mol':float(abs(e).max()),'experimental_validation':False}
(base/'molecular_table_verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
print(json.dumps(checks,ensure_ascii=False,indent=2))
