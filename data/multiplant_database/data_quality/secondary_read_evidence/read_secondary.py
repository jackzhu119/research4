from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

ROOT = Path('/workspace/research4')
OUTPUT = ROOT/'outputs/multiplant-secondary-20261031'
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
MODS = {'吸收系数':'alpha', '折射率':'n', '时域':'time', '频域':'frequency'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def colno(value):
    letters = re.match(r'[A-Z]+',value).group()
    number = 0
    for letter in letters: number = number*26+ord(letter)-64
    return number


def cellvalue(cell, strings):
    kind = cell.attrib.get('t','n')
    if kind == 'inlineStr': return ''.join(x.text or '' for x in cell.iter(NS+'t'))
    v = cell.find(NS+'v')
    if v is None or v.text is None: return None
    if kind == 's': return strings[int(v.text)]
    if kind in ('str','e'): return v.text
    try: return float(v.text)
    except ValueError: return v.text


def numeric(value):
    try: return float(value), False
    except (ValueError,TypeError): return np.nan, value is not None


def read_sheet(z, member, strings):
    header = None; matrix=[]; references=[]; nonnumeric=0; data_rows=[]
    with z.open(member) as handle:
        for _, elem in ET.iterparse(handle,events=('end',)):
            if elem.tag != NS+'row': continue
            values={colno(c.attrib['r']):cellvalue(c,strings) for c in elem.findall(NS+'c')}
            if not any(v is not None for v in values.values()): elem.clear(); continue
            if header is None:
                header=values
                named=[n for n,v in values.items() if n>1 and v is not None and str(v).strip() and '参考' not in str(v) and str(v).lower() not in ('ref','reference')]
                ref=[n for n,v in values.items() if n>1 and v is not None and ('参考' in str(v) or str(v).lower() in ('ref','reference'))]
                elem.clear(); continue
            row=[]
            for n in [1,*named]:
                v, bad=numeric(values.get(n)); row.append(v); nonnumeric+=bad
            rr=[numeric(values.get(n))[0] for n in ref]
            matrix.append(row); references.append(rr);data_rows.append(int(elem.attrib.get('r','0')))
            elem.clear()
    if header is None or not matrix or not named: raise ValueError('Empty worksheet or no named spectral columns')
    a=np.asarray(matrix,dtype=np.float64); refs=np.asarray(references,dtype=np.float64)
    headers=[str(header[n]) for n in named];bases=[];thickness=[];suffixes=[]
    for h in headers:
        m=re.fullmatch(r'(样品|样本)\s*(\d+)(.*)',h.strip())
        bases.append((m.group(1)+str(int(m.group(2)))) if m else '')
        suffixes.append(m.group(3) if m else '')
        t=re.search(r'(?:^|_)(\d+(?:\.\d+)?)\s*mm(?:$|_)',m.group(3) if m else '',re.I)
        thickness.append(float(t.group(1)) if t else np.nan)
    return a[:,0],a[:,1:].T,np.asarray(headers),np.asarray(bases),np.asarray(named),np.asarray(ref),refs.T,np.asarray(thickness),suffixes,nonnumeric,data_rows,header


def process(item):
    path=ROOT/item['source_file']; before=sha(path); folder=OUTPUT/item['source_id'];folder.mkdir(parents=True,exist_ok=True)
    result={**item,'sha256':before,'independence_evidence':'unknown','physical_sample_ids_assigned':False,'sheets':[]}
    with zipfile.ZipFile(path) as z:
        members=[]
        for member in z.infolist():
            try:
                with z.open(member) as h:
                    while h.read(1024*1024): pass
                members.append({'member':member.filename,'crc':'pass'})
            except Exception as exc: members.append({'member':member.filename,'crc':'fail','error':type(exc).__name__+': '+str(exc)})
        result['zip_members']=members;result['container_crc_pass']=all(m['crc']=='pass' for m in members)
        health={m['member']:m for m in members}
        workbook=ET.fromstring(z.read('xl/workbook.xml'))
        rels={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(t.text or '' for t in si.iter(NS+'t')) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall(NS+'si')]
        for sheet in workbook.find(NS+'sheets'):
            name=sheet.attrib['name'];modality=MODS.get(name)
            target=rels[sheet.attrib[REL]]
            member=target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl',target))
            rec={'sheet':name,'modality':modality,'zip_member':member,'crc':health.get(member,{}).get('crc','not_found')}
            if modality is None: rec.update(status='NOT_USED',error='Unrecognized sheet');result['sheets'].append(rec);continue
            if rec['crc']!='pass': rec.update(status='BLOCKED_CORRUPT_SHEET',error=health.get(member,{}).get('error','member missing'));result['sheets'].append(rec);continue
            try:
                axis,X,headers,bases,columns,refs,ref_values,thickness,suffixes,nonnumeric,data_rows,header=read_sheet(z,member,strings)
                duplicates=defaultdict(list)
                for i,row in enumerate(X):
                    canon=row.astype('<f8',copy=True);canon[canon==0]=0
                    duplicates[hashlib.sha256(canon.tobytes()).hexdigest()].append(str(headers[i]))
                exact=[names for names in duplicates.values() if len(names)>1]
                p=folder/(modality+'.npz')
                np.savez_compressed(p,X=X,axis=axis,headers=headers,bases=bases,column_numbers=columns,reference_columns=refs,reference_values=ref_values,thickness_mm=thickness)
                finite=bool(np.isfinite(axis).all());increasing=finite and bool((np.diff(axis)>0).all())
                rec.update(status='READ_COMPLETE',artifact=str(p.relative_to(ROOT)),artifact_sha256=sha(p),coordinate_header=str(header.get(1,'')),coordinate_unit='THz' if 'thz' in str(header.get(1,'')).lower() else 'unknown',coordinate_unit_evidence='literal_axis_header_only',records=len(X),axis_points=len(axis),axis_min=float(np.nanmin(axis)),axis_max=float(np.nanmax(axis)),axis_finite=finite,axis_strictly_increasing=increasing,nonfinite_cells=int((~np.isfinite(X)).sum()),negative_cells=int((X<0).sum()),nonnumeric_cells=nonnumeric,sample_headers=headers.tolist(),sample_bases=bases.tolist(),sample_suffixes=suffixes,sample_excel_columns=columns.tolist(),reference_columns=refs.tolist(),reference_column_headers=[str(header[int(n)]) for n in refs],duplicate_sample_bases=[b for b in set(bases.tolist()) if bases.tolist().count(b)>1],unparsed_sample_headers=[str(h) for h,b in zip(headers,bases) if not b],exact_duplicate_header_groups=exact,thickness_mm=[None if not np.isfinite(v) else float(v) for v in thickness],actual_data_rows=len(data_rows),first_excel_data_row=min(data_rows),last_excel_data_row=max(data_rows),axis_sha256=hashlib.sha256(axis.astype('<f8').tobytes()).hexdigest())
            except Exception as exc: rec.update(status='BLOCKED_READ_ERROR',error=type(exc).__name__+': '+str(exc))
            result['sheets'].append(rec)
            print(json.dumps({'file':path.name,'modality':modality,'status':rec['status'],'records':rec.get('records'),'points':rec.get('axis_points')},ensure_ascii=False),flush=True)
    result['source_hash_after']=sha(path);result['source_bytes_unchanged']=before==result['source_hash_after']
    valid={r['modality']:r for r in result['sheets'] if r['status']=='READ_COMPLETE'}
    alpha=valid.get('alpha')
    result['header_pairing']={}
    if alpha:
        for m,r in valid.items():
            okay=bool(all(alpha['sample_bases'])) and len(set(alpha['sample_bases']))==len(alpha['sample_bases']) and len(set(r['sample_bases']))==len(r['sample_bases']) and set(alpha['sample_bases'])==set(r['sample_bases'])
            result['header_pairing'][m]={'header_base_sets_match':okay,'pairing_is_not_physical_independence_verification':True}
    (folder/'profile.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    registry=json.loads((ROOT/'data/active_workbooks.json').read_text())
    hq=[x for x in registry['workbooks'] if x['dataset']=='huangqi']
    receipt=json.loads((ROOT/'data/incoming/teacher_20261031_v1/receipts.json').read_text())
    tm=[x for x in receipt['uploads'] if x['dataset']=='gastrodia']
    items=[{'dataset':'huangqi','source_id':f'HQ{i+1:02d}','source_file':x['path'],'expected_sha256':x['sha256']} for i,x in enumerate(hq)]
    items += [{'dataset':'tianma','source_id':f'TM{i+1:02d}','source_file':x['archive_path'],'expected_sha256':x['sha256']} for i,x in enumerate(tm)]
    with ProcessPoolExecutor(max_workers=2) as pool: results=list(pool.map(process,items))
    for r in results: assert r['sha256']==r['expected_sha256'] and r['source_bytes_unchanged']
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'reader_script_sha256':sha(Path(__file__)),'source_workbooks':results,'source_workbooks_rewritten':0,'new_model_fits':0,'labels_inferred_from_filenames':False,'physical_ids_inferred_from_columns':False}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':True,'workbooks':len(results),'sheets_read':sum(s['status']=='READ_COMPLETE' for r in results for s in r['sheets']),'blocked_sheets':sum(s['status'].startswith('BLOCKED') for r in results for s in r['sheets'])}),flush=True)
