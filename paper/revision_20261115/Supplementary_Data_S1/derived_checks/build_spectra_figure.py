from pathlib import Path
import hashlib,json,warnings
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl.utils.cell import column_index_from_string

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'data/active_workbooks.json').exists())
OUT=Path(__file__).resolve().parent.parent
registry=json.loads((ROOT/'data/active_workbooks.json').read_text())
split=pd.read_csv(ROOT/'paper/current/inputs/split_manifest.csv')
train=split[(split.species=='酸枣仁')&(split.role=='train')]
assert len(train)==300
names={'ZY03':'Inner Mongolia','ZY04':'Lingshi, Shanxi','ZY05':'Bozhou, Anhui','ZY06':'Shijiazhuang, Hebei','ZY07':'Shanxi'}
rows=[];sources=[]
warnings.filterwarnings('ignore',message='Workbook contains no default style')
for w in registry['workbooks']:
    if w['origin_group_id'] not in names: continue
    path=ROOT/w['path']; assert hashlib.sha256(path.read_bytes()).hexdigest()==w['sha256']
    g=train[train.origin_group_id==w['origin_group_id']];assert len(g)==60
    idx=[column_index_from_string(x)-1 for x in g.excel_column]
    for sheet,mod in [('吸收系数','alpha'),('折射率','n')]:
        frame=pd.read_excel(path,sheet_name=sheet)
        axis=frame.iloc[:,0].to_numpy(float);mask=(axis>=.1)&(axis<=1.5)
        X=frame.iloc[:,idx].to_numpy(float)[mask];axis=axis[mask];assert X.shape==(612,60)
        assert all(frame.columns[i]==str(h) for i,h in zip(idx,g.scan_header))
        for x,m,s in zip(axis,X.mean(axis=1),X.std(axis=1,ddof=1)):
            rows.append({'group':w['origin_group_id'],'modality':mod,'frequency_thz':x,'mean':m,'sd':s,'specimens':60})
    sources.append({'path':w['path'],'sha256':w['sha256'],'records':g.record_id.tolist()})
d=pd.DataFrame(rows);d.to_csv(Path(__file__).resolve().parent/'spectra_means_current.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(7.1,3.0),layout='constrained')
colors=['#126782','#ef8354','#4b8f61','#805e9e','#aa8139']
for ax,mod,title,ylabel in zip(axs,['alpha','n'],['A  Absorption','B  Refractive index'],['Absorption (exported scale)','Refractive index']):
    for (group,label),c in zip(names.items(),colors):
        t=d[(d.group==group)&(d.modality==mod)]
        ax.plot(t.frequency_thz,t['mean'],label=label,color=c,lw=1.35)
    ax.set(title=title,xlabel='Frequency (THz)',ylabel=ylabel,xlim=(.1,1.5))
    ax.grid(alpha=.16,lw=.5)
axs[0].legend(frameon=False,fontsize=7.5,loc='upper left')
for suffix in ['png','pdf','svg']:fig.savefig((ROOT/'paper/revision_20261115/figures')/f'fig1_spectra_current.{suffix}',dpi=300,bbox_inches='tight')
(Path(__file__).resolve().parent/'spectra_figure_sources.json').write_text(json.dumps({'scope':'300 current Suanzaoren TRAIN specimens; raw exported alpha and n; no fitting','workbooks':sources,'frequency_points':612},ensure_ascii=False,indent=2))
print('PASS: 5 unchanged workbooks; 300 TRAIN specimens; 612 paired coordinates; figures saved')
