"""Layout-only second figure; original figure and all numeric results preserved."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/research4-conservative-pilot-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

r=Path(__file__).resolve().parent
v=json.loads((r/'metrics.json').read_text());p=json.loads((r/'protocol.json').read_text())
arms=p['arms'];labels=p['classes']
names=['Scratch','Target-source\npretrain + L2-SP','Huangqi + Tianma\npretrain + L2-SP']
fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
maximum=max(max(max(row) for row in v['per_seed'][f'42.{a}']['confusion_matrix']) for a in arms)
for ax,arm,name in zip(axes,arms,names):
    matrix=np.asarray(v['per_seed'][f'42.{arm}']['confusion_matrix'])
    image=ax.imshow(matrix,vmin=0,vmax=maximum,cmap='Blues')
    for i in range(7):
        for j in range(7):
            ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=7,
                color='white' if matrix[i,j]>.6*maximum else 'black')
    ax.set_xticks(range(7),labels,rotation=45,fontsize=8);ax.set_yticks(range(7),labels,fontsize=8)
    ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(name,fontsize=10,pad=10)
fig.colorbar(image,ax=axes,shrink=.75,label='Prediction count (N=855 per arm)')
fig.suptitle('Seed 42: first predeclared seed; all other seeds retained in CSV/JSON',fontsize=11,y=1.12)
for extension in ('png','pdf'):
    target=r/('confusion_seed42_layout2.'+extension)
    if target.exists():raise FileExistsError(target)
    fig.savefig(target,dpi=200,bbox_inches='tight')
plt.close(fig)
