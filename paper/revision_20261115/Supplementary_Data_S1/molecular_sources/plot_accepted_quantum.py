"""Plot archived, accepted PCA IR sticks without synthetic broadening."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path(__file__).resolve().parent
out=base.parent/'figures';out.mkdir(exist_ok=True)
direct=pd.read_csv(base/'acid_d2f61d35'/'ir_modes.csv');df=pd.read_csv(base/'acid_f626f77f'/'ir_modes.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(10.7,3.8),gridspec_kw={'width_ratios':[1,1.65]})
for ax in axs:
 ax.axvspan(.1,1.5,color='#cbd5e1',alpha=.6,label='Experimental analysis interval')
 ax.vlines(direct.signed_frequency_THz,0,direct.IR_intensity_km_mol,color='#204c71',lw=1.8,label='Direct integrals')
 ax.vlines(df.signed_frequency_THz,0,df.IR_intensity_km_mol,color='#d47b27',lw=1,linestyles='dashed',label='RI-J')
 ax.set_xlabel('Frequency (THz)');ax.set_ylabel('Predicted IR intensity (km/mol)');ax.set_ylim(bottom=0);ax.tick_params(direction='out')
axs[0].set_xlim(0,4);axs[0].set_ylim(0,2.85);axs[0].set_title('(a) Low-frequency region',loc='left',fontweight='bold')
axs[0].text(.8,1.5,'0.1–1.5 THz\nanalysis interval',ha='center',va='center',fontsize=9,color='#475569')
axs[0].annotate('Lowest mode:\n2.21865–2.21940 THz',xy=(2.219,2.14),xytext=(2.5,2.5),ha='center',fontsize=8.5,arrowprops={'arrowstyle':'-','color':'#475569'})
axs[1].set_xlim(0,115);axs[1].set_ylim(0,330);axs[1].set_title('(b) Complete 45-mode prediction',loc='left',fontweight='bold')
axs[1].legend(frameon=False,loc='upper right',fontsize=8.5)
fig.text(.5,.015,'Isolated protocatechuic acid · PBE/def2-SVP · no D3 · no fitted band broadening',ha='center',fontsize=9,color='#475569')
fig.tight_layout(rect=(0,.06,1,1),w_pad=2)
for ext in ['png','pdf','svg']:fig.savefig(out/('Fig_S_quantum_accepted_PCA.'+ext),dpi=300,bbox_inches='tight')
print(out/'Fig_S_quantum_accepted_PCA.png')
