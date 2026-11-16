from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,FancyArrowPatch

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'paper/revision_20261116_r2'
OUT=BASE/'figures';OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'paper/revision_20261115/Supplementary_Tables.xlsx'
MODEL=pd.read_csv(BASE/'analysis/primary_models.csv').set_index('family')
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],
 'font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
 'axes.titlesize':11,'axes.titleweight':'bold','axes.linewidth':.8,
 'mathtext.fontset':'dejavuserif','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'figure.facecolor':'white','savefig.facecolor':'white'})
BLUE='#0072B2';ORANGE='#D55E00';GRAY='#666666';COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']
outputs=[]
def save(fig,stem):
    fig.add_artist(Rectangle((.008,.008),.984,.984,transform=fig.transFigure,fill=False,edgecolor='black',lw=.85,clip_on=False))
    for ax in fig.axes:
        for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(.8)
    for ext in ['png','eps','svg']:
        p=OUT/(stem+'.'+ext);fig.savefig(p,dpi=600)
        if ext=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
        outputs.append({'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    plt.close(fig)
def box(ax,x,y,w,h,text,size=10,fill='#F4F6F8',bold=False):
    patch=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.008,rounding_size=0.008',lw=.85,edgecolor='#333333',facecolor=fill)
    ax.add_patch(patch);ax.text(x,y,text,ha='center',va='center',fontsize=size,fontweight='bold' if bold else 'normal')
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=.85,color='#333333'))

# Figure 1: schematic drawn from the saved validation and estimator definitions.
fig,axs=plt.subplots(1,2,figsize=(7.2,6.6));fig.subplots_adjust(left=.035,right=.965,bottom=.105,top=.93,wspace=.10)
for ax in axs:ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
a,b=axs;a.set_title('(a) Nested model selection',loc='left',pad=10);b.set_title('(b) Training-only feature front end',loc='left',pad=10)
box(a,.5,.93,.92,.10,'300 Suanzaoren TRAIN specimens\nFive recorded groups × 60',9.6)
arrow(a,(.5,.875),(.5,.82));box(a,.5,.77,.92,.09,'Three stratified fivefold partitions',9.5)
arrow(a,(.5,.72),(.27,.66));arrow(a,(.5,.72),(.78,.66))
box(a,.27,.61,.48,.095,'Outer training\n240 specimens',9.7);box(a,.78,.61,.34,.095,'Outer test\n60 specimens',9.7,fill='white')
arrow(a,(.27,.56),(.27,.49));box(a,.27,.43,.48,.12,'Inner twofold validation\n120 fit / 120 validation\nSelect K, C and ρ',9.4)
arrow(a,(.27,.365),(.27,.30));box(a,.27,.245,.48,.10,'Refit selected pipeline\nAll 240 training specimens',9.2)
arrow(a,(.27,.19),(.5,.125));arrow(a,(.78,.56),(.78,.125));arrow(a,(.78,.125),(.69,.125))
box(a,.5,.075,.92,.095,'Predict held-out specimens\nPool folds within each partition',9.4)
box(b,.5,.93,.92,.10,'Spectral candidate matrix\nα: 612; joint controls: 1,224',9.5)
arrow(b,(.5,.875),(.5,.83));box(b,.5,.78,.92,.09,'Per-spectrum SNV as specified\nFit variable scaling on training data',9.1)
arrow(b,(.5,.73),(.5,.68));box(b,.5,.62,.92,.11,'20 class-stratified MC draws\nFive one-versus-rest PLS1 responses\nTwo components per fit',9.2)
arrow(b,(.5,.56),(.5,.51));box(b,.5,.445,.92,.12,'Multiclass coefficient consistency\n'+r'$q_{gj}=|\bar{\beta}_{gj}|/\max(s_{gj},\varepsilon)$'+'\n'+r'$S_j=(\sum_g q_{gj}^{2})^{1/2}$',10)
arrow(b,(.5,.38),(.5,.33));box(b,.5,.275,.92,.10,'Retain the K highest-ranked coordinates\n'+r'$K\in\{31,62,123,184,245\}$',9.3)
arrow(b,(.5,.22),(.5,.16));box(b,.5,.105,.92,.10,'Class-balanced RBF SVM\nUse inner-selected C and ρ',9.6)
fig.text(.5,.052,'Ranking is separate from the classifier; model-specific budgets remain inner-selected.',ha='center',fontsize=9)
fig.text(.5,.027,'Separate tasks: full-spectrum material LOWO and Source/Support/Query controls.',ha='center',fontsize=9)
save(fig,'Figure_1')

spectra=pd.read_excel(SOURCE,sheet_name='Source_mean_spectra')
names={'ZY03':'Inner Mongolia','ZY04':'Lingshi, Shanxi','ZY05':'Bozhou, Anhui','ZY06':'Shijiazhuang, Hebei','ZY07':'Shanxi group'}
fig,axs=plt.subplots(1,2,figsize=(7.2,3.7));fig.subplots_adjust(left=.10,right=.965,bottom=.30,top=.90,wspace=.35)
for ax,mod,title,label in zip(axs,['alpha','n'],['(a) Absorption export','(b) Refractive-index export'],['Absorption (exported scale)','Refractive index, n (dimensionless)']):
    for color,(group,name) in zip(COLORS,names.items()):
        p=spectra[(spectra.group==group)&(spectra.modality==mod)]
        assert len(p)==612 and p.specimens.eq(60).all()
        ax.plot(p.frequency_thz,p['mean'],lw=1.35,color=color,label=name)
    ax.set(xlabel='Frequency (THz)',ylabel=label,xlim=(.1,1.5));ax.set_title(title,loc='left');ax.grid(color='#E2E2E2',lw=.5)
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.025),ncol=3,frameon=False,fontsize=9)
save(fig,'Figure_2')

families=['alpha_snv20','alpha_raw20','alpha_snv100','fullalpha_shrinkLDA','concat_both_snv20','fullalpha_snv_svm_bridge','fullalpha_MLP']
labels=['SNV α MC20','Raw α MC20','SNV α MC100','Full α LDA','Joint SNV MC20','Full SNV α SVM','Full α MLP']
fig,(a,b)=plt.subplots(1,2,figsize=(7.2,4.7));fig.subplots_adjust(left=.19,right=.965,bottom=.16,top=.90,wspace=.90)
y=np.arange(len(families));vals=MODEL.loc[families].accuracy.to_numpy()*100
a.barh(y,vals,color=[BLUE if i==0 else '#BAC7CE' for i in y],height=.64,edgecolor='#333333',lw=.5)
a.set(yticks=y,yticklabels=labels,xlim=(60,101),xlabel='Mean accuracy (%)');a.invert_yaxis();a.set_title('(a) Matched pipelines',loc='left')
for i,v in enumerate(vals):a.text(100.5,i,f'{v:.2f}',ha='right',va='center',fontsize=9,color='black')
a.grid(axis='x',color='#E2E2E2',lw=.5);a.set_axisbelow(True)
eff=pd.read_csv(BASE/'analysis/paired_primary_effects.csv').set_index('comparator')
order=['fullalpha_snv_svm_bridge','fullalpha_shrinkLDA','alpha_raw20','alpha_snv100','concat_both_snv20']
elabels=['Full SVM','Full LDA','Raw α MC20','MC100','Joint SNV']
z=eff.loc[order];p=z.difference_pp.to_numpy();low=z.ci_low_pp.to_numpy();hi=z.ci_high_pp.to_numpy();ys=np.arange(len(order))
b.errorbar(p,ys,xerr=np.vstack([p-low,hi-p]),fmt='o',color=BLUE,ecolor=BLUE,ms=5,capsize=3,lw=1.1)
b.axvline(0,color='#555555',ls='--',lw=.8);b.set(yticks=ys,yticklabels=elabels,xlim=(-4,16),xlabel='MC20 − comparator (pp)');b.invert_yaxis();b.set_title('(b) Paired differences',loc='left');b.grid(axis='x',color='#E2E2E2',lw=.5)
fig.text(.5,.045,'300 physical specimens; three pooled fivefold estimates. Intervals condition on saved predictions.',ha='center',fontsize=9)
save(fig,'Figure_3')

prob=pd.read_csv(BASE/'analysis/v4_primary_selection_probability.csv');freq=spectra[spectra.modality.eq('alpha')].frequency_thz.drop_duplicates().sort_values().to_numpy();assert len(freq)==612
stable=pd.read_csv(BASE/'analysis/v4_stability.csv');fixed=pd.read_csv(BASE/'analysis/fixed_train_stability.csv')
fig=plt.figure(figsize=(7.2,5.1));gs=fig.add_gridspec(2,2,left=.10,right=.965,bottom=.13,top=.93,wspace=.38,hspace=.56);a=fig.add_subplot(gs[0,:]);b=fig.add_subplot(gs[1,0]);c=fig.add_subplot(gs[1,1])
a.plot(freq,prob.selection_probability,color=BLUE,lw=.9);a.set(xlabel='Frequency (THz)',ylabel='Selection proportion',xlim=(.1,1.5),ylim=(-.03,1.03));a.set_title('(a) Primary absorption across 15 outer refits',loc='left');a.grid(color='#E2E2E2',lw=.5)
rows=[stable[(stable.family=='alpha_snv20')&(stable.budget==v)].iloc[0] for v in ['primary','62']]
ix=np.arange(2)
for offset,key,label,color in [(-.16,'Jaccard','Jaccard',BLUE),(.16,'Nogueira_Phi','Nogueira Φ',ORANGE)]:
    vals=np.array([r[key] for r in rows]);b.bar(ix+offset,vals,width=.30,label=label,color=color,edgecolor='black',lw=.4)
    for xx,v in zip(ix+offset,vals):b.text(xx,v+.025,f'{v:.3f}',ha='center',fontsize=8.5)
b.set(xticks=ix,xticklabels=['Inner-selected K','Fixed K62'],ylim=(0,1.09),ylabel='Within-pipeline estimate');b.set_title('(b) Saved-refit stability',loc='left');b.legend(frameon=False,fontsize=8,loc='lower left')
for method,label,color,marker in [('alpha_only/no_gap','Absorption ranking',BLUE,'o'),('true_n_context/no_gap','n+α context; α output',ORANGE,'s')]:
    p=fixed[fixed.method.eq(method)].sort_values('context');c.plot(p.context,p.Jaccard,color=color,marker=marker,ms=5,lw=1,label=label)
c.set(xticks=[0,1,2],xticklabels=['Context 0','Context 1','Context 2'],ylim=(0,1.03),ylabel='Mean Jaccard');c.set_title('(c) Fixed training, 30 MC seeds',loc='left');c.legend(frameon=False,fontsize=8,loc='lower left');c.grid(color='#E2E2E2',lw=.5)
fig.text(.5,.045,'The cross-pipeline K62 overlap of 0.0111 measures a different question.',ha='center',fontsize=9)
save(fig,'Figure_4')

mapping={'枣仁':'Zaoren','酸枣仁':'Suanzaoren','黄芪':'Huangqi','天麻':'Tianma'}
fig,axs=plt.subplots(1,2,figsize=(7.2,4.0));fig.subplots_adjust(left=.13,right=.975,bottom=.22,top=.86,wspace=.50)
for ax,task,title in zip(axs,['binary_kinds','four_kinds'],['(a) Two materials: 98.84%','(b) Four materials: 65.61%']):
    data=json.loads((ROOT/'paper/current/results/kind_lowo_20261115'/task/'summary.json').read_text())['metrics'];counts=np.array(data['confusion_counts']);frac=counts/counts.sum(1,keepdims=True);labels=[mapping[v] for v in data['class_order']]
    # Draw the small matrix as vector cells instead of embedding a large raster
    # in EPS. Counts, labels and the external frame remain vector objects.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=plt.get_cmap('Blues')(frac[i,j]),edgecolor='none'))
    ax.set_xlim(-.5,len(labels)-.5);ax.set_ylim(len(labels)-.5,-.5);ax.set_aspect('equal')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j,i,f'{counts[i,j]:,}\n{100*frac[i,j]:.1f}%',ha='center',va='center',fontsize=9,color='white' if frac[i,j]>.57 else 'black')
    ax.set(xticks=np.arange(len(labels)),yticks=np.arange(len(labels)),xticklabels=labels,yticklabels=labels,xlabel='Predicted material',ylabel='Recorded material');ax.tick_params(axis='x',rotation=35);ax.set_title(title,loc='left',pad=10)
save(fig,'Figure_5')

support=pd.read_excel(SOURCE,sheet_name='Support_controls');p=support.accuracy_percent.to_numpy();labels=['Source only','Valid labels: full refit','Valid labels: frozen transforms','Unlabelled statistics','Cyclically incorrect labels','Valid labels: normalized loss']
fig,ax=plt.subplots(figsize=(7.2,3.8));fig.subplots_adjust(left=.37,right=.95,bottom=.21,top=.91)
y=np.arange(6);ax.scatter(p,y,s=38,color=[GRAY,BLUE,BLUE,GRAY,ORANGE,BLUE],edgecolors='black',linewidths=.5,zorder=3);ax.axvline(80,color='#555555',ls='--',lw=.9)
for yy,v in zip(y,p):ax.text(v+.25,yy,f'{v:.2f}%',va='center',fontsize=9.5)
ax.set(yticks=y,yticklabels=labels,xlim=(71.5,87),xlabel='Query accuracy (%)');ax.invert_yaxis();ax.grid(axis='x',color='#E2E2E2',lw=.5)
fig.text(.5,.055,'15 contexts; Source 190 / Support 50 / Query 60; ten labels per class where used.',ha='center',fontsize=9)
save(fig,'Figure_6')

budget=pd.read_excel(SOURCE,sheet_name='Fixed_feature_budgets');budget.to_csv(BASE/'analysis/fixed_feature_budgets.csv',index=False)
fig,ax=plt.subplots(figsize=(7.2,3.6));fig.subplots_adjust(left=.11,right=.96,bottom=.27,top=.91)
for family,label,color,marker in [('alpha_snv20','SNV α MC20',BLUE,'o'),('alpha_raw20','Raw α MC20',ORANGE,'s'),('concat_both_snv20','Joint SNV MC20','#009E73','^')]:
    p=budget[budget.family.eq(family)].sort_values('k');assert len(p)==5;ax.plot(p.k,p.accuracy*100,lw=1.2,marker=marker,color=color,label=label)
ax.set(xticks=[31,62,123,184,245],xlabel='Prescribed total predictor count, K',ylabel='Mean accuracy (%)',ylim=(65,100));ax.grid(color='#E2E2E2',lw=.5);ax.legend(loc='lower right',frameon=False,fontsize=9)
fig.text(.5,.065,'Sensitivity analysis; primary performance uses inner-selected K.',ha='center',fontsize=9)
save(fig,'Figure_S1')
(OUT/'FIGURE_VERIFICATION.json').write_text(json.dumps({'status':'FILES_GENERATED','method':'Matplotlib; recorded numerical tables and verified saved predictions; no generative image synthesis','outer_frame':True,'production_format':'EPS vector','embedded_png_dpi':600,'outputs':outputs},ensure_ascii=False,indent=2)+'\n')
print('Generated six main figures and one supplementary figure as PNG, EPS and SVG.')
