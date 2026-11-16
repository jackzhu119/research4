from pathlib import Path
from copy import deepcopy
import json,re,subprocess,tempfile,hashlib
import pandas as pd
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH,WD_TAB_ALIGNMENT,WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement,parse_xml
from docx.oxml.ns import qn

from manuscript_content import TITLE,TITLE_ZH,KEYWORDS,KEYWORDS_ZH,BLOCKS,FIGURE_CAPTIONS,EQUATIONS
from supplement_content import BLOCKS as SI_BLOCKS,OPTICAL_EQUATIONS

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'paper/revision_20261116';SRC=Path(__file__).parent
AUTH=json.loads((SRC/'AUTHOR_METADATA_LOCK.json').read_text());REFS=json.loads((SRC/'references.json').read_text())
OLD=ROOT/'paper/revision_20261115/Supplementary_Tables.xlsx';CURRENT=ROOT/'paper/current'
FIG=BASE/'figures';AN=BASE/'analysis'
CAPTURES=[]
METHODS=pd.read_csv(CURRENT/'methods/dataset_by_species.csv').set_index('species')
MODELS=pd.read_csv(AN/'primary_models.csv').set_index('family')
SUPPORT=pd.read_excel(OLD,sheet_name='Support_controls')
FIXED=pd.read_csv(CURRENT/'tables/fixed_classification.csv');FIXED=FIXED[(FIXED.model=='full_svm')&(FIXED.method=='source_only')].set_index('task')
NAMES={'酸枣仁':'Suanzaoren','枣仁':'Zaoren','黄芪':'Huangqi','天麻':'Tianma'}
FAMILY_NAMES={
 'alpha_snv20':('SNV α MC20–SVM','SNV α MC20–SVM'),
 'alpha_raw20':('Raw α MC20–SVM','原始α MC20–SVM'),
 'alpha_snv100':('SNV α MC100–SVM','SNV α MC100–SVM'),
 'alpha_snv_stability5x20':('SNV α, five MC20 streams','SNV α，五条MC20流'),
 'fullalpha_shrinkLDA':('Full α shrinkage LDA','全α收缩LDA'),
 'fullalpha_snv_svm_bridge':('Full SNV α SVM, same IDs','全SNV α SVM，相同ID'),
 'concat_both_snv20':('Joint SNV n+α MC20','联合SNV n+α MC20'),
 'weighted_concat_nraw_asnv20':('Weighted raw n + SNV α','加权原始n + SNV α'),
 'block_half_nraw_asnv20':('Equal-block raw n + SNV α','等块原始n + SNV α'),
 'concat_nraw_asnv20':('Joint raw n + SNV α MC20','联合原始n + SNV α MC20'),
 'n_raw20':('Raw n MC20–SVM','原始n MC20–SVM'),
 'fullalpha_MLP':('Full α MLP reference','全α MLP参照'),
 'n_snv20':('SNV n MC20–SVM','SNV n MC20–SVM'),
}
ORDER=['alpha_snv20','fullalpha_snv_svm_bridge','fullalpha_shrinkLDA','alpha_raw20','alpha_snv100','alpha_snv_stability5x20','concat_both_snv20','weighted_concat_nraw_asnv20','block_half_nraw_asnv20','concat_nraw_asnv20','n_raw20','fullalpha_MLP','n_snv20']
CITE=re.compile(r'\[((?:ref\d+|cawley|ambroise|vabalas)(?:,\s*(?:ref\d+|cawley|ambroise|vabalas))*)\]')
REF_ORDER=[]
for block in BLOCKS:
    for match in CITE.finditer(block['en']):
        for key in (v.strip() for v in match[1].split(',')):
            if key not in REF_ORDER:REF_ORDER.append(key)
assert set(REF_ORDER)==set(REFS) and len(REF_ORDER)==28
REF_NUMBER={key:i+1 for i,key in enumerate(REF_ORDER)}

ALL_EQUATIONS={**EQUATIONS,**OPTICAL_EQUATIONS}
with tempfile.TemporaryDirectory(prefix='research4_math_') as td:
    p=Path(td);(p/'equations.md').write_text('\n\n'.join('$$\n'+v+'\n$$' for v in ALL_EQUATIONS.values()))
    subprocess.run(['pandoc',str(p/'equations.md'),'-o',str(p/'equations.docx')],check=True)
    ed=Document(p/'equations.docx');nodes=ed.element.findall('.//'+qn('m:oMathPara'))
    assert len(nodes)==len(ALL_EQUATIONS)
    MATH={key:deepcopy(node.find(qn('m:oMath'))) for key,node in zip(ALL_EQUATIONS,nodes)}

def run(p,text,bold=None,size=None,italic=None):
    r=p.add_run(text);r.bold=bold;r.italic=italic;r.font.name='Times New Roman';r.font.color.rgb=RGBColor(0,0,0)
    rp=r._r.get_or_add_rPr();rp.rFonts.set(qn('w:eastAsia'),'Noto Serif CJK SC')
    if size:r.font.size=Pt(size)
    return r

def tidy(text):
    protected_tokens=[]
    def protect_token(match):
        marker='ZZPROTECTEDTOKEN'+chr(65+len(protected_tokens))+'ZZ'
        protected_tokens.append((marker,match[0]));return marker
    text=re.sub(r'https?://\S+|(?:doi:)?10\.\d{4,9}/[-._;()/:A-Za-z0-9]+|\be\d{6,8}\b',protect_token,text)
    text=re.sub(r'macro-F1(?=0\.)','macro-F1 = ',text)
    text=re.sub(r'Φ(?=0\.)','Φ = ',text)
    keep={'MC','PLS','B','E','S','V','K','F','L','macro-F','TRAIN','CAL','PRED','ref','ZY','HQ','ZR','TM','XQM-','DHG','FA'}
    text=re.sub(r'\b([A-Za-z][A-Za-z-]*)(\d+(?:\.\d+)?)',lambda m:m[0] if m[1] in keep else m[1]+' '+m[2],text)
    text=re.sub(r'(?<=%),(?=\S)',', ',text)
    text=re.sub(r',(?=[A-Za-zαΦ])',', ',text)
    for marker,value in protected_tokens:text=text.replace(marker,value)
    return text

def paragraph(d,text,style=None,size=None,tidy_text=True):
    text=tidy(text) if tidy_text else text;p=d.add_paragraph(style=style);pos=0
    for match in CITE.finditer(text):
        run(p,text[pos:match.start()],size=size)
        nums=[str(REF_NUMBER[k.strip()]) for k in match[1].split(',')];r=run(p,','.join(nums),size=size);r.font.superscript=True
        pos=match.end()
    run(p,text[pos:],size=size)
    return p

def setup(kind='main'):
    d=Document();s=d.sections[0];s.page_width=Inches(8.5);s.page_height=Inches(11)
    s.top_margin=s.bottom_margin=s.left_margin=s.right_margin=Inches(1)
    s.header_distance=s.footer_distance=Inches(.40)
    for name in ['Normal','Title','Heading 1','Heading 2','Heading 3','Caption']:
        st=d.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
        st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Serif CJK SC')
    normal=d.styles['Normal'];normal.font.size=Pt(12 if kind=='main' else 11)
    normal.paragraph_format.line_spacing=2 if kind=='main' else 1.16
    normal.paragraph_format.space_after=Pt(4 if kind=='main' else 6)
    normal.paragraph_format.widow_control=True
    for name,size in [('Title',16),('Heading 1',13),('Heading 2',12),('Heading 3',11)]:
        st=d.styles[name];st.font.size=Pt(size);st.font.bold=True;st.paragraph_format.keep_with_next=True
        st.paragraph_format.space_before=Pt(10);st.paragraph_format.space_after=Pt(6);st.paragraph_format.line_spacing=1.15
    for root in [d.element,d.styles.element]:
        for node in list(root.iter(qn('w:pBdr'))):node.getparent().remove(node)
    if kind=='main':
        line=OxmlElement('w:lnNumType');line.set(qn('w:countBy'),'1');line.set(qn('w:restart'),'continuous');line.set(qn('w:distance'),'140');s._sectPr.append(line)
    f=s.footer.paragraphs[0];f.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');f._p.append(field)
    d.core_properties.title=TITLE;d.core_properties.author=AUTH['core_author'];d.core_properties.last_modified_by=AUTH['core_last_modified_by'];d.core_properties.comments=''
    return d

def protected(d,entries):
    for item in entries:d.element.body.insert(-1,deepcopy(parse_xml(item['xml'])))

def caption(d,text):
    p=paragraph(d,text);p.paragraph_format.keep_with_next=True;p.paragraph_format.line_spacing=1.05
    for r in p.runs:r.bold=True;r.font.size=Pt(10.5)
    return p
def note(d,text):
    p=paragraph(d,text,size=9.5);p.paragraph_format.line_spacing=1.05;return p

def table(d,headers,rows,widths=None):
    t=d.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    widths=widths or [6.5/len(headers)]*len(headers);widths=[6.5*w/sum(widths) for w in widths]
    for c,w in zip(t.columns,widths):c.width=Inches(w)
    for ri,(row,vals) in enumerate([(t.rows[0],headers)]+[(t.add_row(),row) for row in rows]):
        tr=row._tr.get_or_add_trPr();tr.append(OxmlElement('w:cantSplit'))
        if ri==0:tr.append(OxmlElement('w:tblHeader'))
        for i,(cell,value) in enumerate(zip(row.cells,vals)):
            cell.width=Inches(widths[i]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p=cell.paragraphs[0];p.paragraph_format.line_spacing=1.05;p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(3)
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT if i==0 or len(str(value))>24 else WD_ALIGN_PARAGRAPH.CENTER
            run(p,str(value),bold=ri==0,size=9.3 if len(headers)>6 else 10)
            borders=OxmlElement('w:tcBorders')
            for edge in ['top','bottom','left','right']:
                b=OxmlElement('w:'+edge);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'4');b.set(qn('w:color'),'D9D9D9');borders.append(b)
            cell._tc.get_or_add_tcPr().append(borders)
            if ri==0:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'F0F0F0');cell._tc.get_or_add_tcPr().append(shade)
    d.add_paragraph().paragraph_format.space_after=Pt(1)
    return t

def equation(d,key,label):
    p=d.add_paragraph();p.paragraph_format.keep_together=True
    p.paragraph_format.line_spacing_rule=WD_LINE_SPACING.AT_LEAST
    p.paragraph_format.line_spacing=Pt(66 if key in {'moments','score','kernel','stability'} else 44)
    p.paragraph_format.space_before=Pt(6);p.paragraph_format.space_after=Pt(9)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(3.05),WD_TAB_ALIGNMENT.CENTER)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(6.45),WD_TAB_ALIGNMENT.RIGHT)
    run(p,'\t');p._p.append(deepcopy(MATH[key]));run(p,'\t('+label+')')
    return p

def figure(d,key,bilingual=False):
    p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True;p.paragraph_format.line_spacing=1
    p.add_run().add_picture(str(FIG/f'Figure_{key}.png'),width=Inches(6.45))
    if key=='S1':
        en='Prescribed-budget sensitivity on the original V4 partitions. Each point is the mean of three pooled fivefold evaluations of the same300 Suanzaoren TRAIN specimens. SVM settings are inner-selected at each fixed K; K counts total retained coordinates. These curves do not select the primary budget, which remains inner-selected.'
        zh='原V4划分上的预先指定预算敏感性。每点是相同300份酸枣仁TRAIN样品三次汇总五折评价的均值。各固定K下均在内层选择SVM参数，K为最终保留坐标总数。这些曲线不用于选择主预算；主预算仍由内层选择。'
    else:en,zh=FIGURE_CAPTIONS[key]
    q=paragraph(d,f'Figure {key}. '+en,size=10);q.paragraph_format.line_spacing=1.12
    if bilingual:
        q.paragraph_format.keep_with_next=True;q=paragraph(d,f'图{key}。'+zh,size=10);q.paragraph_format.line_spacing=1.12

def main_table_data(key,zh=False):
    if key=='dataset':
        headers=['Material','Workbooks','Total','TRAIN','CAL','PRED'] if not zh else ['材料','工作簿','合计','TRAIN','CAL','PRED']
        rows=[[s if zh else NAMES[s],*[str(int(METHODS.loc[s,c])) for c in ['workbooks','records','train','calibration','prediction']]] for s in ['酸枣仁','枣仁','黄芪','天麻']]
        rows.append(['合计' if zh else 'Total',18,1672,999,334,339])
        return ('Table I  Material collection and frozen specimen roles' if not zh else '表I  材料集合与冻结样品角色',headers,rows,[1.75,.85,.75,.8,.8,.8],
                'Each specimen is counted once across its linked workbook views. The principal nested analysis uses300 Suanzaoren TRAIN records,60 per source.' if not zh else '每份样品的关联工作表视图仅计数一次。主要嵌套分析采用300份酸枣仁TRAIN记录，每来源60份。')
    if key in ['primary','all_primary']:
        rows=[[FAMILY_NAMES[k][int(zh)],f'{MODELS.loc[k,"mean_K"]:.1f}',f'{100*MODELS.loc[k,"accuracy"]:.2f}',f'{MODELS.loc[k,"macro_f1"]:.4f}'] for k in ORDER]
        return ('Table II  Primary nested representation and matched full-spectrum comparison' if not zh else '表II  主要嵌套表示与匹配全谱比较',
                ['Pipeline','Mean K','Accuracy %','Macro-F1'] if not zh else ['流程','平均K','准确率%','宏F1'],rows,[3.65,.85,1,1],
                'All rows evaluate the same300 TRAIN specimens and three fivefold partitions. Values average the three pooled estimates; balanced accuracy equals accuracy in these balanced pools. K and eligible settings are inner-selected. The full-spectrum SVM is E1; the other rows retain their original V4 recipes.' if not zh else '各行评价相同300份TRAIN样品及三次五折划分。数值为三次汇总指标的均值；这些平衡集合的平衡准确率等于准确率。K及可选择设置在内层确定。全谱SVM为E1，其余行保留原V4配方。')
    if key=='materials':
        labels={'binary_kinds':('Suanzaoren–Zaoren','酸枣仁–枣仁'),'four_kinds':('Four materials','四材料'),'acid_origins':('Suanzaoren sources','酸枣仁来源'),'jujube_origins':('Zaoren sources','枣仁来源'),'huangqi_origins':('Huangqi sources','黄芪来源'),'tianma_origins':('Tianma sources','天麻来源')}
        rows=[]
        for task in ['binary_kinds','four_kinds']:
            f=FIXED.loc[task];n=int(f.prediction_count)
            rows.append([labels[task][int(zh)],'固定PRED' if zh else 'Fixed PRED',f'{round(f.accuracy_mean*n)}/{n}',f'{100*f.accuracy_mean:.2f}',f'{100*f.balanced_accuracy_mean:.2f}',f'{f.macro_f1_mean:.4f}'])
            s=json.loads((CURRENT/'results/kind_lowo_20261115'/task/'summary.json').read_text())['metrics']
            rows.append([labels[task][int(zh)],'LOWO',f'{s["n_correct"]}/{s["n_records"]}',f'{100*s["accuracy"]:.2f}',f'{100*s["balanced_accuracy"]:.2f}',f'{s["macro_f1"]:.4f}'])
        for task in ['acid_origins','jujube_origins','huangqi_origins','tianma_origins']:
            f=FIXED.loc[task];n=int(f.prediction_count);rows.append([labels[task][int(zh)],'固定PRED' if zh else 'Fixed PRED',f'{round(f.accuracy_mean*n)}/{n}',f'{100*f.accuracy_mean:.2f}',f'{100*f.balanced_accuracy_mean:.2f}',f'{f.macro_f1_mean:.4f}'])
        return ('Table III  Protocol-specific material and source classification' if not zh else '表III  按协议区分的材料与来源分类',
                ['Task','Evaluation','Correct / n','Accuracy %','BA %','Macro-F1'] if not zh else ['任务','评价','正确数/n','准确率%','BA%','宏F1'],rows,[1.55,1,.85,.85,.75,.9],
                'The fixed full-spectrum SVM uses612 SNV absorption coordinates,C=10,gamma=scale and training-fitted scaling. LOWO tests all864 or1672 eligible specimens once; it does not use MC20. Fixed PRED rows retain original roles with CAL unused. These protocols overlap in specimens but have different training compositions and evaluation units.' if not zh else '固定全谱SVM采用612个SNV吸收坐标、C=10、gamma=scale及训练集拟合的缩放。LOWO对全部864或1672份合格样品各测试一次，不使用MC20。固定PRED保留原角色，CAL未用于这些行。各协议样品重叠，但训练组成和评价单位不同。')
    if key=='support':
        labels=['Source only','Valid labels, full pipeline refit','Valid labels, frozen scaler and gamma','Unlabelled-statistics update','Cyclically incorrect labels, refit','Valid labels, normalized loss mass']
        zlabels=['仅源数据','正确标签，完整流程重新拟合','正确标签，冻结缩放与gamma','无标签统计量更新','循环错误标签，重新拟合','正确标签，损失质量归一化']
        rows=[[(zlabels if zh else labels)[i],f'{100*r.accuracy:.4f}',f'{r.delta_accuracy_pp:+.4f}'] for i,r in SUPPORT.iterrows()]
        return ('Table IV  Full-spectrum support controls at ten labels per class' if not zh else '表IV  每类十个标签的全谱支持对照',
                ['Intervention','Accuracy %','Difference, pp'] if not zh else ['干预','准确率%','差值，pp'],rows,[4.1,1.2,1.2],
                'Fifteen Source190/Support50/Query60 contexts use the same300 TRAIN specimens and yield900 related Query predictions. Ten labels per class means50 labels in total. These gains do not add to the selected-feature score.' if not zh else '15个Source190/Support50/Query60情境使用相同300份TRAIN样品，产生900条相关Query预测。每类十个标签合计50个，收益不叠加至所选特征分数。')
    raise KeyError(key)

def add_main_table(d,key,bilingual=False):
    for zh in ([False,True] if bilingual else [False]):
        title,headers,rows,widths,foot=main_table_data(key,zh)
        caption(d,title);table(d,headers,rows,widths);note(d,foot)

ORIGINS={'ZY01':'Yan’an, Shaanxi','ZY02':'Yunnan','ZY03':'Inner Mongolia','ZY04':'Lingshi, Shanxi','ZY05':'Bozhou, Anhui','ZY06':'Shijiazhuang, Hebei','ZY07':'Shanxi group','HQ01':'Qujing','HQ02':'Guyang, Inner Mongolia','HQ03':'Gansu','HQ04':'Datong, Shanxi','ZR01':'Yulin, Shaanxi','ZR02':'Liaoning','ZR03':'Inner Mongolia','TM01':'Leishan, Guizhou','TM02':'Changbai Mountain, Jilin','TM03':'Anhui, recorded label','TM04':'Yunnan'}
def supplementary_table_data(key):
    if key=='workbooks':
        x=pd.read_csv(CURRENT/'methods/dataset_by_workbook.csv')
        return 'Workbook groups and frozen roles',['Group','Material','Recorded source','n','TRAIN','CAL','PRED'],[[r.origin_group_id,NAMES[r.species],ORIGINS[r.origin_group_id],r.records,r.train,r.calibration,r.prediction] for r in x.itertuples()],[.65,.95,2.1,.5,.65,.6,.6]
    if key=='interface':
        return 'Reusable stages and model-specific choices',['Stage','Inputs and output','Relation to final classifier'],[
          ['Fit preprocessing','Fitting X → fitted transforms','Fit only inside the corresponding training subset'],
          ['Classwise MC ranking','Fitting X,y → class coefficients and aggregate scores','Ranking does not call the final classifier'],
          ['Coordinate projection','Scores and K → retained coordinate indices','Projection accepts a specified K; its optimal K is model-specific'],
          ['Budget and parameter choice','Inner validation → K and estimator settings','Retune inside each downstream model’s inner validation'],
          ['Final prediction','Fitted transforms, coordinates and estimator','SVM is evaluated here; other-model gains need direct comparison']], [1.4,2.5,2.6]
    if key=='all_primary':
        _,headers,rows,widths,_=main_table_data('primary');return 'Complete primary comparison',headers,rows,widths
    if key=='mlp_diagnostics':
        x=pd.read_csv(CURRENT/'results/focused_followup_20261116/e6/checkpoint_diagnostics.csv')
        rows=[[r.context.replace('matched_',''),r.preprocessing,str(r.hidden_layer_sizes).strip('[]'),int(r.actual_epochs),f'{100*r.best_validation_accuracy:.2f}',int(r.prediction_mismatches)] for r in x.itertuples()]
        return 'Saved MLP checkpoint diagnostics',['Outer context','Preprocess','Hidden','Epochs','Best ES accuracy %','Replay mismatches'],rows,[1.45,1,.65,.7,1.2,1.1]
    if key=='e1_parameters':
        x=pd.read_csv(CURRENT/'results/focused_followup_20261116/e1/outer_parameters.csv')
        return 'Matched full-spectrum SVM outer parameters',['Seed','Fold','C','Kernel factor','Numeric gamma','Fits'],[[int(r.repeat_seed),int(r.outer_fold),f'{r.C:g}',r.gamma,f'{r.numeric_gamma:.10g}',int(r.fits)] for r in x.itertuples()],[1,.6,.6,1.15,1.35,.65]
    if key=='budget_core':
        x=pd.read_csv(AN/'fixed_feature_budgets.csv');families=['alpha_raw20','alpha_snv20','concat_both_snv20']
        return 'Prescribed feature-budget comparison',['K','Raw α MC20 %','SNV α MC20 %','Joint SNV MC20 %'],[[k,*[f'{100*x[(x.family==f)&(x.k==k)].iloc[0].accuracy:.2f}' for f in families]] for k in [31,62,123,184,245]],[.6,1.8,1.8,1.8]
    if key=='stability':
        x=pd.read_csv(AN/'v4_stability.csv');rows=[]
        for family,budget in [('alpha_snv20','primary'),('alpha_snv20','62'),('alpha_raw20','primary'),('concat_both_snv20','primary')]:
            r=x[(x.family==family)&x.budget.eq(budget)].iloc[0]
            rows.append([FAMILY_NAMES[family][0],budget,int(r.universe_p),f'{r.mean_K:.1f}',f'{r.Jaccard:.4f}',f'{r.Nogueira_Phi:.4f}'])
        return 'Within-pipeline saved-refit stability',['Family','Budget','Universe','Mean K','Jaccard','Nogueira Φ'],rows,[2.0,.75,.8,.8,.8,.85]
    if key=='mc_stability':
        x=pd.read_csv(AN/'fixed_train_stability.csv');rows=[]
        for r in x.itertuples():
            label=r.method.replace('alpha_only','α only').replace('true_n_context','n+α context').replace('no_gap','plain K62').replace('fixed_gap4','gap4')
            rows.append([r.context,label,int(r.mask_count),f'{r.Jaccard:.4f}',f'{r.Nogueira_Phi:.4f}'])
        return 'Fixed training IDs with thirty MC seeds',['Context','Rank setting','Masks','Jaccard','Nogueira Φ'],rows,[.65,2.6,.7,1.05,1.2]
    if key=='crossed':
        x=pd.read_csv(AN/'crossed_masks.csv');rows=[[int(r.context),f'{100*r.AA:.2f}',f'{100*r.BA:.2f}',f'{r.mask_effect_pp:+.2f}',f'{r.C_A:g}',f'{r.gamma_A:.8g}'] for r in x.itertuples()]
        return 'Conditional mask exchange at effectively common parameters',['Context','Mask A %','Mask B %','Change pp','Shared C','Shared gamma'],rows,[.7,.9,.9,.9,.9,1.3]
    if key=='paired_effects':
        x=pd.read_csv(AN/'paired_primary_effects.csv');rows=[[FAMILY_NAMES[r.comparator][0],f'{r.difference_pp:+.2f}',f'[{r.ci_low_pp:.2f}, {r.ci_high_pp:.2f}]'] for r in x.itertuples()]
        return 'MC20-minus-comparator conditional accuracy intervals',['Comparator','Difference pp','95% interval pp'],rows,[3.6,1.2,1.7]
    if key=='lowo_folds':
        rows=[]
        for task in ['binary_kinds','four_kinds']:
            for r in pd.read_csv(CURRENT/'results/kind_lowo_20261115'/task/'fold_metrics.csv').itertuples():
                rows.append(['Two' if task=='binary_kinds' else 'Four',r.heldout_group,NAMES[r.true_kind],f'{r.n_correct}/{r.n_test}',f'{100*r.accuracy:.2f}'])
        return 'All workbook-held-out folds',['Task','Held-out group','Recorded material','Correct / n','Recall %'],rows,[.6,1.05,1.8,1.15,1]
    if key=='support':
        _,headers,rows,widths,_=main_table_data('support');return 'Full-spectrum support-label controls',headers,rows,widths
    raise KeyError(key)

END_ZH={
 'Acknowledgements':'致谢',
 AUTH['end_matter'][1]['text']:'第一作者感谢导师及通讯作者Yin Shen提供实验数据，并在论文撰写和投稿方面给予指导与支持。',
 AUTH['ai_disclosure']:'OpenAI ChatGPT/Codex辅助论文修改、文献检索，以及分析和绘图代码的准备与检查。作者审阅了输出，并对内容承担全部责任。',
 'Author contributions':'作者贡献',
 AUTH['author_contributions']:'Ziyuan Zhu负责实验设计、算法优化和论文写作。Jiayi Ren协助检查实验结果及向GitHub仓库推送更新。Shuyu Zhou查询投稿政策并协助整理补充文件。Mengting Zhu检查论文语法并核对最终投稿表单信息。Yin Shen提供实验数据，并指导论文准备和投稿。所有作者均审阅并批准稿件及其向Applied Spectroscopy投稿。',
 'Statements and declarations':'声明', 'Ethical considerations':'伦理事项',
 'This study analyzed herbal-material spectral records and involved neither human participants nor animals. Ethical approval was not required.':'本研究分析药材光谱记录，不涉及人体参与者或动物，无需伦理审批。',
 'Consent to participate':'参与同意','Not applicable.':'不适用。','Consent for publication':'发表同意',
 'Declaration of conflicting interest':'利益冲突声明',
 'The authors declared no potential conflicts of interest with respect to the research, authorship, and/or publication of this article.':'作者声明，本研究、论文署名及发表不存在潜在利益冲突。',
 'Funding statement':'资金声明',
 'The authors received no financial support for the research, authorship or publication of this article.':'本研究、论文撰写和发表未获得资金支持。',
 'Author approval':'作者批准',
 'All authors have read and approved the manuscript and its submission to Applied Spectroscopy.':'所有作者均已阅读并批准稿件及其向Applied Spectroscopy投稿。',
 'All authors accept responsibility for the accuracy and integrity of the work. This manuscript is not under consideration elsewhere.':'所有作者对工作的准确性和完整性承担责任。稿件未同时在其他期刊审理。',
}
DATA_AVAIL='The spectra, recorded material and source labels, specimen allocations, classifier results and analysis protocols supporting this study are available at https://github.com/jackzhu119/research4 and in the accompanying supplementary data. The experimental dataset was supplied by the corresponding author, Yin Shen, as a shared research resource also used in other supervised studies. The input registry and evidence manifest identify the files and protocol-specific analyses used here.'
DATA_AVAIL_ZH='支撑本研究的光谱、记录材料与来源标签、样品分配、分类器结果及分析协议，可从https://github.com/jackzhu119/research4和配套补充数据获得。通讯作者Yin Shen提供实验数据，其属于亦用于其他指导研究的共享资源。输入登记和证据清单明确了本文采用的文件与各协议分析。'

def make_main(bilingual=False):
    d=setup('parallel' if bilingual else 'main');paragraph(d,TITLE,'Title')
    if bilingual:paragraph(d,TITLE_ZH,'Title')
    protected(d,AUTH['header'])
    paragraph(d,'Abstract','Heading 1')
    if bilingual:paragraph(d,'中英文逐段对照','Heading 2')
    for block in BLOCKS:
        kind=block['kind']
        if kind in ['h1','h2']:
            paragraph(d,block['en'],'Heading 1' if kind=='h1' else 'Heading 2')
            if bilingual:paragraph(d,block['zh'],'Heading 2' if kind=='h1' else 'Heading 3')
        elif kind in ['abstract','p']:
            paragraph(d,block['en'])
            if bilingual:paragraph(d,block['zh'])
            if kind=='abstract':
                p=paragraph(d,'Keywords: '+KEYWORDS,size=10.5);p.paragraph_format.line_spacing=1.1
                if bilingual:paragraph(d,'关键词：'+KEYWORDS_ZH,size=10.5)
        elif kind=='equation':equation(d,block['key'],str(list(EQUATIONS).index(block['key'])+1))
        elif kind=='figure':figure(d,block['key'],bilingual)
        elif kind=='table':add_main_table(d,block['key'],bilingual)
    for item in AUTH['end_matter']:
        protected(d,[item])
        if bilingual:
            assert item['text'] in END_ZH,item['text'];paragraph(d,END_ZH[item['text']],item['style'] if item['style'].startswith('Heading') else None)
    paragraph(d,'Data availability','Heading 2');paragraph(d,DATA_AVAIL)
    if bilingual:paragraph(d,'数据可获得性','Heading 3');paragraph(d,DATA_AVAIL_ZH)
    paragraph(d,'Supplemental material','Heading 1');paragraph(d,'Supplemental Material provides specimen allocation, optical definitions, complete model and feature-budget comparisons, stability analyses, workbook-held-out outcomes, support-label controls and neural reference diagnostics. The accompanying workbooks preserve the corrected positive and negative results and source provenance.')
    if bilingual:paragraph(d,'补充材料提供样品分配、光学定义、完整模型与特征预算比较、稳定性分析、工作簿留出结果、支持标签对照及神经网络参照诊断；配套工作簿保留经过标签纠正的正负结果及来源链。')
    paragraph(d,'References','Heading 1')
    for key in REF_ORDER:
        p=paragraph(d,str(REF_NUMBER[key])+'. '+REFS[key],size=10.5,tidy_text=False);p.paragraph_format.line_spacing=1.15;p.paragraph_format.space_after=Pt(6)
    filename='06_Manuscript_EN_ZH.docx' if bilingual else '01_Manuscript.docx'
    d.save(BASE/filename)
    return d

def make_supplement():
    d=setup('supplement');paragraph(d,'Supplementary information','Title');paragraph(d,TITLE,'Heading 1');protected(d,AUTH['header']);n=0
    for block in SI_BLOCKS:
        kind=block['kind']
        if kind in ['h1','h2']:paragraph(d,block['en'],'Heading 1' if kind=='h1' else 'Heading 2')
        elif kind=='p':paragraph(d,block['en'])
        elif kind=='equation':equation(d,block['key'],'S'+str(list(OPTICAL_EQUATIONS).index(block['key'])+1))
        elif kind=='figure':figure(d,block['key'])
        elif kind=='table':
            n+=1;title,headers,rows,widths=supplementary_table_data(block['key']);caption(d,f'Table S{n}  '+title);table(d,headers,rows,widths)
    d.save(BASE/'04_Supplementary_Information.docx')
    return d

def replace_paragraph(p,text):
    p.clear();run(p,text)

def make_auxiliary():
    for template in ['02_Title_Page.docx','Declarations.docx']:
        d=Document(SRC/'templates'/template)
        index=0 if template.startswith('02') else 1
        replace_paragraph(d.paragraphs[index],TITLE if index==0 else 'Manuscript: '+TITLE)
        if template.startswith('02'):
            for p in d.paragraphs:
                if p.text=='Five main figures and one supplementary figure':replace_paragraph(p,'Six main figures and one supplementary figure')
        for root in [d.element,d.styles.element]:
            for node in list(root.iter(qn('w:pBdr'))):node.getparent().remove(node)
        d.save(BASE/('02_Title_Page.docx' if template.startswith('02') else '05_Declarations.docx'))
    d=Document(SRC/'templates/03_Cover_Letter.docx')
    updated={
      2:f'Please consider the manuscript “{TITLE}” for publication in Applied Spectroscopy.',
      3:'Routine medicinal-material identification relies on established morphological, microscopic and chemical methods. This study proposes a classification-oriented analysis route for complementary terahertz discrimination. Multiclass coefficient consistency is aggregated into a spectral ranking, and training-contained feature-budget selection produces traceable inputs. The ranking front end is separate from the final classifier, providing a structure that other estimators can reuse with their own validation and tuning.',
      4:'The collection comprises1,672 specimens in18 workbooks. In balanced five-source Suanzaoren analysis, the selected absorption workflow achieved93.44% mean accuracy with63.95% mean reduction in classifier input width. A newly available full-spectrum SVM on exactly the same inner and outer specimen IDs achieved82.67%, a10.78-percentage-point difference. Full-spectrum shrinkage LDA achieved91.78% and is retained as a strong comparator. The tested accuracy benefit is established for the SVM workflow; architectural reusability is not presented as a measured gain across other classifiers.',
      5:'The ablations identify representation-dependent normalization behavior, changes in absorption subsets when the candidate pool changes, and a valid-supervision effect in separate full-spectrum label-update controls. Workbook-held-out evaluation gives98.84% accuracy for Suanzaoren–Zaoren discrimination and65.61% across all four materials, defining task-specific scope for further validation. These observed phenomena provide concrete directions for developing terahertz-assisted recognition of additional medicinal materials.',
      6:'The manuscript is an application and methodological contribution to medicinal-material spectroscopy. Its28 references, six main figures and accompanying corrected positive and negative comparisons connect the analytical motivation to actual evidence. Source identities, validation units and feature-selection repeatability are supplied with the manuscript.',
    }
    for i,text in updated.items():replace_paragraph(d.paragraphs[i],tidy(text))
    for root in [d.element,d.styles.element]:
        for node in list(root.iter(qn('w:pBdr'))):node.getparent().remove(node)
    d.save(BASE/'03_Cover_Letter.docx')

REVISION_ROWS=[
 ('Main contribution / 主贡献','Strengthened / 强化','A medicinal-material discrimination strategy with a reusable ranking front end; measured SVM gain10.78pp and input reduction63.95%. Other-model gains are not claimed.','提出药材辅助识别的分析策略，强调排名前端可复用；SVM实测提高10.78个百分点，输入维度减少63.95%，不声称已验证其他模型收益。'),
 ('Fixed-K abstract result / 固定K摘要主结果','Adopted / 采纳','Primary results now use inner-selected K. K62 remains a sensitivity comparison.','摘要使用内层选择K的主结果，K62保留为敏感性分析。'),
 ('Jaccard0.0111 / 低Jaccard','Inference corrected / 纠正推断','Cross-pipeline overlap does not measure same-pipeline stability. Primary Jaccard0.8110 and Φ0.8360 are recomputed from saved masks.','跨流程重叠不等于同流程稳定性；从保存掩码重算主流程Jaccard0.8110和Φ0.8360。'),
 ('Matched full-spectrum SVM / 同划分全谱SVM','New evidence incorporated / 纳入新证据','E1 preserves all outer and inner IDs;82.67%,375 historical fits. Three actual gamma factors are verified.','E1保留全部外层与内层ID，82.67%、历史拟合375次；三档实际gamma均已核查。'),
 ('Strong LDA comparator / 强LDA对照','Retained / 保留','LDA91.78%; paired MC20-minus-LDA interval[−1.11,4.44]pp includes zero. No universal-superiority claim.','保留LDA91.78%；MC20减LDA区间[−1.11,4.44]个百分点包含零，不作普遍最优主张。'),
 ('Expanded gamma cache / 扩展gamma缓存','Specific defect confirmed / 确认特定缺陷','The expanded worker reuses its first gamma. Its performance and cost claims are excluded from primary evidence; V4/E1 are checked separately.','扩展worker复用首档gamma；相关性能及成本主张不纳入主证据，V4/E1分别核查。'),
 ('Material validation / 材料验证','Main text updated / 更新正文','Both fixed PRED and LOWO are visible. Binary98.84% and four-material65.61% use complete workbooks and different evaluation populations.','正文同时展示固定PRED与LOWO。二类98.84%、四类65.61%，留出完整工作簿，明确不同评价样品范围。'),
 ('MLP73.22% / 神经网络低分','Qualified / 限定解释','Different feature budgets, narrow search and early stopping are documented. Checkpoints replay correctly; low score alone does not establish a bug.','说明特征预算、有限搜索及早停差异；检查点回放一致，低分本身不证明bug。'),
 ('Export unit / 导出单位','Consistent scale / 统一尺度','No coefficient unit is invented. Optical equations move to physical definitions in SI; figures retain the exported scale.','不编造吸收单位；光学公式作为物理定义移至补充材料，图保留导出尺度。'),
 ('Molecular digression / 分子计算旁支','Removed from paper narrative / 移出稿件叙述','The unrelated molecular section and figure are removed from this submission; protected repository calculations remain unchanged.','本投稿移出缺少直接分类证据联系的分子章节与图，仓库受保护计算不改动。'),
 ('Support-label gains / 支持标签收益','Protocol separated / 分开协议','80.00→83.78% belongs to the full-spectrum control; ten labels per class means50 total. It is not added to93.44%.','80.00→83.78%属于全谱对照；每类十个合计50个，不叠加到93.44%。'),
 ('Crossed-mask mechanism / 交叉掩码机制','Qualified / 限定解释','Mask substitution is measured at nearly identical C/gamma; the parameter factor is not separately identifiable.','掩码替换在近乎相同C/gamma下评价，不能独立识别参数因素。'),
 ('Author and AI fields / 作者及AI信息','Locked / 锁定','The author rows, ORCIDs, emails, address, acknowledgements, contributions, approvals and AI disclosure retain the uploaded baseline wording.','作者行、ORCID、邮箱、地址、致谢、贡献、批准及AI披露均保留上传基线原文。'),
]

def make_revision_notes():
    d=setup('notes');paragraph(d,'Revision record and evidence checks','Title');paragraph(d,'修改记录与证据核查','Title')
    paragraph(d,TITLE);paragraph(d,TITLE_ZH)
    paragraph(d,'2026年11月16日。基线为用户上传的research4.rar，审稿材料为Review.zip。作者及AI披露按基线锁定。此次改稿没有新增科学模型训练。')
    paragraph(d,'The revision follows the original motivation: develop terahertz classification as a complementary approach alongside established medicinal-material identification. Its contribution is the proposed analysis route, measured SVM improvement, reusable front-end structure and the phenomena identified through actual ablations.')
    paragraph(d,'本轮把主线落实为药材识别的新分析思路、实测准确率收益、可复用的前端结构及实际消融发现。可复用结构不等于已证明所有模型提高；不能以挑选最高分或拼接不同协议来增强贡献。')
    for concern,decision,en,zh in REVISION_ROWS:
        paragraph(d,concern,'Heading 2');paragraph(d,decision).paragraph_format.keep_with_next=True
        paragraph(d,tidy(en)).paragraph_format.keep_with_next=True;paragraph(d,tidy(zh))
    paragraph(d,'Evidence and verification / 证据与核验','Heading 1')
    paragraph(d,'The focused-followup source is pinned at5594d9c3326fcd607ee72fb7fc93895047de8f8a. The publication view now verifies182 files and18 unchanged original workbooks. All900 E1 predictions,15 original outer/inner partition sets,360 inner numerical gamma values, primary masks,90 fixed-training MC tasks and15 crossed-mask contexts were checked.')
    paragraph(d,'补充支线固定为5594d9c3326fcd607ee72fb7fc93895047de8f8a。当前结果入口核验182个文件及18份未变原谱，复核E1的900条预测、15组原内外层ID、360次内层数值gamma、主要掩码、90个固定训练MC任务及15个交叉掩码情境。')
    paragraph(d,'Paired primary intervals are newly summarized from saved predictions with10,000 source-stratified physical-ID draws, seed20261116. They are conditional and unadjusted, not an independent confirmation set or a refitting experiment. The two CSV hash differences from the uploaded review are exactly explained by CRLF-to-LF normalization; immutable Git bytes remain the source authority.')
    paragraph(d,'主要配对区间由保存预测统一重算：10,000次按来源分层的实物ID抽样，种子20261116，属于条件性且未校正的分析，不是独立确认集合或重新拟合实验。审稿附件中两份CSV的哈希差异均由CRLF/LF换行归一化解释；固定Git原件仍是来源依据。')
    paragraph(d,'Use the journal-upload files for submission and the bilingual manuscript for reading. This record is an author-facing account of edits and file checks, not an editorial acceptance decision or a claim that every optional research branch was completed.')
    paragraph(d,'投稿使用期刊上传文件，中英文对照稿供阅读。本记录说明修改及文件核验，不代表期刊接收决定，也不宣称所有可选研究支线均已完成。')
    d.save(BASE/'07_Revision_Notes_EN_ZH.docx')

if __name__=='__main__':
    main=make_main();parallel=make_main(True);supplement=make_supplement();make_auxiliary();make_revision_notes()
    authoring={'title':TITLE,'title_zh':TITLE_ZH,'abstract':next(b['en'] for b in BLOCKS if b['kind']=='abstract'),
      'abstract_words':len(next(b['en'] for b in BLOCKS if b['kind']=='abstract').split()),'keywords':KEYWORDS.split('; '),
      'reference_order':REF_ORDER,'main_figures':6,'supplementary_figures':1,'main_tables':4,
      'editable_main_equations':len(main.element.findall('.//'+qn('m:oMath'))),
      'main_document_sha256':hashlib.sha256((BASE/'01_Manuscript.docx').read_bytes()).hexdigest(),
      'bilingual_document_sha256':hashlib.sha256((BASE/'06_Manuscript_EN_ZH.docx').read_bytes()).hexdigest(),
      'science_scope':'Current corrected collection; E1 same-ID bridge; distinct LOWO and support protocols; no scientific training performed during revision'}
    (BASE/'DOCUMENT_BUILD.json').write_text(json.dumps(authoring,ensure_ascii=False,indent=2)+'\n')
    (SRC/'resolved_references.json').write_text(json.dumps({str(REF_NUMBER[k]):REFS[k] for k in REF_ORDER},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(authoring,ensure_ascii=False,indent=2))
