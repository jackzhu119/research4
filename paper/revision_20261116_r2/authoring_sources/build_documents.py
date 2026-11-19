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
from publication_metadata import (
    AFFILIATIONS, AUTHORS, AUTHOR_NAMES, AUTHOR_APPROVAL, AUTHOR_APPROVAL_ZH,
    AI_DISCLOSURE, AI_DISCLOSURE_ZH,
    CONFLICTS, CONFLICTS_ZH, CORRESPONDING, DATA_PROVENANCE, DATA_PROVENANCE_ZH, ORIGINS_EN,
)

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'paper/revision_20261116_r2';SRC=Path(__file__).parent
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
    # Shield literal identifiers before applying English prose spacing. Locked
    # author/declaration XML bypasses this function through protected().
    text=re.sub(r'(?<=[A-Za-z])(?=\d{1,3}(?:,\d{3})+\b)', ' ', text)
    protected_tokens=[]
    def protect_token(match):
        marker='ZZPROTECTEDTOKEN'+chr(65+len(protected_tokens))+'ZZ'
        protected_tokens.append((marker,match[0]));return marker
    literals=(r'https?://\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}'
              r'|(?:doi:)?10\.\d{4,9}/[-._;()/:A-Za-z0-9]+'
              r'|\b\d{4}-\d{4}-\d{4}-[\dX]{4}\b'
              r'|\br\d+_f\d+\b|\b[0-9a-f]{40,64}\b|\be\d{6,8}\b'
              r'|(?<![A-Za-z0-9_])(?:ZY|HQ|ZR|TM)\d+(?:[–-](?:ZY|HQ|ZR|TM)?\d+)?(?![A-Za-z0-9_])'
              r'|\b[A-Za-z][A-Za-z0-9]*-\d+[A-Za-z0-9-]*\b'
              r'|(?<![\w,{\[])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])')
    text=re.sub(literals,protect_token,text)
    text=re.sub(r'macro-F1(?=0\.)','macro-F1 = ',text)
    text=re.sub(r'Φ(?=0\.)','Φ = ',text)
    keep={'MC','PLS','B','E','S','V','K','F','L','macro-F','Macro-F','R','T','ZY','HQ','ZR','TM','hist','TRAIN','CAL','PRED','ref','XQM-','DHG','FA'}
    text=re.sub(r'\b([A-Za-z][A-Za-z-]*)(\d+(?:\.\d+)?)',lambda m:m[0] if m[1] in keep else m[1]+' '+m[2],text)
    # Commas in thousands, URLs and identifiers have already been shielded.
    text=re.sub(r'([,;])(?=[A-Za-z0-9αΦ−+])',r'\1 ',text)
    text=re.sub(r'(?<=[\d\]])pp\b',' pp',text)
    text=re.sub(r'(?<=[{\[])\s+|\s+(?=[}\]])','',text)
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
    d.core_properties.title=TITLE;d.core_properties.author='; '.join(AUTHOR_NAMES);d.core_properties.last_modified_by='';d.core_properties.comments=''
    return d

def protected(d,entries):
    for item in entries:d.element.body.insert(-1,deepcopy(parse_xml(item['xml'])))

def author_byline(paragraph):
    paragraph.paragraph_format.line_spacing = 1.15
    for index, author in enumerate(AUTHORS):
        if index:
            run(paragraph, ', ')
        run(paragraph, author['name'])
        marker = str(author['affiliation_id']) + ('*' if author['corresponding'] else '')
        run(paragraph, marker).font.superscript = True


def author_affiliation(paragraph, affiliation):
    paragraph.paragraph_format.line_spacing = 1.15
    run(paragraph, str(affiliation['id'])).font.superscript = True
    run(paragraph, ' ' + affiliation['name'])


def author_header(d):
    author_byline(d.add_paragraph())
    for affiliation in AFFILIATIONS:
        author_affiliation(d.add_paragraph(), affiliation)
    for item in AUTH['header'][2:]:
        protected(d, [item])

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
    rows=list(rows)
    compact=len(rows)<=8 and sum(len(str(v)) for row in rows for v in row)<1600
    for ri,(row,vals) in enumerate([(t.rows[0],headers)]+[(t.add_row(),row) for row in rows]):
        tr=row._tr.get_or_add_trPr();tr.append(OxmlElement('w:cantSplit'))
        if ri==0:tr.append(OxmlElement('w:tblHeader'))
        for i,(cell,value) in enumerate(zip(row.cells,vals)):
            cell.width=Inches(widths[i]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p=cell.paragraphs[0];p.paragraph_format.line_spacing=1.05;p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(3)
            # Compact tables stay together; longer tables repeat headers and
            # keep the first/last data-row pairs together at page boundaries.
            p.paragraph_format.keep_together=True
            p.paragraph_format.keep_with_next=(ri<len(rows) if compact else ri in {0,1,len(rows)-1})
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT if i==0 or len(str(value))>24 else WD_ALIGN_PARAGRAPH.CENTER
            run(p,tidy(str(value)),bold=ri==0,size=9.3 if len(headers)>6 else 10)
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

def supplementary_table_data(key):
    if key=='workbooks':
        x=pd.read_csv(CURRENT/'methods/dataset_by_workbook.csv')
        ordered = pd.concat([x[x.species.eq(material)] for material in NAMES])
        rows = [[NAMES[r.species], r.origin_group_id, ORIGINS_EN[r.origin_group_id], r.records,
                 r.train, r.calibration, r.prediction] for r in ordered.itertuples()]
        return 'Geographic groups and specimen splits by material',['Material','Group','Origin','Total','TRAIN','CAL','PRED'],rows,[1,.65,2.1,.5,.65,.6,.6]
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

CONTRIBUTIONS_EN = (
    'Ziyuan Zhu designed the experiments, optimized the algorithms and wrote the manuscript. '
    'Jiayi Ren assisted with checking the experimental results and pushing updates to the GitHub repository. '
    'Yiyi Yang performed the spectroscopic measurements and publicly shared the spectral data. '
    'Wenchen Jiang revised the machine-learning code and compiled the experimental results. '
    'Shuyu Zhou researched journal submission policies and assisted with organizing the supplementary files. '
    'Mengting Zhu checked the manuscript grammar and verified the information entered in the final submission forms. '
    'Yin Shen provided the experimental data and guided manuscript preparation and submission.'
)
CONTRIBUTIONS_ZH = (
    'Ziyuan Zhu负责实验设计、算法优化和论文写作。Jiayi Ren协助核查实验结果及推送GitHub仓库更新。'
    '杨易易操作光谱测量并公开分享光谱数据。蒋汶辰修改机器学习代码并汇总实验结果。'
    'Shuyu Zhou查询期刊投稿政策并协助整理补充材料。Mengting Zhu检查论文语法并核对最终投稿表单。'
    'Yin Shen提供实验数据并指导论文准备和投稿。'
)
END_ZH={
 'Acknowledgements':'致谢',
 AUTH['end_matter'][1]['text']:'第一作者感谢导师及通讯作者Yin Shen提供实验数据，并在论文撰写和投稿方面给予指导与支持。',
 'Author contributions':'作者贡献',
 'Statements and declarations':'声明', 'Ethical considerations':'伦理事项',
 'This study analyzed herbal-material spectral records and involved neither human participants nor animals. Ethical approval was not required.':'本研究分析药材光谱记录，不涉及人体参与者或动物，无需伦理审批。',
 'Consent to participate':'参与同意','Not applicable.':'不适用。','Consent for publication':'发表同意',
 'Declaration of conflicting interest':'利益冲突声明',
 'Funding statement':'资金声明',
 'The authors received no financial support for the research, authorship or publication of this article.':'本研究、论文撰写和发表未获得资金支持。',
 'Author approval':'作者批准',
 'All authors accept responsibility for the accuracy and integrity of the work. This manuscript is not under consideration elsewhere.':'所有作者对工作的准确性和完整性承担责任。稿件未同时在其他期刊审理。',
}
END_UPDATES={
    AUTH['author_contributions']: (CONTRIBUTIONS_EN, CONTRIBUTIONS_ZH),
    'The authors declared no potential conflicts of interest with respect to the research, authorship, and/or publication of this article.': (CONFLICTS, CONFLICTS_ZH),
    'All authors have read and approved the manuscript and its submission to Applied Spectroscopy.': (AUTHOR_APPROVAL, AUTHOR_APPROVAL_ZH),
}
DATA_AVAIL='The spectra, material and origin labels, specimen splits, classifier results and analysis protocols are available at https://github.com/jackzhu119/research4 and in the supplementary data. '+DATA_PROVENANCE
DATA_AVAIL_ZH='光谱、药材与产地标签、样品划分、分类结果及分析协议可从 https://github.com/jackzhu119/research4 及补充材料获得。'+DATA_PROVENANCE_ZH

def make_main(bilingual=False):
    d=setup('parallel' if bilingual else 'main');paragraph(d,TITLE,'Title')
    if bilingual:paragraph(d,TITLE_ZH,'Title')
    author_header(d)
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
        if item['text'] == AUTH['ai_disclosure']:
            paragraph(d, 'AI disclosure', 'Heading 2')
            paragraph(d, AI_DISCLOSURE)
            if bilingual:
                paragraph(d, 'AI 使用披露', 'Heading 3')
                paragraph(d, AI_DISCLOSURE_ZH)
        elif item['text'] in END_UPDATES:
            english, chinese = END_UPDATES[item['text']]
            paragraph(d, english)
            if bilingual:
                paragraph(d, chinese)
        else:
            protected(d,[item])
            if bilingual:
                paragraph(d,END_ZH[item['text']],item['style'] if item['style'].startswith('Heading') else None)
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
    d=setup('supplement');paragraph(d,'Supplementary information','Title');paragraph(d,TITLE,'Heading 1');author_header(d);n=0
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
    d = setup('title_page')
    paragraph(d, TITLE, 'Title')
    author_header(d)
    paragraph(d, 'Work addresses', 'Heading 2')
    for affiliation in AFFILIATIONS:
        paragraph(d, f"Affiliation {affiliation['id']}: {affiliation['work_address']}")
    paragraph(d, 'Author email addresses', 'Heading 2')
    for author in AUTHORS:
        paragraph(d, f"{author['name']}: {author['email']}")
    paragraph(d, 'Declarations', 'Heading 2')
    paragraph(d, CONFLICTS + ' No financial support was received for the research, authorship or publication of this article. '
              + AUTHOR_APPROVAL + ' The authors accept responsibility for the accuracy and integrity of the work. '
              'This manuscript is not under consideration elsewhere.')
    paragraph(d, 'Prepared for Applied Spectroscopy. Six main figures and one supplementary figure.')
    d.save(BASE/'02_Title_Page.docx')

    d = Document(SRC/'templates/Declarations.docx')
    for p in d.paragraphs:
        if p.text.startswith('Manuscript:'):
            replace_paragraph(p, 'Manuscript: ' + TITLE)
        elif p.text.startswith('Authors:'):
            replace_paragraph(p, 'Authors: ' + ', '.join(AUTHOR_NAMES))
        elif p.text.startswith('The authors declared no potential conflicts'):
            replace_paragraph(p, CONFLICTS)
        elif p.text.startswith('All authors have read and approved'):
            replace_paragraph(p, AUTHOR_APPROVAL + ' The authors accept responsibility for the accuracy and integrity of the work. This manuscript is not under consideration elsewhere.')
        elif p.text.startswith('The measured experimental dataset was supplied'):
            replace_paragraph(p, DATA_PROVENANCE)
    d.core_properties.author = '; '.join(AUTHOR_NAMES)
    d.core_properties.last_modified_by = ''
    d.core_properties.comments = ''
    for root in [d.element,d.styles.element]:
        for node in list(root.iter(qn('w:pBdr'))):node.getparent().remove(node)
    d.save(BASE/'05_Declarations.docx')
    d=Document(SRC/'templates/03_Cover_Letter.docx')
    updated={
      2:f'Please consider the manuscript “{TITLE}” for publication in Applied Spectroscopy.',
      3:'Routine medicinal-material identification relies on established morphological, microscopic and chemical methods. This study proposes a classification-oriented analysis route for complementary terahertz discrimination. Multiclass coefficient consistency is aggregated into a spectral ranking, and training-contained feature-budget selection produces traceable inputs. The ranking front end is separate from the final classifier, providing a structure that other estimators can reuse with their own validation and tuning.',
      4:'The collection comprises1,672 specimens in18 workbooks. In balanced five-source Suanzaoren analysis, the selected absorption workflow achieved93.44% mean accuracy with63.95% mean reduction in classifier input width. A newly available full-spectrum SVM on exactly the same inner and outer specimen IDs achieved82.67%, a10.78-percentage-point difference. Full-spectrum shrinkage LDA achieved91.78% and is retained as a strong comparator. The tested accuracy benefit is established for the SVM workflow, which supplies an explicit spectral-coordinate representation before classifier fitting.',
      5:'The ablations identify representation-dependent normalization behavior, changes in absorption subsets when the candidate pool changes, and a valid-supervision effect in separate full-spectrum label-update controls. Workbook-held-out evaluation gives98.84% accuracy for Suanzaoren–Zaoren discrimination and65.61% across all four materials, defining task-specific scope for further validation. These observed phenomena provide concrete directions for developing terahertz-assisted recognition of additional medicinal materials.',
      6:'The manuscript connects explicit spectral-coordinate selection with matched validation and controlled label updates, providing a reproducible analysis route for complementary medicinal-material identification. Complete protocol-specific comparisons, selection-repeatability estimates and source provenance accompany the submission.',
      8:DATA_PROVENANCE + ' ' + AUTHOR_APPROVAL + ' The authors accept responsibility for the accuracy and integrity of the work. This manuscript is not under consideration elsewhere. ' + CONFLICTS + ' The authors received no financial support for the research, authorship or publication.',
    }
    for i,text in updated.items():replace_paragraph(d.paragraphs[i],tidy(text))
    obsolete_disclosure = next(p for p in d.paragraphs if p.text.startswith('The use of OpenAI ChatGPT/Codex'))
    obsolete_disclosure._p.getparent().remove(obsolete_disclosure._p)
    for root in [d.element,d.styles.element]:
        for node in list(root.iter(qn('w:pBdr'))):node.getparent().remove(node)
    d.core_properties.author = CORRESPONDING['name']
    d.core_properties.last_modified_by = ''
    d.core_properties.comments = ''
    d.save(BASE/'03_Cover_Letter.docx')

REVISION_ROWS=[('Stability based title / 稳定性标题', 'Accept / 接受', 'Specify coefficient stability in the title. Coefficient consistency defines ranking; saved-refit Φ = 0.8360 describes repeatability and is not the objective or chemical assignment.', '标题明确指向系数稳定性；系数一致性用于排名，保存重拟合的Φ = 0.8360仅描述重复性，不是优化目标或化学归属。'), ('Abstract length / 摘要长度', 'Modify / 修改', 'Keep the requested 200-word ceiling as an editorial constraint. The archived Applied Spectroscopy guidance does not establish this as a journal rule.', '保留所请求的200词上限作为编辑约束；归档期刊指南未确立该项期刊要求。'), ('Matched SVM and LDA / 匹配SVM及LDA', 'Accept with qualification / 限定接受', 'The 10.78-percentage-point gain concerns the matched full-spectrum SVM. The MC20-minus-LDA interval [−1.11, 4.44] pp includes zero; no LDA advantage, equivalence or universal ranking is established.', '10.78个百分点的收益针对匹配全谱SVM；MC20减LDA区间[−1.11, 4.44]个百分点包含零，未确立相对LDA优势、等效性或普遍排名。'), ('Kernel degradation mechanism / 核退化机制', 'Reject / 拒绝', 'The measured pipeline contrast does not isolate redundant coordinates as a cause of degraded kernel distances. Retain traceable coordinate reduction without a causal mechanism or runtime claim.', '流程比较未单独识别冗余坐标导致核距离退化的因果关系；保留可追溯坐标缩减，不声称机制或运行时间优势。'), ('Background expanded SVM scores / 背景扩展SVM成绩', 'Reject as corrected evidence / 不纳入修正证据', 'Do not import background 330-repeat, random/uniform, rank-context or SVM/MC permutation scores: affected SVM searches reuse the first cached gamma. These repeats differ from allowlisted V4 task bundles.', '不引入背景330次重复、随机或均匀、排名情境及SVM/MC置换成绩；相关SVM搜索复用首档缓存gamma。该背景重复不同于白名单V4任务包。'), ('Background non SVM baselines / 背景非SVM对照', 'Modify scope / 修改范围', 'PLS-DA and other non-SVM baselines are not gamma-defective on this evidence. Their numbers remain outside corrected-main evidence pending separate protocol and provenance admission.', '现有证据不表明PLS-DA等非SVM对照存在gamma缺陷；其数值仍须单独核查协议与来源并获准后方可纳入修正主证据。'), ('Permutation and n information / 置换与n信息', 'Reject proposed inference / 拒绝所提推断', 'Chance-level permutation would not certify absence of every leakage mode. Ranking perturbations do not prove n contains no information under other predictors.', '机会水平置换不能认证不存在任何泄漏；排名扰动也不能证明n在其他预测器下无信息。'), ('Recurrent source errors / 重复来源错误', 'Accept descriptively / 描述性接受', 'The three actual repeats contain 900 related primary predictions: ZY07→ZY04 accounts for 30 of 59 errors; eight specimens show it in all three repeats. No chemical or procurement mechanism is inferred.', '三次实际重复共有900条相关主预测；59次错误中30次为ZY07→ZY04，8份样品三次均如此。不推断化学或采购机制。'), ('ZY07 provenance / ZY07来源', 'Modify / 修改', 'Use the confirmed Taiyuan, Shanxi locality for ZY07, with separate workbook and specimen identifiers.', 'ZY07采用已确认的山西太原产地，工作簿及样品标识独立。'), ('Fixed PRED T27 contrasts / 固定PRED的T27比较', 'Accept with qualification / 限定接受', 'Report full_svm 85/100, fixed_MC20 87/100 and R0 94/100 for Suanzaoren sources, and 331/339 versus 313/339 for four materials. R0 also uses 62 coordinates but changes preprocessing and C; these retrospective fixed-role results do not isolate a budget effect.', '报告酸枣仁来源full_svm为85/100、fixed_MC20为87/100、R0为94/100，以及四材料331/339对313/339。R0同为62坐标但改变预处理和C；回顾性固定角色结果不能单独识别预算效应。'), ('Protocol linkage and contribution / 协议联系与贡献', 'Modify / 修改', 'Describe a training-contained reusable ranking interface with measured matched-SVM benefit. LOWO, fixed PRED and full-spectrum support-label controls remain separate; other estimators need direct validation and support gains do not add to primary accuracy.', '描述限定于训练过程、可复用的排名接口及匹配SVM实测收益；LOWO、固定PRED和全谱支持标签对照分开，其他估计器须直接验证，支持收益不叠加主准确率。'), ('Authorship / 作者信息', 'Preserve / 保留', 'The 19 November author list supersedes the earlier byline. The published author details follow the current title page.', '作者署名以11月19日更新后的作者页为准。')]

def make_revision_notes():
    d=setup('notes');paragraph(d,'Revision record and evidence checks','Title');paragraph(d,'修改记录与证据核查','Title')
    paragraph(d,TITLE);paragraph(d,TITLE_ZH)
    paragraph(d,'Author and specimen information updated on 19 November 2026','Heading 2')
    paragraph(d,'2026年11月19日作者与样品资料更新','Heading 3')
    paragraph(d,'Yiyi Yang and Wenchen Jiang were added as the third and fourth authors. The manuscript records pharmacy procurement, Yang’s acquisition of the spectra at Southwest University, and the Taiyuan origin of ZY07. The supplementary table groups the 18 workbooks by commercial material and shows the specimen split for each origin.')
    paragraph(d,'杨易易和蒋汶辰分别列为第三、第四作者。稿件写明药材从中药店购买，由杨易易在西南大学采集光谱，ZY07 产地为山西太原。补充表按商品药材类别归列18份工作簿，并列出各产地的样品划分。')
    paragraph(d,'Second review round, 16 November 2026. This record addresses doubao.txt and codex_revision_instructions.md against the corrected evidence allowlist and the two review audits. No scientific models were trained during this revision.')
    paragraph(d,'2026年11月16日第二轮审阅。本记录依据修正证据白名单及两份审阅核查，回应doubao.txt和codex_revision_instructions.md。本次修改未训练科学模型。')
    paragraph(d,'Accepted suggestions improve the title, scope, source-label description and descriptive contrasts. Modified or rejected suggestions are recorded below with their evidence limits. The contribution remains complementary medicinal-material discrimination with a training-contained ranking and selection workflow.')
    paragraph(d,'接受的建议用于完善标题、适用范围、来源标签描述和描述性比较；修改或拒绝的建议及证据边界逐项记录如下。贡献仍为服务于药材辅助判别、限定于训练过程的排名与选择流程。')
    for concern,decision,en,zh in REVISION_ROWS:
        paragraph(d,concern,'Heading 2');paragraph(d,decision).paragraph_format.keep_with_next=True
        paragraph(d,en).paragraph_format.keep_with_next=True;paragraph(d,zh)
    paragraph(d,'Evidence basis / 证据依据','Heading 1')
    paragraph(d,'The corrected publication view, pinned E1 bridge and saved primary predictions support the retained comparisons. Conditional paired intervals use 10,000 source-stratified physical-ID draws from saved predictions (seed 20261116). Repeated predictions preserve specimen identity and are not counted as independent specimens.')
    paragraph(d,'保留的比较以修正结果入口、固定E1桥接及保存主预测为依据。条件性配对区间使用保存预测，按来源分层进行10,000次实物ID抽样（种子20261116）；重复预测保持实物身份，不另计为独立样品。')
    paragraph(d,'The English manuscript is the submission version. The bilingual manuscript and this record are for author review.')
    paragraph(d,'英文正文为投稿版本。中英文稿与本记录供作者核阅。')
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
