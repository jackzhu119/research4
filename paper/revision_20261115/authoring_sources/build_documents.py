from pathlib import Path
import copy,json,re,subprocess,tempfile,argparse
import pandas as pd
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH,WD_TAB_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'data/active_workbooks.json').exists())
OUT=Path(__file__).resolve().parent
ANALYSIS=ROOT/'paper/revision_20261115/Supplementary_Data_S1/derived_checks'
FIGURES=ROOT/'paper/revision_20261115/figures'
DELIVERY=ROOT/'paper/revision_20261115'
DELIVERY.mkdir(parents=True,exist_ok=True)

EQS={
'transfer':r'T(\omega)=\frac{4n(\omega)}{[n(\omega)+1]^2}\exp\!\left[-\frac{\alpha(\omega)d}{2}\right]\exp\!\left[-i\frac{[n(\omega)-1]\omega d}{c}\right]',
'index':r'n(\omega)=1+\frac{c\Phi(\omega)}{\omega d}',
'absorption':r'\alpha(\omega)=-\frac{2}{d}\ln\!\left\{|T(\omega)|\frac{[n(\omega)+1]^2}{4n(\omega)}\right\}',
'coefficient_moments':r'\begin{aligned}\bar{\beta}_{gj}&=\frac{1}{M}\sum_{m=1}^{M}\beta_{gj}^{(m)}\\s_{gj}&=\left[\frac{1}{M-1}\sum_{m=1}^{M}\left(\beta_{gj}^{(m)}-\bar{\beta}_{gj}\right)^2\right]^{1/2}\end{aligned}',
'multiclass_score':r'q_{gj}=\frac{|\bar{\beta}_{gj}|}{\max(s_{gj},\varepsilon)},\qquad S_j=\left(\sum_{g=1}^{G}q_{gj}^{\,2}\right)^{1/2}',
'budget_selection':r'\mathcal{J}_{K}^{\star}=\underset{\substack{\mathcal{J}\subseteq\{1,\ldots,p\}\\|\mathcal{J}|=K}}{\operatorname{arg\,max}}\ \sum_{j\in\mathcal{J}}S_j',
'inner_selection':r'\theta^{\star}=\underset{\theta\in\Theta}{\operatorname{arg\,max}}\ \frac{1}{L}\sum_{\ell=1}^{L}F_{1,\mathrm{macro}}\!\left(\mathbf{y}_{\ell},\widehat{\mathbf{y}}_{\ell}^{(\theta)}\right),\qquad L=2',
'loss_control':r'C_{\mathrm{norm}}=C_{\mathrm{src}}\frac{N_{\mathrm{src}}}{N_{\mathrm{src}}+N_{\mathrm{sup}}}',
}
with tempfile.TemporaryDirectory() as td:
    f=Path(td)/'equations.md'; f.write_text('\n\n'.join('$$'+x+'$$' for x in EQS.values()))
    subprocess.run(['pandoc',str(f),'-o',str(Path(td)/'equations.docx')],check=True)
    ed=Document(Path(td)/'equations.docx')
    equations={key:copy.deepcopy(node) for key,node in zip(EQS,ed.element.findall('.//'+qn('m:oMathPara')))}
assert len(equations)==8

def text_run(p,text,size=None,bold=False):
    r=p.add_run(text);r.bold=bold
    r.font.name='Times New Roman';r.font.color.rgb=RGBColor(0,0,0)
    if size:r.font.size=Pt(size)
    rp=r._element.get_or_add_rPr();font=rp.find(qn('w:rFonts'))
    if font is None:font=OxmlElement('w:rFonts');rp.insert(0,font)
    font.set(qn('w:eastAsia'),'Noto Serif CJK SC')
    return r

def setup(kind):
    d=Document();s=d.sections[0]
    s.page_width=Inches(8.5);s.page_height=Inches(11)
    s.top_margin=Inches(.85);s.bottom_margin=Inches(.85)
    s.left_margin=Inches(1);s.right_margin=Inches(1)
    s.header_distance=Inches(.35);s.footer_distance=Inches(.35)
    for name in ['Normal','Title','Heading 1','Heading 2','Heading 3','Caption']:
        st=d.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
        st._element.get_or_add_rPr().append(OxmlElement('w:rFonts'))
        st._element.rPr.rFonts.set(qn('w:eastAsia'),'Noto Serif CJK SC')
    normal=d.styles['Normal'];normal.font.size=Pt(11.5 if kind=='main' else 11)
    normal.paragraph_format.line_spacing=1.18 if kind=='main' else 1.12
    normal.paragraph_format.space_after=Pt(6)
    normal.paragraph_format.widow_control=True
    if kind=='report':
        normal.font.size=Pt(10.5)
        normal.paragraph_format.line_spacing=1.06
        normal.paragraph_format.space_after=Pt(5)
    for n,pt in [('Title',16),('Heading 1',13),('Heading 2',12),('Heading 3',11.5)]:
        d.styles[n].font.size=Pt(pt);d.styles[n].font.bold=n!='Title'
        d.styles[n].paragraph_format.space_before=Pt(12 if n!='Title' else 0)
        d.styles[n].paragraph_format.space_after=Pt(6)
        d.styles[n].paragraph_format.keep_with_next=True
    d.styles['Caption'].font.size=Pt(10)
    d.core_properties.author='Ziyuan Zhu; Jiayi Ren; Shuyu Zhou; Mengting Zhu; Yin Shen'
    d.core_properties.comments='';d.core_properties.keywords='';d.core_properties.last_modified_by=''
    foot=s.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=text_run(foot,'',9);field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');r._r.addnext(field)
    return d

def paragraph(d,text,style=None):
    p=d.add_paragraph(style=style)
    is_reference=bool(re.match(r'^\[\d+\]\s',text))
    if is_reference:text=re.sub(r'^\[(\d+)\]\s',r'\1. ',text)
    # Preserve source wording; simple Markdown bold is converted to native runs.
    for i,part in enumerate(re.split(r'\*\*(.*?)\*\*',text)):
        if not part:continue
        for piece in re.split(r'(\[\d+(?:[,–-]\d+)*\])',part):
            if not piece:continue
            citation=not is_reference and bool(re.fullmatch(r'\[\d+(?:[,–-]\d+)*\]',piece))
            value=piece[1:-1] if citation else piece
            if citation:
                value=value.replace('–','\u2060–\u2060').replace('-', '\u2060-\u2060').replace(',', ',\u2060')
            run=text_run(p,value,bold=bool(i%2) or style in ['Heading 1','Heading 2','Heading 3'])
            if citation:run.font.superscript=True
    if is_reference:
        journals=['Hong Kong Chinese Materia Medica Standards','Food Chem X','J Appl Spectrosc','TrAC Trends Anal Chem','Anal Chim Acta','BMC Bioinformatics','J Mach Learn Res','Chemom Intell Lab Syst','Anal Chem','J Chem Phys','Phys Rev Lett','Phys Chem Chem Phys','Phytochem Anal','J Chromatogr B Analyt Technol Biomed Life Sci','Laser Photon Rev','J Pharm Pharmacol','Crit Rev Anal Chem','Microchem J','Infrared Phys Technol','Front Bioeng Biotechnol','IEEE J Sel Top Quantum Electron','Appl Spectrosc','Mach Learn']
        pattern='('+'|'.join(re.escape(j) for j in sorted(journals,key=len,reverse=True))+')'
        content=p.text;p.clear()
        for piece in re.split(pattern,content):
            if piece:
                run=text_run(p,piece)
                if piece in journals:run.italic=True
    return p

def numbered_equation(d,key):
    p=d.add_paragraph()
    p.paragraph_format.tab_stops.add_tab_stop(Inches(3.25),WD_TAB_ALIGNMENT.CENTER)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(6.5),WD_TAB_ALIGNMENT.RIGHT)
    p.paragraph_format.space_before=Pt(4);p.paragraph_format.space_after=Pt(8)
    text_run(p,'\t')
    math=copy.deepcopy(equations[key].find(qn('m:oMath')))
    # The journal sets Greek symbols and the imaginary unit in upright type.
    for run in math.iter(qn('m:r')):
        value=''.join(node.text or '' for node in run.iter(qn('m:t')))
        if re.fullmatch(r'[Α-Ωα-ω]+',value) or value=='i':
            props=run.find(qn('m:rPr'))
            if props is None:props=OxmlElement('m:rPr');run.insert(0,props)
            sty=props.find(qn('m:sty'))
            if sty is None:sty=OxmlElement('m:sty');props.append(sty)
            sty.set(qn('m:val'),'p')
    p._p.append(math)
    text_run(p,'\t('+str(list(EQS).index(key)+1)+')')

def table(d,headers,rows,widths=None):
    t=d.add_table(rows=1, cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    if widths is None:
        n=len(headers)
        if n==7:widths=[.55,.85,2.05,.75,.75,.75,.75]
        elif n==6:widths=[2.2,.55,.9,.9,.85,1.1]
        elif n==5:widths=[2.0,1.1,1.1,1.15,1.15]
        elif n==4:widths=[3.0,.9,1.0,1.6]
        elif n==3:widths=[1.1,2.7,2.7]
        else:widths=[6.5/n]*n
    scale=6.5/sum(widths);widths=[x*scale for x in widths]
    for i,w in enumerate(widths):t.columns[i].width=Inches(w)
    for i,v in enumerate(headers):t.rows[0].cells[i].text=str(v)
    for row in rows:
        cells=t.add_row().cells
        for c,v in zip(cells,row):c.text=str(v)
    borders=OxmlElement('w:tblBorders')
    for name in ['top','left','bottom','right','insideH','insideV']:
        b=OxmlElement('w:'+name);b.set(qn('w:val'),'single' if name in ['top','bottom'] else 'nil');b.set(qn('w:sz'),'6');b.set(qn('w:color'),'000000');borders.append(b)
    t._tbl.tblPr.append(borders)
    for ri,row in enumerate(t.rows):
        trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
        if ri==0:trpr.append(OxmlElement('w:tblHeader'))
        for ci,c in enumerate(row.cells):
            c.width=Inches(widths[ci]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr=c._tc.get_or_add_tcPr();marg=OxmlElement('w:tcMar')
            for edge,val in [('top',70),('bottom',70),('left',85),('right',85)]:
                m=OxmlElement('w:'+edge);m.set(qn('w:w'),str(val));m.set(qn('w:type'),'dxa');marg.append(m)
            tcpr.append(marg)
            if ri==0:
                cb=OxmlElement('w:tcBorders');bottom=OxmlElement('w:bottom');bottom.set(qn('w:val'),'single');bottom.set(qn('w:sz'),'4');bottom.set(qn('w:color'),'000000');cb.append(bottom);tcpr.append(cb)
            for p in c.paragraphs:
                p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.05
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 or len(c.text)>22 else WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.name='Times New Roman';r.font.size=Pt(9.5 if len(headers)>5 else 10)
                    r.font.bold=ri==0;r.font.color.rgb=RGBColor(0,0,0)
                    r._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Serif CJK SC')
    d.add_paragraph().paragraph_format.space_after=Pt(2)
    return t

def caption(d,text):
    p=paragraph(d,text);p.paragraph_format.keep_with_next=True
    for r in p.runs:r.font.bold=True;r.font.size=Pt(10)

def note(d,text):
    p=paragraph(d,text)
    for r in p.runs:r.font.size=Pt(9.5)

def main_table(d,key):
    if key=='dataset':
        caption(d,'Table I  Material collection and specimen allocation')
        x=pd.read_csv(ROOT/'paper/current/methods/dataset_by_species.csv').set_index('species')
        mapping={'酸枣仁':'Suanzaoren','枣仁':'Zaoren','黄芪':'Huangqi','天麻':'Tianma'}
        rows=[[label,*[int(x.loc[s,c]) for c in ['workbooks','records','train','calibration','prediction']]] for s,label in mapping.items()]
        rows.append(['Total',18,1672,999,334,339]);table(d,['Material','Workbooks','Total','TRAIN','CAL','PRED'],rows,[1.7,.7,.85,.85,.8,.8])
        note(d,'The principal nested analysis uses 300 Suanzaoren TRAIN specimens. Each of its five groups contains 100 specimens assigned 60/20/20. Four workbook views describe the same specimens. Frozen roles are specimen partitions within the recorded collection.')
    elif key=='tasks':
        caption(d,'Table II  Absorption classification of materials and source groups')
        x=pd.read_csv(ROOT/'paper/current/tables/fixed_classification.csv');x=x[(x.model=='full_svm')&(x.method=='source_only')].set_index('task')
        names={'binary_kinds':'Suanzaoren vs Zaoren','four_kinds':'Four materials','acid_origins':'Suanzaoren sources','jujube_origins':'Zaoren sources','huangqi_origins':'Huangqi sources','tianma_origins':'Tianma sources'}
        lowo=json.loads((ROOT/'paper/current/results/kind_lowo_20261115/summary.json').read_text())['tasks']
        cls={'binary_kinds':2,'four_kinds':4,'acid_origins':5,'jujube_origins':5,'huangqi_origins':4,'tianma_origins':4}
        rows=[]
        for k,name in names.items():
            if k in lowo:
                result=lowo[k];z=result['metrics'];n=z['n_records']
                rows.append([name,cls[k],f"LOWO ({result['n_folds']} folds)",f"{z['n_correct']}/{n}",f"{100*z['accuracy']:.2f}",f"{z['macro_f1']:.4f}",f"{100*z['balanced_accuracy']:.2f}"])
            else:
                z=x.loc[k];n=int(z.prediction_count)
                rows.append([name,cls[k],'Fixed roles',f'{round(z.accuracy_mean*n)}/{n}',f'{100*z.accuracy_mean:.2f}',f'{z.macro_f1_mean:.4f}',f'{100*z.balanced_accuracy_mean:.2f}'])
        table(d,['Task','Classes','Validation','Correct / evaluated','Accuracy %','Macro F1','Balanced accuracy %'],rows,[1.5,.65,.8,1.0,.85,.8,.95])
        note(d,'LOWO pools one out-of-workbook prediction per material-task specimen; source tasks retain frozen TRAIN/PRED roles. All rows use 612 α coordinates, SNV, balanced RBF SVM, C=10 and gamma=scale, with scaling fitted only on the corresponding training data. Source-task TRAIN-majority accuracies are 20.00%, 94.59%, 26.83% and 25.30% in source-row order. Tasks overlap in specimens; the protocols have different evaluation units.')
    elif key=='primary':
        caption(d,'Table III  Nested validation of spectral representations')
        # Use exact replayed current metrics, with archived retained K means.
        x=pd.read_csv(ANALYSIS/'feature_primary_replayed.csv').set_index('family')
        specs=[('alpha_snv20','SNV α MC20–SVM',220.6),('alpha_snv100','SNV α MC100–SVM',216.5),('alpha_raw20','Raw α MC20–SVM',204.3),('alpha_snv_stability5x20','SNV α five MC20 streams–SVM',216.5),('fullalpha_shrinkLDA','Full α shrinkage LDA',612),('concat_both_snv20','Joint SNV n + SNV α MC20–SVM',216.5),('weighted_concat_nraw_asnv20','Weighted raw n + SNV α',232.8),('block_half_nraw_asnv20','Equal-block raw n + SNV α',236.9),('concat_nraw_asnv20','Joint raw n + SNV α MC20–SVM',228.7),('n_raw20','Raw n MC20–SVM',149.3),('fullalpha_MLP','Full α MLP',612),('n_snv20','SNV n MC20–SVM',224.7)]
        rows=[[label,f'{k:.1f}',f'{100*x.loc[m,"accuracy"]:.2f}',f'{x.loc[m,"macro_f1"]:.4f}'] for m,label,k in specs]
        table(d,['Representation and method','Mean K','Accuracy %','Macro F1'],rows,[3.7,.7,1,1.1])
        note(d,'Means average three pooled fivefold estimates on the same 300 specimens. K and eligible settings are inner-selected. Balanced accuracy equals accuracy for these balanced pooled evaluations. Equal-block ranking uses two selectors and has a different coefficient-draw cost.')
    elif key=='adaptation':
        caption(d,'Table IV  Five-source support controls at 10 labels per class')
        data=pd.read_csv(ANALYSIS/'adaptation_controls.csv')
        names=['Frozen source','True labels, full pipeline refit','True labels, frozen scaler and gamma','Unlabelled-statistics update','Cyclically incorrect labels, refit','True labels, normalized loss mass']
        # Authoritative point endpoints independently extracted by the evidence auditor.
        values=[80,83.88888888888889,83.77777777777777,80.11111111111111,73.33333333333333,83.77777777777777]
        table(d,['Intervention','Accuracy %','Difference from source pp'],[[n,f'{v:.4f}',f'{v-80:+.4f}'] for n,v in zip(names,values)],[3.8,1,1.7])
        note(d,'15 contexts each contain Source190/Support50/Query60 within the same 300 TRAIN specimens. Percentages summarize 900 related Query predictions. Full refitting and frozen-transform refitting use different operations. Complete endpoint sources and paired intervals are supplied in the supplementary data.')

def add_figure(d,key):
    names={'model':'Fig_model_selection.png','spectra':'fig1_spectra_current.png','budget':'fig2_feature_budget.png','modality':'fig3_selected_modalities.png','adaptation':'fig4_adaptation_controls.png'}
    caps=json.loads((OUT/'figure_captions.json').read_text())
    p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run().add_picture(str(FIGURES/names[key]),width=Inches(6.5))
    p.paragraph_format.keep_with_next=True
    p=paragraph(d,caps[key]);p.paragraph_format.line_spacing=1.05
    for r in p.runs:r.font.size=Pt(10)

def from_text(source,filename,kind):
    d=setup(kind);lines=Path(source).read_text().splitlines();i=0
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line:continue
        if line.startswith('|'):
            arr=[line]
            while i<len(lines) and lines[i].strip().startswith('|'):arr.append(lines[i].strip());i+=1
            parsed=[[x.strip() for x in l.strip('|').split('|')] for l in arr]
            table(d,parsed[0],[x for x in parsed[1:] if not all(re.fullmatch('[: -]+',v) for v in x)])
            continue
        if line.startswith('[TABLE:'):main_table(d,line[7:-1]);continue
        if line.startswith('[FIGURE:'):add_figure(d,line[8:-1]);continue
        if line.startswith('[EQUATION:'):
            numbered_equation(d,line[10:-1]);continue
        if line.startswith('# '):
            p=paragraph(d,line[2:],'Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;continue
        if line.startswith('### '):paragraph(d,line[4:],'Heading 2');continue
        if line.startswith('## '):
            if kind=='main' and line=='## Tables':d.add_page_break()
            paragraph(d,line[3:],'Heading 1');continue
        if line.startswith('Table S'):caption(d,line);continue
        if line.startswith('Figure S1.'):
            p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(FIGURES/'Fig_S_quantum_accepted_PCA.png'),width=Inches(6.0));p.paragraph_format.keep_with_next=True
            paragraph(d,line);continue
        p=paragraph(d,line)
        if kind=='supplement' and '## S9 Reproducibility and evidence files' in lines[:i]:
            p.paragraph_format.line_spacing=1.06
            p.paragraph_format.space_after=Pt(4)
            for r in p.runs:r.font.size=Pt(10.5)
        if re.match(r'^\[\d+\]',line):
            p.paragraph_format.line_spacing=1.03
            for r in p.runs:r.font.size=Pt(10)
        if kind=='cover':
            p.paragraph_format.line_spacing=1.03;p.paragraph_format.space_after=Pt(6)
            for r in p.runs:r.font.size=Pt(10.5)
    # The bundled reference template can carry a blue paragraph border in Title.
    for root in [d.element,d.styles.element]:
        for border in list(root.iter(qn('w:pBdr'))):border.getparent().remove(border)
    d.core_properties.title=d.paragraphs[0].text if d.paragraphs else 'Manuscript'
    target=DELIVERY/filename;d.save(target);print(target)

def bilingual_document():
    from docx.text.paragraph import Paragraph
    from docx.table import Table
    translations=json.loads((OUT/'bilingual_translation.json').read_text())
    # Source-form citation keys and the rendered AMA runs must resolve to the same translation.
    for english,chinese in list(translations.items()):
        rendered=re.sub(r'\[(\d+(?:[,–-]\d+)*)\]',lambda m:m.group(1).replace('–','\u2060–\u2060').replace('-','\u2060-\u2060').replace(',',',\u2060'),english)
        translations.setdefault(rendered,chinese)
    source=Document(DELIVERY/'01_Manuscript_Optimized.docx')
    d=setup('bilingual');missing=[];picture=0;references=False;pending_caption=None;table_count=0;translated_paragraphs=0
    image_names=['Fig_model_selection.png','fig1_spectra_current.png','fig2_feature_budget.png','fig3_selected_modalities.png','fig4_adaptation_controls.png']
    def translation(text):
        if text in translations:return translations[text]
        clean=text.replace('\u2060','')
        if clean in translations:return translations[clean]
        missing.append(text);return text
    def chinese_paragraph(text,style=None):
        p=paragraph(d,text,style)
        p.paragraph_format.space_after=Pt(8);p.paragraph_format.line_spacing=1.15
        for run in p.runs:
            run.font.size=Pt(11.5 if style in ['Heading 1','Heading 2'] else 11)
            run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Serif CJK SC')
        if style=='Title':p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        return p
    for element in source.element.body:
        if element.tag==qn('w:sectPr'):continue
        if element.tag==qn('w:tbl'):
            table_count+=1;original=Table(element,source)
            d.element.body.insert(len(d.element.body)-1,copy.deepcopy(element))
            if pending_caption:
                chinese_paragraph(pending_caption,'Caption').paragraph_format.keep_with_next=True
                pending_caption=None
            headers=[translation(c.text) for c in original.rows[0].cells]
            rows=[]
            for row in original.rows[1:]:
                cells=[]
                for c in row.cells:
                    text=c.text
                    cells.append(translation(text) if any(ch.isalpha() for ch in text) else text)
                rows.append(cells)
            widths=[c.width.inches if c.width else 1 for c in original.columns]
            table(d,headers,rows,widths)
            continue
        p=Paragraph(element,source);text=p.text.strip()
        if element.find('.//'+qn('w:drawing')) is not None:
            q=d.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.keep_with_next=True
            q.add_run().add_picture(str(FIGURES/image_names[picture]),width=Inches(6.5));picture+=1
            continue
        if element.find('.//'+qn('m:oMath')) is not None:
            d.element.body.insert(len(d.element.body)-1,copy.deepcopy(element));continue
        if text=='References':references=True
        if text=='Tables':references=False
        cloned=copy.deepcopy(element);d.element.body.insert(len(d.element.body)-1,cloned)
        if not text or (references and re.match(r'^\d+\. ',text)):continue
        chinese=translation(text)
        if chinese==text:continue
        if re.match(r'^Table [IV]+\s',text):
            pending_caption=chinese;continue
        english=Paragraph(cloned,d)
        english.paragraph_format.keep_with_next=True
        st=p.style.name if p.style is not None else 'Normal'
        style=st if st in ['Title','Heading 1','Heading 2','Heading 3','Caption'] else None
        chinese_paragraph(chinese,style);translated_paragraphs+=1
    assert picture==5 and table_count==4
    if missing:raise ValueError('Missing bilingual translations: '+json.dumps(list(dict.fromkeys(missing)),ensure_ascii=False))
    for root in [d.element,d.styles.element]:
        for border in list(root.iter(qn('w:pBdr'))):border.getparent().remove(border)
    d.core_properties.title='Manuscript English Chinese parallel review'
    target=DELIVERY/'07_Manuscript_EN_ZH.docx';d.save(target)
    (DELIVERY/'BILINGUAL_VERIFICATION.json').write_text(json.dumps({'status':'PASS','main_document_sha256':__import__('hashlib').sha256((DELIVERY/'01_Manuscript_Optimized.docx').read_bytes()).hexdigest(),'parallel_document_sha256':__import__('hashlib').sha256(target.read_bytes()).hexdigest(),'translated_prose_blocks':translated_paragraphs,'english_tables':table_count,'chinese_tables':table_count,'figures':picture,'untranslated_requested_strings':[],'references':'Original English bibliography retained','figure_labels':'Original English plot labels retained; captions translated'},ensure_ascii=False,indent=2)+'\n')
    print(target)

if __name__=='__main__':
    targets={
        'main':('manuscript_source.txt','01_Manuscript_Optimized.docx','main'),
        'supplement':('supplementary_source.md','02_Supplementary_Information.docx','supplement'),
        'report':('revision_report_source.txt','03_论文优化与清理说明.docx','report'),
        'cover':('cover_source.txt','04_Cover_Letter.docx','cover'),
        'title':('title_source.txt','05_Title_Page.docx','title'),
        'guide':('submission_guide_source.txt','06_投稿文件说明.docx','guide'),
        'declarations':('declarations_source.txt','08_Declarations.docx','declarations')}
    parser=argparse.ArgumentParser(description='Build all document deliverables or selected targets')
    parser.add_argument('targets',nargs='*',choices=list(targets)+['bilingual'])
    selected=parser.parse_args().targets or list(targets)+['bilingual']
    for key in selected:
        if key=='bilingual':bilingual_document()
        else:
            src,filename,kind=targets[key];from_text(OUT/src,filename,kind)
