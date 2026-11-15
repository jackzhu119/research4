#!/usr/bin/env python3
"""Draw the principal MC20–SVM model-selection design, without fitting models."""
from pathlib import Path
import hashlib
import json
import subprocess
import csv
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path as MplPath
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'data/active_workbooks.json').exists())
BASE = ROOT / 'paper/revision_20261115'
OUT = BASE / 'figures'
NAME = 'Fig_model_selection'
INK = '#263238'
BLUE = '#486B80'
FILL = '#F3F8FB'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.2,
    'mathtext.fontset': 'dejavusans',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})

fig = plt.figure(figsize=(6.5, 5.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.set_axis_off()
labels = []


def text(x, y, value, size=7.2, weight='normal', color=INK, **kwargs):
    kwargs.setdefault('ha', 'center')
    kwargs.setdefault('va', 'center')
    item = ax.text(x, y, value, fontsize=size,
                   fontweight=weight, color=color, linespacing=1.24, **kwargs)
    labels.append(item)
    return item


def box(x, y, width, height, value='', fill=FILL, size=7.2, edge=BLUE):
    ax.add_patch(Rectangle((x, y), width, height, facecolor=fill,
                           edgecolor=edge, linewidth=.75, zorder=2))
    if value:
        text(x+width/2, y+height/2, value, size=size, zorder=3)


def arrow(points):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=7,
                                linewidth=.8, color=INK, zorder=2.5))


# One thin frame encloses both panels, their labels and all connectors.
ax.add_patch(Rectangle((.004, .006), .992, .988, fill=False,
                       edgecolor=INK, linewidth=.8, zorder=5))
text(.035, .963, 'A  Nested model selection', size=9.5, weight='bold', ha='left')

box(.04, .855, .24, .074, '300 Suanzaoren TRAIN\n5 recorded sources × 60')
box(.34, .855, .31, .074,
    'Stratified 5 outer folds × 3 seeds\n42 / 20261031 / 20261108')
arrow([(.282, .892), (.338, .892)])

# Outer-training group: no incoming edge from the outer test branch.
box(.04, .662, .61, .160, fill='#FAFBFC', edge='#9AA7B0')
text(.055, .805, 'Outer training: 240 specimens', size=7.5, weight='bold', ha='left')
box(.055, .679, .355, .103,
    'Two inner folds\nSelect $K$, $C$ and γ by macro-F1\nLearn transforms and ranking\non each inner TRAIN subset')
box(.46, .692, .175, .077,
    'Refit selected\npipeline on all\n240 outer TRAIN')
arrow([(.412, .7305), (.458, .7305)])
arrow([(.495, .853), (.345, .824)])

box(.735, .697, .225, .084,
    'Outer test: 60\nUntouched during\nmodel selection', fill='white', edge='#6B7280')
arrow([(.652, .892), (.8475, .892), (.8475, .783)])
box(.735, .578, .225, .075,
    'Predict outer test: 60\nWith refitted pipeline', fill='#F1F7F4', edge='#60816F')
arrow([(.8475, .695), (.8475, .655)])
arrow([(.637, .7305), (.691, .7305), (.691, .621), (.733, .621)])
box(.04, .565, .61, .074,
    'Pool 300 outer predictions per repeat; average three repeats.\n'
    'The same 300 specimens yield 900 repeated predictions,\n'
    'not 900 independent specimens.', fill='white', size=7.1, edge='#6B7280')
arrow([(.733, .595), (.652, .595)])

ax.plot([.035, .965], [.543, .543], color='#BFC8CE', lw=.6, zorder=1)
text(.035, .518, 'B  MC20–SVM fit pipeline', size=9.5, weight='bold', ha='left')

box(.04, .376, .265, .106,
    'Paired spectral candidates\nα: 612; $n$: 612; joint: 1,224\n'
    'Raw / block-wise SNV\nTraining-fitted standardization', size=7.1)
box(.375, .376, .27, .106,
    'Stratified MC coefficient ranking\n20 draws; ceil(0.8 × class size)\n'
    'Within-draw $X$/$y$ scaling\nOne-vs-rest PLS: 2 components', size=7.1)
box(.71, .376, .25, .106)
text(.835, .463, 'Across-class importance', size=7.2, weight='bold')
text(.835, .434,
     r'$q_{gj}=|\mathrm{\mu}_{gj}|/\max(\mathrm{\sigma}_{gj},\mathrm{\varepsilon})$', size=8.0)
text(.835, .407, r'$S_j=\|q_{\cdot j}\|_2$', size=8.0)
text(.835, .385, 'μ: mean; σ: SD (ddof = 1)', size=7.0)
arrow([(.307, .429), (.373, .429)])
arrow([(.647, .429), (.708, .429)])

box(.04, .202, .265, .105,
    'Retain top $K$ variables\n$K$ ∈ {31, 62, 123, 184, 245}')
box(.375, .202, .27, .105,
    'Class-balanced RBF SVM\n$C$ ∈ {0.1, 1, 10, 100}\n'
    'γ = {0.1, 1, 10} × scale\nInner-fold selection only')
box(.71, .202, .25, .105,
    'Trace retained variables\nModality and frequency\nof the final predictor matrix',
    fill='#F1F7F4', edge='#60816F')
arrow([(.835, .374), (.835, .341), (.1725, .341), (.1725, .309)])
arrow([(.307, .2545), (.373, .2545)])
arrow([(.647, .2545), (.708, .2545)])

box(.04, .065, .92, .09,
    'Primary analysis: $K$ is selected within the inner folds.\n'
    'Fixed-$K$ comparison: $K$ is prescribed; SVM settings remain inner-selected.\n'
    'All learned transformations and rankings are fitted on the relevant training subset.',
    fill='white', size=7.0, edge='#9AA7B0')

# Text extents must remain inside the complete enclosing frame.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
limits = fig.bbox
for item in labels:
    area = item.get_window_extent(renderer)
    assert area.x0 > limits.x0 and area.y0 > limits.y0 and area.x1 < limits.x1 and area.y1 < limits.y1, item.get_text()

OUT.mkdir(exist_ok=True)
outputs = []
for extension in ['png', 'tiff', 'pdf', 'eps', 'svg']:
    path = OUT / f'{NAME}.{extension}'
    if extension == 'tiff':
        with Image.open(OUT / f'{NAME}.png') as image:
            assert image.getextrema()[-1] == (255, 255)
            image.convert('RGB').save(path, compression='tiff_lzw', dpi=(1200, 1200))
    else:
        fig.savefig(path, dpi=1200, facecolor='white')
    entry = {'path': str(path.relative_to(BASE)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    if extension in ['png', 'tiff']:
        with Image.open(path) as image:
            entry.update(pixels=list(image.size), dpi=[float(v) for v in image.info['dpi']], color_mode=image.mode)
            if extension == 'tiff':
                assert image.mode == 'RGB' and image.info['compression'] == 'tiff_lzw'
                entry['compression'] = image.info['compression']
    if extension == 'eps':
        value = path.read_text('ascii')
        assert value.startswith('%!PS-Adobe-3.0 EPSF-3.0') and '/FontType 42 def' in value
        subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-sDEVICE=nullpage', str(path)], check=True, capture_output=True)
        entry['ghostscript_interpretation'] = 'PASS'
    if extension == 'pdf':
        font_report = subprocess.check_output(['pdffonts', str(path)], text=True)
        assert 'Type 3' not in font_report
        assert all('yes' in line for line in font_report.splitlines()[2:] if line.strip())
        entry['font_embedding_report'] = font_report
    outputs.append(entry)
plt.close(fig)

split_path = ROOT / 'paper/current/inputs/split_manifest.csv'
with split_path.open(encoding='utf-8') as handle:
    train = [row for row in csv.DictReader(handle) if row['species'] == '酸枣仁' and row['role'] == 'train']
assert len(train) == 300
assert dict(Counter(row['origin_group_id'] for row in train)) == {f'ZY{i:02d}': 60 for i in range(3, 8)}
sources = [split_path,
           BASE / 'Supplementary_Data_S1/derived_checks/feature_source_verification.json',
           BASE / 'Supplementary_Data_S1/derived_checks/verified_v4_sources/runs/matched_r42_f0/job_manifest.json',
           ROOT / 'paper/current/manifest.json']
receipt = {
    'schema': 'research4-model-selection-schematic-v1', 'status': 'PASS',
    'scope': 'Principal balanced five-source Suanzaoren repeated nested MC20–SVM analysis',
    'new_model_fits': 0, 'raw_or_frozen_data_modified': False,
    'physical_specimens': 300, 'specimens_per_source': 60, 'sources': 5,
    'outer_folds': 5, 'partition_seeds': [42, 20261031, 20261108],
    'outer_train': 240, 'outer_test': 60, 'inner_folds': 2,
    'outer_test_to_selection_edge': False, 'independent_specimens_not_multiplied_by_repeats': True,
    'pipeline': 'Spectral candidates → training-only processing → MC coefficient scores → top K → RBF SVM → retained-variable trace',
    'score': {'q_gj': 'abs(mean_beta_gj) / max(sd_beta_gj, epsilon)', 'sd_ddof': 1, 'S_j': 'sqrt(sum_g(q_gj**2)); g=1..5',
              'epsilon_caption_definition': '1e-12 * max(1, max_gj(sd_beta_gj))'},
    'drawing': {'width_inches': 6.5, 'height_inches': 5.6, 'min_label_font_points': 7,
                'dpi': 1200, 'outer_frame': True, 'white_background': True,
                'transparency': False, 'greek_math_style': 'upright with math roman', 'latin_math_variables': 'italic'},
    'primary_inner_selected_K_distinguished_from_prescribed_K': True,
    'sources': [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
    'outputs': outputs,
}
(BASE / 'authoring_sources/model_selection_figure_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'files': len(outputs), 'pixels': [7800, 6720], 'new_fits': 0}))
