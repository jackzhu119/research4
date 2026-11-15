#!/usr/bin/env python3
"""Restyle the five final figures from certified values; never fit a model.

Frozen evidence remains byte-for-byte unchanged. The hashes below identify the
accepted input tables used by the earlier scientific verification. All output
figures have a complete outer border and four visible spines per panel.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'data/active_workbooks.json').exists())
BASE = ROOT / 'paper/revision_20261115'
EVIDENCE = BASE / 'Supplementary_Data_S1'
FIGURES = BASE / 'figures'
CURRENT = ROOT / 'paper/current'
MANIFEST = json.loads((CURRENT / 'manifest.json').read_text())
ALLOW = {r['path']: r for r in MANIFEST['files']}
INPUTS = []
OUTPUTS = []
HASHES = {
    'derived_checks/spectra_means_current.csv': 'b315a0a398a618eab91bf04281bb9bf0e75d596b3270720685e9d2cca98bef71',
    'derived_checks/feature_budget_all_fixedK_metrics.csv': '76da3a2c286203f2d4334bdb030443a7d68c11f0efebb2a60047caaf3a72064e',
    'derived_checks/feature_budget_paired_bootstrap.csv': 'cde6928c5b48cf6a81fa4980dc507619b64ab359494e2f798f114b39a3bb0066',
    'derived_checks/feature_modality_composition.csv': 'f1e91f2fe40c1c43313586e92e3f3c54bf798af2e1a52e9ae9d883beefe18933',
    'derived_checks/adaptation_controls.csv': 'bf2183acd99f2fb5f7f96a1ffd55ea32582676495d91c7dd98804018d6386cbf',
    'molecular_sources/acid_d2f61d35/ir_modes.csv': 'fc7a44f7ad39d95fa18d66bf20c4a0f1bafaf20ac4682ed4a1bf0828451b6473',
    'molecular_sources/acid_f626f77f/ir_modes.csv': '35d17ce6d7278e529976cf19d9d421cce836ee7523ab6ecff85050ccfa4f80de',
}


def record(path, expected):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == expected, f'Changed certified source: {path}'
    if not any(r['path'] == str(path.relative_to(ROOT)) for r in INPUTS):
        INPUTS.append({'path': str(path.relative_to(ROOT)), 'sha256': digest, 'bytes': len(data)})
    return path


def evidence_csv(name):
    return pd.read_csv(record(EVIDENCE / name, HASHES[name]))


def current_file(name):
    return record(CURRENT / name, ALLOW[name]['sha256'])


BLUE = '#0072B2'
ORANGE = '#D55E00'
GREEN = '#009E73'
PURPLE = '#8B5A9F'
GREY = '#6B7280'
INK = '#263238'
GRID = '#E6E9ED'
BUDGETS = [31, 62, 123, 184, 245]
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.3,
    'axes.labelsize': 8.6, 'axes.titlesize': 9.3,
    'axes.titleweight': 'bold', 'axes.labelcolor': INK,
    'axes.edgecolor': INK, 'axes.linewidth': .75,
    'axes.spines.top': True, 'axes.spines.right': True,
    'xtick.labelsize': 7.8, 'ytick.labelsize': 7.8,
    'xtick.color': INK, 'ytick.color': INK,
    'xtick.major.width': .7, 'ytick.major.width': .7,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'legend.fontsize': 7.6, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'lines.linewidth': 1.25, 'lines.markersize': 4.0,
})


def panel(ax, title, grid='y'):
    ax.set_title(title, loc='left', pad=7)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.55)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.75)
        spine.set_color(INK)
    ax.tick_params(direction='out', top=False, right=False)


def save(fig, name):
    # A figure-coordinate rectangle encloses labels, legends and all panels.
    fig.add_artist(Rectangle((.004, .006), .992, .988,
                             transform=fig.transFigure, fill=False,
                             edgecolor=INK, linewidth=.8, clip_on=False))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    # Axes.get_tightbbox excludes automatic tick labels outside the shown range.
    boxes = [(ax.get_title(), ax.get_tightbbox(renderer)) for ax in fig.axes]
    boxes += [('legend', legend.get_window_extent(renderer)) for legend in fig.legends]
    boxes += [(t.get_text(), t.get_window_extent(renderer)) for t in fig.texts]
    for label, box in boxes:
        assert box.x0 >= bounds.x0 and box.y0 >= bounds.y0 and box.x1 <= bounds.x1 and box.y1 <= bounds.y1, (name, label, tuple(box.bounds))
    for extension in ['png', 'tiff', 'pdf', 'eps', 'svg']:
        path = FIGURES / f'{name}.{extension}'
        if extension == 'tiff':
            # Use the same rendered pixels as PNG, with no alpha channel.
            with Image.open(FIGURES / f'{name}.png') as png:
                assert png.getextrema()[-1] == (255, 255), 'PNG background must be opaque'
                png.convert('RGB').save(path, compression='tiff_lzw', dpi=(1200, 1200))
        else:
            fig.savefig(path, dpi=1200, facecolor='white')
        row = {'path': str(path.relative_to(BASE)),
               'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'bytes': path.stat().st_size}
        if extension in ['png', 'tiff']:
            with Image.open(path) as image:
                row['pixels'] = list(image.size)
                row['dpi'] = [float(value) for value in image.info['dpi']]
                row['color_mode'] = image.mode
                if extension == 'tiff':
                    assert image.mode == 'RGB' and image.info['compression'] == 'tiff_lzw'
                    assert tuple(image.info['dpi']) == (1200, 1200)
                    row['compression'] = image.info['compression']
                    row['opacity'] = 'RGB, no alpha channel'
        if extension == 'eps':
            assert path.read_bytes().startswith(b'%!PS-Adobe-3.0 EPSF-3.0')
            row['eps_header_valid'] = True
            row['transparency'] = 'No partial transparency; original tints rendered as solid RGB'
        OUTPUTS.append(row)
    plt.close(fig)


def spectra_figure():
    d = evidence_csv('derived_checks/spectra_means_current.csv')
    assert len(d) == 5 * 2 * 612 and d.specimens.eq(60).all()
    groups = [
        ('ZY03', 'Inner Mongolia', BLUE, '-'),
        ('ZY04', 'Lingshi, Shanxi', ORANGE, (0, (5, 1.8))),
        ('ZY05', 'Bozhou, Anhui', GREEN, (0, (2.5, 1.6))),
        ('ZY06', 'Shijiazhuang, Hebei', PURPLE, (0, (5, 1.5, 1.1, 1.5))),
        ('ZY07', 'Shanxi', '#A57923', (0, (1, 1.2))),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.55))
    fig.subplots_adjust(left=.092, right=.975, bottom=.29, top=.90, wspace=.31)
    for ax, mod, title, ylabel in zip(
        axs, ['alpha', 'n'], ['A  Absorption', 'B  Refractive index'],
        ['Absorption (exported scale)', 'Refractive index'],
    ):
        for group, label, colour, style in groups:
            t = d[(d.group == group) & (d.modality == mod)].sort_values('frequency_thz')
            assert len(t) == 612
            ax.plot(t.frequency_thz, t['mean'], color=colour, ls=style, lw=1.18, label=label)
        ax.set(xlabel='Frequency (THz)', ylabel=ylabel, xlim=(.1, 1.5))
        panel(ax, title)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .035), ncol=3,
               handlelength=2.8, columnspacing=1.2, labelspacing=.7, fontsize=7.6)
    save(fig, 'fig1_spectra_current')


def budget_figure():
    metrics = evidence_csv('derived_checks/feature_budget_all_fixedK_metrics.csv')
    paired = evidence_csv('derived_checks/feature_budget_paired_bootstrap.csv').sort_values('k')
    accepted = pd.read_csv(current_file('results/v4_330/metrics.csv'))
    specs = [
        ('alpha_snv20', 'SNV absorption', BLUE, 'o', '-'),
        ('concat_both_snv20', 'Joint SNV candidate pool', ORANGE, 's', '-'),
        ('alpha_raw20', 'Raw absorption', GREEN, '^', (0, (4, 1.4))),
        ('n_raw20', 'Raw refractive index', GREY, 'D', (0, (2, 1.3))),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.55), gridspec_kw={'width_ratios': [1.2, 1]})
    fig.subplots_adjust(left=.085, right=.97, bottom=.285, top=.90, wspace=.42)
    for family, label, colour, marker, style in specs:
        d = metrics[metrics.family == family].sort_values('k')
        assert d.k.tolist() == BUDGETS
        for _, row in d.iterrows():
            match = accepted[(accepted.method == family) & (accepted.selection_scope == 'fixed_k') & (accepted.k.astype(str) == str(int(row.k)))]
            assert len(match) == 1 and np.isclose(match.iloc[0].accuracy, row.accuracy, atol=1e-12)
        axs[0].plot(d.k, d.accuracy * 100, color=colour, marker=marker, ls=style,
                    lw=1.3, ms=4.2, mec='white', mew=.4, label=label)
    axs[0].set(xlabel='Retained predictors, K', ylabel='Mean accuracy (%)', xticks=BUDGETS,
               ylim=(65, 97), yticks=[65, 75, 85, 95])
    panel(axs[0], 'A  Classification accuracy')
    y = paired.joint_minus_alpha_pp.to_numpy()
    lo = paired.cluster_bootstrap_95_low_pp.to_numpy()
    hi = paired.cluster_bootstrap_95_high_pp.to_numpy()
    axs[1].errorbar(paired.k, y, yerr=np.vstack([y-lo, hi-y]), color=ORANGE, fmt='o',
                    ms=4.5, capsize=3.2, capthick=1.0, lw=1.15)
    axs[1].axhline(0, color=GREY, ls=(0, (4, 2)), lw=.9)
    axs[1].set(xlabel='Retained predictors, K', ylabel='Joint pool − absorption (pp)',
               xticks=BUDGETS, ylim=(-4.5, 11), yticks=[-4, 0, 4, 8])
    panel(axs[1], 'B  Paired accuracy difference')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .035), ncol=2,
               handlelength=2.8, columnspacing=1.8, labelspacing=.7)
    save(fig, 'fig2_feature_budget')


def modality_figure():
    d = evidence_csv('derived_checks/feature_modality_composition.csv').sort_values('k')
    assert d.k.tolist() == BUDGETS
    assert d.total_feature_occurrences.tolist() == [465, 930, 1845, 2760, 3675]
    assert d.n_occurrences.tolist() == [0, 0, 0, 14, 119]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.4), gridspec_kw={'width_ratios': [1.1, 1]})
    fig.subplots_adjust(left=.09, right=.97, bottom=.255, top=.89, wspace=.35)
    x = np.arange(len(d)); n_pct = d.n_fraction.to_numpy() * 100
    axs[0].bar(x, 100-n_pct, color=BLUE, width=.72, label='Absorption')
    axs[0].bar(x, n_pct, bottom=100-n_pct, color=ORANGE, width=.72, label='Refractive index')
    for i, (_, r) in enumerate(d.iterrows()):
        axs[0].text(i, 102.2, f'{int(r.n_occurrences)}/{int(r.total_feature_occurrences)}',
                    ha='center', va='bottom', fontsize=7.15)
    axs[0].set(xticks=x, xticklabels=BUDGETS, xlabel='Retained predictors, K',
               ylabel='Feature occurrences (%)', ylim=(0, 114), yticks=[0, 25, 50, 75, 100])
    panel(axs[0], 'A  Retained modality composition')
    axs[1].plot(d.k, d.mean_alpha_subset_jaccard, color=GREEN, marker='o', ms=4.5,
                lw=1.45, mec='white', mew=.5)
    axs[1].set(xlabel='Retained predictors, K', ylabel='Absorption-subset Jaccard',
               xticks=BUDGETS, ylim=(-.025, 1), yticks=[0, .2, .4, .6, .8, 1])
    panel(axs[1], 'B  Absorption-subset overlap')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .035), ncol=2,
               handlelength=2.2, columnspacing=2.0)
    save(fig, 'fig3_selected_modalities')


def adaptation_figure():
    d = evidence_csv('derived_checks/adaptation_controls.csv')
    assert len(d) == 6
    expected = [.8, .8388888888888889, .8377777777777777, .8011111111111111, .7333333333333333, .8377777777777777]
    for (_, row), value in zip(d.iterrows(), expected):
        assert abs(row.accuracy-value) < 1e-12
        summary = json.loads(current_file(row.summary_path).read_text())
        key = 'curves_macro_over_jobs' if row['round'] == 'B1' else 'curves_macro_over_job_draws'
        curves = pd.DataFrame(summary[key])
        q = curves[(curves.experiment == 'E03') & (curves.method == row.method) & (curves.k == row.k_per_class)]
        if pd.notna(row.family):
            q = q[q.family == row.family]
        assert len(q) == 1 and abs(float(q.iloc[0].accuracy)-row.accuracy) < 1e-12
    labels = [
        'Source only', 'True support labels:\nfull refit',
        'Frozen scaler and\nkernel scale', 'Unlabeled support\nstatistics',
        'Cyclically incorrect\nsupport labels', 'Normalized total\nloss mass',
    ]
    colors = [GREY, BLUE, GREEN, GREY, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    fig.subplots_adjust(left=.285, right=.97, bottom=.145, top=.88)
    y = np.arange(6)[::-1]
    ax.axvline(80, color=GREY, lw=.95, ls=(0, (4, 2)), zorder=1)
    for i, (_, row) in enumerate(d.iterrows()):
        yy = y[i]; value = row.accuracy_percent
        line_colour = tuple(.65 * c + .35 for c in to_rgb(colors[i]))
        ax.plot([80, value], [yy, yy], color=line_colour, lw=1.5, zorder=2)
        ax.scatter(value, yy, s=34, color=colors[i], edgecolor='white', linewidth=.65, zorder=3)
        left = value < 80
        ax.text(value + (-.32 if left else .32), yy, f'{value:.2f}%',
                va='center', ha='right' if left else 'left', fontsize=8.5,
                color=colors[i], fontweight='bold' if i in [1, 2] else 'normal',
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .6})
    ax.set(yticks=y, yticklabels=labels, xlim=(70, 87.2), ylim=(-.55, 5.55),
           xticks=[72, 76, 80, 84], xlabel='Internal query accuracy (%)')
    panel(ax, 'Support-label and preprocessing controls', grid='x')
    ax.tick_params(axis='y', length=0, pad=7, labelsize=8.5)
    save(fig, 'fig4_adaptation_controls')


def molecular_figure():
    direct = evidence_csv('molecular_sources/acid_d2f61d35/ir_modes.csv')
    ri = evidence_csv('molecular_sources/acid_f626f77f/ir_modes.csv')
    for d in [direct, ri]:
        assert len(d) == 45 and d.signed_frequency_THz.gt(0).all() and d.paper_ready.all() and not d.imaginary.any()
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.55), gridspec_kw={'width_ratios': [1, 1.5]})
    fig.subplots_adjust(left=.085, right=.97, bottom=.24, top=.89, wspace=.37)
    for ax in axs:
        ax.axvspan(.1, 1.5, color='#E1E9F1', label='Analysis interval')
        ax.vlines(direct.signed_frequency_THz, 0, direct.IR_intensity_km_mol,
                  color=BLUE, lw=1.6, label='Direct integrals')
        ax.vlines(ri.signed_frequency_THz, 0, ri.IR_intensity_km_mol,
                  color=ORANGE, lw=.95, linestyles=(0, (3, 2)), label='RI-J')
        ax.set(xlabel='Frequency (THz)', ylabel='Predicted IR intensity (km/mol)')
    axs[0].set(xlim=(0, 4), ylim=(0, 2.85), xticks=[0, 1, 2, 3, 4])
    axs[0].text(.8, 1.4, '0.1–1.5 THz\nanalysis\ninterval', ha='center', va='center',
                fontsize=7.2, color='#4B5967', linespacing=1.25)
    axs[0].annotate('Lowest mode:\n2.21865–2.21940 THz', xy=(2.2194, 2.145),
                    xytext=(2.5, 2.53), ha='center', fontsize=6.9,
                    arrowprops={'arrowstyle': '-', 'color': '#4B5967', 'lw': .7})
    axs[1].set(xlim=(0, 115), ylim=(0, 330), xticks=[0, 20, 40, 60, 80, 100])
    panel(axs[0], 'A  Low-frequency region')
    panel(axs[1], 'B  Complete 45-mode prediction')
    axs[1].legend(loc='upper right', fontsize=7.1, handlelength=2.2)
    fig.text(.5, .053, 'Isolated protocatechuic acid · PBE/def2-SVP · no D3\nNo fitted band broadening',
             ha='center', va='center', fontsize=7.5, color='#4B5967', linespacing=1.45)
    save(fig, 'Fig_S_quantum_accepted_PCA')


if __name__ == '__main__':
    FIGURES.mkdir(exist_ok=True)
    spectra_figure()
    budget_figure()
    modality_figure()
    adaptation_figure()
    molecular_figure()
    receipt = {
        'schema': 'research4-final-figure-style-v1', 'status': 'PASS',
        'purpose': 'User-requested visual refinement and complete rectangular frames',
        'source_manifest_sha256': hashlib.sha256((CURRENT / 'manifest.json').read_bytes()).hexdigest(),
        'width_inches': 6.5, 'png_export_dpi': 1200, 'tiff_export_dpi': 1200,
        'raster_formats': ['PNG', 'TIFF (LZW, RGB, opaque)'],
        'vector_formats': ['PDF', 'EPS', 'SVG'],
        'font_embedding': {'PDF': 'TrueType (pdf.fonttype 42)', 'EPS': 'TrueType (ps.fonttype 42)', 'SVG': 'Editable text'},
        'eps_transparency': 'All partial-alpha tints replaced by equivalent solid RGB colours',
        'artwork_resolution_basis': 'SAGE line artwork minimum 800 dpi; exported at 1200 dpi',
        'frame': {'outer_figure_rectangle': True, 'four_spines_each_panel': True, 'linewidth_points': .8},
        'plot_values_unchanged': True, 'new_classifier_training': False,
        'new_quantum_calculations': False, 'frozen_evidence_modified': False,
        'text_extents_inside_figure': True, 'inputs': INPUTS, 'outputs': OUTPUTS,
    }
    (BASE / 'authoring_sources/figure_style_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'figures': 5, 'output_files': len(OUTPUTS),
                      'inputs_checked': len(INPUTS)}, ensure_ascii=False))
