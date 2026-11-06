"""Recompute completed derivative comparisons and export a scientific summary."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'reports/parallel-cloud-20261106/worker-b/B20261106-01/followup-diagnostics-recovery-v2'
OUTPUT = WORK / 'science/derived-summary-final-001'


def ref(path):
    return {'path': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def checked(binding):
    p = Path(binding['path'])
    p = p if p.is_absolute() else ROOT / p
    assert not p.is_symlink() and p.resolve().is_relative_to(ROOT)
    assert ref(p)['sha256'] == binding['sha256'], str(p)
    return p


def main():
    import numpy as np
    state = json.loads((WORK / 'state.json').read_text())
    ledger = json.loads((WORK / 'cpu_budget.json').read_text())
    assert state['invocations'][-1]['status'] == 'CLOSED'
    assert len(state['inherited_results']) == 34 and len(state['jobs']) == 22
    assert all(j['status'] == 'DONE_SEALED' for j in state['jobs'].values())
    assert all(p['status'] == 'EXITED' for p in ledger['phases'])
    assert len(state['results']) == 56 and not ledger.get('blocked_unknown_cpu_or_process_tree')
    jobs = ('water-control-comparison', 'acid-baseline-comparison', 'calycosin-baseline-comparison',
            'acid-grid3-comparison', 'acid-gridresponse-comparison', 'calycosin-grid3-comparison')
    rows, sources = [], []
    for job in jobs:
        binding = state['results'][job]
        result = json.loads(checked(binding).read_text()); sources.append(binding)
        assert result['status'] == 'AGGREGATE_COMPLETE'
        for row in result['paired_comparisons']:
            minus = json.loads(checked(row['minus_point_result']).read_text())
            plus = json.loads(checked(row['plus_point_result']).read_text())
            center = json.loads(checked(row['center_point_result']).read_text())
            assert minus['settings'] == plus['settings'] == center['settings']
            assert minus['original_frequency_result'] == plus['original_frequency_result'] == center['original_frequency_result']
            array = lambda point, name: np.load(checked(point['arrays'][name]), allow_pickle=False)
            q = array(plus, 'direction.npy'); h = row['step_bohr']
            assert minus['case'] == plus['case'] == center['case'] == row['case']
            assert minus['direction'] == plus['direction'] == row['direction']
            assert minus['sign'] == -1 and plus['sign'] == 1
            assert minus['step_bohr'] == plus['step_bohr'] == h
            assert center['step_bohr'] == center['sign'] == 0
            assert abs(np.linalg.norm(q) - 1) < 1e-12
            assert np.allclose(q, array(minus, 'direction.npy'), atol=1e-13, rtol=0)
            origin = array(center, 'geometry_bohr.npy')
            assert np.allclose(array(minus, 'geometry_bohr.npy'), origin - h * q, atol=2e-12, rtol=0)
            assert np.allclose(array(plus, 'geometry_bohr.npy'), origin + h * q, atol=2e-12, rtol=0)
            gradient_curvature = float(np.sum(q * (array(plus, 'gradient.npy') - array(minus, 'gradient.npy')) / (2 * h)))
            frequency = json.loads(checked(plus['original_frequency_result']).read_text())
            hessian = np.load(checked(frequency['arrays']['hessian_used.npy']), allow_pickle=False)
            cartesian = hessian.transpose(0, 2, 1, 3).reshape(q.size, q.size)
            expected_Hq = (cartesian @ q.ravel()).reshape(q.shape)
            assert np.allclose(expected_Hq, array(plus, 'predicted_Hq.npy'), atol=1e-13, rtol=0)
            analytic = float(np.sum(q * expected_Hq))
            energy_curvature = (plus['energy_hartree'] - 2 * center['energy_hartree'] + minus['energy_hartree']) / h ** 2
            assert abs(gradient_curvature - row['gradient_FD_curvature_Eh_per_Bohr2']) < 1e-11
            assert abs(analytic - row['analytic_directional_curvature_Eh_per_Bohr2']) < 1e-11
            assert abs(energy_curvature - row['energy_FD_curvature_Eh_per_Bohr2']) < 1e-11
            rows.append({'comparison_job': job, 'case': row['case'],
                         'direction_type': row['direction_type'], 'mode_index': row['mode_index'],
                         'translation_axis': row['translation_axis'], 'step_bohr': h,
                         'grid_level': row['grid_level'], 'gradient_grid_response': row['gradient_grid_response'],
                         'analytic_curvature_Eh_per_Bohr2': analytic,
                         'gradient_curvature_Eh_per_Bohr2': gradient_curvature,
                         'energy_curvature_Eh_per_Bohr2': energy_curvature})
    assert len(rows) == 21
    def select(job, mode, step=.005):
        return next(row for row in rows if row['comparison_job'] == job
                    and row['mode_index'] == mode and row['step_bohr'] == step)
    sampled = [select(job, mode) for job, mode in
               (('acid-baseline-comparison', 1), ('acid-baseline-comparison', 4),
                ('calycosin-baseline-comparison', 1), ('calycosin-baseline-comparison', 7))]
    for row in rows:
        if row['comparison_job'] in ('acid-baseline-comparison', 'calycosin-baseline-comparison') and row['analytic_curvature_Eh_per_Bohr2'] < 0:
            assert row['gradient_curvature_Eh_per_Bohr2'] > 0 and row['energy_curvature_Eh_per_Bohr2'] > 0
    base = select('acid-baseline-comparison', 1)
    response = select('acid-gridresponse-comparison', 1)
    dense = select('acid-grid3-comparison', 1)
    error = lambda row: abs(row['energy_curvature_Eh_per_Bohr2'] - row['gradient_curvature_Eh_per_Bohr2'])
    metrics = {
        'acid_mode1_energy_gradient_difference_without_response': error(base),
        'acid_mode1_energy_gradient_difference_with_response': error(response),
        'acid_mode1_energy_gradient_difference_reduction_factor': error(base) / error(response),
        'acid_mode1_response_gradient_curvature_change': response['gradient_curvature_Eh_per_Bohr2'] - base['gradient_curvature_Eh_per_Bohr2'],
        'acid_mode1_grid3_gradient_curvature_change': dense['gradient_curvature_Eh_per_Bohr2'] - base['gradient_curvature_Eh_per_Bohr2'],
        'acid_mode1_original_analytic_to_baseline_gradient_gap': base['gradient_curvature_Eh_per_Bohr2'] - base['analytic_curvature_Eh_per_Bohr2']}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with (OUTPUT / 'verified_directional_curvatures.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {
        'schema': 'research4-b-final-derivative-diagnostic-summary-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_script': ref(Path(__file__).resolve()),
        'manifest': ref(WORK / 'manifest_ROOT_AUTHORIZED.json'),
        'completed_state': ref(WORK / 'state.json'), 'completed_ledger': ref(WORK / 'cpu_budget.json'),
        'new_quantum_evaluations': 0, 'completed_logical_jobs': 56, 'fresh_point_jobs': 48,
        'independently_recomputed_paired_comparisons': len(rows), 'comparison_sources': sources,
        'sampled_original_negative_directions': sampled, 'metrics': metrics,
        'verified_findings_zh': [
            '原儿茶酸模式1/4、毛蕊异黄酮模式1/7的原解析方向曲率为负，但两个步长下独立能量与梯度曲率均为正。',
            '水的整体平移独立曲率接近零，原解析Hessian却给出明显负曲率。',
            '原儿茶酸加密网格后已测方向仍为正；开启梯度网格响应使能量/梯度曲率更接近，但本轮所测设置变化未消除原解析与独立曲率的差距，具体原因仍未定位。',
            '两个正频控制方向也存在曲率幅度偏差，因此不能用简单截断虚频修复这批解析结果。'],
        'hypotheses_zh': [
            '尚待进一步定位解析二阶导数或电子响应链条中不一致的具体来源。',
            '网格响应可解释部分能量/梯度有限差分差异，但目前证据不足以将全部问题归因于网格。'],
        'not_verified_zh': ['未计算完整有限差分Hessian，不能给出已纠正的全频率谱或稳定最低点证书。',
                            '未独立核验完整CPHF方程残差或电子态内部稳定性。',
                            'IR强度、实验峰归属及新的分类性能均未计算或验证。'],
        'method_qualifiers': ['Original analytic Hessian SCF tolerance1e-8; fresh diagnostic tolerance1e-10.',
                              'All displacements use original Cartesian-normalized directions and rebuild atom-centered grids.',
                              'Grid3 results are sensitivity comparisons against the original grid1 Hessian, not a grid3 analytic Hessian.',
                              'Only selected directions are tested; positive directional curvatures do not certify all-direction stability.',
                              'Single-sided energy decrease can reflect a nonzero first derivative; use symmetric curvature.'],
        'resources': {k: ledger[k] for k in ('actual_closed_cpu_seconds', 'conservative_debit_cpu_seconds',
                                            'budget_charged_cpu_seconds', 'remaining_cpu_seconds', 'cpu_cap_seconds', 'deadline_utc')},
        'original_results_modified': False}
    (OUTPUT / 'analysis_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(4); width = .24
    fig, ax = plt.subplots(figsize=(9.3, 4.8))
    for offset, field, color, label in (
        (-width, 'analytic_curvature_Eh_per_Bohr2', '#a63d40', 'Stored analytic Hessian'),
        (0, 'gradient_curvature_Eh_per_Bohr2', '#2878b5', 'Fresh gradient difference'),
        (width, 'energy_curvature_Eh_per_Bohr2', '#229e85', 'Fresh energy difference')):
        ax.bar(x + offset, [row[field] for row in sampled], width, label=label, color=color)
    ax.axhline(0, color='#333333', linewidth=.9)
    ax.set_xticks(x, ['Protocatechuic acid\nmode1', 'Protocatechuic acid\nmode4', 'Calycosin\nmode1', 'Calycosin\nmode7'])
    ax.set_ylabel('Directional curvature (Eh / Bohr$^2$)')
    ax.set_title('Sampled imaginary modes: stored and independent curvatures disagree')
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.grid(axis='y', alpha=.17); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.09, .025, 'Shown: h=0.005 Bohr; signs also reproduce at h=0.01. Fresh SCF tighter than original; grids rebuilt.\nSelected directions only: this is not a corrected full Hessian or a minimum certificate.', fontsize=8, color='#444444')
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(OUTPUT / 'directional_curvature_comparison.png', dpi=180)
    fig.savefig(OUTPUT / 'directional_curvature_comparison.pdf')
    plt.close(fig)
    print(json.dumps({'summary': str(OUTPUT / 'analysis_summary.json'), 'paired_comparisons_recomputed': len(rows),
                      'response_reduction_factor': metrics['acid_mode1_energy_gradient_difference_reduction_factor'],
                      'figure': str(OUTPUT / 'directional_curvature_comparison.png')}, indent=2))


if __name__ == '__main__':
    main()
