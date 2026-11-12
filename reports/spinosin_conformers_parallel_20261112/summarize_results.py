#!/usr/bin/env python3
"""Summarize actual molecular jobs, distinguishing attempts and duplicate basins."""
import argparse
import datetime
import json
from pathlib import Path
import sqlite3


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def summarize(root):
    status = read(root / 'status.json', {})
    jobs = read(root / 'jobs.json', {}).get('jobs', [])
    rows = []
    for job in jobs:
        directory = root / 'jobs' / job['id']
        result = read(directory / 'result.json', {})
        progress = read(directory / 'progress.json', {})
        rows.append({'id': job['id'], 'conformer': job['conformer'],
                     'complete': result.get('complete') is True and result.get('status') == 'complete',
                     'status': result.get('status', progress.get('stage', 'not_started')),
                     'task_outcome': result.get('task_outcome'),
                     'profile': result.get('profile', progress.get('profile', job['profile'])),
                     'energy_hartree': result.get('energy_hartree'),
                     'mode_count': result.get('mode_count'),
                     'basin_identity_status': result.get('basin_identity_status'),
                     'selected_case': result.get('selected_case'),
                     'duplicate_source': result.get('duplicate_source'),
                     'independent_experimental_sample': False})
    attempts = []
    dbpath = root / 'queue.sqlite3'
    if dbpath.exists():
        db = sqlite3.connect(f'file:{dbpath}?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        attempts = [dict(r) for r in db.execute('SELECT job_id,state,exit_code,verified_cpu,charged_cpu,account_source FROM attempts')]
        db.close()
    value = {'generated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'state': status.get('state', 'not_started'),
             'cpu_budget_seconds': status.get('cpu_budget_seconds', 900000),
             'total_cpu_observed_or_conservative': status.get('total_cpu_observed_or_conservative', 0),
             'completed_jobs': sum(r['complete'] for r in rows),
             'duplicate_basin_jobs': sum(r['complete'] and r['task_outcome'] == 'basin_duplicate' for r in rows),
             'independent_experimental_samples': 0,
             'jobs': rows, 'attempts': attempts,
             'scope': 'Isolated molecular models, not independent material samples or proof of solid-state THz band assignment.',
             'unique_final_minima_count': None,
             'unique_minima_note': 'Requires final atom-mapped geometry and energy comparison across local and other-host results.'}
    (root / 'results_summary.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 斯皮诺素补充计算进度', '',
             f"状态：{value['state']}。已完成任务 {value['completed_jobs']}/{len(rows)}；其中复用同一势能盆证据的任务 {value['duplicate_basin_jobs']} 个。",
             f"累计观测/保守记账 CPU：{value['total_cpu_observed_or_conservative']:.2f} 秒；总授权上限 {value['cpu_budget_seconds']} 秒。", '',
             '| 起始构象 | 状态 | 实际方法网格 | 完整振动模式数 |', '|---|---|---|---|']
    for row in rows:
        lines.append(f"| {row['conformer']} | {row['status']} | {row['profile']} | {row['mode_count'] if row['mode_count'] is not None else '尚未验证'} |")
    lines += ['', '进程尝试、构象起点和已完成分子计算分别计数。重复势能盆不算新的独立极小值；全部构象都不代表独立实验样本。尚未完成的 Hessian/IR 不报告为合格结果。', '']
    (root / 'RESULTS.md').write_text('\n'.join(lines))
    return {k: value[k] for k in ('state', 'completed_jobs', 'duplicate_basin_jobs', 'total_cpu_observed_or_conservative')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root.resolve()), ensure_ascii=False))
