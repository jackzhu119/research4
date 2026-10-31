"""Set up the local runtime and execute explicitly labelled usable algorithm paths.

Exit 0 means all selected runnable paths passed, not that all original MATLAB
files are unchanged, scientifically validated or natively MATLAB-compatible.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


REPO=Path(__file__).resolve().parents[1]
QUICK_OCTAVE_CASES=[
    'compat_pls_original','compat_cars_uve_main','compat_uve_noise','compat_mcuve_0308',
    'patched_cars_error_scaling','patched_iriv_small_dimension','compat_libsvm_three_outputs',
    'compat_edge_fixture','compat_sg_filter','core_pls_cv','core_plslda','core_random_frog',
    'core_pca_kmeans','ready_ssa_smoke','ready_vectorized_bp',
]

ORIGINAL_BLOCKERS=[
    {'original':'MATLAB及专有工具箱','reason':'当前未安装或执行原生MATLAB；需要有效许可证和工具箱', 'usable_path':'Octave/Python环境，单独标注兼容及替代实现'},
    {'original':'newff/fitcecoc/templateSVM旧入口','reason':'MathWorks接口及完整依赖缺失','usable_path':'sklearn MLP/SVC 或 Linux LIBSVM；不是原newff复现'},
    {'original':'LSSVM.txt / trainlssvm','reason':'完整原工具箱缺失','usable_path':'独立实现的核LS-SVM回归'},
    {'original':'PCA.txt / my_pca','reason':'图像PCA依赖缺失','usable_path':'保存训练均值与载荷的图像PCA实现'},
    {'original':'UVE分类demo / carsplslda','reason':'分类筛选函数缺失','usable_path':'显式one-hot目标的Python选择器；不是原二分类PLS-LDA接口'},
    {'original':'纹理DOCX、syqpcx、部分PLS/CARS/PCA脚本','reason':'存在已核实的逻辑或形状问题，安装软件不会改变原逻辑','usable_path':'保留原件；运行注明来源和修改内容的修正版/替代工具'},
    {'original':'个人绝对路径、外部原始数据及中药真实标签','reason':'原文件未提供或样品对应关系未确认','usable_path':'随包示例/合成工程夹具；真实中药只验证读取'},
]
EXPECTED_PYTHON={'python_existing_unit_suite','python_example_pls','python_example_svr',
                 'python_example_svc','python_real_herb_reading','python_cnn_structure_migration'}
EXPECTED_REPLACEMENTS={'replacement_lssvm_regression','replacement_image_pca','replacement_glcm_texture',
                       'replacement_sobel_edges','replacement_sg_filter','replacement_bp_classifier',
                       'replacement_scan_row_adapter'}
EXPECTED_SELECTION={'mcuve_absolute_negative_effect','bounded_iriv_three_features',
                    'selector_sample_group_aggregation','cars_multiclass_not_ordinal',
                    'cars_style_training_only_example'}


def read_success_rows(path,expected_ids=None):
    rows=json.loads(Path(path).read_text())
    if not isinstance(rows,list) or not rows:
        raise ValueError(f'Expected nonempty result rows: {path}')
    for row in rows:
        if not isinstance(row,dict) or not row.get('id') or row.get('status')!='pass':
            raise ValueError(f"Runnable check failed: {row.get('id') if isinstance(row,dict) else row}")
        if row.get('stage')=='isolated_defect':
            raise ValueError('Defect reproduction cannot be counted as algorithm readiness')
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate result IDs')
    if expected_ids is not None and not set(expected_ids).issubset(ids):
        raise ValueError('Required algorithm checks are missing: '+', '.join(sorted(set(expected_ids)-set(ids))))
    return rows


def verify_cached_ssa(repo,octave,current_packages):
    evidence=Path(repo)/'reports/runtime/2026-10-30'
    rows=json.loads((evidence/'cases.json').read_text())
    row=next(x for x in rows if x['phase']=='ssa-full' and x['id']=='compat_ssa_full_budget_long')
    if row['status']!='pass':raise ValueError('Previous full SSA run did not pass')
    sources=json.loads((evidence/'source_hashes.json').read_text())
    matches=[x for x in sources if Path(x['archive']).name==Path(row['archive']).name]
    if len(matches)!=1:raise ValueError('Cached SSA archive is not uniquely identified')
    source=matches[0]
    if hashlib.sha256((Path(repo)/source['archive']).read_bytes()).hexdigest()!=source['source_sha256']:
        raise ValueError('SSA archive differs from cached full run')
    member=next(x for x in source['members'] if x['name']==row['source_member'])
    if member['sha256']!=row['source_sha256']:raise ValueError('SSA source member hash mismatch')
    manifest=json.loads((evidence/'evidence_sha256.json').read_text())
    for rel in [row['saved_log'],'harnesses/ssa-full/compat_ssa_full_budget_long/check.m',
                'cases.json','source_hashes.json','octave-environment.log']:
        entry=next(x for x in manifest if x['path']==rel)
        if hashlib.sha256((evidence/rel).read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('SSA cached log or harness hash mismatch')
    cp=subprocess.run([str(octave),'--eval','disp(version);'],capture_output=True,text=True,timeout=30)
    if cp.returncode:raise ValueError('Cannot probe current Octave version')
    version=next((x.strip() for x in cp.stdout.splitlines() if re.fullmatch(r'\d+\.\d+\.\d+',x.strip())),None)
    historical=(evidence/'octave-environment.log').read_text()
    if version is None or f'GNU Octave Version: {version} ' not in historical:
        raise ValueError('Octave version differs from cached full run')
    for name in ['statistics','signal','io','image','optim']:
        value=current_packages.get(name)
        pattern=rf'^\s*{re.escape(name)}\s+\*?\|\s*{re.escape(str(value))}\s*\|'
        if value is None or not re.search(pattern,historical,re.M):
            raise ValueError(f'Octave package {name} differs from cached full run')
    return {'id':row['id'],'status':'cached_verified','original_started_at':row['started_at_shanghai'],
            'original_duration_seconds':row['duration_seconds'],'source_archive':source['archive'],
            'source_sha256':row['source_sha256'],'octave_version':version,
            'package_versions':{k:current_packages[k] for k in ['statistics','signal','io','image','optim']},
            'evidence':str(evidence.relative_to(repo)),'scope':'previous full run, not reexecuted in this invocation'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['quick','full'],default='quick')
    parser.add_argument('--runtime-dir',type=Path,default=REPO/'.runtime')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--offline',action='store_true',help='Set up using already available runtimes/packages')
    parser.add_argument('--skip-setup',action='store_true')
    parser.add_argument('--rerun-slow',action='store_true',help='Actually rerun full 100-iteration SSA instead of verifying cached evidence')
    parser.add_argument('--diagnose-originals',action='store_true',help='Also execute unchanged legacy entry diagnostics; expected failures are listed separately')
    parser.add_argument('--require-originals',action='store_true',help='Exit 2 if original MATLAB scripts remain blocked, even when usable paths pass')
    args=parser.parse_args()
    if args.rerun_slow and args.profile!='full':parser.error('--rerun-slow requires --profile full')
    timestamp=datetime.now(ZoneInfo('Asia/Shanghai'))
    out=(args.output or REPO/'outputs/all-algorithms'/timestamp.strftime('%Y%m%dT%H%M%S')).resolve()
    if out.exists() and any(out.iterdir()):parser.error('Choose an empty/new output directory')
    out.mkdir(parents=True,exist_ok=True)
    runtime=args.runtime_dir.resolve();python=runtime/'bin/python';octave=runtime/'bin/octave'
    summary={'started_at_shanghai':timestamp.isoformat(),'profile':args.profile,'output':str(out),
             'selected_paths_passed':False,'original_files_all_unmodified_runnable':False,
             'matlab_native_execution_verified':False,'research_performance_verified':False,
             'original_blockers':ORIGINAL_BLOCKERS,'steps':[],'cached_cases':[],'case_groups':{}}
    def save():
        (out/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    def step(name,command,results=None,timeout=1200,check_rows=True,expected_ids=None):
        print(f'Running {name} ...',flush=True);t=time.monotonic();log=out/(name+'.log')
        with log.open('w') as f:
            cp=subprocess.run([str(x) for x in command],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        record={'name':name,'command':list(map(str,command)),'returncode':cp.returncode,
                'duration_seconds':round(time.monotonic()-t,4),'log':log.name}
        summary['steps'].append(record);save()
        if cp.returncode:raise RuntimeError(f'{name} failed; see {log}')
        if results is not None:
            rows=read_success_rows(results,expected_ids) if check_rows else json.loads(Path(results).read_text())
            summary['case_groups'][name]=rows;save()
        print(f'Completed {name} ({record["duration_seconds"]:.2f}s)',flush=True)
    exit_code=0
    try:
        if not args.skip_setup:
            command=[sys.executable,REPO/'scripts/setup_research_environment.py','--runtime-dir',runtime]
            if args.offline:command.append('--offline')
            step('setup',command,timeout=1800)
        if not python.exists() or not octave.exists():raise FileNotFoundError('Run setup first; Python/Octave wrappers are missing')
        step('environment',[python,REPO/'scripts/environment_doctor.py','--runtime-dir',runtime,'--output',out/'environment.json'],timeout=120)
        doctor=json.loads((out/'environment.json').read_text())
        if not doctor.get('open_source_ready'):raise RuntimeError('Required open-source environment is not ready')
        source_dir=out/'sources';inventory=source_dir/'source_inventory.json'
        step('prepare_sources',[python,REPO/'scripts/runtime/prepare_sources.py','--output',source_dir],timeout=120)
        step('python',[python,REPO/'scripts/runtime/run_python_checks.py','--inventory',inventory,'--output',out/'python'],out/'python/results.json',timeout=600,expected_ids=EXPECTED_PYTHON)
        step('replacements',[python,REPO/'scripts/runtime/run_replacement_checks.py','--inventory',inventory,'--output',out/'replacements'],out/'replacements/results.json',timeout=300,expected_ids=EXPECTED_REPLACEMENTS)
        step('selection',[python,REPO/'scripts/runtime/run_selection_checks.py','--inventory',inventory,'--output',out/'selection'],out/'selection/results.json',timeout=600,expected_ids=EXPECTED_SELECTION)
        selected=list(QUICK_OCTAVE_CASES)
        if args.profile=='full':selected.append('ready_uve_regression_demo')
        step('octave_ready',[python,REPO/'scripts/runtime/run_octave_checks.py','--phase','ready','--case',*selected,'--octave',octave,'--inventory',inventory,'--output',out/'octave_ready'],out/'octave_ready/results.json',timeout=900,expected_ids=selected)
        if args.profile=='full':
            cached=None
            if not args.rerun_slow:
                try:cached=verify_cached_ssa(REPO,octave,doctor.get('octave',{}).get('packages',{}))
                except (OSError,ValueError,KeyError,StopIteration,subprocess.SubprocessError) as exc:
                    summary['cache_miss_reason']=str(exc)
            if cached:summary['cached_cases'].append(cached);save()
            else:
                step('ssa_full',[python,REPO/'scripts/runtime/run_octave_checks.py','--phase','ssa-full','--octave',octave,'--inventory',inventory,'--output',out/'ssa_full'],out/'ssa_full/results.json',timeout=660,expected_ids={'compat_ssa_full_budget_long'})
        if args.diagnose_originals:
            step('original_diagnostics',[python,REPO/'scripts/runtime/run_octave_checks.py','--phase','first','--octave',octave,'--inventory',inventory,'--output',out/'original_diagnostics'],out/'original_diagnostics/results.json',check_rows=False,timeout=900)
        summary['selected_paths_passed']=True
        if args.require_originals:exit_code=2
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
        summary['error']=f'{type(exc).__name__}: {exc}';exit_code=1
    summary['exit_code']=exit_code;summary['finished_at_shanghai']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat();save()
    total=sum(len(rows) for name,rows in summary['case_groups'].items() if name!='original_diagnostics')
    lines=['# 统一运行结果','',f"选定可运行路径通过：{summary['selected_paths_passed']}。本次包含 {total} 个实时检查及 {len(summary['cached_cases'])} 个已验证历史完整运行。",'',
           '原始MATLAB文件全部未经修改运行：否。原生MATLAB已验证：否。中药科研性能已验证：否。','',
           '| 运行组 | 场景数 | 状态 |','| --- | ---: | --- |']
    for name,rows in summary['case_groups'].items():
        status='diagnostic_only' if name=='original_diagnostics' else ('pass' if all(x.get('status')=='pass' for x in rows) else 'fail')
        lines.append(f'| {name} | {len(rows)} | {status} |')
    lines+=['','原件阻塞项、具体命令、日志、适配说明和缓存日期见 `run_summary.json`。',
            '兼容修改和替代实现保持独立标注；示例/合成夹具成绩不代表中药结果。','']
    (out/'RUN_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'selected_paths_passed':summary['selected_paths_passed'],'live_cases':total,
                      'cached_cases':len(summary['cached_cases']),'original_files_all_unmodified_runnable':False,
                      'matlab_native_execution_verified':False,'summary':str(out/'run_summary.json'),'exit_code':exit_code},ensure_ascii=False,indent=2))
    return exit_code


if __name__=='__main__':sys.exit(main())
