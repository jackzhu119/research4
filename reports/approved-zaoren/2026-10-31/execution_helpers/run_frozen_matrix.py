"""Run the approved internal-CV matrix; scientific inputs and budgets are fixed."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import time

ROOT = Path('/workspace/research4')
BASE = ROOT / 'outputs/approved-zaoren-20261031'
OUT = BASE / 'experiments'
CODE = ['scripts/run_zaoren_classification.py', 'src/herb_spectra/classification.py',
        'src/herb_spectra/benchmark.py', 'src/herb_spectra/selection.py',
        'src/herb_spectra/stability.py', 'src/herb_spectra/validation.py',
        'src/herb_spectra/io.py', 'src/herb_spectra/__init__.py']
NOW = lambda: datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

if OUT.exists():
    raise FileExistsError('Use new experimental output folders; previous runs are preserved')
OUT.mkdir(parents=True)
(OUT / 'logs').mkdir()
snapshot = OUT / 'execution_code'
code_hashes = {}
for name in CODE:
    target = snapshot / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, target)
    code_hashes[name] = SHA(target)
protocol = {'frozen_at_shanghai': NOW(), 'scope': 'user-declared independent acid-jujube samples; anonymous origin-group internal CV, not held-out batches or named geographical origins',
            'data_version': '2026-10-31-zaoren-numbered-v1', 'candidate_interval': [.1, 1.5],
            'interval_status': 'predeclared engineering reference interval, not validated optimal physical band',
            'model_seed': 42, 'split_seed': 20261031, 'outer_splits': 5, 'inner_splits': 2,
            'cars_internal_cv': 2, 'selection_scoring': 'f1_macro', 'thread_limit_per_process': 1,
            'maximum_parallel_processes': 2, 'external_preregistration': False,
            'source_code_sha256': code_hashes,
            'dependency_versions': {p: importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','joblib','openpyxl']},
            'cpu_max': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
            'memory_max': Path('/sys/fs/cgroup/memory.max').read_text().strip()}
write(OUT / 'frozen_protocol.json', protocol)
candidate = BASE / 'datasets/candidate_0p1_1p5.npz'
full = BASE / 'data-preparation/dataset.npz'
jobs = [
    ('alpha_primary', candidate, ['alpha'], ['dummy','svc','plsda','mcuve20_svc','cars_svc']),
    ('modality_controls', candidate, ['n','concat'], ['svc','plsda']),
    ('stability_controls', candidate, ['alpha'], ['mcuve100_svc','stable_svc']),
    ('full_export_sensitivity', full, ['alpha'], ['svc','plsda']),
]
env = os.environ.copy()
env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')

def check_code():
    if any(SHA(ROOT / name) != value for name, value in code_hashes.items()):
        raise RuntimeError('Execution source changed after freezing; preserve current results and review before more runs')


def run(job):
    name, data, representations, methods = job
    check_code()
    command = [str(ROOT / '.runtime/bin/python'), 'scripts/run_zaoren_classification.py',
               '--input', str(data), '--output', str(OUT / name), '--representations', *representations,
               '--methods', *methods, '--outer-splits','5','--inner-splits','2','--cars-cv','2',
               '--seed','42','--split-seed','20261031','--identity-scope',
               'user-declared one ZY record per independent sample (2026-10-31); anonymous origin groups; batches unknown']
    record = {'job': name, 'started_at_shanghai': NOW(), 'command_argv': command,
              'input_sha256': SHA(data), 'source_code_sha256': code_hashes,
              'log': str((OUT / 'logs' / (name + '.log')).relative_to(ROOT))}
    write(OUT / (name + '_execution.json'), record)
    print('Starting ' + name, flush=True)
    started = time.monotonic()
    with (OUT / 'logs' / (name + '.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, ended_at_shanghai=NOW(), seconds=time.monotonic()-started,
                  code_unchanged_after_run=all(SHA(ROOT / p)==h for p,h in code_hashes.items()))
    summary = OUT / name / 'summary.json'
    if summary.exists():
        data = json.loads(summary.read_text())
        record['experiment_status'] = data.get('status')
        record['cases'] = [{'representation':r['representation'],'method':r['method'],'status':r['status']} for r in data.get('results',[])]
    write(OUT / (name + '_execution.json'), record)
    print(json.dumps({'finished':name,'seconds':round(record['seconds'],3),'exit_code':result.returncode,
                      'status':record.get('experiment_status')}, ensure_ascii=False), flush=True)
    check_code()
    return record

records = [run(jobs[0])]
with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(run, job) for job in jobs[1:]]
    for future in as_completed(futures):
        records.append(future.result())
        write(OUT / 'runs_manifest.json', {'protocol':'frozen_protocol.json','runs':sorted(records,key=lambda r:r['job'])})
write(OUT / 'runs_manifest.json', {'protocol':'frozen_protocol.json','runs':sorted(records,key=lambda r:r['job'])})
print('Frozen matrix finished.', flush=True)
