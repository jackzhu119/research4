#!/usr/bin/env python3
"""Upload verified binary shards via authenticated Git Data API, then verify bytes."""
import base64
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = 'jackzhu119/research4'
BRANCH = 'research/spinosin-complete-results-20261115-data'

def api(path, payload=None, method=None):
    cmd = ['gh', 'api', 'repos/' + REPO + '/' + path]
    if method: cmd += ['--method', method]
    if payload is not None: cmd += ['--input', '-']
    p = subprocess.run(cmd, input=json.dumps(payload) if payload is not None else None,
                       text=True, capture_output=True, timeout=180)
    if p.returncode: raise RuntimeError(p.stderr[-1200:])
    return json.loads(p.stdout)

def main():
    latest = json.loads((ROOT / 'complete_export_latest.json').read_text())
    out = Path(latest['output_directory'])
    journal_path = out / 'git_upload_journal.json'
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
    def upload(part):
        data = (out / part['name']).read_bytes()
        assert len(data) == part['bytes'] and hashlib.sha256(data).hexdigest() == part['sha256']
        expected = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        old = journal.get(part['name'])
        if old and old.get('verified') and old.get('git_blob_sha') == expected:
            return old
        blob = api('git/blobs', {'encoding': 'base64', 'content': base64.b64encode(data).decode()})
        assert blob['sha'] == expected
        check = api('git/blobs/' + expected)
        assert check['encoding'] == 'base64'
        downloaded = base64.b64decode(check['content'])
        assert len(downloaded) == part['bytes'] and hashlib.sha256(downloaded).hexdigest() == part['sha256']
        return {**part, 'git_blob_sha': expected, 'verified': True}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(upload, latest['parts']):
            journal[result['name']] = result
            journal_path.write_text(json.dumps(journal, indent=2) + '\n')
            print('verified', result['name'], result['sha256'], flush=True)
    entries = [{'path': p['name'], 'mode': '100644', 'type': 'blob', 'sha': journal[p['name']]['git_blob_sha']}
               for p in latest['parts']]
    readme = ('# 斯皮诺素完整结果快照（2026-11-15）\n\n'
              '本数据分支包含完整结果包分片、逐文件清单和校验值。GitHub Release 上传主机鉴权不可用，故使用仓库独立数据分支保存完整二进制数据。\n\n'
              '10条合格能量/核梯度记录对应8个不同几何点；尚无收敛极小值、Hessian或IR结果。原始结果、密度矩阵、检查点、日志、代码、输入和一致性SQLite账本共293个文件均包含在压缩包中。\n\n'
              '下载本分支全部文件后，在同一目录执行：\n\n'
              '```sh\ncat ' + latest['archive'] + '.part-* > ' + latest['archive'] + '\n'
              'sha256sum -c SHA256SUMS\ntar -xzf ' + latest['archive'] + '\n```\n\n'
              'PARTS.json提供分片顺序和SHA256，MANIFEST.json提供包内全部文件SHA256。每个上传分片已从GitHub重新下载并核对SHA256。\n\n'
              '科研报告与运行代码：[研究分支](https://github.com/' + REPO + '/tree/research/spinosin-conformers-parallel-900000-20261112/reports/spinosin_conformers_parallel_20261112)。\n')
    for name in ['MANIFEST.json', 'PARTS.json', 'SHA256SUMS', 'README_zh.md']:
        entries.append({'path': name, 'mode': '100644', 'type': 'blob', 'content': (out / name).read_text()})
    entries.append({'path': 'README.md', 'mode': '100644', 'type': 'blob', 'content': readme})
    tree = api('git/trees', {'tree': entries})
    commit = api('git/commits', {'message': 'Publish complete verified spinosin result snapshot, 10 records / 8 geometries',
                               'tree': tree['sha'], 'parents': []})
    api('git/refs', {'ref': 'refs/heads/' + BRANCH, 'sha': commit['sha']})
    remote = api('git/trees/' + tree['sha'])
    actual = {item['path']: item['sha'] for item in remote['tree']}
    assert all(actual[p['name']] == journal[p['name']]['git_blob_sha'] for p in latest['parts'])
    result = {'published_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'repository': REPO, 'branch': BRANCH, 'commit': commit['sha'],
              'url': 'https://github.com/' + REPO + '/tree/' + BRANCH,
              'archive_sha256': latest['archive_sha256'], 'archive_bytes': latest['archive_bytes'],
              'part_count': len(latest['parts']), 'all_remote_part_bytes_SHA256_verified': True,
              'manifest_file_count': latest['verified_file_count'], 'source_host_asset_upload_unavailable': True}
    (ROOT / 'full_results_publication_20261115.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__': main()
