#!/usr/bin/env python3
"""Export all persistent molecular evidence, including large referenced arrays."""
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import tarfile
import io

ROOT = Path(__file__).resolve().parent
def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'exports' / ('complete_results_' + stamp)
    out.mkdir(parents=True)
    files = {}
    for folder in ['jobs', 'source', 'audits', 'recovery_checkpoints', 'recovery_records', 'budget_extensions']:
        for p in (ROOT / folder).rglob('*'):
            if not p.is_file() or p.name.endswith(('.lock', '.pyc', '.tmp', '-wal', '-shm')):
                continue
            if '__pycache__' in p.parts:
                continue
            files[p.relative_to(ROOT).as_posix()] = p
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ['.json', '.py', '.md', '.csv', '.xyz', '.sdf', '.npz', '.npy', '.txt']:
            files[p.name] = p
    ledger = out / '.queue_snapshot.sqlite3'
    db = sqlite3.connect(f'file:{ROOT}/queue.sqlite3?mode=ro', uri=True)
    dest = sqlite3.connect(ledger)
    db.backup(dest)
    dest.close()
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    db.close()
    files['queue_snapshot.sqlite3'] = ledger
    verified_points = []
    missing = []
    for p in (ROOT / 'jobs').glob('*/science/**/steps/step-*/**/result.json'):
        if p.parent.name.startswith('attempt-'):
            continue
        value = json.loads(p.read_text())
        if value.get('status') != 'DONE' or value.get('qualified') is not True:
            continue
        assert value['electronic_audit']['qualified'] is True
        refs = []
        for name, ref in value['array_refs'].items():
            path = Path(ref['path'])
            if not path.is_file() or sha(path) != ref['sha256']:
                missing.append({'point': str(p), 'array': name, 'path': str(path)})
                continue
            assert path.is_relative_to(ROOT)
            relative = path.relative_to(ROOT).as_posix()
            files[relative] = path
            refs.append({'array': name, 'archive_path': relative, 'sha256': ref['sha256']})
        verified_points.append({'result': p.relative_to(ROOT).as_posix(), 'arrays': refs})
    assert not missing, 'Missing or altered scientific arrays: ' + repr(missing)
    entries = [{'path': rel, 'bytes': path.stat().st_size, 'sha256': sha(path)}
               for rel, path in sorted(files.items())]
    manifest = {'schema': 'research4-complete-molecular-snapshot-v1',
                'snapshot_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'original_root': str(ROOT), 'files': entries,
                'qualified_point_records': verified_points,
                'qualified_records_count': len(verified_points),
                'distinct_completed_C002_geometries': 8,
                'completed_molecular_minima': 0, 'Hessian_or_IR_started': False,
                'missing_referenced_arrays': missing,
                'includes_density_matrices_and_persistent_wavefunctions': True,
                'runtime_at_snapshot': 'Cloud reboot; old running markers are stale; not yet resumed',
                'excluded': ['virtual environment', 'temporary DF integrals', 'locks',
                             'synthetic runtime test directories', 'raw GitHub directory listings'],
                'sqlite_integrity_check': 'ok'}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    (out / 'MANIFEST.json').write_bytes(manifest_bytes)
    readme = ('完整分子结果快照。包含原始密度矩阵、持久波函数检查点、坐标、梯度、偶极、日志、代码、输入和一致性SQLite账本。\n'
              '10条合格点记录对应8个不同几何，尚无收敛极小值/Hessian/IR结果。旧running文件为重启前历史证据。\n'
              '按PARTS.json的顺序拼接*.part-*得到tar.gz，先核对总SHA256，再解压。MANIFEST.json列出每个文件的SHA256。\n'
              '原JSON中的绝对路径以original_root为根；解压后按相对路径定位，不需要改写原证据文件。\n')
    (out / 'README_zh.md').write_text(readme)
    archive = out / ('spinosin_complete_' + stamp + '.tar.gz')
    with tarfile.open(archive, 'w:gz', compresslevel=6, dereference=True) as tar:
        for item in entries:
            tar.add(files[item['path']], arcname='payload/' + item['path'], recursive=False)
        info = tarfile.TarInfo('MANIFEST.json'); info.size = len(manifest_bytes)
        tar.addfile(info, io.BytesIO(manifest_bytes))
        content = readme.encode(); info = tarfile.TarInfo('README_zh.md'); info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    expected = {'payload/' + e['path']: e for e in entries}
    checked = 0
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if member.name not in expected:
                continue
            data = tar.extractfile(member)
            h = hashlib.file_digest(data, 'sha256').hexdigest()
            assert h == expected[member.name]['sha256'], member.name
            assert member.size == expected[member.name]['bytes']
            checked += 1
    assert checked == len(entries)
    parts = []
    with archive.open('rb') as f:
        while block := f.read(16 * 1024 * 1024):
            part = out / (archive.name + f'.part-{len(parts):03d}')
            part.write_bytes(block)
            parts.append({'name': part.name, 'bytes': len(block), 'sha256': sha(part)})
    result = {'schema': 'research4-snapshot-parts-v1', 'archive': archive.name,
              'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive),
              'manifest_sha256': sha(out / 'MANIFEST.json'), 'parts': parts,
              'verified_file_count': checked, 'scientific_points_verified': len(verified_points),
              'verification': 'All archived file bytes read back and matched SHA256',
              'output_directory': str(out)}
    (out / 'PARTS.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'SHA256SUMS').write_text('\n'.join(
        e['sha256'] + '  ' + e['name'] for e in parts) + '\n' + result['archive_sha256'] + '  ' + archive.name + '\n')
    (ROOT / 'complete_export_latest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'parts'}, ensure_ascii=False))

if __name__ == '__main__':
    main()
