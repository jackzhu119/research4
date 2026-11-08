"""Stream closed, hash-sealed research evidence into bounded Git transport parts.

This independent repository does not write to the running research worktree.
The 600 CPU-second ceiling uses the unused 1900-second preparation allocation;
the frozen training supervisor, grant and source remain unchanged.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import resource
import shutil
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
BRANCH = 'research/species-results-backfill-20261108'
PREFIX = 'data/research/species_relaunch_20261108_v2'
PART_BYTES = 32 * 1024**2
MAX_CPU = 600.
START = time.monotonic()


def need(test, message):
    if not test:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def atomic(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def cpu():
    return sum(u.ru_utime + u.ru_stime for u in
               (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))


def budget():
    need(cpu() < MAX_CPU-10, 'Backfill CPU ceiling reached; keep partial evidence, do not alter training')
    need(time.monotonic()-START < 3600, 'Backfill one-hour wall ceiling reached')
    need(shutil.disk_usage(HERE).free > 2*1024**3, 'Backfill disk floor reached')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    need(not p.is_absolute() and '..' not in p.parts and str(p)==name, 'Unsafe relative path')
    return p


class SealedReader:
    def __init__(self, path, expected):
        self.f = path.open('rb'); self.h = hashlib.sha256(); self.size = 0
        self.expected = expected
    def read(self, n):
        block = self.f.read(n); self.h.update(block); self.size += len(block)
        return block
    def finish(self):
        need(self.f.read(1)==b'', 'Source grew during archive')
        self.f.close()
        need(self.size==self.expected['bytes'] and self.h.hexdigest()==self.expected['sha256'],
             'Immutable source size/hash differs from closure seal')


class PartsWriter:
    def __init__(self):
        self.parts=[];self.current=None;self.position=0;self.n=0;self.current_size=0
    def tell(self):return self.position
    def _finish_part(self):
        if self.current is not None:
            self.current.close()
            final=self.path.with_name(self.path.name.removesuffix('.partial'))
            self.path.rename(final)
            self.parts.append({'path':str(final.relative_to(HERE)), 'bytes':self.current_size,
                               'sha256':self.hasher.hexdigest()})
            self.current=None
    def write(self, data):
        remaining=memoryview(data); length=len(data)
        while remaining:
            if self.current is None:
                self.n+=1;self.path=HERE/'archives'/('closed-results.tar.gz.part%03d.partial'%self.n)
                self.current=self.path.open('xb');self.hasher=hashlib.sha256();self.current_size=0
            take=min(len(remaining), PART_BYTES-self.current_size);block=remaining[:take]
            self.current.write(block);self.hasher.update(block);self.current_size+=take
            self.position+=take;remaining=remaining[take:]
            if self.current_size==PART_BYTES:self._finish_part()
        return length
    def flush(self):
        if self.current is not None:self.current.flush()
    def close(self):self._finish_part()


def checked_selection(source, point):
    root=source/PREFIX
    state=read(root/'control/state.json')
    job_id='new_grant_backup_'+point
    record=state['jobs'].get(job_id,{})
    need(record.get('status')=='completed', 'Chosen backup has not actually closed')
    receipt_path=root/'control/receipts'/(job_id+'.json')
    need(digest(receipt_path)==record['receipt_sha256'], 'Supervisor receipt seal differs')
    supervisor_receipt=read(receipt_path)
    backup_path=root/'backups'/point/'receipt.json'
    need(supervisor_receipt['output_hashes'][str(backup_path)]==digest(backup_path),
         'Closed backup output seal differs')
    backup=read(backup_path)
    need(backup['status']=='DONE' and backup.get('verified_remote_commit'), 'No verified original remote anchor')
    files={}
    for name, info in backup.get('not_backed_up',{}).items():
        safe_relative(name);p=(source/name).resolve()
        need(p.is_relative_to(root) and p.is_file() and not (source/name).is_symlink(), 'Unapproved source path')
        need(p.stat().st_size==info['bytes'], 'Closed file size changed')
        need(not any(x in p.name for x in ['.partial','.tmp','.lock']), 'Unclosed filename')
        need(p.suffix not in ['.npz','.npy','.chk','.hdf','.hdf5','.xlsx'], 'Raw/large input excluded')
        need(name!=PREFIX+'/control/state.json', 'Live ledger excluded')
        files[name]={'path':name,'sha256':info['sha256'],'bytes':info['bytes'],
                     'reason_not_previously_backed_up':info.get('reason')}
    provenance=[receipt_path,backup_path,root/'backups'/point/'complete.json',
                root/'protocol.json',root/'input_lock.json',root/'jobs.json',
                root/'train_only_split.csv',root/'user_authorization_NEW_90000.json',
                root/'derived_source_feature_audit.json',root/'preparation_cpu.json']
    for p in provenance:
        name=str(p.relative_to(source))
        files[name]={'path':name,'sha256':digest(p),'bytes':p.stat().st_size,
                     'reason_not_previously_backed_up':'provenance_anchor_copy'}
    summary_receipt=root/'control/receipts'/('new_grant_snapshot_'+point+'.json')
    summary_record=state['jobs'].get('new_grant_snapshot_'+point,{})
    need(summary_record.get('status')=='completed' and
         summary_record['receipt_sha256']==digest(summary_receipt), 'Summary supervisor closure differs')
    sr=read(summary_receipt)
    md=root/'snapshots'/point/'BACKGROUND_RESULTS.md'
    need(sr['output_hashes'][str(md)]==digest(md), 'Readable summary seal differs')
    shutil.copyfile(md,HERE/'evidence/LATEST_BACKGROUND_RESULTS.md')
    shutil.copyfile(backup_path,HERE/'evidence/source_backup_receipt.json')
    shutil.copyfile(receipt_path,HERE/'evidence/source_supervisor_receipt.json')
    shutil.copyfile(root/'protocol.json',HERE/'evidence/source_protocol.json')
    anchor={'status':'CLOSED_SOURCE_SELECTION_VERIFIED_NOT_ALL_FUTURE_JOBS',
            'observed_at_utc':datetime.now(timezone.utc).isoformat(), 'backup_point':point,
            'source_branch':'research/species-relaunch-90000-20261108',
            'source_verified_remote_commit':backup['verified_remote_commit'],
            'source_supervisor_receipt_sha256':digest(receipt_path),
            'source_backup_receipt_sha256':digest(backup_path),
            'source_job_completed_status':record['status'],
            'source_live_ledger_modified':False,'training_interrupted':False,
            'selected_original_files':len(backup.get('not_backed_up',{})),
            'selected_original_bytes':sum(x['bytes'] for x in backup.get('not_backed_up',{}).values())}
    write_new(HERE/'evidence/source_completion_anchor.json',anchor)
    return sorted(files.values(),key=lambda row:row['path']),anchor


def archive(source, point):
    files,anchor=checked_selection(source,point)
    state={'status':'ARCHIVING','source_point':point,'expected_files':len(files),
           'expected_bytes':sum(x['bytes'] for x in files),'completed_files':0,'completed_bytes':0}
    writer=PartsWriter()
    with gzip.GzipFile(filename='',fileobj=writer,mode='wb',compresslevel=1,mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
            for i,item in enumerate(files,1):
                budget();p=source/item['path'];reader=SealedReader(p,item)
                info=tarfile.TarInfo(item['path']);info.size=item['bytes'];info.mode=0o600
                info.uid=0;info.gid=0;info.mtime=0;info.uname='';info.gname=''
                tar.addfile(info,reader);reader.finish()
                state.update(completed_files=i,completed_bytes=state['completed_bytes']+item['bytes'],
                             cpu_seconds=cpu(),wall_seconds=time.monotonic()-START,
                             updated_at_utc=datetime.now(timezone.utc).isoformat())
                if i%25==0:atomic(HERE/'archive_progress.json',state)
    writer.close()
    need(all(p['bytes']<=PART_BYTES for p in writer.parts),'Part limit exceeded')
    manifest={'schema':'research4-closed-results-backfill-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
              'source':anchor,'transport_branch':BRANCH,'compression':'one continuous tar.gz stream, gzip level1; concatenate parts in manifest order',
              'max_part_bytes':PART_BYTES,'files':files,'parts':writer.parts,
              'source_bytes':sum(x['bytes'] for x in files),'transport_bytes':sum(x['bytes'] for x in writer.parts),
              'source_file_count':len(files),'no_inputs_or_live_models_modified':True,
              'preparation_reserved_cpu_cap':1900,'this_backfill_cpu_cap':MAX_CPU,
              'frozen_original_preparation_cpu_seconds':34.610667,
              'new_training_CPU_allocation_unchanged':88000,'new_controller_CPU_allocation_unchanged':100}
    write_new(HERE/'manifest.json',manifest)
    state.update(status='DONE_ARCHIVE_NOT_YET_REMOTE',parts=len(writer.parts),
                 transport_bytes=manifest['transport_bytes'],cpu_seconds=cpu())
    atomic(HERE/'archive_progress.json',state)
    return manifest


def command(argv, timeout=180):
    budget()
    env={k:v for k,v in os.environ.items() if k not in
         ['GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE','GIT_OBJECT_DIRECTORY','GIT_ALTERNATE_OBJECT_DIRECTORIES','GIT_COMMON_DIR']}
    remaining=max(1,int(MAX_CPU-cpu()-10))
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU,(remaining,remaining))
    r=subprocess.run(argv,cwd=HERE,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,timeout=timeout,preexec_fn=limits)
    need(r.returncode==0,'Bounded subprocess failed: '+argv[0]+' '+argv[1])
    budget()
    return r.stdout


def publish(manifest):
    need(command(['git','branch','--show-current']).decode().strip()==BRANCH,'Independent backup branch required')
    need(not command(['git','ls-remote','origin','refs/heads/'+BRANCH]).strip(),'Backup branch already exists; do not overwrite')
    command([sys.executable,'-B','restore_backfill.py','--manifest','manifest.json','--verify-only',
             '--report','evidence/local_restore_verification.json'],timeout=600)
    need(read(HERE/'evidence/local_restore_verification.json')['status']=='PASS', 'Restore verification failed')
    fixed=['README.md','manifest.json','create_backfill.py','restore_backfill.py','test_restore_backfill.py']
    fixed += [str(p.relative_to(HERE)) for p in sorted((HERE/'evidence').glob('*.json'))]
    fixed += ['evidence/LATEST_BACKGROUND_RESULTS.md']
    parts=[p['path'] for p in manifest['parts']]
    commits=[]
    for i in range(0,len(parts),2):
        batch=parts[i:i+2];paths=(fixed if i==0 else [])+batch
        need(not command(['git','diff','--cached','--name-only']).strip(),'Unknown index content')
        command(['git','add','--',*paths])
        command(['git','commit','-m','research: backfill sealed species results volumes %d-%d'%(i+1,min(i+2,len(parts)))])
        commit=command(['git','rev-parse','HEAD']).decode().strip()
        command(['git','push','origin',commit+':refs/heads/'+BRANCH],timeout=240)
        remote=command(['git','ls-remote','origin','refs/heads/'+BRANCH]).decode().split()
        need(len(remote)==2 and remote[0]==commit,'Remote commit not confirmed')
        commits.append({'sha':commit,'parts':batch,'remote_verified':True,
                        'at_utc':datetime.now(timezone.utc).isoformat()})
        atomic(HERE/'publish_progress.json',{'status':'PARTIAL' if i+2<len(parts) else 'DONE',
               'commits':commits,'parts_uploaded':min(i+2,len(parts)),'parts_total':len(parts),
               'cpu_seconds':cpu(),'wall_seconds':time.monotonic()-START})
    # Verify every remote commit tree contains exactly our expected blob object.
    for p in manifest['parts']:
        need(command(['git','hash-object','--',p['path']]).decode().strip()==
             command(['git','rev-parse','HEAD:'+p['path']]).decode().strip(),'Committed part blob differs')
    return commits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--point',required=True)
    args=parser.parse_args();os.nice(10)
    resource.setrlimit(resource.RLIMIT_CPU,(590,600))
    subprocess.run(['ionice','-c','3','-p',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        manifest=archive(args.source.resolve(),args.point)
        need((HERE/'restore_backfill.py').exists() and (HERE/'test_restore_backfill.py').exists(), 'Verified restore tool required')
        commits=publish(manifest)
        result={'status':'DONE_REMOTE_BACKFILL_VERIFIED','branch':BRANCH,'final_commit':commits[-1]['sha'],
                'source_files':manifest['source_file_count'],'source_bytes':manifest['source_bytes'],
                'transport_bytes':manifest['transport_bytes'],'parts':len(manifest['parts']),
                'cpu_seconds':cpu(),'wall_seconds':time.monotonic()-START,'training_processes_signalled':False,
                'original_git_worktree_or_index_modified':False,'budget_source':'unused1900CPU preparation reserve; no training budget extension',
                'finished_utc':datetime.now(timezone.utc).isoformat()}
        atomic(HERE/'backfill_result.json',result)
        command(['git','add','--','backfill_result.json']);command(['git','commit','-m','research: record verified closed-result backfill receipt'])
        final=command(['git','rev-parse','HEAD']).decode().strip()
        command(['git','push','origin',final+':refs/heads/'+BRANCH],timeout=120)
        need(command(['git','ls-remote','origin','refs/heads/'+BRANCH]).decode().split()[0]==final,'Final receipt push not verified')
        print(json.dumps({**result,'receipt_commit':final}),flush=True)
    except BaseException as error:
        atomic(HERE/'backfill_failure.json',{'status':'PARTIAL_NOT_SCIENTIFIC_FAILURE','error_type':type(error).__name__,
               'error':str(error),'cpu_seconds':cpu(),'wall_seconds':time.monotonic()-START,
               'original_training_unmodified':True,'at_utc':datetime.now(timezone.utc).isoformat()})
        raise


if __name__=='__main__':main()
