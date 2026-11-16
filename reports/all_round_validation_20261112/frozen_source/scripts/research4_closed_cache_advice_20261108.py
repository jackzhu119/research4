"""Release clean file-cache pages of closed evidence; never change file bytes.

This is per-file POSIX advice, not global drop_caches. No process is signalled.
Only completed-supervisor outputs and the verified independent backfill are
eligible. The function can run inside a budgeted admission wait loop.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import stat
import time

OLD=Path('/workspace/research4-relaunch/data/research/species_relaunch_20261108_v2')
BACKFILL=Path('/workspace/research4-backfill-20261108-001')


def advise_closed():
    began=time.process_time();before=int(Path('/sys/fs/cgroup/memory.current').read_text())
    files=set();checked=0
    state=json.loads((OLD/'control/state.json').read_text())
    for job,record in state.get('jobs',{}).items():
        if record.get('status')!='completed':continue
        receipt=OLD/'control/receipts'/(job+'.json')
        if not receipt.is_file():continue
        # Receipt metadata is small. Source data are never read or rewritten.
        proof=json.loads(receipt.read_text())
        for name in proof.get('output_hashes',{}):
            p=Path(name)
            if p.is_absolute() and p.resolve().is_relative_to(OLD) and p.suffix in ('.json','.gz'):
                files.add(p)
    result=BACKFILL/'backfill_result.json'
    if result.exists() and json.loads(result.read_text()).get('status')=='DONE_REMOTE_BACKFILL_VERIFIED':
        files.update((BACKFILL/'archives').glob('closed-results.tar.gz.part*'))
        files.update(p for p in (BACKFILL/'.git/objects').glob('*/*') if p.is_file())
    advised_bytes=0;errors=0
    for p in sorted(files):
        if not p.exists() or p.is_symlink():continue
        st=p.stat()
        if not stat.S_ISREG(st.st_mode) or st.st_size<512*1024:continue
        fd=None
        try:
            fd=os.open(p,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
            os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            after=p.stat()
            if (st.st_ino,st.st_size,st.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
                raise RuntimeError('Closed evidence metadata changed during read-only cache advice')
            checked+=1;advised_bytes+=st.st_size
        except (OSError,ValueError):errors+=1
        finally:
            if fd is not None:os.close(fd)
    return {'status':'READ_ONLY_CLOSED_FILE_CACHE_ADVICE','files':checked,'file_bytes_advised':advised_bytes,
            'memory_before_bytes':before,'memory_after_bytes':int(Path('/sys/fs/cgroup/memory.current').read_text()),
            'actual_cpu_seconds':time.process_time()-began,'errors':errors,
            'file_bytes_or_mtime_modified':False,'signals_sent':False}


if __name__=='__main__':print(json.dumps(advise_closed()))
