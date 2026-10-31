"""Extract selected scientific source/data members; preserve source hashes.

Requires the system libarchive shared library for RAR. Executables, native MEX
binaries and installers are excluded. Extracted files belong under outputs/.
"""
import argparse
import ctypes as c
import ctypes.util
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=args.repo.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    lib=c.CDLL(ctypes.util.find_library('archive'))
    def bind(name,result,*types):
        fn=getattr(lib,name);fn.restype=result;fn.argtypes=list(types);return fn
    new=bind('archive_read_new',c.c_void_p)
    formats=bind('archive_read_support_format_all',c.c_int,c.c_void_p)
    filters=bind('archive_read_support_filter_all',c.c_int,c.c_void_p)
    open_file=bind('archive_read_open_filename',c.c_int,c.c_void_p,c.c_char_p,c.c_size_t)
    next_header=bind('archive_read_next_header',c.c_int,c.c_void_p,c.POINTER(c.c_void_p))
    member_name=bind('archive_entry_pathname',c.c_char_p,c.c_void_p)
    member_size=bind('archive_entry_size',c.c_int64,c.c_void_p)
    member_type=bind('archive_entry_filetype',c.c_uint,c.c_void_p)
    read=bind('archive_read_data',c.c_ssize_t,c.c_void_p,c.c_void_p,c.c_size_t)
    skip=bind('archive_read_data_skip',c.c_int,c.c_void_p)
    error=bind('archive_error_string',c.c_char_p,c.c_void_p)
    free=bind('archive_read_free',c.c_int,c.c_void_p)
    records=[]
    code=repo/'references/paper-materials/code'
    paths=sorted(code.glob('*.rar'))+sorted(code.glob('*.zip'))
    for number,path in enumerate(paths,1):
        folder=out/f'archive_{number:02d}';folder.mkdir(exist_ok=True)
        record=dict(archive=str(path.relative_to(repo)),source_kind='archive',folder=str(folder),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),members=[])
        archive=new();formats(archive);filters(archive)
        try:
            if open_file(archive,str(path).encode(),65536)<0:raise RuntimeError(error(archive))
            while True:
                entry=c.c_void_p();status=next_header(archive,c.byref(entry))
                if status==1:break
                if status< -20:raise RuntimeError(error(archive))
                raw=member_name(entry) or b'unnamed'
                for encoding in ['utf-8','gb18030','latin-1']:
                    try:name=raw.decode(encoding);break
                    except UnicodeError:pass
                rel=Path(name.replace('\\','/'));size=member_size(entry)
                if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe member path')
                selected=rel.suffix.lower() in {'.m','.mat','.xls','.xlsx','.txt','.py','.c','.cpp','.h'}
                if not selected or member_type(entry)!=0o100000 or not 0<=size<=100*1024**2:
                    skip(archive);continue
                target=folder/rel;target.parent.mkdir(parents=True,exist_ok=True)
                digest=hashlib.sha256();total=0;buf=c.create_string_buffer(65536)
                with target.open('wb') as f:
                    while True:
                        n=read(archive,buf,len(buf))
                        if n==0:break
                        if n<0:raise RuntimeError(error(archive))
                        data=buf.raw[:n];f.write(data);digest.update(data);total+=n
                if total!=size:raise ValueError('Member byte count mismatch')
                record['members'].append(dict(name=name,path=str(target),size_bytes=total,sha256=digest.hexdigest()))
        finally:free(archive)
        records.append(record)
    folder=out/'standalone';folder.mkdir(exist_ok=True)
    record=dict(archive='standalone_texts',source_kind='txt_as_m',folder=str(folder),members=[])
    for path in sorted(code.glob('*.txt')):
        target=folder/(path.stem+'.m');target.write_bytes(path.read_bytes())
        record['members'].append(dict(name=target.name,path=str(target),size_bytes=target.stat().st_size,
                                      sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source_file=str(path.relative_to(repo))))
    records.append(record)
    folder=out/'docx_code';folder.mkdir(exist_ok=True)
    record=dict(archive='docx_code',source_kind='docx_paragraph_text',folder=str(folder),members=[])
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    for original,name in [('特征提取.docx','Texture.m'),('边缘检测（可用）.docx','edge_detection.m')]:
        path=code/original
        with zipfile.ZipFile(path) as z:
            tree=ET.fromstring(z.read('word/document.xml'))
            text='\n'.join(''.join(n.text or '' for n in p.findall('.//w:t',ns)) for p in tree.findall('.//w:p',ns))+'\n'
        target=folder/name;target.write_text(text)
        record['members'].append(dict(name=name,path=str(target),size_bytes=target.stat().st_size,
                                      sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source_document=str(path.relative_to(repo))))
    records.append(record)
    (out/'source_inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(f'Prepared {len(records)} source groups, {sum(len(r["members"]) for r in records)} selected members')


if __name__=='__main__':main()
