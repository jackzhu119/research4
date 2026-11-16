from pathlib import Path
import subprocess, json, hashlib, os, argparse

REPO=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser(description='Export explicitly selected immutable Git source objects.')
parser.add_argument('--output-dir',type=Path,default=REPO/'outputs/revision_20261116/evidence')
parser.add_argument('--review-manifest',type=Path,default=None)
args=parser.parse_args();OUT=args.output_dir
FOCUS='5594d9c3326fcd607ee72fb7fc93895047de8f8a'
BACKGROUND='93a607c8ce206a5bbbff22538233bdd0929bd04d'
def git(*args,**kwargs):
    return subprocess.check_output(['git','-C',str(REPO),*args],**kwargs)
def entries(sha,root):
    result={}
    for line in git('ls-tree','-r',sha,root,text=True).splitlines():
        meta,path=line.split('\t',1);mode,kind,oid=meta.split()
        if kind=='blob':result[path]=oid
    return result
selected={}
tree=entries(FOCUS,'reports/focused_followup_20261116')
prefix='reports/focused_followup_20261116/'
for path,oid in tree.items():
    rel=path.removeprefix(prefix)
    include=(rel.startswith(('e1/','e3/','e5/','e6/')) and not path.endswith(('.joblib','.npz')))
    include|=rel.startswith('e2/') and (('/mc/' in path and path.endswith(('result.json','importance.npz','paired_draw_ids.json.gz'))) or any(v in rel for v in ['v4_','fixed_']))
    include|=rel in ['e0/summary.json','e0/common_contexts.json','e0/method_metrics.csv','e0/paired_effects.csv','e4/cost_audit.json','e4/benchmarks/context0_baseline_svm/inner_candidate_means.json']
    if include:selected[(FOCUS,path)]=oid
for path,oid in entries(FOCUS,'experiments/focused_followup_20261116').items():selected[(FOCUS,path)]=oid
for path in ['source/study_worker.py','source/protocol.json','reports/source_accuracy_summary.csv']:
    selected[(BACKGROUND,path)]=entries(BACKGROUND,path)[path]
# Fetch selected blobs in a single protocol exchange rather than requesting
# every object size and lazily downloading the entire experimental tree.
oids=sorted(set(selected.values()))
env=os.environ.copy();env['GIT_NO_LAZY_FETCH']='1'
checks=subprocess.run(['git','-C',str(REPO),'cat-file','--batch-check=%(objectname) %(objecttype)'],input='\n'.join(oids)+'\n',text=True,stdout=subprocess.PIPE,check=True,env=env)
missing=[line.split()[0] for line in checks.stdout.splitlines() if line.endswith(' missing')]
print('Selected files:',len(selected),'missing blobs to fetch:',len(missing),flush=True)
if missing:
    subprocess.run(['git','-C',str(REPO),'fetch','--no-tags','--no-write-fetch-head','--recurse-submodules=no','--filter=blob:none','origin','--stdin'],input='\n'.join(missing)+'\n',text=True,check=True,stdout=subprocess.PIPE)
manifest=[]
for (sha,path),oid in selected.items():
    data=git('cat-file','blob',oid)
    destination=OUT/sha[:8]/path
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
    manifest.append({'source_commit':sha,'source_path':path,'git_blob':oid,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'local_path':str(destination)})
review=json.loads(args.review_manifest.read_text()) if args.review_manifest else {'sources':[]}
matches=0
hash_differences=[]
for source in review['sources']:
    for record in source.get('files',[]):
        if 'path' not in record:
            continue
        key=(source.get('commit'),record['path'])
        if key in selected:
            row=next(x for x in manifest if x['source_commit']==key[0] and x['source_path']==key[1])
            if row['sha256']==record['sha256']:
                matches+=1
            else:
                raw=Path(row['local_path']).read_bytes()
                hash_differences.append({'source_commit':key[0],'path':key[1],
                    'review_sha256':record['sha256'],'git_blob_sha256':row['sha256'],
                    'matches_without_utf8_bom':hashlib.sha256(raw.removeprefix(b'\xef\xbb\xbf')).hexdigest()==record['sha256'],
                    'matches_normalized_newlines':hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==record['sha256']})
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'SOURCE_MANIFEST.json').write_text(json.dumps({'sources':manifest,'matches_to_uploaded_review':matches,'review_hash_differences':hash_differences},ensure_ascii=False,indent=2)+'\n')
print('Exported',len(manifest),'files;',sum(x['bytes'] for x in manifest),'bytes;',matches,'independently fetched files match the review hashes',flush=True)
print('Review hash differences:',json.dumps(hash_differences),flush=True)
