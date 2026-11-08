"""Append-only display-count correction; all checks and conclusions unchanged."""
from pathlib import Path
import hashlib,json

OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not(OUT/'final_review.json').exists()
    p=OUT/'review.json';assert sha(p)=='8e8f7b3b08534e3cbd7f1fd354475d8c2614a6fc443ca8963f88fabfb2f15b3e'
    d=json.loads(p.read_text());assert d['checks_count']==4244 and len(d['checks'])==4266 and all(x['passed']for x in d['checks'])
    d['checks_count']=len(d['checks'])
    d['append_only_count_correction']={'supersedes_count_field_in_sha256':sha(p),'original_index_sha256':sha(OUT/'index.json'),
        'reason':'Twenty-two final source-stability checks were appended after the display count had been captured. All4266 checks passed; no scientific values, source files or findings changed.',
        'correction_source_sha256':sha(__file__)}
    (OUT/'final_review.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    paths=[Path(__file__),OUT/'review.json',OUT/'index.json',OUT/'final_review.json',OUT/'README.md']
    (OUT/'final_review_index.json').write_text(json.dumps({'status':'SEALED_FINAL_INTAKE_REVIEW_WITH_CORRECTED_COUNT','files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size}for p in paths]},indent=2)+'\n')
    print(json.dumps({'checks':d['checks_count'],'final_review_sha256':sha(OUT/'final_review.json'),'final_index_sha256':sha(OUT/'final_review_index.json')}))

if __name__=='__main__':main()
