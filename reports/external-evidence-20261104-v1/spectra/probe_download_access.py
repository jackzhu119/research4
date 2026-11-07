"""Bounded access probe; keep proxy/TLS policy and never overwrite a report.

The original 2026-11-04 result was executed inline with these four GET targets;
this saved replay entry is for a new output path, not evidence of a second run.
No retry, proxy bypass, quantum kernels, training or publication occurs here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise FileExistsError('Preserve the prior probe output')
 targets=[('ginseng_k4_metadata','https://data.mendeley.com/datasets/k4pjjjmgnr/1','metadata'),('maize_zenodo_metadata','https://zenodo.org/records/19150796','metadata'),('dryad_wheat_frequency','https://datadryad.org/downloads/file_stream/108445','numeric frequency vector'),('cambridge_thz_tutorial_metadata','https://www.repository.cam.ac.uk/items/e8f8d203-30a8-4658-8fce-85422f5a71c8','metadata')]
 rows=[]
 for name,url,kind in targets:
  row={'id':name,'url':url,'requested_object':kind,'attempted_at_utc':datetime.now(timezone.utc).isoformat(),'numeric_bytes_downloaded':False}
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'Research4-Data-Feasibility-Audit/1.0'})
   with urllib.request.urlopen(req,timeout=6) as f:
    body=f.read(2*1024*1024+1)
    row.update(http_status=f.status,bytes_read=len(body),content_type=f.headers.get('Content-Type'),status='RESPONSE_READ_NOT_PERSISTED_OR_PARSED_AS_TRAINING_DATA')
  except Exception as exc:row.update(status='BLOCKED_DOWNLOAD',error_type=type(exc).__name__,error_summary=str(exc)[:280])
  rows.append(row)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as h:json.dump({'attempts':rows,'quantum_kernel_calls':0,'model_fits':0,'proxy_bypass':False},h,indent=2);h.write('\n')
 print(str(a.output))
if __name__=='__main__':main()
