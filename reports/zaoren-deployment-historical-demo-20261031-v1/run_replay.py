from pathlib import Path
import subprocess,sys,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
for number in range(1,8):
    target=OUT/f'ZY{number:02d}'
    command=[str(ROOT/'.runtime/bin/python'),str(ROOT/'scripts/predict_zaoren_historical_oof.py'),'--input',str(ROOT/f'data/processed/zaoren_20261031/样本{number}_编号版.xlsx'),'--output',str(target)]
    start=datetime.now(timezone.utc).isoformat()
    with (OUT/f'ZY{number:02d}.log').open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    with (OUT/'command_events.jsonl').open('a') as f:
        f.write(json.dumps({'start_utc':start,'completed_utc':datetime.now(timezone.utc).isoformat(),'argv':command,'returncode':result.returncode},ensure_ascii=False)+'\n')
    if result.returncode:raise SystemExit(result.returncode)
print('Seven original workbooks replayed with original held-out models; no fitting.')
