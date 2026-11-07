"""Append-only correction: group-signal result is not pidfd-signal evidence."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
OUT=HERE/'correction-v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if OUT.exists():raise ValueError('Correction already sealed')
    original=json.loads((HERE/'audit.json').read_text())
    row=original['parent_row']
    assert row['owned_child_pidfd_containment_requested'] is True
    assert all(x['sent'] is False for x in row['identity_checked_signal_attempts'])
    assert row['exit_code']==0 and row['hard_killed'] is True and row['monitor_failed'] is True
    OUT.mkdir()
    correction=dict(original)
    correction.pop('no_signal_sent',None)
    correction.update({'schema':'research4-independent-failed-unit-forensic-audit-v2',
        'corrected_at_utc':datetime.now(timezone.utc).isoformat(),
        'supersedes_signal_wording_in':{'audit_sha256':sha(HERE/'audit.json'),'index_sha256':sha(HERE/'index.json')},
        'correction_source_sha256':sha(__file__),
        'group_KILL_sent':False,'owned_child_pidfd_containment_requested':True,
        'pidfd_KILL_impact':'UNKNOWN; a request was recorded successful; actual wait4 exit0 shows the child was not reaped as SIGKILL-terminated. Do not infer that no signal request occurred.',
        'recommendation':'Eligible for explicit root hash-bound forensic release of only these five complete saved units, conditional on full stock-vs-segment numerical verification. Parent remains failed; group KILL was not sent, but pidfd containment KILL was requested. Do not relabel healthy or assert no signal request.'})
    (OUT/'audit.json').write_text(json.dumps(correction,indent=2)+'\n')
    (OUT/'README.md').write_text('This receipt corrects the earlier claim “no signal sent”. The identity-checked process-group KILL was not sent, but owned_child_pidfd_containment_requested=true records a pidfd KILL request. Actual wait4 exit0 differs from SIGKILL termination; signal-request impact remains UNKNOWN. Original failed supervisor flags remain unchanged. Saved five-record/six-array SHA and MF/proof checks remain valid. Only a separate explicit root release may qualify their reuse, and final full-matrix/mode equivalence remains necessary. Prior receipt and index are preserved for audit.\n')
    index={'status':'APPEND_ONLY_SIGNAL_WORDING_CORRECTION_V2','superseded_index_sha256':sha(HERE/'index.json'),
           'source':{'path':str(Path(__file__)),'sha256':sha(__file__)},
           'files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(OUT.iterdir())]}
    (OUT/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({'corrected_audit_sha256':sha(OUT/'audit.json'),'corrected_index_sha256':sha(OUT/'index.json')}))

if __name__=='__main__':main()
