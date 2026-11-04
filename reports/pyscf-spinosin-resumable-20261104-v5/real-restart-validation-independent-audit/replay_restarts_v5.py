"""Independent NumPy-only replay of completed cross-process control calculations.

Does not import the consumer, PySCF, or any quantum engine. Reads only healthy
closed source/final arrays; exceptional intermediate units need explicit root
forensic receipt, which never makes their original parent healthy.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse,csv,hashlib,json,sys
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
BASE=ROOT/'reports/pyscf-spinosin-resumable-20261104-v5'
MASS={'H':1.008,'C':12.011,'O':15.999}
BOHR=.52917721092
PLANCK,E_MASS,AVOGADRO=6.626070040e-34,9.10938356e-31,6.022140857e23
HBAR=PLANCK/(2*np.pi)
HARTREE2J=HBAR**2/(E_MASS*(BOHR*1e-10)**2)
FACTOR=np.sqrt(HARTREE2J/((1e-3/AVOGADRO)*(BOHR*1e-10)**2))/(2*np.pi*299792458)*1e-2
INPUTS={}
def require(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def record(p,expected=None):
    p=Path(p).resolve();require(p.is_file() and not p.is_symlink(),'Unsafe source')
    digest=sha(p);require(expected is None or digest==expected,'SHA mismatch '+str(p))
    INPUTS[str(p.relative_to(ROOT))]={'sha256':digest,'bytes':p.stat().st_size}
    return p
def read(p,expected=None):return json.loads(record(p,expected).read_text())
def arr(p,shape=None,expected=None):
    a=np.load(record(p,expected),allow_pickle=False)
    require(a.dtype==np.float64 and np.isfinite(a).all() and (shape is None or a.shape==shape),'Invalid array')
    return a
def healthy(r):
    return (r.get('status')=='EXITED' and r.get('exit_code')==0 and isinstance(r.get('ended_at_utc'),str)
        and all(r.get(k) is False for k in ['wall_terminated','cpu_terminated','memory_terminated','monitor_failed','hard_killed'])
        and not r.get('residual_processes_require_review_after_main_payload_exit'))

def qualify(phase,ledger):
    require(ledger['schema']=='research4-pyscf-actual-cpu-budget-v5','V5 ledger required')
    rows=[r for r in ledger['phases']if r['phase']==phase];require(len(rows)==1 and healthy(rows[0]),'Not healthy closed phase '+phase)
    row=rows[0];proof=read(row['supervisor_proof_path'],row['supervisor_proof_sha256'])
    require(proof['schema']=='research4-pyscf-actual-cpu-supervisor-proof-v5','V5 proof required')
    require(proof['supervisor_script_sha256']=='33f1dc3db5280f16c9898c561e345143614730dc052b171b4f11fbbb079cdeee','V5 guard source differs')
    record(proof['supervisor_script_path'],proof['supervisor_script_sha256'])
    record(row['log'],row['log_sha256']);result=read(BASE/phase/'result.json')
    expected_entry='be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8' if phase.endswith('-prepare-001') else 'b40188f747c01e9efa97672af6eedfd281a9e9697d20bc7d57f125bb131fa190'
    require(result['source_worker']['sha256']==expected_entry,'V5 state/consumer source differs')
    if not phase.endswith('-prepare-001'):
        require(result['state_worker_sha256']=='be4727b76d59a0b0dd3defba249ef833ee8ef0ea89473e390dcef389728415c8','State binding differs')
        require(result['incremental_module_sha256']=='ce96055f24b39833337844c178f94597a9ac4ba7721ba019816d03628f971f55','Math core differs')
        require(result['frozen_V8_sha256']=='acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d','Frozen V8 differs')
        record(ROOT/'scripts/pyscf_incremental_rhf_checkpoint_20261103.py',result['incremental_module_sha256'])
        record(ROOT/'scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py',result['frozen_V8_sha256'])
    require(proof==result['resource_supervision']['proof'] and proof['child_pid']==row['pid'],'Parent/proof mismatch')
    require(row['actual_cpu_seconds']>=0 and np.isfinite(row['actual_cpu_seconds']),'No actual wait4 cost')
    record(result['source_worker']['path'],result['source_worker']['sha256'])
    return result,row

def projector(coords,mass):
    centered=coords-(coords*mass[:,None]).sum(0)/mass.sum()
    ext=np.stack([(np.sqrt(mass)[:,None]*v).ravel() for v in np.eye(3)]
        +[(np.sqrt(mass)[:,None]*np.cross(centered,v)).ravel()for v in np.eye(3)],axis=1)
    require(np.linalg.matrix_rank(ext,tol=1e-10)==6,'Nonlinear TR rank not6')
    q,_=np.linalg.qr(ext);p=np.eye(coords.size)-q@q.T;w,v=np.linalg.eigh(p)
    return p,v[:,w>1e-7]

def check_case(phase,result,coords,mass):
    folder=BASE/phase;n=len(mass);nm=3*n-6
    require(result['status']=='FREQUENCIES_COMPUTED' and result['complete_global_Hessian'] is True,'Incomplete scientific result')
    raw=arr(folder/'hessian_hartree_per_bohr2.npy',(n,n,3,3),result['raw_hessian_sha256'])
    sym=arr(folder/'hessian_used_for_analysis.npy',raw.shape,result['analysis_hessian_sha256'])
    require(np.array_equal(sym,(raw+raw.transpose(1,0,3,2))*.5),'Not prescribed symmetrization')
    asym=float(np.max(np.abs(raw-raw.transpose(1,0,3,2))));require(asym<=1e-5,'Asymmetry threshold')
    require(np.array_equal(arr(folder/'normal_mode_geometry_bohr.npy',(n,3)),coords),'Geometry differs')
    modes=arr(folder/'normal_modes.npy',(nm,n,3));frequency=np.array(result['frequencies_cm1']);thz=np.array(result['frequencies_thz'])
    rows=list(csv.DictReader(record(folder/'frequencies.csv').open()))
    require(len(rows)==nm and np.all(frequency>0),'Incomplete/nonpositive frequency list')
    csvcm=np.array([float(r['frequency_cm1'])for r in rows]);csvthz=np.array([float(r['frequency_thz'])for r in rows])
    require(np.array_equal(csvcm,frequency) and np.array_equal(csvthz,thz),'JSON/CSV frequency mismatch')
    require(np.max(np.abs(frequency*.0299792458-thz))<1e-12,'THz conversion mismatch')
    require(all(r['IR_intensity_status']=='NOT_COMPUTED' and r['imaginary']=='False' for r in rows),'Invented IR/imaginary state')
    p,vib=projector(coords,mass)
    matrix=np.einsum('pqxy,p,q->pqxy',sym,mass**-.5,mass**-.5).transpose(0,2,1,3).reshape(3*n,3*n)
    eig,local=np.linalg.eigh(vib.T@matrix@vib);require(len(eig)==nm and np.all(eig>0),'Not complete positive spectrum')
    replay=np.sqrt(eig)*FACTOR;require(np.max(np.abs(replay-frequency))<1e-7,'Independent frequency replay differs')
    z=(modes*np.sqrt(mass)[None,:,None]).reshape(nm,3*n).T
    gram=float(np.max(np.abs(z.T@z-np.eye(nm))));external=float(np.max(np.abs(p@z-z)))
    residual=float(np.max(np.abs(p@matrix@p@z-z*((frequency/FACTOR)**2)[None,:])))
    require(max(gram,external,residual)<1e-10,'Mode normalization/projected eigen-equation failed')
    independent_z=vib@local;overlap=np.abs(np.diag(independent_z.T@z))
    require(float(overlap.min())>1-1e-8,'Saved/independent modes differ modulo sign')
    cache=read(folder/'hessian_cache_manifest.json',result['hessian_cache_manifest_sha256'])
    require(cache['native_global_kernel_returned_normally'] and cache['response_complete_global_atom_ids']==list(range(n)),'Missing full response')
    stats={'modes':nm,'all_positive':True,'raw_max_asymmetry':asym,'frequency_cm1':frequency.tolist(),'frequency_thz':thz.tolist(),
        'independent_frequency_cm1':replay.tolist(),'max_independent_frequency_difference_cm1':float(np.max(np.abs(replay-frequency))),
        'mass_weighted_orthogonality_error':gram,'external_projection_error':external,'projected_eigen_equation_residual':residual,
        'minimum_independent_mode_absolute_overlap':float(overlap.min()),'measured_CPHF_residual_norm':result['measured_CPHF_residual_norm']}
    return {'raw':raw,'sym':sym,'frequency':frequency,'thz':thz,'z':z,'stats':stats}

def audit(molecule,finalphase):
    folder=OUT/molecule;require(not folder.exists(),'Refuse overwrite existing case')
    ledgerbytes=(BASE/'cpu_budget.json').read_bytes();ledger=json.loads(ledgerbytes)
    pre=qualify(f'v5-validation-{molecule}-prepare-001',ledger);restore=qualify(f'v5-validation-{molecule}-restore-gradient-001',ledger)
    stockphase=f'v5-validation-{molecule}-stock-hessian-'+'001'
    stock=qualify(stockphase,ledger);final=qualify(finalphase,ledger)
    source,sourcerow=pre;snapshot=Path(source['snapshot_manifest']['path']).parent
    manifest=read(snapshot/'manifest.json',source['snapshot_manifest']['sha256'])
    symbols=[x.split()[0]for x in record(BASE/sourcerow['phase']/'optimized.xyz').read_text().splitlines()[2:]if x.strip()]
    mass=np.array([MASS[s]for s in symbols]);n=len(mass)
    coords=arr(BASE/restore[1]['phase']/'normal_mode_geometry_bohr.npy',(n,3))
    require(source['scf_converged'] and source['final_gradient_thresholds_passed'],'Source SCF/gradient not good')
    source_gradient=arr(BASE/sourcerow['phase']/'gradient_hartree_per_bohr.npy',(n,3))
    restored_gradient=arr(BASE/restore[1]['phase']/'gradient_hartree_per_bohr.npy',(n,3),restore[0]['gradient_npy_sha256'])
    gd=float(np.max(np.abs(source_gradient-restored_gradient)));require(gd<=1e-8,'Restored gradient drift')
    require(abs(gd-restore[0]['restored_gradient_max_abs_difference_hartree_per_bohr'])<1e-18,'Gradient receipt discrepancy')
    norm=np.linalg.norm(source_gradient,axis=1);require(np.sqrt(np.mean(norm**2))<3e-4 and norm.max()<4.5e-4,'Gradient not stationary')
    phases=[r['phase']for r in ledger['phases']if r['phase'].startswith(f'v5-validation-{molecule}-segment-')
            and (BASE/r['phase']/'result.json').exists()]
    require(finalphase in phases,'Missing final segment')
    rows=[sourcerow,restore[1],stock[1]];segments=[]
    expectedbefore=0;forensic=[]
    for name in phases:
        row=next(r for r in ledger['phases']if r['phase']==name)
        if healthy(row):result,_=qualify(name,ledger)
        else:raise ValueError('No failed or forensic unit producer is allowed in this V5 validation')
        before=result['incremental_before']['total_units_completed'];after=result['incremental_after']['total_units_completed']
        require(before==expectedbefore and after>before,'Segments do not advance a contiguous prefix')
        expectedbefore=after
        require(result['snapshot_manifest_sha256']==source['snapshot_manifest']['sha256'] and result['no_new_SCF_performed'] is True
                and result['stable_MF_state_unchanged'] is True,'SCF/source drift')
        require(result['source_state_binding']==final[0]['source_state_binding'],'Exact MF source binding differs')
        rows.append(row);segments.append({'phase':name,'pid':row['pid'],'healthy_closed_parent':healthy(row),'before':before,'after':after,
            'status':result['status'],'actual_wait4_CPU_seconds':row['actual_cpu_seconds']})
    require(expectedbefore==2*n and final[0]['incremental_after']['complete_RHF_helper'] is True,'Helper prefix incomplete')
    for result,row in [restore,stock,final]:
        require(result['snapshot_manifest_sha256']==source['snapshot_manifest']['sha256'] and result['no_new_SCF_performed'] is True
                and result['stable_MF_state_unchanged'] is True,'Restore/reference MF state differs')
        require(result['source_state_binding']==final[0]['source_state_binding'],'State binding differs')
    require(len({r['pid']for r in rows})==len(rows),'Phases reused a PID instead of process restart')
    accepted=final[0].get('prior_RHF_unit_producer_checks',[]);released=[x for x in accepted if x['qualification']=='EXPLICIT_ROOT_FORENSIC_RELEASE']
    require(not forensic and not released,'V5 controls must contain no forensic reuse')
    require(all(x['qualification']=='CLOSED_HEALTHY_V5_PRODUCER' for x in accepted),'Unexpected V5 producer qualification label')
    a=check_case(stockphase,stock[0],coords,mass);b=check_case(finalphase,final[0],coords,mass)
    difference={key:float(np.max(np.abs(a[key]-b[key])))for key in ['raw','sym','frequency','thz']}
    require(difference['raw']<1e-6 and difference['sym']<1e-6 and difference['frequency']<.01,'Full stock/restart numerical equivalence failed')
    cross=np.abs(a['z'].T@b['z']);overlap=np.diag(cross);require(float(overlap.min())>1-1e-8,'Stock/restart mode vectors differ')
    require('pyscf'not in sys.modules,'Quantum engine imported')
    # Verify scientific files unchanged. Ledger is snapshotted instead of assumed immutable.
    for relative,meta in INPUTS.items():require(sha(ROOT/relative)==meta['sha256'],'Source changed during replay')
    folder.mkdir();(folder/'ledger_observed.json').write_bytes(ledgerbytes)
    report={'schema':'research4-real-cross-process-restart-independent-audit-v5','status':'PASS_INDEPENDENT_NUMPY_REPLAY',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'molecule':molecule,'scope':'Small-molecule engineering control only, not Spinosin frequencies or proof of physical accuracy.',
        'snapshot_manifest_sha256':source['snapshot_manifest']['sha256'],'source_prepare_result_sha256':sha(BASE/sourcerow['phase']/'result.json'),
        'stock_result_sha256':sha(BASE/stockphase/'result.json'),'segmented_result_sha256':sha(BASE/finalphase/'result.json'),
        'all_raw_Hessian_elements_compared':int(a['raw'].size),'all_atom_pair_couplings_compared':True,'max_abs_differences':difference,
        'fixed_tolerances':{'raw_Eh_per_Bohr2':1e-6,'sym_Eh_per_Bohr2':1e-6,'frequency_cm1':.01,'replay_cm1':1e-7,'gradient_Eh_per_Bohr':1e-8},
        'stock':a['stats'],'segmented':b['stats'],'minimum_stock_segment_mode_absolute_overlap':float(overlap.min()),
        'mode_sign_is_arbitrary':True,'mode_comparison_method':'Absolute mass-weighted vector overlaps; independent Cartesian TR-QR projection and NumPy eigh.',
        'restore_gradient_difference_Eh_per_Bohr':gd,'same_exact_MF_snapshot_all_stages':True,'no_new_SCF_recorded_in_consumers':True,
        'segments':segments,'independent_payload_PIDs':[r['pid']for r in rows], 'all_payload_PIDs_unique':True,
        'actual_wait4_CPU_seconds_for_selected_lineage':sum(r['actual_cpu_seconds']for r in rows),
        'parent_rows':rows,'forensically_released_failed_parents':forensic,'failed_parent_never_relabelled_healthy':True,
        'proof_schema':'research4-pyscf-actual-cpu-supervisor-proof-v5','forensic_units_reused':0,
        'unit_qualification_label':'CLOSED_HEALTHY_V5_PRODUCER',
        'mechanically_adapted_from_audit_source_sha256':'aad4cd3abbb7683a3912f2b58a873e7318f88a432e2109714d8c9138168e1f1f',
        'IR_intensities_status':'NOT_COMPUTED','masses_amu':MASS,'frequency_factor_cm1':float(FACTOR),'new_quantum_evaluations':0,
        'PySCF_imported':False,'audit_source_sha256':sha(__file__),'input_files':dict(INPUTS)}
    (folder/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (folder/'frequencies_replayed.csv').open('w',newline='')as f:
        w=csv.writer(f);w.writerow(['case','mode','saved_cm1','NumPy_replay_cm1','saved_THz','IR_intensity_status'])
        for label,value in [('stock',a),('segmented',b)]:
            for i,(cm,replay,thz)in enumerate(zip(value['frequency'],value['stats']['independent_frequency_cm1'],value['thz']),1):w.writerow([label,i,cm,replay,thz,'NOT_COMPUTED'])
    (folder/'README.md').write_text(f"PASS: compared all {a['raw'].size} raw and symmetrized Hessian elements and all {3*n-6} positive modes. Raw max difference {difference['raw']:.16g} Eh/Bohr²; frequency max difference {difference['frequency']:.16g} cm⁻¹. Independent NumPy mass weighting, translation/rotation removal, eigenvalues and sign-independent mode overlaps pass. Source snapshot, fresh-vs-restored gradient, unique process IDs and actual wait4 CPU are traced. Forensic failed parents: {forensic}; their flags are preserved and explicit root receipt is required. IR NOT_COMPUTED. No quantum calculation or PySCF import occurred. These are software controls, not Spinosin modes or a general speed benchmark.\n")
    files=[p for p in folder.iterdir()if p.is_file()]
    index={'status':'SEALED_REAL_RESTART_CONTROL_AUDIT','audit_source_sha256':sha(__file__),
        'files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(files)]}
    (folder/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({'molecule':molecule,'status':report['status'],'differences':difference,'modes':3*n-6,'audit_sha256':sha(folder/'audit.json'),'index_sha256':sha(folder/'index.json')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--molecule',choices=['water','ethanol'],required=True);p.add_argument('--final-phase',required=True)
    args=p.parse_args();audit(args.molecule,args.final_phase)
