"""Independent metadata/artifact review of final mode postprocessor, no QM."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
BASE=ROOT/'reports/pyscf-spinosin-resumable-20261104-v5'
SOURCE_SHA='a2e3588c6dea121be58e4a69afb137bc6e5092199ca4180937fcdb34f6f57154'
TEST_SHA='8ddf29b2a399ead85095c5918be25dcb07a29dcd1f96870c2279422d8c950a72'
INPUTS={}
def require(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p,expected=None):
    p=Path(p).resolve();require(p.is_file()and not p.is_symlink(),'Missing/symlink artifact')
    digest=sha(p);require(expected is None or digest==expected,'SHA mismatch '+str(p))
    INPUTS[str(p.relative_to(ROOT))]={'sha256':digest,'bytes':p.stat().st_size}
    return p
def read(p,expected=None):return json.loads(record(p,expected).read_text())

def main():
    require(not(OUT/'review.json').exists(),'Refuse sealed overwrite')
    source=record(ROOT/'scripts/analyze_spinosin_v5_modes_20261104.py',SOURCE_SHA)
    tests=record(ROOT/'tests/test_analyze_spinosin_v5_modes.py',TEST_SHA)
    text=source.read_text()
    require(text.index('qualified_gradient = consumer.qualify_gradient_receipt',text.index('def analyze('))
            <text.index('w.load_snapshot_payload(',text.index('def analyze(')),'Gradient receipt not qualified before snapshot payload loader')
    log=record(OUT/'unittest-final-v2.log').read_text()
    require('Ran 19 tests' in log and log.rstrip().endswith('OK'),'Independent final19 tests did not pass')
    runtime=BASE/'modes-analysis-preflight-20261104-v2';execution=read(runtime/'execution_records.json')
    require(execution['source_sha256']==SOURCE_SHA and execution['test_sha256']==TEST_SHA
            and execution['unit_test_exit_code']==0 and execution['unit_test_count']==19,'Final CLI source/test binding mismatch')
    require(execution['target_status']=='NOT_RUN' and execution['target_frequencies_or_arrays_read']is False
            and execution['new_quantum_evaluations']==0,'Control preflight promoted to target result')
    for run in execution['runs']:
        require(run['exit_code']==0 and run['new_quantum_evaluations']==0,'Failed/quantum control run')
        record(runtime/(run['label']+'.execution.log'),run['execution_log_sha256'])
    controls=[];artifact_count=0
    for identity,count,phase in [('water',3,'v5-validation-water-segment-003'),('ethanol',21,'v5-validation-ethanol-segment-004')]:
        folder=runtime/'controls'/identity;d=read(folder/'checks.json');index=read(folder/'artifact_index.json')
        require(index['analysis_source_sha256']==SOURCE_SHA,'Output artifact index source mismatch')
        for entry in index['files']:
            p=record(folder/entry['path'],entry['sha256']);require(p.stat().st_size==entry['bytes'],'Artifact byte size mismatch');artifact_count+=1
        require(d['analysis_source_sha256']==SOURCE_SHA and d['status']=='DONE_CONTROL_VALIDATION'
                and d['normal_modes']==count and d['all_frequencies_positive']is True,'Wrong current control modes')
        require(d['qualified_for_mechanism']is False and d['experimental_fold_records']=='NOT_READ'
                and d['IR_intensities_status']=='NOT_COMPUTED' and d['Spinosin_target_status']=='NOT_RUN_CONTROL_ONLY'
                and d['new_quantum_evaluations']==0 and d['PySCF_imported']is False,'Control overclaim')
        original=read(BASE/phase/'result.json',d['source_result_sha256'])
        with record(folder/'all_frequencies_cm1_THz.csv').open(newline='')as f:rows=list(csv.DictReader(f))
        require(len(rows)==count,'Incomplete exported mode count')
        for i,row in enumerate(rows):
            require(float(row['frequency_cm1'])==original['frequencies_cm1'][i]
                    and float(row['frequency_THz'])==original['frequencies_thz'][i]
                    and abs(float(row['frequency_THz'])-float(row['frequency_cm1'])*.0299792458)<1e-12
                    and row['qualified_for_mechanism']=='False'and row['IR_intensity_status']=='NOT_COMPUTED','CSV unit/source/qualification mismatch')
        for stem in ['signed_mode_frequencies','low_frequency_positions_not_IR']:
            require((folder/(stem+'.png')).read_bytes().startswith(b'\x89PNG\r\n\x1a\n'),'Not real PNG')
            require((folder/(stem+'.pdf')).read_bytes().startswith(b'%PDF-'),'Not real PDF')
        for p in (folder/'animations').glob('*.xyz'):
            contents=record(p).read_text();require('DISPLAY_ONLY_NOT_MD'in contents and'source atom order'in contents,'Animation overclaim')
        controls.append({'identity':identity,'modes':count,'checks_sha256':sha(folder/'checks.json'),
            'independent_TR_replay_error_cm1':d['numpy_replay']['max_frequency_difference_cm1'],
            'reported_gradient_difference_Eh_per_Bohr':d['fresh_gradient_max_difference_Eh_per_Bohr'],
            'source_result_sha256':d['source_result_sha256'],'mechanism_qualified':False})
    pre=read(runtime/'target-NOT_RUN/preflight.json');require(pre['status']=='NOT_RUN'and pre['target_frequencies_read']is False,'Target preflight not NOT_RUN')
    require(not list((runtime/'target-NOT_RUN').rglob('*.csv')),'Fake target frequency table')
    (OUT/'reviewed_source_snapshot.py').write_bytes(source.read_bytes());(OUT/'reviewed_tests_snapshot.py').write_bytes(tests.read_bytes())
    checks={'final_healthy_V5_closure_before_arrays':True,'gradient_receipt_metadata_gate_now_before_snapshot_NPY':True,
      'snapshot_producer_closure_before_payload_loading':True,'frozen_consumer_source_and_calc_boot_MF_Greceipt':True,
      'default75atoms219modes_and_explicit_small_controls':True,'source_atom_order_Bohr_Angstrom_geometry_hash':True,
      'mass_weighting_normal_modes_reduced_masses_units':True,'independent_TR_projected_signed_frequency_replay':True,
      'imaginary_and_zero_modes_retained_not_absolute_filtered':True,'any_nonpositive_spectrum_blocks_mechanism_and_animation':True,
      'twenty_overlapping_fold_records_not_independent_peaks':True,'full_cartesian_comparison_not_nearest_assignment':True,
      'IR_NOT_COMPUTED_no_intensity_or_IR_activity_claim':True,'animations_rescaled_source_order_display_not_MD':True,
      'isolated_single_conformer_not_global_minimum_or_solid_spectrum':True,'existing_results_not_overwritten':True}
    result={'schema':'research4-independent-modes-postprocessor-review-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
      'status':'PASS_STATIC_19_PURE_TESTS_AND_FINAL_CLOSED_CONTROLS','blocking_findings':[],
      'reviewed_source_sha256':SOURCE_SHA,'reviewed_tests_sha256':TEST_SHA,'independent_test_exit_code':0,'independent_tests':19,
      'independent_test_log_sha256':sha(OUT/'unittest-final-v2.log'),'static_checks':checks,'final_control_artifacts_hash_verified':artifact_count,
      'final_controls':controls,'final_execution_records_sha256':sha(runtime/'execution_records.json'),
      'resolved_note':'Author moved metadata-only gradient receipt qualification before snapshot payload loading and added a regression test. Preliminary18-test log remains development history.',
      'limits':['Source mode/CSV files lack original producer file SHA; current hashes plus anchored Hessian/JSON eigenchecks are disclosed, not invented old hashes.',
      'Global-kernel completion checks are metadata, not an independently measured CPHF residual.',
      'Only healthy closed water/ethanol software controls validate runtime; Spinosin has no completed-mode evidence in this review.',
      'End-of-run local input hash checks cover locally tracked inputs; frozen source snapshot helper also verifies its manifest/payload hashes at load time.'],
      'active_Spinosin_HDF_NPY_or_result_read':False,'new_quantum_evaluations':0,'Git_operations':0,'reviewer_modified_subject_source':False,
      'review_script_sha256':sha(__file__),'input_files':INPUTS}
    for relative,entry in INPUTS.items():require(sha(ROOT/relative)==entry['sha256'],'Input changed during review')
    (OUT/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'README.md').write_text('PASS: no blocking finding. Final frozen script a2e3588c... and tests8ddf29b2... independently passed19 pure tests. Static review verified healthy V5/fresh-gradient/source gates before any snapshot payload load, exact75/219 default, geometry/atom-order and mass/unit consistency, signed imaginary modes with mechanism blocked for any nonpositive spectrum,20 overlapping-fold records as descriptive Cartesian comparisons, and no IR/MD/global-minimum claims. Author final-v2 healthy water3/ethanol21 CLI outputs bind this exact script;18 artifact hashes and exported frequency units/source values were independently checked. PNG/PDF and display-only animations are real saved outputs. Target preflight remains NOT_RUN without a frequency table. No active Spinosin result/array/checkpoint was read; no QM or Git ran. Prior18-test log is preserved as development history.\n')
    index={'status':'SEALED_FINAL_INDEPENDENT_MODE_POSTPROCESSOR_REVIEW','files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(OUT.iterdir())if p.is_file()and p.name!='index.json']}
    (OUT/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'review_sha256':sha(OUT/'review.json'),'index_sha256':sha(OUT/'index.json')}))

if __name__=='__main__':main()
