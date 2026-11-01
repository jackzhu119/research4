#!/usr/bin/env python3
"""Static source audit only: no PySCF import, numerical arrays, HDF or process access."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[3])
ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
args=ap.parse_args();repo=args.repo.resolve();out=args.output.resolve()
out.mkdir(parents=True,exist_ok=True)
if (out/'audit.json').exists():raise FileExistsError('Use a new output directory; preserve source audit evidence')
site=repo/'.runtime/quantum-upgrade-site'
paths={
 'rhf_hessian':site/'pyscf/hessian/rhf.py',
 'rks_hessian':site/'pyscf/hessian/rks.py',
 'response_functions':site/'pyscf/scf/_response_functions.py',
 'hf_scf':site/'pyscf/scf/hf.py',
 'vhf':site/'pyscf/scf/_vhf.py',
 'cphf':site/'pyscf/scf/cphf.py',
 'v8_worker':repo/'scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py',
 'package_metadata':site/'pyscf-2.14.0.dist-info/METADATA',
}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources={key:{'path':str(p.relative_to(repo)),'sha256':sha(p),'bytes':p.stat().st_size,'line_count':len(p.read_text().splitlines())} for key,p in paths.items()}
texts={key:p.read_text() for key,p in paths.items()}
version=next(line.partition(':')[2].strip() for line in texts['package_metadata'].splitlines() if line.startswith('Version:'))
tree=ast.parse(texts['rhf_hessian'])
partial=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_partial_hess_ejk')
with_k_loads=[n.lineno for n in ast.walk(partial) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id=='with_k']
assert with_k_loads==[], 'Re-audit: installed partial function now uses with_k'
kernel_calls=[]
for node in ast.walk(partial):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_get_jk':
  kernel_calls.append({'line':node.lineno,'intor':ast.literal_eval(node.args[1]),'components':ast.literal_eval(node.args[2]),'aosym':ast.literal_eval(node.args[3]),'explicit_shls_slice':any(k.arg=='shls_slice' for k in node.keywords),'explicit_vhfopt':any(k.arg=='vhfopt' for k in node.keywords)})
kernel_calls.sort(key=lambda x:x['line'])
assert [r['intor'] for r in kernel_calls]==['int2e_ipip1','int2e_ip1ip2','int2e_ipvip1']
assert 'SUPERVISED_PARALLEL_DIRECT_V8_SAVE_ONLY' in texts['v8_worker']
assert 'screened_direct_rks_type(dft.rks.RKS, DirectSCF.get_jk)' in texts['v8_worker']
assert '"partial_kernel_tiling_used": False' in texts['v8_worker']

ranges={
 'rhf_hessian':[(39,60),(99,156),(158,178),(180,208),(239,283),(286,359),(560,573)],
 'rks_hessian':[(60,80),(84,104),(132,178),(212,238),(350,356),(379,410),(1140,1165),(2110,2122)],
 'response_functions':[(44,101)],
 'hf_scf':[(1019,1024),(2124,2164)],
 'vhf':[(151,160),(370,410),(482,516),(535,550)],
 'cphf':[(43,49),(122,137)],
 'v8_worker':[(159,182),(213,236),(353,388),(576,606),(611,652),(678,702),(745,753),(767,817),(930,944),(1000,1020)],
}
excerpt=['STATUS: SOURCE_ANALYSIS_NOT_NEW_QUANTUM_RESULTS',
         'Selected static source excerpts, not an execution log. No PySCF import or quantum call.',
         'PySCF source is Apache-2.0 (source credits original PySCF authors); V8 code is repository-local.',
         'No active output arrays, HDF, checkpoints, progress logs or process state were read.','']
for key,spans in ranges.items():
 lines=texts[key].splitlines()
 excerpt.extend([f'## {key}: {sources[key]["path"]}',f'SHA256: {sources[key]["sha256"]}'])
 for begin,end in spans:
  excerpt.append(f'### Lines{begin}–{end}')
  excerpt.extend(f'L{n}: {lines[n-1]}' for n in range(begin,min(end,len(lines))+1))
 excerpt.append('')
(out/'source_line_excerpts.txt').write_text('\n'.join(excerpt)+'\n')

table=[
 {'stage':'partial','kernel':'int2e_ipip1','components':9,'scope':'One global call; all four shell ranges global','scripts':'lk->s1ij (J); jk->s1il (K-shaped)','explicit_VHFOpt':'_make_vhfopt(ipip1, int2e_ipip1ipip2), conditional native symbol availability','calls_for_75_atoms':'1','pure_PBE_K_skipped':'NO: with_k parameter is unused in this installed function','source':'hessian/rhf.py:120–125'},
 {'stage':'partial','kernel':'int2e_ip1ip2','components':9,'scope':'One call per atom; shell leg1 restricted to that atom, other3 global','scripts':'ji->s1kl (J); li->s1kj and lj->s1ki (K-shaped)','explicit_VHFOpt':'ip1ip2_opt built once; derivative/int2e q-conditions reused and density conditions updated','calls_for_75_atoms':'75','pure_PBE_K_skipped':'NO','source':'hessian/rhf.py:129,138–147'},
 {'stage':'partial','kernel':'int2e_ipvip1','components':9,'scope':'One call per atom; shell leg1 restricted to that atom, other3 global','scripts':'lk->s1ij (J); li->s1kj (K-shaped)','explicit_VHFOpt':'ipvip1_opt built once; derivative/int2e q-conditions reused and density conditions updated','calls_for_75_atoms':'75','pure_PBE_K_skipped':'NO','source':'hessian/rhf.py:130,148–156'},
 {'stage':'H1AO','kernel':'int2e_ip1','components':3,'scope':'One call per atom; shell leg1 restricted to that atom, other3 global','scripts':'ji->s2kl and lk->s1ij (J only in nonhybrid RKS branch)','explicit_VHFOpt':'NOT passed by rks.make_h1; _get_jk defaultsNone; integral optimizer may still be created internally','calls_for_75_atoms':'75','pure_PBE_K_skipped':'YES: nonhybrid branch does not request K scripts','source':'hessian/rks.py:141–172; hessian/rhf.py:261–268; scf/_vhf.py:482–500'},
 {'stage':'CPHF response','kernel':'int2e (sph/cart suffix)','components':1,'scope':'Global AO response density; no nuclear-derivative integral or atom shell truncation','scripts':'J only for each global perturbed density in nonhybrid ground-state response','explicit_VHFOpt':'mf.get_j -> frozenV8 hf.SCF.get_jk -> directSCF cached VHFOpt/J-only prescreen','calls_for_75_atoms':'NOT_DETERMINED_STATICALLY: iterative Krylov response calls plus final response','pure_PBE_K_skipped':'YES: hybrid/range-separated exchange branches bypassed','source':'hessian/rhf.py:343–359; scf/_response_functions.py:67–101; hf.py:2124–2164; _vhf.py:370–410'},
]
with (out/'four_center_call_map.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)

N=75;M=762;nocc=160
def shape_info(shape):
 count=1
 for d in shape:count*=d
 return {'shape':list(shape),'float64_elements':count,'nominal_bytes':8*count,'nominal_GiB':8*count/1024**3}
dimensions={
 'input_basis':'75 atoms and762 AO functions are task-supplied dimensions; electron320/singlet nocc160. No active calculation array was opened. These are exact shape-arithmetic examples, not measured RSS.',
 'final_cartesian_Hessian':shape_info((N,3,N,3)),
 'RKS_XC_deriv2_vmat_global':shape_info((N,3,3,M,M)),
 'H1AO_all_atoms':shape_info((N,3,M,M)),
 'single_nine_component_AO_matrix':shape_info((3,3,M,M)),
 'response_density_batch_for8_atoms':shape_info((8*3,M,M)),
 'MO1_batch_for8_atoms_full_rank_NMO_assumption':shape_info((8*3,M,nocc)),
 'hypothetical_dense_uncontracted_four_center_ERI_NOT_ALLOCATED_BY_V8':shape_info((M,M,M,M)),
 'limitations':'These arrays need not all coexist, and allocator/scratch/grid/SCF/Krylov overhead is omitted. No peak-RSS or runtime estimate follows. mf.max_memory3000 controls numerical block policy, not a hard cap on all global arrays.',
}
audit={
 'status':'SOURCE_ANALYSIS_NOT_NEW_QUANTUM_RESULTS','created_utc':datetime.now(timezone.utc).isoformat(),'pyscf_version_from_text_metadata':version,
 'scope':'Installed static PySCF source and repository V8 source only; no active calculation output or process read.',
 'sources':sources,'AST_checks':{'partial_with_k_load_lines':with_k_loads,'with_k_argument_is_unused_in_installed_partial_function':True,'partial_get_jk_callsites':kernel_calls,'V8_direct_SCF_delegation_present':True,'V8_partial_kernel_tiling_disabled_in_source':True},
 'call_map':table,
 'static_count_for_one_full_partial_atmlst75':{'python_get_jk_contraction_calls':1+2*N,'J_script_bindings':1+2*N,'K_shaped_script_bindings_still_requested':1+3*N,'total_script_bindings':2+5*N,'screening_setup_note':'Three _make_vhfopt invocations; their helper integrals run only when required native symbols exist. Count does not measure integral quartets or timing.'},
 'screening_notes':{'partial':'Explicit derivative VHFOpt is supplied; each construction can fall backNone if required native symbol absent. _get_jk forwards it to direct_bindm.',
 'derivative_optimizer_tolerance':'_make_vhfopt constructs a new _VHFOpt without passing mf.direct_scf_tol; installed _VHFOpt default is1e-14. Do not equate this automatically to V8 mf.direct_scf_tol1e-13.',
 'H1':'No explicit derivative VHFOpt passed in nonhybrid rks.make_h1. This is not proof that all libcint internal pruning is disabled.',
 'CPHF':'SCF get_jk constructs/reuses ordinary int2e VHFOpt from mf.direct_scf_tol and uses J-only prescreen; density conditions update for each call.'},
 'dimension_examples':dimensions,
 'V8_scope':{'full_partial_hessian_original_called_once_before_capture':True,'full_H1_original_called_before_per_atom_save':True,'response_only_perturbation_batching':True,'global_AO_MO_response_space_retained':True,'partial_kernel_tiling':False,'basis_grid_functional_screening_arguments_not_changed_by_thread_guard':True,'save_only_not_restore':True},
 'forbidden_inferences':['No current executing kernel or atom index can be inferred from this static source audit.','An old density-count/components signature is not a current kernel stack trace or completion percentage.','No completion time, ETA, remaining integral fraction, or speedup is estimated.','Pure PBE with_kFalse does not remove the installed partial_hessian K-shaped contraction scripts.','This audit does not recommend modifying or interrupting the running frozen worker.','These static dimensional costs are not measured runtime or peak resident memory.'],
 'new_quantum_computations':0,'PySCF_imported':False,'source_files_modified':0,'active_arrays_or_HDF_or_checkpoints_read':0,'processes_observed_or_signalled':0,
}
(out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
(out/'source_hashes.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
(out/'dimensional_cost_examples.json').write_text(json.dumps(dimensions,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':audit['status'],'pyscf_version':version,'V8_source_sha256':sources['v8_worker']['sha256'],'partial_with_k_load_lines':with_k_loads,'partial_contraction_calls_if_full75_atoms':151,'new_quantum_computations':0,'active_output_reads':0},indent=2))
