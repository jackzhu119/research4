"""stdlib-only static Hessian data-flow review; never imports numeric/QM code."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
FILES={
 'rhf':(ROOT/'.runtime/quantum-upgrade-site/pyscf/hessian/rhf.py','2d986bfc01aac98447034a207df639b3d6897f67ce1ab2a013fe2d0853c85de5'),
 'rks':(ROOT/'.runtime/quantum-upgrade-site/pyscf/hessian/rks.py','d61dbd1727b286845737853b92e1812a3e6a966dca29a9a8a79ff5e9a7725c03'),
 'v8':(ROOT/'scripts/pyscf_spinosin_parallel_hessian_worker_20261031.py','acfe7708748ac237c9912aa9b1d98e047d842794fd0c6f08b1a67b993f5b277d')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
def rootname(n):
    while isinstance(n,(ast.Subscript,ast.Attribute)):n=n.value
    return n.id if isinstance(n,ast.Name)else None
def callname(n):return n.func.attr if isinstance(n.func,ast.Attribute)else n.func.id if isinstance(n.func,ast.Name)else ''
def main():
    assert not(OUT/'contract.json').exists()
    trees={}
    for name,(p,digest)in FILES.items():assert sha(p)==digest;trees[name]=ast.parse(p.read_text())
    elec=function(trees['rhf'],'hess_elec');partial=function(trees['rks'],'partial_hess_elec');h1=function(trees['rks'],'make_h1');solve=function(trees['rhf'],'solve_mo1')
    calls=[(n.lineno,callname(n))for n in ast.walk(elec)if isinstance(n,ast.Call)]
    assert min(l for l,n in calls if n=='partial_hess_elec')<min(l for l,n in calls if n=='make_h1')<min(l for l,n in calls if n=='solve_mo1')
    writes=[n.lineno for n in ast.walk(elec)if isinstance(n,ast.AugAssign)and rootname(n.target)=='de2'];assert len(writes)==3
    assert not any(isinstance(n,ast.AugAssign)and rootname(n.target)=='h1ao'for n in ast.walk(elec))
    assert not any(isinstance(n,ast.AugAssign)and rootname(n.target)=='h1ao'for n in ast.walk(solve))
    h1writes=[n.lineno for n in ast.walk(h1)if isinstance(n,ast.AugAssign)and rootname(n.target)=='h1ao'];assert len(h1writes)==3
    defaults=dict(zip([a.arg for a in elec.args.args][-len(elec.args.defaults):],elec.args.defaults));assert ast.literal_eval(defaults['max_memory'])==4000
    assert not any(isinstance(n,ast.Call)and callname(n)=='hess_nuc'for n in ast.walk(partial))
    base=next(n for n in trees['rhf'].body if isinstance(n,ast.ClassDef)and n.name=='HessianBase');kernel=next(n for n in base.body if isinstance(n,ast.FunctionDef)and n.name=='kernel')
    nuclear=[n.lineno for n in ast.walk(kernel)if isinstance(n,ast.Call)and callname(n)=='hess_nuc'];assert len(nuclear)==1
    assert not any(n=='numpy'or n.startswith('numpy.')or n=='pyscf'or n.startswith('pyscf.')for n in sys.modules)
    report={'schema':'research4-independent-V6-stage-static-contract-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
      'status':'PASS_PINNED_STOCK_DATA_FLOW_CONTRACT_ONLY','source_files':{k:{'path':str(p.relative_to(ROOT)),'sha256':d}for k,(p,d)in FILES.items()},
      'raw_partial':{'meaning':'Full electronic RKS partial=RHF one-electron+J-hybridK+XC; nuclear and orbital response excluded.',
        'shape':[75,75,3,3],'bytes_float64':75*75*3*3*8,'stock_in_place_response_write_lines':sorted(writes),
        'contract':'Return a fresh mutable owner each kernel call; never alias the stored partial, writable mmap, or prior assembly result. Do not add RHF/J/XC a second time.'},
      'H1AO':{'shape':[75,3,762,762],'bytes_float64':75*3*762*762*8,'RKS_build_in_place_lines':sorted(h1writes),
        'stock_hess_elec_and_solver_write_H1AO':False,'contract':'Global atom-indexed75 sequence, every(3,762,762) block complete and valid. Fresh disk loads can provide kernel ownership without retaining a second1GiB cache copy. Native generation mutates its own H1AO; consumers here read it.'},
      'global_assembly':{'stock_nuclear_call_lines':nuclear,'nuclear_added_once':True,
        'contract':'One stock hobj.kernel assembly, preserving all interatomic couplings. Nuclear is added once by kernel; partial cache must exclude it. PBE pilot remains no-dispersion/noNLC.'},
      'implicit_arguments':{'hess_elec_default_max_memory':4000,'MF_and_hobj_resource_context_max_memory':3000,
        'RKS_XC_native_memory_floor_MB':2000,
        'contract':'Bind actual bound-method args including defaults. None resolves to the same MF MO frame, full atom order; explicitMO args must match exact fingerprints. Do not substitute hobj.max_memory3000 for stock default call argument4000. H1 chkfile=None; logger identity is not a physical cache key.'},
      'response':{'block_atoms':8,'global_atoms':75,'MO1_shape':[75,3,762,160],'MO_E1_shape':[75,3,160,160],
        'MO1_bytes_float64':75*3*762*160*8,'MO_E1_bytes_float64':75*3*160*160*8,
        'contract':'Create full-AO/MO rhf.gen_vind once, preserve samefx for all8-atom RHS chunks. Each stock solve returns length75 global lists; retain realglobal indices, never renumber local chunks. Require complete0..74 coverage, no missing/duplicate atoms. CPHF conv_tol1e-8/maxcycle50/levelshift0 and native lindep remain unchanged; residual is NOT_COMPUTED.'},
      'atom_order':'OnlyNone or exact range(75); fullpermutations and subsets are forbidden because partial indices arelocal-atmlst while H1/MO1 lists use global atom IDs.',
      'binding_and_alias':'Before/after native phases bind geometry, atom/basis/env excluding only transient17, MOgauge/energies/occupations, actualgrid coordinates/weights and numerical controls. Scope instance wrappers with MethodType and restore missing/own attributes in finally. Call already-bound originals without an extra self.',
      'stage_gate':'Every cache stage needs actual healthy closed producer/proof/log/result/source/payload hashes. READY in a live child is insufficient. Kernel must not begin without both fullRKS partial and all75H1 blocks qualified.',
      'V6_acceptance_boundary':'V5 cachedRHF producers can be validated by frozenV5 verifier. NewV6 stage/final products require explicit new-source receipts and compatible versioned postprocessor/observer; pinnedV5 consumers do not accept new entry source automatically.',
      'suggested_stdlib_fixtures':[
        'Mutable list/deepcopy fakepartial: response changes returned copy; persistent cache and later return stay unchanged;核counter exactly1.',
        'inspect.signature fakebound methods: None/fullorder/explicit matchingframe accepted; changedMO, atompermutation/subset, maxmemory4001 and nonNonechkfile rejected.',
        'N11/B8 fakeglobal solver produces length11 with entries8..10 in remainder; verify samefx and onefactory, globalindexing, completecoverage and allpairterms.',
        'Exception injection: instance method dictionary and memory settings restored exactly, including previously missing attributes.',
        'Synthetic closed-stage JSON/filehash fixtures: parentRUNNING, missingH1atom, alteredbinding/payload or incompletepartial blocks all prevent kernel invocation.'],
      'new_V6_implementation_reviewed':False,'numerical_equivalence_validated':False,'stdlib_fixture_executions':0,
      'numeric_libraries_imported':False,'arrays_or_HDF_read':0,'QM_calls':0,'Git_operations':0,'frozen_files_modified':False,'audit_source_sha256':sha(__file__)}
    (OUT/'contract.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'README.md').write_text('Pinned-source static contract only. Stock RKS partial is full electronic partial, not the150 RHF helper units; stock hess_elec mutates its returned de2 withresponse, so stagedpartial requiresfresh owner. H1 generation mutatesH1AO, but stock assembly/solve read it. Fullglobal75atom indexing, actualdefaultmaxmemory4000 versusMF/Hessian3000, originalB8fulloperator response and exactly-once nuclear assembly must be preserved. Scope/restore wrappers and bind actual explicit/implicitMO arguments. No newV6 code or numerical equivalence is certified by this note; suggestedstdlib fixtures are designs, not executed tests. No numeric library, array, QM or Git operation occurred.\n')
    (OUT/'index.json').write_text(json.dumps({'status':'SEALED_STATIC_STOCK_CONTRACT','files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(OUT.iterdir())if p.is_file()and p.name!='index.json']},indent=2)+'\n')
    print(json.dumps({'status':report['status'],'contract_sha256':sha(OUT/'contract.json'),'index_sha256':sha(OUT/'index.json')}))

if __name__=='__main__':main()
