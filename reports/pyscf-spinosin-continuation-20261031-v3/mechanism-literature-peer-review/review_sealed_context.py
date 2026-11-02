"""Read-only literature/source consistency review; no quantum module imports."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import email.parser
import hashlib
import json
import re
import sys


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SEALED=ROOT/"reports/pyscf-spinosin-continuation-20261031-v3/mechanism-literature-context"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(title):return re.sub(r"\s+"," ",title).strip().rstrip('.').casefold()


def main():
    target=OUT/"review.json"
    if target.exists():raise FileExistsError("Preserve this completed peer review")
    index_path=SEALED/"files_sha256.json";index=json.loads(index_path.read_text())
    verified=[]
    for name,expected in index['files'].items():
        path=SEALED/name
        assert path.resolve().is_relative_to(SEALED.resolve()) and path.is_file()
        assert sha(path)==expected['sha256'] and path.stat().st_size==expected['bytes'],name
        verified.append({'file':name,'sha256':sha(path),'status':'UNCHANGED'})
    sources={row['id']:row for row in json.loads((SEALED/'sources.json').read_text())['sources']}
    observed=json.loads((OUT/'independent_metadata_records.json').read_text())['sources']
    metadata=[]
    for actual in observed:
        source=sources[actual['id']]
        checks={'title':normalize(actual['observed_title'])==normalize(source['title']),
                'DOI':actual['observed_doi'].casefold()==source['doi'].casefold(),
                'year':actual['observed_year']==source['year']}
        assert all(checks.values())
        metadata.append({'id':actual['id'],'checks':checks,'status':'MATCH_SCOPED_BACKGROUND_ONLY',
                         'claim_scope':actual['claim_scope'],'not_supported':actual['not_supported']})
    source=ROOT/'.runtime/quantum-upgrade-site/pyscf/hessian/thermo.py'
    text=source.read_text();tree=ast.parse(text)
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='harmonic_analysis')
    keys=set()
    for node in ast.walk(function):
        if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id=='results' and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str):
            keys.add(node.slice.value)
    expected={'force_const_au','force_const_dyne','freq_au','freq_error','freq_wavenumber','norm_mode','reduced_mass','vib_temperature'}
    assert keys==expected
    body='\n'.join(text.splitlines()[function.lineno-1:function.end_lineno])
    assert '#TODO: IR intensity' in body and not any('intens' in key.casefold() or key.casefold().startswith('ir') for key in keys)
    original=json.loads((SEALED/'pyscf_harmonic_analysis_static_check.json').read_text())
    assert sha(source)==original['source_file_sha256']
    package_metadata=ROOT/'.runtime/quantum-upgrade-site/pyscf-2.14.0.dist-info/METADATA'
    version=email.parser.Parser().parsestr(package_metadata.read_text())['Version']
    assert version=='2.14.0' and sha(package_metadata)==original['package_metadata_sha256']
    assert 'pyscf' not in sys.modules
    result={'status':'PASS_SCOPED_LITERATURE_METADATA_AND_STATIC_SOURCE_REVIEW',
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'sealed_index_sha256':sha(index_path),'sealed_files_checked':verified,
        'independent_metadata_checks':metadata,
        'S01_S04_focused_review':'See focused-check/ for independent official PubMed metadata and abstract scope review',
        'S06':{'status':'INSTALLED_STANDARD_FUNCTION_CONFIRMED_NO_IR_INTENSITY_OUTPUT',
            'source_version':version,'source_sha256':sha(source),'package_metadata_sha256':sha(package_metadata),
            'function_line_start':function.lineno,'function_line_end':function.end_lineno,'literal_results_keys':sorted(keys),
            'IR_TODO_present':True,'IR_intensity_output_present':False,'standard_function_only_not_all_PySCF_ecosystem':True},
        'new_quantum_evaluations':0,'PySCF_imported':False,'active_calculation_file_reads':0,'checkpoint_reads':0,
        'full_text_downloaded':False,'new_Spinosin_peak_values_generated':0,
        'not_a_systematic_review_or_nonexistence_proof':True,'classification_results_modified':False,
        'peer_review_source_sha256':sha(__file__)}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    assert all(sha(SEALED/entry['file'])==entry['sha256'] for entry in verified)
    print(json.dumps({'status':result['status'],'sealed_files':len(verified),'metadata_records':len(metadata),'S06_output_keys':sorted(keys),'new_quantum_evaluations':0},ensure_ascii=False))


if __name__=='__main__':main()
