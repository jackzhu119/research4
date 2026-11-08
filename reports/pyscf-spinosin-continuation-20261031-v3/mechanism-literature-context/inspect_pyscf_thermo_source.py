#!/usr/bin/env python3
"""Read and AST-inspect installed PySCF source without importing/running PySCF."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[3])
ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
args=ap.parse_args()
repo=args.repo.resolve();out=args.output.resolve()
source=repo/'.runtime/quantum-upgrade-site/pyscf/hessian/thermo.py'
metadata=repo/'.runtime/quantum-upgrade-site/pyscf-2.14.0.dist-info/METADATA'
text=source.read_text();lines=text.splitlines()
version=next(line.partition(':')[2].strip() for line in metadata.read_text().splitlines() if line.startswith('Version:'))
tree=ast.parse(text)
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='harmonic_analysis')
keys=set()
for node in ast.walk(function):
 if isinstance(node,ast.Assign):
  for target in node.targets:
   if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id=='results' and isinstance(target.slice,ast.Constant):
    keys.add(target.slice.value)
expected={'freq_error','freq_au','freq_wavenumber','norm_mode','reduced_mass','vib_temperature','force_const_au','force_const_dyne'}
assert keys==expected
assert not any('ir' in k.lower() or 'intensity' in k.lower() for k in keys)
assert '#TODO: IR intensity' in '\n'.join(lines[function.lineno-1:function.end_lineno])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r={'status':'PASS_STATIC_SOURCE_INSPECTION_NO_QUANTUM','checked_utc':datetime.now(timezone.utc).isoformat(),'pyscf_version_from_package_metadata':version,'source_file':str(source.relative_to(repo)),'source_file_sha256':sha(source),'package_metadata_sha256':sha(metadata),'function_name':function.name,'function_line_start':function.lineno,'function_line_end':function.end_lineno,'literal_results_keys':sorted(keys),'IR_intensity_field_present':False,'IR_TODO_comment_present':True,'PySCF_imported':False,'function_executed':False,'new_quantum_computations':0,'checkpoint_reads':0,'official_source_url':'https://pyscf.org/_modules/pyscf/hessian/thermo.html','official_API_url':'https://pyscf.org/pyscf_api_docs/pyscf.hessian.html#pyscf.hessian.thermo.harmonic_analysis','source_license':'Apache-2.0; installed source credits Tencent America LLC (2019), author Qiming Sun','limitation':'This standard harmonic_analysis function returns frequencies/modes and related harmonic quantities. It does not supply dipole derivatives or IR intensities; independent verified implementations could compute intensities but are outside this inspection.'}
out.mkdir(parents=True,exist_ok=True)
for name in ('pyscf_harmonic_analysis_static_check.json','pyscf_harmonic_analysis_excerpt.txt'):
 if (out/name).exists():raise FileExistsError('Do not overwrite existing evidence: '+str(out/name))
(out/'pyscf_harmonic_analysis_static_check.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
selected=[]
for n,line in enumerate(lines,1):
 if (function.lineno<=n<=function.lineno+4) or (function.lineno<=n<=function.end_lineno and ("results['" in line or 'TODO: IR' in line or 'return results' in line)):
  selected.append(f'L{n}: {line}')
header=f'Static excerpt only. Source: {r["source_file"]}\nPySCF {version}; SHA256 {r["source_file_sha256"]}\nOfficial source: {r["official_source_url"]}\nCopyright2019 Tencent America LLC; author Qiming Sun; Apache License2.0 (https://www.apache.org/licenses/LICENSE-2.0).\nNo PySCF execution, quantum calculation or checkpoint read.\n\n'
(out/'pyscf_harmonic_analysis_excerpt.txt').write_text(header+'\n'.join(selected)+'\n')
print(json.dumps(r,indent=2))
