#!/usr/bin/env python3
"""Independent small-water equivalence test for a reference-lifetime resource fix.

Only change under test: immediately before grids_response_becke yields coords,
w0,w1, delete the three completed partition temporaries pbecke,dpbecke,z.
Installed PySCF sources are inspected and hashed but never edited. No production
worker, sealed result, or large-molecule calculation is touched.
"""
import time
CPU_AT_ENTRY=time.process_time()
WALL_AT_ENTRY=time.monotonic()
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import inspect, hashlib, json, itertools, traceback
import numpy as np
from pyscf import gto,dft,lib,__version__ as pyscf_version
from pyscf.grad import rks as rks_grad

OUT=Path(__file__).resolve().parent
RESULT=OUT/'resource_water_test_result.json'
original=rks_grad.grids_response_becke
source=inspect.getsource(original)
source_file=Path(inspect.getsourcefile(original))
file_sha_before=hashlib.sha256(source_file.read_bytes()).hexdigest()
needle='        yield coords, w0, w1'
assert source.count(needle)==1
patched_source=source.replace(needle,'        del pbecke, dpbecke, z\n'+needle)
namespace=dict(vars(rks_grad))
exec(compile(patched_source,'<memory-only-water-resource-equivalence>','exec'),namespace)
patched=namespace['grids_response_becke']
assert patched is not original
lib.num_threads(1)
record={
 'status':'running','pyscf_version':pyscf_version,
 'scope':'H2O only: 3 atoms; no 75-atom DFT, production worker or sealed-result changes.',
 'source_file':str(source_file),
 'installed_source_file_sha256_before':file_sha_before,
 'original_grids_response_becke_inspect_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
 'patched_grids_response_becke_source_sha256':hashlib.sha256(patched_source.encode()).hexdigest(),
 'patch':'Immediately before yield coords,w0,w1 insert del pbecke, dpbecke, z.',
 'patch_effect':'Release completed intermediates before generator suspension. No arithmetic expression, quadrature, grid, contraction, accumulation order or returned array is changed.',
 'installed_source_modified':False,
 'calculation':{'atom_count':3,'geometry_Angstrom':[['O',0.0,0.0,0.0],['H',0.0,-0.757,0.587],['H',0.0,0.757,0.587]],'charge':0,'spin':0,'basis':'def2-SVP','functional':'PBE','DF_auxbasis':'def2-universal-jkfit','grid_level':5,'grid_response':True,'auxbasis_response':True,'threads':1,'max_memory_MB':2200,'SCF_conv_tol':1e-11,'SCF_conv_tol_grad':1e-8,'SCF_max_cycle':100},
 'timing':{'process_cpu_at_script_entry':CPU_AT_ENTRY}
}
try:
 t=time.process_time()
 mol=gto.M(atom=record['calculation']['geometry_Angstrom'],unit='Angstrom',basis='def2-SVP',charge=0,spin=0,symmetry=False,verbose=0,max_memory=2200)
 grids=dft.gen_grid.Grids(mol);grids.level=5
 assert grids.becke_scheme is dft.gen_grid.original_becke
 record['grid_scheme']=grids.becke_scheme.__name__
 a=original(grids);b=patched(grids);iterator_tests=[]
 missing=object()
 for atom_id,pair in enumerate(itertools.zip_longest(a,b,fillvalue=missing)):
  left,right=pair
  assert left is not missing and right is not missing
  loc=a.gi_frame.f_locals
  original_dp=loc['dpbecke']
  patched_loc=b.gi_frame.f_locals
  fields=[]
  for label,x,y in zip(['coords','w0','w1'],left,right):
   bitwise=bool(np.array_equal(x,y) and x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes())
   fields.append({'name':label,'shape':list(x.shape),'dtype':str(x.dtype),'bitwise_equal':bitwise,'original_owns_data':bool(x.flags.owndata),'patched_owns_data':bool(y.flags.owndata),'original_shares_memory_with_dpbecke':bool(np.shares_memory(x,original_dp))})
  item={'atom_id':atom_id,'symbol':mol.atom_symbol(atom_id),'fields':fields,'original_dpbecke_shape':list(original_dp.shape),'original_dpbecke_nbytes':original_dp.nbytes,'original_frame_partition_temporaries':[k for k in ['pbecke','dpbecke','z'] if k in loc],'patched_frame_partition_temporaries':[k for k in ['pbecke','dpbecke','z'] if k in patched_loc]}
  assert all(f['bitwise_equal'] for f in fields)
  assert all(f['original_owns_data'] and f['patched_owns_data'] and not f['original_shares_memory_with_dpbecke'] for f in fields)
  assert not item['patched_frame_partition_temporaries']
  iterator_tests.append(item)
 assert len(iterator_tests)==3
 record['grid_iterator_tests']=iterator_tests
 record['timing']['grid_equivalence_cpu_seconds']=time.process_time()-t
 del a,b,loc,patched_loc,original_dp,left,right,x,y,pair
 # One strict, shared converged reference removes SCF optimization differences.
 t=time.process_time()
 mf=dft.RKS(mol).density_fit(auxbasis='def2-universal-jkfit')
 mf.xc='PBE';mf.grids.level=5;mf.max_memory=2200
 mf.conv_tol=1e-11;mf.conv_tol_grad=1e-8;mf.max_cycle=100
 mf.chkfile=None
 energy=float(mf.kernel())
 assert mf.converged
 record['SCF']={'converged':bool(mf.converged),'energy_Hartree':energy,'cycles':int(mf.cycles),'nao':int(mol.nao),'naux':int(mf.with_df.auxmol.nao),'reference':'Same converged mf used for both gradient calls. No SCF rerun, grid change, basis change or response removal.'}
 record['timing']['SCF_cpu_seconds']=time.process_time()-t
 t=time.process_time()
 ga=mf.nuc_grad_method();ga.grid_response=True;ga.auxbasis_response=True;ga.max_memory=2200
 with lib.temporary_env(rks_grad,grids_response_becke=original):
  grad_original=np.array(ga.kernel(),copy=True)
 record['timing']['original_gradient_cpu_seconds']=time.process_time()-t
 t=time.process_time()
 gb=mf.nuc_grad_method();gb.grid_response=True;gb.auxbasis_response=True;gb.max_memory=2200
 with lib.temporary_env(rks_grad,grids_response_becke=patched):
  grad_patched=np.array(gb.kernel(),copy=True)
 record['timing']['patched_gradient_cpu_seconds']=time.process_time()-t
 delta=grad_patched-grad_original
 record['gradient_test']={'units':'Hartree/Bohr','original':grad_original.tolist(),'patched':grad_patched.tolist(),'array_equal':bool(np.array_equal(grad_original,grad_patched)),'bytes_equal':bool(grad_original.tobytes()==grad_patched.tobytes()),'max_abs_difference':float(np.max(np.abs(delta))),'norm_difference':float(np.linalg.norm(delta)),'allclose_atol_1e_minus_12_rtol_0':bool(np.allclose(grad_original,grad_patched,atol=1e-12,rtol=0))}
 assert record['gradient_test']['bytes_equal']
 assert rks_grad.grids_response_becke is original
 record['installed_source_file_sha256_after']=hashlib.sha256(source_file.read_bytes()).hexdigest()
 assert record['installed_source_file_sha256_after']==file_sha_before
 record['restored_original_function_identity']=True
 record['status']='passed'
except BaseException as exc:
 record['status']='failed';record['exception_type']=type(exc).__name__;record['exception']=str(exc);record['traceback']=traceback.format_exc()
 raise
finally:
 record['timing']['process_cpu_seconds_including_imports']=time.process_time()
 record['timing']['elapsed_wall_seconds']=time.monotonic()-WALL_AT_ENTRY
 record['timing']['CPU_accounting_note']='process_time() total for this single-threaded process, sampled after all scientific work; includes interpreter startup before script entry and library imports. Does not include orchestration shell CPU.'
 RESULT.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':record['status'],'result_path':str(RESULT),'cpu_seconds_including_imports':record['timing']['process_cpu_seconds_including_imports'],'wall_seconds':record['timing']['elapsed_wall_seconds'],'gradient':record.get('gradient_test',{}),'original_function_source_SHA256':record['original_grids_response_becke_inspect_source_sha256']},ensure_ascii=False),flush=True)
