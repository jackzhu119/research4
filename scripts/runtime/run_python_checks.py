"""Record engineering runs on supplied examples, synthetic fixtures and unlabelled spectra."""
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings
from zoneinfo import ZoneInfo

import numpy as np
from scipy.io import loadmat
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from herb_spectra.io import read_spectra


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--only',nargs='*',help='Optional case IDs to rerun after a targeted change')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    inventory=json.loads(args.inventory.read_text())
    by_name={Path(r['archive']).name:r for r in inventory}
    results=[]
    def source(archive,suffix):
        return next(Path(x['path']) for x in by_name[archive]['members'] if x['name'].endswith(suffix))
    def run(test_id,fn,scope):
        if args.only and test_id not in args.only:return
        started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat();t=time.monotonic()
        try:metrics=fn();status='pass';error=None
        except Exception as e:metrics={};status='fail';error=f'{type(e).__name__}: {e}'
        result=dict(id=test_id,status=status,scope=scope,started_at_shanghai=started,duration_seconds=round(time.monotonic()-t,4),metrics=metrics,error=error)
        results.append(result);(args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False),flush=True)
    def units():
        cp=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],capture_output=True,text=True)
        (args.output/'unit_tests.log').write_text(cp.stdout+cp.stderr)
        if cp.returncode:raise RuntimeError('Unit tests failed; see unit_tests.log')
        return {'tests':7,'returncode':cp.returncode}
    run('python_existing_unit_suite',units,'synthetic engineering fixtures, not medicinal-plant performance')
    corn=loadmat(source('0308 CARS+UVE.rar','corn_m5_starch.mat'))
    for name,estimator in [('pls',PLSRegression(n_components=4,scale=False)),('svr',make_pipeline(StandardScaler(),SVR(C=10)))]:
        def regression(estimator=estimator):
            estimator.fit(corn['Xcal'],corn['ycal'].ravel());p=estimator.predict(corn['Xtest']).ravel()
            assert np.isfinite(p).all()
            if isinstance(estimator,PLSRegression):np.save(args.output/'python_pls_predictions.npy',p)
            return {'train_rows':60,'test_rows':20,'features':700,'example_rmsep':float(np.sqrt(mean_squared_error(corn['ytest'].ravel(),p)))}
        run('python_example_'+name,regression,'supplied corn/starch fixed split; example execution only')
    cnn=loadmat(source('0627卷积神经网络+PCA_Kmeans（可用2）.rar','data.mat'))
    lookup={}
    for i,row in enumerate(cnn['xtrain']):lookup.setdefault(row.tobytes(),[]).append(i)
    overlap=[]
    for i,row in enumerate(cnn['xtest']):
        if row.tobytes() in lookup:
            overlap.append({'test_index_0_based':i,'train_indices_0_based':lookup[row.tobytes()]})
    (args.output/'example_overlap.json').write_text(json.dumps({'train_rows':len(cnn['xtrain']),'test_rows':len(cnn['xtest']),'overlap_test_rows':len(overlap),'matches':overlap},indent=2)+'\n')
    def svm():
        estimator=make_pipeline(StandardScaler(),SVC(C=1,class_weight='balanced'))
        estimator.fit(cnn['xtrain'],cnn['ytrain'].ravel());p=estimator.predict(cnn['xtest'])
        return {'train_rows':252,'test_rows':108,'features':1007,'example_accuracy':float(accuracy_score(cnn['ytest'].ravel(),p))}
    run('python_example_svc',svm,'supplied non-herb multiclass example; not independent research validation')
    def real_io():
        warnings.filterwarnings('ignore',message='Workbook contains no default style')
        detail=[]
        for p in sorted(Path('data/raw').rglob('*.xlsx')):
            table=read_spectra(p,bounds=(0.1,1.5))
            assert table.values.shape[1]==612 and np.isfinite(table.values).all()
            detail.append({'file':p.as_posix(),'scans':table.values.shape[0],'frequency_points':612})
        return {'workbooks':len(detail),'unlabelled_scans':sum(x['scans'] for x in detail),'details':detail}
    run('python_real_herb_reading',real_io,'real unlabelled herb inputs; no classifier/regressor trained')
    def torch_cnn():
        import torch
        from torch import nn
        torch.set_num_threads(1);torch.manual_seed(42)
        def tensor(X):
            # MATLAB reshape uses column-major order. Transpose each C-order
            # 32x32 view so adjacent samples match that layout exactly.
            images=np.pad(X,((0,0),(0,17))).reshape(-1,32,32).transpose(0,2,1).copy()
            return torch.tensor(images,dtype=torch.float32)[:,None,:,:]
        fixture=tensor(np.arange(1007,dtype=float)[None,:])
        assert float(fixture[0,0,0,1])==32 and float(fixture[0,0,1,0])==1
        train=tensor(cnn['xtrain']);test=tensor(cnn['xtest'])
        mean=train.mean(dim=0,keepdim=True);train=train-mean;test=test-mean
        y=torch.tensor(cnn['ytrain'].ravel()-1,dtype=torch.long)
        truth=torch.tensor(cnn['ytest'].ravel()-1,dtype=torch.long)
        assert y.min()>=0 and y.max()<12
        model=nn.Sequential(nn.Conv2d(1,6,5,padding=2),nn.BatchNorm2d(6),nn.ReLU(),nn.MaxPool2d(2),
                            nn.Conv2d(6,16,5,padding=2),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
                            nn.Conv2d(16,120,5,padding=2),nn.BatchNorm2d(120),nn.ReLU(),nn.Flatten(),nn.Linear(120*8*8,12))
        optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
        losses=[]
        for epoch in range(3):
            model.train();order=torch.randperm(len(y));total=0.0
            for indices in order.split(32):
                optimizer.zero_grad();loss=nn.functional.cross_entropy(model(train[indices]),y[indices]);loss.backward();optimizer.step();total+=float(loss.detach())*len(indices)
            losses.append(total/len(y))
        model.eval()
        with torch.no_grad():p=model(test).argmax(dim=1);accuracy=float((p==truth).float().mean())
        assert all(np.isfinite(losses))
        return {'torch_version':torch.__version__,'device':'cpu','epochs':3,'train_rows':252,'test_rows':108,
                'parameters':sum(p.numel() for p in model.parameters()),'losses':losses,'example_accuracy':accuracy,
                'reshape_order':'MATLAB column-major',
                'scope_note':'PyTorch architecture migration; not original MATLAB weights/100-epoch reproduction'}
    run('python_cnn_structure_migration',torch_cnn,'equivalent 2D layer structure; 3-epoch CPU smoke test on supplied example')
    versions={x:importlib.metadata.version(x) for x in ['numpy','scipy','scikit-learn','openpyxl']}
    (args.output/'versions.json').write_text(json.dumps({'python':sys.version,**versions},indent=2)+'\n')


if __name__=='__main__':main()
