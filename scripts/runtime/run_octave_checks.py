"""Execute legacy entry points and small core-function checks in isolated processes.

The source inventory points to separately extracted, immutable uploads. Every
test receives a fresh copy. Results distinguish unchanged entries, core calls,
and explicitly recorded compatibility edits. No MATLAB result is implied.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--octave", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=["first", "compatibility", "focused", "vectorized", "final", "ssa-full", "targeted", "targeted-corrected", "texture-array"], default="first")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.inventory.read_text())
    by_name = {Path(r["archive"]).name: r for r in records}
    jobs = []

    def add(test_id, archive, member, body=None, stage="original_entry", edits=None, timeout=45, recipe=""):
        jobs.append(dict(id=test_id, archive=archive, member=member, body=body,
                         stage=stage, edits=edits or [], timeout=timeout,recipe=recipe))

    add("raw_cars_uve_main", "0308 CARS+UVE.rar", "main.m", timeout=90)
    add("raw_cars_sy_main", "CARS（sy）.rar", "main.m", timeout=90)
    add("raw_uve_regression_demo", "UVE.zip", "demo_PLS_Regression.m", timeout=60)
    add("raw_uve_classification_demo", "UVE.zip", "demo_PLS_Discriminant_Analysis.m", timeout=60)
    for name in ["main0627", "main_cnn0629", "cnn_v2", "cnn_v21", "main_SVM", "main_BPNN"]:
        add("raw_cnn_bundle_"+name, "0627卷积神经网络+PCA_Kmeans（可用2）.rar", name+".m")
    for name in ["BP", "BPDLX"]:
        add("raw_audio_"+name, "BPNN.zip", name+".m", timeout=60)
    for name in ["main_SVM", "main_BPNN"]:
        add("raw_svm_bp_"+name, "支持向量机和BPNN(可用).zip", name+".m")
    add("raw_ssa_main", "麻雀算法优化SVM-沈-2021-0802.zip", "SSA_SVM_main.m", timeout=60)
    for name in ["SVM", "spectralprocessing", "zsp3PCAplot_DL"]:
        add("raw_collection_"+name, "支持向量机和主成分分析及特征提取程序.rar", name+".m")

    # Small deterministic checks use supplied example data, not medicinal-plant labels.
    core_data = "load('corn_m5_starch.mat');"
    add("core_pls_original", "0308 CARS+UVE.rar", "pls.m", body=core_data+" P=pls(Xcal,ycal,4,'center'); pred=[Xtest ones(rows(Xtest),1)]*P.coef_origin(:,end); assert(all(isfinite(pred))); metrics=struct('n_train',rows(Xcal),'n_test',rows(Xtest),'n_features',columns(Xcal),'example_rmsep',sqrt(mean((pred-ytest).^2))); save('-mat7-binary','cross_language_pls.mat','pred','Xcal','Xtest','ycal','ytest');", stage="core_function")
    add("core_cars_uve", "0308 CARS+UVE.rar", "carspls.m", body=core_data+" C=carspls(Xcal,ycal,4,3,'center',10,0,0,0); assert(numel(C.vsel)>0); metrics=struct('variables',numel(C.vsel),'selected_lv',C.optLV,'selection_cv',C.RMSECV_min);", stage="core_function")
    add("core_cars_sy", "CARS（sy）.rar", "carspls.m", body="load('corn_m51.mat'); C=carspls(X,y,4,3,'center',10); assert(numel(C.vsel)>0); metrics=struct('variables',numel(C.vsel),'reported_cv',C.minRMSECV);", stage="core_function")
    add("core_mcuve_uve_zip", "UVE.zip", "mcuvepls.m", body="load('corn_m51.mat'); U=mcuvepls(X,y,4,'center',30,0.7); assert(numel(U.RI)==columns(X)); assert(all(isfinite(U.RI))); metrics=struct('features',columns(X),'runs',30,'finite_stability',true);", stage="core_function")
    add("core_mcuve_0308", "0308 CARS+UVE.rar", "mcuvepls.m", body=core_data+" U=mcuvepls(Xcal,ycal,4,'center',30,0.7); assert(numel(U.RI)==columns(Xcal)); assert(all(isfinite(U.RI))); metrics=struct('features',columns(Xcal),'runs',30,'finite_stability',true);", stage="core_function")
    add("core_uve_noise", "0308 CARS+UVE.rar", "uvepls.m", body=core_data+" U=uvepls(Xcal,ycal,4,3,'center',6,40,0.99); assert(numel(U.vsel)>0); metrics=struct('variables',numel(U.vsel),'jackknife_groups',6);", stage="core_function", timeout=60)
    add("core_pls_cv", "UVE.zip", "plscv.m", body="load('corn_m51.mat'); C=plscv(X,y,4,3,'center',0); metrics=struct('example_cv',C.RMSECV_min,'selected_lv',C.optLV);", stage="core_function")
    add("core_plslda", "UVE.zip", "plslda.m", body="load('DM2.mat'); L=plslda(Xcal,ycal,3,'autoscaling',0); assert(all(isfinite(L.yfit))); metrics=struct('rows',rows(Xcal),'features',columns(Xcal),'training_error_only',L.error);", stage="core_function")
    add("core_iriv", "UVE.zip", "iriv.m", body="X=randn(48,10); y=3*X(:,1)-X(:,4)+0.1*randn(48,1); F=iriv(X,y,3,3,'center'); disp(fieldnames(F)); metrics=struct('synthetic_rows',48,'features',10);", stage="core_function", timeout=90)
    add("core_random_frog", "UVE.zip", "randomfrog_pls.m", body="load('corn_m51.mat'); F=randomfrog_pls(X,y,3,'center',100,5); assert(numel(F.probability)==columns(X)); metrics=struct('runs',100,'features',columns(X));", stage="core_function", timeout=90)
    add("core_pca_kmeans", "0627卷积神经网络+PCA_Kmeans（可用2）.rar", "main0627.m", body="load('data.mat'); [coeff,scores,latent,tsq,explained,mu]=pca(xtrain); k=min(10,columns(scores)); projected=(xtest-mu)*coeff(:,1:k); idx=kmeans(scores(:,1:k),12,'Replicates',1); assert(rows(projected)==rows(xtest)); metrics=struct('train_rows',rows(xtrain),'test_rows',rows(xtest),'features',columns(xtrain),'pca_components',k,'clusters',numel(unique(idx)));", stage="core_function")
    add("core_libsvm", "0627卷积神经网络+PCA_Kmeans（可用2）.rar", "main0627.m", body="load('data.mat'); mu=mean(xtrain); sd=std(xtrain); sd(sd==0)=1; a=(xtrain-mu)./sd; b=(xtest-mu)./sd; M=svmtrain(ytrain,a,sprintf('-q -c 1 -g %.16g',1/columns(a))); [p,acc]=svmpredict(ytest,b,M,'-q'); assert(numel(p)==rows(xtest)); metrics=struct('example_accuracy_percent',acc(1),'train_rows',rows(a),'test_rows',rows(b));", stage="core_function")

    if args.phase in {"compatibility", "focused"}:
        jobs=[]
        for name in ["SVM（可用）","BPNN（可用）","cnn_4","main","PLS","PLS-Prediction-shenyin","PLS1","LSSVM","syqpcx程序(1)","平滑曲线"]:
            add("raw_text_"+name,"standalone_texts",name+".m",edits=["TXT copied byte-for-byte to .m for execution"])
        add("raw_docx_edge","docx_code","edge_detection.m",edits=["DOCX paragraph text extracted to .m"])
        add("core_docx_texture","docx_code","Texture.m",body="imwrite(uint8(255*rand(256,256,3)),'fixture.png'); T=Texture('fixture.png'); assert(numel(T)==8);",stage="core_function",edits=["DOCX paragraph text extracted to .m"])
        add("compat_pls_original","0308 CARS+UVE.rar","pls.m",body=core_data+" P=pls(Xcal,ycal,4,'center'); pred=[Xtest ones(rows(Xtest),1)]*P.coef_origin(:,end); assert(all(isfinite(pred))); metrics=struct('example_rmsep',sqrt(mean((pred-ytest).^2))); save('-mat7-binary','cross_language_pls.mat','pred','Xcal','Xtest','ycal','ytest');",stage="compatibility",recipe="sumsqr",edits=["Add sumsqr compatibility helper; preserve PLS implementation"])
        add("compat_cars_uve_main","0308 CARS+UVE.rar","main.m",stage="compatibility",recipe="sumsqr",edits=["Add sumsqr compatibility helper; original default main settings"],timeout=120)
        add("compat_cars_sy_main","CARS（sy）.rar","main.m",stage="compatibility",recipe="sumsqr",edits=["Add sumsqr compatibility helper; original default main settings"],timeout=120)
        add("compat_uve_noise","0308 CARS+UVE.rar","uvepls.m",body=core_data+" U=uvepls(Xcal,ycal,4,3,'center',6,40,0.99); assert(numel(U.vsel)>0); metrics=struct('variables',numel(U.vsel),'jackknife_groups',6);",stage="compatibility",recipe="sumsqr",edits=["Add sumsqr helper; six jackknife groups for smoke test"])
        add("compat_mcuve_0308","0308 CARS+UVE.rar","mcuvepls.m",body=core_data+" U=mcuvepls(Xcal,ycal,4,'center',30,0.7); assert(all(isfinite(U.RI))); metrics=struct('features',columns(Xcal),'runs',30);",stage="compatibility",recipe="sumsqr",edits=["Add sumsqr helper; 30 Monte Carlo runs for smoke test"])
        cars_metric="load('corn_m51.mat'); C=carspls(X,y,4,3,'center',10); V=plscvfold(X(:,C.vsel),y,4,3,'center',0); metrics=struct('raw_subset_cv',V.RMSECV,'reported_cv',C.minRMSECV,'raw_to_reported_ratio',V.RMSECV/C.minRMSECV,'variables',C.vsel);"
        add("verify_cars_error_scaling","CARS（sy）.rar","carspls.m",body=cars_metric+" assert(abs(metrics.raw_to_reported_ratio-2.8)<1e-7);",stage="isolated_defect",recipe="sumsqr",edits=["Add sumsqr helper; independently recompute selected-subset CV"])
        add("patched_cars_error_scaling","CARS（sy）.rar","carspls.m",body=cars_metric+" assert(abs(metrics.raw_to_reported_ratio-1)<1e-7);",stage="logic_patch",recipe="sumsqr,cars_scaling",edits=["Add sumsqr helper; remove /2.8 in two reported error fields"])
        add("verify_iriv_small_dimension_hang","UVE.zip","iriv.m",body="X=randn(48,3); y=3*X(:,1)-X(:,3)+0.1*randn(48,1); F=iriv(X,y,2,3,'center');",stage="isolated_defect",timeout=4)
        add("patched_iriv_small_dimension","UVE.zip","iriv.m",body="X=randn(48,10); y=3*X(:,1)-X(:,4)+0.1*randn(48,1); F=iriv(X,y,3,3,'center'); assert(numel(F.SelectedVariables)>0); metrics=struct('selected',F.SelectedVariables);",stage="logic_patch",recipe="iriv_guard",edits=["Stop random-screening loop at <=4 variables, hand off to existing backward elimination; Nx() -> Nx"],timeout=60)
        add("compat_libsvm_three_outputs","0627卷积神经网络+PCA_Kmeans（可用2）.rar","main0627.m",body="load('data.mat'); mu=mean(xtrain); sd=std(xtrain); sd(sd==0)=1; a=(xtrain-mu)./sd; b=(xtest-mu)./sd; M=svmtrain(ytrain,a,sprintf('-q -c 1 -g %.16g',1/columns(a))); [p,acc,dec]=svmpredict(ytest,b,M,'-q'); assert(numel(p)==rows(xtest)); metrics=struct('example_accuracy_percent',acc(1),'train_rows',rows(a),'test_rows',rows(b));",stage="compatibility",edits=["Request all three outputs required by this Octave LIBSVM interface"])
        add("compat_pca_main","0627卷积神经网络+PCA_Kmeans（可用2）.rar","main0627.m",stage="compatibility",recipe="mapminmax",edits=["Add limited mapminmax train/apply helper; no PCA logic fix"])
        add("compat_audio_bp","BPNN.zip","BP.m",stage="compatibility",recipe="mapminmax,remove_web",edits=["Remove browser-launch statements; add limited mapminmax helper; retain original 10-epoch manual BP"],timeout=180)
        add("compat_svm_bp_locals","支持向量机和BPNN(可用).zip","main_SVM.m",stage="compatibility",recipe="split_locals,crossvalind",edits=["Split script-local functions into files; add Kfold partition helper"])
        add("compat_cnn_svm_locals","0627卷积神经网络+PCA_Kmeans（可用2）.rar","main_SVM.m",stage="compatibility",recipe="split_locals,crossvalind",edits=["Split script-local functions; add Kfold partition helper"])
        add("compat_cnn_bp_locals","0627卷积神经网络+PCA_Kmeans（可用2）.rar","main_BPNN.m",stage="compatibility",recipe="split_locals,crossvalind",edits=["Split script-local functions; add Kfold partition helper"])
        add("compat_ssa_smoke","麻雀算法优化SVM-沈-2021-0802.zip","SSA_SVM_main.m",stage="compatibility",recipe="split_locals,crossvalind,ssa_smoke",edits=["Split local functions; Kfold helper; three-output svmpredict call; reduce population to 6 and iterations to 3"],timeout=90)
        add("compat_edge_fixture","docx_code","edge_detection.m",stage="compatibility",recipe="edge_fixture",body="imwrite(uint8(255*rand(32,32,3)),'fixture.png'); run('edge_detection.m'); assert(isequal(size(output),size(output1))); metrics=struct('output_rows',rows(output),'max_difference',max(abs(output(:)-output1(:))));",edits=["Replace absolute input bitmap path with synthetic 32x32 RGB fixture"])
        add("verify_syqpcx_concat","standalone_texts","syqpcx程序(1).m",stage="isolated_defect",recipe="syqpcx_fixture",edits=["Replace external input files with 91 synthetic 300x92 matrices; preserve original slice and concatenation lines"])
        add("compat_pca_image_function","standalone_texts","PCA.m",stage="compatibility",body="copyfile('PCA.m','extract_pca.m'); extract_pca(rand(16,16,4));",edits=["Rename function file to match declared extract_pca name; synthetic input image"])
        add("compat_sg_filter","standalone_texts","平滑曲线.m",stage="compatibility",body="x=sin((1:101)'/10)+0.05*randn(101,1); z=sgolayfilt(x,2,15); assert(numel(z)==numel(x)); assert(all(isfinite(z))); metrics=struct('points',101,'window',15,'degree',2);",edits=["Use Octave signal sgolayfilt API for 15-point SG filtering; not a full MATLAB smooth equivalence test"])
    if args.phase in {"focused", "final", "ssa-full"}:
        jobs=[]
        add('compat_audio_bp_complete','BPNN.zip','BP.m',stage='compatibility',recipe='mapminmax,rands,remove_web',edits=['Remove browser calls; limited mapminmax and uniform rands compatibility helpers; retain original 10-epoch BP'],timeout=180)
        add('compat_pca_main_complete','0627卷积神经网络+PCA_Kmeans（可用2）.rar','main0627.m',stage='compatibility',recipe='mapminmax,pca_api',edits=['mapminmax helper; legend NW -> northwest; nearest-centroid pdist2 Smallest option expressed with min'],timeout=60)
        add('compat_ssa_smoke_complete','麻雀算法优化SVM-沈-2021-0802.zip','SSA_SVM_main.m',stage='compatibility',recipe='split_locals,crossvalind,ssa_smoke,plot_apis',edits=['Split local functions; Kfold helper; svmpredict three outputs; plot API adaptations; omit PNG exports; 6 agents x 3 iterations'],timeout=120)
        add('compat_ssa_full_budget','麻雀算法优化SVM-沈-2021-0802.zip','SSA_SVM_main.m',stage='compatibility',recipe='split_locals,crossvalind,ssa_full,plot_apis',edits=['Split local functions; Kfold helper; svmpredict three outputs; plot API adaptations; omit PNG exports; retain original 10 agents x 100 iterations'],timeout=180)
        add('compat_uve_regression_demo_complete','UVE.zip','demo_PLS_Regression.m',stage='compatibility',recipe='plotmcuve',edits=['Add missing plotmcuvepls visualization helper; original demo model settings retained'],timeout=120)
        for name,rows,features in [('PLS',39,40),('PLS-Prediction-shenyin',157,160),('PLS1',80,200)]:
            add('fixture_text_'+name,'standalone_texts',name+'.m',stage='input_fixture',recipe='regression_fixture',edits=[f'Replace absolute Excel path with synthetic {rows}-row {features}-feature numeric fixture; preserve model logic'],timeout=60)
        add('fixture_lssvm_missing_training_api','standalone_texts','LSSVM.m',stage='input_fixture',recipe='lssvm_fixture',edits=['Provide synthetic 2000-row regressionData.mat to isolate missing training API'],timeout=30)
        if args.phase=='final':
            jobs=[j for j in jobs if j['id'] in {'compat_ssa_smoke_complete','compat_ssa_full_budget','fixture_text_PLS','fixture_text_PLS-Prediction-shenyin','fixture_text_PLS1'}]
            for j in jobs:
                if 'regression_fixture' in j['recipe']:
                    j['recipe']='regression_mat_fixture'
                    j['edits']=['Inject a dimension-matched synthetic MAT matrix in place of external Excel input; preserve model logic']
        if args.phase=='ssa-full':
            jobs=[j for j in jobs if j['id']=='compat_ssa_full_budget']
            jobs[0]['id']='compat_ssa_full_budget_long'
            jobs[0]['timeout']=600
    if args.phase == 'vectorized':
        jobs=[]
        add('bp_gradient_equivalence','BPNN.zip','BP.m',stage='equivalence_check',body="x=randn(24,1); W1=randn(25,24);W2=randn(25,4);b1=randn(25,1);b2=randn(4,1); target=randn(4,1);z=1./(1+exp(-(W1*x+b1)));e=target-(W2'*z+b2); G=zeros(24,25);q=zeros(25,1); for k=1:24;for j=1:25;G(k,j)=z(j)*(1-z(j))*x(k)*(e(1)*W2(j,1)+e(2)*W2(j,2)+e(3)*W2(j,3)+e(4)*W2(j,4));q(j)=z(j)*(1-z(j))*(e(1)*W2(j,1)+e(2)*W2(j,2)+e(3)*W2(j,3)+e(4)*W2(j,4));end;end;v=(z.*(1-z)).*(W2*e);difference=max([max(abs(G'-v*x')(:)),max(abs(q-v))]);assert(difference<1e-12);metrics=struct('max_gradient_difference',difference);",edits=['Independent loop-vs-matrix check of the uploaded BP gradient'])
        add('vectorized_audio_bp','BPNN.zip','BP.m',stage='performance_patch',recipe='mapminmax,rands,remove_web,vectorize_bp',edits=['Headless/mapminmax/rands compatibility helpers; algebraically equivalent vectorized sample gradient; original 10 epochs, 1500 train and 500 test rows'],timeout=90)
    if args.phase in {'targeted','targeted-corrected'}:
        jobs=[]
        add('texture_image_independence_defect','docx_code','Texture.m',stage='isolated_defect',recipe='texture_small',body="imwrite(uint8(zeros(16,16,3)),'zero.png');imwrite(uint8(255*rand(16,16,3)),'random.png');a=Texture('zero.png');b=Texture('random.png');assert(all(isfinite(a)));assert(max(abs(a-b))<1e-12);metrics=struct('fixture_size',16,'max_feature_difference',max(abs(a-b)),'first_features',a,'finding','stored GLCM conditions do not use the image values');",edits=['Reduce hard-coded loop dimensions from 256 to 16 for a defect fixture; disable repeated Octave short-circuit warning; preserve stored GLCM conditions'],timeout=30)
        add('pca_centering_defect','0627卷积神经网络+PCA_Kmeans（可用2）.rar','main0627.m',stage='isolated_defect',recipe='mapminmax',body="load('data.mat');[a,ps]=mapminmax(xtrain',0,1);a=a';b=mapminmax('apply',xtest',ps)';[L,S,latent,tsq,explained,mu]=pca(a);C=kmeans(S(:,1:19),12,'Replicates',1);old=b*L(:,1:19);correct=(b-mu)*L(:,1:19);[~,p1]=min(pdist2(C,old),[],1);[~,p2]=min(pdist2(C,correct),[],1);metrics=struct('test_rows',rows(b),'changed_cluster_assignments',sum(p1~=p2),'projection_shift_l2',norm(mu*L(:,1:19)));assert(metrics.projection_shift_l2>0);",edits=['Use the same training-only minmax/PCA fit to compare missing vs applied training mean'])
        add('compat_provided_newff_dependency','支持向量机和BPNN(可用).zip','main_BPNN.m',stage='compatibility',recipe='split_locals,crossvalind',edits=['Split local functions and add Kfold helper; preserve the provided newff.m to identify its actual remaining dependency'],timeout=60)
        if args.phase=='targeted-corrected':
            jobs=jobs[:2]
            jobs[0]['body']=jobs[0]['body'].replace("imwrite(uint8(zeros(16,16,3)),'zero.png')","imwrite(cat(3,uint8(255*ones(16)),uint8(zeros(16)),uint8(zeros(16))),'zero.png')")
            jobs[0]['edits'].append('Use distinct RGB channels so PNG encoding does not convert the first fixture to grayscale')
            jobs[1]['body']=jobs[1]['body'].replace('C=kmeans','[training_idx,C]=kmeans')
    if args.phase=='texture-array':
        jobs=[]
        add('texture_direct_rgb_array','docx_code','Texture.m',stage='isolated_defect',recipe='texture_small,texture_array',body="a=Texture(uint8(cat(3,255*ones(16),zeros(16),zeros(16))));b=Texture(uint8(255*rand(16,16,3)));assert(all(isfinite(a)));assert(max(abs(a-b))<1e-12);metrics=struct('fixture_size',16,'max_feature_difference',max(abs(a-b)),'first_features',a,'finding','stored GLCM conditions do not use image values');",edits=['Reduce hard-coded loop dimensions to 16; pass RGB arrays directly to bypass palette PNG decoding; preserve stored GLCM conditions'],timeout=30)

    prelude = "warning('off','Octave:shadowed-function'); warning('off','Octave:gnuplot-graphics'); pkg load statistics signal io image optim; set(0,'defaultfigurevisible','off'); graphics_toolkit('gnuplot'); rand('seed',42); randn('seed',42);\n"
    results = []
    for job in jobs:
        base = by_name[job["archive"]]
        matches = [m for m in base["members"] if Path(m["name"]).name == job["member"]]
        if len(matches) != 1:
            raise ValueError((job["id"],matches))
        member = matches[0]
        folder = args.output/job["id"]
        work = folder/"work"
        work.mkdir(parents=True,exist_ok=True)
        shutil.copytree(base["folder"],work,dirs_exist_ok=True)
        target = work/member["name"]
        if job['recipe']:
            # Converted work copies are separate from the immutable extracted originals.
            for mfile in work.rglob('*.m'):
                raw=mfile.read_bytes()
                for encoding in ['utf-8-sig','gb18030','latin-1']:
                    try:text=raw.decode(encoding);break
                    except UnicodeError:pass
                mfile.write_text('\n'.join(text.splitlines())+'\n')
            recipe=set(job['recipe'].split(','))
            compat=target.parent
            if 'sumsqr' in recipe:
                (compat/'sumsqr.m').write_text('function s=sumsqr(x)\n s=sum(x(:).^2);\nend\n')
            if 'mapminmax' in recipe:
                (compat/'mapminmax.m').write_text("function [z,ps]=mapminmax(x,varargin)\n if ischar(x); ps=varargin{2}; x=varargin{1}; else; a=-1;b=1;if numel(varargin)>0;a=varargin{1};end;if numel(varargin)>1;b=varargin{2};end;ps.xmin=min(x,[],2);ps.range=max(x,[],2)-ps.xmin;ps.range(ps.range==0)=1;ps.a=a;ps.b=b;end; z=(x-ps.xmin)./ps.range*(ps.b-ps.a)+ps.a;\nend\n")
            if 'crossvalind' in recipe:
                (compat/'crossvalind.m').write_text("function folds=crossvalind(mode,n,k)\n assert(strcmpi(mode,'Kfold')); folds=zeros(n,1); order=randperm(n);folds(order)=1+mod(0:n-1,k);\nend\n")
            if 'rands' in recipe:
                (compat/'rands.m').write_text('function z=rands(varargin)\n z=2*rand(varargin{:})-1;\nend\n')
            if 'plotmcuve' in recipe:
                (compat/'plotmcuvepls.m').write_text("function plotmcuvepls(U)\n figure;plot(abs(U.RI));xlabel('Variable');ylabel('Absolute coefficient stability');\nend\n")
            if 'pca_api' in recipe:
                text=target.read_text().replace("'NW'","'northwest'")
                text=text.replace("pdist2(C,pca_testx,'euclidean','Smallest',1)","min(pdist2(C,pca_testx,'euclidean'),[],1)")
                target.write_text(text)
            if 'remove_web' in recipe:
                target.write_text(re.sub(r'(?m)^web browser.*$','% Browser call omitted for headless execution',target.read_text()))
            if 'vectorize_bp' in recipe:
                text=target.read_text()
                start=text.index('%% 网络训练');end=text.index('%% 语音特征信号分类')
                block="""%% Vectorized equivalent of the uploaded samplewise BP update
for ii=1:10
 E(ii)=0;
 for i=1:1500
  x=inputn(:,i); z=1./(1+exp(-(w1*x+b1))); e=output_train(:,i)-(w2'*z+b2);
  E(ii)=E(ii)+sum(abs(e)); g=(z.*(1-z)).*(w2*e);
  w1=w1+xite*g*x'; b1=b1+xite*g; w2=w2+xite*z*e'; b2=b2+xite*e;
 end
end
"""
                target.write_text(text[:start]+block+text[end:])
            if 'cars_scaling' in recipe:
                target.write_text(target.read_text().replace('RMSEP/2.8','RMSEP').replace('Rmin/2.8','Rmin'))
            if 'iriv_guard' in recipe:
                text=target.read_text().replace('for i=1:Nx()','for i=1:Nx')
                text=text.replace('[~,Nx]=size(X);\n    % determine','[~,Nx]=size(X);\n    if Nx<=4; break; end;\n    % determine')
                target.write_text(text)
            if 'ssa_smoke' in recipe:
                text=target.read_text().replace('SearchAgents_no = 10','SearchAgents_no = 6').replace('Max_iteration   = 100','Max_iteration   = 3')
                text=text.replace('Ytest = svmpredict','[Ytest,svm_acc,svm_dec] = svmpredict')
                target.write_text(text)
            if 'ssa_full' in recipe:
                target.write_text(target.read_text().replace('Ytest = svmpredict','[Ytest,svm_acc,svm_dec] = svmpredict'))
            if 'plot_apis' in recipe:
                (compat/'plotconfusion.m').write_text("function plotconfusion(t,p)\n [~,a]=max(t,[],1);[~,b]=max(p,[],1);C=accumarray([a(:),b(:)],1,[rows(t),rows(t)]);imagesc(C);axis image;colorbar;\nend\n")
                text=target.read_text()
                text=re.sub(r'(?m)^print\([^\n]*$','% PNG export omitted in headless compatibility run',text)
                text=re.sub(r"(?m)^h\.(XTickLabel|YTickLabel|XTickLabelRotation)\s*=\s*(.*);\s*(%[^\n]*)?$",lambda m:"set(h,'"+m.group(1)+"',"+m.group(2)+");",text)
                target.write_text(text)
            if 'regression_fixture' in recipe:
                from openpyxl import Workbook
                import numpy as np
                name=target.stem
                n,p={'PLS':(39,40),'PLS-Prediction-shenyin':(157,160),'PLS1':(80,200)}[name]
                rng=np.random.default_rng(42);X=rng.normal(size=(n,p));y=3*X[:,0]-X[:,2]+rng.normal(scale=.1,size=n)
                data=np.column_stack([y,X]);book=Workbook();sheet=book.active
                for row in data:sheet.append(row.tolist())
                book.save(target.parent/'fixture.xlsx')
                target.write_text(re.sub(r"xlsread\('[^']*'", "xlsread('fixture.xlsx'", target.read_text()))
            if 'regression_mat_fixture' in recipe:
                from scipy.io import savemat
                import numpy as np
                n,p={'PLS':(39,40),'PLS-Prediction-shenyin':(157,160),'PLS1':(80,200)}[target.stem]
                rng=np.random.default_rng(42);X=rng.normal(size=(n,p));y=3*X[:,0]-X[:,2]+rng.normal(scale=.1,size=n)
                savemat(target.parent/'fixture.mat',{'fixture_data':np.column_stack([y,X])})
                target.write_text(re.sub(r'data=xlsread\([^;\n]+\);',"load('fixture.mat');data=fixture_data;",target.read_text()))
            if 'lssvm_fixture' in recipe:
                from scipy.io import savemat
                import numpy as np
                rng=np.random.default_rng(42);X=rng.normal(size=(2000,5));y=(2*X[:,0]-X[:,2])[None,:]
                savemat(target.parent/'regressionData.mat',{'input':X,'output':y})
            if 'split_locals' in recipe:
                text=target.read_text();matches=list(re.finditer(r'(?m)^function\b[^\n]+',text))
                if matches:
                    target.write_text(text[:matches[0].start()])
                    for i,match in enumerate(matches):
                        first=match.group();function_name=re.search(r'(?:=\s*|^function\s+)([A-Za-z_]\w*)\s*\(',first).group(1)
                        (target.parent/(function_name+'.m')).write_text(text[match.start():matches[i+1].start() if i+1<len(matches) else len(text)])
            if 'edge_fixture' in recipe:
                target.write_text(re.sub(r"imread\('[^']*'\)","imread('fixture.png')",target.read_text()))
            if 'syqpcx_fixture' in recipe:
                lines=target.read_text().splitlines()
                target.write_text("for fixture_index=1:91;eval(sprintf('m%d=zeros(300,92);',fixture_index));end;\n"+'\n'.join(lines[91:183])+'\n')
            if 'texture_small' in recipe:
                target.write_text(target.read_text().replace('M = 256;','M = 16;').replace('N = 256;','N = 16;'))
            if 'texture_array' in recipe:
                target.write_text(target.read_text().replace('Image = imread(path);','Image = path;'))
            (folder/'adaptation.json').write_text(json.dumps({'recipe':job['recipe'],'declared_edits':job['edits'],'text_encoding':'work copies normalized to UTF-8; original files preserved'},ensure_ascii=False,indent=2)+'\n')
        body = job["body"] if job["body"] is not None else "run("+quote(target)+");"
        prefix = "cd("+quote(target.parent)+"); addpath(pwd);\n"
        if 'texture_small' in job['recipe']:
            prefix+="warning('off','Octave:possible-matlab-short-circuit-operator');\n"
        harness = folder/"check.m"
        harness.write_text(prelude+prefix+"try\n"+body+"\n if exist('metrics','var'); fprintf('METRICS_JSON=%s\\n',jsonencode(metrics)); end;\n fprintf('CHECK_PASSED\\n');\ncatch err\n fprintf(2,'CHECK_FAILED: %s\\n',err.message); for k=1:numel(err.stack); fprintf(2,'  %s:%d\\n',err.stack(k).file,err.stack(k).line); end; exit(1);\nend;\n")
        command=[str(args.octave),str(harness)]
        started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        start=time.monotonic()
        try:
            cp=subprocess.run(command,capture_output=True,text=True,errors='replace',timeout=job['timeout'])
            log=cp.stdout+cp.stderr
            status='pass' if cp.returncode==0 and 'CHECK_PASSED' in log else 'fail'
            rc=cp.returncode
        except subprocess.TimeoutExpired as exc:
            chunks=[exc.stdout or b'',exc.stderr or b'']
            log=''.join(c.decode(errors='replace') if isinstance(c,bytes) else c for c in chunks)+'\nTIMEOUT\n'
            status='timeout';rc=None
        log_path=folder/'run.log';log_path.write_text(log)
        metric_lines=re.findall(r'METRICS_JSON=([^\n]+)',log)
        failure=re.search(r'CHECK_FAILED: ([^\n]+)',log)
        result={k:v for k,v in job.items() if k!='body'}
        result.update(source_member=member['name'],source_sha256=member['sha256'],
                      dataset='supplied example or synthetic engineering fixture; not medicinal-plant validation',
                      command=command,started_at_shanghai=started,duration_seconds=round(time.monotonic()-start,4),
                      status=status,returncode=rc,log=str(log_path.relative_to(args.output)),
                      error=failure.group(1) if failure else ('time budget exceeded' if status=='timeout' else None),
                      metrics=json.loads(metric_lines[-1]) if metric_lines else {})
        results.append(result)
        (args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'id':job['id'],'status':status,'seconds':result['duration_seconds'],'error':result['error'],'metrics':result['metrics']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
