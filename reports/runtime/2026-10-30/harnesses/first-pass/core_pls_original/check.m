warning('off','Octave:shadowed-function'); warning('off','Octave:gnuplot-graphics'); pkg load statistics signal io image optim; set(0,'defaultfigurevisible','off'); graphics_toolkit('gnuplot'); rand('seed',42); randn('seed',42);
cd('/workspace/scratch/algorithm-execution/first-pass/core_pls_original/work/0308 CARS+UVE'); addpath(pwd);
try
load('corn_m5_starch.mat'); P=pls(Xcal,ycal,4,'center'); pred=[Xtest ones(rows(Xtest),1)]*P.coef_origin(:,end); assert(all(isfinite(pred))); metrics=struct('n_train',rows(Xcal),'n_test',rows(Xtest),'n_features',columns(Xcal),'example_rmsep',sqrt(mean((pred-ytest).^2))); save('-mat7-binary','cross_language_pls.mat','pred','Xcal','Xtest','ycal','ytest');
 if exist('metrics','var'); fprintf('METRICS_JSON=%s\n',jsonencode(metrics)); end;
 fprintf('CHECK_PASSED\n');
catch err
 fprintf(2,'CHECK_FAILED: %s\n',err.message); for k=1:numel(err.stack); fprintf(2,'  %s:%d\n',err.stack(k).file,err.stack(k).line); end; exit(1);
end;
