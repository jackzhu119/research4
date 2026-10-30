warning('off','Octave:shadowed-function'); warning('off','Octave:gnuplot-graphics'); pkg load statistics signal io image optim; set(0,'defaultfigurevisible','off'); graphics_toolkit('gnuplot'); rand('seed',42); randn('seed',42);
cd('/workspace/scratch/algorithm-execution/second-pass/verify_cars_error_scaling/work/CARS'); addpath(pwd);
try
load('corn_m51.mat'); C=carspls(X,y,4,3,'center',10); V=plscvfold(X(:,C.vsel),y,4,3,'center',0); metrics=struct('raw_subset_cv',V.RMSECV,'reported_cv',C.minRMSECV,'raw_to_reported_ratio',V.RMSECV/C.minRMSECV,'variables',C.vsel); assert(abs(metrics.raw_to_reported_ratio-2.8)<1e-7);
 if exist('metrics','var'); fprintf('METRICS_JSON=%s\n',jsonencode(metrics)); end;
 fprintf('CHECK_PASSED\n');
catch err
 fprintf(2,'CHECK_FAILED: %s\n',err.message); for k=1:numel(err.stack); fprintf(2,'  %s:%d\n',err.stack(k).file,err.stack(k).line); end; exit(1);
end;
