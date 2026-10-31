warning('off','Octave:shadowed-function'); warning('off','Octave:gnuplot-graphics'); pkg load statistics signal io image optim; set(0,'defaultfigurevisible','off'); graphics_toolkit('gnuplot'); rand('seed',42); randn('seed',42);
cd('/workspace/scratch/algorithm-execution/first-pass/core_uve_noise/work/0308 CARS+UVE'); addpath(pwd);
try
load('corn_m5_starch.mat'); U=uvepls(Xcal,ycal,4,3,'center',6,40,0.99); assert(numel(U.vsel)>0); metrics=struct('variables',numel(U.vsel),'jackknife_groups',6);
 if exist('metrics','var'); fprintf('METRICS_JSON=%s\n',jsonencode(metrics)); end;
 fprintf('CHECK_PASSED\n');
catch err
 fprintf(2,'CHECK_FAILED: %s\n',err.message); for k=1:numel(err.stack); fprintf(2,'  %s:%d\n',err.stack(k).file,err.stack(k).line); end; exit(1);
end;
