warning('off','Octave:shadowed-function'); warning('off','Octave:gnuplot-graphics'); pkg load statistics signal io image optim; set(0,'defaultfigurevisible','off'); graphics_toolkit('gnuplot'); rand('seed',42); randn('seed',42);
cd('/workspace/scratch/algorithm-execution/first-pass/core_mcuve_0308/work/0308 CARS+UVE'); addpath(pwd);
try
load('corn_m5_starch.mat'); U=mcuvepls(Xcal,ycal,4,'center',30,0.7); assert(numel(U.RI)==columns(Xcal)); assert(all(isfinite(U.RI))); metrics=struct('features',columns(Xcal),'runs',30,'finite_stability',true);
 if exist('metrics','var'); fprintf('METRICS_JSON=%s\n',jsonencode(metrics)); end;
 fprintf('CHECK_PASSED\n');
catch err
 fprintf(2,'CHECK_FAILED: %s\n',err.message); for k=1:numel(err.stack); fprintf(2,'  %s:%d\n',err.stack(k).file,err.stack(k).line); end; exit(1);
end;
