warning('off','Octave:shadowed-function'); warning('off','Octave:gnuplot-graphics'); pkg load statistics signal io image optim; set(0,'defaultfigurevisible','off'); graphics_toolkit('gnuplot'); rand('seed',42); randn('seed',42);
cd('/workspace/research4/outputs/all-algorithms-final-validation/octave_ready/ready_uve_regression_demo/work/uve'); addpath(pwd);
try
run('/workspace/research4/outputs/all-algorithms-final-validation/octave_ready/ready_uve_regression_demo/work/uve/demo_PLS_Regression.m');
 if exist('metrics','var'); fprintf('METRICS_JSON=%s\n',jsonencode(metrics)); end;
 fprintf('CHECK_PASSED\n');
catch err
 fprintf(2,'CHECK_FAILED: %s\n',err.message); for k=1:numel(err.stack); fprintf(2,'  %s:%d\n',err.stack(k).file,err.stack(k).line); end; exit(1);
end;
