warning('off','Octave:shadowed-function'); warning('off','Octave:gnuplot-graphics'); pkg load statistics signal io image optim; set(0,'defaultfigurevisible','off'); graphics_toolkit('gnuplot'); rand('seed',42); randn('seed',42);
cd('/workspace/scratch/algorithm-execution/second-pass/patched_iriv_small_dimension/work/uve'); addpath(pwd);
try
X=randn(48,10); y=3*X(:,1)-X(:,4)+0.1*randn(48,1); F=iriv(X,y,3,3,'center'); assert(numel(F.SelectedVariables)>0); metrics=struct('selected',F.SelectedVariables);
 if exist('metrics','var'); fprintf('METRICS_JSON=%s\n',jsonencode(metrics)); end;
 fprintf('CHECK_PASSED\n');
catch err
 fprintf(2,'CHECK_FAILED: %s\n',err.message); for k=1:numel(err.stack); fprintf(2,'  %s:%d\n',err.stack(k).file,err.stack(k).line); end; exit(1);
end;
