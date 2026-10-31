warning('off','Octave:shadowed-function'); warning('off','Octave:gnuplot-graphics'); pkg load statistics signal io image optim; set(0,'defaultfigurevisible','off'); graphics_toolkit('gnuplot'); rand('seed',42); randn('seed',42);
cd('/workspace/scratch/algorithm-execution/second-pass/compat_edge_fixture/work'); addpath(pwd);
try
imwrite(uint8(255*rand(32,32,3)),'fixture.png'); run('edge_detection.m'); assert(isequal(size(output),size(output1))); metrics=struct('output_rows',rows(output),'max_difference',max(abs(output(:)-output1(:))));
 if exist('metrics','var'); fprintf('METRICS_JSON=%s\n',jsonencode(metrics)); end;
 fprintf('CHECK_PASSED\n');
catch err
 fprintf(2,'CHECK_FAILED: %s\n',err.message); for k=1:numel(err.stack); fprintf(2,'  %s:%d\n',err.stack(k).file,err.stack(k).line); end; exit(1);
end;
