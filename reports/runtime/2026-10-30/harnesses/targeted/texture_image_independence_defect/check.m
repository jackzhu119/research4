warning('off','Octave:shadowed-function'); warning('off','Octave:gnuplot-graphics'); pkg load statistics signal io image optim; set(0,'defaultfigurevisible','off'); graphics_toolkit('gnuplot'); rand('seed',42); randn('seed',42);
cd('/workspace/scratch/algorithm-execution/targeted/texture_image_independence_defect/work'); addpath(pwd);
warning('off','Octave:possible-matlab-short-circuit-operator');
try
imwrite(uint8(zeros(16,16,3)),'zero.png');imwrite(uint8(255*rand(16,16,3)),'random.png');a=Texture('zero.png');b=Texture('random.png');assert(all(isfinite(a)));assert(max(abs(a-b))<1e-12);metrics=struct('fixture_size',16,'max_feature_difference',max(abs(a-b)),'first_features',a,'finding','stored GLCM conditions do not use the image values');
 if exist('metrics','var'); fprintf('METRICS_JSON=%s\n',jsonencode(metrics)); end;
 fprintf('CHECK_PASSED\n');
catch err
 fprintf(2,'CHECK_FAILED: %s\n',err.message); for k=1:numel(err.stack); fprintf(2,'  %s:%d\n',err.stack(k).file,err.stack(k).line); end; exit(1);
end;
