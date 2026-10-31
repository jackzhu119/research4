warning('off','Octave:shadowed-function'); warning('off','Octave:gnuplot-graphics'); pkg load statistics signal io image optim; set(0,'defaultfigurevisible','off'); graphics_toolkit('gnuplot'); rand('seed',42); randn('seed',42);
cd('/workspace/scratch/algorithm-execution/texture-array/texture_direct_rgb_array/work'); addpath(pwd);
warning('off','Octave:possible-matlab-short-circuit-operator');
try
a=Texture(uint8(cat(3,255*ones(16),zeros(16),zeros(16))));b=Texture(uint8(255*rand(16,16,3)));assert(all(isfinite(a)));assert(max(abs(a-b))<1e-12);metrics=struct('fixture_size',16,'max_feature_difference',max(abs(a-b)),'first_features',a,'finding','stored GLCM conditions do not use image values');
 if exist('metrics','var'); fprintf('METRICS_JSON=%s\n',jsonencode(metrics)); end;
 fprintf('CHECK_PASSED\n');
catch err
 fprintf(2,'CHECK_FAILED: %s\n',err.message); for k=1:numel(err.stack); fprintf(2,'  %s:%d\n',err.stack(k).file,err.stack(k).line); end; exit(1);
end;
