warning('off','Octave:shadowed-function'); warning('off','Octave:gnuplot-graphics'); pkg load statistics signal io image optim; set(0,'defaultfigurevisible','off'); graphics_toolkit('gnuplot'); rand('seed',42); randn('seed',42);
cd('/workspace/scratch/algorithm-execution/first-pass/core_pca_kmeans/work/0627卷积神经网络+PCA_Kmeans（可用）/0627卷积神经网络+PCA_Kmeans'); addpath(pwd);
try
load('data.mat'); [coeff,scores,latent,tsq,explained,mu]=pca(xtrain); k=min(10,columns(scores)); projected=(xtest-mu)*coeff(:,1:k); idx=kmeans(scores(:,1:k),12,'Replicates',1); assert(rows(projected)==rows(xtest)); metrics=struct('train_rows',rows(xtrain),'test_rows',rows(xtest),'features',columns(xtrain),'pca_components',k,'clusters',numel(unique(idx)));
 if exist('metrics','var'); fprintf('METRICS_JSON=%s\n',jsonencode(metrics)); end;
 fprintf('CHECK_PASSED\n');
catch err
 fprintf(2,'CHECK_FAILED: %s\n',err.message); for k=1:numel(err.stack); fprintf(2,'  %s:%d\n',err.stack(k).file,err.stack(k).line); end; exit(1);
end;
