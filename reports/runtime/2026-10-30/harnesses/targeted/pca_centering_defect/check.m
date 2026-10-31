warning('off','Octave:shadowed-function'); warning('off','Octave:gnuplot-graphics'); pkg load statistics signal io image optim; set(0,'defaultfigurevisible','off'); graphics_toolkit('gnuplot'); rand('seed',42); randn('seed',42);
cd('/workspace/scratch/algorithm-execution/targeted/pca_centering_defect/work/0627卷积神经网络+PCA_Kmeans（可用）/0627卷积神经网络+PCA_Kmeans'); addpath(pwd);
try
load('data.mat');[a,ps]=mapminmax(xtrain',0,1);a=a';b=mapminmax('apply',xtest',ps)';[L,S,latent,tsq,explained,mu]=pca(a);C=kmeans(S(:,1:19),12,'Replicates',1);old=b*L(:,1:19);correct=(b-mu)*L(:,1:19);[~,p1]=min(pdist2(C,old),[],1);[~,p2]=min(pdist2(C,correct),[],1);metrics=struct('test_rows',rows(b),'changed_cluster_assignments',sum(p1~=p2),'projection_shift_l2',norm(mu*L(:,1:19)));assert(metrics.projection_shift_l2>0);
 if exist('metrics','var'); fprintf('METRICS_JSON=%s\n',jsonencode(metrics)); end;
 fprintf('CHECK_PASSED\n');
catch err
 fprintf(2,'CHECK_FAILED: %s\n',err.message); for k=1:numel(err.stack); fprintf(2,'  %s:%d\n',err.stack(k).file,err.stack(k).line); end; exit(1);
end;
