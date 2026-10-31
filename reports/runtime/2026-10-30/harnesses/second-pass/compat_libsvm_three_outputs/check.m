warning('off','Octave:shadowed-function'); warning('off','Octave:gnuplot-graphics'); pkg load statistics signal io image optim; set(0,'defaultfigurevisible','off'); graphics_toolkit('gnuplot'); rand('seed',42); randn('seed',42);
cd('/workspace/scratch/algorithm-execution/second-pass/compat_libsvm_three_outputs/work/0627卷积神经网络+PCA_Kmeans（可用）/0627卷积神经网络+PCA_Kmeans'); addpath(pwd);
try
load('data.mat'); mu=mean(xtrain); sd=std(xtrain); sd(sd==0)=1; a=(xtrain-mu)./sd; b=(xtest-mu)./sd; M=svmtrain(ytrain,a,sprintf('-q -c 1 -g %.16g',1/columns(a))); [p,acc,dec]=svmpredict(ytest,b,M,'-q'); assert(numel(p)==rows(xtest)); metrics=struct('example_accuracy_percent',acc(1),'train_rows',rows(a),'test_rows',rows(b));
 if exist('metrics','var'); fprintf('METRICS_JSON=%s\n',jsonencode(metrics)); end;
 fprintf('CHECK_PASSED\n');
catch err
 fprintf(2,'CHECK_FAILED: %s\n',err.message); for k=1:numel(err.stack); fprintf(2,'  %s:%d\n',err.stack(k).file,err.stack(k).line); end; exit(1);
end;
