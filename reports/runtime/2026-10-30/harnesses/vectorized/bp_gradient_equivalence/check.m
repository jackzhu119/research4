warning('off','Octave:shadowed-function'); warning('off','Octave:gnuplot-graphics'); pkg load statistics signal io image optim; set(0,'defaultfigurevisible','off'); graphics_toolkit('gnuplot'); rand('seed',42); randn('seed',42);
cd('/workspace/scratch/algorithm-execution/vectorized/bp_gradient_equivalence/work/案例1'); addpath(pwd);
try
x=randn(24,1); W1=randn(25,24);W2=randn(25,4);b1=randn(25,1);b2=randn(4,1); target=randn(4,1);z=1./(1+exp(-(W1*x+b1)));e=target-(W2'*z+b2); G=zeros(24,25);q=zeros(25,1); for k=1:24;for j=1:25;G(k,j)=z(j)*(1-z(j))*x(k)*(e(1)*W2(j,1)+e(2)*W2(j,2)+e(3)*W2(j,3)+e(4)*W2(j,4));q(j)=z(j)*(1-z(j))*(e(1)*W2(j,1)+e(2)*W2(j,2)+e(3)*W2(j,3)+e(4)*W2(j,4));end;end;v=(z.*(1-z)).*(W2*e);difference=max([max(abs(G'-v*x')(:)),max(abs(q-v))]);assert(difference<1e-12);metrics=struct('max_gradient_difference',difference);
 if exist('metrics','var'); fprintf('METRICS_JSON=%s\n',jsonencode(metrics)); end;
 fprintf('CHECK_PASSED\n');
catch err
 fprintf(2,'CHECK_FAILED: %s\n',err.message); for k=1:numel(err.stack); fprintf(2,'  %s:%d\n',err.stack(k).file,err.stack(k).line); end; exit(1);
end;
