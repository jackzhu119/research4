warning('off','Octave:shadowed-function'); pkg load statistics;
rand('seed',42); randn('seed',42);
cd('/workspace/research4/outputs/external-octave-benchmark/ultrair_jyh_lc_target0_fold3__mcuve/work'); addpath(pwd); load('/workspace/research4/outputs/external-octave-benchmark/ultrair_jyh_lc_target0_fold3__mcuve/inputs.mat');
try
A=2; selected=1:columns(X_train); selection_cv=NaN; nonfinite_stability=0;
U=mcuvepls(X_train,y_train,A,'center',20,0.8);
selected=U.SortedVariable(1:100);
nonfinite_stability=sum(~isfinite(U.RI));
assert(all(isfinite(U.RI(selected))));
assert(numel(selected)>=A);
P=pls(X_train(:,selected),y_train,A,'center');
pred=[X_test(:,selected) ones(rows(X_test),1)]*P.coef_origin(:,end);
assert(numel(pred)==rows(X_test)); assert(all(isfinite(pred)));
save('-mat7-binary','/workspace/research4/outputs/external-octave-benchmark/ultrair_jyh_lc_target0_fold3__mcuve/predictions.mat','pred','selected','selection_cv','nonfinite_stability');
fprintf('CHECK_PASSED\n');
catch err
 fprintf(2,'CHECK_FAILED: %s\n',err.message);
for k=1:numel(err.stack); fprintf(2,'  %s:%d\n',err.stack(k).file,err.stack(k).line); end; exit(1);
end;
