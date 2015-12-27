%% Machine Learning Coursework 2015
% Regression Models to predict online news popularity
%%By:  Arshad Ahmed
%%
%%Import and Check data
%%
d = readtable('OnlineNewsPopularity.csv');
[n k] = size(d)
d(1:10,:)
colnames = d.Properties.VariableNames
colnames = colnames(:,3:end)
%Drop unnecessary columns and check
%%
d = d(:,3:end);
d(1:10,:)
[n k] = size(d)
missing = sum(ismissing(d))
% Scale data and split into features and target
%%
summary(d)
d = table2array(d);
zd = abs(zscore(d));
feat = zd(:, 1:58);
target = zd(:,59);

%IQR Median Plot of Dimensions of Features
%%
iqr_d = iqr(feat);
median_d = median(feat);
medianabsdev_feat = mad(feat, 1);%median absolute deviation

figure
subplot(2,1,1);
scatter(median_d, iqr_d, 'k+', 'LineWidth',1.5)
title('Plot of IQR and Median of Standardized Features of Data')
xlabel('Median of Standardized Features')
ylabel('IQR of Standardized Features')
subplot(2,1,2);
scatter(medianabsdev_feat,iqr_d, 'k+', 'LineWidth',1.5) % Use this for poster
title('Plot of IQR and Median Absolute Deviation');
xlabel('Median Absolute Deviation of Standardized Features')
ylabel('IQR of Standardized Features')

%Coordinate Plot of features with lalbels
%%
labels = cellstr(colnames);labels = labels(1:58);
parallelcoords(feat, 'Standardize','PCAStd','Color','r'); 
title('Coordinate Plot of PCA of Feature data with standardized Scores');
xlabel('Features');
ylabel('Standardized PCA Scores')

%Basic Statistics
%%
format short g
%Check pairwise correlations
rho_feat = corr(feat, 'type','Spearman', 'rows','pairwise');rho_feat
%Calculate Correlation Coefficients
R_feat = corrcoef(feat);R_feat
%Calculate covariance matrix
cov_feat = cov(feat);cov_feat
sum(R_feat);R_feat_row = sum(R_feat,2)
sum(rho_feat);rho_feat_row = sum(rho_feat, 2)
sum(cov_feat);cov_feat_row = sum(cov_feat,2)
%Statistics Table
statstable = table(rho_feat_row,R_feat_row,cov_feat_row, 'VariableNames',...
    {'PairwiseCorrelation','CorrelationCoefficients','Covariance'});

%% Machine Learning Models: Regression Trees and Random Forest
% Split Data in training and test
%%

rng(4)%For reproducibility
num_row = size(feat, 1);
split_point = round(num_row * 0.8);
seq = randperm(num_row)';
X_train = feat(seq(1:split_point), :);
Y_train = target(seq(1:split_point), :);
X_test = feat(seq(split_point:end), :);
Y_test = target(seq(split_point:end), :);

%ML method 1: fitrtree
%%
rtree = fitrtree(X_train, Y_train, 'CategoricalPredictors',...
    [12 13 14 15 16 17 30 31 32 33 34 35 36 37], 'CrossVal', 'on');
view(rtree); 
%Calculate Errors
cv_error_rtree = kfoldLoss(rtree) % Cross Validation Error

%Optimum Tree size
%%
leafs = logspace(1,2,200);
N = numel(leafs);
err = zeros(N,1);
for n=1:N
    t = fitrtree(X_train,Y_train,'CrossVal','On',...
        'MinLeaf',leafs(n));
    err(n) = kfoldLoss(t);
end
plot(err, 'k','LineWidth',1.5);
xlabel('Min Leaf Size');
ylabel('Cross-validated error');
title('Cross-Validated Error against increasing minimum leaf size')

OptimalTree = fitrtree(X_train,Y_train,'minleaf',180);%from above
cvOptTree = crossval(OptimalTree, 'kfold', 10);
%Calculate Errors for Optimum Regression Tree
%%
resuberr_opt = resubLoss(OptimalTree)
loss_opt = kfoldLoss(crossval(OptimalTree))
yfit_opt = predict(OptimalTree,X_test);

mseerrOpt =  mean((Y_test - yfit_opt).^2)       %Mean Square Error on test
maberrOpt = mean(abs((Y_test - yfit_opt)))      %Mean Absolute Error on test
medabs_errOpt = median(abs((Y_test - yfit_opt)))%Median Absolute Error
error_varianceOpt = var(Y_test - yfit_opt)      %Error Variance
r2Opt = 1 - sum((Y_test - yfit_opt).^2)/sum((Y_test).^2) %R Square
corrOpt = corr(Y_test,yfit_opt)                          %Correlation
rmsOpt = sqrt(mean((Y_test - yfit_opt).^2))              %RMS Error

%Ensemble Method: Random Forest Regression
%%
ens2 = fitensemble(X_train,Y_train,'Bag',200, 'Tree','Type','Regression',...
     'CategoricalPredictors', [12 13 14 15 16 17 30 31 32 33 34 35 36 37]);
%cross val 
cv_ens2 = crossval(ens2);

%Plot of Errors
%%
figure;
plot(loss(ens2,X_test,Y_test,'mode','cumulative'));
hold on;
plot(kfoldLoss(cv_ens2,'mode','cumulative'),'r.');
plot(oobLoss(ens2,'mode','cumulative'),'k--');
hold off;
xlabel('Number of trees');
ylabel('Error');
legend('Test','Cross-validation','Out of bag','Location','NE');
title('Plot of Errors with increasing number of tree')
comp_ens = compact(ens2);%Compact ensemble

%Optimum Random Forest Regressor after above
%%
ens_reduced = fitensemble(X_train,Y_train,'Bag',20, 'Tree',...
     'Type','Regression','CategoricalPredictors', [12 13 14 15 16 17 30 31 32 33 34 35 36 37]);
 
cv_ens_reduced = crossval(ens_reduced, 'kfold', 10);

%Error plot of Optimum Random Forest
%%
figure;
plot(loss(ens_reduced,X_test,Y_test,'mode','cumulative'));
hold on;
plot(kfoldLoss(cv_ens_reduced,'mode','cumulative'),'r.');
plot(oobLoss(ens_reduced,'mode','cumulative'),'k--');
hold off;
xlabel('Reduced number of trees');
ylabel('Error');
title('Plot of Errors for Optimum Random Forest');
legend('Test','Cross-validation','Out of bag','Location','NE');

%Calculate Errors on Optimum Random Forest
%%
resuberrENS = resubLoss(ens_reduced)
loss_ENS = kfoldLoss(cv_ens_reduced)
yfit_ENS = predict(ens_reduced,X_test);

mseerrENS =  mean((Y_test - yfit_ENS).^2)       %Mean Square Error on test
maberrENS = mean(abs((Y_test - yfit_ENS)))      %Mean Absolute Error on test
medabs_errENS = median(abs((Y_test - yfit_ENS)))%Median Absolute Error
error_varianceENS = var(Y_test - yfit_ENS)      %Error Variance
r2ENS = 1 - sum((Y_test - yfit_ENS).^2)/sum((Y_test).^2) %R Square
corrENS = corr(Y_test,yfit_ENS)                          %Correlation
rmsENS = sqrt(mean((Y_test - yfit_ENS).^2))              % RMS Error

%Bonus SVM Regression
%%
%SVM with Gaussian Kernel
svm_mdl= fitrsvm(X_train, Y_train, 'KernelFunction','gaussian','KernelScale','auto')
svm_mdl.ConvergenceInfo.Converged
svm_mdl.NumIterations

CV_svm_mdl = crossval(svm_mdl, 'kfold',10);
lossSVM = kfoldLoss(CV_svm_mdl)
resubSVM = resubLoss(svm_mdl)
yfit_SVM = predict(svm_mdl,X_test);

mseerrSVM =  mean((Y_test - yfit_SVM).^2)       %Mean Square Error on test
maberrSVM = mean(abs((Y_test - yfit_SVM)))      %Mean Absolute Error on test
medabs_errSVM = median(abs((Y_test - yfit_SVM)))%Median Absolute Error
error_varianceSVM = var(Y_test - yfit_SVM)      %Error Variance
r2SVM = 1 - sum((Y_test - yfit_SVM).^2)/sum((Y_test).^2) %R Square
corrSVM = corr(Y_test,yfit_SVM)                          %Correlation
rmsSVM = sqrt(mean((Y_test - yfit_SVM).^2))              % RMS Error

%SVM with rbf kernel
%%
svm_mdl2= fitrsvm(X_train, Y_train, 'KernelFunction','rbf','KernelScale','auto')
svm_mdl2.ConvergenceInfo.Converged
svm_mdl2.NumIterations

CV_svm_mdl2 = crossval(svm_mdl2)
lossSVM2 = kfoldLoss(CV_svm_mdl2)
resubSVM2 = resubLoss(svm_mdl2)
yfit_SVM2 = predict(svm_mdl2,X_test);

mseerrSVM2 =  mean((Y_test - yfit_SVM2).^2)       %Mean Square Error on test
maberrSVM2 = mean(abs((Y_test - yfit_SVM2)))      %Mean Absolute Error on test
medabs_errSVM2 = median(abs((Y_test - yfit_SVM2)))%Median Absolute Error
error_varianceSVM2 = var(Y_test - yfit_SVM2)      %Error Variance
r2SVM2 = 1 - sum((Y_test - yfit_SVM2).^2)/sum((Y_test).^2) %R Square
corrSVM2 = corr(Y_test,yfit_SVM2)                          %Correlation
rmsSVM2 = sqrt(mean((Y_test - yfit_SVM2).^2))              % RMS Error

%Gaussian and RBF kernel produce identical performance while gaussian
%kernel takes lower number of iterations to achieve convergence
%quote results for SVM with Gaussian kernel

%Evaluating usefullness of dimensionality reduction through PCA
%%
[coefs,score, explained, latent,tsquared, mu_feat] = pca(feat);
%Scree plot from PCA
figure()
pareto(explained)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
title('Pareto Plot Showing Explained Variance within data by PCA Components')

explained
sum(explained)

[coeff1,score1,pcvar1,mu1,v1,S1] = ppca(feat, 25);
pcvar1
sum(pcvar1)
figure()
pareto(pcvar1)
xlabel('Principal Component')
ylabel('Variance Explained (%)')

%PCA is not useful does not explain data sufficiently

%Feature selection with stepwise fit regression
%%

[b, pval, inmodel] = stepwisefit(X_train, Y_train,'penter', 0.05,...
    'premove', 0.10);
%important columns 1 2 6 7 10 11 13 18 19 24 25 26 27 28 30 40 41 43 52 53 

%Create new subset with reduced features
X_train_r = X_train(:, [1 2 6 7 10 11 13 18 19 24 25 26 27 28 30 40 41 43 52 53]);
X_test_r = X_test(:,[1 2 6 7 10 11 13 18 19 24 25 26 27 28 30 40 41 43 52 53]);

%Derive models with feature selection
%%
OptimalTree_fs = fitrtree(X_train_r,Y_train,'minleaf',180);
ens_reduced_fs = fitensemble(X_train_r,Y_train,'Bag',20, 'Tree',...
     'Type','Regression',...
     'CategoricalPredictors', [7 15]); 
svm_mdl_fs= fitrsvm(X_train_r, Y_train, 'KernelFunction','gaussian','KernelScale','auto');

%Calculate new erros from fs models
%%

%Errors for regression tree
%%
resuberr_opt_fs = resubLoss(OptimalTree_fs)
loss_opt_fs = kfoldLoss(crossval(OptimalTree_fs))
yfit_opt_fs = predict(OptimalTree_fs,X_test_r);

mseerrOpt_fs =  mean((Y_test - yfit_opt_fs).^2)       %Mean Square Error on test
maberrOpt_fs = mean(abs((Y_test - yfit_opt_fs)))      %Mean Absolute Error on test
medabs_errOpt_fs = median(abs((Y_test - yfit_opt_fs)))%Median Absolute Error
error_varianceOpt_fs = var(Y_test - yfit_opt_fs)      %Error Variance
r2Opt_fs = 1 - sum((Y_test - yfit_opt_fs).^2)/sum((Y_test).^2) %R Square
corrOpt_fs = corr(Y_test,yfit_opt_fs)                          %Correlation
rmsOpt_fs = sqrt(mean((Y_test - yfit_opt_fs).^2))              %RMS Error

%Errors for Random Forest
%%
cv_ens_reduced_fs = crossval(ens_reduced_fs, 'kfold', 10);
resuberrENS_fs = resubLoss(ens_reduced_fs)
loss_ENS_fs = kfoldLoss(cv_ens_reduced_fs)
yfit_ENS_fs = predict(ens_reduced_fs,X_test_r);

mseerrENS_fs =  mean((Y_test - yfit_ENS_fs).^2)       %Mean Square Error on test
maberrENS_fs = mean(abs((Y_test - yfit_ENS_fs)))      %Mean Absolute Error on test
medabs_errENS_fs = median(abs((Y_test - yfit_ENS_fs)))%Median Absolute Error
error_varianceENS_fs = var(Y_test - yfit_ENS_fs)      %Error Variance
r2ENS_fs = 1 - sum((Y_test - yfit_ENS_fs).^2)/sum((Y_test).^2) %R Square
corrENS_fs = corr(Y_test,yfit_ENS_fs)                          %Correlation
rmsENS_fs = sqrt(mean((Y_test - yfit_ENS_fs).^2))              % RMS Error

%SVM Errors
%%

CV_svm_mdl_fs = crossval(svm_mdl_fs)
lossSVM_fs = kfoldLoss(CV_svm_mdl_fs)
resubSVM_fs = resubLoss(svm_mdl_fs)
yfit_SVM_fs = predict(svm_mdl_fs,X_test_r);

mseerrSVM_fs =  mean((Y_test - yfit_SVM_fs).^2)       %Mean Square Error on test
maberrSVM_fs = mean(abs((Y_test - yfit_SVM_fs)))      %Mean Absolute Error on test
medabs_errSVM_fs = median(abs((Y_test - yfit_SVM_fs)))%Median Absolute Error
error_varianceSVM_fs = var(Y_test - yfit_SVM_fs)      %Error Variance
r2SVM_fs = 1 - sum((Y_test - yfit_SVM_fs).^2)/sum((Y_test).^2) %R Square
corrSVM_fs = corr(Y_test,yfit_SVM_fs)                          %Correlation
rmsSVM_fs = sqrt(mean((Y_test - yfit_SVM_fs).^2))              % RMS Error

%Lasso - further work test
%%

[B FitInfo] = lasso(X_train,Y_train,'CV',10);
figure()
lassoPlot(B,FitInfo,'PlotType','CV');

[B1 FitInfo1] = lasso(X_train,Y_train,'CV',10, 'Alpha',1);%Lasso regression
figure()
lassoPlot(B1,FitInfo1,'PlotType','CV');
title('Lasso Regression');

[B2 FitInfo2] = lasso(X_train,Y_train,'CV',10, 'Alpha',0.1);%Ridge regression
figure()
lassoPlot(B2,FitInfo2,'PlotType','CV');
title('Ridge Regression');

sum(B1,2)
sum(B2,2)

%Hence why this will be further work


[A,B,r,U,V,stats] = canoncorr(feat, target) ;
plot(U(:,1),V(:,1),'.')

[W,H] = nnmf(feat,3);
biplot(H','scores',W);