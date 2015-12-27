"""Introduction To Data Science 
    Coursework 2
    Arshad Ahmed
    Flow 4: ML"""
"""------------------------------------------------------------------------ """

"""------------------------------------------------------------------------ """
"""Import Libraries """
"""------------------------------------------------------------------------ """
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import matplotlib.pyplot as plt
import statsmodels as sm
import sklearn 
import networkx as nx
import seaborn as sns; sns.set()
import re, os , sys
from time import time
from sklearn import decomposition, manifold, cluster, metrics
from scipy.stats import kendalltau, pointbiserialr, pearsonr
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, ExtraTreesClassifier, RandomForestClassifier
"""------------------------------------------------------------------------ """

"""------------------------------------------------------------------------ """
"""Import Data for Machine Learing """
"""------------------------------------------------------------------------ """
datadir = 'C:/Users/arsha_000/Downloads/data ideas/'
d6 = pd.read_excel(datadir + 'input_to_modeling.xlsx')
d6 = d6.ix[34::,:]
d6.head()
printrowname(d6)
target = d6['RowMedian_FertilityRate'];target.head()
d7 = d6.drop(['RowMedian_FertilityRate', 'RowMean_FertilityRate',
              'RowStd_FertilityRate','RowIQR_FertilityRate'], axis=1)
printrowname(d7)

"""Clustergram of all data with additional features included """

corr_d6 = d6.corr()
cmap = sns.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
sns.clustermap(corr_d6, method='weighted', metric='correlation', col_cluster=False,
               figsize=(16, 16), cmap=cmap, linewidths=1)

"""------------------------------------------------------------------------ """
"""Step 14: Dimensionality Reduction """
"""------------------------------------------------------------------------ """

""" 1: PCA """
pca = sklearn.decomposition.PCA()
pca_data_fit = pca.fit_transform(d7)
pca_3c = pca_data_fit[:,0:3]

""" 2:  Factor Analysis """
fa = decomposition.FactorAnalysis()
data_pca_fa = fa.fit_transform(pca_3c)

""" 3:  ICA """
ica = sklearn.decomposition.FastICA()
data_pca_ica = ica.fit_transform(pca_3c)
data_ica = ica.fit_transform(d7)
data_pca_fa_ica =  ica.fit_transform(data_pca_fa)

""" 4:  LLE """
data_lle, errLLE0 = sklearn.manifold.locally_linear_embedding(d7,n_neighbors=12, n_components=3)
data_pca_fa_ica_lle, errLLE = sklearn.manifold.locally_linear_embedding(data_pca_fa_ica,n_neighbors=12, n_components=3)


""" 5:  t-SNE """
tsne = manifold.TSNE(n_components=3)
model_tsne = tsne.fit(d7)
model_tsne.metric

data_tsne = tsne.fit_transform(d7)
data_pca_fa_ica_tsne = tsne.fit_transform(data_pca_fa_ica)

"""------------------------------------------------------------------------ """
"""Step 15: Regression Modelling """
"""------------------------------------------------------------------------ """

"""Regression Models """
forest = ExtraTreesRegressor(n_estimators=250,random_state=0)
rforest = RandomForestRegressor()
params = {'n_estimators': 10, 'max_depth': 10, 'min_samples_split': 10,
          'learning_rate': 0.1, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)


""" 1: ML No Dimensionality Reduction """
"""------------------------------------------------------------------------ """
X = d7.copy(deep=True)
y=target
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

clf.fit(X_train, y_train)
forest.fit(X_train,y_train)
rforest.fit(X_train,y_train) 

mse = mean_squared_error(y_test, clf.predict(X_test))
print("Gradient Boosting MSE: " ,mse)
mse1 = mean_squared_error(y_test, forest.predict(X_test))
print("Extra Trees MSE: " ,mse1)                            
mse2 = mean_squared_error(y_test, rforest.predict(X_test))
print("Random Forest MSE: " ,mse2)

""" Cross Validation """
cvGBR = cross_validation.cross_val_score(clf, X_train, y_train, cv=10).mean()
print("Average CV Error Gradient Boosting Regression:", cvGBR )
cvF  =  cross_validation.cross_val_score(forest, X_train, y_train, cv=10).mean()
print('Average CV Error Extra Trees:', cvF )
cvRF =  cross_validation.cross_val_score(rforest, X_train, y_train, cv=10).mean()
print('Average CV Error Random Forest:', cvRF )


""" 2: PCA + ML """
"""------------------------------------------------------------------------ """

X1 = pca_3c.copy()
y1=target
X1 = X1.astype(np.float32)
offset = int(X1.shape[0] * 0.9)
X1_train, y1_train = X1[:offset], y1[:offset]
X1_test, y1_test = X1[offset:], y1[offset:]

forest.fit(X1_train,y1_train)
rforest.fit(X1_train,y1_train)
clf.fit(X1_train, y1_train)

mse = mean_squared_error(y1_test, clf.predict(X1_test))
print("Gradient Boosting w. PCA MSE: " ,mse)
mse1 = mean_squared_error(y_test, forest.predict(X1_test))
print("Extra Trees w. PCA MSE: " ,mse1)                            
mse2 = mean_squared_error(y_test, rforest.predict(X1_test))
print("Random Forest w. PCA MSE: " ,mse2)

""" Cross Validation """
cvGBR = cross_validation.cross_val_score(clf, X2_train, y1_train, cv=10).mean()
print("Average CV Error Gradient Boosting Regression w. PCA:", cvGBR )
cvF  =  cross_validation.cross_val_score(forest, X2_train, y1_train, cv=10).mean()
print('Average CV Error Extra Trees w. PCA:', cvF )
cvRF =  cross_validation.cross_val_score(rforest, X2_train, y1_train, cv=10).mean()
print('Average CV Error Random Forest w. PCA:', cvRF )

""" 3: PCA + FA + ML"""
"""------------------------------------------------------------------------ """

X2 = data_pca_fa.copy()
y1=target
X2 = X2.astype(np.float32)
offset = int(X1.shape[0] * 0.9)
X2_train, y1_train = X2[:offset], y1[:offset]
X2_test, y1_test = X2[offset:], y1[offset:]

forest.fit(X2_train,y1_train)
rforest.fit(X2_train,y1_train)
clf.fit(X2_train, y1_train)

mse = mean_squared_error(y1_test, clf.predict(X2_test))
print("Gradient Boosting w. PCA + FA MSE: " ,mse)
mse1 = mean_squared_error(y_test, forest.predict(X2_test))
print("Extra Trees w. PCA + FA MSE: " ,mse1)                            
mse2 = mean_squared_error(y_test, rforest.predict(X2_test))
print("Random Forest w. PCA + FA MSE: " ,mse2)

""" Cross Validation """
cvGBR = cross_validation.cross_val_score(clf, X2_train, y1_train, cv=10).mean()
print("Average CV Error Gradient Boosting Regression w. PCA + FA:", cvGBR )
cvF  =  cross_validation.cross_val_score(forest, X2_train, y1_train, cv=10).mean()
print('Average CV Error Extra Treesw. PCA + FA:', cvF )
cvRF =  cross_validation.cross_val_score(rforest, X2_train, y1_train, cv=10).mean()
print('Average CV Error Random Forest w. PCA + FA:', cvRF )


""" 4: PCA + ICA + ML"""
"""------------------------------------------------------------------------ """

X2 = data_pca_ica.copy()
y1=target
X2 = X2.astype(np.float32)
offset = int(X1.shape[0] * 0.9)
X2_train, y1_train = X2[:offset], y1[:offset]
X2_test, y1_test = X2[offset:], y1[offset:]

forest.fit(X2_train,y1_train)
rforest.fit(X2_train,y1_train)
clf.fit(X2_train, y1_train)

mse = mean_squared_error(y1_test, clf.predict(X2_test))
print("Gradient Boosting w. PCA + ICA MSE: " ,mse)
mse1 = mean_squared_error(y_test, forest.predict(X2_test))
print("Extra Trees w. PCA + ICA MSE: " ,mse1)                            
mse2 = mean_squared_error(y_test, rforest.predict(X2_test))
print("Random Forest w. PCA + ICA MSE: " ,mse2)

""" Cross Validation """
cvGBR = cross_validation.cross_val_score(clf, X2_train, y1_train, cv=10).mean()
print("Average CV Error Gradient Boosting Regression w. PCA + ICA:", cvGBR )
cvF  =  cross_validation.cross_val_score(forest, X2_train, y1_train, cv=10).mean()
print('Average CV Error Extra Trees w. PCA + ICA:', cvF )
cvRF =  cross_validation.cross_val_score(rforest, X2_train, y1_train, cv=10).mean()
print('Average CV Error Random Forest w. PCA + ICA:', cvRF )

""" 5: PCA + FA + ICA + ML"""
"""------------------------------------------------------------------------ """

X2 = data_pca_fa_ica.copy()
y1=target
X2 = X2.astype(np.float32)
offset = int(X1.shape[0] * 0.9)
X2_train, y1_train = X2[:offset], y1[:offset]
X2_test, y1_test = X2[offset:], y1[offset:]

forest.fit(X2_train,y1_train)
rforest.fit(X2_train,y1_train)
clf.fit(X2_train, y1_train)

mse = mean_squared_error(y1_test, clf.predict(X2_test))
print("Gradient Boosting w. PCA + FA +ICA MSE: " ,mse)
mse1 = mean_squared_error(y_test, forest.predict(X2_test))
print("Extra Trees w. PCA + FA +ICA MSE: " ,mse1)                            
mse2 = mean_squared_error(y_test, rforest.predict(X2_test))
print("Random Forest w. PCA + FA +ICA MSE: " ,mse2)

""" Cross Validation """
cvGBR = cross_validation.cross_val_score(clf, X2_train, y1_train, cv=10).mean()
print("Average CV Error Gradient Boosting Regression w. PCA + FA +ICA:", cvGBR )
cvF  =  cross_validation.cross_val_score(forest, X2_train, y1_train, cv=10).mean()
print('Average CV Error Extra Trees w. PCA + FA +ICA:', cvF )
cvRF =  cross_validation.cross_val_score(rforest, X2_train, y1_train, cv=10).mean()
print('Average CV Error Random Forest w. PCA + FA +ICA:', cvRF )

#""" 6: LLE + ML"""
#"""------------------------------------------------------------------------ """
#
#X2 = data_lle.copy()
#y1=target
##X2 = X2.astype(np.float32)
#offset = int(X1.shape[0] * 0.9)
#X2_train, y1_train = X2[:offset], y1[:offset]
#X2_test, y1_test = X2[offset:], y1[offset:]
#
#forest.fit(X2_train,y1_train)
#rforest.fit(X2_train,y1_train)
#clf.fit(X2_train, y1_train)
#
#mse = mean_squared_error(y1_test, clf.predict(X2_test))
#print("Gradient Boosting LLE MSE: " ,mse)
#mse1 = mean_squared_error(y_test, forest.predict(X2_test))
#print("Extra Trees LLE MSE: " ,mse1)                            
#mse2 = mean_squared_error(y_test, rforest.predict(X2_test))
#print("Random Forest LLE MSE: " ,mse2)
#
#""" Cross Validation """"
#cvGBR = cross_validation.cross_val_score(clf, X2_train, y1_train, cv=10).mean()
#print("Average CV Error Gradient Boosting Regression w. PCA + FA +ICA:", cvGBR )
#cvF  =  cross_validation.cross_val_score(forest, X2_train, y1_train, cv=10).mean()
#print('Average CV Error Extra Trees w. PCA + FA +ICA:', cvF )
#cvRF =  cross_validation.cross_val_score(rforest, X2_train, y1_train, cv=10).mean()
#print('Average CV Error Random Forest w. PCA + FA +ICA:', cvRF )

#""" 7: PCA + FA + ICA + LLE + ML"""
#"""------------------------------------------------------------------------ """
#
#X2 = data_pca_fa_ica_lle
#y1=target
#X2 = X2.astype(np.float32)
#offset = int(X1.shape[0] * 0.9)
#X2_train, y1_train = X2[:offset], y1[:offset]
#X2_test, y1_test = X2[offset:], y1[offset:]
#
#forest.fit(X2_train,y1_train)
#rforest.fit(X2_train,y1_train)
#clf.fit(X2_train, y1_train)
#
#mse = mean_squared_error(y1_test, clf.predict(X2_test))
#print("Gradient Boosting w. PCA + FA +ICA + LLE MSE: " ,mse)
#mse1 = mean_squared_error(y_test, forest.predict(X2_test))
#print("Extra Trees w. PCA + FA + ICA + LLE MSE: " ,mse1)                            
#mse2 = mean_squared_error(y_test, rforest.predict(X2_test))
#print("Random Forest w. PCA + FA + ICA + LLE MSE: " ,mse2)
#
#""" Cross Validation """"
#cvGBR = cross_validation.cross_val_score(clf, X2_train, y1_train, cv=10).mean()
#print("Average CV Error Gradient Boosting Regression w. PCA + FA +ICA + LLE:", cvGBR )
#cvF  =  cross_validation.cross_val_score(forest, X2_train, y1_train, cv=10).mean()
#print('Average CV Error Extra Trees w. PCA + FA +ICA + LLE:', cvF )
#cvRF =  cross_validation.cross_val_score(rforest, X2_train, y1_train, cv=10).mean()
#print('Average CV Error Random Forest w. PCA + FA +ICA + LLE:', cvRF )

#doesnt work 6,7

""" 8: ICA + ML"""
"""------------------------------------------------------------------------ """

X2 = data_ica.copy()
y1=target
X2 = X2.astype(np.float32)
offset = int(X1.shape[0] * 0.9)
X2_train, y1_train = X2[:offset], y1[:offset]
X2_test, y1_test = X2[offset:], y1[offset:]

forest.fit(X2_train,y1_train)
rforest.fit(X2_train,y1_train)
clf.fit(X2_train, y1_train)

mse = mean_squared_error(y1_test, clf.predict(X2_test))
print("Gradient Boosting ICA MSE: " ,mse)
mse1 = mean_squared_error(y_test, forest.predict(X2_test))
print("Extra Trees ICA  MSE: " ,mse1)                            
mse2 = mean_squared_error(y_test, rforest.predict(X2_test))
print("Random Forest ICA  MSE: " ,mse2)

""" Cross Validation """
cvGBR = cross_validation.cross_val_score(clf, X2_train, y1_train, cv=10).mean()
print("Average CV Error Gradient Boosting Regression w. ICA :", cvGBR )
cvF  =  cross_validation.cross_val_score(forest, X2_train, y1_train, cv=10).mean()
print('Average CV Error Extra Trees w. ICA:', cvF )
cvRF =  cross_validation.cross_val_score(rforest, X2_train, y1_train, cv=10).mean()
print('Average CV Error Random Forest w. ICA:', cvRF )

""" 8: t_SNE + ML"""
"""------------------------------------------------------------------------ """

X2 = data_tsne.copy()
y1=target
X2 = X2.astype(np.float32)
offset = int(X1.shape[0] * 0.9)
X2_train, y1_train = X2[:offset], y1[:offset]
X2_test, y1_test = X2[offset:], y1[offset:]

forest.fit(X2_train,y1_train)
rforest.fit(X2_train,y1_train)
clf.fit(X2_train, y1_train)

mse = mean_squared_error(y1_test, clf.predict(X2_test))
print("Gradient Boosting t-SNE MSE: " ,mse)
mse1 = mean_squared_error(y_test, forest.predict(X2_test))
print("Extra Trees t-SNE  MSE: " ,mse1)                            
mse2 = mean_squared_error(y_test, rforest.predict(X2_test))
print("Random Forest t-SNE  MSE: " ,mse2)

""" Cross Validation """
cvGBR = cross_validation.cross_val_score(clf, X2_train, y1_train, cv=10).mean()
print("Average CV Error Gradient Boosting Regression w. t-SNE :", cvGBR )
cvF  =  cross_validation.cross_val_score(forest, X2_train, y1_train, cv=10).mean()
print('Average CV Error Extra Trees w.t-SNE:', cvF )
cvRF =  cross_validation.cross_val_score(rforest, X2_train, y1_train, cv=10).mean()
print('Average CV Error Random Forest w. t-SNE:', cvRF )

""" 9: PCA + FA + ICA +t_SNE + ML"""
"""------------------------------------------------------------------------ """

X2 = data_pca_fa_ica_tsne.copy()
y1=target
X2 = X2.astype(np.float32)
offset = int(X1.shape[0] * 0.9)
X2_train, y1_train = X2[:offset], y1[:offset]
X2_test, y1_test = X2[offset:], y1[offset:]

forest.fit(X2_train,y1_train)
rforest.fit(X2_train,y1_train)
clf.fit(X2_train, y1_train)

mse = mean_squared_error(y1_test, clf.predict(X2_test))
print("Gradient Boosting PCA + FA + ICA +t_SNE MSE: " ,mse)
mse1 = mean_squared_error(y_test, forest.predict(X2_test))
print("Extra Trees PCA + FA + ICA +t_SNE  MSE: " ,mse1)                            
mse2 = mean_squared_error(y_test, rforest.predict(X2_test))
print("Random Forest PCA + FA + ICA +t_SNE  MSE: " ,mse2)


""" Cross Validation """
cvGBR = cross_validation.cross_val_score(clf, X2_train, y1_train, cv=10).mean()
print("Average CV Error Gradient Boosting Regression PCA + FA + ICA +t_SNE:", cvGBR )
cvF  =  cross_validation.cross_val_score(forest, X2_train, y1_train, cv=10).mean()
print('Average CV Error Extra Trees PCA + FA + ICA +t_SNE:', cvF )
cvRF =  cross_validation.cross_val_score(rforest, X2_train, y1_train, cv=10).mean()
print('Average CV Error Random Forest PCA + FA + ICA +t_SNE:', cvRF )