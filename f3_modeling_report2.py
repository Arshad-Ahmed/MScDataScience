"""------------------------------------------------------------------------ """
"""Introduction To Data Science 
    Coursework 2
    Arshad Ahmed
    Flow 3: Modelling/ML"""
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

"""------------------------------------------------------------------------ """
"""Step 11: Apply Dimensionality Reduction and ML: Testing """
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


"""Step 11_0: Apply PCA """
"""------------------------------------------------------------------------ """

pca = sklearn.decomposition.PCA()
pca_data = pca.fit(d7)
pca_data.explained_variance_ratio_
sum(pca_data.explained_variance_ratio_[0:2])

pca1 = pca_data.components_[:,0]
pca2 = pca_data.components_[:,1]
pca3 = pca_data.components_[:,2]

""" Project Data"""
pca1_data = matrix(pca1) * matrix(d7).transpose()
pca2_data = matrix(pca2) * matrix(d7).transpose()
pca3_data = matrix(pca3) * matrix(d7).transpose()

pca_df1 = pd.DataFrame(pca1_data.transpose())
pca_df2 = pd.DataFrame(pca2_data.transpose())
pca_df3 = pd.DataFrame(pca3_data.transpose())
pca_df_all = pd.concat([pca_df1, pca_df2, pca_df3], axis=1)
pcaname = ('PCA1','PCA2','PCA3')
pca_df_all.columns= pcaname
pca_df_all.head()

sortid_pca1 = pca1.argsort()
featname = np.asanyarray(list(d7))
featrank_pca = pd.DataFrame(featname[sortid_pca1].transpose())

""" PCA fitted data"""
pca_data_fit = pca.fit_transform(d7)
pca_3c = pca_data_fit[:,0:3]

"""Step 11_1: Apply Factor Analysis """
"""------------------------------------------------------------------------ """
fa = decomposition.FactorAnalysis()
data_fa = fa.fit_transform(pca_3c)
fa.components_
fa.loglike_

data_fa =  fa.fit(d7)
data_fa.components_[:,0]
fa1_sortidx = data_fa.components_[:,0].argsort()
featrank_fa = pd.DataFrame(featname[fa1_sortidx].transpose())

data_pca_fa_mdx = fa.fit(pca_data_fit)
pca_fa_sortidx = data_pca_fa_mdx.components_[:,0].argsort()
featname[pca_fa_sortidx]

featrank_pca_fa = pd.DataFrame(featname[pca_fa_sortidx])
"""Step 11_2: Apply ICA """
"""------------------------------------------------------------------------ """
ica = sklearn.decomposition.FastICA()
data_ica = ica.fit(d7)
data_ica.components_[:,0]

ica1_sortidx = data_ica.components_[:,0].argsort()
featrank_ica = pd.DataFrame(featname[ica1_sortidx].transpose())

data_pca_ica_mdx = ica.fit(pca_data_fit)
pca_ica1= data_pca_ica_mdx.components_[:,0]
pca_ica_sortidx = pca_ica1.argsort()
featname[pca_ica_sortidx ]
featrank_pca_ica = pd.DataFrame(featname[pca_ica_sortidx ])

"""Feature Ranking By PCA ,ICA, FA """
featureRank = pd.concat([featrank_ica, featrank_pca, featrank_fa, featrank_pca_fa,
                         featrank_pca_ica], axis=1)
featureRank.columns = ('ICA Comp1', 'PCA Comp1', 'FA Comp1',
                       'PCA + FA Comp1', 'PCA + ICA Comp1')
featureRank[1:11]

featureRank.to_excel(datadir +'feature_ranking_DR.xlsx')

data_pca_ica = ica.fit_transform(pca_3c)
data_pca_fa_ica =  ica.fit_transform(data_pca_fa)

"""Step 11_2: Apply FA to PCA + ICA data """
"""------------------------------------------------------------------------ """
data_pca_ica_fa = fa.fit_transform(data_ica) #produces zeros dont use
data_pca_fa = fa.fit_transform(pca_3c)

 

"""Step 11_3: Apply LLE to data """
"""------------------------------------------------------------------------ """
data_lle = sklearn.manifold.locally_linear_embedding(d7,n_neighbors=12, n_components=3)

data_pca_fa_ica_lle = sklearn.manifold.locally_linear_embedding(data_pca_fa_ica,n_neighbors=12, n_components=3)

"""Step 12: Machine Learning """
"""------------------------------------------------------------------------ """
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, ExtraTreesClassifier, RandomForestClassifier

""" Test and Training Set """
X = d7.copy(deep=True)
y=target
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

"""ML 1: Gradient Boosting Regressor """

params = {'n_estimators': 10, 'max_depth': 10, 'min_samples_split': 10,
          'learning_rate': 0.1, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)

""" Feature Selection """

feature_importance = clf.feature_importances_

# make importances relative to max importance
plt.figure(figsize=(7,18))
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
feature_names = list(X)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feature_names)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


"""Plot training deviance and compute test set deviance """

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance of Gradient Boosting Regression Model')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

"""ML 2: Extra Trees Regressor """

forest = ExtraTreesRegressor(n_estimators=250,random_state=0)
forest.fit(X_train,y_train)

"""ML 3: Random Forest Regressor """

rforest = RandomForestRegressor()
rforest.fit(X_train,y_train)  

mse = mean_squared_error(y_test, clf.predict(X_test))
print("Gradient Boosting MSE: " ,mse)
mse1 = mean_squared_error(y_test, forest.predict(X_test))
print("Extra Trees MSE: " ,mse1)                            
mse2 = mean_squared_error(y_test, rforest.predict(X_test))
print("Random Forest MSE: " ,mse2)

""" ML with PCA """

X1 = pca_df_all.copy(deep=True)
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


""" ML with PCA + FA """
X2 = data_fa.copy()
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

""" ML with PCA + ICA """

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
print("Gradient Boosting w. PCA + ICA MSE: " ,mse)
mse1 = mean_squared_error(y_test, forest.predict(X2_test))
print("Extra Trees w. PCA + ICA MSE: " ,mse1)                            
mse2 = mean_squared_error(y_test, rforest.predict(X2_test))
print("Random Forest w. PCA + ICA MSE: " ,mse2)

""" ML with PCA + ICA + FA """

X2 = data_pca_ica_fa.copy()
y1=target
X2 = X2.astype(np.float32)
offset = int(X1.shape[0] * 0.9)
X2_train, y1_train = X2[:offset], y1[:offset]
X2_test, y1_test = X2[offset:], y1[offset:]

forest.fit(X2_train,y1_train)
rforest.fit(X2_train,y1_train)
clf.fit(X2_train, y1_train)

mse = mean_squared_error(y1_test, clf.predict(X2_test))
print("Gradient Boosting w. PCA + ICA + FA MSE: " ,mse)
mse1 = mean_squared_error(y_test, forest.predict(X2_test))
print("Extra Trees w. PCA + ICA + FA MSE: " ,mse1)                            
mse2 = mean_squared_error(y_test, rforest.predict(X2_test))
print("Random Forest w. PCA + ICA + FA MSE: " ,mse2)


"""Step 13: Merge Country Names back to data for visual analytics """
"""------------------------------------------------------------------------ """

"""Strip Country Names """
countryname = pd.read_excel(datadir + 'merged_data.xlsx')
countryname = countryname['Country Name'].ix[34::];printrowname(countryname)
countryname_df = pd.DataFrame(countryname)
countryname_df.shape[0]
d6.shape[0]
countryname_df.tail(n=59)
"""Merge COuntry Names to Data """
d8 = pd.concat([countryname_df, d6],axis=1)
d8.shape[0]
"""Write Out Data for Visual Analytics"""
fdout = 'C:/Users/arsha_000/OneDrive/MSc Data Science Work/Introduction to Data Science/Coursework 2/'
d8.to_excel(fdout + 'f3_final_merged.xlsx')