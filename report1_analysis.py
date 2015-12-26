# -*- coding: utf-8 -*-
"""
@author: Arshad Ahmed
Introduction to Data Science Coursework 2015
"""

####################################
# Import libraries
####################################
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import matplotlib.pyplot as plt
import statsmodels as sm
import sklearn as skl
import seaborn as sns; sns.set()

###############################################################################
#Get Data
#Smoking Data Adults from Health profiles
###############################################################################
datadir = 'C:/Users/arsha_000/OneDrive/MSc Data Science Work/Introduction to Data Science/Selected CW Data/'
f0_smok = datadir + 'deprivation_clean.xlsx'
f0_smok_raw = pd.read_excel(f0_smok)
colname_smok =list(f0_smok_raw)

def print_el(x):
    for i in x:
        print(i)
        
print_el(colname_smok)
sns.pairplot(f0_smok_raw)
f0_smok_raw.describe()

#We see a strong outlier which is the aggregate value for England
#Aggregate Numerator, Denominator dont show up because they are blank
#Period = 2010
#Drop these columns and trim data

f0_smok_raw = f0_smok_raw.ix[1:]
sns.pairplot(f0_smok_raw) 

f0_smok_raw = f0_smok_raw.drop(['Aggregate Numerator', 'Aggregate Denominator', 'Period','ONS Code (old)', 'ONS Cluster'], axis=1)
f0_smok_raw.head(n=15)
colname_smok =list(f0_smok_raw)
print_el(colname_smok)

sig_split_smok = pd.get_dummies(f0_smok_raw['Significance'])
print(sig_split_smok)

def sig_pct(data):
    tmp1= pd.get_dummies(data['Significance']).sum()
    tmp2 = data.shape[0]
    pct = np.abs((tmp1/tmp2)) * 100
    print('Count of Different Signiface Categories:\n', tmp1)
    print('% of Significance Categories:\n', pct )

#Basic Data Stats post cleanup
sig_pct(f0_smok_raw)
f0_smok_raw.describe()
pd.isnull(f0_smok_raw).sum()
sns.pairplot(f0_smok_raw, hue='Significance', markers=["o", "s", "D"])
f0_smok_raw.head(n=25)
list(f0_smok_raw)
smok_colname_update =(
 'ONS Code (new)',
 'Area Name ',
 'Single Year Numerator Smok',
 'Single Year Denominator Smok',
 'Indicator value Smok',
 'Lower 95% CI Smok',
 'Upper 95% CI Smok',
 'Significance Smok'
)
len(smok_colname_update)
f0_smok_raw_bk = f0_smok_raw #backup data
f0_smok_raw.columns = smok_colname_update
f0_smok_raw.tail(n=50)

###############################################################################
#Deprivation Data
###############################################################################

f0_dep = datadir + 'deprivation_clean.xlsx'
f0_dep_raw = pd.read_excel(f0_dep)
colname_dep =list(f0_dep_raw)
print_el(colname_dep)
f0_dep_raw.head(n=15)
sns.pairplot(f0_dep_raw)
f0_dep_raw.describe()

#We see a strong outlier which is the aggregate value for England
#Aggregate Numerator, Denominator dont show up because they are blank
#Period = 2010
#Drop these columns and trim data

f0_dep_raw = f0_dep_raw.ix[1:]
f0_dep_raw = f0_dep_raw.drop(['Aggregate Numerator', 'Aggregate Denominator', 'Period', 'ONS Code (old)','ONS Cluster' ], axis=1)
f0_dep_raw.head(n=15)
colname_dep =list(f0_dep_raw)
print_el(colname_dep)
sns.pairplot(f0_dep_raw)

#Basic Data Stats post cleanup
sig_pct(f0_dep_raw)
f0_dep_raw.describe()
pd.isnull(f0_dep_raw).sum()
sns.pairplot(f0_dep_raw, hue='Significance', markers=["o", "s", "D"])
f0_dep_raw.head(n=25)
list(f0_dep_raw)
dep_colname_update =(
 'ONS Code (new)',
 'Area Name ',
 'Single Year Numerator Dep',
 'Single Year Denominator Dep',
 'Indicator value Dep',
 'Lower 95% CI Dep',
 'Upper 95% CI Dep',
 'Significance Dep')
len(dep_colname_update)
f0_dep_raw_bk = f0_dep_raw #backup data
f0_dep_raw.columns = dep_colname_update
f0_dep_raw.head(n=25)

###############################################################################
#Unemployment Data
###############################################################################

f0_unemp = datadir + 'lt_unemp_clean.xlsx'
f0_unemp_raw = pd.read_excel(f0_unemp)
colname_unemp =list(f0_unemp_raw)
print_el(colname_unemp)
f0_unemp_raw.head(n=15)
sns.pairplot(f0_unemp_raw)
f0_unemp_raw.describe()


#We see a strong outlier which is the aggregate value for England
#Aggregate Numerator, Denominator dont show up because they are blank
#Period = 2012
#Drop these columns and trim data

f0_unemp_raw = f0_unemp_raw.ix[1:]
f0_unemp_raw = f0_unemp_raw.drop(['Aggregate Numerator', 'Aggregate Denominator', 'Period', 'ONS Code (old)','ONS Cluster' ], axis=1)
f0_unemp_raw.head(n=15)
colname_unemp =list(f0_unemp_raw)
print_el(colname_unemp)
sns.pairplot(f0_unemp_raw)

#Basic Data Stats post cleanup
sig_pct(f0_unemp_raw)
f0_unemp_raw.describe()
pd.isnull(f0_unemp_raw).sum()
sns.pairplot(f0_unemp_raw, hue='Significance', markers=["o", "s", "D"])
list(f0_unemp_raw)
unemp_colname_update =(
 'ONS Code (new)',
 'Area Name ',
 'Single Year Numerator Unemp',
 'Single Year Denominator Unemp',
 'Indicator value Unemp',
 'Lower 95% CI Unemp',
 'Upper 95% CI Unemp',
 'Significance Unemp')
len(unemp_colname_update)
f0_unemp_raw_bk = f0_unemp_raw #backup data
f0_unemp_raw.columns = unemp_colname_update
f0_unemp_raw.head(n=25)
###############################################################################
#Income - Ward level
###############################################################################

f0_inc = 'C:/Users/arsha_000/OneDrive/MSc Data Science Work/Introduction to Data Science/Selected CW Data/gla_income_median.xlsx'
f0_inc_raw = pd.read_excel(f0_inc)
colname_inc =list(f0_inc_raw)
print_el(colname_inc)
f0_inc_raw.head(n=15)
f0_inc_raw.describe()
sns.violinplot(f0_inc_raw)

""" Data ranges from 2001-2013 but other data only from 2010 and 2012
so keep these and drop the rest
"""
f0_inc_raw = f0_inc_raw.drop(['2001/02','2002/03','2003/04','2004/05',
'2005/06','2006/07','2007/08','2008/09','2009/10'], axis=1)
colname_inc =list(f0_inc_raw)
print_el(colname_inc)
sns.violinplot(f0_inc_raw.ix[:,4:7], orient='h')
#Calculating year on year increase between this time period
f0_inc_raw['YoY10/12'] = (f0_inc_raw['2011/12']/f0_inc_raw['2010/11'])
plt.plot(f0_inc_raw['YoY10/12'], 'o')
np.mean(f0_inc_raw['YoY10/12'])

f0_inc_raw['YoY13/11'] = (f0_inc_raw['2012/13']/f0_inc_raw['2011/12']) 
plt.plot(f0_inc_raw['YoY13/11'], 'o')
np.mean(f0_inc_raw['YoY13/11'])

mean_diff_inc = np.mean(f0_inc_raw['YoY13/11'])-np.mean(f0_inc_raw['YoY10/12'])
#Plotting year on year increase
sns.violinplot(data=f0_inc_raw.ix[:,7:], orient='h')

#Try different scaling of data
# First up z-standardization

f0_inc_zstd = f0_inc_raw.ix[:,1:7]
f0_inc_zstd.head(n=15)

normCol= []
for i, col in f0_inc_zstd.iteritems():
    if not isinstance(col[0], str):
        mean = np.mean(col)
        std  = np.std(col)
        normCol = np.abs((col - mean)/std)
        f0_inc_zstd[[i]]= normCol

f0_inc_zstd.head(n=15)
sns.violinplot(data=f0_inc_zstd)

#Next Min/Max scaling
f0_inc_minmax = f0_inc_raw.ix[:,1:7]
f0_inc_minmax.head(n=15)

for i, col in f0_inc_minmax.iteritems():
    if not isinstance(col[0], str):
        min = col.min()
        max = col.max()
        normCol = np.abs((col - min)/(max - min))
        f0_inc_minmax[[i]]= normCol

f0_inc_minmax.head(n=15)
sns.violinplot(f0_inc_minmax)

#Final Log scaling
f0_inc_log = f0_inc_raw.ix[:,1:7]
f0_inc_log.head(n=15)

for i, col in f0_inc_log.iteritems():
    if not isinstance(col[0], str):
        normCol = col.apply(np.log)
        f0_inc_log[[i]]= normCol

f0_inc_log.head(n=15)
sns.violinplot(f0_inc_log) 
f0_inc_minmax['ONS Code (new)'] = f0_inc_minmax['LAD code']
f0_inc_minmax.head(n=25)
list(f0_inc_minmax)
inc_colname_update = ('Ward name',
 'LAD code',
 'Borough',
 '2010/11 Ward',
 '2011/12 Ward',
 '2012/13 Ward',
 'ONS Code (new)')
f0_inc_minmax.columns = inc_colname_update;f0_inc_minmax.head(n=25)

###############################################################################
#Income - Brough level
###############################################################################

f0_inc_bor = 'C:/Users/arsha_000/OneDrive/MSc Data Science Work/Introduction to Data Science/Selected CW Data/gla_income_median_borough.xlsx'
f0_inc_bor_raw = pd.read_excel(f0_inc_bor); f0_inc_bor_raw.shape[0]
colname_inc_bor =list(f0_inc_bor_raw)
print_el(colname_inc_bor)
f0_inc_bor_raw.head(n=15)
f0_inc_bor_raw.describe()
sns.violinplot(f0_inc_bor_raw)

""" Data ranges from 2001-2013 but other data only from 2010 and 2012
so keep these and drop the rest
"""
f0_inc_bor_raw = f0_inc_bor_raw.drop(['2001/02','2002/03','2003/04','2004/05',
'2005/06','2006/07','2007/08','2008/09','2009/10'], axis=1)
f0_inc_bor_raw.head(n=15)
colname_inc_bor =list(f0_inc_bor_raw)
print_el(colname_inc_bor)
sns.violinplot(f0_inc_bor_raw, orient='h')      

f0_inc_bor_raw['YoY12/10'] = (f0_inc_bor_raw['2011/12']/f0_inc_bor_raw['2010/11'])
plt.plot(f0_inc_raw['YoY10/12'], 'o')

f0_inc_bor_raw['YoY13/11'] = (f0_inc_bor_raw['2012/13']/f0_inc_bor_raw['2011/12']) 
plt.plot(f0_inc_raw['YoY13/11'], 'o')

#Do same scaling as above
f0_inc_bor_bk = f0_inc_bor_raw #experienced some issues so backing up

f0_inc_bor_zstd = f0_inc_bor_raw.ix[:,0:]
f0_inc_bor_minmax= f0_inc_bor_raw.ix[:,0:]
f0_inc_bor_log = f0_inc_bor_raw.ix[:,0:]
normCol1= [] ; normCol2= [];normCol3= []

f0_inc_bor_zstd.head(n=15)
f0_inc_bor_minmax.head(n=15)
f0_inc_bor_log.head(n=15)

for i, col in f0_inc_bor_zstd.iteritems():
    if not isinstance(col[0], str):
        mean = np.mean(col)
        std  = np.std(col)
        normCol1 = np.abs((col - mean)/std)
        f0_inc_bor_zstd[[i]]= normCol1
        
for i, col in f0_inc_bor_minmax.iteritems():
    if not isinstance(col[0], str):
        min = col.min()
        max = col.max()
        normCol2 = np.abs((col - min)/(max - min))
        f0_inc_bor_minmax[[i]]= normCol2
        
for i, col in f0_inc_bor_log.iteritems():
    if not isinstance(col[0], str):
        normCol3 = col.apply(np.log)
        f0_inc_bor_log[[i]]= normCol3
           
f0_inc_bor_zstd.head(n=15)
f0_inc_bor_minmax.head(n=15); f0_inc_bor_minmax.shape[0]
f0_inc_bor_log.head(n=15)

sns.violinplot(f0_inc_bor_zstd.ix[:,1:5])#skews relationships
sns.violinplot(f0_inc_bor_minmax.ix[:,1:5]) #elations preserved easier to understand than log
sns.violinplot(f0_inc_bor_log.ix[:,1:5])

list(f0_inc_bor_minmax)
inc_bor_colname_update = ('ONS Code (new)',
 'Borough',
 '2010/11 Bor',
 '2011/12 Bor',
 '2012/13 Bor',
 'YoY12/10 Bor',
 'YoY13/11 Bor')
f0_inc_bor_minmax.columns = inc_bor_colname_update;f0_inc_bor_minmax.head(n=25)

###############################################################################
#Data Fusion Step
###############################################################################
f0_dep_raw.head(n=5)
f0_dep_raw.shape[0]
f0_smok_raw.head(n=5)
f0_unemp_raw.head(n=5)
f0_inc_bor_minmax.head(n=5)

#Merge on ONS Code (new) health profiles first: deprivation, smoking
df_merge1 = pd.merge(f0_dep_raw,f0_smok_raw, on='ONS Code (new)', how='outer')
df_merge1.head(n=25) #QC at each stage
df_merge1.shape[0]
df_merge1.shape[1]
pd.isnull(df_merge1).sum()

#merge deprivation, smoking, unempployment
df_merge2=pd.merge(df_merge1,f0_unemp_raw, on='ONS Code (new)', how='outer')
df_merge2.head(n=25) #QC at each stage
df_merge2.shape[0]
df_merge2.shape[1]
pd.isnull(df_merge2).sum()
#scale data prior to merging
df_merge2_minmax = df_merge2
normCol=[]
for i, col in df_merge2_minmax.iteritems():
    if not isinstance(col[0], str):
        min = col.min()
        max = col.max()
        normCol = np.abs((col - min)/(max - min))
        df_merge2_minmax[[i]]= normCol
df_merge2_minmax.head(n=25)
df_merge2_minmax.shape[0]; df_merge2_minmax.shape[1]

#merge health and income data at borough level
df_merge3 = pd.merge(df_merge2,f0_inc_bor_minmax, on='ONS Code (new)', how='outer')
df_merge3.head(n=25) #QC at each stage
df_merge3.shape[0]
df_merge3.shape[1] 
pd.isnull(df_merge3).sum()
pd.notnull(df_merge3).sum()
df_merge3.dropna(how='all', inplace=True)
pd.isnull(df_merge3).sum()
df_merge3.dropna(how='any', inplace=True)
pd.isnull(df_merge3).sum()
df_merge3.shape[0] ; df_merge3.shape[1]
print(df_merge3)

df_merge3.drop_duplicates(keep='first', inplace=True)

#merging borough and ward level income data using ons code
df_merge_inc = pd.merge(f0_inc_minmax,f0_inc_bor_minmax, on='ONS Code (new)', how='outer')
df_merge_inc.head(n=25) #QC at each stage
df_merge_inc.shape[0]
df_merge_inc.shape[1]
pd.isnull(df_merge_inc).sum()
pd.notnull(df_merge_inc).sum()
df_merge_inc.drop_duplicates(keep='first', inplace=True)
pd.isnull(df_merge_inc).sum(); pd.notnull(df_merge_inc).sum()
df_merge_inc.head(n=25)
df_merge_inc.dropna(how='any',inplace=True)

#merge income based on borough name
df_merge_inc2 = pd.merge(f0_inc_minmax,f0_inc_bor_minmax, on='Borough', how='outer')
df_merge_inc2.head(n=25) #QC at each stage
df_merge_inc2.shape[0]
df_merge_inc2.shape[1]
pd.isnull(df_merge_inc2).sum()
pd.isnull(df_merge_inc2).sum()/ pd.isnull(df_merge_inc).sum() # no impovement merging this
pd.notnull(df_merge_inc2).sum()
df_merge_inc2.drop_duplicates(keep='first', inplace=True)
df_merge_inc2.shape[0] ;df_merge_inc2.shape[1]

#try again merging all data
df_merge4 = pd.merge(df_merge2_minmax,df_merge_inc, on='ONS Code (new)', how='outer')
df_merge4.head(n=25) #QC at each stage
df_merge4.shape[0]
df_merge4.shape[1]
pd.isnull(df_merge4).sum(); pd.notnull(df_merge4).sum() 
df_merge4.dropna(how='any',inplace=True)

#backup everything and write to disk
df_merge4_bk = df_merge4; df_merge4_bk.to_excel('merged_data.xlsx')

#merge health and ward level data 
df_merge5 = pd.merge(df_merge2,f0_inc_minmax, on='ONS Code (new)', how='outer')
df_merge5.head(n=25) #QC at each stage
df_merge5.shape[0]
df_merge5.shape[1]
pd.isnull(df_merge5).sum()
pd.isnull(df_merge5).sum() / pd.isnull(df_merge4).sum() # greater data loss with this method
pd.notnull(df_merge5).sum() 

#Progress with df_merge4 more data than just simple ward level merge

colname_df_m4 = list(df_merge4);print_el(colname_df_m4)

#drop redundant data
df_merge4 = df_merge4.drop(['Area Name _x','Area Name _y','LAD code', 'Borough_y',
                            '2012/13 Ward','2012/13 Bor','YoY13/11 Bor'],axis=1)
df_merge4 = df_merge4.drop(['Lower 95% CI Dep','Upper 95% CI Dep','Lower 95% CI Smok',
                            'Upper 95% CI Smok','Lower 95% CI Unemp',
                            'Upper 95% CI Unemp'],axis=1)                     
colname_df_m4 = list(df_merge4);print_el(colname_df_m4); len(colname_df_m4)
#backup after scaling
df_merge4.to_excel('f2_merged_clean_scaled.xlsx')
df_merge4.head(n=25)

df_merge4.describe()
sns.pairplot(df_merge4, kind='reg') #very useful to see correlations in overall dataset

df_merge4['YoY12/10 Ward'] = df_merge4['2011/12 Ward'] /df_merge4['2010/11 Ward']
var_plot = ('YoY12/10 Bor','Indicator value Dep',
            'Indicator value Unemp', 'Indicator value Smok','YoY12/10 Ward')

sns.pairplot(df_merge4, vars=var_plot, kind='reg', diag_kind='kde', palette="husl" )

###############################################################
#Feature Selection for some interesting model building
###############################################################
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesClassifier

df_merge4.head(n=25)
y = df_merge4['Indicator value Smok']
X = df_merge4.drop(['Indicator value Smok'],axis=1)._get_numeric_data()
pd.isnull(X).sum()#rogue missing value
X['YoY12/10 Ward'] = X['YoY12/10 Ward'].fillna(method='pad');pd.isnull(X).sum()

#http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#example-ensemble-plot-gradient-boosting-regression-py
#Using this example code
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

###############################################################################
# Fit regression model
###########################################################################
params = {'n_estimators': 10, 'max_depth': 10, 'min_samples_split': 10,
          'learning_rate': 0.1, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: " ,mse)

###############################################################################
# Plot training deviance

# compute test set deviance
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

###############################################################################
# Plot feature importance
###############################################################################
feature_importance = clf.feature_importances_

# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
feature_names = list(X)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feature_names)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

              

