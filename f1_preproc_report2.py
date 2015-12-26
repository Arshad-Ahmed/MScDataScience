"""------------------------------------------------------------------------ """
"""Introduction To Data Science 
    Coursework 2
    Arshad Ahmed
    Flow 1: Preprocessing Data for Analysis"""
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
from sklearn import preprocessing
import networkx as nx
import seaborn as sns; sns.set()
import re, os , sys

"""------------------------------------------------------------------------ """
"""Helper Functions """
"""------------------------------------------------------------------------ """
def colmean(data):
    colmean = np.mean(data.ix[:,4:18], axis = 0)
    return colmean

def colmed(data):
    colmed = np.median(data.ix[:,4:18], axis=0)
    return colmed

def colIQR(data):
    colIQR = np.subtract(*np.percentile(data.ix[:,4:18], [75, 25], axis=0))
    return colIQR    

def colIQRall(data):
    colIQR = np.subtract(*np.percentile(data, [75, 25], axis=0))
    return colIQR 

def colmedall(data):
    colmed = np.median(data, axis=0)
    return colmed
    
def rowmean(data):
    rowmean = np.mean(data.ix[:,4:18], axis =1)
    return rowmean

def rowmed(data):
    rowmed = np.median(data.ix[:,4:18], axis =1)
    return rowmed

def rowstd(data):
    rowstd = np.std(data.ix[:,4:18], axis = 1)
    return rowstd
    
def rowIQR(data):
    rowIQR = np.subtract(*np.percentile(data.ix[:,4:18], [75, 25], axis=1))
    return rowIQR

def pctchg13(data):
    try:
        pctchg13 = ((data[2013]-data[2000])/data[2013])*100
    except:
        pctchg13 = ((data['2013']-data['2000'])/data['2013'])*100   
    return pctchg13
    
def pctchg1yr(data):
    try:
        pctchg1yr = ((data[2013]-data[2012])/data[2013])*100
    except:
        pctchg1yr = ((data['2013']-data['2012'])/data['2013'])*100   
    return pctchg1yr

def pctchg5yr(data):
    try:
        pctchg5yr = ((data[2013]-data[2008])/data[2013])*100
    except:
        pctchg5yr = ((data['2013']-data['2008'])/data['2013'])*100   
    return pctchg5yr

def pctchg10yr(data):
    try:
        pctchg10yr = ((data[2013]-data[2003])/data[2013])*100
    except:
        pctchg10yr = ((data['2013']-data['2003'])/data['2013'])*100   
    return pctchg10yr

def numericdata(data):
    numericdata =  data._get_numeric_data()
    return numericdata

def printrowname(data):
    name = list(data)
    for i in name:
        print(i)

def blankpct(data):
    blankpct = (pd.isnull(data).sum()/data.shape[0])*100
    return blankpct
        
"""------------------------------------------------------------------------ """
"""Step 1: Import Data, QC and Extract Indicators """
"""------------------------------------------------------------------------ """
datadir = 'C:/Users/arsha_000/Downloads/data ideas/'
d0_IDS =   pd.read_excel(datadir + 'IDS_data.xlsx')
d0_WDI =   pd.read_excel(datadir + 'WDI_data.xlsx')
d0_GS  =   pd.read_excel(datadir + 'genderstat_data.xlsx')

printrowname(d0_IDS)
blankpct(d0_IDS)
d0_IDS.head(n=30)

printrowname(d0_WDI)
blankpct(d0_WDI)
d0_WDI.head(n=30)

printrowname(d0_GS)
blankpct(d0_GS)
d0_GS.head(n=30)

"""Strip indicators from gender stats"""
FertilityRate = d0_GS[d0_GS['Indicator Name']=='Fertility rate, total (births per woman)']
LabForFemale = d0_GS[d0_GS['Indicator Name']=='Labor force, female (% of total labor force)']

printrowname(LabForFemale)
printrowname(FertilityRate)
blankpct(FertilityRate)# since good completion use these
blankpct(LabForFemale)

"""Strip indicators from WDI"""
GDP = d0_WDI[d0_WDI['Indicator Name']=='GDP (current US$)']
GDPgrowth = d0_WDI[d0_WDI['Indicator Name']=='GDP growth (annual %)']
GDPperCap = d0_WDI[d0_WDI['Indicator Name']=='GDP per capita, PPP (current international $)']
HealthExp = d0_WDI[d0_WDI['Indicator Name']=='Health expenditure, total (% of GDP)']
CPI = d0_WDI[d0_WDI['Indicator Name']=='Consumer price index (2010 = 100)']
Population =  d0_WDI[d0_WDI['Indicator Name']=='Population, total']
GNI = d0_WDI[d0_WDI['Indicator Name']=='GNI (current US$)']
TotDebtService = d0_WDI[d0_WDI['Indicator Name']=='Total debt service (% of exports of goods, services and primary income)']

blankpct(GDP)
blankpct(GDPperCap)
blankpct(GDPgrowth)
blankpct(HealthExp)
blankpct(CPI)
blankpct(Population)#good use this
blankpct(GNI)
blankpct(TotDebtService)

"""fill na with 0"""
GNI = GNI.fillna(value=0)
TotDebtService = TotDebtService.fillna(value=0)
GDP = GDP.fillna(value=0)
GDPgrowth = GDPgrowth.fillna(value=0)
GDPperCap = GDPperCap.fillna(value=0)
HealthExp = HealthExp.fillna(value=0)
FertilityRate =  FertilityRate.fillna(value=0)
LabForFemale=LabForFemale.fillna(value=0)
CPI = CPI.fillna(value=0)
Population= Population.fillna(value=0)

#Check again
blankpct(FertilityRate)
blankpct(LabForFemale)
blankpct(GDP)
blankpct(GDPperCap)
blankpct(GDPgrowth)
blankpct(HealthExp)
blankpct(CPI)
blankpct(Population)
blankpct(GNI)
blankpct(TotDebtService)

"""------------------------------------------------------------------------ """
"""Step 2_0: Calculate Row Statistics i.e countrywise """
"""------------------------------------------------------------------------ """
GDP['RowMean_GDP'] =   np.mean(GDP.ix[:,4:18], axis=1)
GDP['RowMedian_GDP'] = np.median(GDP.ix[:,4:18], axis =1)
GDP['RowStd_GDP'] =    np.std(GDP.ix[:,4:18], axis = 1)
GDP['RowIQR_GDP'] =    np.subtract(*np.percentile(GDP.ix[:,4:18], [75, 25], axis=1))
GDP['%Chg13yrs_GDP']=pctchg13(GDP)
GDP['%Chg1yrs_GDP']= pctchg1yr(GDP)
GDP['%Chg5yrs_GDP']= pctchg5yr(GDP)
GDP['%Chg10yrs_GDP']= pctchg5yr(GDP)
printrowname(GDP);GDP.head()

GNI['RowMean_GNI'] =   np.mean(GNI.ix[:,4:18], axis=1)
GNI['RowMedian_GNI'] = np.median(GNI.ix[:,4:18], axis =1)
GNI['RowStd_GNI'] =    np.std(GNI.ix[:,4:18], axis = 1)
GNI['RowIQR_GNI'] =    np.subtract(*np.percentile(GNI.ix[:,4:18], [75, 25], axis=1))
GNI['%Chg13yrs_GNI']=pctchg13(GNI)
GNI['%Chg1yrs_GNI']= pctchg1yr(GNI)
GNI['%Chg5yrs_GNI']= pctchg5yr(GNI)
GNI['%Chg10yrs_GNI']= pctchg5yr(GNI)
printrowname(GNI);GNI.head(n=50)

TotDebtService['RowMean_TotDebtService'] =   np.mean(TotDebtService.ix[:,4:18], axis=1)
TotDebtService['RowMedian_TotDebtService'] = np.median(TotDebtService.ix[:,4:18], axis =1)
TotDebtService['RowStd_TotDebtService'] =    np.std(TotDebtService.ix[:,4:18], axis = 1)
TotDebtService['RowIQR_TotDebtService'] =    np.subtract(*np.percentile(TotDebtService.ix[:,4:18], [75, 25], axis=1))
TotDebtService['%Chg13yrs_TotDebtService']=    pctchg13(TotDebtService)
TotDebtService['%Chg1yrs_TotDebtService']=     pctchg1yr(TotDebtService)
TotDebtService['%Chg5yrs_TotDebtService']=     pctchg5yr(TotDebtService)
TotDebtService['%Chg10yrs_TotDebtService']=    pctchg5yr(TotDebtService)
printrowname(TotDebtService);TotDebtService.head(n=50)

GDPgrowth['RowMean_GDPgrowth'] =   np.mean(GDPgrowth.ix[:,4:18], axis=1)
GDPgrowth['RowMedian_GDPgrowth'] = np.median(GDPgrowth.ix[:,4:18], axis =1)
GDPgrowth['RowStd_GDPgrowth'] =    np.std(GDPgrowth.ix[:,4:18], axis = 1)
GDPgrowth['RowIQR_GDPgrowth'] =    np.subtract(*np.percentile(GDPgrowth.ix[:,4:18], [75, 25], axis=1))
GDPgrowth['%Chg13yrs_GDPgrowth']=pctchg13(GDPgrowth)
GDPgrowth['%Chg1yrs_GDPgrowth']= pctchg1yr(GDPgrowth)
GDPgrowth['%Chg5yrs_GDPgrowth']= pctchg5yr(GDPgrowth)
GDPgrowth['%Chg10yrs_GDPgrowth']= pctchg5yr(GDPgrowth)
printrowname(GDPgrowth);GDPgrowth.head(n=50)

GDPperCap['RowMean_GDPperCap'] =   np.mean(GDPperCap.ix[:,4:18], axis=1)
GDPperCap['RowMedian_GDPperCap'] = np.median(GDPperCap.ix[:,4:18], axis =1)
GDPperCap['RowStd_GDPperCap'] =    np.std(GDPperCap.ix[:,4:18], axis = 1)
GDPperCap['%Chg13yrs_GDPperCap']=pctchg13(GDPperCap)
GDPperCap['%Chg1yrs_GDPperCap']= pctchg1yr(GDPperCap)
GDPperCap['%Chg5yrs_GDPperCap']= pctchg5yr(GDPperCap)
GDPperCap['%Chg10yrs_GDPperCap']= pctchg5yr(GDPperCap)
printrowname(GDPperCap);GDPperCap.head(n=50)

HealthExp['RowMean_HealthExp'] =   np.mean(HealthExp.ix[:,4:18], axis=1)
HealthExp['RowMedian_HealthExp'] = np.median(HealthExp.ix[:,4:18], axis =1)
HealthExp['RowStd_HealthExp'] =    np.std(HealthExp.ix[:,4:18], axis = 1)
HealthExp['RowIQR_HealthExp'] =    np.subtract(*np.percentile(HealthExp.ix[:,4:18], [75, 25], axis=1))
HealthExp['%Chg13yrs_HealthExp']=pctchg13(HealthExp)
HealthExp['%Chg1yrs_HealthExp']= pctchg1yr(HealthExp)
HealthExp['%Chg5yrs_HealthExp']= pctchg5yr(HealthExp)
HealthExp['%Chg10yrs_HealthExp']= pctchg5yr(HealthExp)
printrowname(HealthExp);HealthExp.head(n=50)

CPI['RowMean_CPI'] =   np.mean(CPI.ix[:,4:18], axis=1)
CPI['RowMedian_CPI'] = np.median(CPI.ix[:,4:18], axis =1)
CPI['RowStd_CPI'] =    np.std(CPI.ix[:,4:18], axis = 1)
CPI['RowIQR_CPI'] =    np.subtract(*np.percentile(CPI.ix[:,4:18], [75, 25], axis=1))
CPI['%Chg13yrs_CPI']=pctchg13(CPI)
CPI['%Chg1yrs_CPI']= pctchg1yr(CPI)
CPI['%Chg5yrs_CPI']= pctchg5yr(CPI)
CPI['%Chg10yrs_CPI']= pctchg5yr(CPI)
printrowname(CPI);CPI.head(n=30) 

FertilityRate['RowMean_FertilityRate'] =   np.mean(FertilityRate.ix[:,4:18], axis=1)
FertilityRate['RowMedian_FertilityRate'] = np.median(FertilityRate.ix[:,4:18], axis =1)
FertilityRate['RowStd_FertilityRate'] =    np.std(FertilityRate.ix[:,4:18], axis = 1)
FertilityRate['RowIQR_FertilityRate'] =    np.subtract(*np.percentile(FertilityRate.ix[:,4:18], [75, 25], axis=1))
FertilityRate['%Chg13yrs_FertilityRate']=pctchg13(FertilityRate)
FertilityRate['%Chg1yrs_FertilityRate']= pctchg1yr(FertilityRate)
FertilityRate['%Chg5yrs_FertilityRate']= pctchg5yr(FertilityRate)
FertilityRate['%Chg10yrs_FertilityRate']= pctchg5yr(FertilityRate)
printrowname(FertilityRate);FertilityRate.head()

LabForFemale['RowMean_LabForFemale'] =   np.mean(LabForFemale.ix[:,4:18], axis=1)
LabForFemale['RowMedian_LabForFemale'] = np.median(LabForFemale.ix[:,4:18], axis =1)
LabForFemale['RowStd_LabForFemale'] =    np.std(LabForFemale.ix[:,4:18], axis = 1)
LabForFemale['RowIQR_LabForFemale'] =    np.subtract(*np.percentile(LabForFemale.ix[:,4:18], [75, 25], axis=1))
LabForFemale['%Chg13yrs_LabForFemale']=pctchg13(LabForFemale)
LabForFemale['%Chg1yrs_LabForFemale']= pctchg1yr(LabForFemale)
LabForFemale['%Chg5yrs_LabForFemale']= pctchg5yr(LabForFemale)
LabForFemale['%Chg10yrs_LabForFemale']= pctchg5yr(LabForFemale)
printrowname(LabForFemale);LabForFemale.head()

Population['RowMean_Population'] =   np.mean(Population.ix[:,4:18], axis=1)
Population['RowMedian_Population'] = np.median(Population.ix[:,4:18], axis =1)
Population['RowStd_Population'] =    np.std(Population.ix[:,4:18], axis = 1)
Population['RowIQR_Population'] =    np.subtract(*np.percentile(Population.ix[:,4:18], [75, 25], axis=1))
Population['%Chg13yrs_Population']=    pctchg13(Population)
Population['%Chg1yrs_Population']=     pctchg1yr(Population)
Population['%Chg5yrs_Population']=     pctchg5yr(Population)
Population['%Chg10yrs_Population']=    pctchg5yr(Population)
printrowname(Population);Population.head()
list(Population)


"""------------------------------------------------------------------------ """
"""Step 2_1: Calculate column stats  """
"""------------------------------------------------------------------------ """
colmeanGNI = colmean(GNI)
colmeanTotDebtService = colmean(TotDebtService)
colmeanGDP = colmean(GDP)
colmeanGDPgrowth = colmean(GDPgrowth)
colmeanGDPperCap = colmean(GDPperCap)
colmeanHealthExp = colmean(HealthExp)
colmeanFertilityRate =  colmean(FertilityRate)
colmeanLabForFemale=    colmean(LabForFemale)
colmeanCPI =            colmean(CPI)
colmeanPopulation= colmean(Population)

plt.plot(colmeanGNI,colmeanGDP,'o')
sns.regplot(colmeanCPI,colmeanFertilityRate)

colname = ('colmeanGDP', 'colmeanGDPperCap', 'colmeanGDPgrowth', 'colmeanHealthExp', 'colmeanFertilityRate', 'colmeanLabForFemale',
           'colmeanCPI', 'colmeanPopulation', 'colmeanGNI','colmeanTotDebtService')

colstatarray = pd.concat([colmeanGDP, colmeanGDPperCap, colmeanGDPgrowth, 
           colmeanHealthExp, colmeanFertilityRate, colmeanLabForFemale,
           colmeanCPI, colmeanPopulation, colmeanGNI,
           colmeanTotDebtService], axis=1, ignore_index=True)
colstatarray.columns=colname
colstatarray.head()
sns.pairplot(colstatarray, diag_kind='kde', markers="+",  palette="Greens_d", plot_kws=dict(s=50, edgecolor="b", linewidth=1),
 diag_kws=dict(shade=False))

colmedGNI = colmed(GNI)
colmedTotDebtService = colmed(TotDebtService)
colmedGDP = colmed(GDP)
colmedGDPgrowth = colmed(GDPgrowth)
colmedGDPperCap = colmed(GDPperCap)
colmedHealthExp = colmed(HealthExp)
colmedFertilityRate =  colmed(FertilityRate)
colmedLabForFemale=    colmed(LabForFemale)
colmedCPI =            colmed(CPI)
colmedPopulation= colmed(Population)

colname1 = ('colmedGDP', 'colmedGDPperCap', 'colmedGDPgrowth', 'colmedHealthExp', 
           'colmedFertilityRate', 'colmedLabForFemale',
           'colmedCPI', 'colmedPopulation', 'colmedGNI','colmedTotDebtService')
colstatarray_median = pd.concat([colmeanGDP, colmeanGDPperCap, colmeanGDPgrowth, 
           colmeanHealthExp, colmeanFertilityRate, colmeanLabForFemale,
           colmeanCPI, colmeanPopulation, colmeanGNI,
           colmeanTotDebtService], axis=1, ignore_index=True)
colstatarray_median.columns=colname1

sns.pairplot(colstatarray_median, diag_kind='kde', markers="+",  palette="Greens_d", plot_kws=dict(s=50, edgecolor="b", linewidth=1),
 diag_kws=dict(shade=False))
plt.tight_layout()
"""------------------------------------------------------------------------ """
""" Step 3: Merge Indicators into one large matrix """
"""------------------------------------------------------------------------ """
coldrop = ['Indicator Name', 'Indicator Code','2000','2001','2002', '2003',
 '2004', '2005', '2006', '2007', '2008', '2009', '2010','2011','2012','2013']
 
GNI = GNI.drop(coldrop, axis=1);printrowname(GNI)
TotDebtService = TotDebtService.drop(coldrop, axis=1)
GDP = GDP.drop(coldrop, axis=1)
GDPgrowth = GDPgrowth.drop(coldrop, axis=1)
GDPperCap = GDPperCap.drop(coldrop, axis=1)
HealthExp = HealthExp.drop(coldrop, axis=1)
FertilityRate =  FertilityRate.drop(coldrop, axis=1)
LabForFemale=LabForFemale.drop(coldrop, axis=1)
CPI = CPI.drop(coldrop, axis=1)
Population= Population.drop(coldrop, axis=1)


merge1 = pd.merge(GNI,TotDebtService, on=['Country Code','Country Name'], how='outer',suffixes=('_GNI', '_TotDebtService'))
merge1.head(n=20)
merge2 = pd.merge(GDP,GDPgrowth, on=['Country Code','Country Name'], how='outer',suffixes=('_GDP', '_GDPgrowth'))
merge2.head(n=20)
merge3 = pd.merge(GDPperCap,HealthExp, on=['Country Code','Country Name'], how='outer',suffixes=('_GDPperCap', '_HealthExp'))
merge3.head(n=10)
merge4= pd.merge(FertilityRate,LabForFemale, on=['Country Code','Country Name'], how='outer',suffixes=('_FertilityRate','_LabForFemale'))
merge4.head(n=10)
merge5= pd.merge(CPI,Population, on=['Country Code','Country Name'], how='outer',suffixes=('_CPI','_Population'))
merge5.head(n=10)

#merge the merged

merge6 = pd.merge(merge1,merge2,  on=['Country Code','Country Name'], how='outer')
merge7 = pd.merge(merge6,merge3,  on=['Country Code','Country Name'], how='outer')
merge8 = pd.merge(merge7,merge4,  on=['Country Code','Country Name'], how='outer')
merge9 = pd.merge(merge8,merge5,  on=['Country Code','Country Name'], how='outer')
merge9.head()

"""------------------------------------------------------------------------ """
"""Step 4: Clean up post merged data """
"""------------------------------------------------------------------------ """
"""Backup merged file """
merge9.to_excel(datadir + 'merged_data.xlsx')
totals = merge9.ix[1:33,:]#exclude totals
countryname = merge9['Country Name'].ix[34::];printrowname(countryname)
merge9 = merge9.ix[34::,:]

"""Fill any remaining blanks after merging """
merge9 = pd.read_excel(datadir + 'merged_data.xlsx')
d1_fillnan = merge9.fillna(value=0) #fill values after merging 
d2_interpolate = d1_fillnan.interpolate(method='values')
d2_interpolate = d2_interpolate.drop(['Country Code'], axis=1)

corr_merged = np.corrcoef(numericdata(d1_fillnan))
plt.figure(figsize=(10,10))
sns.heatmap(corr_merged, cmap="RdYlGn",square=True, xticklabels=10, yticklabels=10, fmt='d')
title('Correlation Matrix Of Data After Merging without scaling')
 
"""------------------------------------------------------------------------ """
"""Step 5: Test Different Scalars """
"""------------------------------------------------------------------------ """
noscale_median = colmedall(d1_fillnan.ix[:,2::])
noscale_iqr = colIQRall(d1_fillnan.ix[:,2::])

d2_input = numericdata(d1_fillnan)
d2_input = np.nan_to_num(d2_input)
blankpct(d2_input)

maxabs_scaler = sklearn.preprocessing.MaxAbsScaler()
minmax_scaler = sklearn.preprocessing.MinMaxScaler()
robust_scaler = sklearn.preprocessing.RobustScaler()
standard_scaler = sklearn.preprocessing.StandardScaler()

d2_maxabs = maxabs_scaler.fit_transform(d2_input)
d2_minmax = minmax_scaler.fit_transform(d2_input)
d2_robust = robust_scaler.fit_transform(d2_input)
d2_standard = standard_scaler.fit_transform(d2_input)

minmax_median = colmedall(d2_minmax)
maxabs_median = colmedall(d2_maxabs)
robust_median = colmedall(d2_robust)
standard_median = colmedall(d2_standard)

standard_iqr = colIQRall(d2_standard)
maxabs_iqr = colIQRall(d2_maxabs)
minmax_iqr = colIQRall(d2_minmax)
robust_iqr = colIQRall(d2_robust)

import scipy.stats 
from scipy.stats import kendalltau, pointbiserialr, pearsonr


sns.jointplot(noscale_median, noscale_iqr,stat_func=pearsonr , color="#4CB391",s=20, edgecolor="w", linewidth=1)
title('Median IQR Plot of data no scaling')

sns.jointplot(standard_median, standard_iqr ,stat_func=pearsonr, color="#4CB391", s=20, edgecolor="w", linewidth=1)
title('Median IQR Plot of data after Standard Scaling')

sns.jointplot(minmax_median, minmax_iqr ,stat_func=pearsonr , color="#4CB391",s=20, edgecolor="w", linewidth=1)
title('Median IQR Plot of data after Min Max Scaling')

sns.jointplot(maxabs_median, maxabs_iqr ,stat_func=pearsonr , color="#4CB391",s=20, edgecolor="w", linewidth=1)
title('Median IQR Plot of data after Max Abs Scaling')


"""Plot grid of scaling graphs """
fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(421)
ax.scatter(noscale_median, noscale_iqr, c="#4CB391", marker='o')
title('Median IQR Plot of data no scaling')

ax = fig.add_subplot(422, projection='polar')
ax.scatter(noscale_median, noscale_iqr, c="#4CB391", marker='o')
title('Median IQR Plot of data no scaling polar projection')

ax = fig.add_subplot(423)
ax.scatter(standard_median, standard_iqr, c="#4CB391", marker='o')
title('Median IQR Plot of data after Standard Scaling')

ax = fig.add_subplot(424, projection='polar')
ax.scatter(standard_median, standard_iqr, c="#4CB391", marker='o')
title('Median IQR Plot of data after Standard Scaling polar projection')

ax = fig.add_subplot(425)
ax.scatter(minmax_median, minmax_iqr, c="#4CB391", marker='o')
title('Median IQR Plot of data after Min Max Scaling')

ax = fig.add_subplot(426, projection='polar')
ax.scatter(minmax_median, minmax_iqr, c="#4CB391", marker='o')
title('Median IQR Plot of data after Min Max Scaling polar projection')

ax = fig.add_subplot(427)
ax.scatter(maxabs_median, maxabs_iqr, c="#4CB391", marker='o')
title('Median IQR Plot of data after Max Abs Scaling')

ax = fig.add_subplot(428, projection='polar')
ax.scatter(maxabs_median, maxabs_iqr, c="#4CB391", marker='o')
title('Median IQR Plot of data after Max Abs Scaling polar projection')
plt.tight_layout()

#pad=0.8, w_pad=0.5, h_pad=2.0
#this one is usless
sns.jointplot(robust_median, robust_iqr ,stat_func=pointbiserialr , color="#4CB391",s=20, edgecolor="w", linewidth=1)

#max abs scaling looks best

corcoef = np.corrcoef(d2_maxabs)
covariance = np.cov(d2_maxabs)

plt.figure(figsize=(10,10))
sns.heatmap(corcoef, cmap="RdYlGn",square=True, xticklabels=10, yticklabels=10, fmt='d')
title('Correlation Matrix of Data after Max Abs Scaling')

sns.heatmap(covariance, cmap="seismic", square=True, xticklabels=8, yticklabels=8, fmt='d')

sns.clustermap(d2_maxabs, cmap='jet',figsize=(13, 13), metric='cityblock')
sns.clustermap(d2_maxabs, cmap='jet',figsize=(13, 13), metric='chebyshev')
sns.clustermap(d2_maxabs, cmap='jet',figsize=(13, 13), metric='minkowski', method='centroid')
sns.clustermap(d2_maxabs, cmap='RdYlGn',figsize=(13, 13), metric='minkowski', method='single', xticklabels=8, yticklabels=8)

printrowname(d2_maxabs)
print(d2_maxabs)
list(d2_maxabs)

""" Winner is MAXABS Scaling """

"""------------------------------------------------------------------------ """
"""Step 6: Apply Max Abs Scaling to full data and construct dataframe """
"""------------------------------------------------------------------------ """

colname_d4 = list(d2_interpolate);colname_d4[1::]
d3_numeric = numericdata(d2_interpolate)
d3_numeric.head()
np.nan_to_num(d3_numeric)
d3_scaled = maxabs_scaler.fit_transform(np.nan_to_num(d3_numeric))
d4 = pd.DataFrame(d3_scaled)
d4.columns = colname_d4[1::]
d4.head()

d4.to_excel(datadir + 'merged_maxabs_scaled_data.xlsx')

"""Visualise post merged data before scaling"""

corr_d1 = d1_fillnan.corr()
corr_d4 = d4.corr()

plt.figure(figsize=(16,16))     
plt.subplot(2,1,1)           
cmap = sns.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
sns.heatmap(corr_d1, cmap=cmap, linewidths=1,yticklabels=2, fmt='d')
title('Correlation Matrix Before Data Scaling')
#Heatmap scaled data
plt.subplot(2,1,2)     
sns.heatmap(corr_d4, cmap=cmap, linewidths=1,yticklabels=2, fmt='d')
title('Correlation Matrix After Data Scaling')
plt.tight_layout()