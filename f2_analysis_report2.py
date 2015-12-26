"""------------------------------------------------------------------------ """
"""Introduction To Data Science 
    Coursework 2
    Arshad Ahmed
    Flow 2: Data Analysis"""
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
"""Step 7: Import Scaled and Reduced Data for Analysis """
"""------------------------------------------------------------------------ """
datadir = 'C:/Users/arsha_000/Downloads/data ideas/'
d4 =pd.read_excel(datadir + 'merged_maxabs_scaled_data.xlsx')
d4 = d4.ix[34::,:]
d4.head()
d4.shape[1]
d4.shape[0]
printrowname(d4)

d4= d4.replace([np.inf,np.nan,-np.inf],0)
sns.heatmap(d4, cmap="RdYlGn", xticklabels=3, yticklabels=8, fmt='d')

corcoef = np.corrcoef(d4)
covariance = np.cov(d4)

sns.heatmap(corcoef, cmap="RdYlGn",square=True, xticklabels=6, yticklabels=8, fmt='d')
sns.heatmap(covariance, cmap="RdYlGn",square=True, xticklabels=6, yticklabels=8, fmt='d')

"""------------------------------------------------------------------------ """
"""Step 8_0: Visualise Means """
"""------------------------------------------------------------------------ """

rowmeans = pd.concat([
            d4['RowMean_TotDebtService'],d4['RowMean_GNI'],
            d4['RowMean_Population'],d4['RowMean_GNI'], d4['RowMean_Population'],
            d4['RowMean_LabForFemale'],d4['RowMean_CPI'],d4['RowMean_FertilityRate'],          
            d4['RowMean_GDP'],d4['RowMean_GDPperCap'],d4['RowMean_GDPgrowth'],
            d4['RowMean_HealthExp']     
             ], axis=1)
rowmeans.head()   
corr_rowmeans = np.corrcoef(rowmeans)          
sns.heatmap(rowmeans, cmap="RdYlGn", yticklabels=False)
sns.heatmap(corr_rowmeans, cmap="RdYlGn", yticklabels=False,xticklabels=5)
sns.pairplot(rowmeans)

""" See some interesting trends it would be worth looking at ratios of these
to compare countries by """

"""------------------------------------------------------------------------ """
"""Step 8_1: Calculate Additional Features """
"""------------------------------------------------------------------------ """
Debt_toGNI =        d4['RowMean_TotDebtService']/d4['RowMean_GNI']
Pop_toGNI=          d4['RowMean_Population']/d4['RowMean_GNI']
Pop_toLabForFem=    d4['RowMean_Population']/d4['RowMean_LabForFemale']
CPI_toFertRate =    d4['RowMean_CPI']/d4['RowMean_FertilityRate']
GDP_toGDPpCap =     d4['RowMean_GDP']/d4['RowMean_GDPperCap']
HealthExp_toGDP =   d4['RowMean_HealthExp']/d4['RowMean_GDP']
HealthExp_toGDPgrowth=d4['RowMean_HealthExp']/d4['RowMean_GDPgrowth']
HealthExp_toFertRate = d4['RowMean_HealthExp']/d4['RowMean_FertilityRate']
HealthExp_toPop =      d4['RowMean_HealthExp']/d4['RowMean_Population']
FertRate_toGDPgrowth = d4['RowMean_FertilityRate']/d4['RowMean_GDPgrowth']
FertRate_toGDPperCap=  d4['RowMean_FertilityRate']/d4['RowMean_GDPperCap']
FertRate_toLabForFem = d4['RowMean_FertilityRate']/d4['RowMean_LabForFemale']
FertRate_toCPI=        d4['RowMean_FertilityRate']/d4['RowMean_CPI']

"""Visualise Ratios """
means_ratio = [Debt_toGNI,Pop_toGNI, Pop_toLabForFem ,CPI_toFertRate, 
GDP_toGDPpCap,HealthExp_toGDP,HealthExp_toGDPgrowth, HealthExp_toFertRate,
HealthExp_toPop, FertRate_toGDPgrowth, FertRate_toGDPperCap,FertRate_toLabForFem,
FertRate_toCPI]

means_ratio_name = ['Debt_toGNI','Pop_toGNI', 'Pop_toLabForFem' ,'CPI_toFertRate', 
'GDP_toGDPpCap','HealthExp_toGDP','HealthExp_toGDPgrowth','HealthExp_toFertRate',
'HealthExp_toPop', 'FertRate_toGDPgrowth', 'FertRate_toGDPperCap','FertRate_toLabForFem',
'FertRate_toCPI']


means_ratio_array = pd.concat(means_ratio,axis=1)
means_ratio_array.columns = means_ratio_name; means_ratio_array.head()

means_ratio_array = means_ratio_array.replace([np.inf,np.nan,-np.inf],0)
np.isnan(means_ratio_array).any()

corr_meanratio = np.corrcoef(means_ratio_array)
sns.heatmap(corr_meanratio, cmap="RdYlGn",xticklabels=6, yticklabels=8, fmt='d')
title('Heatmap of Correlation Matrix of Calculated Means Ratios')

plt.plot(Debt_toGNI, 'go')
plt.plot(Pop_toGNI, 'go')
plt.plot(Pop_toLabForFem, 'go')
plt.plot(Pop_toGNI, 'go')
plt.plot(CPI_toFertRate, 'go')
plt.plot(GDP_toGDPpCap, 'go')
plt.plot(HealthExp_toFertRate, 'go')
plt.semilogy(HealthExp_toGDP, 'go')
plt.plot(FertRate_toCPI, 'go')
plt.plot(FertRate_toGDPgrowth, 'go')
plt.plot(FertRate_toGDPperCap, 'go')
plt.plot(FertRate_toCPI, 'go')

plt.loglog(Debt_toGNI, 'go')
plt.loglog(Pop_toGNI, 'go')
plt.loglog(Pop_toLabForFem, 'go')
plt.loglog(Pop_toGNI, 'go')
plt.loglog(CPI_toFertRate, 'go')
plt.loglog(GDP_toGDPpCap, 'go')
plt.loglog(HealthExp_toFertRate, 'go')
plt.loglog(HealthExp_toGDP, 'go')
plt.loglog(FertRate_toCPI, 'go')
plt.loglog(FertRate_toGDPgrowth, 'go')
plt.loglog(FertRate_toGDPperCap, 'go')
plt.loglog(FertRate_toCPI, 'go')


"""Check if median ratio is worth doing seems same as mean in this case"""
"""-------------------------------------------------------------------------"""
Debt_toGNI_med = d4['RowMedian_TotDebtService']/d4['RowMedian_GNI']
plt.plot(Debt_toGNI_med, 'go')
np.argmax(Debt_toGNI_med)
d4[195:,]
countryname[195]

""" Create a custome score for this data """
"""-------------------------------------------------------------------------"""
#ahmed_score = np.arccosh(np.log10(sum(means_ratio_array,axis=1)\
#                        .replace([np.inf,np.nan,-np.inf],0)**(1.0/3)))

ahmed_score = np.arccosh(np.log10(np.abs(sum(means_ratio_array,axis=1)\
                        .replace([np.inf,np.nan,-np.inf],0)**(1.0/3))))                     
    
ahmed_score.describe()

                   
plt.plot(ahmed_score,'#4CB391')

plt.figure(figsize=(10,10))
plt.subplot(2,2,1) 
ahmed_score_sorted= np.sort(ahmed_score.replace([np.inf,np.nan,-np.inf],0), axis=0)[::-1]
plt.plot(ahmed_score_sorted, "#4CB391")
title('Plot of Sorted Ahmed Score of Data')
ylabel('Ahmed Score')
xlabel('Country Index')

plt.subplot(2,2,2) 
sns.distplot(ahmed_score_sorted, color="#4CB391")
title('Distribution Plot of Sorted Ahmed Score of Data')
ylabel('Ahmed Score')
xlabel('Score Distribution')
plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=2.0)


plt.plot(ahmed_score_sorted[ahmed_score_sorted > 0], "#4CB391")
title('Plot of Sorted Ahmed Score ( >0) of Data')
ylabel('Ahmed Score')
xlabel('Country Index')

sns.distplot(ahmed_score_sorted[ahmed_score_sorted > 0],color="#4CB391")
title('Distribution Plot of Sorted Ahmed Score ( >0) of Data')
ylabel('Ahmed Score')
xlabel('Country Index')

"""------------------------------------------------------------------------ """
"""Step 9: Add Means Ratios to original data, fill missing values """
"""------------------------------------------------------------------------ """

d5_colname = list(d4)
tmp = ('Debt_toGNI','Pop_toGNI', 'Pop_toLabForFem' ,'CPI_toFertRate', 
'GDP_toGDPpCap','HealthExp_toGDP','HealthExp_toGDPgrowth','HealthExp_toFertRate',
'HealthExp_toPop', 'FertRate_toGDPgrowth', 'FertRate_toGDPperCap',
'FertRate_toLabForFem','FertRate_toCPI')
d5_colname.extend(tmp)

print(d5_colname)
d5 = pd.concat([d4, means_ratio_array], axis=1)
d5.head()
d5.tail(n=20)
d5 = pd.DataFrame(np.nan_to_num(d5))
d5.columns = d5_colname
d5.head(n=50)
d5.shape[1]

""" Add score to data """
"""-------------------------------------------------------------------------"""
d5['AhmedScore'] = np.arccosh(np.log10(sum(means_ratio_array,axis=1)\
                    .replace([np.inf,np.nan,-np.inf],0)**(1.0/3)))
d5.head(n=50)
d5['AhmedScore'] =d5['AhmedScore'].replace([np.inf,np.nan,-np.inf],0)
d5_cname = list(d5)
d5.to_excel(datadir + 'input_to_modeling.xlsx')
d5.shape[0]
d5.shape[1]
"""------------------------------------------------------------------------ """
"""Step 10: Decompose Data to explore structures """
"""------------------------------------------------------------------------ """

""" PCA """
"""-------------------------------------------------------------------------"""

pca = sklearn.decomposition.PCA()
pca_data = pca.fit(d5)
pca_data.explained_variance_ratio_
sum(pca_data.explained_variance_ratio_[0:2])#3 components will do the job
pca_data.components_
pca_data.n_components_

pca_data_fit = pca.fit_transform(d5)
pca1 = pca_data.components_[:,0]
pca2 = pca_data.components_[:,1]
pca3 = pca_data.components_[:,2]


pca1.shape[1]

pca1_data = matrix(pca1) * matrix(d5).transpose()
pca2_data = matrix(pca2) * matrix(d5).transpose()
pca3_data = matrix(pca3) * matrix(d5).transpose()


np.argmin(pca1_data)
np.argmin(pca2_data)
np.argmin(pca3_data)

countryname[38]
countryname[195]

sns.jointplot(pca1_data, pca2_data, color='#DC143C',stat_func=pearsonr)
sns.jointplot(pca2_data, pca3_data, color='#DC143C',stat_func=pearsonr)

""" ICA """
"""-------------------------------------------------------------------------"""

ica = sklearn.decomposition.FastICA(n_components=3)
ica_model = ica.fit(d5)
data_ica = ica.fit_transform(d5)
data_ica
ica1 = data_ica[:,0]
ica2 = data_ica[:,1]
ica3 = data_ica[:,2]

ica1.shape[0]

""" Manifold """
"""-------------------------------------------------------------------------"""
data_lle, err_lle = sklearn.manifold.locally_linear_embedding(d5,n_neighbors=12, n_components=3)
lle1 = data_lle[:,0]
lle2 = data_lle[:,1]
lle3 = data_lle[:,2]
data_lle.shape[1]
data_lle.shape[0]
err_lle

""" t-SNE """

tsne = manifold.TSNE(n_components=3)
data_tsne = tsne.fit_transform(d5)
data_tsne.shape[1]
tsne1 =data_tsne[:,0]
tsne2 =data_tsne[:,1]
tsne3 =data_tsne[:,2]

""" Factor Analysis """

fa = decomposition.FactorAnalysis()
data_fa = fa.fit_transform(d5)
fa.components_[:,0]
fa.components_[:,1]

fa1 = data_fa[:,0]
fa2 = data_fa[:,1]
fa3 = data_fa[:,2]

fa.get_precision()
fa.loglike_

data_fa.shape[0]
data_fa.shape[1]

"""Non Negative Matrix Factorization """
from sklearn.decomposition import NMF
nmf = NMF(n_components=7,init='random', random_state=0)
nmf_model = nmf.fit(np.abs(d5))
data_nmf = nmf.fit_transform(np.abs(d5))
nmf_model.reconstruction_err_

nmf_model.n_components_
nmf_model.components_


nmf1=data_nmf[:,0]
nmf2=data_nmf[:,1]
nmf3=data_nmf[:,2]

""" Put all dimensionality reduction plots together """
"""--------------------------------------------------------------""""

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(321, projection='3d')
ax.scatter(ica1, ica2, ica3, c="#4CB391", marker='o')
ax.set_xlabel('ICA 1')
ax.set_ylabel('ICA 2')
ax.set_zlabel('ICA 3')
title('ICA 3 Components Scatter Plot')

ax = fig.add_subplot(322, projection='3d')
ax.scatter(pca1, pca2, pca3, c="#4CB391", marker='o')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
title('PCA 3 Components Scatter Plot')

ax = fig.add_subplot(323, projection='3d')
ax.scatter(fa1, fa2, fa3, c="#4CB391", marker='o')
ax.set_xlabel('FA 1')
ax.set_ylabel('FA 2')
ax.set_zlabel('FA 3')
title('Factor Analysis 3 Components Scatter Plot')

ax = fig.add_subplot(324, projection='3d')
ax.scatter(tsne1, tsne2, tsne3, c="#4CB391", marker='o')
ax.set_xlabel('TSNE 1')
ax.set_ylabel('TSNE 2')
ax.set_zlabel('TSNE 3')
title('TSNE 3 Components Scatter Plot')

ax = fig.add_subplot(325, projection='3d')
ax.scatter(lle1, lle2, lle3, c="#4CB391", marker='o')
ax.set_xlabel('LLE 1')
ax.set_ylabel('LLE 2')
ax.set_zlabel('LLE 3')
title('LLE 3 Components Scatter Plot')

ax = fig.add_subplot(326, projection='3d')
ax.scatter(nmf1, nmf2, nmf3, c="#4CB391", marker='o')
ax.set_xlabel('NMF 1')
ax.set_ylabel('NMF 2')
ax.set_zlabel('NMF 3')
title('Non Negative Matrix Factorization 3 Components Scatter Plot')

plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=2.0)

""" Plot a more informative clustermap """

corr_d4 = d4.corr()

sns.clustermap(corr_d4)
sns.clustermap(corr_d4, method='weighted')
sns.clustermap(corr_d4, method='weighted', metric='correlation')

cmap = sns.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
#with colorbar
sns.clustermap(corr_d4, method='weighted', metric='correlation', col_cluster=False,
               figsize=(16, 16), cmap=cmap, linewidths=1)
#without colorbar
cm = sns.clustermap(corr_d4, method='weighted', metric='correlation', col_cluster=False,
               figsize=(16, 16), cmap=cmap, linewidths=1)
cm.cax.set_visible(False)


from scipy.spatial import distance
from scipy.cluster import hierarchy
corr_array = np.asarray(corr_d4)

row_linkage = hierarchy.linkage(
    distance.pdist(corr_array), method='weighted')

col_linkage = hierarchy.linkage(
    distance.pdist(corr_array.T), method='weighted')
    

sns.clustermap(corr_d4, row_linkage=row_linkage, col_linkage=col_linkage, method="average",
                figsize=(16, 16), cmap='nipy_spectral')

