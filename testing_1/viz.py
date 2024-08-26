# load dataset and visualize things 

# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

#clustering imports 
from sklearn.preprocessing import MinMaxScaler
import gower
from sklearn.metrics import pairwise_distances
# from sklearn_extra.cluster import KMedoids
# from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from mpl_toolkits.mplot3d import Axes3D


from scipy.cluster.hierarchy import dendrogram, linkage

# %%

elso_df = pd.read_csv('ELSOcleaner.csv')
elso_df.head()

# %%

# NOTE the outcomes that we can care about: 
    # Discontinuation, 127 
    # SurvECMO, 0
    # SurvHosp, 0
    # LOSdays, 152
    # VentDur, 0
    # NOTE: RRTduringECMO, 0
    # HoursECMO 0 
# we MAINLY care about RRTduringECMO (a binary value)

# other significant features; 
    # without missing values: 
        # Race, AgeYrs, CDH, Mode, SupType, pH, AKIpresent, CKD, 
        # CPR, CultPos, RRTpreECMO, HypoTerm, NMB, Pressors, Inotropes, 
        # totPress, PreRRT, PreECMOAKI
    
    # With missing values
        # Sex, 183
        # OI, 4965

    # similar features, just use AKIpresent for now and ignore the other two which are Yes or No. Not useful atm 
        # AKIpresent = PreRRt = PreECMOAKI
        # RRTpreECMO is similar to the above as well. 

print(elso_df.isna().sum())
print('---------')
print(elso_df.shape)

# %%

# a couple of interesting graphs to show quick linear relationships
# NOTE: THIS IS PLOTTING CODE, WHICH IS COMMENTED OUT
# feature_list = ['Race', 'AgeYrs', 'CDH', 'Mode', 'SupType', 'pH', 'AKIpresent', 'CKD', 'CPR', 'CultPos', 'RRTpreECMO', 'HypoTerm', 'NMB', 'Pressors', 'Inotropes', 'totPress', 'PreRRT', 'PreECMOAKI', 'Sex', 'OI']

# for i, feature in enumerate(feature_list):
#     plt.figure(figsize=(8, 6))
#     sns.lineplot(data = elso_df, x= feature, y='RRTduringECMO', marker='o')
#     plt.title(f'Line plot of {feature} vs RRTduringECMO')
#     plt.xlabel(feature)
#     plt.grid(True)
#     plt.show()

# %%

# Possible clusters
# NOTE: Do some clustering with 3 feautres 
    # Import library for elbow technique 
    # then apply and graph the clusters. 
        # save the centroids and use a lambda apply function to create "labels" within the dataset.
#  

# don't worry about Sex in list of features

# Mode, SupType, AgeYrs          # Cluster 1 
# CDH, OI, pH, PreECMOAKI        # cluster 2 
# pH, OI, Inotropes, totPress    # cluster 3 

# numerical: AgeYrs, OI, pH

# binary: inotropes, totPress

# %%


scaled_sets = {}
scaler = MinMaxScaler()

# note that OI is not happy with NANs

elso_df['AgeYrs_norm'] = scaler.fit_transform(elso_df[['AgeYrs']])
# cluster_df_1= elso_df[['Mode', 'SupType', 'AgeYrs_norm']]
# cluster_df_2 = elso_df[['CDH', 'pH', 'PreECMOAKI']]
cluster_df_3 = elso_df[['pH', 'Inotropes', 'totPress']]
dm_3 = gower.gower_matrix(cluster_df_3)
# dm_2 = gower.gower_matrix(cluster_df_2)
# dm_3 = gower.gower_matrix(cluster_df_3)
print(dm_3)
# %%
# dbscan clustering 
dbscan = DBSCAN(eps=0.5, metric='precomputed')
labels = dbscan.fit_predict(dm_3)

# %%
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f'Number of clusters: {n_clusters}')
print(f'Number of noise points: {list(labels).count(-1)}')

# %%
elso_df['Cluster'] = labels 

# Pairwise scatter plots colored by cluster labels
sns.pairplot(elso_df, hue='Cluster', vars=['pH', 'Inotropes', 'totPress'])
plt.show()

# %% 

from sklearn.decomposition import PCA 
pca = PCA(n_components=2, random_state= 42)
reduced_data = pca.fit_transform(cluster_df_3)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', marker='o')
plt.colorbar(scatter)
plt.title('PCA Projection of DBSCAN Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
# %%
elso_df['Cluster1'] = labels


elso_df['Mode_cat'] = pd.Categorical(elso_df['Mode']).codes
elso_df['SupType_cat'] = pd.Categorical(elso_df['SupType']).codes

plt.figure(figsize=(10,6))

scatter_plot = sns.scatterplot(data = elso_df, x='Mode', y='AgeYrs', hue='SupType', palette='viridis', style='SupType', s=100)

scatter_plot.set_title("DBSCAN clustering results")
scatter_plot.set_xlabel('Mode (categorical)')
scatter_plot.set_ylabel('Age (numerical)')
plt.legend(title='Cluster Label')
plt.show()

# %%

print(dm_2)

# do the rest 
dbscan = DBSCAN(eps=0.5, metric='precomputed')
labels = dbscan.fit_predict(dm_2)

elso_df['Cluster2'] = labels 

dbscan = DBSCAN(eps=0.5, metric='precomputed')
labels = dbscan.fit_predict(dm_3)

elso_df['Cluster3'] = labels 
# %%

print(elso_df['Cluster1'])
print('--------')
print(elso_df['Cluster2'])
print('--------')
print(elso_df['Cluster3'])

# %%

# export dataframe and test things in xgboost 

elso_df.to_csv('elso_clusters.csv')