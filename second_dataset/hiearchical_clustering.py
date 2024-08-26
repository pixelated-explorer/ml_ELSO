# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from kneefinder import KneeFinder
import seaborn as sns 

# from scipy.cluster.hierarchy import dendogram, linkage 
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.datasets import make_blobs

# %%

elsoCA = pd.read_csv('elsoCA1.csv')

#ELSO CA1
#phNorm, OInorm, SBPnorm, totpress
# %%
# normalize totpress specifically 
scaler = MinMaxScaler()
elsoCA['totPress'] = scaler.fit_transform(elsoCA[['totPress']])
df_cluster = elsoCA[['pHnorm', 'OInorm', 'SBPnorm', 'totPress']]

# %%

k_rng = range(1,10)
sse = []

for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df_cluster)
    sse.append(km.inertia_)
# %%

print(sse)

# %%

kf = KneeFinder(k_rng, sse)
knee_x, knee_y = kf.find_knee()
kf.plot()

# %%
km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(df_cluster)
elsoCA['cluster_1'] = y_predicted


# %%

sns.pairplot(elsoCA, hue='cluster_1', palette='Set2', diag_kind='kde')
plt.suptitle('Cluster Visualization with Pairplot', y=1.02)
plt.show()

# %%

elsoCA.to_csv('elsoCA_cluster1.csv')