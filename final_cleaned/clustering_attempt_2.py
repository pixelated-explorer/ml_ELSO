# NOTE Attempt 1 at clustering code 


# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from kneefinder import KneeFinder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# %%

df = pd.read_csv('elsoOI_tempML.csv')

df.head()

# %%

# NOTE requirements 
# top 6 or 7 feature s to be used in cluster analysis, add RRT outcome as a feature. it is still unsupervised, but able to compared outcomes between the groups. outcome as a feature to create clusters, and do another cluster analysis if necessary
# I don't remember which dataset we are supposed to be working with 

# phNorm, OInorm, SBPnorm, AgeGr, totPress, PreECMOAKI
# with outcome RRTduringECMO?...

features = ['pHnorm', 'OInorm', 'SBPnorm', 'AgeGr', 'totPress', 'PreECMOAKI', 'RRTduringECMO']
X = df[features]

# note this really shouldn't do anything unless we had a weird outlier we didn't catch 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%

sse = []
k_values = range(1,11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_) # inertia is the SSE for KMeans 

# %%

knee_locater = KneeLocator(k_values, sse, curve='convex', direction='decreasing')

# Step to plot the SSE and KneeLocator graph
plt.figure(figsize=(8, 5))
plt.plot(k_values, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE')
plt.title('Elbow Method showing the optimal k')
plt.grid(True)

# Add KneeLocator point to the plot
plt.axvline(x=knee_locater.elbow, color='r', linestyle='--', label=f'Optimal k = {knee_locater.elbow}')
plt.legend()
plt.show()

print(f"Optimal number of clusters: {knee_locater.elbow}")
# %%
kmeans = KMeans(n_clusters=knee_locater.elbow, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters
# %%

outcome_comparison = df.groupby('Cluster')['RRTduringECMO'].mean()
print(outcome_comparison)

# %%

# looking at the rest of the clusters

print(df[['Cluster'] + features].head())

# %%

# plot may show overfitting: 
    # need to figure out how to build some tolerance for error 

# KNNS are clearly not useful here. 

sns.scatterplot(data=df, x='pHnorm', y='SBPnorm', hue='Cluster', palette='viridis')
plt.title('Clusters')
plt.show()

# %%
Z = linkage(X_scaled, method='ward')

# Step 2: Plot the dendrogram to visualize the clusters
plt.figure(figsize=(10, 7))
dendrogram(Z, labels=df.index, leaf_rotation=90, leaf_font_size=10)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Step 3: Cut the dendrogram at a chosen level to get the desired number of clusters
# e.g., get 3 clusters by specifying the 't' threshold or number of clusters
max_d = 7  # You can change this distance to "cut" the dendrogram
clusters = fcluster(Z, max_d, criterion='distance')

# Alternatively, you can specify the number of clusters:
# n_clusters = 3
# clusters = fcluster(Z, n_clusters, criterion='maxclust')

# Add the clusters to the dataframe
df['Cluster_Hierarchical'] = clusters

# %%

# Print a few rows with cluster labels
print(df[['Cluster_Hierarchical'] + features].head())

# %%

# Outcome comparison based on hierarchical clusters
outcome_comparison = df.groupby('Cluster_Hierarchical')['RRTduringECMO'].mean()
print(outcome_comparison)

# %%

# Plot the clusters using a scatterplot (or any other visualization) to compare
sns.scatterplot(data=df, x='pHnorm', y='SBPnorm', hue='Cluster_Hierarchical', palette='viridis')
plt.title('Hierarchical Clusters based on pHnorm and SBPnorm')
plt.show()