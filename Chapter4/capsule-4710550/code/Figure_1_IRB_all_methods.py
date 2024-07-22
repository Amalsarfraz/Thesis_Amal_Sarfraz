#!/usr/bin/env python
# coding: utf-8

# **This routine is to generate Figure 1 for paper "Titled:A Robust Two-step Method for Clustering and Dealing with Outlier Sets in Big Data"**
# 
# 
# 
# For any querries contact asarfraz1@sheffield.ac.uk 

# ## Importing libraries

# In[ ]:


### Run once
#get_ipython().system('pip install pyod')


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from pyod.models.cblof import CBLOF
from sklearn.ensemble import IsolationForest
import time
import psutil
import warnings
warnings.simplefilter('ignore')
## defining the dataset
df_2100 = pd.read_csv('2100_cotton.csv')
data = df_2100[['cotton_withdrawal', 'cotton_scarcity']]


# ## Defining all the methods 

# The following methods are defined in order of their plotting Feature Bagging, Relative-KNN-kernel density-based clustering algorithm (REDCLAN), Cluster-Based Local Outlier Factor (CBLOF), Detection with Explicit Micro-Cluster Assignments (D.MCA) and Differential Potential Spread Loss (DPSL).

# ## Feature Bagging

# In[2]:


# Feature Bagging using Isolation Forest
iso_forest = IsolationForest(n_estimators=100, random_state=42)
outlier_predictions = iso_forest.fit_predict(data)
outlier_labels = np.where(outlier_predictions == -1, 1, 0)
df_2100['fb_outlier'] = outlier_labels


# ## REDCLAN

# In[3]:


#####################REDCLAN functions

def knn(X, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)
    return indices, distances

def compute_core_points(X, k1):
    indices, dists = knn(X, k1+1)
    dists = np.sort(dists[:,1:], axis=1)
    d_avg = np.mean(dists, axis=1)
    d_max = np.max(dists, axis=1)
    bandwidth = d_max + d_avg + 1 - np.sqrt(np.mean(dists**2, axis=1))
    density = 1 / (np.pi * bandwidth**2) 
    rel_density = density / density[indices[:,1:]].sum(axis=1)
    core_pts = np.where(rel_density >= np.percentile(rel_density, 10))[0]  
    return core_pts

def dbscan_cluster(X, core_pts, k2):
    cluster_id = 0
    clusters = {}  # Using a dictionary to store clusters
    visited = set()  # To keep track of visited points

    # Initialize all points as noise (outliers)
    labels = -np.ones(X.shape[0])

    for i in core_pts:
        if i not in visited:
            visited.add(i)
            cluster = [i]
            candidates = [i]

            while candidates:
                point = candidates.pop()
                indices, _ = knn(X, k2)
                neighbors = indices[point][1:]  # Get the neighbors

                for n in neighbors:
                    if n not in visited and n in core_pts:
                        visited.add(n)
                        candidates.append(n)
                        cluster.append(n)

            if len(cluster) >= 10:  # Minimum points to form a cluster
                for point in cluster:
                    labels[point] = cluster_id
                clusters[cluster_id] = cluster
                cluster_id += 1

    return clusters, labels


# REDCLAN
k1 = 50
core_pts = compute_core_points(data.values, k1)  
k2 = 75

_, redclan_labels = dbscan_cluster(data.values, core_pts, k2)
redclan_labels[redclan_labels == -1] = 1  # Outliers
redclan_labels[redclan_labels != 1] = 0   # Inliers
df_2100['redclan_outlier'] = redclan_labels


# ## CBLOF 

# In[4]:


# CBLOF
cblof = CBLOF(alpha=0.7, beta=7, check_estimator=False, clustering_estimator=None, contamination=0.1, n_clusters=5, n_jobs=None, random_state=None, use_weights=False)
cblof.fit(data)
df_2100['cblof_outlier'] = cblof.labels_


# ## DMCA

# In[5]:


##########################DMCA functions
def inne_model(data, psi):
    model = IsolationForest(n_estimators=psi, random_state=42)
    model.fit(data)
    return model
def determine_psi_values():
    # Return a range of psi values
    return range(50, 501, 50)  # Example: psi values from 50 to 500 with an interval of 50

# Define the function to build a hyperensemble of models
def build_hyperensemble(data, psi_values):
    hyperensemble = []
    for psi in psi_values:
        model = IsolationForest(n_estimators=psi, random_state=42)
        model.fit(data)
        hyperensemble.append(model)
    return hyperensemble

# Define the DMCA function
def d_mca_0(data, hyperensemble):
    # Example implementation
    scores = pd.Series([0] * len(data))
    for model in hyperensemble:
        scores += model.decision_function(data)

    outlier_threshold = scores.quantile(0.10)  # Adjust threshold as needed
    return scores < outlier_threshold  # Returns True for outliers
# DMCA
psi_values = determine_psi_values()
hyperensemble = build_hyperensemble(data, psi_values)
df_2100['dmca_outlier'] = d_mca_0(data, hyperensemble)


# ## DPSL

# In[6]:


#### DPSL functions

def establish_potential_chains(data, h_max):
    # This function now returns a list of potential peak points (second nearest neighbors)
    nbrs = NearestNeighbors(n_neighbors=h_max+1, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    peak_points = data[indices[:, h_max]]  # Assuming the h_max-th nearest neighbor as the peak point
    return peak_points

def calculate_psl(point, neighbors):
    distances = np.sqrt(np.sum((neighbors - point)**2, axis=1))
    psl = np.mean(distances)
    return psl

def calculate_dpl(point, peak_point):
    dpl = np.sqrt(np.sum((peak_point - point)**2))
    return dpl

def calculate_anomaly_score(psl, dpl):
    psl_weight = 0.1
    dpl_weight = 0.1
    anomaly_score = (psl_weight * psl) + (dpl_weight * dpl)
    return anomaly_score
def dpsl(data, n_range, t, h_max):
    N = len(data)
    final_scores = np.zeros(N)

    for n in n_range:
        n_samples = N // n
        # Check if n_samples is less than h_max + 1
        if n_samples < h_max + 1:
            continue  # Skip this iteration or adjust h_max

        for _ in range(t):
            D0_indices = np.random.choice(N, n_samples, replace=False)
            D0 = data[D0_indices]
            peak_points = establish_potential_chains(D0, h_max)

            scores = np.zeros(N)
            for i in range(N):
                psl = calculate_psl(data[i], D0)
                # Extract the corresponding peak point for each data[i]
                peak_point = peak_points[np.argmin(np.sum((D0 - data[i])**2, axis=1))]
                dpl = calculate_dpl(data[i], peak_point)
                scores[i] = calculate_anomaly_score(psl, dpl)

            final_scores += scores

    final_scores /= (t * len(n_range))
    return final_scores
# DPSL
n_range=range(2,51,2)
t = 10
h_max = 10
anomaly_scores = dpsl(data.values, n_range, t, h_max)
threshold = np.percentile(anomaly_scores, 95)
df_2100['dpsl_outlier'] = (anomaly_scores > threshold).astype(int)



# ## Plotting Figure 1

# In[8]:


blue_color = '#1f77b4'
edge_color = (1, 1, 1, 0.5)
# Plotting
methods = ['fb_outlier', 'redclan_outlier', 'cblof_outlier', 'dmca_outlier', 'dpsl_outlier']
titles = [ "(b) Feature Bagging","(c) REDCLAN", "(d) CBLOF","(e) D.MCA", "(f) DPSL"]

fig, axs = plt.subplots(2, 3, figsize=(18, 9))  

# IRB Plot
sns.scatterplot(data=df_2100, x="cotton_withdrawal", y="cotton_scarcity", s=70, color=blue_color, alpha=1, ax=axs[0, 0], 
                edgecolor=edge_color, linewidth=0.3)

axs[0, 0].set_title("(a) IRB")
axs[0, 0].tick_params(axis='x', labelsize=16)
axs[0, 0].tick_params(axis='y', labelsize=16)
axs[0, 0].set_xlabel('')  
axs[0, 0].set_ylabel('Water Scarcity', fontsize=16)

# Other methods plots
for i, method in enumerate(methods):
    row = (i + 1) // 3
    col = (i + 1) % 3
    palette = {-1: blue_color, 0: blue_color, 1: 'red', 2: blue_color}  # Adjust as per your data
    sns.scatterplot(data=df_2100, x='cotton_withdrawal', y='cotton_scarcity', hue=method, palette=palette, ax=axs[row, col], s=60, legend=False,
                    edgecolor=edge_color, linewidth=0.5)
    
    axs[row, col].set_title(titles[i])
    axs[row, col].tick_params(axis='x', labelsize=16)
    axs[row, col].tick_params(axis='y', labelsize=16)
    axs[row, col].set_xlabel('Cotton Withdrawals ($km^3$)', fontsize=16)
    axs[row, col].set_ylabel('Water Scarcity', fontsize=16)

    # Only display y-axis label in the first column
    if col != 0:
        axs[row, col].set_ylabel('')

    # Only display x-axis label in the second row
    if row != 1:
        axs[row, col].set_xlabel('')

plt.tight_layout()
#output_filename = "results/Figure_1_All_methods_IRB.png"  # Change this to your preferred file path and name
plt.savefig('../results/Figure_1_All_methods_IRB.png',bbox_inches='tight')
plt.close()


# In[ ]:




